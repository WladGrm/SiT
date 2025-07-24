# distill_ddp.py

import os
import torch
# TF32 speedups on Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
import argparse
from copy import deepcopy
from torchvision.utils import make_grid
from diffusers.models import AutoencoderKL

from models import SiT_models
from train_utils import parse_transport_args
from transport import create_transport

def requires_grad(model: torch.nn.Module, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag

class OneStepGen(torch.nn.Module):
    """One‑step student: maps latent z → one‑step sample via guided teacher net."""
    def __init__(self, base: torch.nn.Module, num_classes: int, cfg_scale: float):
        super().__init__()
        self.net = deepcopy(base)
        requires_grad(self.net, True)
        self.num_classes = num_classes
        self.cfg = cfg_scale

    def forward(self, x, t, y=None):
        if y is not None:
            # classifier‑free guidance
            x = torch.cat([x, x], dim=0)
            y = torch.cat([y, torch.full_like(y, self.num_classes)], dim=0)
            t = torch.cat([t, t], dim=0)
            out = self.net.forward_with_cfg(x, t, y, cfg_scale=self.cfg)
            return out.chunk(2, dim=0)[0]
        else:
            return self.net(x, t, y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        choices=list(SiT_models.keys()), required=True)
    parser.add_argument("--teacher_ckpt", type=str,   required=True)
    parser.add_argument("--batch",        type=int,   default=128)
    parser.add_argument("--iters",        type=int,   default=200_000)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--cfg_scale",    type=float, default=4.0)
    parser.add_argument("--image_size",   type=int,   default=256)
    parser.add_argument("--num_classes",  type=int,   default=1000)
    parser.add_argument("--sample_every", type=int,   default=5_000)
    parser.add_argument("--k_psi",        type=int,   default=5,
                        help="# critic updates per iteration")
    parser.add_argument("--k_G",          type=int,   default=1,
                        help="# generator updates per iteration")
    parse_transport_args(parser)
    args = parser.parse_args()

    # — DDP init —
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        wandb.init(project="sit-distill", config=vars(args))

    # — Load teacher EMA —
    base = SiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        learn_sigma=True
    )
    ema = deepcopy(base).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    ema.load_state_dict(ckpt.get("ema", ckpt.get("model", ckpt)))
    ema.eval()
    requires_grad(ema, False)

    # — Critic v_ψ —
    critic = deepcopy(ema).to(device)
    requires_grad(critic, True)
    critic = DDP(critic, device_ids=[local_rank])
    critic.train()

    # — Student G_θ (1‑step) —
    G = OneStepGen(ema, args.num_classes, cfg_scale=args.cfg_scale).to(device)
    G = DDP(G, device_ids=[local_rank])
    G.train()

    # — Transport & VAE for logging samples —
    transport = create_transport(
        args.path_type, args.prediction, args.loss_weight,
        args.train_eps, args.sample_eps
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    # — Optimizers —
    opt_psi = torch.optim.Adam(critic.parameters(), lr=args.lr)
    opt_G   = torch.optim.Adam(G.parameters(),       lr=args.lr)

    # — Setup loop —
    latent_shape = (args.batch, 4, args.image_size // 8, args.image_size // 8)
    # t_fixed = torch.full((args.batch,), args.sample_eps, device=device)
    t_fixed = torch.full((args.batch,), 0.5, device=device)
    pbar = tqdm(range(1, args.iters + 1),
                disable=(rank != 0),
                desc="Distilling")

    for step in pbar:
        # sample once per iteration
        z = torch.randn(latent_shape, device=device)
        y = torch.randint(0, args.num_classes, (args.batch,), device=device)
        t = t_fixed

        # (a) Critic updates
        G.eval(); critic.train()
        requires_grad(G,     False)
        requires_grad(critic, True)
        for _ in range(args.k_psi):
            with torch.no_grad():
                x1_hat = G(z, t, y)
            loss_psi = transport.training_losses(
                critic, x1_hat, dict(y=y)
            )["loss"].mean()
            opt_psi.zero_grad(); loss_psi.backward(); opt_psi.step()

        # (b) Generator updates
        G.train(); critic.eval()
        requires_grad(G,      True)
        requires_grad(critic, False)
        for _ in range(args.k_G):
            x1_hat = G(z, t, y)
            # L_star = transport.training_losses(
            #     ema, x1_hat, dict(y=y)
            # )["loss"].mean()
            # L_psi  = transport.training_losses(
            #     critic, x1_hat, dict(y=y)
            # )["loss"].mean()
            # gap    = L_star - L_psi
            gap = transport.distillation_loss(
                teacher_model=ema, 
                student_model=critic,       
                x1=x1_hat,
                model_kwargs=dict(y=y)
                )
            opt_G.zero_grad(); gap.backward(); opt_G.step()

        # — W&B logging —
        if rank == 0:
            grad_norm = torch.sqrt(sum(
                p.grad.norm()**2 for p in G.parameters() if p.grad is not None
            ))
            wandb.log({
                "loss_critic": loss_psi.item(),
                "L_star":      L_star.item(),
                "L_psi":       L_psi.item(),
                "gap":         gap.item(),
                "grad_norm":   grad_norm,
                "step":        step,
            })
            pbar.set_postfix(gap=f"{gap.item():.4f}")

        # — Sample & log images every `sample_every` steps —
        if step % args.sample_every == 0 and rank == 0:
            sample_n = 32
            H = args.image_size // 8
            with torch.no_grad():
                zs = torch.randn(sample_n, 4, H, H, device=device)
                ys = torch.randint(0, args.num_classes, (sample_n,), device=device)
                ts = torch.full((sample_n,), args.sample_eps, device=device)
                out  = G(zs, ts, ys)
                imgs = vae.decode(out / 0.18215).sample
                grid = make_grid(imgs, nrow=8, normalize=True,
                                 value_range=(-1,1))

            wandb.log({
                "student_one_step_samples": [
                    wandb.Image(grid,
                                caption=f"Step {step} ▶️ 1‑NFE samples")
                ]
            }, step=step)

        # — Intermediate save every 1 000 steps —
        if step % 1000 == 0 and rank == 0:
            torch.save({
                "student":   G.module.net.state_dict(),
                "critic":    critic.module.state_dict(),
                "cfg_scale": args.cfg_scale,
                "step":      step,
            }, f"student_one_step_{step:07d}.pth")

    # — Final save —
    if rank == 0:
        torch.save({
            "student":   G.module.net.state_dict(),
            "critic":    critic.module.state_dict(),
            "cfg_scale": args.cfg_scale,
            "step":      args.iters,
        }, "student_one_step_final.pth")
        print("✅ Distillation complete!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
