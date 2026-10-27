"""OPSD-I2V trainer — the 4-step on-policy self-distillation loop adapted to I2V.

See docs/METHOD.md for the derivation. This file is the paper's main contribution; keep it
readable and keep comments honest.

Algorithm (per training step):
    1. On-policy query:   sample (x0_src, prompt) and roll v_old forward to intermediate σ_q
    2. Clean anchor:      y0 = z_q - σ_q · v_old(z_q, (c, x0_src), σ_q)         [detached]
    3. Bounded targets:   y0± = y0 ± c_adv · σ_q · normalize(∂R/∂y0)             [detached]
    4. Finite-step fit:   minimize ||y0_hat(θ) - y0+||² - ||y0_hat(θ) - y0-||²
    5. EMA refresh:       every N steps, EMA LoRA(v_theta) → LoRA(v_old)
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dynamicrafter_wrapper import I2VConfig, OPSDPolicies, build_opsd_policies, rollout_until
from .utils.ema import EMA

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================


@dataclass
class OPSDTrainConfig:
    # ---- data ----
    first_frame_dir: str = "data/first_frames"
    train_prompts: str = "data/prompts/train.json"
    batch_size: int = 1              # M3 Pro: must be 1. A100: up to 4.
    num_workers: int = 2

    # ---- optimization ----
    num_steps: int = 500
    lr: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 20

    # ---- OPSD core ----
    c_adv: float = 5.0               # trust-region radius (from OPSD paper)
    sigma_q_min: float = 0.2         # sample σ_q ∈ [σ_q_min, σ_q_max] of total range
    sigma_q_max: float = 0.8
    inner_fit_steps: int = 1         # K in the detached finite-step fit
    reward_grad_rescale: str = "l2"  # 'l2' (OPSD-paper default) or 'sign'

    # ---- EMA ----
    ema_decay: float = 0.995
    ema_refresh_every: int = 25
    ema_warmup_steps: int = 50       # hard-copy for the first refresh after warmup

    # ---- logging / checkpointing ----
    log_every: int = 10
    ckpt_every: int = 100
    out_dir: str = "outputs/opsd_i2v_default"
    seed: int = 0
    wandb_project: Optional[str] = "opsd-i2v"
    wandb_run_name: Optional[str] = None


# =============================================================================
# Trainer
# =============================================================================


class OPSDI2VTrainer:
    """OPSD-I2V trainer.

    Expects:
        policies         : OPSDPolicies     — v_theta / v_old / vae
        reward_fn        : callable(video[B,T,3,H,W], prompts[list[str]]) -> scalar tensor
        scheduler        : the sampler (DDIM / FlowMatchEuler) in training mode
        dataset          : I2VDataset yielding dicts {first_frame, prompt, cond}
    """

    def __init__(
        self,
        cfg: OPSDTrainConfig,
        model_cfg: I2VConfig,
        policies: OPSDPolicies,
        reward_fn: Callable[[torch.Tensor, list], torch.Tensor],
        scheduler,
        dataset,
    ) -> None:
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.p = policies
        self.reward_fn = reward_fn
        self.scheduler = scheduler
        self.dataset = dataset

        torch.manual_seed(cfg.seed)
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

        # Only LoRA params get optimized (lora_utils already froze the rest).
        trainable = [p for p in self.p.v_theta.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99)
        )
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )
        self.ema = EMA(decay=cfg.ema_decay, name_filter="lora")

        self.loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.p.device.type == "cuda"),
            drop_last=True,
        )

        # ---- wandb (optional) ----
        self._wandb = None
        if cfg.wandb_project:
            try:
                import wandb  # type: ignore

                wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name or f"opsd-i2v-seed{cfg.seed}",
                    config={**cfg.__dict__, **model_cfg.__dict__},
                )
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed — skipping cloud logging.")

    # ------------------------------------------------------------------ sched
    def _lr_lambda(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return step / max(1, self.cfg.warmup_steps)
        progress = (step - self.cfg.warmup_steps) / max(
            1, self.cfg.num_steps - self.cfg.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # ---------------------------------------------------------------- step
    def _sample_sigma_q(self) -> torch.Tensor:
        """Uniformly sample intermediate noise level in the configured range."""
        n_steps = len(self.scheduler.timesteps)
        lo = int(self.cfg.sigma_q_min * n_steps)
        hi = int(self.cfg.sigma_q_max * n_steps)
        idx = torch.randint(lo, hi, (1,))
        return self.scheduler.timesteps[idx].to(self.p.device)

    @torch.no_grad()
    def _encode_first_frame(self, x0_src_px: torch.Tensor) -> torch.Tensor:
        """VAE-encode the first frame to the latent conditioning."""
        return self.p.vae.encode(x0_src_px).latent_dist.sample() * 0.18215

    def _rescale_grad(self, g: torch.Tensor) -> torch.Tensor:
        if self.cfg.reward_grad_rescale == "sign":
            return g.sign()
        # Default: L2-normalize per-sample.
        flat = g.flatten(1)
        norm = flat.norm(dim=1, keepdim=True) + 1e-8
        return (flat / norm).view_as(g)

    def _opsd_step(self, batch: Dict) -> Dict[str, float]:
        """One OPSD training step. See docstring at top of file."""
        device = self.p.device
        x0_src_px = batch["first_frame"].to(device, non_blocking=True)   # [B,3,H,W] in [-1,1]
        prompts = batch["prompt"]                                        # list[str]
        cond = batch["cond"]                                             # dict, already on device

        # ---------- Step 1: on-policy query ----------
        with torch.no_grad():
            x0_src_lat = self._encode_first_frame(x0_src_px)
            cond["first_frame_latent"] = x0_src_lat

            sigma_q = self._sample_sigma_q()
            z_q = rollout_until(self.p.v_old, x0_src_lat, cond, sigma_q, self.scheduler)

            # ---------- Step 2: clean anchor y0 ----------
            v_old_pred = self.p.v_old(z_q, sigma_q, cond=cond)
            y0 = z_q - sigma_q * v_old_pred                              # detached

        # ---------- Step 3: reward-guided bounded targets ----------
        y0_leaf = y0.detach().requires_grad_(True)
        with torch.enable_grad():
            video_px = self.p.vae.decode(y0_leaf / 0.18215).sample       # [B,C,T,H,W] or [B,T,3,H,W]
            # normalize to channels-first-time (B, T, 3, H, W)
            if video_px.dim() == 5 and video_px.shape[1] == 3:
                video_px = video_px.permute(0, 2, 1, 3, 4)
            reward = self.reward_fn(video_px, prompts)                   # scalar per batch
            g = torch.autograd.grad(reward.sum(), y0_leaf, create_graph=False)[0]
        g = self._rescale_grad(g.detach())

        step_size = self.cfg.c_adv * sigma_q
        y0_plus = (y0 + step_size * g).detach()
        y0_minus = (y0 - step_size * g).detach()

        # ---------- Step 4: detached finite-step fit on v_theta ----------
        last_loss = 0.0
        last_loss_plus = last_loss_minus = 0.0
        for _ in range(self.cfg.inner_fit_steps):
            v_theta_pred = self.p.v_theta(z_q, sigma_q, cond=cond)
            y0_hat = z_q - sigma_q * v_theta_pred
            loss_plus = F.mse_loss(y0_hat, y0_plus)
            loss_minus = F.mse_loss(y0_hat, y0_minus)
            loss = loss_plus - loss_minus                                # OPSD contrastive form

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.p.v_theta.parameters() if p.requires_grad], self.cfg.grad_clip
            )
            self.optimizer.step()

            last_loss = float(loss.detach().cpu())
            last_loss_plus = float(loss_plus.detach().cpu())
            last_loss_minus = float(loss_minus.detach().cpu())

        return {
            "loss": last_loss,
            "loss_plus": last_loss_plus,
            "loss_minus": last_loss_minus,
            "reward": float(reward.mean().detach().cpu()),
            "sigma_q": float(sigma_q.detach().cpu().item()),
            "grad_norm": float(g.flatten(1).norm(dim=1).mean().cpu()),
        }

    # ---------------------------------------------------------------- loop
    def train(self) -> None:
        self.p.v_theta.train()
        self.p.v_old.eval()

        step = 0
        t0 = time.time()
        data_iter = iter(self.loader)

        while step < self.cfg.num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)

            metrics = self._opsd_step(batch)
            self.lr_sched.step()
            step += 1

            # ---------- EMA refresh ----------
            if step == self.cfg.ema_warmup_steps:
                n = self.ema.hard_copy(self.p.v_old, self.p.v_theta)
                logger.info(f"[step {step}] EMA hard-copy ({n/1e6:.2f}M LoRA params)")
            elif step > self.cfg.ema_warmup_steps and step % self.cfg.ema_refresh_every == 0:
                self.ema.update(self.p.v_old, self.p.v_theta)

            # ---------- logging ----------
            if step % self.cfg.log_every == 0:
                dt = time.time() - t0
                lr = self.lr_sched.get_last_lr()[0]
                msg = (
                    f"[step {step:>5}/{self.cfg.num_steps}] "
                    f"loss={metrics['loss']:+.4f}  "
                    f"R={metrics['reward']:+.4f}  "
                    f"σq={metrics['sigma_q']:.3f}  "
                    f"lr={lr:.2e}  "
                    f"({dt:.1f}s)"
                )
                logger.info(msg)
                print(msg)
                if self._wandb:
                    self._wandb.log({**metrics, "lr": lr, "step": step})

            # ---------- checkpoint ----------
            if step % self.cfg.ckpt_every == 0 or step == self.cfg.num_steps:
                self._save_ckpt(step)

        logger.info(f"Training done. Total time {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- ckpt
    def _save_ckpt(self, step: int) -> None:
        out = Path(self.cfg.out_dir) / f"lora_step{step:05d}"
        out.mkdir(parents=True, exist_ok=True)
        # PEFT save_pretrained saves only the LoRA adapter — small file, portable.
        try:
            self.p.v_theta.save_pretrained(str(out))
            logger.info(f"[ckpt] saved LoRA to {out}")
        except AttributeError:
            # fallback: full state_dict of LoRA params only
            sd = {
                k: v.detach().cpu()
                for k, v in self.p.v_theta.state_dict().items()
                if "lora_" in k
            }
            torch.save(sd, out / "lora.pt")
            logger.info(f"[ckpt] saved raw LoRA state_dict to {out}/lora.pt")
