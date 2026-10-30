"""InstructVideo-style reward fine-tuning baseline.

Reference: Yuan et al., "InstructVideo: Instructing Video Diffusion Models with Human Feedback",
           2024.

Core idea (ReFL applied to video): roll out the full denoising trajectory, decode to pixels,
compute the reward, and back-prop the reward gradient through the final few denoising steps
(``k_backprop`` steps, typically 1-3) into the LoRA.

Compared to OPSD-I2V:
    * No frozen behavior policy / on-policy query stage.
    * Reward signal only at the final clean output, not at intermediate noise levels.
    * Known to collapse motion dynamics in I2V (what we claim to fix).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..dynamicrafter_wrapper import I2VConfig, OPSDPolicies

logger = logging.getLogger(__name__)


@dataclass
class InstructVideoConfig:
    first_frame_dir: str = "data/first_frames"
    train_prompts: str = "data/prompts/train.json"
    batch_size: int = 1
    num_workers: int = 2

    num_steps: int = 500
    lr: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 20

    k_backprop: int = 2              # back-prop through last k denoising steps
    reward_scale: float = 1.0

    log_every: int = 10
    ckpt_every: int = 100
    out_dir: str = "outputs/instructvideo"
    seed: int = 0
    wandb_project: Optional[str] = "opsd-i2v"
    wandb_run_name: Optional[str] = None


class InstructVideoTrainer:
    def __init__(
        self,
        cfg: InstructVideoConfig,
        model_cfg: I2VConfig,
        policies: OPSDPolicies,          # we only use `v_theta` (trainable) + vae
        reward_fn: Callable,
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

        trainable = [p for p in self.p.v_theta.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99)
        )
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )
        self.loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, drop_last=True,
        )

        self._wandb = None
        if cfg.wandb_project:
            try:
                import wandb  # type: ignore
                wandb.init(project=cfg.wandb_project,
                           name=cfg.wandb_run_name or f"instructvideo-seed{cfg.seed}",
                           config={**cfg.__dict__, **model_cfg.__dict__})
                self._wandb = wandb
            except ImportError:
                pass

    def _lr_lambda(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return step / max(1, self.cfg.warmup_steps)
        p = (step - self.cfg.warmup_steps) / max(1, self.cfg.num_steps - self.cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    @torch.no_grad()
    def _encode_first_frame(self, x: torch.Tensor) -> torch.Tensor:
        return self.p.vae.encode(x).latent_dist.sample() * 0.18215

    def _refl_step(self, batch: Dict) -> Dict[str, float]:
        device = self.p.device
        x0_src_px = batch["first_frame"].to(device)
        prompts = batch["prompt"]
        cond = batch["cond"]
        cond["first_frame_latent"] = self._encode_first_frame(x0_src_px)

        B = x0_src_px.shape[0]
        T = cond.get("video_length", 16)
        H, W = self.model_cfg.resolution
        C = cond["first_frame_latent"].shape[1]
        H_, W_ = cond["first_frame_latent"].shape[-2], cond["first_frame_latent"].shape[-1]

        # No-grad rollout for the first (num_inference - k_backprop) steps.
        n_steps = len(self.scheduler.timesteps)
        n_nograd = max(0, n_steps - self.cfg.k_backprop)
        z = torch.randn(B, C, T, H_, W_, device=device, dtype=cond["first_frame_latent"].dtype)
        with torch.no_grad():
            for i, t in enumerate(self.scheduler.timesteps):
                if i >= n_nograd:
                    break
                noise_pred = self.p.v_theta(z, t, cond=cond)
                z = self.scheduler.step(noise_pred, t, z).prev_sample

        # Back-prop through last k steps.
        for t in self.scheduler.timesteps[n_nograd:]:
            noise_pred = self.p.v_theta(z, t, cond=cond)
            z = self.scheduler.step(noise_pred, t, z).prev_sample

        video_px = self.p.vae.decode(z / 0.18215).sample
        if video_px.dim() == 5 and video_px.shape[1] == 3:
            video_px = video_px.permute(0, 2, 1, 3, 4)
        reward = self.reward_fn(video_px, prompts).mean()
        loss = -self.cfg.reward_scale * reward

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.p.v_theta.parameters() if p.requires_grad], self.cfg.grad_clip
        )
        self.optimizer.step()

        return {"loss": float(loss.detach().cpu()), "reward": float(reward.detach().cpu())}

    def train(self) -> None:
        self.p.v_theta.train()
        step = 0
        t0 = time.time()
        it = iter(self.loader)
        while step < self.cfg.num_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.loader)
                batch = next(it)
            metrics = self._refl_step(batch)
            self.lr_sched.step()
            step += 1
            if step % self.cfg.log_every == 0:
                msg = (f"[step {step}/{self.cfg.num_steps}] "
                       f"loss={metrics['loss']:+.4f} R={metrics['reward']:+.4f} "
                       f"lr={self.lr_sched.get_last_lr()[0]:.2e} ({time.time()-t0:.1f}s)")
                logger.info(msg); print(msg)
                if self._wandb:
                    self._wandb.log({**metrics, "step": step})
            if step % self.cfg.ckpt_every == 0 or step == self.cfg.num_steps:
                out = Path(self.cfg.out_dir) / f"lora_step{step:05d}"
                out.mkdir(parents=True, exist_ok=True)
                try:
                    self.p.v_theta.save_pretrained(str(out))
                except AttributeError:
                    sd = {k: v.detach().cpu() for k, v in self.p.v_theta.state_dict().items()
                          if "lora_" in k}
                    torch.save(sd, out / "lora.pt")
