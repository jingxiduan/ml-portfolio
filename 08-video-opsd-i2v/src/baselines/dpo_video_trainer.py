"""DPO-Video baseline.

Reference: Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023);
           extended to video diffusion by VideoDPO / Diffusion-DPO.

Core idea: for each prompt, sample two videos (winner w, loser l) using the current policy;
label them with a reward model (we use our ``combined`` reward); minimize
    L = -log σ(β · [ log π_θ(w|c) - log π_ref(w|c) - log π_θ(l|c) + log π_ref(l|c) ])

For diffusion, log π is approximated by the per-step denoising MSE between the policy and
the reference at the winner/loser trajectory latents.

Compared to OPSD-I2V:
    * Needs **two** rollouts per prompt (2× compute vs OPSD's single on-policy rollout).
    * No intermediate-step supervision.
    * No EMA refresh of the reference — reference is the frozen base.
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
class DPOVideoConfig:
    first_frame_dir: str = "data/first_frames"
    train_prompts: str = "data/prompts/train.json"
    batch_size: int = 1
    num_workers: int = 2

    num_steps: int = 500
    lr: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 20

    beta: float = 1000.0             # DPO temperature; large β → sharper preference
    rollout_steps: int = 25          # number of denoising steps per rollout (half of inference)

    log_every: int = 10
    ckpt_every: int = 100
    out_dir: str = "outputs/dpo_video"
    seed: int = 0
    wandb_project: Optional[str] = "opsd-i2v"
    wandb_run_name: Optional[str] = None


class DPOVideoTrainer:
    def __init__(
        self,
        cfg: DPOVideoConfig,
        model_cfg: I2VConfig,
        policies: OPSDPolicies,        # we use v_theta (trainable) + v_old (as reference)
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
                           name=cfg.wandb_run_name or f"dpo-video-seed{cfg.seed}",
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

    @torch.no_grad()
    def _rollout_and_decode(self, cond, model) -> torch.Tensor:
        B = cond["first_frame_latent"].shape[0]
        T = cond.get("video_length", 16)
        C = cond["first_frame_latent"].shape[1]
        H_, W_ = cond["first_frame_latent"].shape[-2], cond["first_frame_latent"].shape[-1]
        z = torch.randn(B, C, T, H_, W_, device=self.p.device,
                        dtype=cond["first_frame_latent"].dtype)
        for t in self.scheduler.timesteps[: self.cfg.rollout_steps]:
            noise_pred = model(z, t, cond=cond)
            z = self.scheduler.step(noise_pred, t, z).prev_sample
        return z

    def _trajectory_logprob(self, z: torch.Tensor, cond, model) -> torch.Tensor:
        """Approximation: sum of per-step denoising MSE over a sub-sampled trajectory."""
        total = 0.0
        for t in self.scheduler.timesteps[: self.cfg.rollout_steps : 5]:
            noise = torch.randn_like(z)
            z_noisy = self.scheduler.add_noise(z, noise, t)
            pred = model(z_noisy, t, cond=cond)
            total = total + F.mse_loss(pred, noise, reduction="mean")
        return -total  # higher = more likely

    def _dpo_step(self, batch: Dict) -> Dict[str, float]:
        device = self.p.device
        x0_src_px = batch["first_frame"].to(device)
        prompts = batch["prompt"]
        cond = batch["cond"]
        cond["first_frame_latent"] = self._encode_first_frame(x0_src_px)

        # Sample two rollouts and score with reward model.
        with torch.no_grad():
            z_a = self._rollout_and_decode(cond, self.p.v_theta)
            z_b = self._rollout_and_decode(cond, self.p.v_theta)
            vid_a = self.p.vae.decode(z_a / 0.18215).sample
            vid_b = self.p.vae.decode(z_b / 0.18215).sample
            if vid_a.dim() == 5 and vid_a.shape[1] == 3:
                vid_a = vid_a.permute(0, 2, 1, 3, 4)
                vid_b = vid_b.permute(0, 2, 1, 3, 4)
            r_a = self.reward_fn(vid_a, prompts)
            r_b = self.reward_fn(vid_b, prompts)

        win = torch.where(r_a > r_b, z_a, z_b).detach()
        lose = torch.where(r_a > r_b, z_b, z_a).detach()

        logp_theta_win = self._trajectory_logprob(win, cond, self.p.v_theta)
        logp_theta_lose = self._trajectory_logprob(lose, cond, self.p.v_theta)
        with torch.no_grad():
            logp_ref_win = self._trajectory_logprob(win, cond, self.p.v_old)
            logp_ref_lose = self._trajectory_logprob(lose, cond, self.p.v_old)

        logits = self.cfg.beta * (
            (logp_theta_win - logp_ref_win) - (logp_theta_lose - logp_ref_lose)
        )
        loss = -F.logsigmoid(logits).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.p.v_theta.parameters() if p.requires_grad], self.cfg.grad_clip
        )
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "r_gap": float((r_a - r_b).abs().mean().cpu()),
        }

    def train(self) -> None:
        self.p.v_theta.train()
        self.p.v_old.eval()
        step = 0
        t0 = time.time()
        it = iter(self.loader)
        while step < self.cfg.num_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.loader); batch = next(it)
            metrics = self._dpo_step(batch)
            self.lr_sched.step()
            step += 1
            if step % self.cfg.log_every == 0:
                msg = (f"[step {step}/{self.cfg.num_steps}] "
                       f"loss={metrics['loss']:+.4f} r_gap={metrics['r_gap']:+.4f} "
                       f"({time.time()-t0:.1f}s)")
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
