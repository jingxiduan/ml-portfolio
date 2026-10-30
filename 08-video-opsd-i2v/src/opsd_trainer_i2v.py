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

from .dynamicrafter_wrapper import I2VConfig, OPSDPolicies, build_cond, build_opsd_policies, rollout_until
from .utils.ema import EMA

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers — bridge raw DynamiCrafter UNetModel / 2D VAE to our training loop
# =============================================================================

def _call_unet(unet, x_latent: torch.Tensor, t: torch.Tensor, cond: Dict) -> torch.Tensor:
    """Call raw ``UNetModel.forward`` with a DynamiCrafter cond dict.

    DynamiCrafter's ``DiffusionWrapper`` expects ``cond`` to be *unpacked* as
    kwargs; passing ``cond=cond`` would be swallowed by the UNet's ``**kwargs``
    and silently lose conditioning. This helper replicates the "hybrid" /
    "crossattn" branch logic:

      - ``c_concat`` (list of tensors) is concatenated to ``x`` on channel dim.
      - ``c_crossattn`` (list of tensors) is cat'd on dim=1 and passed as ``context``.
      - ``fs`` / ``features_adapter`` passed through if present.
    """
    x = x_latent
    if "c_concat" in cond and cond["c_concat"]:
        x = torch.cat([x] + list(cond["c_concat"]), dim=1)
    context = None
    if "c_crossattn" in cond and cond["c_crossattn"]:
        context = torch.cat(list(cond["c_crossattn"]), dim=1)
    fs = cond.get("fs", None)
    features_adapter = cond.get("features_adapter", None)
    # DynamiCrafter's timestep_embedding wants a 1-D tensor (`timesteps[:, None]`).
    # scheduler.timesteps[i] is 0-D; broadcast up here.
    if t.dim() == 0:
        t = t.unsqueeze(0).expand(x.shape[0])
    # Use autocast for CUDA to handle mixed fp16/fp32 precision automatically.
    # DynamiCrafter + LoRA + DiffusionWrapper all expect consistent dtypes;
    # autocast handles this without brittle manual casting.
    device = x.device
    if device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return unet(x, t, context=context, features_adapter=features_adapter, fs=fs)
    return unet(x, t, context=context, features_adapter=features_adapter, fs=fs)


def _vae_decode_video(vae, z: torch.Tensor, scale_factor: float = 0.18215,
                      force_fp32: bool = False, chunk_size: int = 4) -> torch.Tensor:
    """Decode a 5-D latent video with DynamiCrafter's 2-D VAE.

    Replicates ``LatentVisualDiffusion.decode_core``: reshape 5-D → 4-D batch,
    decode per-frame, rearrange back. Returns ``[B, T, 3, H, W]``.

    Args:
        force_fp32: if True, temporarily casts the VAE to fp32 before decoding.
            Required on CUDA when gradients must flow through the decoder (step 3
            of OPSD) — fp16 VAE gradients underflow to zero on H20 precision.
        chunk_size: decode this many frames at a time to limit peak GPU memory.
    """
    B, C, T, H, W = z.shape
    z2d = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    z2d_in = z2d / scale_factor

    if force_fp32 or z2d.device.type != "cuda":
        # Full fp32 — needed for gradient flow in the reward-guided step.
        # NOTE: do NOT restore VAE to fp16 here — the backward pass (autograd.grad)
        # will need the weights in fp32. Caller must call vae.half() after grad.
        vae.float()
        # Decode in chunks to limit peak memory (16 frames at 320×512 + gradients)
        chunks = z2d_in.float().split(chunk_size, dim=0)
        dec_chunks = []
        for chunk in chunks:
            dec_chunks.append(vae.decode(chunk))
        dec = torch.cat(dec_chunks, dim=0)
    else:
        vae_dtype = next(vae.parameters()).dtype
        dec = vae.decode(z2d_in.to(dtype=vae_dtype))

    _, C_out, H_out, W_out = dec.shape
    return dec.view(B, T, C_out, H_out, W_out).float()


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

        # ---- metrics log (JSON lines) ----
        self._metrics_path = Path(cfg.out_dir) / "train_metrics.jsonl"
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on new run
        self._metrics_path.write_text("")

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
        """VAE-encode the first frame to the latent conditioning.

        DynamiCrafter's ``AutoencoderKL.encode`` returns a
        ``DiagonalGaussianDistribution`` directly (no ``.latent_dist`` wrapper
        à la diffusers), so we call ``.sample()`` on it.
        """
        # Cast to model dtype (fp16 on CUDA, fp32 on MPS) before VAE encode
        vae_dtype = next(self.p.vae.parameters()).dtype
        return self.p.vae.encode(x0_src_px.to(dtype=vae_dtype)).sample() * 0.18215

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

        # ---------- Step 1: on-policy query ----------
        with torch.no_grad():
            x0_src_lat = self._encode_first_frame(x0_src_px)

            # Build the full DynamiCrafter cond dict (CLIP text + image + proj + img_cat_cond).
            # The dataset provides only {text, video_length, fps}; the real cond requires
            # three frozen encoders which live on policies.cond_stage.
            if self.p.cond_stage is None:
                raise RuntimeError(
                    "OPSDPolicies.cond_stage is None — DynamiCrafter cond-stage modules "
                    "(cond_stage_model / embedder / image_proj_model) were not found on the "
                    "base checkpoint. Cannot train without conditioning."
                )
            cond = build_cond(
                self.p.cond_stage,
                prompts=list(prompts),
                first_frame_latent=x0_src_lat,
                first_frame_px=x0_src_px,
                T=self.p.cfg.num_frames,
            )

            sigma_q = self._sample_sigma_q()
            z_q = rollout_until(self.p.v_old, x0_src_lat, cond, sigma_q, self.scheduler)

            # ---------- Step 2: clean anchor y0 ----------
            # Convert timestep to actual alpha_t / sigma_t for v-prediction reconstruction.
            # DynamiCrafter runs in v-prediction mode where:
            #   v = alpha_t * epsilon - sigma_t * x_0
            #   x_0 = alpha_t * z_t - sigma_t * v
            t_int = sigma_q.long().clamp(0, 999)
            alpha_prod = self.scheduler.alphas_cumprod[t_int.cpu()].to(device).view(-1, 1, 1, 1, 1)
            alpha_t = alpha_prod.sqrt()
            sigma_t = (1.0 - alpha_prod).sqrt()

            v_old_pred = _call_unet(self.p.v_old, z_q, sigma_q, cond)
            y0 = alpha_t * z_q - sigma_t * v_old_pred.float()           # v-prediction reconstruction

        # ---------- Step 3: reward-guided bounded targets ----------
        # Free CUDA cache before the memory-intensive reward gradient computation
        if device.type == "cuda":
            torch.cuda.empty_cache()
        y0_leaf = y0.detach().requires_grad_(True)
        with torch.enable_grad():
            # force_fp32: on CUDA, fp16 VAE gradients underflow to zero
            video_px = _vae_decode_video(self.p.vae, y0_leaf, force_fp32=True)  # [B,T,3,H,W]
            # Downsample for RAFT-based rewards to limit GPU memory.
            # RAFT optical flow is the dominant memory consumer; halving resolution
            # cuts its memory 4x while preserving the motion signal.
            _B, _T, _C, _H, _W = video_px.shape
            reward_res = (max(160, _H // 2), max(256, _W // 2))
            if (_H > reward_res[0]) or (_W > reward_res[1]):
                video_ds = F.interpolate(
                    video_px.flatten(0, 1),     # [B*T, C, H, W]
                    size=reward_res, mode="bilinear", align_corners=False
                ).view(_B, _T, _C, reward_res[0], reward_res[1])
            else:
                video_ds = video_px
            reward = self.reward_fn(video_ds, prompts)                   # scalar per batch
            reward_val = reward.detach().clone()                          # save for metrics
            # Log per-component reward values (no-grad, for training curves)
            self._last_reward_components = {}
            if hasattr(self.reward_fn, 'components'):
                with torch.no_grad():
                    for k, v in self.reward_fn.components(video_ds, prompts).items():
                        self._last_reward_components[k] = float(v.mean().cpu())
            g = torch.autograd.grad(reward.sum(), y0_leaf, create_graph=False)[0]
        g = self._rescale_grad(g.detach())
        # Restore VAE to fp16 now that backward is done
        self.p.vae.half()
        # Free reward intermediates
        del video_px, video_ds, reward
        if device.type == "cuda":
            torch.cuda.empty_cache()

        step_size = self.cfg.c_adv * sigma_t.view(-1)                   # use actual sigma, not timestep
        y0_plus = (y0 + step_size.view(-1, 1, 1, 1, 1) * g).detach()
        y0_minus = (y0 - step_size.view(-1, 1, 1, 1, 1) * g).detach()

        # ---------- Step 4: detached finite-step fit on v_theta ----------
        last_loss = 0.0
        last_loss_plus = last_loss_minus = 0.0
        for _ in range(self.cfg.inner_fit_steps):
            v_theta_pred = _call_unet(self.p.v_theta, z_q, sigma_q, cond)
            y0_hat = alpha_t * z_q - sigma_t * v_theta_pred.float()     # v-prediction reconstruction
            loss_plus = F.mse_loss(y0_hat, y0_plus)
            loss_minus = F.mse_loss(y0_hat, y0_minus)
            loss = loss_plus - loss_minus                                # OPSD contrastive form

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _param_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.p.v_theta.parameters() if p.requires_grad], self.cfg.grad_clip
            )
            self.optimizer.step()

            last_loss = float(loss.detach().cpu())
            last_loss_plus = float(loss_plus.detach().cpu())
            last_loss_minus = float(loss_minus.detach().cpu())

        # --- diagnostic metrics (logging only, does not affect training) ---
        # shift_ratio: ||c_adv * sigma_t * g|| / ||y0||
        _shift_norm = float((step_size.view(-1, 1, 1, 1, 1) * g).flatten(1).norm(dim=1).mean().cpu())
        _y0_norm = float(y0.flatten(1).norm(dim=1).mean().cpu())
        _shift_ratio = _shift_norm / max(_y0_norm, 1e-8)

        # lora_B_norm: L2 norm of all LoRA B matrices
        _lora_b_sq = 0.0
        for n, p in self.p.v_theta.named_parameters():
            if "lora_B" in n:
                _lora_b_sq += float(p.detach().float().norm().cpu().item()) ** 2
        _lora_b_norm = _lora_b_sq ** 0.5

        metrics = {
            "loss": last_loss,
            "loss_plus": last_loss_plus,
            "loss_minus": last_loss_minus,
            "reward": float(reward_val.mean().cpu()),
            "sigma_q": float(sigma_q.detach().cpu().item()),
            "grad_norm": float(g.flatten(1).norm(dim=1).mean().cpu()),
            "param_grad_norm": float(_param_grad_norm.cpu()),
            "shift_ratio": _shift_ratio,
            "lora_B_norm": _lora_b_norm,
        }
        # Attach per-component reward breakdown
        metrics.update({f"R_{k}": v for k, v in self._last_reward_components.items()})
        return metrics

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
            # Always write metrics to JSON lines file (every step, for training curves)
            import json as _json
            dt = time.time() - t0
            lr = self.lr_sched.get_last_lr()[0]
            log_entry = {**metrics, "lr": lr, "step": step, "wall_sec": dt,
                         "gpu_hours": dt / 3600.0}
            with open(self._metrics_path, "a") as f:
                f.write(_json.dumps(log_entry) + "\n")

            if step % self.cfg.log_every == 0:
                msg = (
                    f"[step {step:>5}/{self.cfg.num_steps}] "
                    f"loss={metrics['loss']:+.4f}  "
                    f"R={metrics['reward']:+.4f}  "
                    f"σq={metrics['sigma_q']:.3f}  "
                    f"lr={lr:.2e}  "
                    f"({dt:.1f}s)"
                )
                # Also print per-component rewards
                comp_str = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k.startswith("R_"))
                if comp_str:
                    msg += f"  [{comp_str}]"
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
