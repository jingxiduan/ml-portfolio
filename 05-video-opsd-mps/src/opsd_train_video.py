"""
OPSD (On-Policy Self-Distillation) training loop for TEXT-TO-VIDEO, extending
`04-diffusion-opsd-mps`'s single-image implementation to AnimateDiff (a
frozen motion adapter stacked on the SD1.5 UNet).

The 4-step mechanism is identical to the image version:
  1. On-policy query sampling: a frozen *behavior* LoRA adapter rolls out the
     first denoising step from pure noise to a low-noise "query" latent
     (now a (C, frames, H, W) video latent instead of a single image latent).
  2. Reward-guided target construction: the behavior policy's one-step clean
     prediction (the "anchor" -- now a short clip) is scored with the
     differentiable video reward (frame-averaged CLIPScore + temporal
     consistency, see reward_model.py), and the reward gradient w.r.t. the
     query latent is followed by a small bounded step in both directions to
     build a positive/negative target, each via one more behavior-policy
     forward pass, then detached.
  3. Detached distillation fit: a separate *trainable* LoRA adapter predicts
     the clean clip from the SAME query latent and is fit towards the
     positive target / away from the negative target. Gradients never flow
     through the rollout, only through the LoRA weights.
  4. Periodic EMA refresh of the behavior adapter towards the trainable one.

What's different from the image version (and why):
  - Base model: AnimateDiff motion adapter (`guoyww/animatediff-motion-adapter-v1-5-2`,
    frozen) on top of SD1.5's UNet (860M, same param count class as SD-Turbo),
    instead of SD-Turbo directly -- SD-Turbo has no released video variant.
  - LoRA is added ONLY to the UNet's spatial attention projections
    (to_q/to_k/to_v/to_out.0 inside the ORIGINAL SD1.5 blocks), explicitly
    excluding the motion adapter's own temporal-attention layers (which are
    named identically but live under `*.motion_modules.*`) -- the motion
    adapter is a separately-pretrained module for temporal coherence and is
    kept frozen throughout; OPSD here only ever edits "what the model draws
    per frame", not "how it moves between frames".
  - Reward: ClipVideoRewardModel (see reward_model.py) averages CLIPScore
    over `--num-frames` sampled frames and adds a temporal-consistency term.
  - sigma0/sigma1/t0/t1 constants are unchanged from the image version:
    SD1.5 and SD-Turbo share the exact same beta schedule
    (beta_start=0.00085, beta_end=0.012, scaled_linear), so swapping the
    pipeline scheduler to EulerDiscreteScheduler gives identical sigmas.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from peft import LoraConfig

from reward_model import ClipVideoRewardModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"


def load_prompts(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def build_pipe(device: str):
    adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=torch.float32)
    pipe = AnimateDiffPipeline.from_pretrained(BASE_MODEL_ID, motion_adapter=adapter, torch_dtype=torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.motion_modules_frozen = True  # just a marker, motion params never get requires_grad_(True)
    return pipe


def spatial_only_target_modules(unet) -> list[str]:
    """Full qualified module names for to_q/to_k/to_v/to_out.0 projections that
    belong to the original SD1.5 spatial attention blocks, excluding the
    AnimateDiff motion adapter's temporal-attention projections (which share
    the same leaf names but live under `*.motion_modules.*`)."""
    return [n for n, _ in unet.named_modules()
            if "motion_modules" not in n and n.split(".")[-1] in ("to_q", "to_k", "to_v", "to_out.0")]


def add_lora_adapters(unet, r: int, alpha: int):
    targets = spatial_only_target_modules(unet)
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=targets, init_lora_weights=True)
    unet.add_adapter(cfg, adapter_name="trainable")
    unet.add_adapter(cfg, adapter_name="behavior")
    return len(targets)


def lora_params(unet, adapter_name: str):
    return [p for n, p in unet.named_parameters() if f".{adapter_name}." in n and "lora_" in n]


@torch.no_grad()
def ema_update(unet, decay: float):
    trainable = dict(unet.named_parameters())
    for n, p in trainable.items():
        if ".trainable." in n and "lora_" in n:
            behavior_name = n.replace(".trainable.", ".behavior.")
            trainable[behavior_name].mul_(decay).add_(p, alpha=1 - decay)


def encode_prompts(pipe, prompts: list[str], device: str):
    tok = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                          truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        ctx = pipe.text_encoder(**tok).last_hidden_state
    return ctx


def decode_frames_to_image01(vae, latents_5d: torch.Tensor) -> torch.Tensor:
    """latents_5d: (B, 4, n_frames, h, w) -> (B, n_frames, 3, H, W) in [0, 1]."""
    b, c, n_frames, h, w = latents_5d.shape
    flat = latents_5d.permute(0, 2, 1, 3, 4).reshape(b * n_frames, c, h, w)
    image = vae.decode(flat / vae.config.scaling_factor).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    _, c3, H, W = image.shape
    return image.reshape(b, n_frames, c3, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "train_prompts.txt"))
    ap.add_argument("--rounds", type=int, default=60, help="number of OPSD optimizer steps")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--reward-step-size", type=float, default=1.2,
                     help="bounded step size (in latent units) used to build +/- targets around the anchor")
    ap.add_argument("--lambda-neg", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.9)
    ap.add_argument("--ema-every", type=int, default=4)
    ap.add_argument("--temporal-weight", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    torch.manual_seed(args.seed)
    prompts = load_prompts(Path(args.prompts))
    print(f"[info] {len(prompts)} training prompts loaded from {args.prompts}")

    print("[info] loading AnimateDiff (SD1.5 + frozen motion adapter) pipeline...")
    pipe = build_pipe(args.device)
    unet = pipe.unet
    n_targets = add_lora_adapters(unet, args.lora_r, args.lora_alpha)
    print(f"[info] LoRA attached to {n_targets} spatial attention projections (motion adapter untouched)")

    for n, p in unet.named_parameters():
        p.requires_grad_(".trainable." in n and "lora_" in n)
    train_params = lora_params(unet, "trainable")
    print(f"[info] trainable LoRA params: {sum(p.numel() for p in train_params):,}")

    optimizer = torch.optim.AdamW(train_params, lr=args.lr)

    print("[info] loading CLIP video reward model...")
    reward_model = ClipVideoRewardModel(device=args.device, temporal_weight=args.temporal_weight)

    latent_size = args.resolution // 8
    n_frames = args.num_frames
    sigma0, sigma1 = 14.6146, 1.6129
    t0, t1 = 999.0, 499.0

    metrics_f = metrics_path.open("w")
    t_start = time.time()

    for step in range(1, args.rounds + 1):
        batch_prompts = [prompts[i % len(prompts)] for i in
                          torch.randint(0, len(prompts), (args.batch_size,)).tolist()]
        ctx = encode_prompts(pipe, batch_prompts, args.device)
        ctx_rep = ctx.repeat_interleave(n_frames, dim=0)
        text_feats = reward_model.encode_text(batch_prompts)

        z0 = torch.randn(args.batch_size, 4, n_frames, latent_size, latent_size, device=args.device) * sigma0

        # ---- step 0: behavior policy rollout (no grad) ----
        unet.set_adapter("behavior")
        with torch.no_grad():
            scaled0 = z0 / ((sigma0 ** 2 + 1) ** 0.5)
            t0_b = torch.full((args.batch_size,), t0, device=args.device)
            eps0 = unet(scaled0, t0_b, encoder_hidden_states=ctx_rep).sample
            z_query = z0 + eps0 * (sigma1 - sigma0)  # on-policy query state (noisy, sigma1)

        # ---- anchor estimate + reward gradient (behavior policy, grad only wrt input) ----
        z_query = z_query.detach().requires_grad_(True)
        t1_b = torch.full((args.batch_size,), t1, device=args.device)
        scaled_q = z_query / ((sigma1 ** 2 + 1) ** 0.5)
        eps1_anchor = unet(scaled_q, t1_b, encoder_hidden_states=ctx_rep).sample
        anchor_x0 = z_query - sigma1 * eps1_anchor
        anchor_frames = decode_frames_to_image01(pipe.vae, anchor_x0)
        reward, align_score, consistency = reward_model.score_video(anchor_frames, text_feats)
        grad_z = torch.autograd.grad(reward.sum(), z_query)[0]
        direction = grad_z / (grad_z.flatten(1).norm(dim=1).view(-1, 1, 1, 1, 1) + 1e-8)

        z_query = z_query.detach()
        z_pos = (z_query + args.reward_step_size * direction).detach()
        z_neg = (z_query - args.reward_step_size * direction).detach()

        with torch.no_grad():
            scaled_pos = z_pos / ((sigma1 ** 2 + 1) ** 0.5)
            eps_pos = unet(scaled_pos, t1_b, encoder_hidden_states=ctx_rep).sample
            target_pos = z_pos - sigma1 * eps_pos

            scaled_neg = z_neg / ((sigma1 ** 2 + 1) ** 0.5)
            eps_neg = unet(scaled_neg, t1_b, encoder_hidden_states=ctx_rep).sample
            target_neg = z_neg - sigma1 * eps_neg

        # ---- trainable policy: fit the same query state towards target_pos / away from target_neg ----
        unet.set_adapter("trainable")
        scaled_q_train = z_query / ((sigma1 ** 2 + 1) ** 0.5)
        eps1_train = unet(scaled_q_train, t1_b, encoder_hidden_states=ctx_rep).sample
        pred_x0_train = z_query - sigma1 * eps1_train

        loss_pos = F.mse_loss(pred_x0_train, target_pos)
        loss_neg = F.mse_loss(pred_x0_train, target_neg)
        loss = loss_pos + args.lambda_neg * F.relu(args.margin - loss_neg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()

        if step % args.ema_every == 0:
            ema_update(unet, args.ema_decay)

        rec = {
            "step": step,
            "loss": float(loss.item()),
            "loss_pos": float(loss_pos.item()),
            "loss_neg": float(loss_neg.item()),
            "mean_anchor_reward": float(reward.mean().item()),
            "mean_anchor_align": float(align_score.mean().item()),
            "mean_anchor_consistency": float(consistency.mean().item()),
            "elapsed_s": round(time.time() - t_start, 1),
        }
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        print(f"[step {step:3d}/{args.rounds}] loss={rec['loss']:.4f} "
              f"(pos={rec['loss_pos']:.4f} neg={rec['loss_neg']:.4f}) "
              f"reward={rec['mean_anchor_reward']:.2f} "
              f"align={rec['mean_anchor_align']:.2f} cons={rec['mean_anchor_consistency']:.2f} "
              f"t={rec['elapsed_s']:.0f}s")

    metrics_f.close()

    unet.set_adapter("trainable")
    unet.save_lora_adapter(str(out_dir / "lora_trainable"), adapter_name="trainable")
    print(f"[ok] saved trainable LoRA adapter to {out_dir / 'lora_trainable'}")
    print(f"[ok] metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
