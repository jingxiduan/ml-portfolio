"""
OPSD (On-Policy Self-Distillation) training loop, reimplemented at small scale
for a single Apple-silicon machine (MPS, no CUDA).

This follows the structure described in the original paper / repo
(worldbench/DiffusionOPSD) in spirit, but is NOT a line-for-line port:

  1. On-policy query sampling: a frozen *behavior* policy (a LoRA adapter on
     the SD-Turbo UNet) rolls out the first denoising step from pure noise,
     producing a low-noise "query" latent -- a state the current policy would
     actually visit, not an offline / re-noised sample.
  2. Reward-guided target construction: from that query latent, we take the
     behavior policy's own final-step clean prediction (the "anchor"), score
     it with a differentiable CLIP reward, and follow the reward gradient
     w.r.t. the query latent by a small bounded step in both directions to
     build a "better" (positive) and "worse" (negative) target -- both
     produced by one more behavior-policy forward pass, then detached.
  3. Detached distillation fit: a separate *trainable* LoRA adapter predicts
     the clean sample from the SAME query latent, and is fit towards the
     positive target / pushed away from the negative target. Gradients never
     flow back through the rollout itself, only through the LoRA weights.
  4. Periodic EMA refresh: every `ema_every` steps, the behavior adapter is
     EMA-updated towards the trainable adapter, so the next round's on-policy
     queries come from an (slowly) improving policy.

Scale-down vs. the paper (and why):
  - SD-Turbo (865M UNet, 1-4 step) instead of SD3.5-M (10-step, gated,
    multi-GB) / Z-Image-Turbo -- fits in unified memory, no HF gating.
  - CLIPScore (openai/clip-vit-base-patch32, 151M) instead of HPSv3 (7B) /
    DeQA (8B) -- the paper's "differentiable reward" interface is preserved,
    just with a much smaller multimodal reward model.
  - ~40 training prompts / tens of optimizer steps instead of ~100 updates
    over the full 25k-prompt Pick-a-Pic split -- this is a from-scratch LoRA
    on top of a frozen base model, meant to demonstrate the OPSD mechanism
    end-to-end and produce a visible before/after reward gain, not to match
    the paper's benchmark numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image
from peft import LoraConfig

from reward_model import ClipRewardModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "stabilityai/sd-turbo"


def load_prompts(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def build_pipe(device: str):
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    return pipe


def add_lora_adapters(unet, r: int, alpha: int):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights=True,
    )
    unet.add_adapter(cfg, adapter_name="trainable")
    unet.add_adapter(cfg, adapter_name="behavior")
    # B is zero-initialized by peft, so both adapters start as a zero delta
    # (= identical to the frozen base model) regardless of A's random init.


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


def decode_to_image01(vae, latents: torch.Tensor) -> torch.Tensor:
    image = vae.decode(latents / vae.config.scaling_factor).sample
    return (image / 2 + 0.5).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "train_prompts.txt"))
    ap.add_argument("--rounds", type=int, default=60, help="number of OPSD optimizer steps")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--reward-step-size", type=float, default=0.6,
                     help="bounded step size (in latent units) used to build +/- targets around the anchor")
    ap.add_argument("--lambda-neg", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.9)
    ap.add_argument("--ema-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    torch.manual_seed(args.seed)
    prompts = load_prompts(Path(args.prompts))
    print(f"[info] {len(prompts)} training prompts loaded from {args.prompts}")

    print("[info] loading SD-Turbo pipeline...")
    pipe = build_pipe(args.device)
    unet = pipe.unet
    add_lora_adapters(unet, args.lora_r, args.lora_alpha)

    for n, p in unet.named_parameters():
        p.requires_grad_(".trainable." in n and "lora_" in n)
    train_params = lora_params(unet, "trainable")
    print(f"[info] trainable LoRA params: {sum(p.numel() for p in train_params):,}")

    optimizer = torch.optim.AdamW(train_params, lr=args.lr)

    print("[info] loading CLIP reward model...")
    reward_model = ClipRewardModel(device=args.device)

    latent_size = args.resolution // 8
    sigma0, sigma1, sigma2 = 14.6146, 1.6129, 0.0
    t0, t1 = 999.0, 499.0

    metrics_f = metrics_path.open("w")
    t_start = time.time()

    for step in range(1, args.rounds + 1):
        batch_prompts = [prompts[i % len(prompts)] for i in
                          torch.randint(0, len(prompts), (args.batch_size,)).tolist()]
        ctx = encode_prompts(pipe, batch_prompts, args.device)
        text_feats = reward_model.encode_text(batch_prompts)

        z0 = torch.randn(args.batch_size, 4, latent_size, latent_size, device=args.device) * sigma0

        # ---- step 0: behavior policy rollout (no grad) ----
        unet.set_adapter("behavior")
        with torch.no_grad():
            scaled0 = z0 / ((sigma0 ** 2 + 1) ** 0.5)
            t0_b = torch.full((args.batch_size,), t0, device=args.device)
            eps0 = unet(scaled0, t0_b, encoder_hidden_states=ctx).sample
            z_query = z0 + eps0 * (sigma1 - sigma0)  # on-policy query state (noisy, sigma1)

        # ---- anchor estimate + reward gradient (behavior policy, grad only wrt input) ----
        z_query = z_query.detach().requires_grad_(True)
        t1_b = torch.full((args.batch_size,), t1, device=args.device)
        scaled_q = z_query / ((sigma1 ** 2 + 1) ** 0.5)
        eps1_anchor = unet(scaled_q, t1_b, encoder_hidden_states=ctx).sample
        anchor_x0 = z_query - sigma1 * eps1_anchor
        anchor_image = decode_to_image01(pipe.vae, anchor_x0)
        reward = reward_model.score(anchor_image, text_feats)
        grad_z = torch.autograd.grad(reward.sum(), z_query)[0]
        direction = grad_z / (grad_z.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)

        z_query = z_query.detach()
        z_pos = (z_query + args.reward_step_size * direction).detach()
        z_neg = (z_query - args.reward_step_size * direction).detach()

        with torch.no_grad():
            scaled_pos = z_pos / ((sigma1 ** 2 + 1) ** 0.5)
            eps_pos = unet(scaled_pos, t1_b, encoder_hidden_states=ctx).sample
            target_pos = z_pos - sigma1 * eps_pos

            scaled_neg = z_neg / ((sigma1 ** 2 + 1) ** 0.5)
            eps_neg = unet(scaled_neg, t1_b, encoder_hidden_states=ctx).sample
            target_neg = z_neg - sigma1 * eps_neg

        # ---- trainable policy: fit the same query state towards target_pos / away from target_neg ----
        unet.set_adapter("trainable")
        scaled_q_train = z_query / ((sigma1 ** 2 + 1) ** 0.5)
        eps1_train = unet(scaled_q_train, t1_b, encoder_hidden_states=ctx).sample
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
            "elapsed_s": round(time.time() - t_start, 1),
        }
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        print(f"[step {step:3d}/{args.rounds}] loss={rec['loss']:.4f} "
              f"(pos={rec['loss_pos']:.4f} neg={rec['loss_neg']:.4f}) "
              f"anchor_reward={rec['mean_anchor_reward']:.2f} "
              f"t={rec['elapsed_s']:.0f}s")

    metrics_f.close()

    # save the final trainable LoRA adapter
    unet.set_adapter("trainable")
    unet.save_lora_adapter(str(out_dir / "lora_trainable"), adapter_name="trainable")
    print(f"[ok] saved trainable LoRA adapter to {out_dir / 'lora_trainable'}")
    print(f"[ok] metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
