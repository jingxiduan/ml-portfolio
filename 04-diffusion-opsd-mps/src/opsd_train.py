"""Single-GPU canonical-structure DiffusionOPSD adaptation for SDXL-Turbo."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import save_file

from reward_model import ClipRewardModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "stabilityai/sdxl-turbo"


def load_prompt_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def adapter_parameters(unet, adapter_name: str) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in unet.named_parameters()
        if f".{adapter_name}." in name and "lora_" in name
    }


def copy_adapter(unet, source: str, target: str) -> None:
    params = dict(unet.named_parameters())
    copied = 0
    with torch.no_grad():
        for name, value in params.items():
            marker = f".{source}."
            if marker in name and "lora_" in name:
                target_name = name.replace(marker, f".{target}.")
                params[target_name].copy_(value)
                copied += 1
    if copied == 0:
        raise RuntimeError("no LoRA parameters copied")


@torch.no_grad()
def ema_adapter(unet, source: str, target: str, decay: float) -> None:
    params = dict(unet.named_parameters())
    for name, value in params.items():
        marker = f".{source}."
        if marker in name and "lora_" in name:
            target_value = params[name.replace(marker, f".{target}.")]
            target_value.mul_(decay).add_(value, alpha=1.0 - decay)


def trust_region_targets(
    anchor: torch.Tensor, gradient: torch.Tensor, rho: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dims = tuple(range(1, anchor.ndim))
    anchor_norm = anchor.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    grad_norm = gradient.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
    displacement = rho * anchor_norm * gradient / grad_norm
    plus = (anchor + displacement).detach()
    minus = (anchor - displacement).detach()
    ratio = displacement.flatten(1).norm(dim=1) / anchor.flatten(1).norm(dim=1).clamp_min(1e-12)
    return plus, minus, ratio


def dual_branch_loss(
    z_query: torch.Tensor,
    sigma: torch.Tensor,
    eps_theta: torch.Tensor,
    eps_old: torch.Tensor,
    target_plus: torch.Tensor,
    target_minus: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v_pos = beta * eps_theta.float() + (1.0 - beta) * eps_old.float()
    v_neg = (1.0 + beta) * eps_old.float() - beta * eps_theta.float()
    y_pos = z_query.float() - sigma * v_pos
    y_neg = z_query.float() - sigma * v_neg
    dims = tuple(range(1, y_pos.ndim))
    with torch.no_grad():
        weight_pos = (y_pos - target_plus).abs().mean(dim=dims, keepdim=True).clamp_min(1e-5)
        weight_neg = (y_neg - target_minus).abs().mean(dim=dims, keepdim=True).clamp_min(1e-5)
    loss_pos = (((y_pos - target_plus) ** 2) / weight_pos).mean(dim=dims)
    loss_neg = (((y_neg - target_minus) ** 2) / weight_neg).mean(dim=dims)
    return 0.5 * (loss_pos + loss_neg).mean(), loss_pos.mean(), loss_neg.mean()


def encode_prompts(pipe, prompts: list[str], device: torch.device):
    with torch.no_grad():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return prompt_embeds, pooled_prompt_embeds


def make_time_ids(batch: int, resolution: int, device: torch.device, dtype: torch.dtype):
    values = [resolution, resolution, 0, 0, resolution, resolution]
    return torch.tensor([values], device=device, dtype=dtype).repeat(batch, 1)


def predict_noise(
    unet,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    sigma: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    time_ids: torch.Tensor,
) -> torch.Tensor:
    scaled = latents / torch.sqrt(sigma * sigma + 1.0)
    return unet(
        scaled,
        timestep,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids},
        return_dict=False,
    )[0]


def decode_for_reward(vae, clean_latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(clean_latents / vae.config.scaling_factor, return_dict=False)[0]
    image_raw = decoded / 2.0 + 0.5
    image_clamped = image_raw.clamp(0.0, 1.0)
    return image_raw + (image_clamped - image_raw).detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("scene", "portrait"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.inference_steps < 2:
        raise ValueError("inference-steps must be at least 2")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this reproducible run targets CUDA")
    out_dir = Path(args.out or ROOT / "outputs" / args.domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = ROOT / "data" / f"train_{args.domain}.jsonl"
    rows = load_prompt_rows(prompt_path)
    prompts = [item["prompt"] for item in rows]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    if pipe.scheduler.config.prediction_type != "epsilon":
        raise RuntimeError(f"unsupported prediction type: {pipe.scheduler.config.prediction_type}")
    pipe.vae.to(dtype=torch.float32).requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights=True,
    )
    pipe.unet.add_adapter(config, adapter_name="trainable")
    pipe.unet.add_adapter(config, adapter_name="behavior")
    copy_adapter(pipe.unet, "trainable", "behavior")

    trainable = adapter_parameters(pipe.unet, "trainable")
    behavior = adapter_parameters(pipe.unet, "behavior")
    if trainable.keys() != behavior.keys():
        train_suffixes = {name.replace(".trainable.", ".ADAPTER.") for name in trainable}
        behavior_suffixes = {name.replace(".behavior.", ".ADAPTER.") for name in behavior}
        if train_suffixes != behavior_suffixes:
            raise RuntimeError("trainable and behavior adapter structures differ")
    for parameter in trainable.values():
        parameter.requires_grad_(True)
        parameter.data = parameter.data.float()
    train_params = list(trainable.values())
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
    reward_model = ClipRewardModel(device=args.device)
    quality_positive, quality_negative = reward_model.encode_quality_targets()

    pipe.scheduler.set_timesteps(args.inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps.clone()
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)
    query_index = args.inference_steps - 1
    sigma_query_value = float(sigmas[query_index].item())
    if not 0.0 < sigma_query_value < float(sigmas[0].item()):
        raise RuntimeError("query state is not low-noise")

    module_names = sorted({name.split(".lora_")[0] for name in trainable})
    run_config = vars(args) | {
        "model_id": MODEL_ID,
        "prompt_manifest": str(prompt_path.relative_to(ROOT)),
        "n_training_prompts": len(prompts),
        "scheduler": pipe.scheduler.__class__.__name__,
        "timesteps": [float(value) for value in timesteps.cpu()],
        "sigmas": [float(value) for value in sigmas.cpu()],
        "query_index": query_index,
        "query_sigma": sigma_query_value,
        "trainable_parameters": sum(value.numel() for value in train_params),
        "adapter_tensor_count": len(train_params),
        "target_module_count": len(module_names),
        "dtype": "float16 backbone / float32 LoRA and reward path",
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    (out_dir / "target_modules.json").write_text(json.dumps(module_names, indent=2) + "\n")

    metrics_path = out_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("w")
    start = time.time()
    prompt_rng = random.Random(args.seed)

    for step in range(1, args.rounds + 1):
        batch_prompts = [prompt_rng.choice(prompts) for _ in range(args.batch_size)]
        prompt_embeds, pooled_prompt_embeds = encode_prompts(pipe, batch_prompts, device)
        time_ids = make_time_ids(
            args.batch_size, args.resolution, device, prompt_embeds.dtype
        )
        text_features = reward_model.encode_text(batch_prompts)

        latent_size = args.resolution // pipe.vae_scale_factor
        z = torch.randn(
            args.batch_size,
            pipe.unet.config.in_channels,
            latent_size,
            latent_size,
            device=device,
            dtype=torch.bfloat16,
        ) * sigmas[0]

        pipe.unet.set_adapter("behavior")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in range(query_index):
                sigma = sigmas[index].view(1, 1, 1, 1)
                eps = predict_noise(
                    pipe.unet,
                    z,
                    timesteps[index],
                    sigma,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    time_ids,
                )
                z = z + eps * (sigmas[index + 1] - sigmas[index])
            sigma_query = sigmas[query_index].view(1, 1, 1, 1)
            eps_old = predict_noise(
                pipe.unet,
                z,
                timesteps[query_index],
                sigma_query,
                prompt_embeds,
                pooled_prompt_embeds,
                time_ids,
            )
            anchor = z.float() - sigma_query * eps_old.float()

        anchor_leaf = anchor.detach().requires_grad_(True)
        with torch.autocast("cuda", enabled=False):
            image = decode_for_reward(pipe.vae, anchor_leaf)
            reward, reward_parts = reward_model.score(
                image,
                text_features,
                quality_positive,
                quality_negative,
                quality_weight=args.quality_weight,
            )
            reward_gradient = torch.autograd.grad(reward.sum(), anchor_leaf)[0]
        if not torch.isfinite(reward_gradient).all():
            raise RuntimeError(f"non-finite reward gradient at step {step}")
        gradient_norm = reward_gradient.flatten(1).norm(dim=1)
        if (gradient_norm <= 1e-12).any():
            raise RuntimeError(f"zero reward gradient at step {step}")
        target_plus, target_minus, displacement_ratio = trust_region_targets(
            anchor_leaf.detach(), reward_gradient.detach(), args.rho
        )
        del image, anchor_leaf, reward_gradient

        pipe.unet.set_adapter("trainable")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eps_theta = predict_noise(
                pipe.unet,
                z.detach(),
                timesteps[query_index],
                sigma_query,
                prompt_embeds,
                pooled_prompt_embeds,
                time_ids,
            )
        loss, loss_pos, loss_neg = dual_branch_loss(
            z.detach(),
            sigma_query,
            eps_theta,
            eps_old.detach(),
            target_plus,
            target_minus,
            args.beta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"non-finite policy gradient at step {step}")
        optimizer.step()
        ema_adapter(pipe.unet, "trainable", "behavior", args.ema_decay)

        record = {
            "step": step,
            "loss": float(loss.item()),
            "loss_pos": float(loss_pos.item()),
            "loss_neg": float(loss_neg.item()),
            "anchor_reward": float(reward.mean().item()),
            "anchor_alignment": float(reward_parts["alignment"].mean().item()),
            "anchor_quality_margin": float(reward_parts["quality_margin"].mean().item()),
            "display_penalty": float(reward_parts["display_penalty"].mean().item()),
            "reward_grad_norm": float(gradient_norm.mean().item()),
            "policy_grad_norm_pre_clip": float(grad_norm),
            "target_displacement_ratio": float(displacement_ratio.mean().item()),
            "elapsed_s": round(time.time() - start, 2),
            "max_cuda_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        }
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        if step == 1 or step % args.log_every == 0:
            print(
                f"[{args.domain} {step:04d}/{args.rounds}] "
                f"loss={record['loss']:.5f} reward={record['anchor_reward']:.3f} "
                f"rgrad={record['reward_grad_norm']:.4g} "
                f"pgrad={record['policy_grad_norm_pre_clip']:.4g} "
                f"mem={record['max_cuda_gib']:.1f}GiB",
                flush=True,
            )

    metrics_file.close()
    for adapter_name, directory_name in (("behavior", "lora_ema"), ("trainable", "lora_raw")):
        adapter_dir = out_dir / directory_name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(pipe.unet, adapter_name=adapter_name)
        state = {name: value.detach().cpu().contiguous() for name, value in state.items()}
        save_file(state, str(adapter_dir / "adapter.safetensors"))
    print(f"saved adapters and metrics to {out_dir}")


if __name__ == "__main__":
    main()
