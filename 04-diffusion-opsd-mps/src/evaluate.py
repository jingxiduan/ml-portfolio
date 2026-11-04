"""Paired multi-seed evaluation with an independent CLIP-H evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from diffusers import AutoPipelineForText2Image
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "stabilityai/sdxl-turbo"
EVALUATOR_ID = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ci(values: np.ndarray, seed: int = 3407, draws: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


class IndependentClipHEvaluator:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            EVALUATOR_ID, device=self.device
        )
        self.model.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(EVALUATOR_ID)

    @torch.no_grad()
    def score(self, image, prompt: str) -> float:
        pixels = self.preprocess(image).unsqueeze(0).to(self.device)
        tokens = self.tokenizer([prompt]).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image_features = self.model.encode_image(pixels)
            text_features = self.model.encode_text(tokens)
        image_features = image_features.float() / image_features.float().norm(dim=-1, keepdim=True)
        text_features = text_features.float() / text_features.float().norm(dim=-1, keepdim=True)
        return float((image_features * text_features).sum().item() * 100.0)


def generate(pipe, prompt: str, seed: int, resolution: int, steps: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,
        height=resolution,
        width=resolution,
        generator=generator,
    ).images[0]


def image_diagnostics(image) -> dict[str, float]:
    array = np.asarray(image, dtype=np.float32) / 255.0
    luma = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    saturation = array.max(axis=-1) - array.min(axis=-1)
    return {
        "clipped_fraction": float(((luma <= 0.02) | (luma >= 0.98)).mean()),
        "mean_saturation": float(saturation.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("scene", "portrait"), required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--lora-dir", default=None)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    manifest = ROOT / "data" / f"{args.split}_{args.domain}.jsonl"
    rows = load_rows(manifest)
    lora_dir = Path(args.lora_dir or ROOT / "outputs" / args.domain / "lora_ema")
    out_path = Path(args.out or ROOT / "outputs" / args.domain / "eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    evaluator = IndependentClipHEvaluator(args.device)

    records = []
    print(f"evaluating base on {len(rows)} frozen prompts x {len(args.seeds)} seeds")
    for prompt_index, item in enumerate(rows):
        for seed in args.seeds:
            image = generate(pipe, item["prompt"], seed, args.resolution, args.inference_steps)
            records.append({
                "source_id": item["source_id"],
                "prompt": item["prompt"],
                "seed": seed,
                "base_score": evaluator.score(image, item["prompt"]),
                "base_diagnostics": image_diagnostics(image),
            })
        print(f"base {prompt_index + 1:02d}/{len(rows)}", flush=True)

    run_config = json.loads((lora_dir.parent / "run_config.json").read_text())
    pipe.unet.add_adapter(
        LoraConfig(
            r=int(run_config["lora_r"]),
            lora_alpha=int(run_config["lora_alpha"]),
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
        adapter_name="candidate",
    )
    state = load_file(str(lora_dir / "adapter.safetensors"), device=args.device)
    load_result = set_peft_model_state_dict(pipe.unet, state, adapter_name="candidate")
    if load_result.unexpected_keys:
        raise RuntimeError(f"unexpected LoRA keys: {load_result.unexpected_keys}")
    pipe.unet.set_adapters("candidate", weights=args.adapter_scale)
    print(f"evaluating EMA candidate at scale={args.adapter_scale:g} with matched prompts and seeds")
    cursor = 0
    for prompt_index, item in enumerate(rows):
        for seed in args.seeds:
            image = generate(pipe, item["prompt"], seed, args.resolution, args.inference_steps)
            records[cursor]["candidate_score"] = evaluator.score(image, item["prompt"])
            records[cursor]["candidate_diagnostics"] = image_diagnostics(image)
            records[cursor]["delta"] = records[cursor]["candidate_score"] - records[cursor]["base_score"]
            cursor += 1
        print(f"candidate {prompt_index + 1:02d}/{len(rows)}", flush=True)

    prompt_deltas = []
    for item in rows:
        values = [record["delta"] for record in records if record["source_id"] == item["source_id"]]
        prompt_deltas.append(float(np.mean(values)))
    deltas = np.asarray([record["delta"] for record in records], dtype=np.float64)
    base_scores = np.asarray([record["base_score"] for record in records], dtype=np.float64)
    candidate_scores = np.asarray([record["candidate_score"] for record in records], dtype=np.float64)
    prompt_delta_array = np.asarray(prompt_deltas, dtype=np.float64)

    result = {
        "domain": args.domain,
        "model_id": MODEL_ID,
        "evaluator": EVALUATOR_ID,
        "evaluator_independent_of_training_reward": True,
        "adapter_scale": args.adapter_scale,
        "manifest": str(manifest.relative_to(ROOT)),
        "n_prompts": len(rows),
        "seeds": args.seeds,
        "n_pairs": len(records),
        "base_mean": float(base_scores.mean()),
        "candidate_mean": float(candidate_scores.mean()),
        "paired_delta_mean": float(deltas.mean()),
        "prompt_cluster_bootstrap_95_ci": bootstrap_ci(prompt_delta_array),
        "pair_win_rate": float((deltas > 0).mean()),
        "prompt_win_rate": float((prompt_delta_array > 0).mean()),
        "records": records,
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
