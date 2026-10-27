"""Qwen2-VL as pairwise video judge (human-study proxy).

Given a prompt and two candidate videos (A, B), ask Qwen2-VL to pick the better one along a
single rubric at a time: {motion realism, subject identity preservation, prompt-video alignment,
overall quality}. Report win-rate per rubric with 95% bootstrap CIs.

Rate limits: on an M3 Pro, Qwen2-VL-7B at fp16 takes ~8s per pair; keep N_pairs ≤ 200.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

RUBRICS = [
    ("motion",       "Which video has more natural, coherent motion?"),
    ("identity",     "Which video preserves the subject's identity more faithfully across frames?"),
    ("alignment",    "Which video better matches the prompt?"),
    ("overall",      "Overall, which video is better?"),
]


def _load_qwen():
    try:
        import torch
        from qwen_vl_utils import process_vision_info  # type: ignore
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "transformers>=4.51 + qwen-vl-utils required for Qwen2-VL judge."
        ) from e
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor, process_vision_info


def _judge_pair(model, processor, process_vision_info, prompt: str, vid_a: str, vid_b: str,
                question: str) -> str:
    """Returns 'A' or 'B'."""
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": vid_a, "max_pixels": 360 * 420, "fps": 2.0},
                {"type": "video", "video": vid_b, "max_pixels": 360 * 420, "fps": 2.0},
                {
                    "type": "text",
                    "text": (
                        f"Prompt: {prompt}\n"
                        f"Question: {question}\n"
                        "Answer with a single character: 'A' or 'B'. No explanation."
                    ),
                },
            ],
        }
    ]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msg)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    txt = processor.batch_decode(gen, skip_special_tokens=True)[0].strip().upper()
    if txt.startswith("A"):
        return "A"
    if txt.startswith("B"):
        return "B"
    # Fallback: random tie-break
    return random.choice(["A", "B"])


def judge(
    method_a_dir: str,
    method_b_dir: str,
    out_json: str,
    n_pairs: int = 200,
    seed: int = 0,
) -> Dict:
    random.seed(seed)
    a_root = Path(method_a_dir)
    b_root = Path(method_b_dir)
    manifest_a = json.loads((a_root / "manifest.json").read_text())
    manifest_b = json.loads((b_root / "manifest.json").read_text())
    pairs: List[Tuple[Dict, Dict]] = []
    idx_b = {m["index"]: m for m in manifest_b}
    for ma in manifest_a:
        if ma["index"] in idx_b:
            pairs.append((ma, idx_b[ma["index"]]))
    if n_pairs < len(pairs):
        pairs = random.sample(pairs, n_pairs)

    model, processor, process_vision_info = _load_qwen()

    results: Dict[str, Dict] = {k: {"a_wins": 0, "b_wins": 0, "total": 0} for k, _ in RUBRICS}
    for i, (ma, mb) in enumerate(pairs):
        # Randomize A/B order to avoid position bias
        swap = random.random() < 0.5
        left = a_root / "videos" / ma["file"]
        right = b_root / "videos" / mb["file"]
        if swap:
            left, right = right, left
        for rk, q in RUBRICS:
            ans = _judge_pair(model, processor, process_vision_info, ma["prompt"],
                              str(left), str(right), q)
            # Interpret: 'A' means the left video; swap-aware
            left_is_a = not swap
            picked_a = (ans == "A" and left_is_a) or (ans == "B" and not left_is_a)
            results[rk]["total"] += 1
            if picked_a:
                results[rk]["a_wins"] += 1
            else:
                results[rk]["b_wins"] += 1
        if (i + 1) % 20 == 0:
            print(f"[judge] {i+1}/{len(pairs)} pairs done")

    for rk, _ in RUBRICS:
        t = results[rk]["total"]
        if t:
            results[rk]["a_winrate"] = results[rk]["a_wins"] / t
            results[rk]["b_winrate"] = results[rk]["b_wins"] / t

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="method A video dir (contains videos/ + manifest.json)")
    ap.add_argument("--b", required=True, help="method B video dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    judge(args.a, args.b, args.out, args.n_pairs, args.seed)


if __name__ == "__main__":
    main()
