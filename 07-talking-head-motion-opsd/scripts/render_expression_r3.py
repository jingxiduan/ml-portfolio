"""Render a matched base/seed0/seed1/seed2 R3 expression comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from engine import RenderOptions, TalkingHeadEngine  # noqa: E402
from render_industry_demo import HD_OPTS, load_lora  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--portrait",
        type=Path,
        default=ROOT / "data" / "portraits" / "host_f.png",
    )
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "expression_demo_r3")
    args = parser.parse_args()

    if args.audio is None:
        candidates = sorted((ROOT / "data" / "audio_public" / "eval").glob("*.wav"))
        if not candidates:
            raise SystemExit("no held-out audio found")
        args.audio = candidates[0]
    if not args.portrait.is_file() or not args.audio.is_file():
        raise SystemExit("portrait or audio input is missing")

    args.out.mkdir(parents=True, exist_ok=True)
    engine = TalkingHeadEngine(load_renderer=True)
    audio = engine.load_audio(str(args.audio), args.seconds)
    options = RenderOptions(**HD_OPTS)

    variants = [("base", None)] + [
        (f"seed{seed}", ROOT / "outputs" / f"opsd_v5_r3_seed{seed}" / "lora.pt")
        for seed in range(3)
    ]
    entries = []
    for name, adapter in variants:
        if adapter is not None and not adapter.is_file():
            raise SystemExit(f"missing adapter: {adapter}")
        load_lora(engine, None if adapter is None else str(adapter))
        coef = engine.gen_motion(audio, cfg_scale=2.0, seed=args.seed)
        driving = engine.coef_to_driving(coef, smooth_alpha=0.3)
        engine.render(
            str(args.portrait),
            str(args.audio),
            driving,
            str(args.out),
            f"expression_r3__{name}",
            options,
        )
        entries.append(
            {
                "variant": name,
                "video": str((args.out / f"expression_r3__{name}.mp4").relative_to(ROOT)),
                "adapter": None if adapter is None else str(adapter.relative_to(ROOT)),
            }
        )
        print(f"[render] {name}", flush=True)

    manifest = {
        "portrait": str(args.portrait.relative_to(ROOT)),
        "audio": str(args.audio.relative_to(ROOT)),
        "seconds": args.seconds,
        "generation_seed": args.seed,
        "matched_inputs": True,
        "entries": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
