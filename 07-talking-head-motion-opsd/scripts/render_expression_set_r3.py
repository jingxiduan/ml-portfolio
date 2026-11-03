"""Render the held-out R3 base/seed0/seed1/seed2 comparison set."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from engine import RenderOptions, TalkingHeadEngine  # noqa: E402
from render_industry_demo import HD_OPTS, load_lora  # noqa: E402


def trim_audio(source: Path, destination: Path, seconds: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-t",
            str(seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portrait", type=Path, default=ROOT / "data" / "portraits" / "host_f.png")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "data" / "audio_public" / "eval")
    parser.add_argument("--n-clips", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "expression_render_set_r3")
    args = parser.parse_args()
    args.portrait = args.portrait.resolve()
    args.audio_dir = args.audio_dir.resolve()
    args.out = args.out.resolve()

    clips = sorted(args.audio_dir.glob("*.wav"))[: args.n_clips]
    if len(clips) != args.n_clips:
        raise SystemExit(f"requested {args.n_clips} clips, found {len(clips)}")
    if not args.portrait.is_file():
        raise SystemExit(f"missing portrait: {args.portrait}")

    args.out.mkdir(parents=True, exist_ok=True)
    audio_out = args.out / "audio"
    audio_out.mkdir(exist_ok=True)
    variants = [("base", None)] + [
        (f"seed{seed}", ROOT / "outputs" / f"opsd_v5_r3_seed{seed}" / "lora.pt")
        for seed in range(3)
    ]
    for _, adapter in variants:
        if adapter is not None and not adapter.is_file():
            raise SystemExit(f"missing adapter: {adapter}")

    engine = TalkingHeadEngine(load_renderer=True)
    options = RenderOptions(**HD_OPTS)
    entries = []
    for clip_index, source_audio in enumerate(clips):
        trimmed_audio = audio_out / f"clip{clip_index:03d}.wav"
        trim_audio(source_audio, trimmed_audio, args.seconds)
        audio = engine.load_audio(str(trimmed_audio), args.seconds)
        for name, adapter in variants:
            load_lora(engine, None if adapter is None else str(adapter))
            coef = engine.gen_motion(audio, cfg_scale=2.0, seed=args.seed)
            driving = engine.coef_to_driving(coef, smooth_alpha=0.3)
            stem = f"clip{clip_index:03d}__{name}"
            engine.render(
                str(args.portrait),
                str(trimmed_audio),
                driving,
                str(args.out),
                stem,
                options,
            )
            entries.append(
                {
                    "clip_index": clip_index,
                    "source_audio": str(source_audio.relative_to(ROOT)),
                    "matched_audio": str(trimmed_audio.relative_to(ROOT)),
                    "variant": name,
                    "video": str((args.out / f"{stem}.mp4").relative_to(ROOT)),
                    "adapter": None if adapter is None else str(adapter.relative_to(ROOT)),
                }
            )
            print(f"[render] {clip_index + 1}/{len(clips)} {name}", flush=True)

    manifest = {
        "portrait": str(args.portrait.relative_to(ROOT)),
        "n_clips": len(clips),
        "seconds": args.seconds,
        "generation_seed": args.seed,
        "matched_inputs": True,
        "entries": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
