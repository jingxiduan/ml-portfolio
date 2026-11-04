#!/usr/bin/env python3
"""One-command CPU mock and GPU talking-head demo entry point."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _gpu_snapshot() -> list[dict[str, Any]]:
    query = "index,memory.used,memory.total,utilization.gpu"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "memory_used_mib": int(fields[1]),
                "memory_total_mib": int(fields[2]),
                "utilization_gpu_percent": int(fields[3]),
            }
        )
    return rows


def _mock_svg() -> str:
    return """<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1200 540\" role=\"img\" aria-label=\"CPU mock talking-head comparison\">
  <defs>
    <linearGradient id=\"bg\" x1=\"0\" x2=\"1\"><stop stop-color=\"#101827\"/><stop offset=\"1\" stop-color=\"#243b62\"/></linearGradient>
    <linearGradient id=\"skin\" x1=\"0\" x2=\"1\"><stop stop-color=\"#f3c7ab\"/><stop offset=\"1\" stop-color=\"#d89477\"/></linearGradient>
    <style>
      .panel { fill: #0b1220; stroke: #405477; stroke-width: 3; }
      .label { font: 700 25px sans-serif; fill: #e8eefc; }
      .sub { font: 16px sans-serif; fill: #9eb0d2; }
      .head { fill: url(#skin); stroke: #8a5a4b; stroke-width: 4; transform-box: fill-box; transform-origin: center; }
      .mouth { fill: #9c4b55; transform-box: fill-box; transform-origin: center; }
      .calibrated .head { animation: nod 1.2s ease-in-out infinite alternate; }
      .calibrated .mouth { animation: speak 0.42s ease-in-out infinite alternate; }
      .base .mouth { animation: speak 0.42s ease-in-out infinite alternate; animation-delay: .15s; }
      @keyframes nod { from { transform: rotate(-2deg) translateY(2px); } to { transform: rotate(2deg) translateY(-3px); } }
      @keyframes speak { from { transform: scaleY(.45); } to { transform: scaleY(1.55); } }
    </style>
  </defs>
  <rect width=\"1200\" height=\"540\" rx=\"28\" fill=\"url(#bg)\"/>
  <g transform=\"translate(35 32)\"><rect class=\"panel\" width=\"545\" height=\"476\" rx=\"22\"/><text class=\"label\" x=\"28\" y=\"42\">BASE</text><text class=\"sub\" x=\"28\" y=\"68\">frozen motion policy</text>
    <g class=\"base\" transform=\"translate(272 270)\"><ellipse class=\"head\" cx=\"0\" cy=\"0\" rx=\"125\" ry=\"155\"/><circle cx=\"-45\" cy=\"-18\" r=\"10\" fill=\"#182235\"/><circle cx=\"45\" cy=\"-18\" r=\"10\" fill=\"#182235\"/><path d=\"M-42 55 Q0 65 42 55\" fill=\"none\" stroke=\"#7f3c45\" stroke-width=\"9\" stroke-linecap=\"round\"/><ellipse class=\"mouth\" cx=\"0\" cy=\"56\" rx=\"22\" ry=\"7\"/></g>
  </g>
  <g transform=\"translate(620 32)\"><rect class=\"panel\" width=\"545\" height=\"476\" rx=\"22\"/><text class=\"label\" x=\"28\" y=\"42\">CALIBRATED POLICY</text><text class=\"sub\" x=\"28\" y=\"68\">bounded expression target</text>
    <g class=\"calibrated\" transform=\"translate(272 270)\"><ellipse class=\"head\" cx=\"0\" cy=\"0\" rx=\"125\" ry=\"155\"/><circle cx=\"-45\" cy=\"-18\" r=\"10\" fill=\"#182235\"/><circle cx=\"45\" cy=\"-18\" r=\"10\" fill=\"#182235\"/><path d=\"M-42 55 Q0 75 42 55\" fill=\"none\" stroke=\"#7f3c45\" stroke-width=\"9\" stroke-linecap=\"round\"/><ellipse class=\"mouth\" cx=\"0\" cy=\"56\" rx=\"22\" ry=\"7\"/></g>
  </g>
</svg>"""


def run_mock(out: Path) -> None:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    svg = _mock_svg()
    (out / "comparison.svg").write_text(svg, encoding="utf-8")
    html = f"""<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><title>Talking-head CPU mock</title>
<style>body{{margin:0;background:#080d18;color:#e8eefc;font:16px system-ui,sans-serif}}main{{max-width:1200px;margin:32px auto;padding:0 20px}}img{{width:100%;border-radius:18px;box-shadow:0 20px 60px #0008}}a{{color:#8ab4ff}}.note{{color:#9eb0d2}}</style>
<main><h1>Talking-head motion: CPU mock</h1><p class=\"note\">This dependency-free animation validates the demo contract only; it is not a model render.</p><img src=\"comparison.svg\" alt=\"Base versus calibrated policy mock comparison\"><h2>Pre-generated model render</h2><p><a href=\"../../new_face_demo_r3/female/female_new_face_base_vs_r3.gif\">Female held-out Base vs calibrated GIF</a> · <a href=\"../../new_face_demo_r3/male/male_new_face_base_vs_r3.gif\">Male held-out Base vs calibrated GIF</a></p></main></html>\n"""
    (out / "index.html").write_text(html, encoding="utf-8")
    elapsed_ms = (time.perf_counter() - started) * 1000
    manifest = {
        "mode": "mock",
        "cpu_only": True,
        "elapsed_ms": round(elapsed_ms, 2),
        "output": "index.html",
        "model_render": False,
        "limitations": ["No model weights are loaded", "The animation is a UI and pipeline smoke test"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[mock] wrote {out / 'index.html'} ({elapsed_ms:.0f} ms)")


def run_gpu(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("GPU demo requires PyTorch; run scripts/h20_render_setup.sh first") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Use --mode mock on CPU or install a CUDA PyTorch build.")
    portrait = args.portrait.resolve()
    audio = args.audio.resolve()
    out = args.out.resolve()
    if not portrait.is_file():
        raise SystemExit(f"missing portrait: {portrait}")
    if not audio.is_file():
        raise SystemExit(f"missing audio: {audio}")
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi is required to record GPU memory")
    out.mkdir(parents=True, exist_ok=True)
    before = _gpu_snapshot()
    peak = list(before)
    stop = threading.Event()

    def monitor() -> None:
        nonlocal peak
        while not stop.is_set():
            sample = _gpu_snapshot()
            for row in sample:
                previous = next((item for item in peak if item["index"] == row["index"]), None)
                if previous is None or row["memory_used_mib"] > previous["memory_used_mib"]:
                    if previous is not None:
                        peak.remove(previous)
                    peak.append(row)
            stop.wait(1.0)

    command = [
        sys.executable,
        str(Path(__file__).with_name("render_expression_r3.py")),
        "--portrait",
        str(portrait),
        "--audio",
        str(audio),
        "--seconds",
        str(args.seconds),
        "--seed",
        str(args.seed),
        "--out",
        str(out),
    ]
    print("[gpu]", " ".join(command), flush=True)
    started = time.perf_counter()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        stop.set()
        thread.join(timeout=2)
    elapsed_s = time.perf_counter() - started
    after = _gpu_snapshot()
    manifest = {
        "mode": "gpu",
        "command": command,
        "seconds": args.seconds,
        "elapsed_seconds": round(elapsed_s, 3),
        "gpu_before": before,
        "gpu_peak_sampled": peak,
        "gpu_after": after,
        "note": "Peak memory is sampled with nvidia-smi once per second; it is not a CUDA allocator trace.",
    }
    (out / "benchmark.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[gpu] finished in {elapsed_s:.1f}s; benchmark: {out / 'benchmark.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "gpu"), required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "demo" / "cpu_mock")
    parser.add_argument("--portrait", type=Path, default=ROOT / "data" / "portraits" / "host_f.png")
    parser.add_argument("--audio", type=Path, default=ROOT / "data" / "audio" / "demo" / "demo_00.wav")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "mock":
        run_mock(args.out.resolve())
    else:
        run_gpu(args)


if __name__ == "__main__":
    main()
