#!/usr/bin/env bash
# Generate paper figures from the collected JSON summaries + wandb logs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
source .venv/bin/activate
source .envrc

mkdir -p outputs/figures

python3 - <<'PY'
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Main table bar chart
csv = Path("outputs/tables/main_table.csv")
if csv.is_file():
    df = pd.read_csv(csv)
    dims = [c for c in df.columns if c != "method"]
    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(dims))
    width = 0.2
    for i, row in enumerate(df.itertuples(index=False)):
        offsets = [xx + (i - len(df)/2) * width for xx in x]
        ax.bar(offsets, [getattr(row, d) for d in dims], width=width, label=row.method)
    ax.set_xticks(list(x))
    ax.set_xticklabels(dims, rotation=20, ha="right")
    ax.set_ylabel("VBench-I2V score")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/main_bar.pdf")
    print("[fig] outputs/figures/main_bar.pdf")

# Judge win-rate
judge_dir = Path("outputs/judge")
if judge_dir.is_dir():
    rows = []
    for p in sorted(judge_dir.glob("opsd_vs_*.json")):
        tag = p.stem.replace("opsd_vs_", "")
        data = json.loads(p.read_text())
        for rk, r in data.items():
            if "a_winrate" in r:
                rows.append({"vs": tag, "rubric": rk, "opsd_winrate": r["a_winrate"]})
    if rows:
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 4))
        for vs, sub in df.groupby("vs"):
            ax.bar([f"{r}\n(vs {vs})" for r in sub.rubric], sub.opsd_winrate, label=vs)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, ls="--", c="gray")
        ax.set_ylabel("OPSD-I2V win rate")
        plt.tight_layout()
        plt.savefig("outputs/figures/judge_winrate.pdf")
        print("[fig] outputs/figures/judge_winrate.pdf")
PY

echo "[done] figures in outputs/figures/"
