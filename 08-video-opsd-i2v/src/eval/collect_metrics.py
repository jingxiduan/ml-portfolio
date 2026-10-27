"""Aggregate VBench + LLM-judge results into a single Markdown table + CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_vbench_summary(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    results = data.get("results", data)
    out: Dict[str, float] = {}
    for k, v in results.items():
        # VBench returns {"<dim>": [score, {...}]}; take the scalar.
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            out[k] = float(v[0])
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="method:summary_json pairs, e.g. base:outputs/vbench_eval/base/summary.json "
             "opsd_i2v:outputs/vbench_eval/opsd_i2v/summary.json",
    )
    ap.add_argument("--out_dir", default="outputs/tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for item in args.methods:
        name, path = item.split(":", 1)
        vals = read_vbench_summary(Path(path))
        rows.append({"method": name, **vals})

    dims = sorted({k for r in rows for k in r if k != "method"})

    # CSV
    csv_path = out_dir / "main_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + dims)
        for r in rows:
            w.writerow([r["method"]] + [f"{r.get(d, float('nan')):.4f}" for d in dims])
    print(f"[collect] wrote {csv_path}")

    # Markdown
    md_path = out_dir / "main_table.md"
    with md_path.open("w") as f:
        f.write("| method | " + " | ".join(dims) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(dims)) + "|\n")
        for r in rows:
            cells = [f"{r.get(d, float('nan')):.3f}" for d in dims]
            f.write(f"| {r['method']} | " + " | ".join(cells) + " |\n")
    print(f"[collect] wrote {md_path}")


if __name__ == "__main__":
    main()
