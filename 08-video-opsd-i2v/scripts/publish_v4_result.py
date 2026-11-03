"""Publish the immutable V4 run/evaluation summary into repository Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

START = "<!-- V4_RESULT_START -->"
END = "<!-- V4_RESULT_END -->"


def replace_block(path: Path, block: str) -> None:
    text = path.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"missing V4 markers in {path}")
    before, remainder = text.split(START, 1)
    _, after = remainder.split(END, 1)
    path.write_text(f"{before}{START}\n{block.rstrip()}\n{END}{after}")


def mean(rows: list[dict], key: str, count: int = 100, tail: bool = False) -> float:
    chosen = rows[-count:] if tail else rows[:count]
    return sum(float(row[key]) for row in chosen) / len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--candidate-tag", default="v4_step2000")
    args = parser.parse_args()

    metrics_path = args.run_dir / "train_metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    last_step = int(rows[-1]["step"])
    complete = last_step == 2000 and (args.run_dir / "lora_step02000").is_dir() and not (args.run_dir / "GUARD_STOP.json").exists()
    figure = str((args.run_dir / "fig_v4_summary.png").relative_to(args.repo_root))

    report_lines = [
        "# V4 final report",
        "",
        f"- Recorded steps: **{last_step}/2000**",
        f"- Guard stop: **{'no' if complete else 'yes/incomplete'}**",
        f"- `loss_plus`, first→last 100: **{mean(rows, 'loss_plus'):.5f} → {mean(rows, 'loss_plus', tail=True):.5f}**",
        f"- Raw parameter-gradient norm, first→last 100: **{mean(rows, 'raw_param_grad_norm'):.6f} → {mean(rows, 'raw_param_grad_norm', tail=True):.6f}**",
        f"- Effective post-clip norm, last 100: **{mean(rows, 'param_grad_norm', tail=True):.6f}**",
        "",
    ]

    if complete:
        eval_path = args.eval_dir / f"metrics_{args.candidate_tag}.json"
        if not eval_path.is_file():
            raise SystemExit(f"completed training but missing evaluation: {eval_path}")
        evaluation = json.loads(eval_path.read_text())
        summaries = evaluation["summaries"]
        deltas = evaluation["candidate_minus_base"]
        improved = [name for name, value in deltas.items() if value["favorable_ci_excludes_zero"]]
        report_lines.extend(
            [
                "## Paired held-out result",
                "",
                "| Metric | Base | V4 | V4−Base mean [95% CI] |",
                "|---|---:|---:|---:|",
            ]
        )
        labels = {"appeal": "Appeal", "nat": "Naturalness", "dynamic": "Dynamic degree"}
        for key in ("appeal", "nat", "dynamic"):
            value = deltas[key]
            report_lines.append(
                f"| {labels[key]} | {summaries['base'][key]:.4f} | {summaries[args.candidate_tag][key]:.4f} | "
                f"{value['mean']:+.4f} [{value['lo']:+.4f}, {value['hi']:+.4f}] |"
            )
        claim = ", ".join(labels[key] for key in improved) if improved else "none"
        report_lines.extend(["", f"Metrics with a favorable 95% interval excluding zero: **{claim}**.", ""])
        demo_cells = []
        for index in range(6):
            gif = f"08-video-opsd-i2v/docs/figures/eval_v4/eval_{index:03d}_base_vs_candidate.gif"
            base_mp4 = f"08-video-opsd-i2v/outputs/eval_v4/videos_base/seed_00/{index:03d}.mp4"
            candidate_mp4 = f"08-video-opsd-i2v/outputs/eval_v4/videos_{args.candidate_tag}/seed_00/{index:03d}.mp4"
            demo_cells.append(
                f'<img src="{gif}" alt="COCO held-out {index}: Base versus V4"><br>'
                f'<a href="{base_mp4}">Base MP4</a> · <a href="{candidate_mp4}">V4 MP4</a>'
            )
        demo_table = "\n".join(
            f"<tr><td width=\"50%\">{demo_cells[index]}</td><td width=\"50%\">{demo_cells[index + 1]}</td></tr>"
            for index in range(0, 6, 2)
        )
        block = f"""### 08 · V4 held-out I2V result

![V4 training and stability curves]({figure})

{chr(10).join(report_lines[report_lines.index('## Paired held-out result') + 2:])}

<table>
{demo_table}
</table>

The six demos are fixed indices 0–5 from a 40-pair COCO-validation set not used for training or checkpoint selection. Base and V4 use identical first frame, caption, and generation seed. [Full operation record](08-video-opsd-i2v/operations/v4/README.md)."""
    else:
        guard = {}
        guard_path = args.run_dir / "GUARD_STOP.json"
        if guard_path.is_file():
            guard = json.loads(guard_path.read_text())
        reason = guard.get("reason", "training process ended before step 2000")
        block = f"""### 08 · V4 stopped before completion

![V4 partial training and stability curves]({figure})

V4 recorded **{last_step}/2000** steps and did not enter held-out quality evaluation. Stop reason: `{reason}`. This is retained as a failed stability result, not presented as a quality improvement. [Full operation record](08-video-opsd-i2v/operations/v4/README.md)."""
        report_lines.extend(["## Decision", "", block])

    report = args.run_dir / "FINAL_REPORT.md"
    report.write_text("\n".join(report_lines) + "\n")
    replace_block(args.repo_root / "README.md", block)

    project_block = block.replace("08-video-opsd-i2v/", "")
    project_readme = args.repo_root / "08-video-opsd-i2v" / "README.md"
    if START in project_readme.read_text():
        replace_block(project_readme, project_block)
    results_doc = args.repo_root / "08-video-opsd-i2v" / "docs" / "RESULTS_LONG.md"
    if START in results_doc.read_text():
        replace_block(results_doc, project_block)
    print(f"published V4 block; complete={complete}; report={report}")


if __name__ == "__main__":
    main()
