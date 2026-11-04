"""Score blinded A/B responses without overstating participant independence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def wilson(wins: float, total: float, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def participant_bootstrap(rows: list[dict], seed: int = 20261104, draws: int = 10000) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["participant_id"]].append(row["candidate_score"])
    participant_ids = sorted(grouped)
    if not participant_ids:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        chosen = [rng.choice(participant_ids) for _ in participant_ids]
        values = [value for participant in chosen for value in grouped[participant]]
        samples.append(sum(values) / len(values))
    samples.sort()
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("human-study/results.json"))
    args = parser.parse_args()

    key = {item["trial_id"]: item for item in json.loads(args.key.read_text())["key"]}
    records = []
    seen = set()
    with args.responses.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            participant = raw["participant_id"].strip()
            trial_id = raw["trial_id"].strip()
            choice = raw["choice"].strip().upper()
            if not participant or choice not in {"A", "B", "TIE"} or trial_id not in key:
                continue
            identity = (participant, trial_id)
            if identity in seen:
                raise SystemExit(f"duplicate response: {identity}")
            seen.add(identity)
            answer = key[trial_id]
            score = 0.5 if choice == "TIE" else float(choice == answer["candidate_side"])
            records.append(
                {
                    "participant_id": participant,
                    "trial_id": trial_id,
                    "project": answer["project"],
                    "choice": choice,
                    "candidate_score": score,
                }
            )

    summaries = {}
    for project in ("07", "08", "all"):
        rows = records if project == "all" else [row for row in records if row["project"] == project]
        wins = sum(row["candidate_score"] for row in rows)
        ties = sum(row["choice"] == "TIE" for row in rows)
        n = len(rows)
        summaries[project] = {
            "n_responses": n,
            "n_participants": len({row["participant_id"] for row in rows}),
            "candidate_preference_rate_ties_half": wins / n if n else None,
            "tie_rate": ties / n if n else None,
            "wilson_ci_response_level": wilson(wins, n),
            "participant_cluster_bootstrap_ci95": participant_bootstrap(rows),
        }
    result = {"summaries": summaries, "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
