"""
Exports the gold-standard preference data into the standard DPO training format
({"prompt", "chosen", "rejected"} JSONL), ready to feed directly into the Hugging Face
`trl` library's `DPOTrainer`.
"""
import json
import os

import pandas as pd

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def export(split: str):
    df = pd.read_parquet(os.path.join(PROC_DIR, f"{split}_preference.parquet"))
    gold = df[df["is_gold"]]

    out_path = os.path.join(OUT_DIR, f"dpo_{split}.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        for _, row in gold.iterrows():
            f.write(json.dumps({
                "prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"],
            }, ensure_ascii=False) + "\n")

    funnel = {
        "split": split,
        "n_raw_pairs": int(len(df)),
        "n_ambiguous_sent_to_judge": int(df["is_ambiguous"].sum()),
        "n_noisy_discarded_by_judge": int(df["is_noisy"].sum()),
        "n_gold_for_dpo": int(len(gold)),
        "gold_ratio": float(len(gold) / len(df)),
    }
    print(json.dumps(funnel, indent=2, ensure_ascii=False))
    return funnel


def main():
    funnels = [export(s) for s in ["train", "test"]]
    with open(os.path.join(OUT_DIR, "dpo_data_funnel.json"), "w") as f:
        json.dump(funnels, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
