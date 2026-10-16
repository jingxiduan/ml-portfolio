"""
Downloads the public preference dataset Anthropic/hh-rlhf (helpful-base subset, official
Hugging Face dataset, no login/token required): each sample contains, under the same
multi-turn conversation context, a human-labeled better (chosen) and worse (rejected)
candidate response — i.e. the standard "same input, different candidates, chosen-rejected
preference pair" structure.
"""
import json
import os

from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
N_TRAIN = 8000
N_TEST = 1500


def parse_example(chosen: str, rejected: str):
    # hh-rlhf format: "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ..."
    # take the content before the last "Assistant:" as the prompt, and the content after it as the response
    def split(text):
        idx = text.rfind("\n\nAssistant:")
        prompt = text[:idx].strip()
        response = text[idx + len("\n\nAssistant:"):].strip()
        return prompt, response

    prompt_c, resp_c = split(chosen)
    prompt_r, resp_r = split(rejected)
    prompt = prompt_c if len(prompt_c) >= len(prompt_r) else prompt_r
    return prompt, resp_c, resp_r


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=f"train[:{N_TRAIN}]")
    test_ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=f"test[:{N_TEST}]")

    for name, ds in [("train", train_ds), ("test", test_ds)]:
        out_path = os.path.join(OUT_DIR, f"{name}.jsonl")
        n_written = 0
        with open(out_path, "w") as f:
            for ex in ds:
                prompt, resp_c, resp_r = parse_example(ex["chosen"], ex["rejected"])
                if not resp_c or not resp_r or resp_c == resp_r:
                    continue
                f.write(json.dumps({
                    "prompt": prompt, "chosen": resp_c, "rejected": resp_r,
                }, ensure_ascii=False) + "\n")
                n_written += 1
        print(f"{name}: wrote {n_written} pairs -> {out_path}")


if __name__ == "__main__":
    main()
