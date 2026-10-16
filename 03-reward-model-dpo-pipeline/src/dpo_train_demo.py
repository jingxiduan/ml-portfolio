"""
DPO training smoke test: runs a few dozen steps of `trl.DPOTrainer` on the gold DPO data
produced earlier (outputs/dpo_train.jsonl), verifying that the full "preference data
construction -> Reward Model filtering -> DPO training" pipeline runs end-to-end.

Defaults to the tiny model sshleifer/tiny-gpt2 for a CPU smoke test (finishes in minutes);
training quality is not the goal here. For real training results, swap MODEL_NAME for
something like Qwen2.5-0.5B-Instruct / gpt2, run on a GPU, and increase max_steps — the
rest of this script can be reused as-is.
"""
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_NAME = os.environ.get("DPO_DEMO_MODEL", "sshleifer/tiny-gpt2")
MAX_STEPS = int(os.environ.get("DPO_DEMO_STEPS", "20"))


def main():
    train_path = os.path.join(OUT_DIR, "dpo_train.jsonl")
    ds = load_dataset("json", data_files=train_path, split="train")
    ds = ds.select(range(min(300, len(ds))))  # only use a small subset for the smoke test, to run faster

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    config = DPOConfig(
        output_dir=os.path.join(OUT_DIR, "dpo_smoke_test"),
        per_device_train_batch_size=2,
        max_steps=MAX_STEPS,
        learning_rate=1e-5,
        logging_steps=5,
        report_to=[],
        max_length=256,
        max_prompt_length=128,
        beta=0.1,
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=ds,
    )
    result = trainer.train()
    print("DPO smoke-test training finished:", result)


if __name__ == "__main__":
    main()
