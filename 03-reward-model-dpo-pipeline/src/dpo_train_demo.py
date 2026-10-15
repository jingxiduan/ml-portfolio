"""
DPO 训练冒烟测试: 用 `trl.DPOTrainer` 在上一步产出的金标 DPO 数据(outputs/dpo_train.jsonl)
上跑几十个 step, 验证"偏好数据构建 -> Reward Model 筛选 -> DPO 训练"全链路可以端到端跑通。

默认用极小模型 sshleifer/tiny-gpt2 在 CPU 上做 smoke test(几分钟内完成), 不追求训练效果;
如需真实效果, 把 MODEL_NAME 换成 Qwen2.5-0.5B-Instruct / gpt2 等并接入 GPU 环境、增大
max_steps 即可直接复用本脚本。
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
    ds = ds.select(range(min(300, len(ds))))  # smoke test 只取一小部分, 加速跑通

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
