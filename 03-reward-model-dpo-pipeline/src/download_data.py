"""
下载公开偏好数据集 Anthropic/hh-rlhf(helpful-base 子集, Hugging Face 官方数据集, 无需
登录/Token), 作为"数字人生成质量用户偏好"场景的公开替代数据: 每条样本包含同一段多轮对话
上下文下, 人类标注员更偏好(chosen)与较差(rejected)的两个候选回复, 结构上与简历中
"同一输入、不同候选结果的 chosen-rejected 偏好对"完全一致。
"""
import json
import os

from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
N_TRAIN = 8000
N_TEST = 1500


def parse_example(chosen: str, rejected: str):
    # hh-rlhf 格式: "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ..."
    # 取最后一轮 "Assistant:" 之前的内容作为 prompt, 之后的内容作为回复
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
