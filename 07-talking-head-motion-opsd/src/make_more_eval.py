"""Append 12 new eval clips (eval_08..eval_19) without disturbing existing
train/eval/demo splits.  Keeps BoN's `selected_coefs.pt` for clips 00-07
reusable; the new ones are freshly selected on top.
"""
import json
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "audio")
EVAL = os.path.join(OUT, "eval")

# 12 fresh scripts (different domains, deliberately varied length / register)
EXTRA_SCRIPTS = [
    # tech explainer / long
    "大语言模型的训练分为预训练和微调两个阶段，预训练用的是海量无标注文本，让模型学会语言的统计规律。",
    "量子计算机和经典计算机最大的区别在于，它可以同时处于多个状态，这让某些问题的求解速度呈指数级提升。",
    "半导体制程缩小到五纳米以下之后，漏电流问题变得非常严重，这也是先进制程成本飙升的主要原因之一。",
    # health / medium
    "每天保持三十分钟的中等强度运动，比如快走或者游泳，可以显著降低心血管疾病的发病风险。",
    "维生素D主要是通过晒太阳合成的，上班族如果长期待在室内，可以考虑适当补充。",
    "睡眠分成浅睡期、深睡期和快速眼动期，深睡期对身体修复最重要，而快速眼动期对记忆巩固最重要。",
    # finance / short
    "指数基金的核心理念就是不去预测哪只股票会涨，而是跟随整个市场的平均表现。",
    "美联储加息会抬高美元存款的吸引力，所以通常会伴随着其他货币对美元的贬值。",
    "分散投资的关键不只是买很多只股票，更重要的是买不同类型、不同行业的资产。",
    # lifestyle / varied tone
    "养猫的朋友一定要定期给它们梳毛，不然舔进肚子里的毛太多，容易引起消化道堵塞。",
    "拍摄人像的时候，背景虚化不是越强越好，适度的环境信息反而能让照片更有故事感。",
    "泡茶的水温会影响口感，绿茶适合八十度左右，红茶和普洱可以用接近沸腾的水。",
]

VOICES = ["Tingting", "Eddy (中文（中国大陆）)", "Flo (中文（中国大陆）)", "Reed (中文（中国大陆）)",
          "Sandy (中文（中国大陆）)", "Shelley (中文（中国大陆）)", "Rocko (中文（中国大陆）)"]


def synth(text, voice, rate, wav):
    aiff = wav.replace(".wav", ".aiff")
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", aiff, text], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", aiff, "-ac", "1", "-ar", "16000", wav],
                   check=True)
    os.remove(aiff)


def main():
    assert os.path.exists(EVAL), "run src/make_audio.py first"
    rng = random.Random(42)
    os.makedirs(EVAL, exist_ok=True)
    manifest_add = []
    for i, text in enumerate(EXTRA_SCRIPTS, start=8):
        voice = VOICES[(i * 3 + 1) % len(VOICES)]
        rate = rng.choice([170, 185, 200, 215])
        wav = os.path.join(EVAL, f"eval_{i:02d}.wav")
        if os.path.exists(wav):
            print(f"skip {wav} (exists)")
            continue
        synth(text, voice, rate, wav)
        manifest_add.append({"split": "eval", "path": os.path.relpath(wav, ROOT),
                             "voice": voice, "rate": rate, "text": text})
        print(f"wrote {wav}")

    # append to manifest
    mpath = os.path.join(ROOT, "data", "audio_manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
    else:
        manifest = []
    manifest.extend(manifest_add)
    with open(mpath, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[done] +{len(manifest_add)} eval clips, manifest now has {len(manifest)} entries")


if __name__ == "__main__":
    main()
