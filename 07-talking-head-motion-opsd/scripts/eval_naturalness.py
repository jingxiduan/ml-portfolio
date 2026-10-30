"""Score every variant against the REAL professional-speech band, same units.

For base / v2 / v3 / v4 this reports exactly the quantities measured on real
HDTF footage by scripts/calibrate_reward_targets.py, so the comparison is
apples-to-apples:

    low-band pose std (deg)   real median 1.878
    blink rate (per min)      real median 15.0
    high-band jitter std(deg) real median 0.107

"Distance to real" is the whole point: a bigger reward number means nothing if
the motion sits outside the human band.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))

FPS = 25.0
POSE_DIMS = [67, 68, 69]

REAL = {"low_std_deg": 1.878, "blink_per_min": 15.0, "high_std_deg": 0.107}

VARIANTS = [
    ("base", None),
    ("v2", "outputs/opsd_v2_large/lora.pt"),
    ("v3", "outputs/opsd_v3_large/lora.pt"),
    ("v4", "outputs/opsd_v4/lora.pt"),
]


def band_std(x, lo, hi, fs=FPS):
    from scipy.signal import butter, filtfilt
    n = len(x)
    if n < 24:
        return 0.0
    ny = fs / 2.0
    b, a = butter(2, [lo / ny, min(hi / ny, 0.99)], btype="band")
    return float(np.std(filtfilt(b, a, x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "naturalness_eval.json"))
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()

    from engine import TalkingHeadEngine
    import opsd_train_motion as T
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    eng = TalkingHeadEngine(load_renderer=False)
    layers = T.add_lora(eng.mg.denoising_net, 8, 8)
    lo = torch.tensor([float(eng.tpl[f"min_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    hi = torch.tensor([float(eng.tpl[f"max_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    rng = (hi - lo).numpy()

    clips = a.clips or sorted(glob.glob(os.path.join(ROOT, "data", "audio_public", "celebv", "*.wav")))[: a.n]
    if not clips:
        clips = sorted(glob.glob(os.path.join(ROOT, "data", "audio_public", "eval", "*.wav")))[: a.n]
    print(f"[eval] {len(clips)} clips")

    rows = []
    for name, lora in VARIANTS:
        if lora and not os.path.exists(lora):
            print(f"  [skip] {name}: no checkpoint")
            continue
        if lora:
            ck = torch.load(lora, map_location="cpu")["lora"]
            with torch.no_grad():
                for i, l in enumerate(layers):
                    l.A_t.data.copy_(ck[f"{i}.A_t"].to(l.A_t.device))
                    l.B_t.data.copy_(ck[f"{i}.B_t"].to(l.B_t.device))
            T.set_mode("trainable")     # MUST match the weights we loaded
        else:
            T.set_mode("base")
        ls, hs = [], []
        for w in clips:
            au = eng.load_audio(w, a.max_seconds)
            with torch.no_grad():
                c = eng.gen_motion(au, cfg_scale=2.0, seed=0)
            P = c[:, POSE_DIMS].numpy() * rng
            ls.append(np.mean([band_std(P[:, j], 0.3, 3.0) for j in range(3)]))
            hs.append(np.mean([band_std(P[:, j], 6.0, 12.5) for j in range(3)]))
        r = {"variant": name,
             "low_std_deg": float(np.median(ls)),
             "high_std_deg": float(np.median(hs)),
             "ratio_vs_real_amp": float(np.median(ls)) / REAL["low_std_deg"]}
        rows.append(r)
        print(f"  {name:<6} low_std={r['low_std_deg']:6.3f}deg  "
              f"high_std={r['high_std_deg']:6.3f}deg  "
              f"amp/real={r['ratio_vs_real_amp']:5.2f}x")

    out = {"real_band": REAL, "variants": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n[ok] REAL target: low_std={REAL['low_std_deg']} deg, "
          f"blink={REAL['blink_per_min']}/min")
    print("[ok] wrote", a.out)


if __name__ == "__main__":
    main()
