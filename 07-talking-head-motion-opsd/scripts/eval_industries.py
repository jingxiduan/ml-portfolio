"""Score every industry persona against its own measured natural band.

This is the evaluation that answers "did the method help, and is it natural?".
It is deliberately independent of the training reward in one respect: the clips
come from the AISHELL-1 **eval** split, whose 15 speakers are disjoint from the
60 training speakers (asserted in build_audio_segments.py).

For each variant we generate motion on the same clips and measure, in degrees:
    low_std  0.3-3.0 Hz   deliberate head motion   -> compared to the industry target
    high_std 6.0-12.5 Hz  tremor / jitter          -> compared to the industry target
and report the ratio to the target. 1.0 means the persona sits exactly on the
band measured from that industry's real footage. Below 1.0 = too stiff,
above 1.0 = overshooting into uncanny territory.

Writes outputs/industry_eval.json.
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


def band_std(x, lo, hi, fs=FPS):
    from scipy.signal import butter, filtfilt
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 24:
        return 0.0
    ny = fs / 2.0
    b, a = butter(2, [lo / ny, min(hi / ny, 0.99)], btype="band")
    return float(np.std(filtfilt(b, a, x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default=os.path.join(ROOT, "outputs", "industry_bands.json"))
    ap.add_argument("--n-clips", type=int, default=12)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "industry_eval.json"))
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    from engine import TalkingHeadEngine
    import opsd_train_motion as T

    eng = TalkingHeadEngine(load_renderer=False)
    layers = T.add_lora(eng.mg.denoising_net, 8, 8)
    lo = torch.tensor([float(eng.tpl[f"min_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    hi = torch.tensor([float(eng.tpl[f"max_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    rng = (hi - lo).numpy()

    clips = sorted(glob.glob(os.path.join(ROOT, "data", "audio_public", "eval", "*.wav")))[: a.n_clips]
    bands = json.load(open(a.bands))["industries"]
    print(f"[eval] {len(clips)} held-out clips (speakers disjoint from training)")

    variants = [("base", None)] + [(k, os.path.join(ROOT, "outputs", f"opsd_ind_{k}", "lora.pt"))
                                   for k in bands]
    rows = []
    for name, path in variants:
        if path and not os.path.exists(path):
            print(f"  [skip] {name}: no checkpoint")
            continue
        if path:
            ck = torch.load(path, map_location="cpu")["lora"]
            with torch.no_grad():
                for i, l in enumerate(layers):
                    l.A_t.data.copy_(ck[f"{i}.A_t"].to(l.A_t.device))
                    l.B_t.data.copy_(ck[f"{i}.B_t"].to(l.B_t.device))
            T.set_mode("trainable")
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
             "low_std_deg": float(np.median(ls)), "high_std_deg": float(np.median(hs))}
        # compare against this variant's own target band when it has one
        tgt = bands.get(name)
        if tgt:
            r["target_low_deg"] = tgt["low_std_deg"]["median"]
            r["target_high_deg"] = tgt["high_std_deg"]["median"]
            r["ratio_vs_target_amp"] = r["low_std_deg"] / max(tgt["low_std_deg"]["median"], 1e-6)
            r["ratio_vs_target_jitter"] = r["high_std_deg"] / max(tgt["high_std_deg"]["median"], 1e-6)
        rows.append(r)
        s = (f"  target={r['target_low_deg']:.2f}/{r['target_high_deg']:.3f} "
             f"amp_ratio={r['ratio_vs_target_amp']:.2f}x jit_ratio={r['ratio_vs_target_jitter']:.2f}x"
             if tgt else "  (reference)")
        print(f"  {name:<12} low={r['low_std_deg']:6.3f}deg high={r['high_std_deg']:6.4f}deg{s}")

    json.dump({"n_clips": len(clips), "variants": rows}, open(a.out, "w"), indent=1)
    print("[ok] wrote", a.out)


if __name__ == "__main__":
    main()
