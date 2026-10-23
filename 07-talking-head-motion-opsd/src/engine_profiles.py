"""Engine feature profiles: render a fixed probe set with each engine config and summarize the
engine-agnostic video metrics (mean and spread across faces x audios).

"Engine" here = a concrete, runnable configuration of the talking-head stack:
  motion model (base JoyVASA / OPSD-trained) x CFG scale x pose smoothing x driving amplitude.
The same profiling would apply unchanged to a third-party engine: only its output mp4s are needed.

Stability check: per-feature coefficient of variation across the probe set, plus a split-half
rank correlation (are engines ordered the same way on two disjoint halves of the probe set?).
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import soundfile as sf
import torch
from scipy.stats import spearmanr

from engine import RenderOptions, TalkingHeadEngine
from evaluate_motion import load_trained
from opsd_train_motion import set_mode
from video_metrics import evaluate_video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["head_pitch_std", "head_yaw_std", "head_roll_std", "head_speed_mean", "head_jitter", "nod_rate",
            "speech_head_corr", "still_ratio", "blink_rate", "gaze_std", "gaze_shift_rate", "jaw_open_mean",
            "jaw_audio_corr"]


def engine_grid():
    grid = []
    for model, cfg, smooth, mult in itertools.product(["base", "opsd"], [2.0], [0.01, 0.3], [0.8, 1.0, 1.25]):
        grid.append({"name": f"{model}_cfg{cfg}_s{smooth}_m{mult}", "model": model, "cfg": cfg, "smooth": smooth,
                     "mult": mult})
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "profiles"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eng = TalkingHeadEngine(load_renderer=True)
    load_trained(eng, os.path.join(ROOT, "outputs", "opsd", "lora.pt"))

    faces = [os.path.join(ROOT, "data", "portraits", f) for f in ["host_f.png", "host_m.png", "host_f2.png"]]
    audios = sorted(glob.glob(os.path.join(ROOT, "data", "audio", "eval", "*.wav")))[:1]
    probes = []
    for ai, ap_ in enumerate(audios):
        clip = eng.load_audio(ap_, a.seconds)
        p = os.path.join(a.out, f"_probe_audio{ai}.wav")
        sf.write(p, clip.numpy(), 16000)
        probes.append((ai, p, clip))

    cache = os.path.join(a.out, "profile_rows.jsonl")
    done = set()
    if os.path.exists(cache):
        done = {(r["engine"], r["face"], r["audio"]) for r in map(json.loads, open(cache))}
    fo = open(cache, "a")
    for e in engine_grid():
        set_mode("base" if e["model"] == "base" else "trainable")
        for ai, apath, clip in probes:
            coef = eng.gen_motion(clip, cfg_scale=e["cfg"], seed=ai)
            drv = eng.coef_to_driving(coef, smooth_alpha=e["smooth"])
            for fi, face in enumerate(faces):
                if (e["name"], fi, ai) in done:
                    continue
                vid = eng.render(face, apath, drv, a.out, "_tmp",
                                 RenderOptions(flag_pasteback=False, driving_multiplier=e["mult"]))
                feats = evaluate_video(vid, apath)
                row = {"engine": e["name"], **e, "face": fi, "audio": ai, **feats}
                fo.write(json.dumps(row) + "\n")
                fo.flush()
                print(e["name"], fi, ai, {k: round(feats[k], 2) for k in ("head_speed_mean", "nod_rate",
                                                                            "blink_rate", "gaze_std")}, flush=True)
    fo.close()
    if os.path.exists(os.path.join(a.out, "_tmp.mp4")):
        os.remove(os.path.join(a.out, "_tmp.mp4"))

    rows = [json.loads(l) for l in open(cache)]
    engines = [e["name"] for e in engine_grid()]
    prof, stab = {}, {}
    for en in engines:
        rs = [r for r in rows if r["engine"] == en]
        prof[en] = {f: float(np.mean([r[f] for r in rs])) for f in FEATURES}
        prof[en].update({f + "_sd": float(np.std([r[f] for r in rs])) for f in FEATURES})
    # split-half stability: profile from faces {0} vs faces {1,2}, ranks of engines per feature
    for f in FEATURES:
        h1 = [np.mean([r[f] for r in rows if r["engine"] == en and r["face"] == 0]) for en in engines]
        h2 = [np.mean([r[f] for r in rows if r["engine"] == en and r["face"] != 0]) for en in engines]
        between = np.std([prof[en][f] for en in engines])
        within = np.mean([prof[en][f + "_sd"] for en in engines])
        stab[f] = {"split_half_spearman": float(spearmanr(h1, h2).correlation),
                   "between_engine_sd": float(between), "within_engine_sd": float(within),
                   "separability": float(between / (within + 1e-8))}
    json.dump({"profiles": prof, "stability": stab, "engines": engine_grid()},
              open(os.path.join(a.out, "engine_profiles.json"), "w"), indent=1)
    print(json.dumps(stab, indent=1))


if __name__ == "__main__":
    main()
