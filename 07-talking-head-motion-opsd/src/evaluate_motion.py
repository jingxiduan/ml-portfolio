"""Held-out evaluation: base JoyVASA vs OPSD-trained, same audio / same noise seed.

Two levels:
  motion   reward terms on generated coefficients, 8 eval clips x N seeds (cheap, no rendering)
  video    render each eval clip with both models (same face, same render config) and score the
           videos with the engine-agnostic MediaPipe metrics in video_metrics.py. This is the
           honest test: it does not reuse the eye probe or the training reward.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from engine import RenderOptions, TalkingHeadEngine
from eye_probe import EyeProbe
from motion_reward import LivelinessReward
from opsd_train_motion import add_lora, audio_env_frames, set_mode
from video_metrics import evaluate_video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES = ["host_f.png", "host_m.png", "host_f2.png"]


def load_trained(eng, path):
    ck = torch.load(path, map_location="cpu")
    layers = add_lora(eng.mg.denoising_net, ck["r"], ck["alpha"])
    for i, l in enumerate(layers):
        l.A_t.data.copy_(ck["lora"][f"{i}.A_t"].to(l.A_t.device))
        l.B_t.data.copy_(ck["lora"][f"{i}.B_t"].to(l.B_t.device))
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=os.path.join(ROOT, "outputs", "opsd", "lora.pt"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smooth", type=float, default=0.3)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "eval"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    eng = TalkingHeadEngine(load_renderer=not a.no_video)
    load_trained(eng, a.lora)
    probe = EyeProbe()
    probe.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "eye_probe", "eye_probe.pt"), map_location="cpu"))
    rew = LivelinessReward(eng.tpl, eye_probe=probe.to(eng.device)).to(eng.device)

    clips = sorted(glob.glob(os.path.join(ROOT, "data", "audio", "eval", "*.wav")))
    motion_rows = []
    coefs = {}
    for ci, clip in enumerate(clips):
        audio = eng.load_audio(clip)
        for seed in range(a.seeds):
            for model in ("base", "opsd"):
                set_mode("base" if model == "base" else "trainable")
                coef = eng.gen_motion(audio, cfg_scale=2.0, seed=seed)
                env = audio_env_frames(audio, coef.shape[0])
                with torch.no_grad():
                    r, terms = rew(coef.unsqueeze(0).to(eng.device), env.unsqueeze(0).to(eng.device),
                                   return_terms=True)
                motion_rows.append({"clip": ci, "seed": seed, "model": model, "reward": r.item(),
                                    **{k: v.item() for k, v in terms.items()}})
                if seed == 0:
                    coefs[(ci, model)] = coef
    summ = {}
    for m in ("base", "opsd"):
        rows = [r for r in motion_rows if r["model"] == m]
        summ[m] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("clip", "seed", "model")}
    pairs = [(r1, r2) for r1 in motion_rows if r1["model"] == "base" for r2 in motion_rows
             if r2["model"] == "opsd" and r2["clip"] == r1["clip"] and r2["seed"] == r1["seed"]]
    summ["win_rate"] = float(np.mean([r2["reward"] > r1["reward"] for r1, r2 in pairs]))
    print("[motion]", json.dumps(summ, indent=1))
    result = {"motion_space": summ, "motion_rows": motion_rows}

    if not a.no_video:
        vrows = []
        for ci, clip in enumerate(clips):
            face = os.path.join(ROOT, "data", "portraits", FACES[ci % len(FACES)])
            for model in ("base", "opsd"):
                drv = eng.coef_to_driving(coefs[(ci, model)], smooth_alpha=a.smooth)
                vid = eng.render(face, clip, drv, a.out, f"eval{ci:02d}_{model}",
                                 RenderOptions(flag_pasteback=False))
                f = evaluate_video(vid, clip)
                vrows.append({"clip": ci, "model": model, **f})
                print("[video]", ci, model, json.dumps({k: round(v, 3) for k, v in f.items()}), flush=True)
        keys = [k for k in vrows[0] if k not in ("clip", "model")]
        vsum = {m: {k: float(np.mean([r[k] for r in vrows if r["model"] == m])) for k in keys} for m in ("base", "opsd")}
        result["video"] = {"summary": vsum, "rows": vrows}
        print("[video summary]", json.dumps(vsum, indent=1))
    with open(os.path.join(a.out, "eval_results.json"), "w") as fo:
        json.dump(result, fo, indent=1)


if __name__ == "__main__":
    main()
