"""Render one audio/image pair under several engine configs and score each video.

Used for the smoke test and later for building engine "feature profiles".
"""
import argparse
import json
import os
import time

import torch

from engine import RenderOptions, TalkingHeadEngine
from video_metrics import evaluate_video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = {
    # what JoyVASA's CLI does out of the box: whole image squeezed to 256, 512 output, EMA a=0.01 on pose
    "official_default": dict(smooth=0.01, opts=RenderOptions(
        flag_do_crop=False, flag_pasteback=False, flag_stitching=False, flag_relative_motion=False,
        driving_option="expression-friendly", driving_multiplier=1.1)),
    # face crop rendered at 512 and pasted back into the original-resolution photo, light smoothing
    "hd_pasteback_light_smooth": dict(smooth=0.3, opts=RenderOptions()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "render_compare"))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    eng = TalkingHeadEngine(load_renderer=True)
    audio = eng.load_audio(a.audio, a.max_seconds)
    audio_path = a.audio
    if a.max_seconds is not None:
        import soundfile as sf

        audio_path = os.path.join(a.out, "_audio_clip.wav")
        sf.write(audio_path, audio.numpy(), 16000)
    coef = eng.gen_motion(audio, cfg_scale=a.cfg_scale, seed=a.seed)
    torch.save(coef, os.path.join(a.out, "motion_coef.pt"))

    results = {}
    for name in a.configs.split(","):
        c = CONFIGS[name]
        t0 = time.time()
        drv = eng.coef_to_driving(coef, smooth_alpha=c["smooth"])
        vid = eng.render(a.image, audio_path, drv, a.out, name, c["opts"])
        feats = evaluate_video(vid, audio_path)
        feats["render_sec"] = time.time() - t0
        results[name] = feats
        print(name, json.dumps({k: round(v, 3) for k, v in feats.items()}), flush=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
