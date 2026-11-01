"""C1 · measure REAL expression dynamics in LivePortrait coefficient space.

This is the number the whole `expr` reward term is waiting for.

Why it must be measured here, not with MediaPipe
------------------------------------------------
`motion_reward_v5.py` scores `coef[..., :63]`, which is *normalised* LivePortrait
expression: `exp_raw = coef * std_exp + mean_exp` (see `engine.denorm_pose`).
MediaPipe blendshapes are a different parameterisation, so a target read off
MediaPipe cannot be used directly -- that is exactly the mistake documented in
`INDUSTRY_ROUND.md` §7.7 (a unit-mismatched target turned the eyes half-closed).

So: run LivePortrait's own motion extractor over real public footage, crop with
LivePortrait's own Cropper (identical alignment), normalise with the model's own
template, and band-pass with the *same* FFT mask the reward uses. Then the median
is directly usable as `--expr-target`.

Output: outputs/behaviour_bands.json
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))

FPS = 25.0
N_MOTIONS = None          # filled from the engine (JoyVASA window length)


def read_frames(path, max_frames, target_fps=FPS):
    """Return list of RGB uint8 frames, subsampled to ~target_fps."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, int(round(src_fps / target_fps)))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdtf-dir", default=os.path.join(ROOT, "data", "videos_public", "hdtf"))
    ap.add_argument("--celebv-dir", default=os.path.join(ROOT, "data", "videos_public", "celebv"))
    ap.add_argument("--max-frames", type=int, default=250, help="~10 s per clip at 25 fps")
    ap.add_argument("--limit", type=int, default=0, help="cap clips per source (0 = all)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "behaviour_bands.json"))
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))

    from engine import TalkingHeadEngine                      # noqa: E402
    from motion_reward import _bandpass, POSE_DIMS            # noqa: E402
    from motion_reward_v5 import NON_LIP                      # noqa: E402

    from src.config.argument_config import ArgumentConfig     # noqa: E402
    from src.config.crop_config import CropConfig             # noqa: E402
    from src.utils.cropper import Cropper                     # noqa: E402

    eng = TalkingHeadEngine(load_renderer=False)
    W = eng.n_motions                       # frames per training window (4 s)
    tpl = eng.tpl
    std_exp = torch.as_tensor(tpl["std_exp"], dtype=torch.float32)     # (63,)
    mean_exp = torch.as_tensor(tpl["mean_exp"], dtype=torch.float32)
    pose_lo = torch.tensor([float(tpl[f"min_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    pose_hi = torch.tensor([float(tpl[f"max_{k}"][0]) for k in ("pitch", "yaw", "roll")])

    args = ArgumentConfig()
    crop_cfg = CropConfig(**{k: v for k, v in args.__dict__.items() if hasattr(CropConfig, k)})
    cropper = Cropper(crop_cfg=crop_cfg)
    print(f"[c1] window={W} frames ({W / FPS:.1f}s)  device={eng.device}", flush=True)

    groups = {
        "finance": sorted(glob.glob(os.path.join(a.hdtf_dir, "*.mp4"))),
        "celebv": sorted(glob.glob(os.path.join(a.celebv_dir, "*.mp4"))),
    }
    if a.limit:
        groups = {k: v[: a.limit] for k, v in groups.items()}

    rows = []
    for gname, paths in groups.items():
        for pi, p in enumerate(paths):
            try:
                fr = read_frames(p, a.max_frames)
                if len(fr) < W:
                    print(f"  [skip] {os.path.basename(p)}: only {len(fr)} frames", flush=True)
                    continue
                out = cropper.crop_source_video(fr, crop_cfg)
                crops = out["frame_crop_lst"]
                if len(crops) < W:
                    print(f"  [skip] {os.path.basename(p)}: crop failed", flush=True)
                    continue
                x = torch.from_numpy(np.asarray(crops)).permute(0, 3, 1, 2).float() / 255.0
                x = x.to(eng.device)
                exp_norm, pose_deg = [], []
                with torch.no_grad():
                    for s in range(0, len(x), 32):
                        kp = eng.wrapper.get_kp_info(x[s:s + 32])
                        e = kp["exp"].reshape(-1, 63).float().cpu()
                        exp_norm.append((e - mean_exp) / std_exp)
                        pose_deg.append(torch.stack(
                            [kp["pitch"].reshape(-1), kp["yaw"].reshape(-1), kp["roll"].reshape(-1)], -1)
                            .float().cpu())
                exp_norm = torch.cat(exp_norm)                       # (T,63) normalised
                pose_deg = torch.cat(pose_deg)                       # (T,3) degrees
                pose_n = (pose_deg - pose_lo) / (pose_hi - pose_lo)  # -> coef space

                n_win = len(exp_norm) // W
                for w in range(n_win):
                    e = exp_norm[w * W:(w + 1) * W][:, NON_LIP].unsqueeze(0)
                    pn = pose_n[w * W:(w + 1) * W].unsqueeze(0)
                    low_e = _bandpass(e, 0.3, 3.0)
                    std_e = (low_e.pow(2).mean(1) + 1e-6).sqrt().mean(-1).item()
                    low_p, high_p = _bandpass(pn, 0.3, 3.0), _bandpass(pn, 6.0, FPS / 2)
                    std_lp = (low_p.pow(2).mean(1) + 1e-6).sqrt()
                    std_hp = (high_p.pow(2).mean(1) + 1e-6).sqrt()
                    deg = (std_lp * (pose_hi - pose_lo)).norm(dim=-1).item() / np.sqrt(3)
                    deg_hi = (std_hp * (pose_hi - pose_lo)).norm(dim=-1).item() / np.sqrt(3)
                    rows.append({"group": gname, "clip": os.path.basename(p),
                                 "expr_low_std": std_e, "pose_low_deg": deg, "pose_high_deg": deg_hi})
                print(f"  [ok] {gname}/{os.path.basename(p)}: {n_win} windows", flush=True)
            except Exception as e:
                print(f"  [skip] {gname}/{os.path.basename(p)}: {type(e).__name__}: {e}", flush=True)

    if not rows:
        print("[c1] no rows -- nothing measured", flush=True)
        return


    def stat(vals):
        v = np.asarray(vals, dtype=float)
        if len(v) == 0:
            return {"n": 0, "median": None, "p25": None, "p75": None,
                    "mean": None, "max": None}
        return {"n": int(len(v)), "median": float(np.median(v)),
                "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
                "mean": float(v.mean()), "max": float(v.max())}

    def band(rows_sel):
        return {"expr_low_std": stat([r["expr_low_std"] for r in rows_sel]),
                "pose_low_deg": stat([r["pose_low_deg"] for r in rows_sel]),
                "pose_high_deg": stat([r["pose_high_deg"] for r in rows_sel])}

    fin = [r for r in rows if r["group"] == "finance"]
    cel = [r for r in rows if r["group"] == "celebv"]
    # same proxy split as INDUSTRY_ROUND: CelebV-HQ by measured head amplitude
    if cel:
        amps = sorted(r["pose_low_deg"] for r in cel)
        cut = amps[len(amps) // 2]
        low = [r for r in cel if r["pose_low_deg"] <= cut]
        high = [r for r in cel if r["pose_low_deg"] > cut]
    else:
        low, high = [], []

    result = {"source": "LivePortrait motion extractor, coefficient space",
              "window_frames": W, "fps": FPS,
              "n_windows_total": len(rows),
              "industries": {"finance": band(fin),
                             "realestate": band(low),
                             "shortvideo": band(high)},
              "all": band(rows),
              "rows": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=1)
    print(f"\n[c1] {len(rows)} windows -> {a.out}", flush=True)
    for k, v in result["industries"].items():
        e = v["expr_low_std"]
        print(f"  {k:12s} expr_low_std median={e['median']:.4f} "
              f"[IQR {e['p25']:.4f}-{e['p75']:.4f}]  n={e['n']}", flush=True)


if __name__ == "__main__":
    main()
