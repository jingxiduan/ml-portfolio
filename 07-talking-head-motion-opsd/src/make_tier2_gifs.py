"""Build 3-way comparison GIFs from Tier-2 BoN output mp4s.

Reads outputs/tier2/bon_cXX_{base_naive,opsd_naive,opsd_bon_thin}.mp4 (or
opsd_bon if _thin missing) and side-by-side composites them with labels.
"""
import argparse
import glob
import os

import cv2
import imageio.v3 as iio
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def label(frame: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    # black translucent bar
    bar_h = max(26, h // 15)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = bar_h / 36
    cv2.putText(out, text, (8, int(bar_h * 0.72)), font, scale, color, 1, cv2.LINE_AA)
    return out


def load_frames(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def hstack_with_label(frames_dict, labels):
    n = min(len(fr) for fr in frames_dict.values())
    h = min(fr[0].shape[0] for fr in frames_dict.values())
    out = []
    for i in range(n):
        row = []
        for key, lbl in labels:
            f = frames_dict[key][i]
            if f.shape[0] != h:
                scale = h / f.shape[0]
                f = cv2.resize(f, (int(f.shape[1] * scale), h))
            row.append(label(f, lbl))
        out.append(np.hstack(row))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(ROOT, "outputs", "tier2"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "tier2"))
    ap.add_argument("--prefix", default="bon_c")
    ap.add_argument("--max-side", type=int, default=320, help="downscale each panel to this height")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    base_vids = sorted(glob.glob(os.path.join(a.in_dir, f"{a.prefix}*_base_naive.mp4")))
    for base_vid in base_vids:
        stem = os.path.basename(base_vid).replace("_base_naive.mp4", "")  # bon_cXX
        tag = stem.replace(a.prefix, "")
        # Prefer the 3-way comparison that tells the strongest story:
        # base | opsd v1 naive (reward-hacked) | opsd v2 + BoN (fixed via retraining)
        # Fall back to v1-only story (base | opsd naive | opsd + BoN + thin) if v2 absent.
        cand_v2 = base_vid.replace("_base_naive.mp4", "_opsd_v2_bon.mp4")
        cand_v1_thin = base_vid.replace("_base_naive.mp4", "_opsd_bon_thin.mp4")
        cand_v1_bon = base_vid.replace("_base_naive.mp4", "_opsd_bon.mp4")
        if os.path.exists(cand_v2):
            fix_path = cand_v2
            fix_label = "+ OPSD v2 + BoN"
        elif os.path.exists(cand_v1_thin):
            fix_path = cand_v1_thin
            fix_label = "+ OPSD v1 + BoN + thin"
        elif os.path.exists(cand_v1_bon):
            fix_path = cand_v1_bon
            fix_label = "+ OPSD v1 + BoN"
        else:
            print(f"[skip] {stem}: no fix variant found", flush=True)
            continue
        paths = {
            "base": base_vid,
            "opsd": base_vid.replace("_base_naive.mp4", "_opsd_naive.mp4"),
            "fix": fix_path,
        }
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"[skip] {stem}: missing variants", flush=True)
            continue
        frames = {k: load_frames(p)[0] for k, p in paths.items()}
        # subsample to target fps (source is 25 fps)
        src_fps = 25
        step = max(1, src_fps // a.fps)
        for k in frames:
            frames[k] = frames[k][::step]
        # downscale
        for k in frames:
            frs = frames[k]
            if frs[0].shape[0] > a.max_side:
                scale = a.max_side / frs[0].shape[0]
                frames[k] = [cv2.resize(f, (int(f.shape[1] * scale), a.max_side)) for f in frs]
        labels = [("base", "base (JoyVASA)"),
                  ("opsd", "+ OPSD v1 (naive, hacked)"),
                  ("fix", fix_label)]
        composite = hstack_with_label(frames, labels)
        out_path = os.path.join(a.out_dir, f"tier2_compare_{tag}.gif")
        iio.imwrite(out_path, composite, duration=1000 // a.fps, loop=0)
        print(f"[write] {out_path}  ({len(composite)} frames, {composite[0].shape[1]}x{composite[0].shape[0]})",
              flush=True)


if __name__ == "__main__":
    main()
