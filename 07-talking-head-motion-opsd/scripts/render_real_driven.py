"""Drive a portrait with the MOTION COEFFICIENTS OF A REAL HUMAN VIDEO.

Why this exists
---------------
Everything so far generates motion from audio only (JoyVASA). The audio-driven
model has no reason to move brow / cheek / eye-region, so the face is a static
mask inherited from the source portrait -- the "面具脸" the user sees.

This script takes the opposite route: run LivePortrait's own motion extractor
over a real English-speaking talking-head video, and use those 73-d coefficients
to drive the portrait directly. The result is a digital human that moves
*exactly like a real person moved*, and the coefficient statistics we read off
it are the ground-truth parameters the reward has never had.

It also renders the audio-only base with the same portrait / audio / seed, so the
pair is a controlled "before / after".

Outputs
-------
  <out>/<tag>__real.mp4     driven by the real human's coefficients
  <out>/<tag>__base.mp4     audio-only (JoyVASA), the baseline
  <out>/real_coef_stats.json   measured statistics of the real coefficients
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))

import cv2  # noqa: E402

from engine import RenderOptions, TalkingHeadEngine  # noqa: E402

FPS = 25.0
HD_OPTS = dict(
    flag_do_crop=True,
    flag_pasteback=True,
    flag_stitching=True,
    flag_relative_motion=True,
    driving_option="pose-friendly",
    driving_multiplier=1.0,
    source_max_dim=1280,
    animation_region="all",
)


def _eye_mouth_scores(rgb):
    """(eye aspect ratio, jaw-open blendshape) via MediaPipe, or (0, 1) if no face."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from video_metrics import MP_MODEL
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MP_MODEL),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=True)
        with vision.FaceLandmarker.create_from_options(opts) as lm:
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return 0.0, 1.0
        pts = np.array([[q.x, q.y] for q in res.face_landmarks[0]])
        # EAR on the two eyes (MediaPipe indices)
        def ear(ids):
            p = pts[ids]
            v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
            h = np.linalg.norm(p[0] - p[3])
            return float(v / (2.0 * h + 1e-9))
        e = min(ear([33, 159, 158, 133, 153, 145]), ear([362, 386, 385, 263, 380, 374]))
        jaw = 0.0
        if res.face_blendshapes:
            for b in res.face_blendshapes[0]:
                if b.category_name == "jawOpen":
                    jaw = float(b.score)
        return e, jaw
    except Exception as exc:
        print(f"  [warn] eye/mouth scoring failed: {type(exc).__name__}: {exc}")
        return 0.0, 1.0


def stabilize_scale_translation(coef, scale_band=0.02, win=51):
    """Kill the head-size pulsing.

    Measured on the rendered clips, face width fluctuated by ~2% std (p2/p98
    spread ~8%), which reads as the head swelling and shrinking. The `scale`
    and `t` coefficients carry that jitter. Median-filter them over ~2 s and
    clamp scale to +/-scale_band around its median: slow approach/withdraw
    survives, per-frame jitter does not.
    """
    import scipy.ndimage as ndi
    c = coef.clone()
    s = c[:, 63]
    med = torch.median(s)
    sm = torch.as_tensor(ndi.median_filter(s.cpu().numpy(), size=win, mode="nearest"),
                         dtype=s.dtype, device=s.device)
    c[:, 63] = torch.clamp(sm, med * (1 - scale_band), med * (1 + scale_band))
    t = c[:, 64:67]
    tm = torch.as_tensor(ndi.median_filter(t.cpu().numpy(), size=(win, 1), mode="nearest"),
                         dtype=t.dtype, device=t.device)
    c[:, 64:67] = tm
    return c


def read_frames(path, max_frames, target_fps=FPS):
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


def ffmpeg_executable():
    """Use system ffmpeg when present, otherwise imageio-ffmpeg's bundled binary."""
    import shutil

    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video, wav):
    if os.path.exists(wav) and os.path.getsize(wav) > 0:
        return wav
    subprocess.run([ffmpeg_executable(), "-y", "-loglevel", "error", "-i", video,
                    "-ac", "1", "-ar", "16000", "-vn", wav], check=True)
    return wav


def real_coefficients(eng, frames, cropper, crop_cfg):
    """Return (T, 73) normalised coefficients extracted from real frames."""
    from motion_reward import POSE_DIMS  # noqa: F401
    tpl = eng.tpl
    std_exp = torch.as_tensor(tpl["std_exp"], dtype=torch.float32).reshape(1, 63)
    std_exp = std_exp.clamp(min=1e-4)   # 17 dims are degenerate: no divide-by-zero
    mean_exp = torch.as_tensor(tpl["mean_exp"], dtype=torch.float32).reshape(1, 63)
    p_lo = torch.tensor([float(tpl[f"min_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    p_hi = torch.tensor([float(tpl[f"max_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    s_lo, s_hi = float(tpl["min_scale"][0]), float(tpl["max_scale"][0])
    t_lo = torch.as_tensor(tpl["min_t"], dtype=torch.float32).reshape(-1)
    t_hi = torch.as_tensor(tpl["max_t"], dtype=torch.float32).reshape(-1)

    out = cropper.crop_source_video(frames, crop_cfg)
    crops = out["frame_crop_lst"]
    x = torch.from_numpy(np.asarray(crops)).permute(0, 3, 1, 2).float() / 255.0
    x = x.to(eng.device)

    coefs = []
    with torch.no_grad():
        for s in range(0, len(x), 32):
            kp = eng.wrapper.get_kp_info(x[s:s + 32])
            exp = kp["exp"].reshape(-1, 63).float().cpu()
            exp_n = (exp - mean_exp) / std_exp
            pyr = torch.stack([kp["pitch"].reshape(-1), kp["yaw"].reshape(-1),
                               kp["roll"].reshape(-1)], -1).float().cpu()
            pyr_n = (pyr - p_lo) / (p_hi - p_lo)
            sc = kp["scale"].reshape(-1, 1).float().cpu()
            sc_n = (sc - s_lo) / (s_hi - s_lo)
            tt = kp["t"].reshape(-1, 3).float().cpu()
            tt_n = (tt - t_lo) / (t_hi - t_lo)
            coefs.append(torch.cat([exp_n, sc_n, tt_n, pyr_n], dim=-1))
    return torch.cat(coefs)      # (T, 70...) — padded to 73 below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="real English talking-head mp4")
    ap.add_argument("--portrait", default=None,
                    help="omit (or pass 'auto') to use the driving video's own first "
                         "frame: same person, same voice, so no gender/identity mismatch")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--smooth-alpha", type=float, default=0.3)
    ap.add_argument("--scale-band", type=float, default=0.02,
                    help="clamp head scale to +/- this fraction of its median (kills size pulsing)")
    ap.add_argument("--stab-win", type=int, default=51, help="median-filter window, frames")
    ap.add_argument("--no-stabilize", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="real01")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "real_driven"))
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "audio_real"), exist_ok=True)

    from src.config.argument_config import ArgumentConfig  # noqa: E402
    from src.config.crop_config import CropConfig          # noqa: E402
    from src.utils.cropper import Cropper                  # noqa: E402

    eng = TalkingHeadEngine(load_renderer=True)
    args = ArgumentConfig()
    crop_cfg = CropConfig(**{k: v for k, v in args.__dict__.items() if hasattr(CropConfig, k)})
    cropper = Cropper(crop_cfg=crop_cfg)

    n_frames = int(a.seconds * FPS)
    frames = read_frames(a.video, n_frames)
    print(f"[real] {os.path.basename(a.video)}: {len(frames)} frames", flush=True)

    wav = extract_audio(a.video, os.path.join(
        ROOT, "data", "audio_real", os.path.basename(a.video).replace(".mp4", ".wav")))
    audio = eng.load_audio(wav, a.seconds)

    coef = real_coefficients(eng, frames, cropper, crop_cfg)
    T = min(coef.shape[0], int(len(audio) / 16000 * FPS))
    coef = coef[:T]
    if coef.shape[1] < 73:
        coef = torch.cat([coef, torch.zeros(coef.shape[0], 73 - coef.shape[1])], dim=-1)
    print(f"[real] coefficients {tuple(coef.shape)}", flush=True)

    # ---- measured statistics: these are the parameters we have been missing --
    from motion_reward import _bandpass                     # noqa: E402
    from motion_reward_v5 import NON_LIP                    # noqa: E402
    std_exp = torch.as_tensor(eng.tpl["std_exp"], dtype=torch.float32).reshape(1, 63)
    exp_raw = coef[:, :63] * std_exp                        # raw displacement
    W = eng.n_motions
    n_win = coef.shape[0] // W
    dyn, rng = [], []
    for w in range(n_win):
        e = exp_raw[w * W:(w + 1) * W][:, NON_LIP].unsqueeze(0)
        dyn.append(float(_bandpass(e, 0.3, 3.0).pow(2).mean(1).add(1e-6).sqrt().mean(-1)))
        rng.append(exp_raw[w * W:(w + 1) * W][:, NON_LIP].abs().max().item())
    pyr = coef[:, 67:70]
    p_lo = torch.tensor([float(eng.tpl[f"min_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    p_hi = torch.tensor([float(eng.tpl[f"max_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    deg = pyr * (p_hi - p_lo) + p_lo
    stats = {
        "video": os.path.basename(a.video), "portrait": os.path.basename(a.portrait),
        "frames": int(coef.shape[0]), "windows": int(n_win),
        "expr_dyn_std_per_window": dyn,
        "expr_dyn_std_median": float(np.median(dyn)) if dyn else None,
        "expr_abs_max_median": float(np.median(rng)) if rng else None,
        "pose_low_deg_std": None,
    }
    if n_win:
        pn = pyr[: n_win * W].reshape(n_win, W, 3)
        lo = _bandpass(pn, 0.3, 3.0).pow(2).mean(1).add(1e-6).sqrt() * (p_hi - p_lo)
        stats["pose_low_deg_std"] = float(lo.norm(dim=-1).div(np.sqrt(3)).median())
    json.dump(stats, open(os.path.join(a.out, f"{a.tag}_coef_stats.json"), "w"), indent=1)
    print(f"[stats] expr_dyn_std median={stats['expr_dyn_std_median']:.5f}  "
          f"pose_low_deg_std={stats['pose_low_deg_std']}", flush=True)

    coef = stabilize_scale_translation(coef, scale_band=a.scale_band, win=a.stab_win)
    print(f"[real] scale stabilised: band=+/-{a.scale_band:.0%} win={a.stab_win}f", flush=True)

    opts = RenderOptions(**HD_OPTS)
    if not a.portrait or a.portrait == "auto":
        # the driving video's own first frame -> same identity, same voice
        pdir = os.path.join(ROOT, "data", "portraits_real")
        os.makedirs(pdir, exist_ok=True)
        port = os.path.join(pdir, os.path.basename(a.video).replace(".mp4", ".png"))
        # Frame 0 is sometimes a fade-in / title card with no detectable face.
        # Among the early frames that DO crop, pick the one with eyes open and
        # MOUTH CLOSED: if the source mouth is open, its teeth get baked into the
        # texture and LivePortrait's warping then smears them on every wide
        # opening (the "broken teeth" the user saw).
        best, best_score = None, -1e9
        for idx in (0, 6, 12, 24, 40, 60, 90):
            if idx >= len(frames):
                break
            cand = frames[idx]
            try:
                out = cropper.crop_source_image(cand, crop_cfg)
                if out is None or out.get("img_crop_256x256") is None:
                    continue
            except Exception:
                continue
            ear, jaw = _eye_mouth_scores(cand)
            luma = float(cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY).mean())
            score = ear * 3.0 - jaw * 6.0 + (luma / 255.0) * 0.5
            if score > best_score:
                best, best_score = cand, score
        img = best if best is not None else frames[0]
        print(f"[real] portrait score={best_score:.3f}", flush=True)
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            s = 1280 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        cv2.imwrite(port, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[real] portrait = own frame -> {port}", flush=True)
    else:
        port = a.portrait if os.path.isabs(a.portrait) else os.path.join(ROOT, a.portrait)

    driving = eng.coef_to_driving(coef, smooth_alpha=a.smooth_alpha)
    eng.render(port, wav, driving, a.out, f"{a.tag}__real", opts)
    print(f"[ok] {a.tag}__real.mp4", flush=True)

    base_coef = eng.gen_motion(audio, cfg_scale=2.0, seed=0)
    base_coef = base_coef[:T]
    driving_b = eng.coef_to_driving(base_coef, smooth_alpha=a.smooth_alpha)
    eng.render(port, wav, driving_b, a.out, f"{a.tag}__base", opts)
    print(f"[ok] {a.tag}__base.mp4", flush=True)


if __name__ == "__main__":
    main()
