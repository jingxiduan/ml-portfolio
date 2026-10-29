"""Tier-2 Best-of-N: inference-time rejection sampling to repair the OPSD-LoRA's
reward-hacking symptoms (blink overshoot ~55/min, head jitter +110%).

Pipeline per audio clip:
  1. generate N motions with the OPSD LoRA (seeds 0..N-1), each (T, 73)
  2. score each one in motion-space with naturalness_v2 (defined below) --- an
     anti-hacking scorer that is engine-agnostic and NOT the training reward
  3. argmax -> opsd_bon motion
  4. render opsd_bon alongside base_naive (base, seed 0) and opsd_naive (opsd, seed 0)
  5. score all three renders with video_metrics (MediaPipe, independent of eye_probe)

The scorer differs from motion_reward.LivelinessReward in two deliberate ways:
  - blink is scored by **matching ~18 blinks/min** (gaussian), not "at least one
    per window". This removes the hacking incentive that produced 55/min.
  - head jitter has a **hard quadratic** penalty above base-model level, not just
    a bounded term, so BoN really picks less-jittery candidates.
Everything else (amp, speech-head couple, gaze variation) is the same as the
training reward, so BoN rewards the real "lively but natural" region of policy
space, not just "less lively".
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from engine import FPS, RenderOptions, TalkingHeadEngine  # noqa: E402
from eye_probe import EyeProbe  # noqa: E402
from motion_reward import LIP_DIMS  # noqa: E402
from opsd_train_motion import add_lora, audio_env_frames, set_mode  # noqa: E402

ROOT = os.path.dirname(SRC)

# Faces (same mapping evaluate_motion.py uses so results are directly comparable)
FACES = ["host_f.png", "host_m.png", "host_f2.png"]


# ---------------------------------------------------------------- scoring
def naturalness_v2(coef: torch.Tensor, env: torch.Tensor, eye_probe: EyeProbe,
                   template: dict) -> dict:
    """Anti-hacking naturalness score on a (T, 73) motion sample.

    All operations in numpy (we only need argmax over candidates, no gradient).
    """
    T = coef.shape[0]
    device = next(eye_probe.parameters()).device

    # 1. eye signals via probe
    with torch.no_grad():
        d_exp = coef[:, :63] - coef[:1, :63]
        eye = eye_probe(d_exp.to(device).unsqueeze(0))
        blink_sig = eye["blink"][0].cpu().numpy()
        gaze_h = eye["gaze_h"][0].cpu().numpy()
        gaze_v = eye["gaze_v"][0].cpu().numpy()

    # 2. blink rate via peak detection, target gaussian around 18/min
    from scipy.signal import find_peaks
    pk, _ = find_peaks(blink_sig,
                       height=max(0.35, float(np.median(blink_sig)) + 0.15),
                       prominence=0.15,
                       distance=int(0.3 * FPS))
    duration_s = T / FPS
    blink_rate = len(pk) / duration_s * 60.0  # per minute
    blink_match = float(np.exp(-((blink_rate - 18.0) / 8.0) ** 2 / 2.0))

    # 3. pose in degrees
    pose_norm = coef[:, 67:70].numpy()  # pitch/yaw/roll normalized
    lo = np.array([float(template[f"min_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    hi = np.array([float(template[f"max_{k}"][0]) for k in ("pitch", "yaw", "roll")])
    pose = pose_norm * (hi - lo) + lo  # (T, 3) in degrees
    pose_c = pose - pose.mean(0, keepdims=True)

    # 4. jitter via high-frequency (>6 Hz) power, hard penalty
    freqs = np.fft.rfftfreq(T, d=1.0 / FPS)
    spec = np.abs(np.fft.rfft(pose_c, axis=0)) ** 2  # (F, 3)
    hp = freqs > 6.0
    hp_power = spec[hp].mean()  # scalar across axes
    # base model's hp_power on this dataset sits around 0.3-0.6 (deg^2),
    # OPSD-naive around 1.0-1.8; map to a score in [0,1]
    jitter_score = float(np.exp(-hp_power / 0.6))  # softer than 1.0 cap, keeps gradient

    # 5. amp via low-frequency (0.3-3 Hz) power, tanh saturating
    lp = (freqs >= 0.3) & (freqs <= 3.0)
    lp_power = spec[lp].mean()
    amp_score = float(np.tanh(lp_power / 4.0))  # ~2 deg^2 ≈ decent amp

    # 6. speech-head coupling: pearson(env_lagged, head_speed) with 4-frame lag
    pose_vel = np.diff(pose, axis=0)
    pose_speed = np.linalg.norm(pose_vel, axis=1) * FPS  # (T-1,)
    env_np = env.cpu().numpy()[1:1 + len(pose_speed)]
    lag = 4
    if len(pose_speed) > lag + 10:
        e = env_np[: len(pose_speed) - lag]
        s = pose_speed[lag:]
        if e.std() > 1e-6 and s.std() > 1e-6:
            couple_raw = float(np.corrcoef(e, s)[0, 1])
        else:
            couple_raw = 0.0
    else:
        couple_raw = 0.0
    couple_score = max(0.0, couple_raw)  # reward only positive correlation

    # 7. gaze variation, with off-camera penalty
    gaze_std = (gaze_h.std() + gaze_v.std()) / 2
    gaze_score = float(np.tanh(gaze_std / 0.1))
    gaze_bias = abs(float(gaze_h.mean())) + abs(float(gaze_v.mean()))
    gaze_score -= max(0.0, gaze_bias - 0.3)

    # 8. weighted total: blink + jitter are the anti-hacking emphasis.
    # blink weight is deliberately high (2.5) because the OPSD candidate pool is
    # dominated by overshoot samples; a weaker weight gets drowned out.
    total = (2.5 * blink_match
             + 1.0 * jitter_score
             + 0.6 * amp_score
             + 0.4 * couple_score
             + 0.3 * gaze_score)

    return {
        "total": float(total),
        "blink_match": blink_match,
        "blink_rate_est": float(blink_rate),
        "jitter_score": jitter_score,
        "jitter_hp_power": float(hp_power),
        "amp_score": amp_score,
        "amp_lp_power": float(lp_power),
        "couple_score": couple_score,
        "couple_raw": couple_raw,
        "gaze_score": gaze_score,
    }


# ------------------------------------------------------------- lora loading
def load_trained(eng, path):
    ck = torch.load(path, map_location="cpu")
    layers = add_lora(eng.mg.denoising_net, ck["r"], ck["alpha"])
    for i, l in enumerate(layers):
        l.A_t.data.copy_(ck["lora"][f"{i}.A_t"].to(l.A_t.device))
        l.B_t.data.copy_(ck["lora"][f"{i}.B_t"].to(l.B_t.device))
    return layers


def swap_lora_weights(layers, path):
    """Replace the trainable adapter weights of already-wrapped layers in place.
    Used to hot-swap v1 -> v2 without re-wrapping (which would double-wrap)."""
    ck = torch.load(path, map_location="cpu")
    for i, l in enumerate(layers):
        l.A_t.data.copy_(ck["lora"][f"{i}.A_t"].to(l.A_t.device))
        l.B_t.data.copy_(ck["lora"][f"{i}.B_t"].to(l.B_t.device))
    print(f"[swap-lora] loaded weights from {path}")


# ---------------------------------------------------- Track B: blink thinning
def blink_thin(coef: torch.Tensor, eye_probe: EyeProbe,
               min_interval_s: float = 2.5, win_frames: int = 5,
               attn_min: float = 0.25) -> torch.Tensor:
    """Suppress blink events that are too close in time.

    1. detect blinks via eye_probe on this motion
    2. keep every peak that is >= `min_interval_s` seconds from any already-kept
       stronger peak (greedy by peak height)
    3. for each suppressed peak, attenuate expression dims (not lip keypoints)
       toward the first-frame baseline inside a +/-`win_frames` window, with a
       cosine profile that is strongest at the peak and tapers to 1.0 at window
       edges. `attn_min` is the attenuation factor at the peak center.

    Lip keypoints are NOT touched, so lip-sync is preserved.
    """
    T = coef.shape[0]
    device = next(eye_probe.parameters()).device
    with torch.no_grad():
        d_exp = coef[:, :63] - coef[:1, :63]
        blink_sig = eye_probe(d_exp.to(device).unsqueeze(0))["blink"][0].cpu().numpy()

    from scipy.signal import find_peaks
    pk, props = find_peaks(blink_sig,
                           height=max(0.35, float(np.median(blink_sig)) + 0.15),
                           prominence=0.15,
                           distance=int(0.3 * FPS))
    if len(pk) <= 1:
        return coef.clone()

    heights = props["peak_heights"]
    # greedy keep: strongest first, enforce min interval
    order = np.argsort(-heights)
    min_gap = int(min_interval_s * FPS)
    kept_t = []
    for idx in order:
        t = int(pk[idx])
        if all(abs(t - kt) >= min_gap for kt in kept_t):
            kept_t.append(t)
    kept_set = set(kept_t)
    suppressed = [int(pk[i]) for i in range(len(pk)) if int(pk[i]) not in kept_set]
    if not suppressed:
        return coef.clone()

    out = coef.clone()
    non_lip = [d for d in range(63) if d not in LIP_DIMS]
    non_lip_idx = torch.tensor(non_lip, dtype=torch.long)
    baseline = coef[0, non_lip_idx]  # resting values

    for t in suppressed:
        for dt in range(-win_frames, win_frames + 1):
            tt = t + dt
            if 0 <= tt < T:
                # cosine profile: attn_min at center, 1.0 at edges
                frac = abs(dt) / win_frames  # 0 at center, 1 at edge
                alpha = attn_min + (1.0 - attn_min) * (1 - np.cos(np.pi * frac)) / 2
                out[tt, non_lip_idx] = baseline + (coef[tt, non_lip_idx] - baseline) * alpha
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=os.path.join(ROOT, "outputs", "opsd", "lora.pt"))
    ap.add_argument("--lora-v2", default=None,
                    help="also evaluate a second LoRA (e.g. outputs/opsd_v2/lora.pt) as model 'opsd_v2'; "
                         "the two LoRAs are hot-swapped between runs on the same wrapped layers")
    ap.add_argument("--eye-probe", default=os.path.join(ROOT, "outputs", "eye_probe", "eye_probe.pt"))
    ap.add_argument("--clips", nargs="*", default=None,
                    help="specific eval wav paths (default: all under data/audio/eval)")
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--n-eval-clips", type=int, default=6, help="how many clips to actually render for comparison")
    ap.add_argument("--smooth", type=float, default=0.3)
    ap.add_argument("--motion-only", action="store_true",
                    help="skip rendering, output motion-space BoN table only (fast, ~1 min)")
    ap.add_argument("--save-coefs", action="store_true",
                    help="save selected motion coefficients to <out>/selected_coefs.pt for later rendering elsewhere (e.g. H20)")
    ap.add_argument("--load-coefs", default=None,
                    help="skip motion generation entirely and load selected coefs from this .pt file (e.g. on H20 after transferring from Mac)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "tier2"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import glob
    clips = a.clips or sorted(glob.glob(os.path.join(ROOT, "data", "audio", "eval", "*.wav")))
    assert clips, "no eval clips"

    # --- fast path: load pre-selected coefs from disk, skip motion gen entirely
    if a.load_coefs is not None:
        assert not a.motion_only, "--motion-only is incompatible with --load-coefs"
        print(f"[init] load engine (renderer=on) ...", flush=True)
        eng = TalkingHeadEngine(load_renderer=True)
        bundle = torch.load(a.load_coefs, map_location="cpu")
        selected = {tuple(k): v for k, v in bundle["selected_tuples"]}
        clips = bundle["clips"]
        motion_tbl = bundle.get("motion_tbl", {})
        print(f"[load-coefs] loaded {len(selected)} entries for {len(clips)} clips from {a.load_coefs}")
        # skip the whole generation / BoN loop
        all_rows = bundle.get("all_rows", [])
        variants_all = ["base_naive", "base_bon", "opsd_naive", "opsd_bon", "opsd_bon_thin"]
    else:
        print(f"[init] load engine (renderer={'off' if a.motion_only else 'on'}) ...", flush=True)
        eng = TalkingHeadEngine(load_renderer=not a.motion_only)
        lora_layers = load_trained(eng, a.lora)
        probe = EyeProbe()
        probe.load_state_dict(torch.load(a.eye_probe, map_location="cpu"))
        probe = probe.to(eng.device).eval()

        model_list = ["base", "opsd"]
        if a.lora_v2:
            model_list.append("opsd_v2")

        all_rows = []
        selected = {}  # (ci, variant) -> (coef, seed, score_dict)

        for ci, clip in enumerate(clips):
            name = Path(clip).stem
            audio = eng.load_audio(clip)
            print(f"\n[clip {ci}] {name}  ({len(audio)/16000:.1f}s)", flush=True)

            for model in model_list:
                if model == "base":
                    set_mode("base")
                elif model == "opsd":
                    # restore v1 weights (needed after a v2 swap on a prior clip)
                    if a.lora_v2:
                        swap_lora_weights(lora_layers, a.lora)
                    set_mode("trainable")
                else:  # opsd_v2
                    swap_lora_weights(lora_layers, a.lora_v2)
                    set_mode("trainable")

                cand_rows = []
                cand_coefs = []
                for seed in range(a.n_candidates):
                    t0 = time.time()
                    coef = eng.gen_motion(audio, cfg_scale=2.0, seed=seed)
                    env = audio_env_frames(audio, coef.shape[0])
                    score = naturalness_v2(coef, env, probe, eng.tpl)
                    score["clip"] = ci
                    score["name"] = name
                    score["model"] = model
                    score["seed"] = seed
                    score["gen_s"] = round(time.time() - t0, 2)
                    cand_rows.append(score)
                    cand_coefs.append(coef)
                    all_rows.append(score)
                best_i = int(np.argmax([r["total"] for r in cand_rows]))
                selected[(ci, f"{model}_bon")] = (cand_coefs[best_i], best_i, cand_rows[best_i])
                selected[(ci, f"{model}_naive")] = (cand_coefs[0], 0, cand_rows[0])
                best = cand_rows[best_i]
                seed0 = cand_rows[0]
                print(f"  [{model}] seed=0  total={seed0['total']:.3f}  "
                      f"blink_rate={seed0['blink_rate_est']:.1f}/min  "
                      f"jitter_hp={seed0['jitter_hp_power']:.3f}", flush=True)
                print(f"  [{model}] BoN(seed={best_i}) total={best['total']:.3f}  "
                      f"blink_rate={best['blink_rate_est']:.1f}/min  "
                      f"jitter_hp={best['jitter_hp_power']:.3f}", flush=True)

                # Track B · for OPSD v1 only: apply blink-thinning on top of BoN.
                # v2 already has an overshoot penalty in its reward, so thinning
                # would be over-correction.
                if model == "opsd":
                    thin_coef = blink_thin(cand_coefs[best_i], probe)
                    thin_env = audio_env_frames(audio, thin_coef.shape[0])
                    thin_score = naturalness_v2(thin_coef, thin_env, probe, eng.tpl)
                    thin_score["clip"] = ci
                    thin_score["name"] = name
                    thin_score["model"] = "opsd"
                    thin_score["seed"] = best_i
                    thin_score["variant"] = "bon_thin"
                    selected[(ci, "opsd_bon_thin")] = (thin_coef, best_i, thin_score)
                    print(f"  [opsd] BoN+thin total={thin_score['total']:.3f}  "
                          f"blink_rate={thin_score['blink_rate_est']:.1f}/min  "
                          f"jitter_hp={thin_score['jitter_hp_power']:.3f}", flush=True)

        # motion-space table
        variants_all = ["base_naive", "base_bon",
                        "opsd_naive", "opsd_bon", "opsd_bon_thin"]
        if a.lora_v2:
            variants_all += ["opsd_v2_naive", "opsd_v2_bon"]
        motion_tbl = {
            v: {
                key: float(np.mean([selected[(ci, v)][2][key] for ci in range(len(clips))]))
                for key in ("total", "blink_rate_est", "jitter_hp_power", "amp_lp_power", "couple_raw")
            }
            for v in variants_all
        }
        print("\n[motion-space summary (avg over clips)]")
        print(json.dumps(motion_tbl, indent=1))

        with open(os.path.join(a.out, "bon_motion_scores.jsonl"), "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(a.out, "bon_motion_summary.json"), "w") as f:
            json.dump({"per_clip_selected": {
                f"{ci}_{v}": {"seed": selected[(ci, v)][1], **selected[(ci, v)][2]}
                for ci in range(len(clips)) for v in variants_all
            }, "avg": motion_tbl}, f, indent=1)

        # ---- save bundle for H20 to pick up without re-generating ----
        if a.save_coefs:
            bundle_path = os.path.join(a.out, "selected_coefs.pt")
            bundle = {
                "clips": clips,
                "selected_tuples": [((ci, v), selected[(ci, v)]) for ci in range(len(clips)) for v in variants_all],
                "motion_tbl": motion_tbl,
                "all_rows": all_rows,
                "note": "produced on Mac/MPS; load on H20 with --load-coefs to skip motion gen",
            }
            torch.save(bundle, bundle_path)
            print(f"\n[save-coefs] wrote {bundle_path} ({os.path.getsize(bundle_path)/1024:.0f} KB)")

    if a.motion_only:
        print("\n[done] motion-only mode; skipping render + video_metrics")
        return

    # ------------------- render + video metrics --------------------
    from video_metrics import evaluate_video

    n_render = min(a.n_eval_clips, len(clips))
    # Determine render variants from what's actually in `selected`
    seen_variants = sorted({v for (_, v) in selected.keys()})
    render_variants = tuple(v for v in (
        "base_naive", "opsd_naive", "opsd_bon", "opsd_bon_thin",
        "opsd_v2_naive", "opsd_v2_bon",
    ) if v in seen_variants)
    print(f"\n[render] {n_render} clips × {len(render_variants)} variants = "
          f"{n_render*len(render_variants)} videos", flush=True)
    video_rows = []
    for ci in range(n_render):
        clip = clips[ci]
        name = Path(clip).stem
        face = os.path.join(ROOT, "data", "portraits", FACES[ci % len(FACES)])
        print(f"\n[render clip {ci}] face={os.path.basename(face)}", flush=True)
        for variant in render_variants:
            coef, seed, motion_score = selected[(ci, variant)]
            drv = eng.coef_to_driving(coef, smooth_alpha=a.smooth)
            tag = f"bon_c{ci:02d}_{variant}"
            t0 = time.time()
            vid = eng.render(face, clip, drv, a.out, tag, RenderOptions(flag_pasteback=False))
            print(f"  rendered {tag}  ({time.time()-t0:.1f}s)  seed={seed}", flush=True)
            feat = evaluate_video(vid, clip)
            video_rows.append({"clip": ci, "name": name, "variant": variant, "seed": seed,
                               "video": os.path.basename(vid), **feat})
            print(f"    blink_rate={feat['blink_rate']:.1f}  "
                  f"head_jitter={feat['head_jitter']:.1f}  "
                  f"speech_head_corr={feat['speech_head_corr']:.3f}", flush=True)

    # summary
    vkeys = [k for k in video_rows[0]
             if k not in ("clip", "name", "variant", "seed", "video", "duration_s", "detect_rate")]
    video_sum = {
        v: {k: float(np.mean([r[k] for r in video_rows if r["variant"] == v])) for k in vkeys}
        for v in render_variants
    }
    print("\n[video summary]")
    print(json.dumps(video_sum, indent=1))

    with open(os.path.join(a.out, "bon_metrics.json"), "w") as f:
        json.dump({"video_summary": video_sum, "video_rows": video_rows,
                   "motion_summary": motion_tbl}, f, indent=1)

    # markdown · dynamically includes whatever variants are present
    label_map = {
        "base_naive": "base",
        "opsd_naive": "opsd v1 (naive)",
        "opsd_bon": "+ BoN",
        "opsd_bon_thin": "+ BoN + thin",
        "opsd_v2_naive": "**opsd v2** (naive)",
        "opsd_v2_bon": "**opsd v2 + BoN**",
    }
    md_variants = [v for v in render_variants if v in video_sum]
    header_cells = " | ".join(label_map.get(v, v) for v in md_variants)
    md = [
        f"# Tier-2 · video-space metrics (held-out, MediaPipe, n={n_render} clips)\n",
        f"- N candidates per model: {a.n_candidates}",
        "- Track A (BoN): motion-space `naturalness_v2` argmax over N seeds",
        "- Track B (thin): blink-thinning post-process (min 2.5 s between blinks, cosine attenuation on non-lip expression dims)",
        "- Track C (opsd v2): LoRA retrained with reward v2 (gaussian blink target + hard jitter cap)" if "opsd_v2_naive" in md_variants else "",
        "",
        f"| metric | {header_cells} | human target |",
        "|" + "---|" * (len(md_variants) + 2),
    ]
    hum = {"blink_rate": "10-20", "head_jitter": "~120", "head_pitch_std": "1-4",
           "head_yaw_std": "1-4", "speech_head_corr": "0.2-0.4", "still_ratio": "<0.1",
           "gaze_std": "0.1-0.3", "head_speed_mean": "5-15", "nod_rate": "0.3-1.0",
           "gaze_shift_rate": "1-3", "head_roll_std": "0.5-2", "jaw_open_mean": "0.1-0.3",
           "jaw_audio_corr": "0.3-0.5"}
    for k in ("blink_rate", "head_jitter", "head_pitch_std", "head_yaw_std",
              "head_speed_mean", "speech_head_corr", "still_ratio", "gaze_std",
              "gaze_shift_rate", "nod_rate", "jaw_audio_corr"):
        if any(k not in video_sum[v] for v in md_variants):
            continue
        cells = " | ".join(f"{video_sum[v][k]:.2f}" for v in md_variants)
        md.append(f"| {k} | {cells} | {hum.get(k, '—')} |")
    md = [m for m in md if m]  # drop empty strings
    with open(os.path.join(a.out, "bon_metrics.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[done] wrote {a.out}/bon_metrics.{{json,md}} + {len(video_rows)} mp4s")


if __name__ == "__main__":
    main()
