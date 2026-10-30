"""Render HD side-by-side comparison videos for the 07 qualitative gallery.

Renders every (clip x variant x portrait) combination at source_max_dim=1280 with
pasteback enabled, so the face is composited back into the full-resolution
portrait instead of being squeezed to 256. This is the "HD" path referenced in
src/render_compare.py as `hd_pasteback_light_smooth`.

Outputs:
  outputs/hd_compare/<portrait>__<clip>__<variant>.mp4
  outputs/hd_compare/manifest.json
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))

from engine import RenderOptions, TalkingHeadEngine  # noqa: E402

# LoRA variants to compare. None = base model.
VARIANTS = [
    ("base", None),
    ("v2", "outputs/opsd_v2_large/lora.pt"),
    ("v3", "outputs/opsd_v3_large/lora.pt"),
]

HD_OPTS = dict(
    flag_do_crop=True,
    flag_pasteback=True,      # composite back into the full-resolution portrait
    flag_stitching=True,
    flag_relative_motion=True,
    driving_option="pose-friendly",
    driving_multiplier=1.0,
    source_max_dim=1280,      # HD: do not downscale the source below 1280
    animation_region="all",
)


def load_lora(eng, path):
    """Hot-swap a LoRA onto the motion generator; None restores base weights."""
    import opsd_train_motion as T
    if not hasattr(eng, "_lora_layers"):
        eng._lora_layers = T.add_lora(eng.mg.denoising_net, 8, 8)
        eng._lora_sd = {f"{i}.{k}": v for i, l in enumerate(eng._lora_layers)
                        for k, v in (("A_t", l.A_t.detach().clone()),
                                     ("B_t", l.B_t.detach().clone()),
                                     ("A_b", l.A_b.clone()),
                                     ("B_b", l.B_b.clone()))}
    if path is None:
        T.set_mode("none")
        return
    sd = torch.load(path, map_location="cpu")
    lora = sd["lora"]
    with torch.no_grad():
        for i, l in enumerate(eng._lora_layers):
            l.A_t.copy_(lora[f"{i}.A_t"])
            l.B_t.copy_(lora[f"{i}.B_t"])
            if f"{i}.A_b" in lora:
                l.A_b.copy_(lora[f"{i}.A_b"])
                l.B_b.copy_(lora[f"{i}.B_b"])
    T.set_mode("behavior")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None,
                    help="audio wavs (default: first N of the AISHELL eval set)")
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--portraits", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=[v for v, _ in VARIANTS])
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "hd_compare"))
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    clips = a.clips or sorted(glob.glob(os.path.join(ROOT, "data", "audio_public", "eval", "*.wav")))[: a.n_clips]
    portraits = a.portraits or sorted(glob.glob(os.path.join(ROOT, "data", "portraits", "*.png")))
    variants = [(n, p) for n, p in VARIANTS if n in a.variants]

    print(f"[hd] {len(clips)} clips x {len(portraits)} portraits x {len(variants)} variants "
          f"= {len(clips)*len(portraits)*len(variants)} videos, source_max_dim=1280")

    eng = TalkingHeadEngine(load_renderer=True)
    opts = RenderOptions(**HD_OPTS)

    manifest = []
    t_all = time.time()
    for ci, wav in enumerate(clips):
        audio = eng.load_audio(wav, a.max_seconds)
        for pi, port in enumerate(portraits):
            for vname, lora_path in variants:
                tag = f"c{ci:02d}_{os.path.basename(port)[:-4]}__{vname}"
                dst = os.path.join(a.out, f"{tag}.mp4")
                if os.path.exists(dst):
                    print(f"  [skip] {tag}")
                    manifest.append({"tag": tag, "clip": wav, "portrait": port,
                                     "variant": vname, "video": dst})
                    continue
                load_lora(eng, lora_path)
                coef = eng.gen_motion(audio, cfg_scale=2.0, seed=a.seed)
                driving = eng.coef_to_driving(coef, smooth_alpha=0.3)
                t0 = time.time()
                eng.render(port, wav, driving, a.out, tag, opts)
                print(f"  [ok] {tag} ({time.time()-t0:.1f}s)")
                manifest.append({"tag": tag, "clip": os.path.basename(wav),
                                 "portrait": os.path.basename(port),
                                 "variant": vname, "video": dst})

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({"source_max_dim": HD_OPTS["source_max_dim"],
                   "pasteback": HD_OPTS["flag_pasteback"],
                   "seed": a.seed, "entries": manifest}, f, indent=1)
    print(f"[ok] {len(manifest)} videos in {time.time()-t_all:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
