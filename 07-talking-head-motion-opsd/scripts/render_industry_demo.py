"""Render the three-industry demo: HD talking-head video, base vs persona.

For each industry we render the same (portrait, real-speech audio) pair twice --
once with the untouched base model and once with that industry's OPSD LoRA -- so
the only thing that differs is the post-training. Portraits and audio come from
`scripts/prep_industry_demo.py` (real public footage, real human speech).

Render settings are the HD path: source_max_dim=1280 with pasteback, meaning the
animated face is composited back into the full-resolution portrait instead of
being cropped to the face box.

Outputs:
  outputs/industry_demo/<industry>/<tag>__base.mp4
  outputs/industry_demo/<industry>/<tag>__<industry>.mp4
  outputs/industry_demo/manifest.json
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))

from engine import RenderOptions, TalkingHeadEngine  # noqa: E402

HD_OPTS = dict(
    flag_do_crop=True,
    flag_pasteback=True,       # composite back into the full-resolution portrait
    flag_stitching=True,
    flag_relative_motion=True,
    driving_option="pose-friendly",
    driving_multiplier=1.0,
    source_max_dim=1280,
    animation_region="all",
)


def load_lora(eng, path):
    """Hot-swap a LoRA onto the motion generator; None restores pure base.

    DualLoRALinear.forward branches on the mode string: "trainable" uses the
    trained A_t/B_t, "behavior" uses the EMA copy (B_b starts at zero), and
    anything else contributes no LoRA at all. We load A_t/B_t, so the mode MUST
    be "trainable" -- otherwise the persona silently renders as base.
    """
    import opsd_train_motion as T
    if not hasattr(eng, "_lora_layers"):
        eng._lora_layers = T.add_lora(eng.mg.denoising_net, 8, 8)
    if path is None:
        T.set_mode("base")
        return
    ck = torch.load(path, map_location="cpu")["lora"]
    with torch.no_grad():
        for i, l in enumerate(eng._lora_layers):
            l.A_t.data.copy_(ck[f"{i}.A_t"].to(l.A_t.device))
            l.B_t.data.copy_(ck[f"{i}.B_t"].to(l.B_t.device))
    T.set_mode("trainable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "industry_demo_manifest.json"))
    ap.add_argument("--lora-dir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "industry_demo"))
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--industries", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap pairs per industry")
    ap.add_argument("--portrait-override", nargs="*", default=None,
                    help="render these portraits for EVERY industry instead of each "
                         "industry's own -- used for the same-face cross-industry control")
    ap.add_argument("--tag-prefix", default="", help="prefix added to output tags")
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    man = json.load(open(a.manifest))
    eng = TalkingHeadEngine(load_renderer=True)
    opts = RenderOptions(**HD_OPTS)

    entries, t_all = [], time.time()
    for ind, items in man["industries"].items():
        if a.industries and ind not in a.industries:
            continue
        lora = os.path.join(a.lora_dir, f"opsd_ind_{ind}", "lora.pt")
        if not os.path.exists(lora):
            print(f"  [skip] {ind}: no checkpoint at {lora}")
            continue
        pairs = items[: a.limit] if a.limit else items
        if a.portrait_override:
            # same-face control: identical portrait + identical audio for every
            # industry, so any difference on screen is the persona and nothing else
            pairs = [{"portrait": p if os.path.isabs(p) else os.path.relpath(p, ROOT),
                      "audio": items[0]["audio"]} for p in a.portrait_override]
        odir = os.path.join(a.out, ind + ("_control" if a.portrait_override else ""))
        os.makedirs(odir, exist_ok=True)
        for pi, it in enumerate(pairs):
            port = it["portrait"] if os.path.isabs(it["portrait"]) else os.path.join(ROOT, it["portrait"])
            wav = os.path.join(ROOT, it["audio"])
            tag = a.tag_prefix + f"{ind}_{pi:02d}_{os.path.basename(it['portrait'])[:-4]}"
            audio = eng.load_audio(wav, a.max_seconds)
            for vname, lpath in (("base", None), (ind, lora)):
                dst = os.path.join(odir, f"{tag}__{vname}.mp4")
                if os.path.exists(dst):
                    entries.append({"industry": ind, "tag": tag, "variant": vname,
                                    "video": os.path.relpath(dst, ROOT),
                                    "portrait": it["portrait"], "audio": it["audio"]})
                    continue
                load_lora(eng, lpath)
                coef = eng.gen_motion(audio, cfg_scale=2.0, seed=a.seed)
                driving = eng.coef_to_driving(coef, smooth_alpha=0.3)
                t0 = time.time()
                try:
                    eng.render(port, wav, driving, odir, f"{tag}__{vname}", opts)
                except Exception as e:
                    # Some first frames are a fade-in / cut and carry no detectable
                    # face; skip that portrait instead of aborting the whole industry.
                    print(f"  [skip] {ind}/{tag}__{vname}: {type(e).__name__}: {e}", flush=True)
                    entries.append({"industry": ind, "tag": tag, "variant": vname,
                                    "video": None, "error": str(e),
                                    "portrait": it["portrait"], "audio": it["audio"]})
                    continue
                print(f"  [ok] {ind}/{tag}__{vname} ({time.time()-t0:.1f}s)", flush=True)
                entries.append({"industry": ind, "tag": tag, "variant": vname,
                                "video": os.path.relpath(dst, ROOT),
                                "portrait": it["portrait"], "audio": it["audio"]})

    json.dump({"render": HD_OPTS, "seed": a.seed, "entries": entries},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"[ok] {len(entries)} videos in {time.time()-t_all:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
