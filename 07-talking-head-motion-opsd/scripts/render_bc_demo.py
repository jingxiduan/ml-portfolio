"""Render a BASE vs TRAINED pair from one portrait + one audio clip.

Used to show what the behaviour-cloning (+OPSD) checkpoint actually does, without
touching the broken industry manifest.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "src")))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "scripts")))

from engine import RenderOptions, TalkingHeadEngine  # noqa: E402
from render_real_driven import HD_OPTS              # noqa: E402


def load_lora(eng, path):
    """Hot-swap a LoRA onto the motion generator; None restores pure base."""
    import torch
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
    ap.add_argument("--portrait", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "bc_demo"))
    ap.add_argument("--tag", default="bc")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    os.makedirs(a.out, exist_ok=True)
    port = a.portrait if os.path.isabs(a.portrait) else os.path.join(ROOT, a.portrait)
    wav = a.audio if os.path.isabs(a.audio) else os.path.join(ROOT, a.audio)

    eng = TalkingHeadEngine(load_renderer=True)
    audio = eng.load_audio(wav, a.seconds)
    opts = RenderOptions(**HD_OPTS)

    load_lora(eng, None)
    c = eng.gen_motion(audio, cfg_scale=2.0, seed=0)
    eng.render(port, wav, eng.coef_to_driving(c), a.out, f"{a.tag}__base", opts)
    print(f"[ok] {a.tag}__base.mp4", flush=True)

    if a.lora:
        load_lora(eng, a.lora)
        c2 = eng.gen_motion(audio, cfg_scale=2.0, seed=0)
        eng.render(port, wav, eng.coef_to_driving(c2), a.out, f"{a.tag}__trained", opts)
        print(f"[ok] {a.tag}__trained.mp4", flush=True)
        import numpy as np
        d = (c[:len(c2)] - c2[:len(c)]).abs().mean().item()
        print(f"[diff] mean |base - trained| = {d:.5f}  (0 => LoRA not applied)", flush=True)


if __name__ == "__main__":
    main()
