"""Apply the minimal MPS/newer-PyTorch fixes to a fresh JoyVASA checkout.

third_party/ is gitignored (it's downloaded), so these small edits that make JoyVASA run on
Apple Silicon are recorded here instead. Idempotent: safe to run twice.

Fixes:
  1. helper.py  - allowlist argparse.Namespace + pathlib.PosixPath for torch.load weights_only,
                  and pass device=... into DitTalkingHead so submodules land on MPS.
  2. common.py  - enc_dec_mask default device 'cuda' -> 'cpu' (buffer, moved with the model).
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JV = os.path.join(ROOT, "third_party", "JoyVASA", "src")


def patch(path, subs):
    with open(path) as f:
        txt = f.read()
    for old, new in subs:
        if new in txt:
            continue
        if old not in txt:
            raise SystemExit(f"anchor not found in {path}:\n{old}")
        txt = txt.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(txt)
    print("patched", os.path.relpath(path, ROOT))


patch(os.path.join(JV, "utils", "helper.py"), [
    ("        model_data = torch.load(ckpt_path, map_location=device)\n"
     "        model_args = NullableArgs(model_data['args'])",
     "        import argparse, pathlib\n"
     "        with torch.serialization.safe_globals([argparse.Namespace, pathlib.PosixPath]):\n"
     "            model_data = torch.load(ckpt_path, map_location=device)\n"
     "        model_args = NullableArgs(model_data['args'])"),
    ("        model = DitTalkingHead(motion_feat_dim=model_args.motion_feat_dim, ",
     "        model = DitTalkingHead(device=device,\n"
     "                               motion_feat_dim=model_args.motion_feat_dim, "),
])

patch(os.path.join(JV, "modules", "common.py"), [
    ("def enc_dec_mask(T, S, frame_width=2, expansion=0, device='cuda'):",
     "def enc_dec_mask(T, S, frame_width=2, expansion=0, device='cpu'):"),
])

print("done")
