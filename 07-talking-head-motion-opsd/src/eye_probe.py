"""Learn a differentiable probe: normalized expression offset (63-d) -> eye state (blink, gaze_h, gaze_v).

The motion generator only emits implicit-keypoint expression offsets, so "eyes look alive" is not
directly measurable in motion space. We render single frames with LivePortrait for random expression
offsets around the generator's own distribution, label them with MediaPipe blendshapes, and fit a
small MLP. The probe is then a differentiable eye-state reward for OPSD.
"""
import argparse
import json
import os

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from engine import TalkingHeadEngine, EXP_SLICE
from video_metrics import MP_MODEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "eye_probe")


class EyeProbe(nn.Module):
    def __init__(self, d_in=63, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))

    def forward(self, d_exp):  # (..., 63) normalized exp offset vs first frame
        o = self.net(d_exp)
        # gaze outputs are changes relative to the face's resting gaze
        return {"blink": torch.sigmoid(o[..., 0]), "gaze_h": o[..., 1], "gaze_v": o[..., 2]}


class SourceFace:
    """Pre-computed LivePortrait source state for one portrait (mirrors live_portrait_wmg_pipeline)."""

    def __init__(self, eng: TalkingHeadEngine, image_path: str):
        from src.utils.io import load_image_rgb, resize_to_limit
        from src.utils.camera import get_rotation_matrix

        w = eng.wrapper
        img = resize_to_limit(load_image_rgb(image_path), 1280, 2)
        crop = eng.pipeline.cropper.crop_source_image(img, eng.pipeline.cropper.crop_cfg)
        I_s = w.prepare_source(crop["img_crop_256x256"])
        self.info = w.get_kp_info(I_s)
        self.R_s = get_rotation_matrix(self.info["pitch"], self.info["yaw"], self.info["roll"])
        self.f_s = w.extract_feature_3d(I_s)
        self.x_s = w.transform_keypoint(self.info)
        self.w = w
        self.std_exp = torch.as_tensor(eng.tpl["std_exp"], dtype=torch.float32, device=w.device).reshape(1, 21, 3)

    @torch.no_grad()
    def render_exp_offset(self, d_exp_norm: np.ndarray) -> np.ndarray:
        """Render 512x512 RGB with source pose and expression = source + d_exp_norm * std_exp."""
        d = torch.as_tensor(d_exp_norm, dtype=torch.float32, device=self.w.device).reshape(1, 21, 3)
        delta = self.info["exp"] + d * self.std_exp
        x_d = self.info["scale"] * (self.info["kp"] @ self.R_s + delta) + self.info["t"]
        x_d = self.w.stitching(self.x_s, x_d)
        out = self.w.warp_decode(self.f_s, self.x_s, x_d)
        return self.w.parse_output(out["out"])[0]


def mp_eye_labels(rgb: np.ndarray, lm):
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_blendshapes:
        return None
    bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
    blink = (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2
    gh = (bs["eyeLookOutLeft"] - bs["eyeLookInLeft"] + bs["eyeLookInRight"] - bs["eyeLookOutRight"]) / 2
    gv = (bs["eyeLookUpLeft"] + bs["eyeLookUpRight"] - bs["eyeLookDownLeft"] - bs["eyeLookDownRight"]) / 2
    return [blink, gh, gv]


def collect(eng, images, audio_paths, n_per_face, seed):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL), output_face_blendshapes=True, num_faces=1))
    rng = np.random.default_rng(seed)
    # empirical offsets from the generator itself (what OPSD will actually see) ...
    gen_offsets = []
    for ap in audio_paths:
        coef = eng.gen_motion(eng.load_audio(ap, 12), cfg_scale=2.0, seed=int(rng.integers(1e6)))
        e = coef[:, EXP_SLICE].numpy()
        gen_offsets.append(e - e[:1])
    gen_offsets = np.concatenate(gen_offsets)
    dim_std = gen_offsets.std(0) + 1e-3
    X, Y, F = [], [], []
    for fi, img in enumerate(images):
        face = SourceFace(eng, img)
        for k in range(n_per_face):
            r = rng.random()
            if r < 0.4:  # generator samples
                d = gen_offsets[rng.integers(len(gen_offsets))]
            elif r < 0.8:  # widened random directions (needed to ever see closed eyes / gaze shifts)
                d = rng.normal(0, 1, 63) * dim_std * rng.uniform(1.5, 4.0)
            else:  # sparse large pushes on a few dims
                d = np.zeros(63)
                idx = rng.choice(63, size=rng.integers(1, 5), replace=False)
                d[idx] = rng.normal(0, 1, len(idx)) * dim_std[idx] * 6
            y = mp_eye_labels(face.render_exp_offset(d), lm)
            if y is not None:
                X.append(d)
                Y.append(y)
                F.append(fi)
        print(f"face {fi}: {sum(1 for f in F if f == fi)} labeled frames", flush=True)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(F)


def fit(X, Y, F, epochs=400):
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    val = torch.tensor(F == F.max())  # hold out the last face entirely
    probe = EyeProbe()
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=1e-4)
    for ep in range(epochs):
        p = probe(Xt[~val])
        loss = (nn.functional.binary_cross_entropy(p["blink"], Yt[~val, 0].clamp(0, 1))
                + ((p["gaze_h"] - Yt[~val, 1]) ** 2).mean() + ((p["gaze_v"] - Yt[~val, 2]) ** 2).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = probe(Xt[val])
        rep = {}
        for i, k in enumerate(["blink", "gaze_h", "gaze_v"]):
            y, yh = Yt[val, i].numpy(), p[k].numpy()
            rep[k] = {"r2_heldout_face": float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                      "corr": float(np.corrcoef(y, yh)[0, 1]), "label_range": [float(y.min()), float(y.max())]}
    # refit on everything for the final probe
    probe_all = EyeProbe()
    opt = torch.optim.AdamW(probe_all.parameters(), lr=3e-3, weight_decay=1e-4)
    for ep in range(epochs):
        p = probe_all(Xt)
        loss = (nn.functional.binary_cross_entropy(p["blink"], Yt[:, 0].clamp(0, 1))
                + ((p["gaze_h"] - Yt[:, 1]) ** 2).mean() + ((p["gaze_v"] - Yt[:, 2]) ** 2).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probe_all, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-face", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jv = os.path.join(ROOT, "third_party", "JoyVASA", "assets", "examples")
    images = [os.path.join(ROOT, "data", "portraits", f) for f in ["host_f.png", "host_m.png", "host_f2.png"]]
    audios = [os.path.join(jv, "audios", f"joyvasa_00{i}.wav") for i in [1, 3, 5]]
    eng = TalkingHeadEngine(load_renderer=True)
    cache = os.path.join(OUT, "probe_data.npz")
    if os.path.exists(cache):
        d = dict(np.load(cache))
        X, Y, F = d["X"], d["Y"], d["F"]
    else:
        X, Y, F = collect(eng, images, audios, a.n_per_face, a.seed)
        d = {"X": X, "Y": Y, "F": F}
    if "B" not in d:  # resting-face labels, so gaze targets are changes, not identity-specific offsets
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL), output_face_blendshapes=True,
            num_faces=1))
        d["B"] = np.array([mp_eye_labels(SourceFace(eng, im).render_exp_offset(np.zeros(63)), lm) for im in images],
                          np.float32)
    np.savez(cache, **d)
    B = d["B"]
    Y = Y.copy()
    Y[:, 1:] = Y[:, 1:] - B[F, 1:]  # gaze as delta from resting gaze; blink stays absolute
    probe, rep = fit(X, Y, F)
    torch.save(probe.state_dict(), os.path.join(OUT, "eye_probe.pt"))
    rep["n_samples"] = int(len(X))
    rep["blink_label_frac_gt_0.4"] = float((Y[:, 0] > 0.4).mean())
    json.dump(rep, open(os.path.join(OUT, "probe_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
