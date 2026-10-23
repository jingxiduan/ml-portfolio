"""Thin wrapper around JoyVASA (audio -> motion diffusion) + LivePortrait (motion -> pixels).

Separates the two stages so motion can be generated / scored / trained without
paying the rendering cost, and rendering is only done for demos.
"""
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOYVASA_DIR = os.path.join(ROOT, "third_party", "JoyVASA")
if JOYVASA_DIR not in sys.path:
    sys.path.insert(0, JOYVASA_DIR)

import librosa  # noqa: E402

from src.config.argument_config import ArgumentConfig  # noqa: E402
from src.config.crop_config import CropConfig  # noqa: E402
from src.config.inference_config import InferenceConfig  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402

FPS = 25
EXP_SLICE = slice(0, 63)
SCALE_IDX = 63
T_SLICE = slice(64, 67)
PITCH_IDX, YAW_IDX, ROLL_IDX = 67, 68, 69


def _partial(cls, kw):
    return cls(**{k: v for k, v in kw.items() if hasattr(cls, k)})


@dataclass
class RenderOptions:
    flag_do_crop: bool = True
    flag_pasteback: bool = True
    flag_stitching: bool = True
    flag_relative_motion: bool = True
    driving_option: str = "pose-friendly"
    driving_multiplier: float = 1.0
    source_max_dim: int = 1280
    animation_region: str = "all"
    extra: dict = field(default_factory=dict)


class TalkingHeadEngine:
    def __init__(self, load_renderer: bool = True):
        args = ArgumentConfig()
        inf_cfg = _partial(InferenceConfig, args.__dict__)
        crop_cfg = _partial(CropConfig, args.__dict__)
        if load_renderer:
            from src.live_portrait_wmg_pipeline import LivePortraitPipeline

            self.pipeline = LivePortraitPipeline(inference_cfg=inf_cfg, crop_cfg=crop_cfg)
            self.wrapper = self.pipeline.live_portrait_wrapper
        else:
            from src.live_portrait_wmg_wrapper import LivePortraitWrapper  # still loads renderer weights

            self.pipeline = None
            self.wrapper = LivePortraitWrapper(inference_cfg=inf_cfg)
        self.device = self.wrapper.device
        self.mg = self.wrapper.motion_generator
        self.tpl = self.wrapper.templete_dict
        self.n_motions = self.wrapper.n_motions
        self.n_prev = self.wrapper.n_prev_motions
        self.audio_unit = 16000.0 / FPS

    # ------------------------------------------------------------------ audio
    def load_audio(self, path: str, max_seconds: Optional[float] = None) -> torch.Tensor:
        audio, _ = librosa.load(path, sr=16000, mono=True)
        if max_seconds is not None:
            audio = audio[: int(max_seconds * 16000)]
        return torch.from_numpy(audio)

    def audio_windows(self, audio: torch.Tensor):
        """Split audio exactly the way JoyVASA does. Returns list of (1, n_samples) windows,
        list of indicators, and number of padded frames at the end."""
        audio = F.pad(audio, (1280, 640), "constant", 0)
        n_audio_samples = round(self.audio_unit * self.n_motions)
        clip_len = int(len(audio) / 16000 * FPS)
        n_sub = 1 if clip_len <= self.n_motions else math.ceil(clip_len / self.n_motions)
        n_pad_samples = n_audio_samples * n_sub - len(audio)
        n_pad_frames = math.ceil(n_pad_samples / self.audio_unit)
        if n_pad_samples > 0:
            audio = F.pad(audio, (0, n_pad_samples), value=0)
        windows, indicators = [], []
        for i in range(n_sub):
            s, e = i * self.n_motions, (i + 1) * self.n_motions
            windows.append(audio[round(s * self.audio_unit): round(e * self.audio_unit)].unsqueeze(0))
            ind = torch.ones((1, self.n_motions))
            if i == n_sub - 1 and n_pad_frames > 0:
                ind[:, -n_pad_frames:] = 0
            indicators.append(ind)
        return windows, indicators, max(n_pad_frames, 0)

    # ----------------------------------------------------------------- motion
    @torch.no_grad()
    def gen_motion(self, audio: torch.Tensor, cfg_scale: float = 2.0, seed: int = 0,
                   model=None) -> torch.Tensor:
        """Returns normalized motion coefficients (T, 73) on CPU."""
        model = model or self.mg
        torch.manual_seed(seed)
        windows, indicators, n_pad = self.audio_windows(audio)
        prev_m, prev_a, noise, out = None, None, None, []
        for i, (w, ind) in enumerate(zip(windows, indicators)):
            w, ind = w.to(self.device), ind.to(self.device)
            m, noise, a = model.sample(w, prev_m, prev_a, noise, indicator=ind, cfg_mode="incremental",
                                       cfg_cond=None, cfg_scale=cfg_scale, dynamic_threshold=0)
            prev_m = m[:, -self.n_prev:].clone()
            prev_a = a[:, -self.n_prev:]
            if i == len(windows) - 1 and n_pad > 0:
                m = m[:, :-n_pad]
            out.append(m)
        return torch.cat(out, dim=1)[0].float().cpu()

    def denorm_pose(self, coef: torch.Tensor) -> dict:
        """Head pose in degrees + translation, differentiable, works for (T,73) or (B,T,73)."""
        t = self.tpl

        def mm(x, k):
            lo, hi = float(t[f"min_{k}"][0]), float(t[f"max_{k}"][0])
            return x * (hi - lo) + lo

        return {
            "pitch": mm(coef[..., PITCH_IDX], "pitch"),
            "yaw": mm(coef[..., YAW_IDX], "yaw"),
            "roll": mm(coef[..., ROLL_IDX], "roll"),
            "exp": coef[..., EXP_SLICE] * torch.as_tensor(t["std_exp"], dtype=coef.dtype, device=coef.device)
            + torch.as_tensor(t["mean_exp"], dtype=coef.dtype, device=coef.device),
        }

    def coef_to_driving(self, coef: torch.Tensor, smooth_alpha: Optional[float] = None) -> dict:
        """Convert (T,73) normalized coefficients to LivePortrait driving dict.
        smooth_alpha: EMA weight for the new frame on pose/scale/t (JoyVASA default is 0.01)."""
        t = self.tpl
        coef = coef.cpu()
        exp = coef[:, EXP_SLICE] * t["std_exp"] + t["mean_exp"]
        scale = coef[:, SCALE_IDX:SCALE_IDX + 1] * (t["max_scale"] - t["min_scale"]) + t["min_scale"]
        tr = coef[:, T_SLICE] * (t["max_t"] - t["min_t"]) + t["min_t"]
        pose = self.denorm_pose(coef)
        pyr = torch.stack([pose["pitch"], pose["yaw"], pose["roll"]], dim=-1)
        rigid = torch.cat([scale, tr, pyr], dim=-1).numpy().astype(np.float32)
        if smooth_alpha is not None:
            s = rigid[0].copy()
            for i in range(len(rigid)):
                s = smooth_alpha * rigid[i] + (1 - smooth_alpha) * s if i else s
                rigid[i] = s
        motion = []
        for i in range(coef.shape[0]):
            p, y, r = (torch.tensor(rigid[i, 4:5]), torch.tensor(rigid[i, 5:6]), torch.tensor(rigid[i, 6:7]))
            R = get_rotation_matrix(p, y, r).reshape(1, 3, 3).numpy().astype(np.float32)
            motion.append({
                "exp": exp[i].reshape(1, 21, 3).numpy().astype(np.float32),
                "scale": rigid[i, 0:1].reshape(1, 1), "t": rigid[i, 1:4].reshape(1, 3),
                "pitch": rigid[i, 4:5].reshape(1, 1), "yaw": rigid[i, 5:6].reshape(1, 1),
                "roll": rigid[i, 6:7].reshape(1, 1), "R": R,
            })
        return {"n_frames": coef.shape[0], "output_fps": FPS, "motion": motion}

    # ----------------------------------------------------------------- render
    def render(self, image_path: str, audio_path: str, driving: dict, out_dir: str, tag: str,
               opts: Optional[RenderOptions] = None) -> str:
        assert self.pipeline is not None, "renderer not loaded"
        opts = opts or RenderOptions()
        cfg = self.wrapper.inference_cfg
        for k, v in {**opts.__dict__, **opts.extra}.items():
            if k != "extra" and hasattr(cfg, k):
                setattr(cfg, k, v)
        args = ArgumentConfig(reference=image_path, audio=audio_path, output_dir=out_dir)
        orig = self.wrapper.gen_motion_sequence
        self.wrapper.gen_motion_sequence = lambda _args: driving
        try:
            final = self.pipeline.execute(args)
        finally:
            self.wrapper.gen_motion_sequence = orig
        dst = os.path.join(out_dir, f"{tag}.mp4")
        os.replace(final, dst)
        tmp = final.replace(".mp4", "_temp.mp4")
        if os.path.exists(tmp):
            os.remove(tmp)
        return dst
