"""OPSD (On-Policy Self-Distillation) post-training of JoyVASA's audio->motion diffusion transformer,
with the differentiable liveliness reward in motion_reward.py.

Same 4-step mechanism as 04/05 (image / video OPSD), applied to a (T=100 frames, 73-d) motion
sequence instead of an image latent:
  1. On-policy query: the frozen *behavior* adapter runs JoyVASA's own DDPM sampler (with CFG)
     from pure noise down to an intermediate step t1, giving a query state z_t1.
  2. Reward-guided targets: the behavior policy's clean prediction (anchor) at z_t1 is scored by
     the reward; the reward gradient w.r.t. z_t1 (lip keypoint dims masked out) gives a direction;
     z_t1 +/- step * direction are re-denoised by the behavior policy to positive / negative targets.
  3. Detached distillation: the *trainable* adapter's clean prediction at the same z_t1 is pulled to
     the positive target and pushed (with a margin) from the negative one.
  4. Periodic EMA refresh of the behavior adapter towards the trainable one.

LoRA (rank r) goes on every nn.Linear inside the denoising transformer's FFNs plus the motion
input/output projections; the audio encoder (HuBERT) and attention in_proj stay frozen.
"""
import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import TalkingHeadEngine
from eye_probe import EyeProbe
from motion_reward import LIP_DIMS, LivelinessReward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- dual LoRA
class DualLoRALinear(nn.Module):
    mode = "behavior"  # class-level switch shared by all wrapped layers

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        d_in, d_out = base.in_features, base.out_features
        self.A_t = nn.Parameter(torch.randn(r, d_in) / math.sqrt(d_in))
        self.B_t = nn.Parameter(torch.zeros(d_out, r))
        self.register_buffer("A_b", self.A_t.detach().clone())
        self.register_buffer("B_b", self.B_t.detach().clone())

    def forward(self, x):
        if DualLoRALinear.mode == "trainable":
            A, Bm = self.A_t, self.B_t
        elif DualLoRALinear.mode == "behavior":
            A, Bm = self.A_b, self.B_b
        else:  # "base"
            return self.base(x)
        return self.base(x) + F.linear(F.linear(x, A), Bm) * self.scale

    @torch.no_grad()
    def ema(self, decay):
        self.A_b.mul_(decay).add_(self.A_t.detach(), alpha=1 - decay)
        self.B_b.mul_(decay).add_(self.B_t.detach(), alpha=1 - decay)


def add_lora(denoiser: nn.Module, r: int, alpha: float):
    targets = []
    for name, mod in denoiser.named_modules():
        for cname, child in mod.named_children():
            if isinstance(child, nn.Linear) and not isinstance(child, nn.modules.linear.NonDynamicallyQuantizableLinear):
                full = f"{name}.{cname}" if name else cname
                if full.startswith("transformer") and not (cname.startswith("linear")):
                    continue
                targets.append((mod, cname, child))
    layers = []
    for mod, cname, child in targets:
        w = DualLoRALinear(child, r, alpha).to(child.weight.device)
        setattr(mod, cname, w)
        layers.append(w)
    return layers


def set_mode(m):
    DualLoRALinear.mode = m


# --------------------------------------------------------------------------- data
def audio_env_frames(audio: torch.Tensor, n_frames: int) -> torch.Tensor:
    """RMS loudness per 25 fps frame, aligned with JoyVASA's 1280-sample front padding."""
    a = F.pad(audio, (1280, 640))
    hop = 640
    need = n_frames * hop
    if len(a) < need:
        a = F.pad(a, (0, need - len(a)))
    fr = a[:need].reshape(n_frames, hop)
    return fr.pow(2).mean(1).sqrt()


@torch.no_grad()
def build_windows(eng, paths, stride_frames=25):
    mg = eng.mg
    feats, envs = [], []
    n = eng.n_motions
    n_samp = int(round(eng.audio_unit * n))
    for p in paths:
        audio = eng.load_audio(p)
        padded = F.pad(audio, (1280, 640))
        total_frames = int(len(padded) / eng.audio_unit)
        env_all = audio_env_frames(audio, max(total_frames, n))
        for start in range(0, max(total_frames - n, 0) + 1, stride_frames):
            seg = padded[int(start * eng.audio_unit): int(start * eng.audio_unit) + n_samp]
            if len(seg) < n_samp:
                seg = F.pad(seg, (0, n_samp - len(seg)))
            af = mg.extract_audio_feature(seg.unsqueeze(0).to(eng.device))
            feats.append(af[0].cpu())
            envs.append(env_all[start:start + n].cpu())
    return torch.stack(feats), torch.stack(envs)


# --------------------------------------------------------------------------- diffusion helpers
def guided_x0(mg, x_t, t, audio_feat, cfg_scale):
    """CFG-guided clean prediction exactly as DitTalkingHead.sample computes it (incremental, 1 cond)."""
    B = x_t.shape[0]
    null = mg.null_audio_feat.expand(B, mg.n_motions, -1)
    prev_m = mg.start_motion_feat.expand(B, -1, -1)
    prev_a = mg.start_audio_feat.expand(B, -1, -1)
    step = torch.full((B,), int(t), device=x_t.device, dtype=torch.long)
    out = mg.denoising_net(torch.cat([x_t, x_t]), torch.cat([null, audio_feat]), torch.cat([prev_m, prev_m]),
                           torch.cat([prev_a, prev_a]), torch.cat([step, step]), None)
    out = out[:, -mg.n_motions:]
    r_null, r_cond = out.chunk(2)
    return r_null + cfg_scale * (r_cond - r_null)


def ddpm_step(mg, x_t, x0, t, noise):
    s = mg.diffusion_sched
    alpha, ab, ab_prev = s.alphas[t], s.alpha_bars[t], s.alpha_bars[t - 1]
    sigma = s.get_sigmas(t, 0)
    c0 = (1 - ab_prev) * torch.sqrt(alpha) / (1 - ab)
    c1 = (1 - alpha) * torch.sqrt(ab_prev) / (1 - ab)
    return c0 * x_t + c1 * x0 + (sigma * noise if t > 1 else 0)


@torch.no_grad()
def rollout_to(mg, audio_feat, t_stop, cfg_scale, gen):
    B = audio_feat.shape[0]
    x = torch.randn(B, mg.n_motions, mg.motion_feat_dim, generator=gen).to(audio_feat.device)
    for t in range(mg.diffusion_sched.num_steps, t_stop, -1):
        x0 = guided_x0(mg, x, t, audio_feat, cfg_scale)
        x = ddpm_step(mg, x, x0, t, torch.randn(x.shape, generator=gen).to(x.device))
    return x


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--t1", type=int, default=12, help="query step (of 50)")
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--reward-step-size", type=float, default=3.0)
    ap.add_argument("--lambda-neg", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--ema-every", type=int, default=10)
    ap.add_argument("--ema-decay", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "opsd"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    eng = TalkingHeadEngine(load_renderer=False)
    mg = eng.mg
    dev = eng.device
    for p in mg.parameters():
        p.requires_grad_(False)
    mg.eval()

    probe = EyeProbe()
    probe.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "eye_probe", "eye_probe.pt"), map_location="cpu"))
    reward_fn = LivelinessReward(eng.tpl, eye_probe=probe).to(dev)
    probe.to(dev)

    layers = add_lora(mg.denoising_net, a.lora_r, a.lora_alpha)
    train_params = [p for l in layers for p in (l.A_t, l.B_t)]
    print(f"[lora] {len(layers)} layers, {sum(p.numel() for p in train_params)/1e3:.1f}K trainable params "
          f"(base denoiser {sum(p.numel() for p in mg.denoising_net.parameters() if not p.requires_grad)/1e6:.2f}M frozen)")
    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=0.0)

    train_paths = sorted(glob.glob(os.path.join(ROOT, "data", "audio", "train", "*.wav")))
    feats, envs = build_windows(eng, train_paths)
    print(f"[data] {len(train_paths)} train clips -> {len(feats)} 4-second windows")

    lip_mask = torch.ones(mg.motion_feat_dim, device=dev)
    lip_mask[LIP_DIMS] = 0.0
    gen = torch.Generator().manual_seed(a.seed)
    log = open(os.path.join(a.out, "metrics.jsonl"), "w")
    t_start = time.time()
    for step in range(1, a.rounds + 1):
        idx = torch.randint(len(feats), (a.batch_size,), generator=gen)
        af, env = feats[idx].to(dev), envs[idx].to(dev)

        # 1. on-policy query from the behavior policy
        set_mode("behavior")
        z = rollout_to(mg, af, a.t1, a.cfg_scale, gen)

        # 2. reward-guided targets
        z_q = z.detach().requires_grad_(True)
        anchor = guided_x0(mg, z_q, a.t1, af, a.cfg_scale)
        r, terms = reward_fn(anchor, env, return_terms=True)
        g, = torch.autograd.grad(r.sum(), z_q)
        g = g * lip_mask
        d = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-8)
        with torch.no_grad():
            tgt_pos = guided_x0(mg, z + a.reward_step_size * d, a.t1, af, a.cfg_scale)
            tgt_neg = guided_x0(mg, z - a.reward_step_size * d, a.t1, af, a.cfg_scale)

        # 3. detached distillation fit of the trainable adapter
        set_mode("trainable")
        pred = guided_x0(mg, z.detach(), a.t1, af, a.cfg_scale)
        loss_pos = F.mse_loss(pred, tgt_pos)
        loss_neg = F.mse_loss(pred, tgt_neg)
        loss = loss_pos + a.lambda_neg * F.relu(a.margin - loss_neg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()

        # 4. EMA refresh
        if step % a.ema_every == 0:
            for l in layers:
                l.ema(a.ema_decay)

        rec = {"step": step, "loss": loss.item(), "loss_pos": loss_pos.item(), "loss_neg": loss_neg.item(),
               "reward": r.mean().item(), **{k: v.mean().item() for k, v in terms.items()},
               "elapsed_s": round(time.time() - t_start, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0 or step == 1:
            print(f"[{step:4d}/{a.rounds}] loss={rec['loss']:.5f} R={rec['reward']:.3f} "
                  + " ".join(f"{k}={rec[k]:.3f}" for k in terms) + f" t={rec['elapsed_s']:.0f}s", flush=True)
        if a.dry_run and step >= 3:
            break

    log.close()
    sd = {f"{i}.{k}": v for i, l in enumerate(layers) for k, v in
          (("A_t", l.A_t.detach().cpu()), ("B_t", l.B_t.detach().cpu()), ("A_b", l.A_b.cpu()), ("B_b", l.B_b.cpu()))}
    torch.save({"lora": sd, "r": a.lora_r, "alpha": a.lora_alpha, "args": vars(a)}, os.path.join(a.out, "lora.pt"))
    print("[ok] saved", os.path.join(a.out, "lora.pt"))


if __name__ == "__main__":
    main()
