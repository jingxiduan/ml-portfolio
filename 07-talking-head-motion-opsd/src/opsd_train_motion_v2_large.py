"""Track C — retrain OPSD-LoRA with the fixed reward v2 (gaussian-around-human
blink + hard jitter cap). Same OPSD mechanism as v1, swapping only the reward.

Produces outputs/opsd_v2/lora.pt and metrics.jsonl — intentionally does NOT
overwrite outputs/opsd/lora.pt so v1 remains comparable.
"""
import argparse
import glob
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from engine import TalkingHeadEngine
from eye_probe import EyeProbe
from motion_reward import LIP_DIMS
from motion_reward_v2 import LivelinessRewardV2
from opsd_train_motion import add_lora, build_windows, guided_x0, rollout_to, set_mode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--t1", type=int, default=12)
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
    # reward v2 knobs
    ap.add_argument("--blink-target-mean", type=float, default=0.15)
    ap.add_argument("--blink-overshoot-penalty", type=float, default=4.0)
    ap.add_argument("--jitter-cap-deg", type=float, default=1.0)
    ap.add_argument("--jitter-cap-penalty", type=float, default=5.0)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "audio", "train"),
                    help="directory of training wavs (default: legacy 25 TTS clips)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "opsd_v2"))
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
    probe.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "eye_probe", "eye_probe.pt"),
                                     map_location="cpu"))
    reward_fn = LivelinessRewardV2(
        eng.tpl, eye_probe=probe,
        blink_target_mean=a.blink_target_mean,
        blink_overshoot_penalty=a.blink_overshoot_penalty,
        jitter_cap_deg=a.jitter_cap_deg,
        jitter_cap_penalty=a.jitter_cap_penalty,
    ).to(dev)
    probe.to(dev)

    layers = add_lora(mg.denoising_net, a.lora_r, a.lora_alpha)
    train_params = [p for l in layers for p in (l.A_t, l.B_t)]
    print(f"[lora-v2] {len(layers)} layers, {sum(p.numel() for p in train_params)/1e3:.1f}K trainable params")
    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=0.0)

    train_paths = sorted(glob.glob(os.path.join(a.data_dir, "*.wav")))
    feats, envs = build_windows(eng, train_paths)
    print(f"[data] {len(train_paths)} train clips -> {len(feats)} 4-sec windows")

    lip_mask = torch.ones(mg.motion_feat_dim, device=dev)
    lip_mask[LIP_DIMS] = 0.0
    gen = torch.Generator().manual_seed(a.seed)
    log = open(os.path.join(a.out, "metrics.jsonl"), "w")
    t_start = time.time()

    for step in range(1, a.rounds + 1):
        idx = torch.randint(len(feats), (a.batch_size,), generator=gen)
        af, env = feats[idx].to(dev), envs[idx].to(dev)

        set_mode("behavior")
        z = rollout_to(mg, af, a.t1, a.cfg_scale, gen)

        z_q = z.detach().requires_grad_(True)
        anchor = guided_x0(mg, z_q, a.t1, af, a.cfg_scale)
        r, terms = reward_fn(anchor, env, return_terms=True)
        g, = torch.autograd.grad(r.sum(), z_q)
        g = g * lip_mask
        d = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-8)
        with torch.no_grad():
            tgt_pos = guided_x0(mg, z + a.reward_step_size * d, a.t1, af, a.cfg_scale)
            tgt_neg = guided_x0(mg, z - a.reward_step_size * d, a.t1, af, a.cfg_scale)

        set_mode("trainable")
        pred = guided_x0(mg, z.detach(), a.t1, af, a.cfg_scale)
        loss_pos = F.mse_loss(pred, tgt_pos)
        loss_neg = F.mse_loss(pred, tgt_neg)
        loss = loss_pos + a.lambda_neg * F.relu(a.margin - loss_neg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()

        if step % a.ema_every == 0:
            for l in layers:
                l.ema(a.ema_decay)

        rec = {"step": step, "loss": loss.item(), "loss_pos": loss_pos.item(),
               "loss_neg": loss_neg.item(), "reward": r.mean().item(),
               **{k: v.mean().item() for k, v in terms.items()},
               "elapsed_s": round(time.time() - t_start, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 20 == 0 or step == 1:
            print(f"[{step:4d}/{a.rounds}] loss={rec['loss']:.5f} R={rec['reward']:.3f} "
                  + " ".join(f"{k}={rec[k]:.3f}" for k in terms)
                  + f"  t={rec['elapsed_s']:.0f}s", flush=True)

    log.close()
    sd = {f"{i}.{k}": v for i, l in enumerate(layers) for k, v in
          (("A_t", l.A_t.detach().cpu()), ("B_t", l.B_t.detach().cpu()),
           ("A_b", l.A_b.cpu()), ("B_b", l.B_b.cpu()))}
    torch.save({"lora": sd, "r": a.lora_r, "alpha": a.lora_alpha, "args": vars(a)},
               os.path.join(a.out, "lora.pt"))
    print("[ok] saved", os.path.join(a.out, "lora.pt"))


if __name__ == "__main__":
    main()
