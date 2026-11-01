"""OPSD training with reward v5 (= v4 + the `expr` expression-dynamics term).

Identical to `opsd_train_motion_v4.py` except:
  * the reward class is LivelinessRewardV5
  * three new CLI knobs: --expr-target / --expr-tol / --expr-weight
    (plus --expr-env-weight, default 0 = off)
  * `expr` and `expr_active_frac` are logged to metrics.jsonl

Everything else -- LoRA config, lr, rounds, EMA, OPSD step, lip masking --
is byte-identical to v4 so the diff is attributable to the new term alone.
"""
import argparse
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from engine import TalkingHeadEngine
from eye_probe import EyeProbe
from motion_reward import LIP_DIMS
from motion_reward_v5 import LivelinessRewardV5
from opsd_train_motion import add_lora, build_windows, guided_x0, rollout_to, set_mode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=200,
                    help="linear warmup steps before the cosine decay")
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--t1", type=int, default=12)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--reward-step-size", type=float, default=3.0)
    ap.add_argument("--lambda-neg", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--ema-every", type=int, default=10)
    ap.add_argument("--ema-decay", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blink-target-mean", type=float, default=0.15)
    ap.add_argument("--blink-overshoot-penalty", type=float, default=4.0)
    ap.add_argument("--jitter-cap-deg", type=float, default=0.2)
    ap.add_argument("--jitter-cap-penalty", type=float, default=5.0)
    ap.add_argument("--amp-target-deg", type=float, default=1.878)
    ap.add_argument("--amp-tol-deg", type=float, default=1.0)
    ap.add_argument("--weights", default=None,
                    help='JSON reward weights, e.g. \'{"amp":0.35,"jitter":1.2}\'')
    # ---- v5: expression dynamics -------------------------------------------
    ap.add_argument("--expr-target", type=float, default=1.0,
                    help="measured non-lip expression low-band std, normalised units")
    ap.add_argument("--expr-tol", type=float, default=1.0)
    ap.add_argument("--expr-weight", type=float, default=1.0)
    ap.add_argument("--expr-env-weight", type=float, default=0.0,
                    help=">0 adds speech<->expression coupling (off by default in P1)")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "audio", "train"),
                    help="directory of training wavs (default: legacy 25 TTS clips)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "opsd_v5"))
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
    weights = json.loads(a.weights) if a.weights else None
    if weights:
        print(f"[style] reward weights: {weights}", flush=True)
    reward_fn = LivelinessRewardV5(
        eng.tpl, eye_probe=probe, weights=weights,
        blink_target_mean=a.blink_target_mean,
        blink_overshoot_penalty=a.blink_overshoot_penalty,
        amp_target_deg=a.amp_target_deg,
        amp_tol_deg=a.amp_tol_deg,
        jitter_cap_deg=a.jitter_cap_deg,
        jitter_cap_penalty=a.jitter_cap_penalty,
        expr_target=a.expr_target,
        expr_tol=a.expr_tol,
        expr_weight=a.expr_weight,
        expr_env_weight=a.expr_env_weight,
    ).to(dev)
    probe.to(dev)
    print(f"[reward-v5] expr_target={a.expr_target} expr_tol={a.expr_tol} "
          f"expr_weight={a.expr_weight} expr_env_weight={a.expr_env_weight}", flush=True)

    layers = add_lora(mg.denoising_net, a.lora_r, a.lora_alpha)
    train_params = [p for l in layers for p in (l.A_t, l.B_t)]
    print(f"[lora-v5] {len(layers)} layers, "
          f"{sum(p.numel() for p in train_params)/1e3:.1f}K trainable params")
    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=0.0)

    def lr_at(step):
        if step < a.warmup:                       # linear warmup 0 -> lr
            return float(step) / float(max(a.warmup, 1))
        prog = (step - a.warmup) / float(max(a.rounds - a.warmup, 1))
        prog = min(max(prog, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return a.min_lr_frac + (1.0 - a.min_lr_frac) * cos
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    print(f"[opt] AdamW lr={a.lr} warmup={a.warmup} -> cosine to {a.min_lr_frac}x", flush=True)

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
        sched.step()

        if step % a.ema_every == 0:
            for l in layers:
                l.ema(a.ema_decay)

        rec = {"step": step, "loss": loss.item(), "loss_pos": loss_pos.item(),
               "loss_neg": loss_neg.item(), "reward": r.mean().item(),
               **{k: v.mean().item() for k, v in terms.items()},
               "cap_active_frac": getattr(reward_fn, '_last_cap_active_frac', 0.0),
               "expr_active_frac": getattr(reward_fn, '_last_expr_active_frac', 0.0),
               "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t_start, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 20 == 0 or step == 1:
            print(f"[{step:4d}/{a.rounds}] loss={rec['loss']:.5f} R={rec['reward']:.3f} "
                  + " ".join(f"{k}={rec[k]:.3f}" for k in terms)
                  + f" cap_frac={rec['cap_active_frac']:.2f}"
                  + f" expr_frac={rec['expr_active_frac']:.2f}"
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
