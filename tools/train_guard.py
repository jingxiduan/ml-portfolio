#!/usr/bin/env python3
"""Training guard for 07 (motion OPSD) and 08 (I2V OPSD).

Tails a trainer's per-step JSONL log and stops the run when it starts to
collapse, so a bad run does not burn the GPU window.

Thresholds are calibrated on the existing logs (see docs/NEXT_H20_TASKS.md,
section "Safety constraints"). Do NOT loosen them to make a run pass.

Live mode (run next to the trainer):
    python tools/train_guard.py --kind i2v    --log outputs/X/train_metrics.jsonl --pid $TRAIN_PID
    python tools/train_guard.py --kind motion --log outputs/X/metrics.jsonl       --pid $TRAIN_PID

Replay mode (check thresholds against an old log, no process involved):
    python tools/train_guard.py --kind i2v --log outputs/opsd_tier2_v2/train_metrics.jsonl --replay

Self test:
    python tools/train_guard.py --selftest

Exit codes: 0 finished/clean, 3 guard tripped, 2 usage error.
On trip it writes GUARD_STOP.json next to the log and sends SIGTERM to --pid.
"""
import argparse
import json
import math
import os
import signal
import sys
import time

W = 50  # window length in steps

# (name, key, mode, threshold, min_steps)
#   abs_below       : window mean < threshold
#   ratio_above     : window mean > threshold * first-window mean
#   drop_from_base  : window mean < first-window mean - threshold
#   drop_from_best  : window mean < best earlier (non-overlapping) window mean - threshold
RULES = {
    # 08: reward-grad "grad_norm" is always 1.0 (it is the normalised reward gradient,
    # not the parameter gradient), so it is deliberately NOT monitored.
    "i2v": [
        ("loss_blowup", "loss_plus", "ratio_above", 10.0, 2 * W),
        ("subject_drift", "R_subject_consistency", "abs_below", 0.95, W),
        ("reward_collapse", "reward", "drop_from_base", 1.0, 2 * W),
        ("naturalness_worse", "R_naturalness_penalty", "drop_from_base", 0.5, 2 * W),
    ],
    "motion": [
        ("loss_blowup", "loss", "ratio_above", 3.0, 2 * W),
        ("ood_blowup", "ood", "abs_below", -0.05, W),
        ("reward_collapse", "reward", "drop_from_best", 0.3, 3 * W),
        ("jitter_blowup", "jitter", "drop_from_base", 0.5, 2 * W),
        # tightening the jitter penalty can freeze the head; amp starts ~0.26 and rises to ~0.85
        ("head_frozen", "amp", "abs_below", 0.15, W),
    ],
}


def _wmean(rows, key, a, b):
    vals = [r[key] for r in rows[a:b] if key in r and isinstance(r[key], (int, float))]
    return sum(vals) / len(vals) if vals else None


def evaluate(rows, kind):
    """Return (rule_name, message) if the run should stop, else None."""
    n = len(rows)
    if n == 0:
        return None
    last = rows[-1]
    step = last.get("step", n)
    for k, v in last.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(v):
            return "nonfinite", f"{k}={v} at step {step}"
    for name, key, mode, thr, min_steps in RULES[kind]:
        if n < min_steps or n < W:
            continue
        cur = _wmean(rows, key, n - W, n)
        if cur is None:
            continue
        if mode == "abs_below":
            if cur < thr:
                return name, f"{key} window-mean {cur:.4f} < {thr} at step {step}"
            continue
        base = _wmean(rows, key, 0, W)
        if base is None:
            continue
        if mode == "ratio_above":
            if base > 0 and cur > thr * base:
                return name, f"{key} window-mean {cur:.5f} > {thr}x baseline {base:.5f} at step {step}"
        elif mode == "drop_from_base":
            if cur < base - thr:
                return name, f"{key} window-mean {cur:.4f} dropped > {thr} below baseline {base:.4f} at step {step}"
        elif mode == "drop_from_best":
            vals = [r.get(key) for r in rows]
            best = None
            if all(isinstance(v, (int, float)) for v in vals):
                pref = [0.0]
                for v in vals:
                    pref.append(pref[-1] + v)
                for end in range(W, n - W + 1):
                    m = (pref[end] - pref[end - W]) / W
                    best = m if best is None else max(best, m)
            if best is not None and cur < best - thr:
                return name, f"{key} window-mean {cur:.4f} fell > {thr} below earlier best {best:.4f} at step {step}"
    return None


def read_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # partially written last line
    return rows


def trip(log, pid, rule, msg):
    out = os.path.join(os.path.dirname(os.path.abspath(log)), "GUARD_STOP.json")
    with open(out, "w") as f:
        json.dump({"rule": rule, "message": msg, "time": time.strftime("%F %T")}, f, indent=2)
    print(f"[GUARD] TRIPPED rule={rule}: {msg}", flush=True)
    print(f"[GUARD] wrote {out}", flush=True)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[GUARD] sent SIGTERM to pid {pid}", flush=True)
        except ProcessLookupError:
            pass


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def replay(log, kind):
    rows = read_rows(log)
    for i in range(1, len(rows) + 1):
        res = evaluate(rows[:i], kind)
        if res:
            print(f"TRIP  {log}: {res[0]}: {res[1]}")
            return 3
    print(f"OK    {log}: {len(rows)} steps, no trip")
    return 0


def live(log, kind, pid, poll, stall_sec):
    print(f"[GUARD] kind={kind} log={log} pid={pid} window={W}", flush=True)
    last_n, last_change = 0, time.time()
    while True:
        rows = read_rows(log)
        if len(rows) != last_n:
            last_n, last_change = len(rows), time.time()
        res = evaluate(rows, kind)
        if res:
            trip(log, pid, *res)
            return 3
        if pid and not pid_alive(pid):
            print(f"[GUARD] pid {pid} exited after {len(rows)} steps, no trip", flush=True)
            return 0
        if pid and time.time() - last_change > stall_sec:
            trip(log, pid, "stalled", f"no new log line for {stall_sec}s (last step count {last_n})")
            return 3
        time.sleep(poll)


def selftest():
    import random
    random.seed(0)
    ok = True

    def mk(n, f):
        return [dict(step=i + 1, **f(i)) for i in range(n)]

    good_i2v = mk(300, lambda i: dict(loss_plus=3.8e-4 + random.random() * 1e-4, reward=-0.3 + random.gauss(0, 0.3),
                                      R_subject_consistency=0.984, R_naturalness_penalty=-1.2 + 0.001 * i))
    good_motion = mk(600, lambda i: dict(loss=0.0101, amp=0.3 + 0.0009 * i, reward=0.33 + 0.0015 * i + random.gauss(0, 0.05), ood=-0.001, jitter=-0.31 - 0.0002 * i))
    cases = [
        ("good i2v", "i2v", good_i2v, False),
        ("good motion", "motion", good_motion, False),
        ("nan", "i2v", good_i2v[:100] + [dict(step=101, loss_plus=float("nan"), reward=0.0)], True),
        ("subject drift", "i2v", good_i2v[:100] + mk(60, lambda i: dict(loss_plus=3.8e-4, reward=-0.3, R_subject_consistency=0.80, R_naturalness_penalty=-1.2)), True),
        ("reward collapse i2v", "i2v", good_i2v[:100] + mk(100, lambda i: dict(loss_plus=3.8e-4, reward=-2.5, R_subject_consistency=0.984, R_naturalness_penalty=-1.2)), True),
        ("loss blowup motion", "motion", good_motion[:200] + mk(60, lambda i: dict(loss=0.08, reward=1.0, ood=-0.001, jitter=-0.4, amp=0.8)), True),
        ("head frozen motion", "motion", good_motion[:200] + mk(60, lambda i: dict(loss=0.0101, reward=1.0, ood=-0.001, jitter=-0.1, amp=0.05)), True),
        ("ood blowup motion", "motion", good_motion[:200] + mk(60, lambda i: dict(loss=0.0101, reward=1.0, ood=-0.3, jitter=-0.4, amp=0.8)), True),
    ]
    for name, kind, rows, expect in cases:
        tripped = False
        for i in range(1, len(rows) + 1):
            if evaluate(rows[:i], kind):
                tripped = True
                break
        status = "PASS" if tripped == expect else "FAIL"
        ok &= tripped == expect
        print(f"  {status}  {name}: tripped={tripped} expected={expect}")
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(RULES))
    ap.add_argument("--log")
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--stall-sec", type=float, default=1200.0)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.kind or not a.log:
        ap.error("--kind and --log are required")
    sys.exit(replay(a.log, a.kind) if a.replay else live(a.log, a.kind, a.pid, a.poll, a.stall_sec))


if __name__ == "__main__":
    main()
