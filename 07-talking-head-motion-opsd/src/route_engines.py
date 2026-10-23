"""Engine routing prototype: learn which engine each user segment prefers from feedback alone,
using engine feature profiles (engine_profiles.py) as item features.

IMPORTANT: user preferences here are SIMULATED. There is no real user data in this repo.
Three synthetic segments get a hidden linear utility over standardized engine features plus
logistic choice noise. The point is to show the pipeline (measurable engine features ->
contextual bandit -> routing) works and how fast it learns, not to claim what real users like.

Policies compared over the same simulated traffic:
  random          pick any engine
  global_best     the single engine with the best average utility across segments (oracle, no personalization)
  linucb          LinUCB, context = segment one-hot (x) engine feature profile
  oracle          per-segment best engine (upper bound)
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USE = ["head_speed_mean", "head_jitter", "nod_rate", "speech_head_corr", "still_ratio", "blink_rate", "gaze_std",
       "jaw_open_mean"]

SEGMENTS = {
    # livestream ecommerce: energetic, lots of head movement, nodding, eye contact changes
    "ecommerce_live": {"head_speed_mean": 1.0, "nod_rate": 0.8, "still_ratio": -0.8, "blink_rate": 0.4,
                       "gaze_std": 0.3, "head_jitter": -0.6, "speech_head_corr": 0.5},
    # news / corporate announcement: composed, low jitter, natural blinks, modest movement
    "news_corporate": {"head_speed_mean": -0.6, "head_jitter": -1.2, "nod_rate": -0.3, "blink_rate": 0.6,
                       "gaze_std": -0.4, "still_ratio": 0.3},
    # knowledge explainer: in between, values speech-synced movement and lively eyes
    "knowledge": {"speech_head_corr": 1.0, "blink_rate": 0.7, "gaze_std": 0.5, "head_jitter": -0.7,
                  "nod_rate": 0.3},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6000)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--noise", type=float, default=1.0, help="logit noise scale of the simulated choice")
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()
    prof = json.load(open(os.path.join(ROOT, "outputs", "profiles", "engine_profiles.json")))["profiles"]
    engines = list(prof)
    X = np.array([[prof[e][f] for f in USE] for e in engines])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)  # standardized engine features
    seg_names = list(SEGMENTS)
    W = np.array([[SEGMENTS[s].get(f, 0.0) for f in USE] for s in seg_names])
    U = W @ X.T  # (segments, engines) true utility
    p_seg = np.array([0.4, 0.3, 0.3])
    n_e, n_s, d = len(engines), len(seg_names), X.shape[1]

    def ctx(s, e):
        v = np.zeros(n_s * (d + 1))
        v[s * (d + 1): (s + 1) * (d + 1)] = np.r_[1.0, X[e]]
        return v

    def p_like(s, e):  # probability the simulated user is satisfied (e.g. keeps the video / doesn't regenerate)
        return 1 / (1 + np.exp(-(U[s, e] - U[s].mean()) / a.noise))

    best_global = int(np.argmax(p_seg @ U))
    res = {k: [] for k in ["random", "global_best", "linucb", "oracle"]}
    curves = []
    for run in range(a.runs):
        rng = np.random.default_rng(run)
        D = n_s * (d + 1)
        A, b = np.eye(D), np.zeros(D)
        tot = {k: 0.0 for k in res}
        curve = []
        for t in range(a.rounds):
            s = rng.choice(n_s, p=p_seg)
            Ainv = np.linalg.inv(A) if t % 25 == 0 else Ainv
            theta = Ainv @ b
            C = np.stack([ctx(s, e) for e in range(n_e)])
            ucb = C @ theta + a.alpha * np.sqrt(np.einsum("ij,jk,ik->i", C, Ainv, C))
            picks = {"random": rng.integers(n_e), "global_best": best_global, "linucb": int(np.argmax(ucb)),
                     "oracle": int(np.argmax(U[s]))}
            for k, e in picks.items():
                tot[k] += p_like(s, e)
            e = picks["linucb"]
            r = float(rng.random() < p_like(s, e))
            x = C[e]
            A += np.outer(x, x)
            b += r * x
            if (t + 1) % 250 == 0:
                curve.append(tot["linucb"] / (t + 1))
        for k in res:
            res[k].append(tot[k] / a.rounds)
        curves.append(curve)
        # what did the router learn? final greedy choice per segment
    theta = np.linalg.inv(A) @ b
    learned = {seg_names[s]: engines[int(np.argmax([ctx(s, e) @ theta for e in range(n_e)]))] for s in range(n_s)}
    truth = {seg_names[s]: engines[int(np.argmax(U[s]))] for s in range(n_s)}
    out = {
        "mean_satisfaction": {k: float(np.mean(v)) for k, v in res.items()},
        "global_best_engine": engines[best_global],
        "per_segment_best_engine_true": truth,
        "per_segment_choice_learned_run_last": learned,
        "learning_curve_linucb_every_250": np.mean(curves, 0).round(4).tolist(),
        "note": "simulated users; demonstrates the routing pipeline, not real preferences",
    }
    os.makedirs(os.path.join(ROOT, "outputs", "routing"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "routing", "routing_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "learning_curve_linucb_every_250"}, indent=1))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = np.arange(1, len(out["learning_curve_linucb_every_250"]) + 1) * 250
        plt.figure(figsize=(6, 3.6))
        plt.plot(xs, out["learning_curve_linucb_every_250"], label="LinUCB (cumulative avg)")
        for k, c in [("random", "gray"), ("global_best", "tab:orange"), ("oracle", "tab:green")]:
            plt.axhline(out["mean_satisfaction"][k], ls="--", color=c, label=k)
        plt.xlabel("requests")
        plt.ylabel("simulated satisfaction rate")
        plt.title("Engine routing on simulated user segments")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT, "outputs", "routing", "routing_curve.png"), dpi=130)
    except Exception as ex:  # plotting is optional
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
