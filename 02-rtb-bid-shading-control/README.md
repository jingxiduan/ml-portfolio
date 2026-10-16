# RTB Win-Rate Modeling + PID/MPC Bid Shading Feedback Control Simulation

Reproduces the core technical pipeline of "win-rate/bid-landscape modeling + framing Bid Shading as a feedback control problem" using a public CTR dataset plus an auction simulation environment: pCTR drives the bid -> win-rate model estimates win probability -> greedy search over candidate bids -> PID / MPC controllers dynamically adjust the bid coefficient based on ROI deviation.

> Note: real RTB bidding logs (bid price, winning price, clearing price) involve advertisers' commercial secrets, so there is essentially no public dataset that directly includes all these fields. This project therefore follows the standard academic approach for "bid landscape prediction" (e.g. Wu et al., KDD'15, *Predicting Winning Price in RTB with Censored Data*): **real features come from the public [Avazu CTR dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2)** (official LIBSVM mirror, ~40M mobile ad click logs, 400K used here) to train a real pCTR model; **market competing prices (winning prices) are parameterized via a log-normal distribution**. This simulation assumption is stated explicitly in the README, and all strategy comparisons share the same set of random market prices to ensure a fair, internally consistent comparison.

## Key Techniques

- Omni-channel win-rate model `omni_pwrm`: jointly estimates win probability from pCTR, bid price, and traffic competition features
- Greedy candidate-bid search: scans candidate bid ratios on top of the win-rate model to maximize expected profit
- Bid Shading as a feedback control problem: implements both PID and a simplified MPC controller, dynamically adjusting the bid coefficient based on ROI/spend deviation
- Staged ramp-up evaluation: simulates incremental business metric changes when scaling traffic from 5% to 20%

## Method & Code Structure

```
02-rtb-bid-shading-control/
├── src/
│   ├── download_data.py    # download the public Avazu dataset (official LIBSVM mirror, no login)
│   ├── prepare_pctr.py     # train a pCTR model on real click data (LogisticRegression, AUC~0.78)
│   ├── auction_env.py      # second-price auction simulation environment (log-normal market price)
│   ├── winrate_model.py    # win-rate model (rolling online logistic regression) + greedy candidate-bid search
│   ├── controllers.py      # PIDShadingController / MPCShadingController
│   └── run_experiment.py   # end-to-end experiment: 3-strategy windowed online comparison + staged ramp-up estimate
└── outputs/                 # json metrics and curve plots produced by the experiment
```

## Run

```bash
pip install -r requirements.txt
python src/download_data.py      # download and slice the Avazu data subset
python src/prepare_pctr.py       # train the pCTR model, produce the simulation traffic pool sim_pool.npz
python src/run_experiment.py     # run Fixed / PID / MPC strategy comparison + staged ramp-up experiment
```

## Results

**pCTR model** (400K real Avazu click records, 70/15/15 split): validation AUC = **0.7796**, LogLoss = 0.2734.

**Three Bid Shading strategies compared** (60K simulated traffic pool, 60-window rolling online comparison, target ROI=1.4):

| Strategy | Total GMV | Total Spend | Profit | Overall ROI | Avg Win Rate |
|---|---|---|---|---|---|
| Fixed (coefficient 0.8) | 5,874 | 3,196 | 2,678 | 1.84 | 10.3% |
| PID feedback control | 16,054 | 10,497 | **5,557** | 1.53 | 27.1% |
| MPC feedback control | 15,459 | 10,021 | 5,438 | 1.54 | 25.8% |

The fixed-coefficient strategy is overly conservative: a constant bid coefficient results in only a 10% win rate, leaving a lot of high-value traffic unwon; PID/MPC dynamically raise the bid coefficient based on ROI deviation, keeping overall ROI stable around the target (1.4) while **boosting win rate by ~2.5x and profit by ~1.0x**. MPC's lookahead prediction gives smoother, less oscillating coefficient adjustments than PID (ROI closer to target with lower spend).

**Staged ramp-up simulation** (PID strategy scaled from 5% to 20% of traffic, relative to the full-Fixed baseline):

| Ramp Ratio | GMV Uplift | Spend Change |
|---|---|---|
| 5% | +8.7% | +11.4% |
| 10% | +17.3% | +22.8% |
| 15% | +26.0% | +34.3% |
| 20% | +34.7% | +45.7% |

Curve plot: `outputs/bid_shading_curves.png` (left: per-window ROI convergence for each strategy; right: bid coefficient trajectory over time). Full data: `outputs/strategy_comparison.json`, `outputs/staged_ramp.json`, `outputs/window_logs.json`.
