# V4 final report

- Recorded steps: **2000/2000**
- Guard stop: **no**
- `loss_plus`, first→last 100: **0.00671 → 0.02984**
- Raw parameter-gradient norm, first→last 100: **0.000070 → 0.003071**
- Effective post-clip norm, last 100: **0.000371**

## Paired held-out result

| Metric | Base | V4 | V4−Base mean [95% CI] |
|---|---:|---:|---:|
| Appeal | 0.7146 | 0.7104 | -0.0042 [-0.0062, -0.0024] |
| Naturalness | 0.9434 | 0.9429 | -0.0005 [-0.0031, +0.0013] |
| Dynamic degree | 0.4858 | 0.4779 | -0.0079 [-0.0115, -0.0048] |

Metrics with a favorable 95% interval excluding zero: **none**.
