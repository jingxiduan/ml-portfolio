# Gate Log — 08-video-opsd-i2v

| Time | Gate | Key Values | Result |
|------|------|------------|--------|
| 2026-10-30 20:20 | T1 probe P1 (lr=1e-4, c_adv=5) 200 steps | shift_ratio=0.56%, param_grad=1.44e-05, lora_B=0.2765, reward delta=+0.043, subj=0.9837, guard=no trip | PARTIAL (shift ok, reward delta +0.043 < +0.3) |
| 2026-10-30 20:20 | T1 probe P2 (lr=1e-5, c_adv=40) 200 steps | shift_ratio=4.51%, param_grad=1.48e-04 (10x P1), lora_B=0.0769, reward delta=+0.041, subj=0.9837, guard=no trip | PARTIAL (shift in range, reward delta +0.041 < +0.3) |
| 2026-10-30 20:20 | T1 root cause | c_adv dominant: param_grad_norm 10x larger at c_adv=40 vs 5; confirms A2-1 (target offset too small) | CONFIRMED |
| 2026-11-02 | 500-step V2 held-out eval | appeal 0.7291 vs base 0.7295; naturalness 0.9513 vs 0.9515; dynamic 0.5069 vs 0.5075 | FAIL — no measurable held-out gain |
| 2026-11-02 | NoNat 2,000-step stability control | completed step 2,000; first/final 20-step `loss_plus` 0.00660/0.00640; final LoRA-B norm 0.9875 | PASS numerical completion only; quality not established |
| 2026-11-02 | V2 2,000-step attempt | stopped at step 306; rolling `loss_plus` reached 10.95× baseline; LoRA-B norm 2.13 | FAIL — guard stop |
| 2026-11-03 | V3 unit preflight | anchored objective, legacy objective compatibility, and tanh extreme-pixel gradient tests | PASS — 3 tests |
| 2026-11-03 | V3 100-step training preflight | completed in 1,219s; no guard trip; `loss_plus` first/final 20-step mean 0.00660/0.00681; max shift 0.1378; checkpoints 50/100 | PASS stability preflight |
