# Gate Log — 08-video-opsd-i2v

| Time | Gate | Key Values | Result |
|------|------|------------|--------|
| 2026-10-30 20:20 | T1 probe P1 (lr=1e-4, c_adv=5) 200 steps | shift_ratio=0.56%, param_grad=1.44e-05, lora_B=0.2765, reward delta=+0.043, subj=0.9837, guard=no trip | PARTIAL (shift ok, reward delta +0.043 < +0.3) |
| 2026-10-30 20:20 | T1 probe P2 (lr=1e-5, c_adv=40) 200 steps | shift_ratio=4.51%, param_grad=1.48e-04 (10x P1), lora_B=0.0769, reward delta=+0.041, subj=0.9837, guard=no trip | PARTIAL (shift in range, reward delta +0.041 < +0.3) |
| 2026-10-30 20:20 | T1 root cause | c_adv dominant: param_grad_norm 10x larger at c_adv=40 vs 5; confirms A2-1 (target offset too small) | CONFIRMED |
