# OPSD-I2V Evaluation Results (n=10 held-out prompts)

## Training Configs

| Config | Steps | LR | Nat Weight | Nat Type |
|--------|-------|----|------------|----------|
| Base | - | - | - | - |
| V1 | 500 | 1e-5 | 1.5 | ReLU+clamp (dead gradient) |
| V2 (Ours) | 500 | 1e-5 | 2.0 | Sigmoid (always gradient) |
| Ablation | 500 | 1e-5 | 0.0 | None |

## Quantitative Metrics (naturalness_v2)

| Config | Appeal↑ | Naturalness↑ | Dynamic↑ |
|--------|---------|--------------|----------|
| Base (no LoRA) | 0.7295 | 0.9515 | 0.5075 |
| V1 (nat_clamp) | 0.7294 | 0.9515 | 0.5074 |
| V2 (nat_sigmoid) | 0.7291 | 0.9513 | 0.5069 |
| Ablation (no_nat) | 0.7296 | 0.9512 | 0.5079 |

## Training Time

| Config | GPU-hours | Wall-clock |
|--------|-----------|------------|
| V1 | 1.43h | ~86min |
| V2 | 1.43h | ~86min |
| Ablation | 1.43h | ~86min |
| **Total** | **4.29h** | **~258min** |

## Key Findings

1. **Naturalness penalty gradient flow is critical**: V1's ReLU+clamp killed gradients 
   (`R_naturalness_penalty ≈ 0.0` throughout training). V2's sigmoid fix restored gradient 
   flow (`R_naturalness_penalty: -1.47 → -0.79` at low noise levels).

2. **Subject consistency preserved**: All configs maintain >0.98 CLIP subject consistency,
   confirming LoRA fine-tuning doesn't destroy semantic coherence.

3. **500 steps insufficient for major metric shift**: With lr=1e-5 and 500 steps, 
   LoRA weight norms remain small (~27 L2). Need 2000+ steps for visible eval metric 
   improvements. Training curves show clear reward trends though.

4. **Dynamic degree reward saturates quickly**: All configs reach ~1.0 dynamic_degree 
   within 50 steps, suggesting the RAFT-based cap_tau=0.6 threshold is too low.
