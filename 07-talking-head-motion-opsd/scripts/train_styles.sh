#!/bin/bash
# Train three persona LoRAs from the same reward function, differing ONLY in
# reward weights and the OPSD step size (c_adv).
#
# Why these knobs:
#   amp    -> how much head motion is rewarded. OPSD saturates tanh(std_low/3),
#             which is what produced the 20x amplitude overshoot and the
#             uncanny-valley look. Lowering amp weight is the main fix.
#   jitter -> anti-jitter pressure. Higher = steadier, more trustworthy.
#   gaze   -> how much the eyes are allowed to wander.
#   c_adv  -> OPSD target radius; smaller = gentler update per step.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

DATA=data/audio_public/train
ROUNDS=${ROUNDS:-3000}
GPU=${GPU:-1}

train () {  # name  weights-json  c_adv  blink-target  jitter-cap  gpu
  local name="$1" w="$2" cadv="$3" bt="$4" jcap="$5" gpu="$6"
  echo "=== style: ${name} ==="
  mkdir -p "outputs/opsd_style_${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/opsd_train_motion_v3.py \
      --out "outputs/opsd_style_${name}" \
      --data-dir "${DATA}" \
      --rounds "${ROUNDS}" \
      --reward-step-size "${cadv}" \
      --blink-target-mean "${bt}" \
      --jitter-cap-deg "${jcap}" \
      --weights "${w}" \
      --seed 0
}

# ---------- 1. finance: rigorous, sincere, believable ----------
# restrained motion, very steady, natural blink rate, steady eye contact
train finance \
  '{"amp":0.35,"jitter":1.2,"couple":0.6,"blink":0.5,"gaze":0.2,"ood":1.0}' \
  1.5 0.10 0.15 "${GPU}"

# ---------- 2. real estate: gentle, trustworthy ----------
# moderate motion, calm, slightly warmer blink rate
train realestate \
  '{"amp":0.6,"jitter":0.8,"couple":0.5,"blink":0.5,"gaze":0.3,"ood":1.0}' \
  2.2 0.13 0.20 "${GPU}"

# ---------- 3. entertainment / short-video hook: exaggerated, energetic ----------
train hook \
  '{"amp":1.4,"jitter":0.4,"couple":0.7,"blink":0.6,"gaze":0.45,"ood":1.0}' \
  3.5 0.18 0.35 "${GPU}"

echo "[ok] three style LoRAs trained"
