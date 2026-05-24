#!/bin/bash
set -e

SCRIPT="scripts/rsl_rl/train_mimic_mesh.py"
TASK="Tracking-Mesh-G1-v0"
NUM_ENVS=8192
SEEDS=(2 3 4)
MAX_ITERATIONS=2000

REGISTRY="sajelian-eth-zurich-org/wandb-registry-Motions/stepping_stones"
USD_NAME="stepping_stones"

# Gaussian σ (metres) added to reference body positions each step.
# 0.0 = clean reference; higher values simulate terrain-alignment inaccuracy.
REF_NOISE_LEVELS=(0.0 0.01 0.02 0.05 0.1)

for seed in "${SEEDS[@]}"; do
  for noise in "${REF_NOISE_LEVELS[@]}"; do
    GROUP="ref_noise_${noise}"
    RUN_NAME_BASE="ref_noise_${noise}"

    export WANDB_RUN_GROUP="$GROUP"

    python "$SCRIPT" \
      --task "$TASK" \
      --num_envs $NUM_ENVS \
      --seed $seed \
      --max_iterations $MAX_ITERATIONS \
      --registry_name "$REGISTRY" \
      --usd_name "$USD_NAME" \
      --ref_noise "$noise" \
      --logger wandb \
      --log_project_name RT_IL \
      --run_name "${RUN_NAME_BASE}_seed${seed}" \
      --headless
  done
done
