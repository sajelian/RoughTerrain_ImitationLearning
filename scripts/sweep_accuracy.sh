#!/bin/bash
set -e

SCRIPT="scripts/rsl_rl/train_mimic_mesh.py"
TASK="Tracking-Mesh-G1-v0"
NUM_ENVS=2048
SEEDS=(1 2 3)
MAX_ITERATIONS=4000

# 5 (registry_name, usd_name) pairs
declare -A PAIRS
PAIRS[0]="sajelian-eth-zurich-org/wandb-registry-Motions/stepping_stones|stepping_stones"
for fcs in "${REF_NOISE[@]}"; do
for fcs in "${REF_NOISE[@]}"; do

REF_NOISE=(0.0, 0.01, 0.05, 0.1, 0.25)

for fcs in "${REF_NOISE[@]}"; do
  for i in "${!PAIRS[@]}"; do
    IFS="|" read -r registry usd <<< "${PAIRS[$i]}"

    # runs differing only in seed are grouped together
    motion=$(basename "$registry")
    GROUP="${fcs}_${motion}"
    RUN_NAME_BASE="${fcs}_${motion}"

    for seed in "${SEEDS[@]}"; do
      export WANDB_RUN_GROUP="$GROUP"

      EXTRA_ARGS=()
      if [[ "$fcs" == "without_fcs" ]]; then
        EXTRA_ARGS=("env.rewards.foot_contact_schedule.weight=0.0")
      fi

      python "$SCRIPT" \
        --task "$TASK" \
        --num_envs $NUM_ENVS \
        --seed $seed \
        --max_iterations $MAX_ITERATIONS \
        --registry_name "$registry" \
        --usd_name "$usd" \
        --logger wandb \
        --log_project_name RT_IL \
        --run_name "${RUN_NAME_BASE}_seed${seed}" \
        --headless \
        "${EXTRA_ARGS[@]}"
    done
  done
done
