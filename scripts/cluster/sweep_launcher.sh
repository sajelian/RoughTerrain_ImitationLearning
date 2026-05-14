#!/bin/bash
# Run from the repo root on the login node:
#   bash scripts/cluster/sweep_launcher.sh
set -e

SBATCH="scripts/cluster/train.sbatch"
SCRIPT="scripts/rsl_rl/train_mimic_mesh.py"
TASK="Tracking-Mesh-G1-v0"
NUM_ENVS=16384
MAX_ITERATIONS=2000
SEEDS=(1 2 3 4 5)

# usd must be copied to the cluster, motions can be fetched from the registry
# take the atomic motions only (newest format and footlock given)
declare -A PAIRS
PAIRS[0]="sajelian-eth-zurich-org/wandb-registry-Motions/walking|flat"
PAIRS[1]="sajelian-eth-zurich-org/wandb-registry-Motions/box_up|box"
PAIRS[2]="sajelian-eth-zurich-org/wandb-registry-Motions/beam_straight|beam"
PAIRS[3]="sajelian-eth-zurich-org/wandb-registry-Motions/gap|gap"
PAIRS[4]="sajelian-eth-zurich-org/wandb-registry-Motions/stepping_stones|stepping_stones"

FOOT_CONTACT_OPTIONS=("with_fcs" "without_fcs")

for fcs in "${FOOT_CONTACT_OPTIONS[@]}"; do
  for i in "${!PAIRS[@]}"; do
    IFS="|" read -r registry usd <<< "${PAIRS[$i]}"

    GROUP="${fcs}_pair${i}_${usd}"
    RUN_NAME_BASE="${fcs}_pair${i}"

    EXTRA_ARGS=()
    if [[ "$fcs" == "without_fcs" ]]; then
      EXTRA_ARGS=("env.rewards.foot_contact_schedule.weight=0.0")
    fi

    for seed in "${SEEDS[@]}"; do
      export WANDB_GROUP="$GROUP"

      sbatch --export=ALL "$SBATCH" \
        "$SCRIPT" \
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

      echo "Submitted: $fcs | pair $i ($usd) | seed $seed"
    done
  done
done
