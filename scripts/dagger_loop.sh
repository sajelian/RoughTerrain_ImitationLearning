#!/usr/bin/env bash
# DAgger iteration loop.
#
# Each round:
#   1) Roll out the current FQL student in the env, log states it visits with
#      PPO's correct actions as labels  ->  data/fql_rollouts/dagger_round_N.pt
#   2) Retrain FQL on the UNION of all prior datasets (the original PPO-driven
#      data + every dagger_round_*.pt collected so far)
#   3) Advance CURRENT_FQL to the latest checkpoint, ready for the next round
#
# After each round you should manually verify quality (e.g. compare_ppo_fql.py
# or just deploy in the play env). When you're satisfied, break out of the loop
# by setting MAX_ROUNDS appropriately, or just Ctrl+C.
#
# Required envvars (set below or override on the command line):
#   PPO_WANDB_PATH    - wandb run path for the velocity PPO teacher
#   INITIAL_FQL_CKPT  - the FQL checkpoint to start the loop from
#                       (the one trained on PPO-driven rollouts, with some
#                       compounding-error drift remaining)
#
# Optional:
#   MAX_ROUNDS              - default 3
#   DAGGER_TRANSITIONS      - default 1_000_000 per round
#   TRAIN_ITERATIONS        - default 300_000 per round
#   ALPHA                   - FQL BC coefficient, default 3.0
#   DATA_DIR                - default data/fql_rollouts
#   LOG_DIR                 - default logs/fql/dagger
#   TASK                    - default Velocity-G1-v0 (env for DAgger collection)
#   REGISTRY_NAME           - W&B motion artifact (required for tracking-env tasks,
#                             e.g. sajelian-eth-zurich-org/wandb-registry-Motions/box_up)
#   USD_NAME                - obstacle USD name (for tracking-env tasks, e.g. flat_box_convex)
#   TEACHER_CFG             - 'ppo' (default, tracking PPO) or 'ppo_velocity'
#
# Usage:
#   PPO_WANDB_PATH=user/project/runid \
#   INITIAL_FQL_CKPT=logs/fql/base/agent_500000.pt \
#       bash scripts/dagger_loop.sh

set -euo pipefail

# ---- Config ----
MAX_ROUNDS="${MAX_ROUNDS:-3}"
DAGGER_TRANSITIONS="${DAGGER_TRANSITIONS:-1000000}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-300000}"
ALPHA="${ALPHA:-3.0}"
DATA_DIR="${DATA_DIR:-data/fql_rollouts}"
LOG_DIR="${LOG_DIR:-logs/fql/dagger}"
TASK="${TASK:-Velocity-G1-v0}"
NUM_ENVS="${NUM_ENVS:-2048}"
# RAM cap for the FQL replay buffer (proportional subsampling across all
# datasets if exceeded). Rough budget: 4 GB per 1M transitions @ obs_dim≈988.
# Default 10M ≈ 40 GB — adjust to your machine.
MAX_TOTAL_TRANSITIONS="${MAX_TOTAL_TRANSITIONS:-10000000}"

# Required
if [[ -z "${PPO_WANDB_PATH:-}" ]]; then
    echo "ERROR: PPO_WANDB_PATH not set. Set it to the velocity PPO's wandb run path." >&2
    exit 1
fi
if [[ -z "${INITIAL_FQL_CKPT:-}" ]]; then
    echo "ERROR: INITIAL_FQL_CKPT not set. Point it at the FQL checkpoint you want to iterate from." >&2
    exit 1
fi
if [[ ! -f "$INITIAL_FQL_CKPT" ]]; then
    echo "ERROR: INITIAL_FQL_CKPT does not exist: $INITIAL_FQL_CKPT" >&2
    exit 1
fi

CURRENT_FQL="$INITIAL_FQL_CKPT"

mkdir -p "$DATA_DIR" "$LOG_DIR"

# Optional: start a fresh log of which datasets were used at each round
ROUND_LOG="$LOG_DIR/_round_log.txt"
echo "DAgger loop started $(date)" > "$ROUND_LOG"
echo "  PPO:  $PPO_WANDB_PATH" >> "$ROUND_LOG"
echo "  Init: $INITIAL_FQL_CKPT" >> "$ROUND_LOG"

for ROUND in $(seq 1 "$MAX_ROUNDS"); do
    echo
    echo "============================================================"
    echo " DAgger ROUND $ROUND / $MAX_ROUNDS"
    echo " Current FQL: $CURRENT_FQL"
    echo "============================================================"
    echo

    # ---- 1) Collect DAgger augmentation ----
    DAGGER_NAME="dagger_round_${ROUND}"
    DAGGER_FILE="$DATA_DIR/${DAGGER_NAME}.pt"

    echo ">>> Step 1/2: Collecting DAgger data..."
    # Optional args for tracking envs (motion clip + obstacle USD)
    EXTRA_COLLECT_ARGS=()
    if [[ -n "${REGISTRY_NAME:-}" ]]; then
        EXTRA_COLLECT_ARGS+=(--registry_name "$REGISTRY_NAME")
    fi
    if [[ -n "${USD_NAME:-}" ]]; then
        EXTRA_COLLECT_ARGS+=(--usd_name "$USD_NAME")
    fi
    if [[ -n "${TEACHER_CFG:-}" ]]; then
        EXTRA_COLLECT_ARGS+=(--teacher_cfg "$TEACHER_CFG")
    fi

    python scripts/rsl_rl/collect_dagger_dataset.py \
        --task "$TASK" \
        --wandb_path "$PPO_WANDB_PATH" \
        --fql_checkpoint "$CURRENT_FQL" \
        --output_dir "$DATA_DIR" \
        --output_name "$DAGGER_NAME" \
        --num_envs "$NUM_ENVS" \
        --num_transitions "$DAGGER_TRANSITIONS" \
        "${EXTRA_COLLECT_ARGS[@]}" \
        --headless

    if [[ ! -f "$DAGGER_FILE" ]]; then
        echo "ERROR: DAgger collection produced no file at $DAGGER_FILE" >&2
        exit 1
    fi
    echo "  -> wrote $DAGGER_FILE"

    # ---- 2) Retrain FQL on union of everything in DATA_DIR ----
    ROUND_OUT="$LOG_DIR/round_${ROUND}"
    mkdir -p "$ROUND_OUT"

    echo
    echo ">>> Step 2/2: Retraining FQL on union of all datasets..."
    ALL_DATA=("$DATA_DIR"/*.pt)
    echo "  Datasets in union (${#ALL_DATA[@]} files):"
    for f in "${ALL_DATA[@]}"; do echo "    - $f"; done

    python scripts/rsl_rl/train_fql.py \
        --datasets "${ALL_DATA[@]}" \
        --alpha "$ALPHA" \
        --num_iterations "$TRAIN_ITERATIONS" \
        --max_total_transitions "$MAX_TOTAL_TRANSITIONS" \
        --output_dir "$ROUND_OUT" \
        --wandb \
        --wandb_name "dagger_round_${ROUND}"

    # Newest checkpoint becomes CURRENT_FQL for the next round
    NEW_CKPT="$ROUND_OUT/agent_${TRAIN_ITERATIONS}.pt"
    if [[ ! -f "$NEW_CKPT" ]]; then
        echo "ERROR: training did not produce expected checkpoint $NEW_CKPT" >&2
        exit 1
    fi
    CURRENT_FQL="$NEW_CKPT"

    # Log this round
    {
        echo "Round $ROUND ($(date)):"
        echo "  Dagger file: $DAGGER_FILE"
        echo "  New ckpt:    $CURRENT_FQL"
    } >> "$ROUND_LOG"

    echo
    echo ">>> Round $ROUND complete. New checkpoint: $CURRENT_FQL"
    echo ">>> To verify before continuing, in another shell:"
    echo "    python scripts/rsl_rl/compare_ppo_fql.py \\"
    echo "        --task Velocity-G1-Play-v0 \\"
    echo "        --wandb_path $PPO_WANDB_PATH \\"
    echo "        --fql_checkpoint $CURRENT_FQL \\"
    echo "        --drive_with fql --num_steps 500"
    echo
done

echo
echo "============================================================"
echo " DAgger loop done. Final FQL checkpoint: $CURRENT_FQL"
echo "============================================================"
echo "Round log: $ROUND_LOG"
