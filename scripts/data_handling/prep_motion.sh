#!/usr/bin/env bash
set -euo pipefail

# ---------------- user config ----------------
ISAACLAB_SETUP="${HOME}/IsaacLab/_isaac_sim/setup_conda_env.sh"
ENV_ISAACLAB="env_isaaclab"
ENV_GMR="gmr"

# External GMR repo script
GMR_REPO="${GMR_REPO:-${HOME}/GMR}"
SMPLX_TO_G1_SH="${GMR_REPO}/scripts/smplx_to_g1_csv.sh"
# --------------------------------------------

usage() {
  cat >&2 <<EOF
Usage:
  $0 --motion-npz <motion.npz> --out-dir <output_dir> [--leica-z <float>] [--output-name <name>] [--headless] [--keep-intermediate]

Behavior:
  - By default, intermediate files are created in a temp dir and deleted on exit.
  - With --keep-intermediate, intermediates are written to --out-dir and preserved.

Outputs (always in --out-dir):
  <stem>_rt2g1_rdy.csv
EOF
  exit 1
}

MOTION_NPZ=""
OUT_DIR=""
LEICA_Z="0.0"
OUTPUT_NAME=""
HEADLESS_FLAG=""
KEEP_INTERMEDIATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-npz) MOTION_NPZ="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --leica-z) LEICA_Z="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --headless) HEADLESS_FLAG="--headless"; shift 1 ;;
    --keep-intermediate) KEEP_INTERMEDIATE=1; shift 1 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "${MOTION_NPZ}" && -n "${OUT_DIR}" ]] || usage
[[ -f "${MOTION_NPZ}" ]] || { echo "Missing file: ${MOTION_NPZ}" >&2; exit 1; }

mkdir -p "${OUT_DIR}"

# Make input absolute
MOTION_NPZ="$(cd "$(dirname "${MOTION_NPZ}")" && pwd)/$(basename "${MOTION_NPZ}")"

# Derive stem from input file name
STEM="$(basename "${MOTION_NPZ}")"
STEM="${STEM%.npz}"

# Working directory for intermediates
if [[ "${KEEP_INTERMEDIATE}" -eq 1 ]]; then
  WORK_DIR="${OUT_DIR}"
else
  WORK_DIR="$(mktemp -d)"
  trap 'rm -rf "${WORK_DIR}"' EXIT
fi

PREPPED_NPZ="${WORK_DIR}/${STEM}_prepped.npz"
RT_CSV="${WORK_DIR}/${STEM}_rt2g1.csv"
RDY_CSV="${WORK_DIR}/${STEM}_rt2g1_rdy.csv"
FINAL_RDY_CSV="${OUT_DIR}/$(basename "${RDY_CSV}")"

# Default wandb collection name
if [[ -z "${OUTPUT_NAME}" ]]; then
  OUTPUT_NAME="${STEM}"
fi

# Validate external script
[[ -f "${SMPLX_TO_G1_SH}" ]] || { echo "Missing GMR script: ${SMPLX_TO_G1_SH}" >&2; exit 1; }
[[ -f "${SMPLX_TO_G1_SH}" ]] || {
  echo "Missing GMR script: ${SMPLX_TO_G1_SH}" >&2
  exit 1
}

echo "== Pipeline =="
echo "Input NPZ       : ${MOTION_NPZ}"
echo "Out dir         : ${OUT_DIR}"
echo "Work dir        : ${WORK_DIR}"
echo "Keep interm     : ${KEEP_INTERMEDIATE}"
echo "Prepped NPZ     : ${PREPPED_NPZ}"
echo "Retargeted CSV  : ${RT_CSV}"
echo "Postproc CSV    : ${RDY_CSV}"
echo "Final CSV       : ${FINAL_RDY_CSV}"
echo "W&B name        : ${OUTPUT_NAME}"
echo

# Helper: run a command in IsaacLab env with required sourcing, isolated in a subshell
run_in_isaaclab() {
  (
    # Ensure conda function exists in non-interactive shell
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_ISAACLAB}"

    # IsaacLab setup script assumes ZSH_VERSION may be unset; avoid 'set -u' crash.
    set +u
    source "${ISAACLAB_SETUP}"
    set -u

    "$@"
  )
}

# Helper: run a command in gmr env isolated in a subshell
run_in_gmr() {
  (
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_GMR}"
    "$@"
  )
}

# ---- 1) prepare_data.py (env_isaaclab + IsaacLab setup) ----
echo "== Step 1: prepare_data.py (isaaclab) =="
run_in_isaaclab python scripts/data_handling/prepare_data.py \
  --motion-file "${MOTION_NPZ}" \
  --output-path "${PREPPED_NPZ}"
[[ -f "${PREPPED_NPZ}" ]] || { echo "Expected file not found: ${PREPPED_NPZ}" >&2; exit 1; }
echo

# ---- 2) smplx_to_g1_csv.sh (gmr) ----
echo "== Step 2: smplx_to_g1_csv.sh (gmr) =="
run_in_gmr bash "${SMPLX_TO_G1_SH}" "${PREPPED_NPZ}" "${WORK_DIR}"
[[ -f "${RT_CSV}" ]] || {
  echo "Expected retargeted CSV not found: ${RT_CSV}" >&2
  echo "Check smplx_to_g1_csv.sh output directory/stem logic." >&2
  exit 1
}
echo

# ---- 3) post_retargeting.py (env_isaaclab + IsaacLab setup) ----
echo "== Step 3: post_retargeting.py (isaaclab) =="
run_in_isaaclab python scripts/data_handling/post_retargeting.py \
  --original-npz "${PREPPED_NPZ}" \
  --retargeted-csv "${RT_CSV}" \
  --output-csv "${RDY_CSV}" \
  --leica-z "${LEICA_Z}"
[[ -f "${RDY_CSV}" ]] || { echo "Expected file not found: ${RDY_CSV}" >&2; exit 1; }
echo

# ---- 4) csv_to_npz.py (env_isaaclab + IsaacLab setup) ----
# NOTE: this launches Isaac Sim; use --headless on servers/CI.
echo "== Step 4: csv_to_npz.py (isaaclab) =="
run_in_isaaclab python scripts/csv_to_npz.py \
  --input_file "${RDY_CSV}" \
  --input_fps 30 \
  --output_name "${OUTPUT_NAME}" \
  ${HEADLESS_FLAG}

# Ensure final output is in OUT_DIR even when intermediates are temp
if [[ "${KEEP_INTERMEDIATE}" -eq 0 ]]; then
  echo
  echo "Copying final outputs to ${OUT_DIR}"
  cp -v "${RDY_CSV}" "${FINAL_RDY_CSV}"
fi

echo
echo "Done."
echo "Final CSV: ${FINAL_RDY_CSV}"
