#!/usr/bin/env bash
set -euo pipefail

# ---------------- user config ----------------
ISAACLAB_SETUP="${HOME}/IsaacLab/_isaac_sim/setup_conda_env.sh"
ENV_ISAACLAB="env_isaaclab"
ENV_GMR="gmr"

# Path to smplx_to_g1_csv.sh (your script #2)
GMR_REPO="${GMR_REPO:-${HOME}/IsaacLab_DMC/GMR}"
SMPLX_TO_G1_SH="${GMR_REPO}/scripts/smplx_to_g1_csv.sh"
# --------------------------------------------

usage() {
  cat >&2 <<EOF
Usage:
  $0 --motion-npz <motion.npz> --out-dir <output_dir> [--leica-z <float>] [--output-name <name>] [--headless]

Outputs:
  <out-dir>/<stem>_prepped.npz
  <out-dir>/<stem>_rt2g1.csv
  <out-dir>/<stem>_rt2g1_rdy.csv
  (csv_to_npz logs /tmp/motion.npz to wandb under --output-name)
EOF
  exit 1
}

MOTION_NPZ=""
OUT_DIR=""
LEICA_Z="0.0"
OUTPUT_NAME=""
HEADLESS_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-npz) MOTION_NPZ="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --leica-z) LEICA_Z="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --headless) HEADLESS_FLAG="--headless"; shift 1 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "${MOTION_NPZ}" && -n "${OUT_DIR}" ]] || usage
[[ -f "${MOTION_NPZ}" ]] || { echo "Missing file: ${MOTION_NPZ}" >&2; exit 1; }
[[ -f "${SMPLX_TO_G1_SH}" ]] || { echo "Missing smplx_to_g1_csv.sh at: ${SMPLX_TO_G1_SH}" >&2; exit 1; }

mkdir -p "${OUT_DIR}"

# Make everything absolute + derive stems
MOTION_NPZ="$(cd "$(dirname "${MOTION_NPZ}")" && pwd)/$(basename "${MOTION_NPZ}")"
STEM="$(basename "${MOTION_NPZ}")"
STEM="${STEM%.npz}"

PREPPED_NPZ="${OUT_DIR}/${STEM}_prepped.npz"
RT_CSV="${OUT_DIR}/${STEM}_rt2g1.csv"
RDY_CSV="${OUT_DIR}/${STEM}_rt2g1_rdy.csv"

# If you want the wandb collection name to default nicely:
if [[ -z "${OUTPUT_NAME}" ]]; then
  OUTPUT_NAME="${STEM}"
fi

echo "== Pipeline =="
echo "Input NPZ : ${MOTION_NPZ}"
echo "Out dir   : ${OUT_DIR}"
echo "Prepped   : ${PREPPED_NPZ}"
echo "Retarget  : ${RT_CSV}"
echo "Postproc  : ${RDY_CSV}"
echo "W&B name  : ${OUTPUT_NAME}"
echo

# Helper: run a command in IsaacLab env with required sourcing, isolated in a subshell
run_in_isaaclab() {
  (
    # Ensure conda function exists in non-interactive shell
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_ISAACLAB}"

    # IsaacLab/IsaacSim env setup required by you
    # (this often sets PYTHONPATH/LD_LIBRARY_PATH/OMNI vars)
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
echo

# ---- 2) smplx_to_g1_csv.sh (gmr) ----
echo "== Step 2: smplx_to_g1_csv.sh (gmr) =="
run_in_gmr bash "${SMPLX_TO_G1_SH}" "${PREPPED_NPZ}" "${OUT_DIR}"
# smplx_to_g1_csv.sh writes <outdir>/<stem>_rt2g1.csv (based on your script logic)
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
echo

# ---- 4) csv_to_npz.py (env_isaaclab + IsaacLab setup) ----
# NOTE: this launches Isaac Sim; you might want --headless on servers/CI.
echo "== Step 4: csv_to_npz.py (isaaclab) =="
run_in_isaaclab python scripts/csv_to_npz.py \
  --input_file "${RDY_CSV}" \
  --input_fps 30 \
  --output_name "${OUTPUT_NAME}" \
  ${HEADLESS_FLAG}

echo "Done."
