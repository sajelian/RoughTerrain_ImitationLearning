#!/usr/bin/env bash
set -euo pipefail

# ---------------- user config ----------------
ISAACLAB_SETUP="${HOME}/IsaacLab/_isaac_sim/setup_conda_env.sh"
ENV_ISAACLAB="env_isaaclab"
ENV_GMR="gmr_custom"
ENV_MESH="mesh"

# External GMR repo script
GMR_REPO="${GMR_REPO:-${HOME}/GMR}"
SMPLX_TO_G1_SH="${GMR_REPO}/scripts/smplx_to_g1_csv.sh"

# Mesh conversion tool (IsaacLab)
CONVERT_MESH_PY="${HOME}/IsaacLab/scripts/tools/convert_mesh.py"
# --------------------------------------------

usage() {
  cat >&2 <<EOF
Usage:
  $0 --motion-npz <motion.npz> --out-dir <output_dir>
     [--leica-z <float>] [--output-name <name>] [--headless] [--keep-intermediate]
     [--mesh <mesh.obj>] [--mesh-out-dir <dir>] [--mesh-cut-script <path>] [--skip-mesh]

Behavior:
  - By default, intermediate files are created in a temp dir and deleted on exit.
  - With --keep-intermediate, intermediates are written to --out-dir and preserved.

Trajectory output (always in --out-dir):
  <stem>_rt2g1_rdy.csv

Mesh outputs (if --mesh is provided and not --skip-mesh), written to --mesh-out-dir (defaults to --out-dir):
  <stem>_<meshname>_cropped.obj
  <stem>_<meshname>_cropped/<stem>_<meshname>_cropped.usd
EOF
  exit 1
}

MOTION_NPZ=""
OUT_DIR="trajectories/mesh"
LEICA_Z="0.0"
OUTPUT_NAME=""
HEADLESS_FLAG=""
KEEP_INTERMEDIATE=0

MESH_OBJ=""
MESH_OUT_DIR="${HOME}/RT_IL/source/RT_IL/RT_IL/assets/usd"
MESH_CUT_SCRIPT="${HOME}/RT_IL/scripts/terrain/mesh_traj_cut.py"
SKIP_MESH=0
    
while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-npz) MOTION_NPZ="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --leica-z) LEICA_Z="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --headless) HEADLESS_FLAG="--headless"; shift 1 ;;
    --keep-intermediate) KEEP_INTERMEDIATE=1; shift 1 ;;
    --mesh) MESH_OBJ="$2"; shift 2 ;;
    --mesh-out-dir) MESH_OUT_DIR="$2"; shift 2 ;;
    --mesh-cut-script) MESH_CUT_SCRIPT="$2"; shift 2 ;;
    --skip-mesh) SKIP_MESH=1; shift 1 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "${MOTION_NPZ}" && -n "${OUT_DIR}" ]] || usage
[[ -f "${MOTION_NPZ}" ]] || { echo "Missing file: ${MOTION_NPZ}" >&2; exit 1; }

mkdir -p "${OUT_DIR}"

# Make input absolute
MOTION_NPZ="$(cd "$(dirname "${MOTION_NPZ}")" && pwd)/$(basename "${MOTION_NPZ}")"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

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

# Helper: run a command in mesh env isolated in a subshell
run_in_mesh() {
  (
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_MESH}"
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
echo "== Step 4: csv_to_npz.py (isaaclab) =="
run_in_isaaclab python scripts/csv_to_npz.py \
  --input_file "${RDY_CSV}" \
  --input_fps 30 \
  --output_name "${OUTPUT_NAME}" \
  ${HEADLESS_FLAG}

# Ensure final trajectory output is in OUT_DIR even when intermediates are temp
if [[ "${KEEP_INTERMEDIATE}" -eq 0 ]]; then
  echo
  echo "Copying final trajectory output to ${OUT_DIR}"
  cp -v "${RDY_CSV}" "${FINAL_RDY_CSV}"
else
  # If intermediates are kept in OUT_DIR, FINAL_RDY_CSV == RDY_CSV only if WORK_DIR==OUT_DIR; still print final path.
  FINAL_RDY_CSV="${OUT_DIR}/$(basename "${RDY_CSV}")"
fi

# ---- Optional mesh pipeline (mesh env -> cropped OBJ, then isaaclab -> USD) ----
if [[ "${SKIP_MESH}" -eq 0 && -n "${MESH_OBJ}" ]]; then
  [[ -f "${MESH_OBJ}" ]] || { echo "Missing mesh file: ${MESH_OBJ}" >&2; exit 1; }
  [[ -f "${MESH_CUT_SCRIPT}" ]] || { echo "Missing mesh cut script: ${MESH_CUT_SCRIPT}" >&2; exit 1; }
  [[ -f "${CONVERT_MESH_PY}" ]] || { echo "Missing convert_mesh.py: ${CONVERT_MESH_PY}" >&2; exit 1; }

  if [[ -z "${MESH_OUT_DIR}" ]]; then
    MESH_OUT_DIR="${OUT_DIR}"
  fi
  mkdir -p "${MESH_OUT_DIR}"
  MESH_OUT_DIR="$(cd "${MESH_OUT_DIR}" && pwd)"
  MESH_OBJ="$(cd "$(dirname "${MESH_OBJ}")" && pwd)/$(basename "${MESH_OBJ}")"

  motion_name="${STEM}"
  terrain_name="$(basename "${MESH_OBJ}")"; terrain_name="${terrain_name%.*}"

  # Where to write cropped obj:
  # - if keep-intermediate, keep alongside mesh outputs
  # - else, crop in WORK_DIR then copy to MESH_OUT_DIR at end
  if [[ "${KEEP_INTERMEDIATE}" -eq 1 ]]; then
    CROPPED_OBJ="${WORK_DIR}/${motion_name}_${terrain_name}_cropped.obj"
  else
    CROPPED_OBJ="${WORK_DIR}/${motion_name}_${terrain_name}_cropped.obj"
  fi

  echo
  echo "== Step 5: mesh_traj_cut.py (mesh env) =="
  echo "Trajectory CSV : ${FINAL_RDY_CSV}"
  echo "Mesh OBJ       : ${MESH_OBJ}"
  echo "Cropped OBJ    : ${CROPPED_OBJ}"
  run_in_mesh python "${MESH_CUT_SCRIPT}" \
    --trajectory "${FINAL_RDY_CSV}" \
    --mesh "${MESH_OBJ}" \
    --output "${CROPPED_OBJ}"
  [[ -f "${CROPPED_OBJ}" ]] || { echo "Expected cropped OBJ not found: ${CROPPED_OBJ}" >&2; exit 1; }

  # If intermediates are not kept, copy cropped obj to mesh out dir now
  if [[ "${KEEP_INTERMEDIATE}" -eq 0 ]]; then
    echo "Copying cropped OBJ to ${MESH_OUT_DIR}"
    cp -v "${CROPPED_OBJ}" "${MESH_OUT_DIR}/"
    CROPPED_OBJ="${MESH_OUT_DIR}/$(basename "${CROPPED_OBJ}")"
  fi

  cropped_stem="$(basename "${CROPPED_OBJ}")"; cropped_stem="${cropped_stem%.*}"
  USD_SUBDIR="${MESH_OUT_DIR}/${cropped_stem}"
  USD_FILE="${USD_SUBDIR}/${cropped_stem}.usd"
  mkdir -p "${USD_SUBDIR}"

  echo
  echo "== Step 6: convert_mesh.py (isaaclab) =="
  echo "USD output     : ${USD_FILE}"
  run_in_isaaclab python "${CONVERT_MESH_PY}" \
    "${CROPPED_OBJ}" \
    "${USD_FILE}" \
    --make-instanceable \
    --collision-approximation sdf \
    ${HEADLESS_FLAG}

  [[ -f "${USD_FILE}" ]] || { echo "Expected USD not found: ${USD_FILE}" >&2; exit 1; }
  echo "Mesh USD: ${USD_FILE}"

  # Delete cropped OBJ if intermediates are not kept
  if [[ "${KEEP_INTERMEDIATE}" -eq 0 ]]; then
    echo "Removing intermediate cropped OBJ: ${CROPPED_OBJ}"
    rm -f "${CROPPED_OBJ}"
  fi
  
fi

echo
echo "Done."
echo "Successfully completed data preparation."
