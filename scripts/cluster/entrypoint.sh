#!/bin/bash

echo "[INFO] Running inside Isaac Lab container"
echo "[INFO] Isaac Sim: $(cat /isaac-sim/VERSION 2>/dev/null || echo 'unknown')"

# Ensure PATH includes isaac-sim python
export PATH="/isaac-sim/kit/python/bin:/isaac-sim:/usr/local/bin:/usr/bin:/bin:$PATH"
export ISAACLAB_PATH=/workspace/isaaclab

# Fix URDF extension version mismatch (container has 2.4.30, code expects 2.4.31)
# 1. Downgrade version string
# 2. Remove set_merge_fixed_ignore_inertia call (not available in 2.4.30)
sed -i -e 's/urdf-2.4.31/urdf-2.4.30/g' \
       -e '/set_merge_fixed_ignore_inertia/d' \
    /workspace/isaaclab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py 2>/dev/null || true

# Install RT_IL (IsaacLab extensions are pre-installed in the image)
/isaac-sim/python.sh -m pip install --editable /workspace/RT_IL/source/RT_IL 2>&1

echo "[INFO] Starting training..."
cd /workspace/RT_IL
/isaac-sim/python.sh "$@" 2>&1
EXIT_CODE=$?
echo "[INFO] Training exited with code: $EXIT_CODE"
exit $EXIT_CODE
