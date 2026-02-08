#!/bin/bash

echo "[INFO] Running inside Isaac Lab container"
echo "[INFO] Isaac Sim: $(cat /isaac-sim/VERSION 2>/dev/null || echo 'unknown')"

# Ensure PATH includes isaac-sim python
export PATH="/isaac-sim/kit/python/bin:/isaac-sim:/usr/local/bin:/usr/bin:/bin:$PATH"
export ISAACLAB_PATH=/workspace/isaaclab

# Install RT_IL (IsaacLab extensions are pre-installed in the image)
/isaac-sim/python.sh -m pip install --editable /workspace/RT_IL/source/RT_IL 2>&1

echo "[INFO] Starting training..."
cd /workspace/RT_IL
/isaac-sim/python.sh "$@" 2>&1
EXIT_CODE=$?
echo "[INFO] Training exited with code: $EXIT_CODE"
exit $EXIT_CODE
