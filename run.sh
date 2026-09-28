#!/usr/bin/env bash
#
# run.sh — build the Day-11 SVM/360 fisheye demo on local CVAT (split by camera_id).
#
#   ./run.sh                 # download locked subsets + build all tasks + golden GT
#   ./run.sh --recreate      # delete existing Day11 tasks first, then rebuild
#   ./run.sh --only object   # build one family {object,freespace,lines,ignore}
#
# Datasets are NOT vendored (licenses forbid re-hosting). This downloads only the
# locked file subset in assets/picks/ into data/ at build time.
#
# Prereqs: local CVAT running + admin; Python deps (requirements.txt);
#          CVAT_* in .env/../cvat/.env; KAGGLE_API_TOKEN for WoodScape (B/C/D).
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${PY:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
  elif [ -x "$HOME/miniconda3/envs/ai-lab/bin/python" ]; then
    PY="$HOME/miniconda3/envs/ai-lab/bin/python"
  else
    PY="$(which python3)"
  fi
fi
[ -x "$PY" ] || { echo "python not found at $PY (set PY=/path/to/python)"; exit 1; }

echo ">> Task A: FishEye8K object subset + golden GT (HuggingFace, no token) ..."
"$PY" scripts/prepare_fisheye8k.py
echo ">> Tasks B/C/D: WoodScape subset + golden GT (Kaggle, needs KAGGLE_API_TOKEN) ..."
"$PY" scripts/prepare_woodscape.py
echo ">> creating CVAT tasks (split by camera_id) + a golden GT job for each ..."
"$PY" scripts/setup_cvat.py "$@"
echo ">> ALL DONE. Open the task URLs printed above."
