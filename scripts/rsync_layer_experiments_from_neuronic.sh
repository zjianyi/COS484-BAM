#!/usr/bin/env bash
# Pull Neuronic COS484-BAM metrics artifacts listed below into metrics/layer-experiments/
# (metrics/ is gitignored).
#
# Remote layout (Neuronic):
#   ${REMOTE_REPO}/metrics/
#     fewshot_babilong/
#     logs/
#     metrics/
#     interval_k4_layer2.json
#     interval_k4_layer2.cells.jsonl
#     loss_k4_layer6.json
#     loss_k4_layer6.cells.jsonl
#     loss_k4_layer16.json
#     loss_k4_layer16.cells.jsonl
#
# Run on your laptop:
#   bash scripts/rsync_layer_experiments_from_neuronic.sh
#
# Override:
#   REMOTE_HOST=user@host REMOTE_REPO=/path/to/COS484-BAM bash scripts/rsync_layer_experiments_from_neuronic.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REPO_ROOT}/metrics/layer-experiments"

REMOTE_HOST="${REMOTE_HOST:-bw7520@neuronic.cs.princeton.edu}"
REMOTE_REPO="${REMOTE_REPO:-/u/bw7520/COS484-BAM}"
REMOTE_METRICS="${REMOTE_REPO}/metrics"

mkdir -p "${DEST}"

echo "Remote metrics dir: ${REMOTE_HOST}:${REMOTE_METRICS}/"
echo "Local dest:        ${DEST}/"
echo

sync_dir() {
  local name="$1"
  echo "[dir] ${name}/"
  rsync -avz --progress "${REMOTE_HOST}:${REMOTE_METRICS}/${name}/" "${DEST}/${name}/"
}

sync_file() {
  local name="$1"
  echo "[file] ${name}"
  rsync -avz --progress "${REMOTE_HOST}:${REMOTE_METRICS}/${name}" "${DEST}/"
}

# Subdirectories under remote metrics/
sync_dir fewshot_babilong
sync_dir logs
sync_dir metrics

# Layer-run JSON + cells (train_babilong_ablation inline eval + optional checkpoint eval artifacts)
sync_file interval_k4_layer2.json
sync_file interval_k4_layer2.cells.jsonl
sync_file loss_k4_layer6.json
sync_file loss_k4_layer6.cells.jsonl
sync_file loss_k4_layer16.json
sync_file loss_k4_layer16.cells.jsonl

echo
echo "Done → ${DEST}/"
