#!/usr/bin/env bash
# scp Neuronic metrics files into metrics/layer-experiments/ only if missing locally.
# Layer sweep artifacts (under remote .../metrics/):
#   3× .json + 3× .jsonl = 6 files for interval@L2 + loss@L6 + loss@L16.
#
# Run on your laptop:
#   bash scripts/scp_missing_layer_metrics_from_neuronic.sh
#
# Force re-download even when present:
#   FORCE=1 bash scripts/scp_missing_layer_metrics_from_neuronic.sh
#
# Override:
#   REMOTE_HOST=user@host REMOTE_REPO=/path/to/COS484-BAM DEST=metrics/layer-experiments bash scripts/scp_missing_layer_metrics_from_neuronic.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-bw7520@neuronic.cs.princeton.edu}"
REMOTE_REPO="${REMOTE_REPO:-/u/bw7520/COS484-BAM}"
REMOTE_METRICS="${REMOTE_REPO}/metrics"

DEST="${DEST:-${REPO_ROOT}/metrics/layer-experiments}"
FORCE="${FORCE:-0}"

mkdir -p "${DEST}"

# Flat files from Neuronic ${REMOTE_REPO}/metrics/ (same layout as `ls` on cluster).
FILES=(
  interval_k4_layer2.json
  interval_k4_layer2.cells.jsonl
  loss_k4_layer6.json
  loss_k4_layer6.cells.jsonl
  loss_k4_layer16.json
  loss_k4_layer16.cells.jsonl
)

echo "Remote: ${REMOTE_HOST}:${REMOTE_METRICS}/"
echo "Local:  ${DEST}/"
echo

for f in "${FILES[@]}"; do
  local_path="${DEST}/${f}"
  if [[ -f "${local_path}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${f} (already exists: ${local_path})"
    continue
  fi
  echo "[scp] ${f}"
  scp "${REMOTE_HOST}:${REMOTE_METRICS}/${f}" "${DEST}/"
done

echo
echo "Done."
