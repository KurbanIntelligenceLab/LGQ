#!/usr/bin/env bash
# Build a flat-named staging dir of best/latest checkpoints and rclone-copy it to gdrive.
#
# Usage:
#   shell/upload_to_gdrive.sh                          # uploads to gdrive:vq-checkpoints/
#   shell/upload_to_gdrive.sh gdrive:my/dest           # custom remote path
#   KINDS="best latest" shell/upload_to_gdrive.sh      # filter kinds

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEST="${1:-gdrive:vq-checkpoints}"
KINDS="${KINDS:-best latest}"
STAGING="uploads/staging"
MANIFEST="uploads/gdrive_manifest.txt"
INDEX="uploads/gdrive_index.csv"
LOG="uploads/rclone_upload.log"

PY="${PYTHON:-python3}"

echo "[upload] building manifest (kinds=$KINDS)"
$PY scripts/build_upload_manifest.py \
    --kinds $KINDS \
    --staging-dir "$STAGING" \
    --manifest "$MANIFEST" \
    --index "$INDEX" \
    --clean-staging

echo "[upload] rclone copy → $DEST"
rclone copy \
    --files-from "$MANIFEST" \
    --transfers 4 \
    --checkers 8 \
    --stats 30s \
    --log-file "$LOG" \
    --log-level INFO \
    . "$DEST"

echo "[upload] done. log: $LOG"
echo "[upload] index uploaded as: $DEST/$INDEX"
