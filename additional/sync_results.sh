#!/bin/bash
# Pull the additional-experiment results from the remote host into the local
# additional_exp/ tree.
#
# Training and evaluation run on TG-A40G, but the study's deliverable is local:
# every table, figure, raw record and run log is kept under additional_exp/ so
# the results survive independently of the remote scratch space.
#
# Only results are pulled. Checkpoints (several GB each) and the extracted SMD
# frames (about 46 GB) stay on the remote; the manifest written here records
# where they live so a rerun can find them.
#
# Usage:
#   bash additional/sync_results.sh [local_additional_exp_dir]

set -euo pipefail

REMOTE_SSH_PORT=11022
REMOTE_HOST=yvvyee@129.254.222.148
REMOTE_WORK=/mnt/ssd1/yvvyee/gaa_additional
LOCAL_DIR="${1:-/media/ywlee/T9/Documents/WORK/Workspace/GAA/additional_exp}"

mkdir -p "$LOCAL_DIR/raw" "$LOCAL_DIR/logs" "$LOCAL_DIR/data_stats"

echo "[sync] results  $REMOTE_HOST:$REMOTE_WORK/results -> $LOCAL_DIR/raw"
rsync -a --info=stats2 -e "ssh -p $REMOTE_SSH_PORT -o BatchMode=yes" \
  "$REMOTE_HOST:$REMOTE_WORK/results/" "$LOCAL_DIR/raw/" | tail -4

echo "[sync] logs     $REMOTE_HOST:$REMOTE_WORK/logs -> $LOCAL_DIR/logs"
rsync -a -e "ssh -p $REMOTE_SSH_PORT -o BatchMode=yes" \
  --include='*/' --include='*.log' --include='*.txt' --exclude='*' \
  "$REMOTE_HOST:$REMOTE_WORK/logs/" "$LOCAL_DIR/logs/"

echo "[sync] run scripts"
rsync -a -e "ssh -p $REMOTE_SSH_PORT -o BatchMode=yes" \
  --include='run_*.sh' --exclude='*' \
  "$REMOTE_HOST:$REMOTE_WORK/src/" "$LOCAL_DIR/logs/run_scripts/"

echo "[sync] manifest"
{
  echo "# Additional-experiment artefact manifest"
  echo
  echo "Generated: $(date -Iseconds)"
  echo
  echo "## Remote locations (not synced)"
  echo
  echo "Host: $REMOTE_HOST (port $REMOTE_SSH_PORT), conda env: eva"
  echo "Work root: $REMOTE_WORK"
  echo
  echo '```'
  ssh -p "$REMOTE_SSH_PORT" -o BatchMode=yes "$REMOTE_HOST" \
    "du -sh $REMOTE_WORK/checkpoints/* $REMOTE_WORK/frames $REMOTE_WORK/SMD \
       $REMOTE_WORK/data 2>/dev/null | sort -k2"
  echo '```'
  echo
  echo "## Checkpoint inventory"
  echo
  echo '```'
  ssh -p "$REMOTE_SSH_PORT" -o BatchMode=yes "$REMOTE_HOST" \
    "for d in $REMOTE_WORK/checkpoints/*/; do
       printf '%-28s ' \"\$(basename \$d)\"
       if [ -f \"\$d/train_meta.json\" ]; then
         python3 -c \"
import json,sys
m=json.load(open('\$d/train_meta.json'))
print(f\\\"variant={m['variant']:12s} lam={m['lam']:<5} n={m['n_samples']:<6} steps={m['total_steps']:<5} {m['wall_seconds']}s\\\")\"
       else
         echo '(no train_meta.json)'
       fi
     done"
  echo '```'
} > "$LOCAL_DIR/MANIFEST.md"

echo "[sync] done -> $LOCAL_DIR"
du -sh "$LOCAL_DIR"
