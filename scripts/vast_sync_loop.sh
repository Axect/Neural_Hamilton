#!/bin/bash
# Stability-gated periodic sync of Vast.ai campaign artifacts to local.
# Phase 1: every 5 min, check campaign health; 3 consecutive healthy checks = stable.
# Phase 2: every 30 min, snapshot on the instance (sqlite backup + rsync -a), then
#          rsync the frozen snapshot to local. Stops after the campaign's ALL DONE marker.
# Usage: VAST_HOST=sshN.vast.ai VAST_PORT=NNNNN bash scripts/vast_sync_loop.sh
set -u
HOST="${VAST_HOST:?set VAST_HOST}"
PORT="${VAST_PORT:?set VAST_PORT}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20)
REMOTE=/workspace/Neural_Hamilton
SNAP=/workspace/snapshot

rssh() { ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

echo "[sync] phase 1: stability check every 5 min (need 3 consecutive healthy)"
healthy=0
while [ "$healthy" -lt 3 ]; do
  util=$(rssh "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | tr -d ' \r')
  alive=$(rssh "tmux has-session -t hpo 2>/dev/null && echo yes || echo no" 2>/dev/null | tr -d '\r')
  bad=$(rssh "grep -cE 'Traceback|CUDA error|RuntimeError' /workspace/provision.log 2>/dev/null" 2>/dev/null | tr -d '\r')
  echo "[check $(date +%H:%M)] util=${util:-?}% tmux=${alive:-?} errors=${bad:-?}"
  if [ "${alive:-no}" = "yes" ] && [ "${util:-0}" -ge 30 ] 2>/dev/null && [ "${bad:-1}" -eq 0 ] 2>/dev/null; then
    healthy=$((healthy + 1))
  else
    healthy=0
  fi
  [ "$healthy" -ge 3 ] && break
  sleep 300
done

echo "[sync] stable; phase 2: sync every 30 min"
while true; do
  # Freeze a consistent snapshot on the instance: sqlite backup API for hot DBs,
  # rsync -a for runs/ (model.pt written at seed end; latest_model.pt may be torn
  # in the snapshot but is only a resume convenience, never a result artifact).
  rssh "mkdir -p $SNAP && rsync -a --delete $REMOTE/runs/ $SNAP/runs/ 2>/dev/null;
        cp /workspace/provision.log $SNAP/ 2>/dev/null;
        cd $REMOTE && python3 - <<'PY'
import sqlite3, glob, os
for f in glob.glob('*.db'):
    src = sqlite3.connect(f)
    dst = sqlite3.connect(os.path.join('$SNAP', f))
    src.backup(dst)
    dst.close(); src.close()
PY" || { echo "[sync] remote snapshot failed, retrying next tick"; sleep 1800; continue; }

  rsync -az --partial --partial-dir=.rsync-partial -e "ssh ${SSH_OPTS[*]} -p $PORT" \
    "root@$HOST:$SNAP/runs/" ./runs/ || echo "[sync] runs rsync failed"
  rsync -az --partial -e "ssh ${SSH_OPTS[*]} -p $PORT" \
    "root@$HOST:$SNAP/*.db" ./ || echo "[sync] db rsync failed"
  rsync -az -e "ssh ${SSH_OPTS[*]} -p $PORT" \
    "root@$HOST:$SNAP/provision.log" ./vast_provision.log || true
  echo "[sync] done $(date +%H:%M)"

  if rssh "grep -q 'ALL DONE' /workspace/provision.log" 2>/dev/null; then
    echo "[sync] CAMPAIGN COMPLETE, final sync finished"
    break
  fi
  sleep 1800
done
