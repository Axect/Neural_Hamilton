#!/bin/bash
set -euo pipefail
exec > >(tee -a /workspace/training.log) 2>&1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd /workspace/Neural_Hamilton
source .venv/bin/activate
echo "=== [8] Final training campaign $(date -u) ==="
for m in mambonet fno deeponet traonet; do
  echo "--- TRAIN: $m start $(date -u) ---"
  python main.py --run_config configs/v1.33/${m}_best.yaml --device cuda:0 --resume
  echo "--- TRAIN: $m done $(date -u) ---"
done
echo "=== ALL DONE $(date -u) ==="
