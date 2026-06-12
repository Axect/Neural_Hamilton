#!/bin/bash
# Neural_Hamilton v1.33 — Vast.ai provisioning + HPO campaign
# Run inside tmux on the instance. Logs to /workspace/provision.log
set -euo pipefail
exec > >(tee -a /workspace/provision.log) 2>&1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive

echo "=== [0] prerequisites $(date -u) ==="
apt-get update -qq
apt-get install -y -qq rsync curl git build-essential tmux pkg-config libssl-dev jq

mkdir -p /workspace
cd /workspace

echo "=== [1] clone repo ==="
if [ ! -d Neural_Hamilton ]; then
  git clone --depth 1 https://github.com/Axect/Neural_Hamilton.git
fi
cd Neural_Hamilton
git log --oneline -1

echo "=== [2] uv + python ==="
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13
rm -rf .venv
uv venv --python 3.13
source .venv/bin/activate
uv pip install -U torch wandb polars numpy optuna matplotlib scienceplots beaupy rich mambapy scipy scikit-learn typer tqdm pyyaml

echo "=== [3] verify CUDA torch (pin down to driver CUDA if needed) ==="
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DRV_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
  TAG="cu$(echo "$DRV_CUDA" | tr -d '.')"
  echo "latest torch wheel incompatible with driver CUDA $DRV_CUDA; pinning to $TAG"
  uv pip install torch --index-url "https://download.pytorch.org/whl/$TAG"
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "=== [4] rust + data generation ==="
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
fi
cargo build --release --bin neural_hamilton
./target/release/neural_hamilton 0
./target/release/neural_hamilton 2
./target/release/neural_hamilton 1

echo "=== [5] validate data ==="
python scripts/validate_data.py data_normal/train.parquet
python scripts/validate_data.py data_more/train.parquet

echo "=== [6] preflight ==="
for m in mambonet fno deeponet traonet; do
  python -m cli preflight configs/v1.33/${m}_run.yaml --device cuda:0 --json | python -c "import sys,json; r=json.load(sys.stdin); print('${m}:', 'PASS' if r['passed'] else 'FAIL'); sys.exit(0 if r['passed'] else 1)"
done

echo "=== [7] HPO campaign (sequential) $(date -u) ==="
for m in mambonet fno deeponet traonet; do
  echo "--- HPO: $m start $(date -u) ---"
  python main.py --run_config configs/v1.33/${m}_run.yaml --optimize_config configs/v1.33/${m}_opt.yaml --device cuda:0
  echo "--- HPO: $m done $(date -u) ---"
done

echo "=== ALL DONE $(date -u) ==="
