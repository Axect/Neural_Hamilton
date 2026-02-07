default: build

install:
    uv python install 3.13
    uv venv
    sh install_requirements.sh

# Build only the required binaries (neural_hamilton, relevant, solvers)
build-cargo:
    cargo build --release --bin neural_hamilton --bin relevant --bin solvers

# Build all binaries including experimental ones
build-cargo-all:
    cargo build --release

data-gen:
    for i in {0..2}; do \
        cargo run --release --bin neural_hamilton -- $i; \
    done
    cargo run --release --bin relevant

data-filter:
    .venv/bin/python scripts/umap_filter.py --data_file="data_test/test_cand.parquet"
    .venv/bin/python scripts/umap_filter.py --data_file="data_normal/train_cand.parquet"
    .venv/bin/python scripts/umap_filter.py --data_file="data_normal/val_cand.parquet"
    .venv/bin/python scripts/umap_filter.py --data_file="data_more/train_cand.parquet"
    .venv/bin/python scripts/umap_filter.py --data_file="data_more/val_cand.parquet"
    mv "data_test/test_cand_samples.parquet" "data_test/test.parquet"
    mv "data_normal/train_cand_samples.parquet" "data_normal/train.parquet"
    mv "data_normal/val_cand_samples.parquet" "data_normal/val.parquet"
    mv "data_more/train_cand_samples.parquet" "data_more/train.parquet"
    mv "data_more/val_cand_samples.parquet" "data_more/val.parquet"

post-process:
    julia -t auto scripts/true_trajectories.jl # True data via Kahan-Li 8th order
    cargo run --release --bin solvers # Various solvers to compare

build: build-cargo

all: install build-cargo data-gen post-process

# ============================================================
# Experimental/Utility binaries (not part of main pipeline)
# ============================================================
# These binaries are available but not used in the main workflow:
#   - adjust_eta_inf: Learning rate adjustment utility
#   - unbounded: Unbounded potential experiment
#   - cliff: Cliff potential experiment
#   - is_conserved: Energy conservation verification
#   - rk4: RK4 standalone test (functionality in solvers.rs)
#   - gl4: GL4 standalone test (functionality in solvers.rs)
#   - rkf78: RKF78 adaptive integrator test
