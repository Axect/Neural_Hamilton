default: build

install:
    uv python install 3.13
    uv venv
    sh install_requirements.sh

build-cargo:
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
    julia -t 32 scripts/true_trajectories.jl # True data via Kahan-Li 8th order
    cargo run --release --bin solvers # Various solvers to compare

build: build-cargo

#all: install build-cargo data-gen data-filter data-rk4
all: install build-cargo data-gen post-process
