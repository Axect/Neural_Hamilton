default: build

install:
    uv python install 3.13
    uv venv
    sh install_requirements.sh

activate:
    source .venv/bin/activate

build-cargo:
    cargo build --release

data-gen:
    for i in {0..2}; do \
        cargo run --release --bin neural_hamilton -- $i; \
    done 
    cargo run --release --bin relevant

data-filter: activate
    python scripts/umap_filter.py --data_file="data_test/test_cand.parquet"
    python scripts/umap_filter.py --data_file="data_normal/train_cand.parquet"
    python scripts/umap_filter.py --data_file="data_normal/val_cand.parquet"
    python scripts/umap_filter.py --data_file="data_more/train_cand.parquet"
    python scripts/umap_filter.py --data_file="data_more/val_cand.parquet"
    mv "data_test/test_cand_samples.parquet" "data_test/test.parquet"
    mv "data_normal/train_cand_samples.parquet" "data_normal/train.parquet"
    mv "data_normal/val_cand_samples.parquet" "data_normal/val.parquet"
    mv "data_more/train_cand_samples.parquet" "data_more/train.parquet"
    mv "data_more/val_cand_samples.parquet" "data_more/val.parquet"

build: build-cargo

all: install build-cargo data-gen data-filter
