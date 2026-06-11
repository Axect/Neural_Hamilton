# Ver 1.33

- **Phase-space IC sampling**: training samples are no longer windows of a single origin-started orbit (which pinned every sample to the energy shell E = V0 = 2). Each potential now yields `NDIFFCONFIG=4` orbits with stratified energy fraction u = (E - V(q0)) / (V0 - V(q0)) over [0.05, 0.35) / [0.35, 0.65) / [0.65, 0.95) / [0.95, 0.99). Sub-well libration and near-bottom regimes enter the training distribution (~23% of orbits cover less than 60% of the domain).
- **Explicit depth sampling**: interior potential values are mapped to [V0 - D, V0 - 0.1] with D ~ U(0.5, 3.5) per sample, replacing batch-global min-max scaling (which made the depth distribution depend on batch composition).
- **Confinement by construction**: E < V0 guarantees orbits stay in [0, L], removing the boundary-rejection bias of the previous pipeline. Rejection counters are printed per filter.
- **Metadata columns**: parquet now carries `pid, b, l, depth, E0, u` per sample for OOD analysis, regime-stratified evaluation, and leakage audits. `scripts/validate_data.py` checks strata coverage, depth, and phase-space coverage.
- **Energy drift filter**: evaluated on dense solver points relative to the per-potential energy scale (the old relative formula was unsafe for E0 < 0).
- **pytorch_template migration (v0 -> current)**: adds `callbacks.py`, `checkpoint.py`, `provenance.py`, `cli.py`, `metrics.py`; replaces `config.py`, `main.py`, `pruner.py`; rewrites `util.py` on the template base with the (V, t, q, p, ic) batch signature kept in `Trainer`. VaRONet variational path removed. New commands: `python -m cli preflight/doctor/hpo_report`. Resume via `--resume` and `latest_model.pt`.
- **configs/v1.33**: run + opt configs for DeepONet/FNO/MambONet/TraONet with `data`/`criterion`/`wandb` fields, HPO at 10 epochs (scheduler max_iter 10), seeds [89, 231, 928].
- **Breaking**: smoke mode added to data generator (`neural_hamilton 3 <n>`); old datasets remain loadable but lack metadata columns and are on-shell (E = 2) only.

# Ver 1.31

- **Initial Condition (IC) Support**: All models now require `(q0, p0)` as input
  - New forward signature: `model(V, t, ic)` where `ic = (batch, 2)`
  - `ICEmbedding` module added for conditioning
  - Additive conditioning in decoder: `x = x + ic_embed(ic)`
- **Autoregressive Long-term Prediction**: Chain multiple windows using predicted endpoints
  - New function: `autoregressive_predict(model, V, n_windows, ...)`
- **Data Pipeline Updates**:
  - `load_data()` now returns `(V, t, q, p, ic)` with IC extracted from `q[:, 0], p[:, 0]`
  - `solvers.rs` and `true_trajectories.jl` use actual IC from data (not origin)
- **Build System**: `just build-cargo` now builds only required binaries (neural_hamilton, relevant, solvers)
- **Breaking Change**: Existing checkpoints incompatible (new `ic_embed` layers)

# Ver 1.30

- Total data = n * NDIFFCONFIG
- Batch size: 256

# Ver 1.29

- Remove umap

# Ver 1.28

- Introduce `NDIFFCONFIG` to control the number of different time configurations per potential.
  - This version uses `NDIFFCONFIG=2` by default, which means two different time configurations per potential.
