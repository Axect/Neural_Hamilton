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
