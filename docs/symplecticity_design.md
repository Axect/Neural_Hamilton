# Symplecticity violation analysis (G4.3)

Design for measuring how far learned flow maps deviate from the symplectic structure of the true Hamiltonian flow. This is the differentiating analysis for the MLST submission: no prior work measures whether operator-learned flow maps spontaneously acquire approximate symplecticity without built-in structure preservation.

## Object and metric

For a fixed potential V, the learned model defines a flow map Phi_t: (q0, p0) -> (q(t), p(t)). The true flow satisfies J(t)^T Omega J(t) = Omega with J(t) = d(q(t), p(t)) / d(q0, p0) in R^{2x2} and Omega = [[0, 1], [-1, 0]].

For 1 degree of freedom, J^T Omega J = det(J) Omega identically, so symplecticity is exactly area preservation and the deviation reduces to a scalar:

    s(t) = |det J(t) - 1|

This is the primary metric. Report it directly; do not dress it up as a Frobenius norm of J^T Omega J - Omega, which is just sqrt(2) * s(t) in 1-DOF. State this equivalence in the paper because it makes the analysis interpretable (Liouville area preservation).

Secondary metric (accuracy of the tangent map, not just its structure):

    e_J(t) = || J_model(t) - J_true(t) ||_F

J_true comes from integrating the variational equation dJ/dt = A(t) J, A(t) = [[0, 1], [-V''(q(t)), 0]], J(0) = I, alongside the reference trajectory. V'' is available analytically from the cubic spline of the sensor values. This separates two failure modes: a model can have det J close to 1 (structure right) while J itself is wrong (dynamics wrong), and vice versa.

## Computing J_model

Default: central finite differences on the IC input, which is model-agnostic and avoids vmap incompatibilities with the Mamba pscan kernel:

    J[:, 0] = (Phi(q0 + eps, p0) - Phi(q0 - eps, p0)) / (2 eps)
    J[:, 1] = (Phi(q0, p0 + eps) - Phi(q0, p0 - eps)) / (2 eps)

with eps = 1e-3 in normalized units (q in [0, 1], p in [-2.6, 2.6]); sweep eps over {1e-2, 1e-3, 1e-4} once to confirm a plateau. Cost: 4 extra batched forward passes for the whole evaluation set, negligible.

Validation: torch.func.jacrev on a subset (DeepONet and TraONet, which are vmap-safe) to confirm FD agrees to <1% before trusting FD for MambONet.

Caveat: eps perturbation moves the IC slightly off the energy used at sampling time. That is fine; J is defined on open phase space, and the model accepts any (q0, p0). But perturbed ICs near the wall (E close to V0) can leave the bounded regime; mask samples with E0 > V0 - 0.05.

## Baseline references

- Yoshida 4th order: symplectic by construction; |det J - 1| at machine precision. Floor reference.
- RK4: violates area preservation at O(dt^5) per step; accumulated violation over [0, 2] at the matched dt gives the "how bad is a standard non-symplectic method" reference scale. Compute by the same FD through the integrator.
- True flow: |det J - 1| = 0 exactly; J_true from variational integration is the accuracy reference.

Both solver baselines use the same V spline and dt conventions as fair_compare.py.

## Measurement axes

1. Time within window: s(t) at the 100 output time points, t in [0, 2].
2. Model: DeepONet / FNO / TraONet / MambONet (all v1.33-trained, best configs, all seeds).
3. Energy stratum: u in the 4 training strata, read from the test parquet metadata.
4. Autoregressive chaining: for the composed map Phi^(n) = Phi o ... o Phi over n windows, det J multiplies, so log|det J_n| = sum of per-window log dets. Hypothesis: log-det error grows linearly in n, and its sign/magnitude correlates with the energy drift direction from fair_compare Approach D. Test n up to 50 (t = 100).
5. Correlation: per-sample scatter of s(T) vs |Delta H(T)| and vs trajectory MSE. Question: is symplectic violation a better predictor of long-term failure than instantaneous MSE?

Sample budget: 256 test potentials x 4 strata x 100 time points. With FD this is ~1k batched forwards per model. Minutes on GPU.

## Paper figures

- F-S1: median and IQR of s(t) vs t per model, one panel per energy stratum. Y4 and RK4 reference lines.
- F-S2: log|det J_n| vs window count n per model, with the linear-accumulation fit; RK4/Y4 references.
- F-S3: scatter s(T) vs |Delta H(T)| per model, with rank correlation in the legend.

Expected headline sentences (to be confirmed by data): "learned flow maps violate area preservation at a level intermediate between RK4 and Y4 within the training window, but the violation compounds multiplicatively under autoregressive chaining" or its refutation. Either outcome is a publishable claim.

## Implementation

`scripts/symplecticity.py`:

1. Load model(s) via `util.load_model`, test set via `util.load_data("data_test/test.parquet")` plus metadata columns read directly with polars.
2. FD Jacobian at all output times in one batched pass per perturbation direction.
3. Variational J_true via scipy `solve_ivp` (RK45, rtol 1e-10) on the 6-dim system (q, p, J11, J12, J21, J22) using CubicSpline V'' from the 100 sensor values.
4. Solver baselines: numpy Y4/RK4 with FD Jacobian at matched dt.
5. Output: one parquet of (pid, model, seed, u, t, det_J, e_J, dH) rows; plotting kept separate.

Open items: eps plateau check; whether to subsample seeds for J_true (variational integration is CPU-bound, ~5k solves are fine); 2-DOF extension would replace det J with the full 4x4 J^T Omega J - Omega norm (design holds, metric generalizes).
