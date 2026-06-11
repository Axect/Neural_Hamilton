"""Validate a generated dataset: energy strata coverage, depth distribution, phase-space coverage, and sub-well occupancy.

Usage: python scripts/validate_data.py <path/to/data.parquet>
"""

import sys

import numpy as np
import polars as pl

NSENSORS = 100
V0 = 2.0


def main(path: str):
    df = pl.read_parquet(path)
    n = df.height // NSENSORS
    V = df["V"].to_numpy().reshape(n, NSENSORS)
    q = df["q"].to_numpy().reshape(n, NSENSORS)
    p = df["p"].to_numpy().reshape(n, NSENSORS)

    has_meta = "E0" in df.columns
    print(f"== {path}: {n} samples, metadata={'yes' if has_meta else 'no'} ==")

    qgrid = np.linspace(0, 1, NSENSORS)
    q0, p0 = q[:, 0], p[:, 0]
    Vq0 = np.array([np.interp(q0[i], qgrid, V[i]) for i in range(n)])
    E_emp = 0.5 * p0**2 + Vq0

    if has_meta:
        E0 = df["E0"].to_numpy().reshape(n, NSENSORS)[:, 0]
        u = df["u"].to_numpy().reshape(n, NSENSORS)[:, 0]
        depth = df["depth"].to_numpy().reshape(n, NSENSORS)[:, 0]
        pid = df["pid"].to_numpy().reshape(n, NSENSORS)[:, 0]
        b = df["b"].to_numpy().reshape(n, NSENSORS)[:, 0]

        # Stored E0 vs empirical H(q0, p0) from linear interp of V grid
        err = np.abs(E_emp - E0)
        print(f"E0 stored vs empirical: max abs diff {err.max():.2e} (interp-level agreement expected)")

        print(f"E0: min {E0.min():.4f}, max {E0.max():.4f}, mean {E0.mean():.4f}, std {E0.std():.4f}")
        strata = [(0.05, 0.35), (0.35, 0.65), (0.65, 0.95), (0.95, 0.99)]
        for lo, hi in strata:
            frac = ((u >= lo) & (u < hi)).mean()
            print(f"  stratum u in [{lo:.2f},{hi:.2f}): {frac:.3f}")
        print(f"depth: min {depth.min():.3f}, max {depth.max():.3f}, mean {depth.mean():.3f}")
        print(f"unique pid: {len(np.unique(pid))}, samples/pid: {n / len(np.unique(pid)):.2f}")
        print(f"b values: {sorted(np.unique(b).astype(int).tolist())}")
    else:
        print(f"empirical E: min {E_emp.min():.4f}, max {E_emp.max():.4f}, std {E_emp.std():.4f}")
        print(f"fraction |E-2|<0.01 (legacy on-shell): {(np.abs(E_emp - 2) < 0.01).mean():.4f}")

    print(f"q0 range: [{q0.min():.4f}, {q0.max():.4f}], p0 range: [{p0.min():.4f}, {p0.max():.4f}]")
    print(f"q range overall: [{q.min():.4f}, {q.max():.4f}] (must be within [0, 1])")

    # Sub-well occupancy: fraction of orbits whose q-excursion covers less than
    # 60% of the domain (these regimes were absent in the legacy on-shell data)
    excursion = q.max(axis=1) - q.min(axis=1)
    print(f"orbit q-excursion: mean {excursion.mean():.3f}, frac < 0.6: {(excursion < 0.6).mean():.3f}, frac < 0.3: {(excursion < 0.3).mean():.3f}")

    # Depth diversity check on V itself
    Vmin = V.min(axis=1)
    print(f"V min per sample: [{Vmin.min():.3f}, {Vmin.max():.3f}], boundaries V[0]={V[:, 0].min():.4f}..{V[:, 0].max():.4f}")

    # Energy conservation along stored trajectory points (interp-level check)
    idx = np.random.default_rng(0).choice(n, size=min(500, n), replace=False)
    drifts = []
    for i in idx:
        Vq = np.interp(q[i], qgrid, V[i])
        H = 0.5 * p[i] ** 2 + Vq
        drifts.append(H.max() - H.min())
    drifts = np.array(drifts)
    print(f"H drift along stored points (interp-level): median {np.median(drifts):.2e}, p95 {np.percentile(drifts, 95):.2e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_smoke/smoke.parquet")
