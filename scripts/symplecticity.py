"""Symplecticity violation measurement for learned Hamiltonian flow maps (G4.3).

For 1-DOF systems symplecticity is area preservation, so the primary metric is
s(t) = |det J(t) - 1| with J(t) = d(q(t), p(t)) / d(q0, p0).
See docs/symplecticity_design.md for the full design.

Usage:
    python scripts/symplecticity.py --project <project> --group <group> --seed <seed> \
        --data data_test/test.parquet --out results/symplecticity/<model>.parquet \
        [--n-potentials 256] [--eps 1e-3] [--device cuda:0]

Outputs one parquet of per-(sample, time) rows:
    pid, u, depth, E0, t, det_J, e_J, dH
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util import load_model  # noqa: E402

NSENSORS = 100
V0 = 2.0
E_WALL_MARGIN = 0.05  # mask samples whose FD perturbation can leave the bounded regime


def load_test_with_meta(path: str, n_potentials: int):
    """Load test parquet keeping one sample per (pid, stratum), up to n_potentials pids."""
    df = pl.read_parquet(path)
    n = df.height // NSENSORS
    cols = {c: df[c].to_numpy().reshape(n, NSENSORS) for c in ["V", "t", "q", "p"]}
    meta = {c: df[c].to_numpy().reshape(n, NSENSORS)[:, 0] for c in ["pid", "u", "depth", "E0"]}

    keep_pids = np.unique(meta["pid"])[:n_potentials]
    mask = np.isin(meta["pid"], keep_pids) & (meta["E0"] < V0 - E_WALL_MARGIN)
    idx = np.where(mask)[0]
    return cols, meta, idx


def fd_jacobian(model, V, t, ic, eps, device, batch_size=512):
    """Central FD Jacobian of the flow map wrt the IC, at all output times.

    Returns J with shape (N, N_t, 2, 2): J[i, k] = d(q_k, p_k)/d(q0, p0).
    """
    model.eval()
    offsets = [
        torch.tensor([eps, 0.0]), torch.tensor([-eps, 0.0]),
        torch.tensor([0.0, eps]), torch.tensor([0.0, -eps]),
    ]
    outs = []
    with torch.inference_mode():
        for off in offsets:
            qs, ps = [], []
            for i in range(0, V.shape[0], batch_size):
                sl = slice(i, i + batch_size)
                q_pred, p_pred = model(
                    V[sl].to(device), t[sl].to(device), (ic[sl] + off).to(device)
                )
                qs.append(q_pred.cpu())
                ps.append(p_pred.cpu())
            outs.append((torch.cat(qs), torch.cat(ps)))

    (qpx, ppx), (qmx, pmx), (qpy, ppy), (qmy, pmy) = outs
    J = torch.empty(V.shape[0], t.shape[1], 2, 2)
    J[..., 0, 0] = (qpx - qmx) / (2 * eps)  # dq/dq0
    J[..., 0, 1] = (qpy - qmy) / (2 * eps)  # dq/dp0
    J[..., 1, 0] = (ppx - pmx) / (2 * eps)  # dp/dq0
    J[..., 1, 1] = (ppy - pmy) / (2 * eps)  # dp/dp0
    return J


def true_tangent_map(V_row, t_row, q0, p0):
    """Integrate (q, p, J) variational system with the spline potential.

    Returns (q(t), p(t), J(t)) at the requested times; J has shape (N_t, 2, 2).
    """
    qgrid = np.linspace(0.0, 1.0, NSENSORS)
    spline = CubicSpline(qgrid, V_row)
    dV = spline.derivative(1)
    d2V = spline.derivative(2)

    def rhs(_t, y):
        q, p, j11, j12, j21, j22 = y
        a = -d2V(q)
        return [p, -dV(q), j21, j22, a * j11, a * j12]

    sol = solve_ivp(
        rhs, (0.0, float(t_row[-1])), [q0, p0, 1.0, 0.0, 0.0, 1.0],
        t_eval=t_row, method="RK45", rtol=1e-10, atol=1e-12,
    )
    J = np.stack(
        [sol.y[2], sol.y[3], sol.y[4], sol.y[5]], axis=-1
    ).reshape(-1, 2, 2)
    return sol.y[0], sol.y[1], J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--data", default="data_test/test.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-potentials", type=int, default=256)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-true-J", action="store_true", help="skip CPU-bound variational integration")
    args = ap.parse_args()

    model, _config = load_model(args.project, args.group, args.seed)
    model = model.to(args.device)

    cols, meta, idx = load_test_with_meta(args.data, args.n_potentials)
    V = torch.tensor(cols["V"][idx], dtype=torch.float32)
    t = torch.tensor(cols["t"][idx], dtype=torch.float32)
    q = torch.tensor(cols["q"][idx], dtype=torch.float32)
    p = torch.tensor(cols["p"][idx], dtype=torch.float32)
    ic = torch.stack([q[:, 0], p[:, 0]], dim=1)
    print(f"evaluating {len(idx)} samples on {args.device}")

    J = fd_jacobian(model, V, t, ic, args.eps, args.device)
    det_J = (J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]).numpy()

    # Energy drift of the model prediction along the trajectory
    with torch.inference_mode():
        q_pred, p_pred = [], []
        for i in range(0, V.shape[0], 512):
            sl = slice(i, i + 512)
            qq, pp = model(V[sl].to(args.device), t[sl].to(args.device), ic[sl].to(args.device))
            q_pred.append(qq.cpu())
            p_pred.append(pp.cpu())
    q_pred = torch.cat(q_pred).numpy()
    p_pred = torch.cat(p_pred).numpy()
    qgrid = np.linspace(0.0, 1.0, NSENSORS)
    Vq = np.stack([np.interp(q_pred[i], qgrid, cols["V"][idx][i]) for i in range(len(idx))])
    H = 0.5 * p_pred**2 + Vq
    dH = np.abs(H - H[:, :1])

    e_J = np.full_like(det_J, np.nan)
    if not args.skip_true_J:
        for i in range(len(idx)):
            _, _, Jt = true_tangent_map(
                cols["V"][idx][i], cols["t"][idx][i], float(ic[i, 0]), float(ic[i, 1])
            )
            e_J[i] = np.linalg.norm(J[i].numpy() - Jt, axis=(-2, -1))

    n_s, n_t = det_J.shape
    out = pl.DataFrame({
        "pid": np.repeat(meta["pid"][idx], n_t),
        "u": np.repeat(meta["u"][idx], n_t),
        "depth": np.repeat(meta["depth"][idx], n_t),
        "E0": np.repeat(meta["E0"][idx], n_t),
        "t": cols["t"][idx].ravel(),
        "det_J": det_J.ravel(),
        "e_J": e_J.ravel(),
        "dH": dH.ravel(),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    s = np.abs(det_J - 1.0)
    print(f"wrote {args.out}: median |det J - 1| = {np.median(s):.3e}, p95 = {np.percentile(s, 95):.3e}")


if __name__ == "__main__":
    main()
