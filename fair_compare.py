"""
Fair comparison of Neural Hamilton model vs traditional ODE solvers.

The existing comparison (solvers.rs) is unfair because Y4/RK4 solvers use
CubicHermiteSpline interpolation to evaluate at arbitrary time points, adding
interpolation error. The neural model directly outputs at arbitrary times
without interpolation. This module implements three fair comparison approaches:

A. Native Grid Comparison: Evaluate all methods at the solver's native grid
B. Work-Precision Diagram: Compare accuracy vs computational cost
C. Energy Conservation: Compare Hamiltonian preservation across methods
"""

import os
import time as time_module

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch
import matplotlib.pyplot as plt
import scienceplots
import polars as pl
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================
NSENSORS = 100
NDENSE = 1000

# Yoshida 4th-order coefficients (must match Rust exactly)
W0 = -1.7024143839193153
W1 = 1.3512071919596578
YOSHIDA_C = [W1 / 2.0, (W0 + W1) / 2.0, (W0 + W1) / 2.0, W1 / 2.0]
YOSHIDA_D = [W1, W0, W1, 0.0]

# Paper-quality colors
COLORS = {
    "model": "#E74C3C",
    "y4": "#3498DB",
    "rk4": "#2ECC71",
    "kl8": "#95A5A6",
}


# =============================================================================
# 1. Python ODE Solvers (matching Rust implementations exactly)
# =============================================================================
class PotentialODE:
    """
    Represents a Hamiltonian system H = p^2/2 + V(q) where V is given as
    100 discrete samples on q in [0, 1].

    Uses PCHIP interpolation for V(q) and its derivative, matching the
    CubicHermiteSpline approach in the Rust code.
    """

    def __init__(self, V_values):
        """
        Parameters
        ----------
        V_values : array-like, shape (100,)
            Potential energy values sampled uniformly on q in [0, 1].
        """
        V_values = np.asarray(V_values, dtype=np.float64)
        x = np.linspace(0.0, 1.0, len(V_values))
        self._V_interp = PchipInterpolator(x, V_values)
        self._dV_interp = self._V_interp.derivative()

    def dVdq(self, q):
        """Evaluate dV/dq at q, clamping q to [0, 1]."""
        q_clamped = np.clip(q, 0.0, 1.0)
        return float(self._dV_interp(q_clamped))

    def V_eval(self, q):
        """Evaluate V(q), clamping q to [0, 1]."""
        q_clamped = np.clip(q, 0.0, 1.0)
        return float(self._V_interp(q_clamped))

    def V_eval_vec(self, q_arr):
        """Vectorized V(q) evaluation."""
        q_clamped = np.clip(np.asarray(q_arr, dtype=np.float64), 0.0, 1.0)
        return self._V_interp(q_clamped)


def yoshida4_solve(ode, tspan, dt, q0, p0):
    """
    Yoshida 4th-order symplectic integrator.

    Matches the Rust implementation in solvers.rs and relevant.rs exactly:
      for j in 0..4:
          rhs(q, p) -> dy  (dy[0] = p, dy[1] = -dV/dq)
          q += c[j] * dy[0] * dt   =>  q += c[j] * p * dt
          rhs(q_new, p) -> dy
          p += d[j] * dy[1] * dt   =>  p -= d[j] * dV/dq_new * dt

    Parameters
    ----------
    ode : PotentialODE
        The Hamiltonian system.
    tspan : tuple (t0, tf)
        Time interval.
    dt : float
        Time step.
    q0, p0 : float
        Initial conditions.

    Returns
    -------
    t_arr, q_arr, p_arr : numpy arrays
        Solution at native grid points t=0, dt, 2*dt, ..., tf.
    """
    t0, tf = tspan
    n_steps = int(round((tf - t0) / dt))
    t_arr = np.zeros(n_steps + 1, dtype=np.float64)
    q_arr = np.zeros(n_steps + 1, dtype=np.float64)
    p_arr = np.zeros(n_steps + 1, dtype=np.float64)

    t_arr[0] = t0
    q_arr[0] = q0
    p_arr[0] = p0

    q = float(q0)
    p = float(p0)

    for i in range(1, n_steps + 1):
        for j in range(4):
            # Position update: q += c[j] * p * dt (uses current p)
            q = q + YOSHIDA_C[j] * p * dt
            # Momentum update: p -= d[j] * dV/dq * dt (uses UPDATED q)
            p = p - YOSHIDA_D[j] * ode.dVdq(q) * dt
        t_arr[i] = t0 + i * dt
        q_arr[i] = q
        p_arr[i] = p

    return t_arr, q_arr, p_arr


def rk4_solve(ode, tspan, dt, q0, p0):
    """
    Classical 4th-order Runge-Kutta for the Hamiltonian system:
        dq/dt = p
        dp/dt = -dV/dq

    Parameters
    ----------
    ode : PotentialODE
        The Hamiltonian system.
    tspan : tuple (t0, tf)
        Time interval.
    dt : float
        Time step.
    q0, p0 : float
        Initial conditions.

    Returns
    -------
    t_arr, q_arr, p_arr : numpy arrays
        Solution at native grid points t=0, dt, 2*dt, ..., tf.
    """
    t0, tf = tspan
    n_steps = int(round((tf - t0) / dt))
    t_arr = np.zeros(n_steps + 1, dtype=np.float64)
    q_arr = np.zeros(n_steps + 1, dtype=np.float64)
    p_arr = np.zeros(n_steps + 1, dtype=np.float64)

    t_arr[0] = t0
    q_arr[0] = q0
    p_arr[0] = p0

    q = float(q0)
    p = float(p0)

    for i in range(1, n_steps + 1):
        # k1
        dq1 = p
        dp1 = -ode.dVdq(q)

        # k2
        q2 = q + 0.5 * dt * dq1
        p2 = p + 0.5 * dt * dp1
        dq2 = p2
        dp2 = -ode.dVdq(q2)

        # k3
        q3 = q + 0.5 * dt * dq2
        p3 = p + 0.5 * dt * dp2
        dq3 = p3
        dp3 = -ode.dVdq(q3)

        # k4
        q4 = q + dt * dq3
        p4 = p + dt * dp3
        dq4 = p4
        dp4 = -ode.dVdq(q4)

        # Update
        q = q + (dt / 6.0) * (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4)
        p = p + (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)

        t_arr[i] = t0 + i * dt
        q_arr[i] = q
        p_arr[i] = p

    return t_arr, q_arr, p_arr


# =============================================================================
# 2. Helper Functions
# =============================================================================
def compute_hamiltonian(ode, q_arr, p_arr):
    """
    Compute Hamiltonian H(q, p) = p^2/2 + V(q) (vectorized).

    Parameters
    ----------
    ode : PotentialODE
        The Hamiltonian system (provides V(q)).
    q_arr, p_arr : numpy arrays
        Position and momentum arrays.

    Returns
    -------
    H : numpy array
        Hamiltonian values.
    """
    q_arr = np.asarray(q_arr, dtype=np.float64)
    p_arr = np.asarray(p_arr, dtype=np.float64)
    V_vals = ode.V_eval_vec(q_arr)
    return 0.5 * p_arr ** 2 + V_vals


def mse_loss(pred, target):
    """Calculate MSE loss between numpy arrays."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.mean((pred - target) ** 2)


def load_test_data():
    """
    Load test data from data_test/test.parquet.

    Returns
    -------
    V : numpy array, shape (N, 100)
        Potential function values.
    t : numpy array, shape (N, 100)
        Time points.
    q : numpy array, shape (N, 100)
        Position values.
    p : numpy array, shape (N, 100)
        Momentum values.
    ic : numpy array, shape (N, 2)
        Initial conditions [q0, p0].
    """
    df = pl.read_parquet("data_test/test.parquet")
    V = df["V"].to_numpy().reshape(-1, NSENSORS)
    t = df["t"].to_numpy().reshape(-1, NSENSORS)
    q = df["q"].to_numpy().reshape(-1, NSENSORS)
    p = df["p"].to_numpy().reshape(-1, NSENSORS)
    ic = np.stack([q[:, 0], p[:, 0]], axis=1)
    return V, t, q, p, ic


def load_dense_kl8():
    """
    Load dense KL8 data from data_true/test_kl8_dense.parquet.

    Returns
    -------
    t_dense : numpy array, shape (N, 1000)
    q_dense : numpy array, shape (N, 1000)
    p_dense : numpy array, shape (N, 1000)
    Or None if file not found.
    """
    try:
        df = pl.read_parquet("data_true/test_kl8_dense.parquet")
        t_dense = df["t"].to_numpy().reshape(-1, NDENSE)
        q_dense = df["q_true"].to_numpy().reshape(-1, NDENSE)
        p_dense = df["p_true"].to_numpy().reshape(-1, NDENSE)
        return t_dense, q_dense, p_dense
    except FileNotFoundError:
        return None


def load_physical_dense(name):
    """
    Load dense KL8 data for a physical potential.

    Parameters
    ----------
    name : str
        One of: sho, double_well, morse, pendulum, stw, sstw, atw.

    Returns
    -------
    (t_dense, q_dense, p_dense) or None if not found.
    """
    try:
        df = pl.read_parquet(f"data_true/{name}_dense.parquet")
        t_dense = df["t"].to_numpy()
        q_dense = df["q_true"].to_numpy()
        p_dense = df["p_true"].to_numpy()
        return t_dense, q_dense, p_dense
    except FileNotFoundError:
        return None


def load_physical_potentials():
    """
    Load V values for each physical potential from data_analyze/{name}.parquet.

    Returns
    -------
    dict mapping name (str) to V array (numpy array, shape (100,)).
    """
    names = ["sho", "double_well", "morse", "pendulum", "stw", "sstw", "atw"]
    potentials = {}
    for name in names:
        try:
            df = pl.read_parquet(f"data_analyze/{name}.parquet")
            V = df["V"].to_numpy()[:NSENSORS]
            potentials[name] = V
        except FileNotFoundError:
            pass
    return potentials


# =============================================================================
# 3. Approach A: Native Grid Comparison
# =============================================================================
def compare_native_grid(model, device, dt=0.02, variational=False, fig_dir="figs/fair"):
    """
    Fair comparison at the solver's native time grid (no interpolation).

    For each test potential:
    1. Run Y4 and RK4 at dt -> native grid (no spline interpolation)
    2. Interpolate dense KL8 reference to the SAME native grid via PCHIP
    3. Run neural model with native grid time points as input
    4. Compute MSE of each vs KL8

    Parameters
    ----------
    model : torch.nn.Module
        Trained Neural Hamilton model.
    device : str or torch.device
        Device for model inference.
    dt : float
        Time step for solvers.
    variational : bool
        Whether the model is VaRONet.
    fig_dir : str
        Directory for saving figures.

    Returns
    -------
    results : dict
        Contains 'model_losses', 'y4_losses', 'rk4_losses' arrays and timing info.
    """
    os.makedirs(fig_dir, exist_ok=True)

    # Load data
    print("Loading test data...")
    V_all, t_all, q_all, p_all, ic_all = load_test_data()
    n_samples = V_all.shape[0]

    print("Loading dense KL8 reference...")
    dense_data = load_dense_kl8()
    if dense_data is None:
        print("ERROR: Dense KL8 data not found. Cannot perform fair comparison.")
        return None
    t_dense, q_dense, p_dense = dense_data

    # Native grid: t = 0, dt, 2*dt, ...
    # Solvers produce n_steps+1 points, but model is limited to NSENSORS points.
    # Select NSENSORS evenly-spaced points from the solver's native grid.
    tspan = (0.0, 2.0)
    n_solver = int(round(2.0 / dt)) + 1
    n_compare = min(n_solver, NSENSORS)
    compare_idx = np.round(np.linspace(0, n_solver - 1, n_compare)).astype(int)

    model.eval()

    model_losses = []
    y4_losses = []
    rk4_losses = []
    model_times = []
    y4_times = []
    rk4_times = []

    print(f"Running native grid comparison (dt={dt}, {n_compare} points from {n_solver} solver grid)...")
    for i in tqdm(range(n_samples), desc="Native Grid"):
        V_i = V_all[i]
        q0 = ic_all[i, 0]
        p0 = ic_all[i, 1]

        ode = PotentialODE(V_i)

        # Y4 solver at native grid
        t_start = time_module.time()
        t_y4, q_y4, p_y4 = yoshida4_solve(ode, tspan, dt, q0, p0)
        y4_time = time_module.time() - t_start
        y4_times.append(y4_time)

        # RK4 solver at native grid
        t_start = time_module.time()
        _, q_rk4, p_rk4 = rk4_solve(ode, tspan, dt, q0, p0)
        rk4_time = time_module.time() - t_start
        rk4_times.append(rk4_time)

        # Subsample to evenly-spaced comparison points
        t_compare = t_y4[compare_idx]
        q_y4 = q_y4[compare_idx]
        p_y4 = p_y4[compare_idx]
        q_rk4 = q_rk4[compare_idx]
        p_rk4 = p_rk4[compare_idx]

        # KL8 reference at comparison grid (PCHIP interpolation from 1000 dense points)
        kl8_interp_q = PchipInterpolator(t_dense[i], q_dense[i])
        kl8_interp_p = PchipInterpolator(t_dense[i], p_dense[i])
        q_kl8_native = kl8_interp_q(t_compare)
        p_kl8_native = kl8_interp_p(t_compare)

        # Neural model at comparison grid
        V_tensor = torch.tensor(V_i, dtype=torch.float32).unsqueeze(0).to(device)
        t_tensor = torch.tensor(t_compare, dtype=torch.float32).unsqueeze(0).to(device)
        ic_tensor = torch.tensor([[q0, p0]], dtype=torch.float32).to(device)

        t_start = time_module.time()
        with torch.no_grad():
            if not variational:
                q_pred, p_pred = model(V_tensor, t_tensor, ic_tensor)
            else:
                q_pred, p_pred, _, _ = model(V_tensor, t_tensor, ic_tensor)
        model_time = time_module.time() - t_start
        model_times.append(model_time)

        q_model = q_pred.cpu().numpy().flatten().astype(np.float64)
        p_model = p_pred.cpu().numpy().flatten().astype(np.float64)

        # Compute MSE vs KL8
        loss_q_model = mse_loss(q_model, q_kl8_native)
        loss_p_model = mse_loss(p_model, p_kl8_native)
        loss_model = 0.5 * (loss_q_model + loss_p_model)

        loss_q_y4 = mse_loss(q_y4, q_kl8_native)
        loss_p_y4 = mse_loss(p_y4, p_kl8_native)
        loss_y4 = 0.5 * (loss_q_y4 + loss_p_y4)

        loss_q_rk4 = mse_loss(q_rk4, q_kl8_native)
        loss_p_rk4 = mse_loss(p_rk4, p_kl8_native)
        loss_rk4 = 0.5 * (loss_q_rk4 + loss_p_rk4)

        model_losses.append(loss_model)
        y4_losses.append(loss_y4)
        rk4_losses.append(loss_rk4)

    model_losses = np.array(model_losses)
    y4_losses = np.array(y4_losses)
    rk4_losses = np.array(rk4_losses)

    # Print summary
    print("\n--- Approach A: Native Grid Comparison ---")
    print(f"  dt = {dt}, grid points = {n_compare}")
    print(f"  Model  Mean Loss: {model_losses.mean():.4e}  (mean time: {np.mean(model_times):.4e} s)")
    print(f"  Y4     Mean Loss: {y4_losses.mean():.4e}  (mean time: {np.mean(y4_times):.4e} s)")
    print(f"  RK4    Mean Loss: {rk4_losses.mean():.4e}  (mean time: {np.mean(rk4_times):.4e} s)")

    # Save loss data
    df_losses = pl.DataFrame({
        "model_loss": model_losses,
        "y4_loss": y4_losses,
        "rk4_loss": rk4_losses,
    })
    df_losses.write_parquet(f"{fig_dir}/native_grid_losses.parquet")

    # Plot histograms
    _plot_native_grid_histograms(model_losses, y4_losses, rk4_losses, fig_dir)

    results = {
        "model_losses": model_losses,
        "y4_losses": y4_losses,
        "rk4_losses": rk4_losses,
        "model_times": np.array(model_times),
        "y4_times": np.array(y4_times),
        "rk4_times": np.array(rk4_times),
        "dt": dt,
        "n_compare": n_compare,
    }
    return results


def _plot_native_grid_histograms(model_losses, y4_losses, rk4_losses, fig_dir):
    """Generate loss distribution histograms for Approach A (paper quality)."""
    all_losses = np.concatenate([model_losses, y4_losses, rk4_losses])
    all_losses = all_losses[all_losses > 0]
    if len(all_losses) == 0:
        print("Warning: All losses are zero, skipping histogram.")
        return

    loss_min_log = np.floor(np.log10(all_losses.min())) - 0.5
    loss_max_log = np.ceil(np.log10(all_losses.max())) + 0.5
    logbins = np.logspace(loss_min_log, loss_max_log, 60)

    with plt.style.context(["science", "nature"]):
        # Combined histogram (main paper figure)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for name, losses, key in [
            ("Model", model_losses, "model"),
            ("Yoshida4", y4_losses, "y4"),
            ("RK4", rk4_losses, "rk4"),
        ]:
            valid = losses[losses > 0]
            ax.hist(
                valid, bins=logbins, histtype="stepfilled",
                color=COLORS[key], alpha=0.3, edgecolor=COLORS[key], linewidth=0.8,
                label=name,
            )
            ax.axvline(
                losses.mean(), color=COLORS[key], linestyle="--", linewidth=0.8,
            )

        ax.set_xlabel(r"MSE vs KahanLi8")
        ax.set_ylabel("Count")
        ax.set_xscale("log")
        ax.legend(loc="upper left", fontsize=5, frameon=True, fancybox=False, edgecolor="gray")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/native_hist_combined.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Individual histograms
        for name, losses, key in [
            ("Model", model_losses, "model"),
            ("Y4", y4_losses, "y4"),
            ("RK4", rk4_losses, "rk4"),
        ]:
            fig, ax = plt.subplots()
            valid = losses[losses > 0]
            ax.hist(valid, bins=logbins, color=COLORS[key], alpha=0.7, edgecolor="white", linewidth=0.3)
            ax.axvline(losses.mean(), color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel(r"MSE vs KahanLi8")
            ax.set_ylabel("Count")
            ax.set_xscale("log")
            ax.text(
                0.95, 0.95, f"mean: {losses.mean():.2e}",
                transform=ax.transAxes, fontsize=5, ha="right", va="top",
            )
            fig.savefig(f"{fig_dir}/native_hist_{key}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)


# =============================================================================
# 4. Approach B: Work-Precision Diagram
# =============================================================================
def work_precision_diagram(model, device, variational=False, n_samples=100, fig_dir="figs/fair"):
    """
    Work-precision diagram comparing computational cost vs accuracy.

    For multiple dt values, run Y4 and RK4, measure wall-clock time and MSE
    vs KL8. Plot the neural model as a single point.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Neural Hamilton model.
    device : str or torch.device
        Device for model inference.
    variational : bool
        Whether the model is VaRONet.
    n_samples : int
        Number of test potentials to use.
    fig_dir : str
        Directory for saving figures.

    Returns
    -------
    results : dict
        Contains work-precision data for all methods.
    """
    os.makedirs(fig_dir, exist_ok=True)

    # Load data
    print("Loading test data...")
    V_all, t_all, q_all, p_all, ic_all = load_test_data()
    n_samples = min(n_samples, V_all.shape[0])

    print("Loading dense KL8 reference...")
    dense_data = load_dense_kl8()
    if dense_data is None:
        print("ERROR: Dense KL8 data not found. Cannot perform work-precision analysis.")
        return None
    t_dense, q_dense, p_dense = dense_data

    tspan = (0.0, 2.0)
    dt_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    # Build ODE objects and KL8 interpolators once
    print("Building ODE objects and KL8 interpolators...")
    odes = []
    kl8_q_interps = []
    kl8_p_interps = []
    for i in range(n_samples):
        odes.append(PotentialODE(V_all[i]))
        kl8_q_interps.append(PchipInterpolator(t_dense[i], q_dense[i]))
        kl8_p_interps.append(PchipInterpolator(t_dense[i], p_dense[i]))

    # Solver results: for each dt, compute average time and average MSE
    y4_results = {"dt": [], "mean_time": [], "mean_mse": []}
    rk4_results = {"dt": [], "mean_time": [], "mean_mse": []}

    for dt in tqdm(dt_values, desc="Work-Precision (dt sweep)"):
        n_native = int(round(2.0 / dt)) + 1
        t_native = np.linspace(0.0, 2.0, n_native)

        y4_times = []
        y4_mses = []
        rk4_times = []
        rk4_mses = []

        for i in range(n_samples):
            q0 = ic_all[i, 0]
            p0 = ic_all[i, 1]

            # KL8 reference at this native grid
            q_kl8 = kl8_q_interps[i](t_native)
            p_kl8 = kl8_p_interps[i](t_native)

            # Y4
            t_start = time_module.time()
            _, q_y4, p_y4 = yoshida4_solve(odes[i], tspan, dt, q0, p0)
            y4_times.append(time_module.time() - t_start)
            loss_q = mse_loss(q_y4, q_kl8)
            loss_p = mse_loss(p_y4, p_kl8)
            y4_mses.append(0.5 * (loss_q + loss_p))

            # RK4
            t_start = time_module.time()
            _, q_rk4, p_rk4 = rk4_solve(odes[i], tspan, dt, q0, p0)
            rk4_times.append(time_module.time() - t_start)
            loss_q = mse_loss(q_rk4, q_kl8)
            loss_p = mse_loss(p_rk4, p_kl8)
            rk4_mses.append(0.5 * (loss_q + loss_p))

        y4_results["dt"].append(dt)
        y4_results["mean_time"].append(np.mean(y4_times))
        y4_results["mean_mse"].append(np.mean(y4_mses))

        rk4_results["dt"].append(dt)
        rk4_results["mean_time"].append(np.mean(rk4_times))
        rk4_results["mean_mse"].append(np.mean(rk4_mses))

    # Neural model: evaluate at NSENSORS evenly-spaced points (model's max capacity)
    print("Running neural model inference...")
    t_native_model = np.linspace(0.0, 2.0, NSENSORS)

    model.eval()
    model_mses = []
    model_times = []

    for i in tqdm(range(n_samples), desc="Model inference"):
        q0 = ic_all[i, 0]
        p0 = ic_all[i, 1]

        # KL8 reference at finest grid
        q_kl8 = kl8_q_interps[i](t_native_model)
        p_kl8 = kl8_p_interps[i](t_native_model)

        V_tensor = torch.tensor(V_all[i], dtype=torch.float32).unsqueeze(0).to(device)
        t_tensor = torch.tensor(t_native_model, dtype=torch.float32).unsqueeze(0).to(device)
        ic_tensor = torch.tensor([[q0, p0]], dtype=torch.float32).to(device)

        t_start = time_module.time()
        with torch.no_grad():
            if not variational:
                q_pred, p_pred = model(V_tensor, t_tensor, ic_tensor)
            else:
                q_pred, p_pred, _, _ = model(V_tensor, t_tensor, ic_tensor)
        model_times.append(time_module.time() - t_start)

        q_model = q_pred.cpu().numpy().flatten().astype(np.float64)
        p_model = p_pred.cpu().numpy().flatten().astype(np.float64)

        loss_q = mse_loss(q_model, q_kl8)
        loss_p = mse_loss(p_model, p_kl8)
        model_mses.append(0.5 * (loss_q + loss_p))

    model_mean_time = np.mean(model_times)
    model_mean_mse = np.mean(model_mses)

    # Print summary
    print("\n--- Approach B: Work-Precision Diagram ---")
    print("  NOTE: Y4/RK4 timings are from pure Python loops.")
    print("        Model timing uses compiled PyTorch (C++/CUDA).")
    print("        Time comparison reflects implementation, not algorithmic complexity.")
    print(f"  {'Method':<8} {'dt':<10} {'Mean MSE':<14} {'Mean Time (s)':<14}")
    print("  " + "-" * 46)
    for j, dt in enumerate(dt_values):
        print(f"  {'Y4':<8} {dt:<10.4f} {y4_results['mean_mse'][j]:<14.4e} {y4_results['mean_time'][j]:<14.4e}")
    print("  " + "-" * 46)
    for j, dt in enumerate(dt_values):
        print(f"  {'RK4':<8} {dt:<10.4f} {rk4_results['mean_mse'][j]:<14.4e} {rk4_results['mean_time'][j]:<14.4e}")
    print("  " + "-" * 46)
    print(f"  {'Model':<8} {'--':<10} {model_mean_mse:<14.4e} {model_mean_time:<14.4e}")

    # Plot work-precision diagram
    _plot_work_precision(y4_results, rk4_results, model_mean_time, model_mean_mse, fig_dir)

    results = {
        "y4": y4_results,
        "rk4": rk4_results,
        "model_mean_time": model_mean_time,
        "model_mean_mse": model_mean_mse,
        "n_samples": n_samples,
    }
    return results


def _plot_work_precision(y4_results, rk4_results, model_time, model_mse, fig_dir):
    """Generate work-precision diagram (paper quality)."""
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        # Y4 line
        ax.plot(
            y4_results["mean_time"], y4_results["mean_mse"],
            "o-", color=COLORS["y4"], label="Yoshida4", markersize=3, linewidth=0.8,
        )
        # RK4 line
        ax.plot(
            rk4_results["mean_time"], rk4_results["mean_mse"],
            "s-", color=COLORS["rk4"], label="RK4", markersize=3, linewidth=0.8,
        )

        # dt annotations (only a few key ones to avoid clutter)
        key_dts = {0.1, 0.02, 0.001}
        for results, name in [(y4_results, "Y4"), (rk4_results, "RK4")]:
            for j, dt in enumerate(results["dt"]):
                if dt in key_dts:
                    ax.annotate(
                        f"$\\Delta t$={dt}",
                        (results["mean_time"][j], results["mean_mse"][j]),
                        textcoords="offset points", xytext=(5, 3), fontsize=3.5,
                        color="dimgray",
                    )

        # Model star
        ax.plot(
            model_time, model_mse,
            "*", color=COLORS["model"], label="Model", markersize=10, zorder=5,
            markeredgecolor="darkred", markeredgewidth=0.3,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wall-clock time per sample (s)")
        ax.set_ylabel(r"MSE vs KahanLi8")
        ax.legend(loc="upper right", fontsize=5, frameon=True, fancybox=False, edgecolor="gray")
        ax.text(
            0.02, 0.02,
            r"$\dagger$ Y4/RK4: pure Python; Model: PyTorch",
            transform=ax.transAxes, fontsize=3, alpha=0.5, verticalalignment="bottom",
        )
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/work_precision.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 5. Approach C: Energy Conservation Analysis
# =============================================================================
def energy_conservation_analysis(model, device, dt=0.02, variational=False, n_samples=100, fig_dir="figs/fair"):
    """
    Compare energy (Hamiltonian) conservation across methods.

    For each test potential, compute H(q,p) = p^2/2 + V(q) for all methods
    and analyze the deviation from initial energy.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Neural Hamilton model.
    device : str or torch.device
        Device for model inference.
    dt : float
        Time step for solvers.
    variational : bool
        Whether the model is VaRONet.
    n_samples : int
        Number of test potentials to use.
    fig_dir : str
        Directory for saving figures.

    Returns
    -------
    results : dict
        Contains energy conservation statistics for all methods.
    """
    os.makedirs(fig_dir, exist_ok=True)

    # Load data
    print("Loading test data...")
    V_all, t_all, q_all, p_all, ic_all = load_test_data()
    n_samples = min(n_samples, V_all.shape[0])

    print("Loading dense KL8 reference...")
    dense_data = load_dense_kl8()
    if dense_data is None:
        print("ERROR: Dense KL8 data not found. Cannot perform energy analysis.")
        return None
    t_dense, q_dense, p_dense = dense_data

    tspan = (0.0, 2.0)
    n_solver = int(round(2.0 / dt)) + 1
    n_compare = min(n_solver, NSENSORS)
    compare_idx = np.round(np.linspace(0, n_solver - 1, n_compare)).astype(int)

    model.eval()

    # Storage for energy statistics
    stats = {
        "y4": {"max_dH": [], "mean_dH": [], "final_dH": []},
        "rk4": {"max_dH": [], "mean_dH": [], "final_dH": []},
        "kl8": {"max_dH": [], "mean_dH": [], "final_dH": []},
        "model": {"max_dH": [], "mean_dH": [], "final_dH": []},
    }

    # Store one example for time series plot
    example_idx = 0
    example_data = {}

    print(f"Running energy conservation analysis (n_samples={n_samples})...")
    for i in tqdm(range(n_samples), desc="Energy Conservation"):
        V_i = V_all[i]
        q0 = ic_all[i, 0]
        p0 = ic_all[i, 1]

        ode = PotentialODE(V_i)

        # Y4 at native grid (subsample evenly)
        t_y4, q_y4, p_y4 = yoshida4_solve(ode, tspan, dt, q0, p0)
        t_compare = t_y4[compare_idx]
        q_y4 = q_y4[compare_idx]
        p_y4 = p_y4[compare_idx]
        H_y4 = compute_hamiltonian(ode, q_y4, p_y4)

        # RK4 at native grid (subsample evenly)
        _, q_rk4, p_rk4 = rk4_solve(ode, tspan, dt, q0, p0)
        q_rk4 = q_rk4[compare_idx]
        p_rk4 = p_rk4[compare_idx]
        H_rk4 = compute_hamiltonian(ode, q_rk4, p_rk4)

        # KL8 subsampled to comparison grid
        kl8_interp_q = PchipInterpolator(t_dense[i], q_dense[i])
        kl8_interp_p = PchipInterpolator(t_dense[i], p_dense[i])
        q_kl8 = kl8_interp_q(t_compare)
        p_kl8 = kl8_interp_p(t_compare)
        H_kl8 = compute_hamiltonian(ode, q_kl8, p_kl8)

        # Neural model at comparison grid
        V_tensor = torch.tensor(V_i, dtype=torch.float32).unsqueeze(0).to(device)
        t_tensor = torch.tensor(t_compare, dtype=torch.float32).unsqueeze(0).to(device)
        ic_tensor = torch.tensor([[q0, p0]], dtype=torch.float32).to(device)

        with torch.no_grad():
            if not variational:
                q_pred, p_pred = model(V_tensor, t_tensor, ic_tensor)
            else:
                q_pred, p_pred, _, _ = model(V_tensor, t_tensor, ic_tensor)

        q_model = q_pred.cpu().numpy().flatten().astype(np.float64)
        p_model = p_pred.cpu().numpy().flatten().astype(np.float64)
        H_model = compute_hamiltonian(ode, q_model, p_model)

        # Compute Delta H = |H(t) - H(0)|
        for method_name, H_vals in [
            ("y4", H_y4), ("rk4", H_rk4), ("kl8", H_kl8), ("model", H_model)
        ]:
            dH = np.abs(H_vals - H_vals[0])
            stats[method_name]["max_dH"].append(np.max(dH))
            stats[method_name]["mean_dH"].append(np.mean(dH))
            stats[method_name]["final_dH"].append(dH[-1])

        # Store example data (first potential)
        if i == example_idx:
            example_data = {
                "t": t_compare,
                "y4_dH": np.abs(H_y4 - H_y4[0]),
                "rk4_dH": np.abs(H_rk4 - H_rk4[0]),
                "kl8_dH": np.abs(H_kl8 - H_kl8[0]),
                "model_dH": np.abs(H_model - H_model[0]),
            }

    # Convert to arrays
    for method_name in stats:
        for key in stats[method_name]:
            stats[method_name][key] = np.array(stats[method_name][key])

    # Print summary table
    print("\n--- Approach C: Energy Conservation Statistics ---")
    print(f"  {'Method':<8} {'max|dH| (mean)':<18} {'mean|dH| (mean)':<18} {'final|dH| (mean)':<18}")
    print("  " + "-" * 62)
    for method_name in ["model", "y4", "rk4", "kl8"]:
        s = stats[method_name]
        print(
            f"  {method_name.upper():<8} "
            f"{s['max_dH'].mean():<18.4e} "
            f"{s['mean_dH'].mean():<18.4e} "
            f"{s['final_dH'].mean():<18.4e}"
        )

    # Plot: Example Delta H time series
    _plot_energy_example(example_data, fig_dir)

    # Plot: Distribution of max|Delta H|
    _plot_energy_distribution(stats, fig_dir)

    return stats


def _plot_energy_example(example_data, fig_dir):
    """Plot Delta H(t) time series for a single example (paper quality)."""
    if not example_data:
        return

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        t = example_data["t"]
        # Replace exact zeros with NaN to avoid log(0) issues
        for key in ["y4_dH", "rk4_dH", "kl8_dH", "model_dH"]:
            arr = example_data[key]
            arr[arr == 0] = np.nan

        ax.plot(t, example_data["kl8_dH"], color=COLORS["kl8"], label="KL8", linewidth=0.8, linestyle=":", alpha=0.8)
        ax.plot(t, example_data["rk4_dH"], color=COLORS["rk4"], label="RK4", linewidth=0.8)
        ax.plot(t, example_data["model_dH"], color=COLORS["model"], label="Model", linewidth=0.8)
        ax.plot(t, example_data["y4_dH"], color=COLORS["y4"], label="Yoshida4", linewidth=0.8)

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|\Delta H(t)| = |H(t) - H(0)|$")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=5, frameon=True, fancybox=False, edgecolor="gray")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/energy_example.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_energy_distribution(stats, fig_dir):
    """Plot distribution of max|Delta H| across all potentials (paper quality)."""
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        data = [
            stats["kl8"]["max_dH"],
            stats["rk4"]["max_dH"],
            stats["model"]["max_dH"],
            stats["y4"]["max_dH"],
        ]
        labels = ["KL8", "RK4", "Model", "Yoshida4"]
        colors_list = [COLORS["kl8"], COLORS["rk4"], COLORS["model"], COLORS["y4"]]

        positions = list(range(1, len(labels) + 1))

        bp = ax.boxplot(
            data,
            positions=positions,
            labels=labels,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=0.8),
            whiskerprops=dict(linewidth=0.6),
            capprops=dict(linewidth=0.6),
            flierprops=dict(markersize=2, markerfacecolor="gray", markeredgewidth=0.3),
        )

        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
            patch.set_edgecolor(color)

        ax.set_ylabel(r"$\max_t |\Delta H(t)|$")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/energy_distribution.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 6. Symplectic Fragility Verification
# =============================================================================
def verify_symplectic_fragility(fig_dir="figs/fair"):
    """
    Verify Yoshida4 implementation correctness and demonstrate symplectic
    fragility under spline interpolation.

    Compares analytic SHO (V=q^2/2, exact C^inf) vs spline-interpolated SHO
    to isolate the effect of spline discontinuities on symplectic properties.
    """
    os.makedirs(fig_dir, exist_ok=True)

    class AnalyticSHO:
        def dVdq(self, q):
            return q
        def V_eval(self, q):
            return 0.5 * q ** 2
        def V_eval_vec(self, q_arr):
            return 0.5 * np.asarray(q_arr) ** 2

    q0, p0 = 0.5, 0.0
    tspan = (0.0, 2.0)

    dt_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    # Analytic SHO
    analytic_ode = AnalyticSHO()
    # Spline SHO (100 samples of V=q^2/2)
    x_samples = np.linspace(0.0, 1.0, NSENSORS)
    V_samples = 0.5 * x_samples ** 2
    spline_ode = PotentialODE(V_samples)

    H0 = 0.5 * (q0 ** 2 + p0 ** 2)

    results = {"analytic": {"y4": [], "rk4": []}, "spline": {"y4": [], "rk4": []}}
    dH_results = {"analytic": {"y4": [], "rk4": []}, "spline": {"y4": [], "rk4": []}}

    for dt in dt_values:
        for label, ode in [("analytic", analytic_ode), ("spline", spline_ode)]:
            t_y4, q_y4, p_y4 = yoshida4_solve(ode, tspan, dt, q0, p0)
            t_rk4, q_rk4, p_rk4 = rk4_solve(ode, tspan, dt, q0, p0)

            q_exact = q0 * np.cos(t_y4) + p0 * np.sin(t_y4)
            err_y4 = np.max(np.abs(q_y4 - q_exact))
            err_rk4 = np.max(np.abs(q_rk4 - q_exact))
            results[label]["y4"].append(err_y4)
            results[label]["rk4"].append(err_rk4)

            H_y4 = 0.5 * p_y4 ** 2 + ode.V_eval_vec(q_y4)
            H_rk4 = 0.5 * p_rk4 ** 2 + ode.V_eval_vec(q_rk4)
            dH_results[label]["y4"].append(np.max(np.abs(H_y4 - H0)))
            dH_results[label]["rk4"].append(np.max(np.abs(H_rk4 - H0)))

    dt_arr = np.array(dt_values)

    # Print summary
    print("\n--- Symplectic Fragility Verification ---")
    print(f"  {'dt':<10} {'Y4 analytic':<14} {'Y4 spline':<14} {'RK4 analytic':<14} {'RK4 spline':<14}")
    print("  " + "-" * 66)
    for j, dt in enumerate(dt_values):
        print(
            f"  {dt:<10.4f} "
            f"{results['analytic']['y4'][j]:<14.4e} "
            f"{results['spline']['y4'][j]:<14.4e} "
            f"{results['analytic']['rk4'][j]:<14.4e} "
            f"{results['spline']['rk4'][j]:<14.4e}"
        )

    print(f"\n  {'dt':<10} {'Y4 dH anal.':<14} {'Y4 dH spline':<14} {'RK4 dH anal.':<14} {'RK4 dH spline':<14}")
    print("  " + "-" * 66)
    for j, dt in enumerate(dt_values):
        print(
            f"  {dt:<10.4f} "
            f"{dH_results['analytic']['y4'][j]:<14.4e} "
            f"{dH_results['spline']['y4'][j]:<14.4e} "
            f"{dH_results['analytic']['rk4'][j]:<14.4e} "
            f"{dH_results['spline']['rk4'][j]:<14.4e}"
        )

    _plot_symplectic_fragility(dt_arr, results, dH_results, fig_dir)


def _plot_symplectic_fragility(dt_arr, results, dH_results, fig_dir):
    """Generate symplectic fragility comparison plots (paper quality)."""
    with plt.style.context(["science", "nature"]):
        # Figure 1: Convergence order (trajectory error vs dt)
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

        ax = axes[0]
        ax.loglog(dt_arr, results["analytic"]["y4"], "o-", color=COLORS["y4"], markersize=3, linewidth=0.8, label="Y4 (analytic)")
        ax.loglog(dt_arr, results["analytic"]["rk4"], "s-", color=COLORS["rk4"], markersize=3, linewidth=0.8, label="RK4 (analytic)")
        ax.loglog(dt_arr, results["spline"]["y4"], "o--", color=COLORS["y4"], markersize=3, linewidth=0.8, alpha=0.5, label="Y4 (spline)")
        ax.loglog(dt_arr, results["spline"]["rk4"], "s--", color=COLORS["rk4"], markersize=3, linewidth=0.8, alpha=0.5, label="RK4 (spline)")
        # Reference slope dt^4
        ref = dt_arr ** 4 * (results["analytic"]["rk4"][0] / dt_arr[0] ** 4)
        ax.loglog(dt_arr, ref, "k:", linewidth=0.5, alpha=0.4, label=r"$\mathcal{O}(\Delta t^4)$")
        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel(r"$\max_t |q(t) - q_{\mathrm{exact}}(t)|$")
        ax.legend(loc="lower right", fontsize=4, frameon=True, fancybox=False, edgecolor="gray")
        ax.set_title("(a) Trajectory Error", fontsize=6)

        # Figure 2: Energy conservation vs dt
        ax = axes[1]
        ax.loglog(dt_arr, dH_results["analytic"]["y4"], "o-", color=COLORS["y4"], markersize=3, linewidth=0.8, label="Y4 (analytic)")
        ax.loglog(dt_arr, dH_results["analytic"]["rk4"], "s-", color=COLORS["rk4"], markersize=3, linewidth=0.8, label="RK4 (analytic)")
        ax.loglog(dt_arr, dH_results["spline"]["y4"], "o--", color=COLORS["y4"], markersize=3, linewidth=0.8, alpha=0.5, label="Y4 (spline)")
        ax.loglog(dt_arr, dH_results["spline"]["rk4"], "s--", color=COLORS["rk4"], markersize=3, linewidth=0.8, alpha=0.5, label="RK4 (spline)")
        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel(r"$\max_t |\Delta H(t)|$")
        ax.legend(loc="lower right", fontsize=4, frameon=True, fancybox=False, edgecolor="gray")
        ax.set_title("(b) Energy Conservation", fontsize=6)

        fig.tight_layout()
        fig.savefig(f"{fig_dir}/symplectic_fragility.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 7. Approach D: Long-term Autoregressive Prediction Comparison
# =============================================================================
NDENSE_LONGTERM = 10000
PHYSICAL_POTENTIALS = ["sho", "double_well", "morse", "pendulum", "stw", "sstw", "atw"]
PHYSICAL_LABELS = {
    "sho": "SHO",
    "double_well": "Double Well",
    "morse": "Morse",
    "pendulum": "Pendulum",
    "stw": "STW",
    "sstw": "SSTW",
    "atw": "ATW",
}


def _autoregressive_rollout(model, V_tensor, n_windows, window_size, dt, device, variational):
    """
    Autoregressive model rollout for long-term prediction.

    Predicts trajectory window by window, using the last (q, p) from each
    window as the initial condition for the next.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Neural Hamilton model.
    V_tensor : torch.Tensor, shape (1, 100)
        Potential function values.
    n_windows : int
        Number of prediction windows.
    window_size : int
        Number of time points per window.
    dt : float
        Time step between points.
    device : str or torch.device
        Device for model inference.
    variational : bool
        Whether the model is VaRONet.

    Returns
    -------
    q_full, p_full, t_full : numpy float64 arrays
        Concatenated trajectory across all windows.
    """
    model.eval()
    V_tensor = V_tensor.to(device)

    q_full = []
    p_full = []
    t_full = []

    ic = torch.tensor([[0.0, 0.0]], device=device)
    t_current = 0.0

    # Match training data grid: linspace(0, 2.0, 100) where dt = 2/(N-1)
    window_T = window_size * dt

    with torch.no_grad():
        for i in range(n_windows):
            t_window = torch.linspace(0, window_T, window_size, device=device)
            t_window = t_window.unsqueeze(0)

            if not variational:
                q_pred, p_pred = model(V_tensor, t_window, ic)
            else:
                q_pred, p_pred, _, _ = model(V_tensor, t_window, ic)

            if i == 0:
                q_full.append(q_pred.cpu())
                p_full.append(p_pred.cpu())
                t_full.append(t_window.cpu() + t_current)
            else:
                q_full.append(q_pred[:, 1:].cpu())
                p_full.append(p_pred[:, 1:].cpu())
                t_full.append((t_window[:, 1:] + t_current).cpu())

            ic = torch.stack([q_pred[:, -1], p_pred[:, -1]], dim=-1)
            t_current += window_T

    q_full = torch.cat(q_full, dim=1).squeeze(0).numpy().astype(np.float64)
    p_full = torch.cat(p_full, dim=1).squeeze(0).numpy().astype(np.float64)
    t_full = torch.cat(t_full, dim=1).squeeze(0).numpy().astype(np.float64)

    return q_full, p_full, t_full


def load_longterm_kl8(name):
    """
    Load long-term KL8 reference from data_true/{name}_longterm_dense.parquet.

    Parameters
    ----------
    name : str
        Physical potential name (e.g., 'sho', 'double_well').

    Returns
    -------
    (t_dense, q_dense, p_dense) : tuple of numpy float64 arrays, or None.
    """
    try:
        df = pl.read_parquet(f"data_true/{name}_longterm_dense.parquet")
        t_dense = df["t"].to_numpy().astype(np.float64)
        q_dense = df["q_true"].to_numpy().astype(np.float64)
        p_dense = df["p_true"].to_numpy().astype(np.float64)
        return t_dense, q_dense, p_dense
    except FileNotFoundError:
        return None


def longterm_comparison(model, device, n_windows=10, window_size=100, dt=0.02,
                        variational=False, fig_dir="figs/fair"):
    """
    Compare long-term autoregressive model prediction vs continuous Y4/RK4
    integration, using KL8 (T=100s) as reference.

    For each of 7 physical potentials:
    1. Load KL8 long-term reference (T=100s, 10000 dense points)
    2. Run model autoregressive rollout (n_windows windows)
    3. Run Y4/RK4 continuously over the full time span
    4. Align all methods to model's time grid via PCHIP interpolation
    5. Compute cumulative MSE and energy drift metrics
    6. Generate per-potential 4-panel figures and summary figure

    Parameters
    ----------
    model : torch.nn.Module
        Trained Neural Hamilton model.
    device : str or torch.device
        Device for model inference.
    n_windows : int
        Number of autoregressive windows (T_max ~ n_windows * (window_size-1) * dt).
    window_size : int
        Number of time points per window.
    dt : float
        Time step between points.
    variational : bool
        Whether the model is VaRONet.
    fig_dir : str
        Directory for saving figures.
    """
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("  APPROACH D: Long-term Autoregressive Comparison")
    print("=" * 60)

    # Load physical potential V values
    phys_potentials = load_physical_potentials()
    if not phys_potentials:
        print("ERROR: No physical potential data found in data_analyze/.")
        return

    window_T = window_size * dt
    t_max_model = n_windows * window_T
    print(f"  Model T_max: {t_max_model:.2f}s ({n_windows} windows x {window_T:.2f}s)")

    # Checkpoint windows for summary
    checkpoints = [w for w in [2, 4, 6, 10] if w <= n_windows]

    # Storage for summary across potentials
    summary_data = {
        "model": {cp: [] for cp in checkpoints},
        "y4": {cp: [] for cp in checkpoints},
        "rk4": {cp: [] for cp in checkpoints},
    }
    summary_maxdH = {
        "model": {cp: [] for cp in checkpoints},
        "y4": {cp: [] for cp in checkpoints},
        "rk4": {cp: [] for cp in checkpoints},
    }

    model.eval()

    print(f"\n--- Per-potential results ---")
    print(f"  {'Potential':<14} {'Model MSE':<14} {'Y4 MSE':<14} {'RK4 MSE':<14} {'Model maxdH':<14} {'Y4 maxdH':<14} {'RK4 maxdH':<14}")
    print("  " + "-" * 84)

    for name in PHYSICAL_POTENTIALS:
        if name not in phys_potentials:
            print(f"  {PHYSICAL_LABELS.get(name, name):<14} SKIPPED (no V data)")
            continue

        # Load KL8 reference
        kl8_data = load_longterm_kl8(name)
        if kl8_data is None:
            print(f"  {PHYSICAL_LABELS.get(name, name):<14} SKIPPED (no KL8 longterm data)")
            continue

        t_kl8, q_kl8, p_kl8 = kl8_data

        V_values = phys_potentials[name]
        V_tensor = torch.tensor(V_values, dtype=torch.float32).unsqueeze(0)

        # 1. Model autoregressive rollout
        q_model, p_model, t_model = _autoregressive_rollout(
            model, V_tensor, n_windows, window_size, dt, device, variational
        )

        # 2. Y4 and RK4 continuous integration over full model time span
        ode = PotentialODE(V_values)
        tspan_full = (0.0, t_model[-1])

        t_y4, q_y4, p_y4 = yoshida4_solve(ode, tspan_full, dt, 0.0, 0.0)
        t_rk4, q_rk4, p_rk4 = rk4_solve(ode, tspan_full, dt, 0.0, 0.0)

        # 3. Align all to model's time grid via PCHIP
        kl8_interp_q = PchipInterpolator(t_kl8, q_kl8)
        kl8_interp_p = PchipInterpolator(t_kl8, p_kl8)
        # Clamp model times to KL8 range
        t_eval = np.clip(t_model, t_kl8[0], t_kl8[-1])
        q_kl8_aligned = kl8_interp_q(t_eval)
        p_kl8_aligned = kl8_interp_p(t_eval)

        y4_interp_q = PchipInterpolator(t_y4, q_y4)
        y4_interp_p = PchipInterpolator(t_y4, p_y4)
        q_y4_aligned = y4_interp_q(t_eval)
        p_y4_aligned = y4_interp_p(t_eval)

        rk4_interp_q = PchipInterpolator(t_rk4, q_rk4)
        rk4_interp_p = PchipInterpolator(t_rk4, p_rk4)
        q_rk4_aligned = rk4_interp_q(t_eval)
        p_rk4_aligned = rk4_interp_p(t_eval)

        # 4. Compute metrics
        # Cumulative MSE at each time point
        cum_mse_model = np.cumsum((q_model - q_kl8_aligned) ** 2 + (p_model - p_kl8_aligned) ** 2) / (2.0 * np.arange(1, len(t_model) + 1))
        cum_mse_y4 = np.cumsum((q_y4_aligned - q_kl8_aligned) ** 2 + (p_y4_aligned - p_kl8_aligned) ** 2) / (2.0 * np.arange(1, len(t_model) + 1))
        cum_mse_rk4 = np.cumsum((q_rk4_aligned - q_kl8_aligned) ** 2 + (p_rk4_aligned - p_kl8_aligned) ** 2) / (2.0 * np.arange(1, len(t_model) + 1))

        # Energy drift |ΔH(t)|
        # NOTE: All methods are evaluated using the same spline-based V(q) via
        # compute_hamiltonian(ode, ...). KL8 trajectories were generated with
        # analytic dV/dq in Julia, so dH_kl8 here reflects spline approximation
        # mismatch rather than integration error. This is consistent with
        # Approach C and ensures all methods are compared on the same footing.
        H_kl8 = compute_hamiltonian(ode, q_kl8_aligned, p_kl8_aligned)
        H_model = compute_hamiltonian(ode, q_model, p_model)
        H_y4 = compute_hamiltonian(ode, q_y4_aligned, p_y4_aligned)
        H_rk4 = compute_hamiltonian(ode, q_rk4_aligned, p_rk4_aligned)

        dH_model = np.abs(H_model - H_model[0])
        dH_y4 = np.abs(H_y4 - H_y4[0])
        dH_rk4 = np.abs(H_rk4 - H_rk4[0])

        # Print per-potential summary
        final_mse_model = cum_mse_model[-1]
        final_mse_y4 = cum_mse_y4[-1]
        final_mse_rk4 = cum_mse_rk4[-1]
        max_dH_model = np.max(dH_model)
        max_dH_y4 = np.max(dH_y4)
        max_dH_rk4 = np.max(dH_rk4)

        label = PHYSICAL_LABELS.get(name, name)
        print(f"  {label:<14} {final_mse_model:<14.4e} {final_mse_y4:<14.4e} {final_mse_rk4:<14.4e} {max_dH_model:<14.4e} {max_dH_y4:<14.4e} {max_dH_rk4:<14.4e}")

        # Collect checkpoint data for summary
        for cp in checkpoints:
            # Number of time points at this checkpoint
            n_pts_cp = cp * (window_size - 1) + 1  # first window full, rest skip 1 overlap
            n_pts_cp = min(n_pts_cp, len(t_model))
            idx_cp = n_pts_cp - 1

            summary_data["model"][cp].append(cum_mse_model[idx_cp])
            summary_data["y4"][cp].append(cum_mse_y4[idx_cp])
            summary_data["rk4"][cp].append(cum_mse_rk4[idx_cp])

            summary_maxdH["model"][cp].append(np.max(dH_model[:idx_cp + 1]))
            summary_maxdH["y4"][cp].append(np.max(dH_y4[:idx_cp + 1]))
            summary_maxdH["rk4"][cp].append(np.max(dH_rk4[:idx_cp + 1]))

        # 5. Plot per-potential 4-panel figure
        _plot_longterm_single(
            name, t_model, t_kl8, q_kl8, p_kl8,
            q_model, p_model, q_y4_aligned, p_y4_aligned, q_rk4_aligned, p_rk4_aligned,
            cum_mse_model, cum_mse_y4, cum_mse_rk4,
            dH_model, dH_y4, dH_rk4,
            fig_dir,
        )

    # 6. Summary figure
    _plot_longterm_summary(summary_data, summary_maxdH, checkpoints, window_size, dt, fig_dir)

    print(f"\n  Figures saved to: {fig_dir}/")
    print("=" * 60)


def _plot_longterm_single(name, t_model, t_kl8, q_kl8, p_kl8,
                          q_model, p_model, q_y4, p_y4, q_rk4, p_rk4,
                          cum_mse_model, cum_mse_y4, cum_mse_rk4,
                          dH_model, dH_y4, dH_rk4, fig_dir):
    """Generate a 4-panel figure for a single potential's long-term comparison."""
    label = PHYSICAL_LABELS.get(name, name)

    # Clip KL8 reference to model's time range for plotting
    t_max = t_model[-1]
    kl8_mask = t_kl8 <= t_max
    t_kl8_clip = t_kl8[kl8_mask]
    q_kl8_clip = q_kl8[kl8_mask]
    p_kl8_clip = p_kl8[kl8_mask]

    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(2, 2, figsize=(7, 5))

        # (a) q(t) trajectory
        ax = axes[0, 0]
        ax.plot(t_kl8_clip, q_kl8_clip, color=COLORS["kl8"], linewidth=0.5, alpha=0.6, label="KL8", zorder=0)
        ax.plot(t_model, q_y4, color=COLORS["y4"], linewidth=0.5, alpha=0.7, label="Y4", zorder=1)
        ax.plot(t_model, q_rk4, color=COLORS["rk4"], linewidth=0.5, alpha=0.7, label="RK4", zorder=1)
        ax.plot(t_model, q_model, color=COLORS["model"], linewidth=0.5, alpha=0.8, label="Model", zorder=2)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$q(t)$")
        ax.set_title("(a) Position trajectory", fontsize=6)
        ax.legend(loc="best", fontsize=4, frameon=True, fancybox=False, edgecolor="gray")

        # (b) Phase space (q vs p)
        ax = axes[0, 1]
        ax.plot(q_kl8_clip, p_kl8_clip, color=COLORS["kl8"], linewidth=0.3, alpha=0.5, label="KL8", zorder=0)
        ax.plot(q_y4, p_y4, color=COLORS["y4"], linewidth=0.3, alpha=0.6, label="Y4", zorder=1)
        ax.plot(q_rk4, p_rk4, color=COLORS["rk4"], linewidth=0.3, alpha=0.6, label="RK4", zorder=1)
        ax.plot(q_model, p_model, color=COLORS["model"], linewidth=0.3, alpha=0.7, label="Model", zorder=2)
        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$p$")
        ax.set_title("(b) Phase space", fontsize=6)
        ax.legend(loc="best", fontsize=4, frameon=True, fancybox=False, edgecolor="gray")

        # (c) |ΔH(t)| energy drift (log scale)
        ax = axes[1, 0]
        # Replace zeros with NaN for log scale
        dH_model_plot = dH_model.copy()
        dH_y4_plot = dH_y4.copy()
        dH_rk4_plot = dH_rk4.copy()
        dH_model_plot[dH_model_plot == 0] = np.nan
        dH_y4_plot[dH_y4_plot == 0] = np.nan
        dH_rk4_plot[dH_rk4_plot == 0] = np.nan

        ax.plot(t_model, dH_y4_plot, color=COLORS["y4"], linewidth=0.5, alpha=0.7, label="Y4")
        ax.plot(t_model, dH_rk4_plot, color=COLORS["rk4"], linewidth=0.5, alpha=0.7, label="RK4")
        ax.plot(t_model, dH_model_plot, color=COLORS["model"], linewidth=0.5, alpha=0.8, label="Model")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|\Delta H(t)|$")
        ax.set_yscale("log")
        ax.set_title("(c) Energy drift", fontsize=6)
        ax.legend(loc="best", fontsize=4, frameon=True, fancybox=False, edgecolor="gray")

        # (d) Cumulative MSE vs t
        ax = axes[1, 1]
        ax.plot(t_model, cum_mse_y4, color=COLORS["y4"], linewidth=0.8, alpha=0.7, label="Y4")
        ax.plot(t_model, cum_mse_rk4, color=COLORS["rk4"], linewidth=0.8, alpha=0.7, label="RK4")
        ax.plot(t_model, cum_mse_model, color=COLORS["model"], linewidth=0.8, alpha=0.8, label="Model")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("Cumulative MSE vs KL8")
        ax.set_yscale("log")
        ax.set_title("(d) Error accumulation", fontsize=6)
        ax.legend(loc="best", fontsize=4, frameon=True, fancybox=False, edgecolor="gray")

        fig.suptitle(f"Long-term comparison: {label}", fontsize=8, y=1.02)
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/longterm_{name}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def _plot_longterm_summary(summary_data, summary_maxdH, checkpoints, window_size, dt, fig_dir):
    """Generate summary figure showing average metrics across all potentials at checkpoints."""
    if not checkpoints:
        return

    # Convert checkpoint windows to approximate time in seconds
    checkpoint_times = [cp * (window_size - 1) * dt for cp in checkpoints]

    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

        # (a) Mean cumulative MSE growth at checkpoints
        ax = axes[0]
        for method, key, marker in [("Model", "model", "o"), ("Y4", "y4", "s"), ("RK4", "rk4", "^")]:
            means = [np.mean(summary_data[key][cp]) for cp in checkpoints]
            ax.plot(
                checkpoint_times, means,
                f"{marker}-", color=COLORS[key], markersize=4, linewidth=0.8,
                label=method,
            )
        ax.set_xlabel(r"$T$ (s)")
        ax.set_ylabel("Mean cumulative MSE vs KL8")
        ax.set_yscale("log")
        ax.set_title("(a) MSE growth", fontsize=6)
        ax.legend(loc="best", fontsize=5, frameon=True, fancybox=False, edgecolor="gray")

        # (b) Mean max|ΔH| growth at checkpoints
        ax = axes[1]
        for method, key, marker in [("Model", "model", "o"), ("Y4", "y4", "s"), ("RK4", "rk4", "^")]:
            means = [np.mean(summary_maxdH[key][cp]) for cp in checkpoints]
            ax.plot(
                checkpoint_times, means,
                f"{marker}-", color=COLORS[key], markersize=4, linewidth=0.8,
                label=method,
            )
        ax.set_xlabel(r"$T$ (s)")
        ax.set_ylabel(r"Mean $\max_t |\Delta H|$")
        ax.set_yscale("log")
        ax.set_title(r"(b) Energy drift growth", fontsize=6)
        ax.legend(loc="best", fontsize=5, frameon=True, fancybox=False, edgecolor="gray")

        fig.suptitle("Long-term comparison summary (averaged over potentials)", fontsize=7, y=1.02)
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/longterm_summary.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# 8. Main Entry
# =============================================================================
def run_fair_comparison(model, device, fig_dir="figs/fair", variational=False):
    """
    Run all three fair comparison approaches.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Neural Hamilton model.
    device : str or torch.device
        Device for model inference.
    fig_dir : str
        Directory for saving figures.
    variational : bool
        Whether the model is VaRONet.
    """
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("  FAIR COMPARISON: Neural Hamilton vs ODE Solvers")
    print("=" * 60)

    # Verification: Symplectic Fragility
    print("\n[V] Symplectic Fragility Verification")
    print("-" * 40)
    verify_symplectic_fragility(fig_dir=fig_dir)

    # Approach A: Native Grid Comparison
    print("\n[A] Native Grid Comparison")
    print("-" * 40)
    results_a = compare_native_grid(
        model, device, dt=0.02, variational=variational, fig_dir=fig_dir,
    )

    # Approach B: Work-Precision Diagram
    print("\n[B] Work-Precision Diagram")
    print("-" * 40)
    results_b = work_precision_diagram(
        model, device, variational=variational, n_samples=100, fig_dir=fig_dir,
    )

    # Approach C: Energy Conservation
    print("\n[C] Energy Conservation Analysis")
    print("-" * 40)
    results_c = energy_conservation_analysis(
        model, device, dt=0.02, variational=variational, n_samples=100, fig_dir=fig_dir,
    )

    # Combined summary
    print("\n" + "=" * 60)
    print("  COMBINED SUMMARY")
    print("=" * 60)

    if results_a is not None:
        print(f"\n  [A] Native Grid (dt=0.02, {results_a['n_compare']} pts):")
        print(f"      Model  MSE: {results_a['model_losses'].mean():.4e}")
        print(f"      Y4     MSE: {results_a['y4_losses'].mean():.4e}")
        print(f"      RK4    MSE: {results_a['rk4_losses'].mean():.4e}")

    if results_b is not None:
        print(f"\n  [B] Work-Precision ({results_b['n_samples']} samples):")
        print(f"      Model  MSE: {results_b['model_mean_mse']:.4e}  Time: {results_b['model_mean_time']:.4e} s")
        # Show best Y4 and RK4 (smallest dt)
        best_y4_idx = -1  # last element = smallest dt
        best_rk4_idx = -1
        print(f"      Y4     MSE: {results_b['y4']['mean_mse'][best_y4_idx]:.4e}  Time: {results_b['y4']['mean_time'][best_y4_idx]:.4e} s  (dt={results_b['y4']['dt'][best_y4_idx]})")
        print(f"      RK4    MSE: {results_b['rk4']['mean_mse'][best_rk4_idx]:.4e}  Time: {results_b['rk4']['mean_time'][best_rk4_idx]:.4e} s  (dt={results_b['rk4']['dt'][best_rk4_idx]})")

    if results_c is not None:
        print(f"\n  [C] Energy Conservation ({len(results_c['model']['max_dH'])} samples):")
        print(f"      Model  mean max|dH|: {results_c['model']['max_dH'].mean():.4e}")
        print(f"      Y4     mean max|dH|: {results_c['y4']['max_dH'].mean():.4e}")
        print(f"      RK4    mean max|dH|: {results_c['rk4']['max_dH'].mean():.4e}")
        print(f"      KL8    mean max|dH|: {results_c['kl8']['max_dH'].mean():.4e}")

    print("\n" + "=" * 60)
    print(f"  Figures saved to: {fig_dir}/")
    print("=" * 60)

    return results_a, results_b, results_c
