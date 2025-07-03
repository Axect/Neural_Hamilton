import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import scienceplots
import polars as pl
import beaupy
from rich.console import Console

import os
import time

from util import (
    select_project,
    select_group,
    select_seed,
    select_device,
    load_model,
    load_data,
    load_study,
    load_best_model,
    log_cosh_loss,
    np_log_cosh_loss
)


torch.set_float32_matmul_precision("medium")
criterion = log_cosh_loss


def load_relevant_data(potential: str):
    # Load existing data (Y4, RK4)
    file_name = f"./data_analyze/{potential}.parquet"
    df = pl.read_parquet(file_name)
    V = df["V"].to_numpy()
    t = df["t"].to_numpy()
    q = df[f"q"].to_numpy()
    p = df[f"p"].to_numpy()
    ds = TensorDataset(
        torch.tensor(V, dtype=torch.float32).unsqueeze(0),
        torch.tensor(t, dtype=torch.float32).unsqueeze(0),
        torch.tensor(q, dtype=torch.float32).unsqueeze(0),
        torch.tensor(p, dtype=torch.float32).unsqueeze(0),
    )

    q_rk4 = df[f"q_rk4"].to_numpy()
    p_rk4 = df[f"p_rk4"].to_numpy()
    ds_rk4 = TensorDataset(
        torch.tensor(V, dtype=torch.float32).unsqueeze(0),
        torch.tensor(t, dtype=torch.float32).unsqueeze(0),
        torch.tensor(q_rk4, dtype=torch.float32).unsqueeze(0),
        torch.tensor(p_rk4, dtype=torch.float32).unsqueeze(0),
    )

    # Load true data (high-order symplectic integrator results from Julia)
    true_file_name = f"./data_true/{potential}.parquet"
    df_true = pl.read_parquet(true_file_name)
    q_true = df_true["q_true"].to_numpy()
    p_true = df_true["p_true"].to_numpy()
    ds_true = TensorDataset(
        torch.tensor(V, dtype=torch.float32).unsqueeze(0),
        torch.tensor(t, dtype=torch.float32).unsqueeze(0),
        torch.tensor(q_true, dtype=torch.float32).unsqueeze(0),
        torch.tensor(p_true, dtype=torch.float32).unsqueeze(0),
    )
    return ds_true, ds, ds_rk4  # True, Y4, RK4


def load_test_data_with_true():
    """
    Load test data with KahanLi8 reference data
    """
    # Load Y4 test data
    y4_file_name = "./data_test/test.parquet"
    df_y4 = pl.read_parquet(y4_file_name)
    V = df_y4["V"].to_numpy()
    t = df_y4["t"].to_numpy()
    q_y4 = df_y4["q"].to_numpy()
    p_y4 = df_y4["p"].to_numpy()

    # Load KahanLi8 reference data
    true_file_name = "./data_true/test_kl8.parquet"
    df_true = pl.read_parquet(true_file_name)
    q_true = df_true["q_true"].to_numpy()
    p_true = df_true["p_true"].to_numpy()

    # Load RK4 test data
    rk4_file_name = "./data_analyze/rk4.parquet"
    df_rk4 = pl.read_parquet(rk4_file_name)
    q_rk4 = df_rk4["q"].to_numpy()
    p_rk4 = df_rk4["p"].to_numpy()

    # Create TensorDatasets
    ds_true = TensorDataset(
        torch.tensor(V, dtype=torch.float32).reshape(-1, NSENSORS),
        torch.tensor(t, dtype=torch.float32).reshape(-1, NSENSORS),
        torch.tensor(q_true, dtype=torch.float32).reshape(-1, NSENSORS),
        torch.tensor(p_true, dtype=torch.float32).reshape(-1, NSENSORS),
    )

    ds_y4 = TensorDataset(
        torch.tensor(V, dtype=torch.float32).reshape(-1, NSENSORS),
        torch.tensor(t, dtype=torch.float32).reshape(-1, NSENSORS),
        torch.tensor(q_y4, dtype=torch.float32).reshape(-1, NSENSORS),
        torch.tensor(p_y4, dtype=torch.float32).reshape(-1, NSENSORS),
    )

    ds_rk4 = TensorDataset(
        torch.tensor(V, dtype=torch.float32).reshape(-1, NSENSORS),
        torch.tensor(t, dtype=torch.float32).reshape(-1, NSENSORS),
        torch.tensor(q_rk4, dtype=torch.float32).reshape(-1, NSENSORS),
        torch.tensor(p_rk4, dtype=torch.float32).reshape(-1, NSENSORS),
    )

    return ds_true, ds_y4, ds_rk4


# RK4 Solver
NSENSORS = 100  # Global constant needed for dataset reshaping


class TestResults:
    def __init__(self, model, dl_val, device, variational=False, precise=False):
        self.model = model
        if precise:
            self.dl_val = None
        else:
            self.dl_val = dl_val
        self.device = device
        self.variational = variational

        if precise:
            self.run_precise_test()
        else:
            self.run_test()
            self.load_rk4()

    def run_test(self):
        self.model.eval()

        V_vec = []
        t_vec = []
        q_preds = []
        p_preds = []
        q_targets = []
        p_targets = []

        total_loss_q_vec = []
        total_loss_p_vec = []
        total_loss_vec = []
        total_time_vec = []

        with torch.no_grad():
            for V, t, q, p in self.dl_val:
                V, t, q, p = (
                    V.to(self.device),
                    t.to(self.device),
                    q.to(self.device),
                    p.to(self.device),
                )

                t_start = time.time()
                if not self.variational:
                    self.reparameterize = False
                    q_pred, p_pred = self.model(V, t)
                else:
                    q_pred, p_pred, _, _ = self.model(V, t)
                t_total = (time.time() - t_start) / V.shape[0]

                loss_q_vec = criterion(q_pred, q, reduction="none")
                loss_p_vec = criterion(p_pred, p, reduction="none")
                loss_vec = 0.5 * (loss_q_vec + loss_p_vec)

                loss_q = loss_q_vec.mean(dim=1)
                loss_p = loss_p_vec.mean(dim=1)
                loss = loss_vec.mean(dim=1)

                V_vec.extend(V.cpu().numpy())
                t_vec.extend(t.cpu().numpy())
                q_preds.extend(q_pred.cpu().numpy())
                p_preds.extend(p_pred.cpu().numpy())
                q_targets.extend(q.cpu().numpy())
                p_targets.extend(p.cpu().numpy())
                total_loss_vec.extend(loss.cpu().numpy())
                total_loss_q_vec.extend(loss_q.cpu().numpy())
                total_loss_p_vec.extend(loss_p.cpu().numpy())
                total_time_vec.extend([t_total] * V.shape[0])

        self.total_loss_vec = np.array(total_loss_vec)
        self.total_loss_q_vec = np.array(total_loss_q_vec)
        self.total_loss_p_vec = np.array(total_loss_p_vec)
        self.total_time_vec = np.array(total_time_vec)
        self.V_vec = np.array(V_vec)
        self.t_vec = np.array(t_vec)
        self.q_preds = np.array(q_preds)
        self.p_preds = np.array(p_preds)
        self.q_targets = np.array(q_targets)
        self.p_targets = np.array(p_targets)

    def load_rk4(self):
        df_rk4 = pl.read_parquet("./data_analyze/rk4.parquet")
        self.rk4_q = df_rk4["q"].to_numpy().reshape(-1, 100)
        self.rk4_p = df_rk4["p"].to_numpy().reshape(-1, 100)
        self.rk4_loss_q = df_rk4["loss_q"].to_numpy().reshape(-1, 100).mean(axis=1)
        self.rk4_loss_p = df_rk4["loss_p"].to_numpy().reshape(-1, 100).mean(axis=1)
        self.rk4_loss = df_rk4["loss"].to_numpy().reshape(-1, 100).mean(axis=1)

    def print_results(self):
        print(f"Shape: {self.total_loss_vec.shape}")
        print(f"Total Loss: {self.total_loss_vec.mean():.4e}")
        print(f"Total Loss q: {self.total_loss_q_vec.mean():.4e}")
        print(f"Total Loss p: {self.total_loss_p_vec.mean():.4e}")

    def hist_loss(self, name: str):
        losses = self.total_loss_vec
        times = self.total_time_vec
        df_losses = pl.DataFrame({"loss": losses, "time": times})
        df_losses.write_parquet(f"{name}.parquet")
        loss_min_log = np.log10(losses.min())
        loss_max_log = np.log10(losses.max())
        if loss_min_log < 0:
            loss_min_log *= 1.01
        else:
            loss_min_log *= 0.99
        if loss_max_log < 0:
            loss_max_log *= 0.99
        else:
            loss_max_log *= 1.01
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            logbins = np.logspace(loss_min_log, loss_max_log, 100)
            ax.hist(losses, bins=logbins)
            ax.axvline(losses.mean(), color="red", linestyle="--")
            ax.set_xlabel("Total Loss")
            ax.set_ylabel("Count")
            ax.set_xscale("log")
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def hist_loss_rk4(self, name: str):
        losses = self.rk4_loss
        df_losses = pl.DataFrame({"loss": losses, "time": times})
        df_losses.write_parquet(f"{name}.parquet")
        loss_min_log = np.log10(losses.min())
        loss_max_log = np.log10(losses.max())
        if loss_min_log < 0:
            loss_min_log *= 1.01
        else:
            loss_min_log *= 0.99
        if loss_max_log < 0:
            loss_max_log *= 0.99
        else:
            loss_max_log *= 1.01
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            logbins = np.logspace(loss_min_log, loss_max_log, 100)
            ax.hist(losses, bins=logbins)
            ax.axvline(losses.mean(), color="red", linestyle="--")
            ax.set_xlabel("Total Loss")
            ax.set_ylabel("Count")
            ax.set_xscale("log")
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_V(self, name: str, index: int):
        q = np.linspace(0, 1, 100)
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(q, self.V_vec[index])
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$V(q)$")
            ax.autoscale(tight=True)
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_q(self, name: str, index: int):
        t = self.t_vec[index]
        loss_q = self.total_loss_q_vec[index]
        cmap = plt.get_cmap("gist_heat")
        colors = cmap(np.linspace(0, 0.75, len(t)))
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(
                t,
                self.q_targets[index],
                color="gray",
                label=r"$q$",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )
            ax.scatter(
                t,
                self.q_preds[index],
                color=colors,
                marker=".",
                s=9,
                label=r"$\hat{q}$",
                zorder=1,
                edgecolors="none",
            )
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$q(t)$")
            ax.autoscale(tight=True)
            ax.text(
                0.05, 0.9, f"Loss: {loss_q:.4e}", transform=ax.transAxes, fontsize=5
            )
            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_p(self, name: str, index: int):
        t = self.t_vec[index]
        loss_p = self.total_loss_p_vec[index]
        cmap = plt.get_cmap("gist_heat")
        colors = cmap(np.linspace(0, 0.75, len(t)))
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(
                t,
                self.p_targets[index],
                color="gray",
                label=r"$p$",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )
            ax.scatter(
                t,
                self.p_preds[index],
                color=colors,
                marker=".",
                s=9,
                label=r"$\hat{p}$",
                zorder=1,
                edgecolors="none",
            )
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$p(t)$")
            ax.autoscale(tight=True)
            ax.text(
                0.05, 0.1, f"Loss: {loss_p:.4e}", transform=ax.transAxes, fontsize=5
            )
            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_phase(self, name: str, index: int):
        t = self.t_vec[index]
        loss = self.total_loss_vec[index]
        cmap = plt.get_cmap("gist_heat")
        colors = cmap(np.linspace(0, 0.75, len(t)))
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(
                self.q_targets[index],
                self.p_targets[index],
                color="gray",
                label=r"$(q,p)$",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )
            ax.scatter(
                self.q_preds[index],
                self.p_preds[index],
                color=colors,
                marker=".",
                s=9,
                label=r"$(\hat{q}, \hat{p})$",
                zorder=1,
                edgecolors="none",
            )
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$p$")
            ax.autoscale(tight=True)
            ax.text(0.05, 0.5, f"Loss: {loss:.4e}", transform=ax.transAxes, fontsize=5)
            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_V_expand(self, name: str, index: int):
        q = np.linspace(0, 1, 100)
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(q, self.V_vec[index])
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$V(q)$")
            ax.axvline(0.5, color="olive", linestyle="--")
            ax.autoscale(tight=True)
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_q_expand(self, name: str, index: int):
        t = self.t_vec[index]
        loss_q = self.total_loss_q_vec[index]
        cmap = plt.get_cmap("gist_heat")
        colors = cmap(np.linspace(0, 0.75, len(t)))
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(
                t,
                self.q_targets[index],
                color="gray",
                label=r"$q$",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )
            ax.scatter(
                t,
                self.q_preds[index],
                color=colors,
                marker=".",
                s=9,
                label=r"$\hat{q}$",
                zorder=1,
                edgecolors="none",
            )
            ax.axvline(0.5, color="olive", linestyle="--")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$q(t)$")
            ax.autoscale(tight=True)
            ax.text(
                0.03, 0.9, f"Loss: {loss_q:.4e}", transform=ax.transAxes, fontsize=5
            )
            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_p_expand(self, name: str, index: int):
        t = self.t_vec[index]
        loss_p = self.total_loss_p_vec[index]
        cmap = plt.get_cmap("gist_heat")
        colors = cmap(np.linspace(0, 0.75, len(t)))
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(
                t,
                self.p_targets[index],
                color="gray",
                label=r"$p$",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )
            ax.scatter(
                t,
                self.p_preds[index],
                color=colors,
                marker=".",
                s=9,
                label=r"$\hat{p}$",
                zorder=1,
                edgecolors="none",
            )
            ax.axvline(0.5, color="olive", linestyle="--")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$p(t)$")
            ax.autoscale(tight=True)
            ax.text(
                0.03, 0.1, f"Loss: {loss_p:.4e}", transform=ax.transAxes, fontsize=5
            )
            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)


def calculate_comparison_results(
    model, device, dl_true, dl_y4, dl_rk4, variational=False
):
    """
    Compare model, Y4, and RK4 performance against KahanLi8 reference
    """
    model.eval()

    test_results_dict = {
        "model": {"q": [], "p": [], "loss_q": [], "loss_p": [], "loss": []},
        "true": {"q": [], "p": []},
        "y4": {"q": [], "p": [], "loss_q": [], "loss_p": [], "loss": []},
        "rk4": {"q": [], "p": [], "loss_q": [], "loss_p": [], "loss": []},
    }

    all_V = []
    all_t = []

    # Compare each potential
    for true_batch, y4_batch, rk4_batch in zip(dl_true, dl_y4, dl_rk4):
        # True data (KahanLi8)
        V, t, q_true, p_true = [x.to(device) for x in true_batch]

        # Y4 data
        _, _, q_y4, p_y4 = [x.to(device) for x in y4_batch]

        # RK4 data
        _, _, q_rk4, p_rk4 = [x.to(device) for x in rk4_batch]

        # Model predictions
        with torch.no_grad():
            if not variational:
                q_pred, p_pred = model(V, t)
            else:
                q_pred, p_pred, _, _ = model(V, t)

        # Calculate MSE loss (against KahanLi8 reference)
        # Model
        loss_q_model = criterion(q_pred, q_true, reduction="none").mean(dim=1)
        loss_p_model = criterion(p_pred, p_true, reduction="none").mean(dim=1)
        loss_model = 0.5 * (loss_q_model + loss_p_model)

        # Y4
        loss_q_y4 = criterion(q_y4, q_true, reduction="none").mean(dim=1)
        loss_p_y4 = criterion(p_y4, p_true, reduction="none").mean(dim=1)
        loss_y4 = 0.5 * (loss_q_y4 + loss_p_y4)

        # RK4
        loss_q_rk4 = criterion(q_rk4, q_true, reduction="none").mean(dim=1)
        loss_p_rk4 = criterion(p_rk4, p_true, reduction="none").mean(dim=1)
        loss_rk4 = 0.5 * (loss_q_rk4 + loss_p_rk4)

        # Store results
        test_results_dict["model"]["q"].extend(q_pred.cpu().numpy())
        test_results_dict["model"]["p"].extend(p_pred.cpu().numpy())
        test_results_dict["model"]["loss_q"].extend(loss_q_model.cpu().numpy())
        test_results_dict["model"]["loss_p"].extend(loss_p_model.cpu().numpy())
        test_results_dict["model"]["loss"].extend(loss_model.cpu().numpy())

        test_results_dict["true"]["q"].extend(q_true.cpu().numpy())
        test_results_dict["true"]["p"].extend(p_true.cpu().numpy())

        test_results_dict["y4"]["q"].extend(q_y4.cpu().numpy())
        test_results_dict["y4"]["p"].extend(p_y4.cpu().numpy())
        test_results_dict["y4"]["loss_q"].extend(loss_q_y4.cpu().numpy())
        test_results_dict["y4"]["loss_p"].extend(loss_p_y4.cpu().numpy())
        test_results_dict["y4"]["loss"].extend(loss_y4.cpu().numpy())

        test_results_dict["rk4"]["q"].extend(q_rk4.cpu().numpy())
        test_results_dict["rk4"]["p"].extend(p_rk4.cpu().numpy())
        test_results_dict["rk4"]["loss_q"].extend(loss_q_rk4.cpu().numpy())
        test_results_dict["rk4"]["loss_p"].extend(loss_p_rk4.cpu().numpy())
        test_results_dict["rk4"]["loss"].extend(loss_rk4.cpu().numpy())

        # Collect V and t for plotting
        all_V.append(V.cpu().numpy())
        all_t.append(t.cpu().numpy())

    all_V = np.array(all_V)
    all_t = np.array(all_t)

    # Print loss statistics
    print("\n--- Performance Comparison (KahanLi8 Reference) ---")
    print(f"Model Loss: {np.mean(test_results_dict['model']['loss']):.4e}")
    print(f"Y4 Loss: {np.mean(test_results_dict['y4']['loss']):.4e}")
    print(f"RK4 Loss: {np.mean(test_results_dict['rk4']['loss']):.4e}")

    return test_results_dict, all_V, all_t


def plot_comparison_histograms(results_dict, fig_dir):
    """
    Generate loss histograms for model, Y4, and RK4
    """
    # Extract loss data
    model_losses = np.array(results_dict["model"]["loss"])
    y4_losses = np.array(results_dict["y4"]["loss"])
    rk4_losses = np.array(results_dict["rk4"]["loss"])

    # Calculate min/max log scale range
    all_losses = np.concatenate([model_losses, y4_losses, rk4_losses])
    min_loss = all_losses.min()
    min_loss = min_loss if min_loss != 0 else 1e-10  # Avoid log(0)
    loss_min_log = (
        np.log10(min_loss) * 1.01
        if min_loss < 0
        else np.log10(min_loss) * 0.99
    )
    loss_max_log = (
        np.log10(all_losses.max()) * 0.99
        if all_losses.max() < 0
        else np.log10(all_losses.max()) * 1.01
    )

    # Create log bins
    logbins = np.logspace(loss_min_log, loss_max_log, 100)

    # Save parquet
    df_losses = pl.DataFrame(
        {
            "model_loss": model_losses,
            "y4_loss": y4_losses,
            "rk4_loss": rk4_losses,
        }
    )
    df_losses.write_parquet(f"{fig_dir}/losses.parquet")

    # Plot histograms
    with plt.style.context(["science", "nature"]):
        # Model histogram
        fig, ax = plt.subplots()
        ax.hist(model_losses, bins=logbins)
        ax.axvline(model_losses.mean(), color="red", linestyle="--")
        ax.set_xlabel("Loss (vs KahanLi8)")
        ax.set_ylabel("Count")
        ax.set_xscale("log")
        fig.savefig(f"{fig_dir}/00_1_Loss_model_hist.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Y4 histogram
        fig, ax = plt.subplots()
        ax.hist(y4_losses, bins=logbins)
        ax.axvline(y4_losses.mean(), color="red", linestyle="--")
        ax.set_xlabel("Loss (vs KahanLi8)")
        ax.set_ylabel("Count")
        ax.set_xscale("log")
        fig.savefig(f"{fig_dir}/00_2_Loss_y4_hist.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # RK4 histogram
        fig, ax = plt.subplots()
        ax.hist(rk4_losses, bins=logbins)
        ax.axvline(rk4_losses.mean(), color="red", linestyle="--")
        ax.set_xlabel("Loss (vs KahanLi8)")
        ax.set_ylabel("Count")
        ax.set_xscale("log")
        fig.savefig(f"{fig_dir}/00_3_Loss_rk4_hist.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Combined histogram - compare distributions
        fig, ax = plt.subplots()
        ax.hist(
            [model_losses, y4_losses, rk4_losses],
            bins=logbins,
            histtype="step",
            label=["Model", "Y4", "RK4"],
            alpha=0.6,
        )
        ax.axvline(model_losses.mean(), color="C0", linestyle="--")
        ax.axvline(y4_losses.mean(), color="C1", linestyle="--")
        ax.axvline(rk4_losses.mean(), color="C2", linestyle="--")
        ax.set_xlabel("Loss (vs KahanLi8)")
        ax.set_ylabel("Count")
        ax.set_xscale("log")
        # Adjust legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
        fig.tight_layout()
        fig.savefig(
            f"{fig_dir}/00_4_Loss_comparison_hist.png", dpi=600, bbox_inches="tight"
        )
        plt.close(fig)


def plot_detailed_comparisons(results_dict, V, t, fig_dir, indices=None):
    """
    Generate comparison plots for selected potentials
    """
    if indices is None:
        # Default: first 10 and worst case
        model_losses = np.array(results_dict["model"]["loss"])
        worst_idx = np.argmax(model_losses)
        indices = list(range(10)) + [worst_idx]

    num_potentials = len(V)

    for idx in indices:
        if idx >= num_potentials:
            print(
                f"Warning: Index {idx} exceeds available potentials ({num_potentials})"
            )
            continue

        # Current potential data
        V_i = V[idx].reshape(-1)  # Flatten for plotting
        t_array = t[idx].reshape(-1)  # Flatten for plotting
        q_true_i = results_dict["true"]["q"][idx]
        p_true_i = results_dict["true"]["p"][idx]
        q_model_i = results_dict["model"]["q"][idx]
        p_model_i = results_dict["model"]["p"][idx]
        q_y4_i = results_dict["y4"]["q"][idx]
        p_y4_i = results_dict["y4"]["p"][idx]
        q_rk4_i = results_dict["rk4"]["q"][idx]
        p_rk4_i = results_dict["rk4"]["p"][idx]

        # Loss values
        loss_model = results_dict["model"]["loss"][idx]
        loss_q_model = results_dict["model"]["loss_q"][idx]
        loss_p_model = results_dict["model"]["loss_p"][idx]
        loss_y4 = results_dict["y4"]["loss"][idx]
        loss_q_y4 = results_dict["y4"]["loss_q"][idx]
        loss_p_y4 = results_dict["y4"]["loss_p"][idx]
        loss_rk4 = results_dict["rk4"]["loss"][idx]
        loss_q_rk4 = results_dict["rk4"]["loss_q"][idx]
        loss_p_rk4 = results_dict["rk4"]["loss_p"][idx]

        # Color settings
        cmap = plt.get_cmap("gist_heat")
        colors = cmap(np.linspace(0, 0.75, len(t_array)))

        # 1. Potential plot
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            q_range = np.linspace(0, 1, 100)
            ax.plot(q_range, V_i)
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$V(q)$")
            ax.autoscale(tight=True)
            fig.savefig(
                f"{fig_dir}/{idx:02}_potential.png", dpi=600, bbox_inches="tight"
            )
            plt.close(fig)

            # 2. Position (q) comparison plot
            fig, ax = plt.subplots()

            # True (KahanLi8)
            ax.plot(
                t_array,
                q_true_i,
                color="gray",
                label=r"$q$ (KL8)",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )

            # Y4, RK4 (lines)
            ax.plot(
                t_array,
                q_y4_i,
                color="blue",
                linestyle="-",
                label=r"$q$ (Y4)",
                alpha=0.8,
                linewidth=1,
                zorder=1,
            )
            ax.plot(
                t_array,
                q_rk4_i,
                color="green",
                linestyle="--",
                label=r"$q$ (RK4)",
                alpha=0.8,
                linewidth=1,
                zorder=1,
            )

            # Model predictions (points)
            ax.scatter(
                t_array,
                q_model_i,
                color=colors,
                marker=".",
                s=9,
                label=r"$\hat{q}$ (Model)",
                zorder=2,
                edgecolors="none",
            )

            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$q(t)$")
            ax.autoscale(tight=True)

            # Display loss info
            ax.text(
                0.05,
                0.95,
                f"Model Loss: {loss_q_model:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )
            ax.text(
                0.05,
                0.9,
                f"Y4 Loss: {loss_q_y4:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )
            ax.text(
                0.05,
                0.85,
                f"RK4 Loss: {loss_q_rk4:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )

            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()
            fig.savefig(
                f"{fig_dir}/{idx:02}_q_comparison.png", dpi=600, bbox_inches="tight"
            )
            plt.close(fig)

            # 3. Momentum (p) comparison plot
            fig, ax = plt.subplots()

            # True (KahanLi8)
            ax.plot(
                t_array,
                p_true_i,
                color="gray",
                label=r"$p$ (KL8)",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )

            # Y4, RK4 (lines)
            ax.plot(
                t_array,
                p_y4_i,
                color="blue",
                linestyle="-",
                label=r"$p$ (Y4)",
                alpha=0.8,
                linewidth=1,
                zorder=1,
            )
            ax.plot(
                t_array,
                p_rk4_i,
                color="green",
                linestyle="--",
                label=r"$p$ (RK4)",
                alpha=0.8,
                linewidth=1,
                zorder=1,
            )

            # Model predictions (points)
            ax.scatter(
                t_array,
                p_model_i,
                color=colors,
                marker=".",
                s=9,
                label=r"$\hat{p}$ (Model)",
                zorder=2,
                edgecolors="none",
            )

            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$p(t)$")
            ax.autoscale(tight=True)

            # Display loss info
            ax.text(
                0.05,
                0.15,
                f"Model Loss: {loss_p_model:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )
            ax.text(
                0.05,
                0.10,
                f"Y4 Loss: {loss_p_y4:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )
            ax.text(
                0.05,
                0.05,
                f"RK4 Loss: {loss_p_rk4:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )

            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()

            fig.savefig(
                f"{fig_dir}/{idx:02}_p_comparison.png", dpi=600, bbox_inches="tight"
            )
            plt.close(fig)

            # 4. Phase space (q-p) comparison plot
            fig, ax = plt.subplots()

            # True (KahanLi8)
            ax.plot(
                q_true_i,
                p_true_i,
                color="gray",
                label=r"$(q,p)$ (KL8)",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )

            # Y4, RK4 (lines)
            ax.plot(
                q_y4_i,
                p_y4_i,
                color="blue",
                linestyle="-",
                label=r"$(q,p)$ (Y4)",
                alpha=0.8,
                linewidth=1,
                zorder=1,
            )
            ax.plot(
                q_rk4_i,
                p_rk4_i,
                color="green",
                linestyle="--",
                label=r"$(q,p)$ (RK4)",
                alpha=0.8,
                linewidth=1,
                zorder=1,
            )

            # Model predictions (points)
            ax.scatter(
                q_model_i,
                p_model_i,
                color=colors,
                marker=".",
                s=9,
                label=r"$(\hat{q},\hat{p})$ (Model)",
                zorder=2,
                edgecolors="none",
            )

            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$p$")
            ax.autoscale(tight=True)

            # Display loss info
            ax.text(
                0.05,
                0.5,
                f"Model Loss: {loss_model:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )
            ax.text(
                0.05,
                0.45,
                f"Y4 Loss: {loss_y4:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )
            ax.text(
                0.05,
                0.4,
                f"RK4 Loss: {loss_rk4:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )

            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()

            fig.savefig(
                f"{fig_dir}/{idx:02}_phase_comparison.png", dpi=600, bbox_inches="tight"
            )
            plt.close(fig)


def plot_kl8_model_only(results_dict, V, t, fig_dir, indices=None):
    """
    Generate plots showing only KahanLi8 reference and Model predictions
    """
    if indices is None:
        # Default: first 10 and worst case
        model_losses = np.array(results_dict["model"]["loss"])
        worst_idx = np.argmax(model_losses)
        indices = list(range(10)) + [worst_idx]

    num_potentials = len(V)

    for idx in indices:
        if idx >= num_potentials:
            print(
                f"Warning: Index {idx} exceeds available potentials ({num_potentials})"
            )
            continue

        # Current potential data
        V_i = V[idx].reshape(-1)  # Flatten for plotting
        t_array = t[idx].reshape(-1)  # Flatten for plotting
        q_true_i = results_dict["true"]["q"][idx]
        p_true_i = results_dict["true"]["p"][idx]
        q_model_i = results_dict["model"]["q"][idx]
        p_model_i = results_dict["model"]["p"][idx]

        # Loss values
        loss_model = results_dict["model"]["loss"][idx]
        loss_q_model = results_dict["model"]["loss_q"][idx]
        loss_p_model = results_dict["model"]["loss_p"][idx]

        # Color settings
        cmap = plt.get_cmap("gist_heat")
        colors = cmap(np.linspace(0, 0.75, len(t_array)))

        # 1. Potential plot
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            q_range = np.linspace(0, 1, 100)
            ax.plot(q_range, V_i)
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$V(q)$")
            ax.autoscale(tight=True)
            fig.savefig(
                f"{fig_dir}/{idx:02}_kl8_potential.png", dpi=600, bbox_inches="tight"
            )
            plt.close(fig)

            # 2. Position (q) plot - KL8 vs Model only
            fig, ax = plt.subplots()

            # True (KahanLi8)
            ax.plot(
                t_array,
                q_true_i,
                color="gray",
                label=r"$q$ (KL8)",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )

            # Model predictions (points)
            ax.scatter(
                t_array,
                q_model_i,
                color=colors,
                marker=".",
                s=9,
                label=r"$\hat{q}$ (Model)",
                zorder=1,
                edgecolors="none",
            )

            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$q(t)$")
            ax.autoscale(tight=True)

            # Display loss info
            ax.text(
                0.05,
                0.95,
                f"Loss: {loss_q_model:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )

            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()
            fig.savefig(
                f"{fig_dir}/{idx:02}_kl8_q_only.png", dpi=600, bbox_inches="tight"
            )
            plt.close(fig)

            # 3. Momentum (p) plot - KL8 vs Model only
            fig, ax = plt.subplots()

            # True (KahanLi8)
            ax.plot(
                t_array,
                p_true_i,
                color="gray",
                label=r"$p$ (KL8)",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )

            # Model predictions (points)
            ax.scatter(
                t_array,
                p_model_i,
                color=colors,
                marker=".",
                s=9,
                label=r"$\hat{p}$ (Model)",
                zorder=1,
                edgecolors="none",
            )

            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$p(t)$")
            ax.autoscale(tight=True)

            # Display loss info
            ax.text(
                0.05,
                0.15,
                f"Loss: {loss_p_model:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )

            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()
            fig.savefig(
                f"{fig_dir}/{idx:02}_kl8_p_only.png", dpi=600, bbox_inches="tight"
            )
            plt.close(fig)

            # 4. Phase space (q-p) plot - KL8 vs Model only
            fig, ax = plt.subplots()

            # True (KahanLi8)
            ax.plot(
                q_true_i,
                p_true_i,
                color="gray",
                label=r"$(q,p)$ (KL8)",
                alpha=0.5,
                linewidth=1.75,
                zorder=0,
            )

            # Model predictions (points)
            ax.scatter(
                q_model_i,
                p_model_i,
                color=colors,
                marker=".",
                s=9,
                label=r"$(\hat{q},\hat{p})$ (Model)",
                zorder=1,
                edgecolors="none",
            )

            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$p$")
            ax.autoscale(tight=True)

            # Display loss info
            ax.text(
                0.05,
                0.5,
                f"Loss: {loss_model:.4e}",
                transform=ax.transAxes,
                fontsize=5,
            )

            # Adjust legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=5)
            fig.tight_layout()
            fig.savefig(
                f"{fig_dir}/{idx:02}_kl8_phase_only.png", dpi=600, bbox_inches="tight"
            )
            plt.close(fig)


def main():
    # Test run
    console = Console()
    console.print("[bold green]Analyzing the model...[/bold green]")
    console.print("Select a project to analyze:")
    project = select_project()
    console.print("Select a group to analyze:")
    group_name = select_group(project)
    console.print("Select a seed to analyze:")
    seed = select_seed(project, group_name)
    console.print("Select a device:")
    device = select_device()
    model, config = load_model(project, group_name, seed)
    model = model.to(device)

    variational = False
    if "VaRONet" in config.net:
        variational = True

    test_options = ["test", "physical"]
    console.print("Select a test option:")
    test_option = beaupy.select(test_options)
    if test_option == "test":
        # Create result directory
        fig_dir = f"figs/{project}"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Provide two analysis modes
        test_modes = ["Original (Y4 Reference)", "Enhanced (KahanLi8 Reference)"]
        console.print("Select test mode:")
        test_mode = beaupy.select(test_modes)

        if test_mode == "Original (Y4 Reference)":
            # Original analysis (using Y4 as reference)
            ds_val = load_data("./data_test/test.parquet")
            dl_val = DataLoader(ds_val, batch_size=100)

            test_results = TestResults(model, dl_val, device, variational)
            test_results.print_results()

            # Histogram for loss
            test_results.hist_loss(f"{fig_dir}/00_0_Loss_hist")
            test_results.hist_loss_rk4(f"{fig_dir}/00_0_Loss_rk4_hist")

            losses = test_results.total_loss_vec
            worst_idx = int(np.argmax(losses))

            # Plot the results
            for index in range(10):
                test_results.plot_V(f"{fig_dir}/{index:02}_0_V_plot", index)
                test_results.plot_q(f"{fig_dir}/{index:02}_1_q_plot", index)
                test_results.plot_p(f"{fig_dir}/{index:02}_2_p_plot", index)
                test_results.plot_phase(f"{fig_dir}/{index:02}_3_phase_plot", index)

            # Plot the worst result
            test_results.plot_V(f"{fig_dir}/{worst_idx:02}_0_V_plot", worst_idx)
            test_results.plot_q(f"{fig_dir}/{worst_idx:02}_1_q_plot", worst_idx)
            test_results.plot_p(f"{fig_dir}/{worst_idx:02}_2_p_plot", worst_idx)
            test_results.plot_phase(f"{fig_dir}/{worst_idx:02}_3_phase_plot", worst_idx)

        else:  # Enhanced (KahanLi8 Reference)
            # New analysis (using KahanLi8 as reference)
            console.print(
                "[bold yellow]Loading KahanLi8 reference data...[/bold yellow]"
            )

            try:
                # Load KahanLi8, Y4, RK4 data
                ds_true, ds_y4, ds_rk4 = load_test_data_with_true()

                # Create dataloaders
                batch_size = 1  # Appropriate batch size
                dl_true = DataLoader(ds_true, batch_size=batch_size)
                dl_y4 = DataLoader(ds_y4, batch_size=batch_size)
                dl_rk4 = DataLoader(ds_rk4, batch_size=batch_size)

                # Compare model with reference data
                console.print("[bold green]Analyzing model performance...[/bold green]")
                results_dict, V, t = calculate_comparison_results(
                    model, device, dl_true, dl_y4, dl_rk4, variational
                )

                # Visualize results
                console.print("[bold green]Generating histograms...[/bold green]")
                plot_comparison_histograms(results_dict, fig_dir)

                # Create detailed comparison plots
                console.print("[bold green]Generating comparison plots...[/bold green]")

                # Find worst case
                model_losses = np.array(results_dict["model"]["loss"])
                worst_idx = np.argmax(model_losses)

                # Generate detailed plots for first 10 cases and worst case
                indices = list(range(0, len(V), len(V) // 10))
                if worst_idx not in indices:
                    indices.append(worst_idx)

                # Generate comparison plots (all methods)
                plot_detailed_comparisons(results_dict, V, t, fig_dir, indices)

                # Generate KL8 vs Model only plots
                console.print(
                    "[bold green]Generating KL8 vs Model plots...[/bold green]"
                )
                plot_kl8_model_only(results_dict, V, t, fig_dir, indices)

                console.print("[bold green]Analysis complete![/bold green]")

            except FileNotFoundError as e:
                console.print(f"[bold red]Error: {e}[/bold red]")
                console.print(
                    "[bold yellow]KahanLi8 reference data not found. Run the Julia script test_reference.jl first.[/bold yellow]"
                )
    elif test_option == "physical":
        potentials = {
            "sho": "SHO",
            "double_well": "DoubleWell",
            "morse": "Morse",
            "pendulum": "Pendulum",
            "atw": "ATW",
            "stw": "STW",
            "sstw": "SSTW",
        }
        results = [load_relevant_data(name) for name in potentials.keys()]
        ds_trues, ds_y4s, ds_rk4s = zip(*results)
        tests_name = list(potentials.values())

        for i in range(len(ds_trues)):
            print()
            print(f"Test {tests_name[i]}:")
            ds_true = ds_trues[i]
            ds_y4 = ds_y4s[i]
            ds_rk4 = ds_rk4s[i]
            test_name = tests_name[i]

            dl_true = DataLoader(ds_true, batch_size=1)
            dl_y4 = DataLoader(ds_y4, batch_size=1)
            dl_rk4 = DataLoader(ds_rk4, batch_size=1)

            test_results = TestResults(model, dl_true, device, variational)
            test_results.print_results()

            fig_dir = f"figs/{project}/{test_name}"
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            test_results.plot_V(f"{fig_dir}/{i:02}_{test_name}_0_V_plot", 0)
            test_results.plot_q(f"{fig_dir}/{i:02}_{test_name}_1_q_plot", 0)
            test_results.plot_p(f"{fig_dir}/{i:02}_{test_name}_2_p_plot", 0)
            test_results.plot_phase(f"{fig_dir}/{i:02}_{test_name}_3_phase_plot", 0)

            for batch_data in zip(dl_true, dl_y4, dl_rk4):
                (_, _, q_true, p_true) = batch_data[0]
                (_, _, q_y4, p_y4) = batch_data[1]
                (_, _, q_rk4, p_rk4) = batch_data[2]

                q_true = q_true.numpy().astype(np.float64).reshape(-1)
                p_true = p_true.numpy().astype(np.float64).reshape(-1)
                q_y4 = q_y4.numpy().astype(np.float64).reshape(-1)
                p_y4 = p_y4.numpy().astype(np.float64).reshape(-1)
                q_rk4 = q_rk4.numpy().astype(np.float64).reshape(-1)
                p_rk4 = p_rk4.numpy().astype(np.float64).reshape(-1)
                q_test = test_results.q_preds.astype(np.float64)
                p_test = test_results.p_preds.astype(np.float64)

                loss_q_test = np_log_cosh_loss(q_test, q_true)
                loss_p_test = np_log_cosh_loss(p_test, p_true)
                loss_test = 0.5 * (loss_q_test + loss_p_test)

                loss_q_y4 = np_log_cosh_loss(q_y4, q_true)
                loss_p_y4 = np_log_cosh_loss(p_y4, p_true)
                loss_y4 = 0.5 * (loss_q_y4 + loss_p_y4)

                loss_q_rk4 = np_log_cosh_loss(q_rk4, q_true)
                loss_p_rk4 = np_log_cosh_loss(p_rk4, p_true)
                loss_rk4 = 0.5 * (loss_q_rk4 + loss_p_rk4)

                print(f"Model Loss: {loss_test:.4e}")
                print(f"Y4 Loss: {loss_y4:.4e}")
                print(f"RK4 Loss: {loss_rk4:.4e}")

                t = np.linspace(0, 2, len(q_true))
                cmap = plt.get_cmap("gist_heat")
                colors = cmap(np.linspace(0, 0.75, len(t)))

                with plt.style.context(["science", "nature"]):
                    fig, ax = plt.subplots()

                    # True plot
                    ax.plot(
                        t,
                        q_true,
                        color="gray",
                        label=r"$q$ (KL8)",
                        alpha=0.5,
                        linewidth=1.75,
                        zorder=0,
                    )

                    ax.plot(
                        t,
                        q_y4,
                        color="blue",
                        linestyle="-",
                        label=r"$q$ (Y4)",
                        alpha=0.8,
                        linewidth=1,
                        zorder=1,
                    )
                    ax.plot(
                        t,
                        q_rk4,
                        color="green",
                        linestyle="--",
                        label=r"$q$ (RK4)",
                        alpha=0.8,
                        linewidth=1,
                        zorder=1,
                    )

                    ax.scatter(
                        t,
                        q_test,
                        color=colors,
                        marker=".",
                        s=9,
                        label=r"$\hat{q}$",
                        zorder=2,
                        edgecolors="none",
                    )

                    ax.set_xlabel(r"$t$")
                    ax.set_ylabel(r"$q(t)$")
                    ax.autoscale(tight=True)

                    ax.text(
                        0.05,
                        0.95,
                        f"Model Loss: {loss_q_test:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.9,
                        f"Y4 Loss: {loss_q_y4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.85,
                        f"RK4 Loss: {loss_q_rk4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )

                    # Adjust legend
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(
                        by_label.values(), by_label.keys(), loc="best", fontsize=5
                    )
                    fig.tight_layout()

                    fig.savefig(
                        f"{fig_dir}/{i:02}_{test_name}_compare_q_plot.png",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

                    fig, ax = plt.subplots()

                    ax.plot(
                        t,
                        p_true,
                        color="gray",
                        label=r"$p$ (KL8)",
                        alpha=0.5,
                        linewidth=1.75,
                        zorder=0,
                    )

                    ax.plot(
                        t,
                        p_y4,
                        color="blue",
                        linestyle="-",
                        label=r"$p$ (Y4)",
                        alpha=0.8,
                        linewidth=1,
                        zorder=1,
                    )
                    ax.plot(
                        t,
                        p_rk4,
                        color="green",
                        linestyle="--",
                        label=r"$p$ (RK4)",
                        alpha=0.8,
                        linewidth=1,
                        zorder=1,
                    )

                    ax.scatter(
                        t,
                        p_test,
                        color=colors,
                        marker=".",
                        s=9,
                        label=r"$\hat{p}$",
                        zorder=2,
                        edgecolors="none",
                    )

                    ax.set_xlabel(r"$t$")
                    ax.set_ylabel(r"$p(t)$")
                    ax.autoscale(tight=True)

                    ax.text(
                        0.05,
                        0.15,
                        f"Model Loss: {loss_p_test:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.10,
                        f"Y4 Loss: {loss_p_y4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.05,
                        f"RK4 Loss: {loss_p_rk4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )

                    # Adjust legend
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(
                        by_label.values(), by_label.keys(), loc="best", fontsize=5
                    )
                    fig.tight_layout()

                    fig.savefig(
                        f"{fig_dir}/{i:02}_{test_name}_compare_p_plot.png",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

                    fig, ax = plt.subplots()

                    ax.plot(
                        q_true,
                        p_true,
                        color="gray",
                        label=r"$(q,p)$ (KL8)",
                        alpha=0.5,
                        linewidth=1.75,
                        zorder=0,
                    )

                    ax.plot(
                        q_y4,
                        p_y4,
                        color="blue",
                        linestyle="-",
                        label=r"$(q,p)$ (Y4)",
                        alpha=0.8,
                        linewidth=1,
                        zorder=1,
                    )
                    ax.plot(
                        q_rk4,
                        p_rk4,
                        color="green",
                        linestyle="--",
                        label=r"$(q,p)$ (RK4)",
                        alpha=0.8,
                        linewidth=1,
                        zorder=1,
                    )

                    ax.scatter(
                        q_test,
                        p_test,
                        color=colors,
                        marker=".",
                        s=9,
                        label=r"$(\hat{q}, \hat{p})$",
                        zorder=2,
                        edgecolors="none",
                    )

                    ax.set_xlabel(r"$q$")
                    ax.set_ylabel(r"$p$")
                    ax.autoscale(tight=True)

                    ax.text(
                        0.05,
                        0.5,
                        f"Model Loss: {loss_test:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.45,
                        f"Y4 Loss: {loss_y4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.4,
                        f"RK4 Loss: {loss_rk4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )

                    # Adjust legend
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(
                        by_label.values(), by_label.keys(), loc="best", fontsize=5
                    )
                    fig.tight_layout()

                    fig.savefig(
                        f"{fig_dir}/{i:02}_{test_name}_compare_phase_plot.png",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)


if __name__ == "__main__":
    console = Console()
    main()
