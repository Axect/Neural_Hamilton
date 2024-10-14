import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import scienceplots
import polars as pl
import survey

import os
import time

from util import select_project, select_group, select_seed, select_device, load_model, load_data, load_study, load_best_model


# ┌──────────────────────────────────────────────────────────┐
#  RK4 Solver
# └──────────────────────────────────────────────────────────┘
def rk4_step(f, y, t, dt):
    k1 = dt * f(y, t)
    k2 = dt * f(y + k1 / 2, t + dt / 2)
    k3 = dt * f(y + k2 / 2, t + dt / 2)
    k4 = dt * f(y + k3, t + dt)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def solve_hamilton(V, x0, p0, t):
    x = np.linspace(0, 1, 100)
    V_interp = PchipInterpolator(x, V)
    dVdx_interp = V_interp.derivative()

    def hamilton_eqs(y, _t):
        x, p = y
        dxdt = p
        dpdt = -dVdx_interp(x)
        return np.array([dxdt, dpdt])

    y0 = np.array([x0, p0])
    dt = t[1] - t[0]

    solution = np.zeros((len(t), 2))
    solution[0] = y0

    for i in range(1, len(t)):
        solution[i] = rk4_step(hamilton_eqs, solution[i - 1], t[i - 1], dt)

    return solution[:,0], solution[:,1]


class TestResults:
    def __init__(self, model, dl_val, device, variational=False):
        self.model = model
        self.dl_val = dl_val
        self.device = device
        self.variational = variational
        
        self.run_test()
        self.load_rk4()
        self.measure_rk4()

    def run_test(self):
        self.model.eval()

        V_vec = []
        x_preds = []
        p_preds = []
        x_targets = []
        p_targets = []

        total_loss_x_vec = []
        total_loss_p_vec = []
        total_loss_vec = []
        total_time_vec = []

        with torch.no_grad():
            for V, t, x, p in self.dl_val:
                V, t, x, p = V.to(self.device), t.to(self.device), x.to(self.device), p.to(self.device)
                
                t_start = time.time()
                if not self.variational:
                    self.reparameterize=False
                    x_pred, p_pred = self.model(V, t)
                else:
                    x_pred, p_pred, _, _ = self.model(V, t)
                t_total = (time.time() - t_start) / V.shape[0]
                
                loss_x_vec = F.mse_loss(x_pred, x, reduction="none")
                loss_p_vec = F.mse_loss(p_pred, p, reduction="none")
                loss_vec = 0.5 * (loss_x_vec + loss_p_vec)

                loss_x = loss_x_vec.mean(dim=1)
                loss_p = loss_p_vec.mean(dim=1)
                loss = loss_vec.mean(dim=1)
                
                V_vec.extend(V.cpu().numpy())
                x_preds.extend(x_pred.cpu().numpy())
                p_preds.extend(p_pred.cpu().numpy())
                x_targets.extend(x.cpu().numpy())
                p_targets.extend(p.cpu().numpy())
                total_loss_vec.extend(loss.cpu().numpy())
                total_loss_x_vec.extend(loss_x.cpu().numpy())
                total_loss_p_vec.extend(loss_p.cpu().numpy())
                total_time_vec.extend([t_total] * V.shape[0])

        self.total_loss_vec = np.array(total_loss_vec)
        self.total_loss_x_vec = np.array(total_loss_x_vec)
        self.total_loss_p_vec = np.array(total_loss_p_vec)
        self.total_time_vec = np.array(total_time_vec)
        self.V_vec = np.array(V_vec)
        self.x_preds = np.array(x_preds)
        self.p_preds = np.array(p_preds)
        self.x_targets = np.array(x_targets)
        self.p_targets = np.array(p_targets)

    def load_rk4(self):
        df_rk4 = pl.read_parquet("./data_analyze/rk4.parquet")
        self.rk4_x = df_rk4["x"].to_numpy().reshape(-1, 100)
        self.rk4_p = df_rk4["p"].to_numpy().reshape(-1, 100)
        self.rk4_loss_x = df_rk4["loss_x"].to_numpy().reshape(-1, 100).mean(axis=1)
        self.rk4_loss_p = df_rk4["loss_p"].to_numpy().reshape(-1, 100).mean(axis=1)
        self.rk4_loss = df_rk4["loss"].to_numpy().reshape(-1, 100).mean(axis=1)

    def measure_rk4(self):
        t_total_vec = []
        for V, _, x, p in self.dl_val:
            for i in range(V.shape[0]):
                V_i = V[i].detach().cpu().numpy()
                t = np.linspace(0, 2, 100)
                t_start = time.time()
                _, _ = solve_hamilton(V_i, 0, 0, t)
                t_total_vec.append(time.time() - t_start)
        self.t_total_time_vec_rk4 = np.array(t_total_vec)

    def print_results(self):
        print(self.total_loss_vec.shape)
        print(f"Total Loss: {self.total_loss_vec.mean():.4e}")
        print(f"Total Loss x: {self.total_loss_x_vec.mean():.4e}")
        print(f"Total Loss p: {self.total_loss_p_vec.mean():.4e}")

    def hist_loss(self, name:str):
        losses = self.total_loss_vec
        times = self.total_time_vec
        df_losses = pl.DataFrame({"loss":losses, "time":times})
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
            ax.axvline(losses.mean(), color='red', linestyle='--')
            ax.set_xlabel("Total Loss")
            ax.set_ylabel("Count")
            ax.set_xscale("log")
            ax.set_yscale("log")
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def hist_loss_rk4(self, name:str):
        losses = self.rk4_loss
        times = self.t_total_time_vec_rk4
        df_losses = pl.DataFrame({"loss":losses, "time":times})
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
            ax.axvline(losses.mean(), color='red', linestyle='--')
            ax.set_xlabel("Total Loss")
            ax.set_ylabel("Count")
            ax.set_xscale("log")
            ax.set_yscale("log")
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_V(self, name:str, index:int):
        q = np.linspace(0, 1, 100)
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(q, self.V_vec[index])
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$V(q)$")
            ax.autoscale(tight=True)
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_q(self, name:str, index:int):
        t = np.linspace(0, 2, len(self.x_preds[index]))
        loss_x = self.total_loss_x_vec[index]
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(t, self.x_targets[index], color='gray', label=r"$q$", alpha=0.65, linewidth=1.75)
            ax.plot(t, self.x_preds[index], ':', color='red', label=r"$\hat{q}$")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$q(t)$")
            ax.autoscale(tight=True)
            ax.text(0.05, 0.9, f"Loss: {loss_x:.4e}", transform=ax.transAxes, fontsize=5)
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_p(self, name:str, index:int):
        t = np.linspace(0, 2, len(self.p_preds[index]))
        loss_p = self.total_loss_p_vec[index]
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(t, self.p_targets[index], color='gray', label=r"$p$", alpha=0.65, linewidth=1.75)
            ax.plot(t, self.p_preds[index], ':', color='red', label=r"$\hat{p}$")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$p(t)$")
            ax.autoscale(tight=True)
            ax.text(0.05, 0.1, f"Loss: {loss_p:.4e}", transform=ax.transAxes, fontsize=5)
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_phase(self, name:str, index:int):
        loss = self.total_loss_vec[index]
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(self.x_targets[index], self.p_targets[index], color='gray', label=r"$(q,p)$", alpha=0.65, linewidth=1.75)
            ax.plot(self.x_preds[index], self.p_preds[index], ':', color='red', label=r"$(\hat{q}, \hat{p})$")
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$p$")
            ax.autoscale(tight=True)
            ax.text(0.05, 0.5, f"Loss: {loss:.4e}", transform=ax.transAxes, fontsize=5)
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_compare_q(self, name:str, index:int):
        t = np.linspace(0, 2, len(self.x_preds[index]))
        loss_nn = self.total_loss_x_vec[index]
        loss_rk4 = self.rk4_loss_x[index]
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(t, self.x_targets[index], color='gray', label=r"$q$ (Target)", alpha=0.65, linewidth=1.75)
            ax.plot(t, self.x_preds[index], ':', color='red', label=r"$\hat{q}$ (Neural Network)")
            ax.plot(t, self.rk4_x[index], '--', color='blue', label=r"$q$ (RK4)")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$q(t)$")
            ax.autoscale(tight=True)
            ax.text(0.05, 0.95, f"NN Loss: {loss_nn:.4e}", transform=ax.transAxes, fontsize=5)
            ax.text(0.05, 0.9, f"RK4 Loss: {loss_rk4:.4e}", transform=ax.transAxes, fontsize=5)
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_compare_p(self, name:str, index:int):
        t = np.linspace(0, 2, len(self.p_preds[index]))
        loss_nn = self.total_loss_p_vec[index]
        loss_rk4 = self.rk4_loss_p[index]
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(t, self.p_targets[index], color='gray', label=r"$p$ (Target)", alpha=0.65, linewidth=1.75)
            ax.plot(t, self.p_preds[index], ':', color='red', label=r"$\hat{p}$ (Neural Network)")
            ax.plot(t, self.rk4_p[index], '--', color='blue', label=r"$p$ (RK4)")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$p(t)$")
            ax.autoscale(tight=True)
            ax.text(0.05, 0.95, f"NN Loss: {loss_nn:.4e}", transform=ax.transAxes, fontsize=5)
            ax.text(0.05, 0.9, f"RK4 Loss: {loss_rk4:.4e}", transform=ax.transAxes, fontsize=5)
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_compare_phase(self, name:str, index:int):
        loss_nn = self.total_loss_vec[index]
        loss_rk4 = self.rk4_loss[index]
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(self.x_targets[index], self.p_targets[index], color='gray', label=r"$(q,p)$", alpha=0.65, linewidth=1.75)
            ax.plot(self.x_preds[index], self.p_preds[index], ':', color='red', label=r"$(\hat{q}, \hat{p})$")
            ax.plot(self.rk4_x[index], self.rk4_p[index], '--', color='blue', label=r"$(q,p)$ (RK4)")
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$p$")
            ax.autoscale(tight=True)
            ax.text(0.05, 0.9, f"NN Loss: {loss_nn:.4e}", transform=ax.transAxes, fontsize=5)
            ax.text(0.05, 0.85, f"RK4 Loss: {loss_rk4:.4e}", transform=ax.transAxes, fontsize=5)
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)


def main():
    # Test run
    project = select_project()
    group_name = select_group(project)
    seed = select_seed(project, group_name)
    device = select_device()
    model, config = load_model(project, group_name, seed)
    model = model.to(device)

    variational = False
    if "VaRONet" in config.net:
        variational = True

    test_options = ["test", "physical"]
    test_option = survey.routines.select("Select test option:", options=test_options)
    if test_option == 0:
        ds_val = load_data("./data_normal/test.parquet")
        dl_val = DataLoader(ds_val, batch_size=100)


        test_results = TestResults(model, dl_val, device, variational)
        test_results.print_results()

        fig_dir = f"figs/{project}"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

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
    else:
        ds_test_sho = load_data("./data_analyze/sho.parquet")
        ds_test_quartic = load_data("./data_analyze/quartic.parquet")
        ds_test_morse = load_data("./data_analyze/morse.parquet")
        ds_test_smff = load_data("./data_analyze/smff.parquet")
        ds_test_mff = load_data("./data_analyze/mff.parquet")
        ds_test_unbounded = load_data("./data_analyze/unbounded.parquet")

        ds_sho_rk4 = load_data("./data_analyze/sho_rk4.parquet")
        ds_quartic_rk4 = load_data("./data_analyze/quartic_rk4.parquet")
        ds_morse_rk4 = load_data("./data_analyze/morse_rk4.parquet")
        ds_smff_rk4 = load_data("./data_analyze/smff_rk4.parquet")
        ds_mff_rk4 = load_data("./data_analyze/mff_rk4.parquet")
        ds_unbounded_rk4 = load_data("./data_analyze/unbounded_rk4.parquet")

        ds_tests = [ds_test_sho, ds_test_quartic, ds_test_morse, ds_test_smff, ds_test_mff, ds_test_unbounded]
        ds_rk4s = [ds_sho_rk4, ds_quartic_rk4, ds_morse_rk4, ds_smff_rk4, ds_mff_rk4, ds_unbounded_rk4]
        dl_tests = [DataLoader(ds_test, batch_size=1) for ds_test in ds_tests]
        dl_rk4s = [DataLoader(ds_rk4, batch_size=1) for ds_rk4 in ds_rk4s]
        tests_name = ["SHO", "Quartic", "Morse", "SMFF", "MFF", "Unbounded"]
        for name, dl, dl_rk4 in zip(tests_name, dl_tests, dl_rk4s):
            print(f"Test {name}:")
            test_results = TestResults(model, dl, device, variational)
            test_results.print_results()

            fig_dir = f"figs/{project}"
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            test_results.plot_V(f"{fig_dir}/{name}_0_V_plot", 0)
            test_results.plot_q(f"{fig_dir}/{name}_1_q_plot", 0)
            test_results.plot_p(f"{fig_dir}/{name}_2_p_plot", 0)
            test_results.plot_phase(f"{fig_dir}/{name}_3_phase_plot", 0)

            # RK4
            for (_, _, x, p), (_, _, x_hat, p_hat) in zip(dl, dl_rk4):
                x = x.numpy().reshape(-1)
                p = p.numpy().reshape(-1)
                x_hat = x_hat.numpy().reshape(-1)
                p_hat = p_hat.numpy().reshape(-1)
                loss_x = np.mean(np.square(x - x_hat))
                loss_p = np.mean(np.square(p - p_hat))
                loss = 0.5 * (loss_x + loss_p)
                print(f"RK4 Loss: {loss:.4e}")

                t = np.linspace(0, 2, len(x))
                with plt.style.context(["science", "nature"]):
                    fig, ax = plt.subplots()
                    ax.plot(t, x, color='gray', label=r"$q$", alpha=0.65, linewidth=1.75)
                    ax.plot(t, x_hat, ':', color='red', label=r"$\hat{q}$")
                    ax.set_xlabel(r"$t$")
                    ax.set_ylabel(r"$q(t)$")
                    ax.autoscale(tight=True)
                    ax.text(0.05, 0.9, f"Loss: {loss_x:.4e}", transform=ax.transAxes, fontsize=5)
                    ax.legend()
                    fig.savefig(f"{fig_dir}/{name}_RK4_0_q_plot.png", dpi=600, bbox_inches="tight")
                    plt.close(fig)

                    fig, ax = plt.subplots()
                    ax.plot(t, p, color='gray', label=r"$p$", alpha=0.65, linewidth=1.75)
                    ax.plot(t, p_hat, ':', color='red', label=r"$\hat{p}$")
                    ax.set_xlabel(r"$t$")
                    ax.set_ylabel(r"$p(t)$")
                    ax.autoscale(tight=True)
                    ax.text(0.05, 0.1, f"Loss: {loss_p:.4e}", transform=ax.transAxes, fontsize=5)
                    ax.legend()
                    fig.savefig(f"{fig_dir}/{name}_RK4_1_p_plot.png", dpi=600, bbox_inches="tight")
                    plt.close(fig)

                    fig, ax = plt.subplots()
                    ax.plot(x, p, color='gray', label=r"$p$", alpha=0.65, linewidth=1.75)
                    ax.plot(x_hat, p_hat, ':', color='red', label=r"$\hat{p}$")
                    ax.set_xlabel(r"$q$")
                    ax.set_ylabel(r"$p$")
                    ax.autoscale(tight=True)
                    ax.text(0.05, 0.5, f"Loss: {loss:.4e}", transform=ax.transAxes, fontsize=5)
                    ax.legend()
                    fig.savefig(f"{fig_dir}/{name}_RK4_2_phase_plot.png", dpi=600, bbox_inches="tight")
                    plt.close(fig)


if __name__ == "__main__":
    main()
