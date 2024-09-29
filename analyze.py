import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import polars as pl

import os

from util import select_project, select_group, select_seed, select_device, load_model, load_data, load_study, load_best_model

class TestResults:
    def __init__(self, model, dl_val, device, variational=False):
        self.model = model
        self.dl_val = dl_val
        self.device = device
        self.variational = variational
        
        self.run_test()
        self.load_rk4()

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

        with torch.no_grad():
            for V, t, x, p in self.dl_val:
                V, t, x, p = V.to(self.device), t.to(self.device), x.to(self.device), p.to(self.device)
                
                if not self.variational:
                    x_pred, p_pred = self.model(V, t)
                else:
                    x_pred, p_pred, _, _ = self.model(V, t)
                
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

        self.total_loss_vec = np.array(total_loss_vec)
        self.total_loss_x_vec = np.array(total_loss_x_vec)
        self.total_loss_p_vec = np.array(total_loss_p_vec)
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

    def print_results(self):
        print(self.total_loss_vec.shape)
        print(f"Total Loss: {self.total_loss_vec.mean():.4e}")
        print(f"Total Loss x: {self.total_loss_x_vec.mean():.4e}")
        print(f"Total Loss p: {self.total_loss_p_vec.mean():.4e}")

    def hist_loss(self, name:str):
        losses = self.total_loss_vec
        df_losses = pl.DataFrame({"loss":losses})
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
        df_losses = pl.DataFrame({"loss":losses})
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
        t = np.linspace(0, 1, len(self.x_preds[index]))
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
        t = np.linspace(0, 1, len(self.p_preds[index]))
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
        t = np.linspace(0, 1, len(self.x_preds[index]))
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
        t = np.linspace(0, 1, len(self.p_preds[index]))
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

    # Load the best model
    #study_name = "Optimize_Template"
    #model, config = load_best_model(project, study_name)
    #device = select_device()
    #model = model.to(device)

    ds_val = load_data("./data_normal/test.parquet")
    dl_val = DataLoader(ds_val, batch_size=100)

    variational = False
    if "VaRONet" in config.net:
        variational = True

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
    #for index in [0, 5, 9, 10, 11, 26, 34, 44, 49, 64]:
    for index in range(10):
        test_results.plot_V(f"{fig_dir}/{index:02}_0_V_plot", index)
        test_results.plot_q(f"{fig_dir}/{index:02}_1_q_plot", index)
        test_results.plot_p(f"{fig_dir}/{index:02}_2_p_plot", index)
        test_results.plot_phase(f"{fig_dir}/{index:02}_3_phase_plot", index)
        #test_results.plot_compare_q(f"{fig_dir}/{index:02}_4_compare_q_plot", index)
        #test_results.plot_compare_p(f"{fig_dir}/{index:02}_5_compare_p_plot", index)
        #test_results.plot_compare_phase(f"{fig_dir}/{index:02}_6_compare_phase_plot", index)

    # Plot the worst result
    test_results.plot_V(f"{fig_dir}/{worst_idx:02}_0_V_plot", worst_idx)
    test_results.plot_q(f"{fig_dir}/{worst_idx:02}_1_q_plot", worst_idx)
    test_results.plot_p(f"{fig_dir}/{worst_idx:02}_2_p_plot", worst_idx)
    test_results.plot_phase(f"{fig_dir}/{worst_idx:02}_3_phase_plot", worst_idx)
    #test_results.plot_compare_q(f"{fig_dir}/{worst_idx:02}_4_compare_q_plot", worst_idx)
    #test_results.plot_compare_p(f"{fig_dir}/{worst_idx:02}_5_compare_p_plot", worst_idx)
    #test_results.plot_compare_phase(f"{fig_dir}/{worst_idx:02}_6_compare_phase_plot", worst_idx)

    # Additional custom analysis can be added here
    # ...

if __name__ == "__main__":
    main()
