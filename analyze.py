import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.integrate import odeint
from scipy.interpolate import PchipInterpolator

import os

from util import select_project, select_group, select_seed, select_device, load_model, load_data, load_study, load_best_model

class TestResults:
    def __init__(self, model, dl_val, device, variational=False):
        self.model = model
        self.dl_val = dl_val
        self.device = device
        self.variational = variational
        
        self.run_test()

    def run_test(self):
        self.model.eval()

        V_vec = []
        x_preds = []
        p_preds = []
        x_targets = []
        p_targets = []
        rk4_x_vec = []
        rk4_p_vec = []

        total_loss_x_vec = []
        total_loss_p_vec = []
        total_loss_vec = []
        rk4_loss_x_vec = []
        rk4_loss_p_vec = []
        rk4_loss_vec = []

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
                
                V_vec.extend(V.cpu().numpy())
                x_preds.extend(x_pred.cpu().numpy())
                p_preds.extend(p_pred.cpu().numpy())
                x_targets.extend(x.cpu().numpy())
                p_targets.extend(p.cpu().numpy())
                total_loss_vec.extend(loss_vec.cpu().numpy())
                total_loss_x_vec.extend(loss_x_vec.cpu().numpy())
                total_loss_p_vec.extend(loss_p_vec.cpu().numpy())

                ## Compute RK4 solution for comparison
                #t_np = t.cpu().numpy()
                #for i, V_single in enumerate(V):
                #    rk4_x, rk4_p = self.solve_hamilton(V_single.cpu().numpy(), 0, 0, t_np[0])
                #    rk4_loss_x = F.mse_loss(torch.tensor(rk4_x), x[i, :].cpu(), reduction="none")
                #    rk4_loss_p = F.mse_loss(torch.tensor(rk4_p), p[i, :].cpu(), reduction="none")
                #    rk4_loss = 0.5 * (rk4_loss_x + rk4_loss_p)
                #    rk4_x_vec.append(rk4_x)
                #    rk4_p_vec.append(rk4_p)
                #    rk4_loss_x_vec.append(rk4_loss_x)
                #    rk4_loss_p_vec.append(rk4_loss_p)
                #    rk4_loss_vec.append(rk4_loss)

        self.total_loss_vec = np.array(total_loss_vec)
        self.total_loss_x_vec = np.array(total_loss_x_vec)
        self.total_loss_p_vec = np.array(total_loss_p_vec)
        self.V_vec = np.array(V_vec)
        self.x_preds = np.array(x_preds)
        self.p_preds = np.array(p_preds)
        self.x_targets = np.array(x_targets)
        self.p_targets = np.array(p_targets)
        #self.rk4_x = np.array(rk4_x_vec)
        #self.rk4_p = np.array(rk4_p_vec)
        #self.rk4_loss_x_vec = np.array(rk4_loss_x_vec)
        #self.rk4_loss_p_vec = np.array(rk4_loss_p_vec)
        #self.rk4_loss_vec = np.array(rk4_loss_vec)

    @staticmethod
    def rk4_step(f, y, t, dt):
        k1 = f(y, t)
        k2 = f(y + k1 * dt / 2, t + dt / 2)
        k3 = f(y + k2 * dt / 2, t + dt / 2)
        k4 = f(y + k3 * dt, t + dt)
        return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve_hamilton(self, V, x0, p0, t):
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
            solution[i] = self.rk4_step(hamilton_eqs, solution[i - 1], t[i - 1], dt)
        return solution[:, 0], solution[:, 1]

    def print_results(self):
        print(f"Total Loss: {self.total_loss_vec.mean():.4e}")
        print(f"Total Loss x: {self.total_loss_x_vec.mean():.4e}")
        print(f"Total Loss p: {self.total_loss_p_vec.mean():.4e}")

    def print_rk4_results(self):
        print(f"RK4 Total Loss: {self.rk4_loss_vec.mean():.4e}")
        print(f"RK4 Total Loss x: {self.rk4_loss_x_vec.mean():.4e}")
        print(f"RK4 Total Loss p: {self.rk4_loss_p_vec.mean():.4e}")

    def hist_loss(self, name:str):
        losses = self.total_loss_vec.mean(axis=1)
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            logbins = np.logspace(np.log10(losses.min()), np.log10(losses.max()), 100)
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
        loss_x = F.mse_loss(torch.tensor(self.x_preds[index]), torch.tensor(self.x_targets[index]))
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
        loss_p = F.mse_loss(torch.tensor(self.p_preds[index]), torch.tensor(self.p_targets[index]))
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(t, self.p_targets[index], color='gray', label=r"$p$", alpha=0.65, linewidth=1.75)
            ax.plot(t, self.p_preds[index], ':', color='red', label=r"$\hat{p}$")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$p(t)$")
            ax.autoscale(tight=True)
            ax.text(0.05, 0.9, f"Loss: {loss_p:.4e}", transform=ax.transAxes, fontsize=5)
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_phase(self, name:str, index:int):
        loss_x = F.mse_loss(torch.tensor(self.x_preds[index]), torch.tensor(self.x_targets[index]))
        loss_p = F.mse_loss(torch.tensor(self.p_preds[index]), torch.tensor(self.p_targets[index]))
        loss = 0.5 * (loss_x + loss_p)
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            ax.plot(self.x_targets[index], self.p_targets[index], color='gray', label=r"$(q,p)$", alpha=0.65, linewidth=1.75)
            ax.plot(self.x_preds[index], self.p_preds[index], ':', color='red', label=r"$(\hat{q}, \hat{p})$")
            ax.set_xlabel(r"$q$")
            ax.set_ylabel(r"$p$")
            ax.autoscale(tight=True)
            ax.text(0.05, 0.9, f"Loss: {loss:.4e}", transform=ax.transAxes, fontsize=5)
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_compare_q(self, name:str, index:int):
        t = np.linspace(0, 1, len(self.x_preds[index]))
        loss_nn = F.mse_loss(torch.tensor(self.x_preds[index]), torch.tensor(self.x_targets[index]))
        loss_rk4 = F.mse_loss(torch.tensor(self.rk4_x[index]), torch.tensor(self.x_targets[index]))
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
        loss_nn = F.mse_loss(torch.tensor(self.p_preds[index]), torch.tensor(self.p_targets[index]))
        loss_rk4 = F.mse_loss(torch.tensor(self.rk4_p[index]), torch.tensor(self.p_targets[index]))
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
    #test_results.print_rk4_results()

    fig_dir = f"figs/{project}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Histogram for loss
    test_results.hist_loss(f"{fig_dir}/00_0_loss_hist")

    losses = test_results.total_loss_vec.mean(axis=1)
    worst_idx = int(np.argmax(losses))

    # Plot the results
    #for index in [0, 5, 9, 10, 11, 26, 34, 44, 49, 64]:
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

    # Additional custom analysis can be added here
    # ...

if __name__ == "__main__":
    main()
