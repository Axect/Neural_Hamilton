import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os

from util import select_project, select_group, select_seed, select_device, load_model, load_data, load_study, load_best_model


class TestResults:
    def __init__(self, total_loss_vec, total_loss_x_vec, total_loss_p_vec, V_vec, x_preds, p_preds, x_targets, p_targets):
        self.total_loss_vec = np.array(total_loss_vec)
        self.total_loss_x_vec = np.array(total_loss_x_vec)
        self.total_loss_p_vec = np.array(total_loss_p_vec)
        self.V_vec = V_vec
        self.x_preds = x_preds
        self.p_preds = p_preds
        self.x_targets = x_targets
        self.p_targets = p_targets

    def print(self):
        print(f"Total Loss: {self.total_loss_vec.mean():.4e}")
        print(f"Total Loss x: {self.total_loss_x_vec.mean():.4e}")
        print(f"Total Loss p: {self.total_loss_p_vec.mean():.4e}")

    def hist_loss(self, name:str):
        losses = self.total_loss_vec.mean(axis=1)
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            logbins = np.logspace(np.log10(losses.min()), np.log10(losses.max()), 100)
            ax.hist(self.total_loss_vec.mean(axis=1), bins=logbins)
            ax.axvline(losses.sum() / losses.shape[0], color='red', linestyle='--')
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


def test_model(model, dl_val, device, variational=False):
    model.eval()
    total_loss = 0
    total_loss_x = 0
    total_loss_p = 0
    total_loss_vec = []
    total_loss_x_vec = []
    total_loss_p_vec = []
    V_vec = []
    x_preds = []
    p_preds = []
    x_targets = []
    p_targets = []
    with torch.no_grad():
        for V, t, x, p in dl_val:
            V, t, x, p = V.to(device), t.to(device), x.to(device), p.to(device)
            if not variational:
                x_pred, p_pred = model(V, t)
            else:
                x_pred, p_pred, _, _ = model(V, t)
            loss_x_vec = F.mse_loss(x_pred, x, reduction="none")
            loss_p_vec = F.mse_loss(p_pred, p, reduction="none")
            loss_vec = 0.5 * (loss_x_vec + loss_p_vec)
            loss_x = F.mse_loss(x_pred, x)
            loss_p = F.mse_loss(p_pred, p)
            loss = 0.5 * (loss_x + loss_p)
            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_p += loss_p.item()
            V_vec.extend(V.cpu().numpy())
            x_preds.extend(x_pred.cpu().numpy())
            p_preds.extend(p_pred.cpu().numpy())
            x_targets.extend(x.cpu().numpy())
            p_targets.extend(p.cpu().numpy())
            total_loss_vec.extend(loss_vec.cpu().numpy())
            total_loss_x_vec.extend(loss_x_vec.cpu().numpy())
            total_loss_p_vec.extend(loss_p_vec.cpu().numpy())

    total_loss = total_loss / len(dl_val)
    total_loss_x = total_loss_x / len(dl_val)
    total_loss_p = total_loss_p / len(dl_val)

    print(np.array(total_loss_vec).shape)
    print(np.array(total_loss_x_vec).shape)
    print(np.array(total_loss_p_vec).shape)
    print(np.array(V_vec).shape)
    print(np.array(x_preds).shape)
    print(np.array(p_preds).shape)
    print(np.array(x_targets).shape)
    print(np.array(p_targets).shape)

    test_results = TestResults(total_loss_vec, total_loss_x_vec, total_loss_p_vec, V_vec, x_preds, p_preds, x_targets, p_targets)
    return test_results


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

    test_results = test_model(model, dl_val, device, variational)
    test_results.print()

    fig_dir = f"figs/{project}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Histogram for loss
    test_results.hist_loss(f"{fig_dir}/000_loss_hist")

    losses = test_results.total_loss_vec.mean(axis=1)
    worst_idx = np.argmax(losses)

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
