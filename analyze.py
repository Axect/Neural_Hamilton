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
)


torch.set_float32_matmul_precision("medium")


def load_relevant_data(potential: str):
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

    q_gl4 = df[f"q_gl4"].to_numpy()
    p_gl4 = df[f"p_gl4"].to_numpy()
    ds_gl4 = TensorDataset(
        torch.tensor(V, dtype=torch.float32).unsqueeze(0),
        torch.tensor(t, dtype=torch.float32).unsqueeze(0),
        torch.tensor(q_gl4, dtype=torch.float32).unsqueeze(0),
        torch.tensor(p_gl4, dtype=torch.float32).unsqueeze(0),
    )
    return ds_gl4, ds, ds_rk4  # GL4, Y4, RK4


def load_precise_data():
    y4_ds = load_data("./data_precise/test.parquet")
    y8_ds = load_data("./data_precise/compare.parquet")
    return y4_ds, y8_ds


# ┌──────────────────────────────────────────────────────────┐
#  RK4 Solver
# └──────────────────────────────────────────────────────────┘
def rk4_step(f, y, t, dt):
    k1 = dt * f(y, t)
    k2 = dt * f(y + k1 / 2, t + dt / 2)
    k3 = dt * f(y + k2 / 2, t + dt / 2)
    k4 = dt * f(y + k3, t + dt)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def solve_hamilton(V, q0, p0, t):
    q = np.linspace(0, 1, 100)
    V_interp = PchipInterpolator(q, V)
    dVdq_interp = V_interp.derivative()

    def hamilton_eqs(y, _t):
        q, p = y
        dqdt = p
        dpdt = -dVdq_interp(q)
        return np.array([dqdt, dpdt])

    y0 = np.array([q0, p0])
    dt = t[1] - t[0]

    solution = np.zeros((len(t), 2))
    solution[0] = y0

    for i in range(1, len(t)):
        solution[i] = rk4_step(hamilton_eqs, solution[i - 1], t[i - 1], dt)

    return solution[:, 0], solution[:, 1]


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
            self.measure_rk4()

    def run_test(self):
        self.model.eval()

        V_vec = []
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

                loss_q_vec = F.mse_loss(q_pred, q, reduction="none")
                loss_p_vec = F.mse_loss(p_pred, p, reduction="none")
                loss_vec = 0.5 * (loss_q_vec + loss_p_vec)

                loss_q = loss_q_vec.mean(dim=1)
                loss_p = loss_p_vec.mean(dim=1)
                loss = loss_vec.mean(dim=1)

                V_vec.extend(V.cpu().numpy())
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

    def measure_rk4(self):
        t_total_vec = []
        for V, _, _, p in self.dl_val:
            for i in range(V.shape[0]):
                V_i = V[i].detach().cpu().numpy()
                t = np.linspace(0, 2, 100)
                t_start = time.time()
                _, _ = solve_hamilton(V_i, 0, 0, t)
                t_total_vec.append(time.time() - t_start)
        self.t_total_time_vec_rk4 = np.array(t_total_vec)

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
        times = self.t_total_time_vec_rk4
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
        t = np.linspace(0, 2, len(self.q_preds[index]))
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
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_p(self, name: str, index: int):
        t = np.linspace(0, 2, len(self.p_preds[index]))
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
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_phase(self, name: str, index: int):
        t = np.linspace(0, 2, len(self.p_preds[index]))
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
            ax.legend()
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
        t = np.linspace(0, 2, len(self.q_preds[index]))
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
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_p_expand(self, name: str, index: int):
        t = np.linspace(0, 2, len(self.p_preds[index]))
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
            ax.legend()
            fig.savefig(f"{name}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)


def main():
    # Test run
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
        ds_val = load_data("./data_test/test.parquet")
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
    elif test_option == "physical":
        potentials = {
            "sho": "SHO",
            "double_well": "DoubleWell",
            "morse": "Morse",
            "pendulum": "Pendulum",
            "mff": "MFF",
            "smff": "SMFF",
            "sawtooth": "Sawtooth",
        }
        results = [load_relevant_data(name) for name in potentials.keys()]
        ds_tests, ds_y4s, ds_rk4s = zip(*results)
        tests_name = list(potentials.values())

        for i in range(len(ds_tests)):
            print()
            print(f"Test {tests_name[i]}:")
            ds_test = ds_tests[i]
            ds_y4 = ds_y4s[i]
            ds_rk4 = ds_rk4s[i]
            test_name = tests_name[i]

            # 데이터로더 생성
            dl_test = DataLoader(ds_test, batch_size=1)
            dl_y4 = DataLoader(ds_y4, batch_size=1)
            dl_rk4 = DataLoader(ds_rk4, batch_size=1)

            # 테스트 결과 계산
            test_results = TestResults(model, dl_test, device, variational)
            test_results.print_results()

            # 결과 저장 디렉토리 생성
            fig_dir = f"figs/{project}/{test_name}"
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            # 기본 플롯 생성
            test_results.plot_V(f"{fig_dir}/{i:02}_{test_name}_0_V_plot", 0)
            test_results.plot_q(f"{fig_dir}/{i:02}_{test_name}_1_q_plot", 0)
            test_results.plot_p(f"{fig_dir}/{i:02}_{test_name}_2_p_plot", 0)
            test_results.plot_phase(f"{fig_dir}/{i:02}_{test_name}_3_phase_plot", 0)

            # 모든 방법에 대한 비교 그래프 생성
            for batch_data in zip(dl_test, dl_y4, dl_rk4):
                # 데이터 추출 및 형태 조정
                (_, _, q_gl4, p_gl4) = batch_data[0]
                (_, _, q_y4, p_y4) = batch_data[1]
                (_, _, q_rk4, p_rk4) = batch_data[2]

                # NumPy 배열로 변환
                q_gl4 = q_gl4.numpy().reshape(-1)
                p_gl4 = p_gl4.numpy().reshape(-1)
                q_y4 = q_y4.numpy().reshape(-1)
                p_y4 = p_y4.numpy().reshape(-1)
                q_rk4 = q_rk4.numpy().reshape(-1)
                p_rk4 = p_rk4.numpy().reshape(-1)
                q_test = test_results.q_preds
                p_test = test_results.p_preds

                # 손실 계산
                loss_q_test = np.mean(np.square(q_gl4 - q_test))
                loss_p_test = np.mean(np.square(p_gl4 - p_test))
                loss_test = 0.5 * (loss_q_test + loss_p_test)

                loss_q_y4 = np.mean(np.square(q_gl4 - q_y4))
                loss_p_y4 = np.mean(np.square(p_gl4 - p_y4))
                loss_y4 = 0.5 * (loss_q_y4 + loss_p_y4)

                loss_q_rk4 = np.mean(np.square(q_gl4 - q_rk4))
                loss_p_rk4 = np.mean(np.square(p_gl4 - p_rk4))
                loss_rk4 = 0.5 * (loss_q_rk4 + loss_p_rk4)

                # 손실 출력
                print(f"Test Loss: {loss_test:.4e}")
                print(f"Y4 Loss: {loss_y4:.4e}")
                print(f"RK4 Loss: {loss_rk4:.4e}")

                # 시간 배열 생성
                t = np.linspace(0, 2, len(q_gl4))
                cmap = plt.get_cmap("gist_heat")
                colors = cmap(np.linspace(0, 0.75, len(t)))

                # q 그래프 생성
                with plt.style.context(["science", "nature"]):
                    fig, ax = plt.subplots()

                    # GL4를 True로 취급 (회색 굵은 선)
                    ax.plot(
                        t,
                        q_gl4,
                        color="gray",
                        label=r"$q$ (GL4)",
                        alpha=0.5,
                        linewidth=1.75,
                        zorder=0,
                    )

                    # 나머지 데이터셋 (실선, 점선, 점실선)
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

                    # 예측값은 점으로 표시
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

                    # 손실 표시
                    ax.text(
                        0.05,
                        0.9,
                        f"Test Loss: {loss_q_test:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.85,
                        f"Y4 Loss: {loss_q_y4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.8,
                        f"RK4 Loss: {loss_q_rk4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )

                    # 범례는 선과 점을 합쳐서 표시하지 않도록 핸들 조정
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())

                    fig.savefig(
                        f"{fig_dir}/{i:02}_{test_name}_compare_q_plot.png",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

                    # p 그래프 생성
                    fig, ax = plt.subplots()

                    # GL4를 True로 취급 (회색 굵은 선)
                    ax.plot(
                        t,
                        p_gl4,
                        color="gray",
                        label=r"$p$ (GL4)",
                        alpha=0.5,
                        linewidth=1.75,
                        zorder=0,
                    )

                    # 나머지 데이터셋 (실선, 점선, 점실선)
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

                    # 예측값은 점으로 표시
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

                    # 손실 표시
                    ax.text(
                        0.05,
                        0.1,
                        f"Test Loss: {loss_p_test:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.05,
                        f"Y4 Loss: {loss_p_y4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )
                    ax.text(
                        0.05,
                        0.0,
                        f"RK4 Loss: {loss_p_rk4:.4e}",
                        transform=ax.transAxes,
                        fontsize=5,
                    )

                    # 범례는 선과 점을 합쳐서 표시하지 않도록 핸들 조정
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())

                    fig.savefig(
                        f"{fig_dir}/{i:02}_{test_name}_compare_p_plot.png",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

                    # 위상 그래프 생성
                    fig, ax = plt.subplots()

                    # GL4를 True로 취급 (회색 굵은 선)
                    ax.plot(
                        q_gl4,
                        p_gl4,
                        color="gray",
                        label=r"$(q,p)$ (GL4)",
                        alpha=0.5,
                        linewidth=1.75,
                        zorder=0,
                    )

                    # 나머지 데이터셋 (실선, 점선, 점실선)
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

                    # 예측값은 점으로 표시
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

                    # 손실 표시
                    ax.text(
                        0.05,
                        0.5,
                        f"Test Loss: {loss_test:.4e}",
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

                    # 범례는 선과 점을 합쳐서 표시하지 않도록 핸들 조정
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys())

                    fig.savefig(
                        f"{fig_dir}/{i:02}_{test_name}_compare_phase_plot.png",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close(fig)


if __name__ == "__main__":
    console = Console()
    main()
