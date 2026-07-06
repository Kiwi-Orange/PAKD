"""Build an A4-ready Nature-style main figure for the Fisher-KPP example.

The figure is redrawn from Fisher-KPP data/checkpoints with the same visual
style used by the MMReaction and POLLU main figures. Existing result figures
are not modified.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fisher_kpp_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from Fisher_KPP_simulation import SimulationConfig, solve_fisher_kpp
from models import MLP, ResidualMLP


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "nature_figure"
PANEL_DIR = OUT_DIR / "panels"

TEACHER_MODEL = ROOT / "models" / "ResidualMLP_fisher_kpp_1cond_n100_best.pt"
STUDENT_MODEL = (
    ROOT
    / "models"
    / "students"
    / "student_PAKD_ResidualMLP_from_teacher_high_res_fisher_kpp_1cond_5000times_n100_blocks1_wp7.0_lasthidden.pt"
)
GAMMA_DATA = ROOT / "data" / "fisher_kpp" / "teacher_high_res_fisher_kpp_1cond_5000times_n100_with_gammas.npy"
METADATA = ROOT / "data" / "fisher_kpp" / "teacher_high_res_fisher_kpp_1cond_5000times_n100_metadata.npz"
DYNAMICS_DATA = ROOT / "results" / "student_dynamics" / "ResidualMLP_step_two_regime" / "discovered_dynamics.npz"

N_GRID = 100
EPSILON = 0.01
TRUE_R = 1.0
TIME_POINTS = np.logspace(-4, 1, 700)

COLORS = {
    "truth": "#111111",
    "teacher": "#1F78B4",
    "student": "#D62728",
    "fast": "#E64B35",
    "slow": "#3C8DBC",
    "hidden": "#6A51A3",
    "output": "#F28E2B",
    "fit": "#2CA25F",
    "grid": "#D8DDE6",
    "text": "#1F2933",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "font.weight": "bold",
            "axes.labelsize": 8.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 1.05,
            "lines.linewidth": 2.1,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.tick_params(axis="both", which="major", width=0.9, length=3.0, pad=1.5)
    ax.tick_params(axis="both", which="minor", width=0.6, length=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
    if grid:
        ax.grid(True, which="major", color=COLORS["grid"], lw=0.45, alpha=0.65)


def set_sparse_log_time_ticks(ax: plt.Axes) -> None:
    ax.set_xticks([1e-4, 1e-2, 1e0, 1e1])
    ax.get_xaxis().set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def set_sparse_log_y_ticks(ax: plt.Axes) -> None:
    ax.set_yticks([1e-4, 1e-2, 1e0, 1e1])
    ax.get_yaxis().set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.075, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="black",
    )


def count_residual_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    block_ids = set()
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    return max(block_ids) + 1 if block_ids else 1


def infer_mlp_hidden_sizes(state_dict: dict[str, torch.Tensor]) -> list[int]:
    hidden = []
    for key, value in state_dict.items():
        if key.startswith("network.") and key.endswith(".weight") and value.ndim == 2:
            layer_idx = int(key.split(".")[1])
            if layer_idx != max(
                int(k.split(".")[1])
                for k, v in state_dict.items()
                if k.startswith("network.") and k.endswith(".weight") and v.ndim == 2
            ):
                hidden.append(int(value.shape[0]))
    return hidden or [128]


def make_model(checkpoint: dict, is_student: bool) -> torch.nn.Module:
    state_dict = checkpoint["model_state_dict"]
    training_args = checkpoint.get("training_args", {})
    model_type = checkpoint.get("model_type", training_args.get("student_type", "ResidualMLP"))
    input_size = int(checkpoint.get("input_size", N_GRID + 1))
    output_size = int(checkpoint.get("output_size", checkpoint.get("n_grid", N_GRID)))

    if model_type == "ResidualMLP":
        hidden_dim = checkpoint.get("hidden_dim", training_args.get("student_hidden_dim"))
        if hidden_dim is None:
            hidden_dim = int(state_dict["input_proj.weight"].shape[0])
        if is_student:
            num_blocks = checkpoint.get("num_blocks", training_args.get("student_num_blocks"))
        else:
            num_blocks = checkpoint.get("num_layers", training_args.get("num_layers"))
        if num_blocks is None:
            num_blocks = count_residual_blocks(state_dict)
        dropout = checkpoint.get("dropout", training_args.get("student_dropout", 0.0))
        return ResidualMLP(
            input_size=input_size,
            output_size=output_size,
            hidden_dim=int(hidden_dim),
            num_blocks=int(num_blocks),
            dropout=float(dropout),
        )

    if model_type == "MLP":
        hidden_sizes = checkpoint.get("hidden_sizes")
        if hidden_sizes is None:
            hidden_dim = checkpoint.get("hidden_dim", training_args.get("student_hidden_dim"))
            num_layers = checkpoint.get("num_layers", training_args.get("student_num_blocks"))
            if hidden_dim is not None and num_layers is not None:
                hidden_sizes = [int(hidden_dim)] * int(num_layers)
            else:
                hidden_sizes = infer_mlp_hidden_sizes(state_dict)
        dropout = checkpoint.get("dropout", training_args.get("student_dropout", 0.0))
        return MLP(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes, dropout=float(dropout))

    raise ValueError(f"Unsupported model type: {model_type}")


def load_checkpoint_model(path: Path, is_student: bool) -> tuple[torch.nn.Module, object, object, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(checkpoint, is_student=is_student)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["X_scaler"], checkpoint.get("y_scaler"), checkpoint


def base_initial_condition() -> np.ndarray:
    metadata = np.load(METADATA)
    return np.asarray(metadata["initial_conditions"][0], dtype=float)


def analytical_solution(initial_condition: np.ndarray, time_points: np.ndarray = TIME_POINTS) -> tuple[np.ndarray, np.ndarray]:
    config = SimulationConfig(
        n_interior=len(initial_condition),
        epsilon=EPSILON,
        t_span=(0.0, float(time_points[-1])),
        n_time_points=len(time_points),
        atol=1e-10,
        rtol=1e-8,
        solver_method="BDF",
    )
    sol = solve_fisher_kpp(config, initial_condition, t_eval=time_points, use_jacobian=True)
    if not sol["success"]:
        raise RuntimeError("BDF Fisher-KPP solve failed")
    return sol["x"], np.maximum(sol["u"].T, 0.0)


def model_predict(
    model: torch.nn.Module,
    x_scaler,
    y_scaler,
    initial_condition: np.ndarray,
    time_points: np.ndarray = TIME_POINTS,
) -> np.ndarray:
    X = np.zeros((len(time_points), len(initial_condition) + 1), dtype=np.float32)
    X[:, 0] = np.log10(time_points + 1.0)
    X[:, 1:] = initial_condition.astype(np.float32)
    Xn = x_scaler.transform(X)
    with torch.no_grad():
        pred = model(torch.tensor(Xn, dtype=torch.float32)).cpu().numpy()
    if y_scaler is not None:
        pred = y_scaler.inverse_transform(pred)
    return np.clip(pred, 0.0, 1.05)


def compute_trajectories(teacher_bundle, student_bundle) -> dict[str, np.ndarray]:
    initial_condition = base_initial_condition()
    x_grid, truth = analytical_solution(initial_condition)
    teacher = model_predict(teacher_bundle[0], teacher_bundle[1], teacher_bundle[2], initial_condition)
    student = model_predict(student_bundle[0], student_bundle[1], student_bundle[2], initial_condition)
    return {
        "x": x_grid,
        "initial": initial_condition,
        "truth": truth,
        "teacher": teacher,
        "student": student,
    }


def plot_teacher_heatmaps(fig: plt.Figure, spec, trajectories: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(2, 3, hspace=0.10, wspace=0.04)
    axes = [
        [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1]), fig.add_subplot(sub[0, 2])],
        [fig.add_subplot(sub[1, 0]), fig.add_subplot(sub[1, 1]), fig.add_subplot(sub[1, 2])],
    ]

    x_grid = trajectories["x"]
    truth = trajectories["truth"]
    teacher = trajectories["teacher"]
    student = trajectories["student"]
    err_student_truth = np.abs(student - truth)
    err_teacher_truth = np.abs(teacher - truth)
    err_student_teacher = np.abs(student - teacher)
    # Shared error vmax: max of the three 95th percentiles
    err_vmax = max(
        np.percentile(err_student_truth, 95),
        np.percentile(err_teacher_truth, 95),
        np.percentile(err_student_teacher, 95),
        0.02,
    )

    heatmaps = [
        [(truth, "Truth", "viridis", mpl.colors.Normalize(vmin=0.0, vmax=1.0)),
         (teacher, "Teacher", "viridis", mpl.colors.Normalize(vmin=0.0, vmax=1.0)),
         (student, "Student", "viridis", mpl.colors.Normalize(vmin=0.0, vmax=1.0))],
        [(err_teacher_truth, "|Teacher − Truth|", "hot", mpl.colors.Normalize(vmin=0.0, vmax=err_vmax)),
         (err_student_truth, "|Student − Truth|", "hot", mpl.colors.Normalize(vmin=0.0, vmax=err_vmax)),
         (err_student_teacher, "|Student − Teacher|", "hot", mpl.colors.Normalize(vmin=0.0, vmax=err_vmax))],
    ]

    im_viridis = None
    im_hot = None

    for row in range(2):
        for col in range(3):
            ax = axes[row][col]
            values, title, cmap, norm = heatmaps[row][col]
            im = ax.pcolormesh(
                x_grid,
                TIME_POINTS,
                values,
                shading="auto",
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
            if cmap == "viridis" and im_viridis is None:
                im_viridis = im
            if cmap == "hot" and im_hot is None:
                im_hot = im
            ax.set_yscale("log")
            ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
            ax.set_ylim(TIME_POINTS[0], TIME_POINTS[-1])
            ax.set_xticks([0.0, 0.5, 1.0])
            if row == 1:
                if col == 0:
                    ax.set_xticklabels(["0.0", "0.5", ""])
                elif col == 2:
                    ax.set_xticklabels(["", "0.5", "1.0"])
                else:
                    ax.set_xticklabels(["", "0.5", ""])
            else:
                ax.tick_params(labelbottom=False)
            set_sparse_log_y_ticks(ax)
            ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([1e-4, 1e-2, 1e0, 1e1]))
            style_axis(ax, grid=False)
            if row == 1:
                ax.set_xlabel(r"$x$", labelpad=1.5)
            if col == 0:
                ax.set_ylabel(r"$t$", labelpad=1.5)
            else:
                ax.tick_params(labelleft=False)
            # 内部标签（左上角）
            ax.text(
                0.04,
                0.96,
                title,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                fontweight="bold",
                color="white",
                bbox=dict(facecolor="black", edgecolor="none", alpha=0.55, pad=0.3),
            )

    # Compact colorbars on the right of the last column
    cax_v = axes[0][2].inset_axes([1.02, 0.0, 0.03, 1.0])
    cb_v = fig.colorbar(im_viridis, cax=cax_v)
    cb_v.set_label(r"$u$", rotation=0, labelpad=1.5, fontsize=6.5, fontweight="bold", va="center")
    cb_v.ax.tick_params(labelsize=5.0, width=0.4, length=2)
    cb_v.outline.set_linewidth(0.5)

    cax_h = axes[1][2].inset_axes([1.02, 0.0, 0.03, 1.0])
    cb_h = fig.colorbar(im_hot, cax=cax_h)
    cb_h.set_label("|error|", rotation=90, labelpad=1.5, fontsize=6.5, fontweight="bold")
    cb_h.ax.tick_params(labelsize=5.0, width=0.4, length=2)
    cb_h.outline.set_linewidth(0.5)

    # Panel label 和 title 由 build_figure 统一放置


def plot_hmm_panel(ax: plt.Axes) -> None:
    data = np.load(GAMMA_DATA)
    time = np.maximum(data[:, 0], TIME_POINTS[0])
    gammas = data[:, -2:]
    raw_phase = np.argmax(gammas, axis=1)
    medians = {phase: np.median(time[raw_phase == phase]) for phase in np.unique(raw_phase)}
    fast_phase = min(medians, key=medians.get)
    slow_phase = max(medians, key=medians.get)
    fast_gamma = gammas[:, fast_phase]
    slow_gamma = gammas[:, slow_phase]

    ax.fill_between(time, 0.0, 1.0, where=fast_gamma >= slow_gamma, color=COLORS["fast"], alpha=0.10, lw=0)
    ax.fill_between(time, 0.0, 1.0, where=slow_gamma > fast_gamma, color=COLORS["slow"], alpha=0.10, lw=0)
    ax.semilogx(time, fast_gamma, color=COLORS["fast"], lw=2.3, label="Fast")
    ax.semilogx(time, slow_gamma, color=COLORS["slow"], lw=2.3, label="Slow")

    switch_idx = np.where(np.diff((slow_gamma > fast_gamma).astype(int)) != 0)[0]
    if len(switch_idx):
        t_star = float(time[switch_idx[0] + 1])
        ax.axvline(t_star, color="black", lw=1.1, ls=":", alpha=0.85)
        ax.text(
            t_star * 1.06,
            0.54,
            r"$t_{\mathrm{HMM}}$",
            fontsize=6.8,
            fontweight="bold",
            ha="left",
            va="bottom",
            rotation=90,
        )

    ax.set_xlim(TIME_POINTS[0], TIME_POINTS[-1])
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(r"$t$", labelpad=1.5)
    ax.set_ylabel("Phase prob.", labelpad=1.5)
    set_sparse_log_time_ticks(ax)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.92), handlelength=1.4, labelspacing=0.2)
    style_axis(ax)
    # Panel label 和 title 由 build_figure 统一放置


def plot_pakd_loss(ax: plt.Axes, student_checkpoint: dict) -> None:
    losses = student_checkpoint["training_losses"]
    epochs = np.arange(1, len(losses["total"]) + 1)
    curves = [
        ("total", "Total", COLORS["teacher"]),
        ("output", "Output", COLORS["output"]),
        ("hidden", "Hidden", COLORS["hidden"]),
    ]
    for key, label, color in curves:
        ax.semilogy(epochs, np.asarray(losses[key], dtype=float), color=color, lw=2.25, label=label)
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Loss", labelpad=1.5)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, labelspacing=0.18)
    style_axis(ax)
    # Panel label 和 title 由 build_figure 统一放置


def plot_student_profiles(fig: plt.Figure, spec, trajectories: dict[str, np.ndarray], dynamics: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(2, 2, hspace=0.25, wspace=0.18)
    axes = []
    x_grid = trajectories["x"]
    truth = trajectories["truth"]
    teacher = trajectories["teacher"]
    student = trajectories["student"]
    t_transition = float(dynamics["t_transition"])
    target_times = [TIME_POINTS[0], t_transition, 0.7, TIME_POINTS[-1]]
    marker_idx = np.unique(np.linspace(0, len(x_grid) - 1, 22, dtype=int))

    for i, target in enumerate(target_times):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        axes.append(ax)
        idx = int(np.argmin(np.abs(TIME_POINTS - target)))
        ax.plot(
            x_grid[marker_idx],
            truth[idx, marker_idx],
            ls="none",
            marker="o",
            ms=3.4,
            mfc="white",
            mec=COLORS["truth"],
            mew=0.9,
            color=COLORS["truth"],
            label="Truth" if i == 0 else None,
        )
        ax.plot(x_grid, teacher[idx], color=COLORS["teacher"], lw=2.0, ls="--", label="Teacher" if i == 0 else None)
        ax.plot(x_grid, student[idx], color=COLORS["student"], lw=2.35, ls="-", label="Student" if i == 0 else None)
        ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
        ax.set_ylim(-0.055, 1.08)
        ax.set_xticks([0.0, 0.5, 1.0])
        if i % 2 == 0:
            ax.set_xticklabels(["0.0", "0.5", ""])
        else:
            ax.set_xticklabels(["", "0.5", "1.0"])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        style_axis(ax)
        # Time label inside the plot
        # t=10 (last subplot, i=3) at bottom-center; others at top-right
        if i == 3:
            ax.text(
                0.50,
                0.08,
                rf"$t={TIME_POINTS[idx]:.2g}$",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=COLORS["text"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.3),
            )
        else:
            ax.text(
                0.96,
                0.92,
                rf"$t={TIME_POINTS[idx]:.2g}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color=COLORS["text"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.3),
            )
        if i // 2 == 1:
            ax.set_xlabel(r"$x$", labelpad=1.5)
        else:
            ax.tick_params(labelbottom=False)
        if i % 2 == 0:
            ax.set_ylabel(r"$u(x,t)$", labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    # d 图：图例放到 t=0.0001 子图(axes[0])右侧中间
    axes[0].legend(
        handles,
        labels,
        loc="center right",
        ncol=1,
        frameon=False,
        handlelength=1.25,
        borderpad=0.0,
        labelspacing=0.15,
        columnspacing=0.75,
    )
    # Panel label 和 title 由 build_figure 统一放置


def two_regime_values(times: np.ndarray, t_transition: float, fast: float, slow: float) -> np.ndarray:
    return np.where(times < t_transition, fast, slow)


def plot_knowledge_discovery(fig: plt.Figure, spec, dynamics: dict[str, np.ndarray]) -> None:
    times = np.asarray(dynamics["times"], dtype=float)
    time_mask = np.isfinite(times) & (times > 0)
    times = times[time_mask]
    D_t = np.asarray(dynamics["D_t"], dtype=float)[time_mask]
    r_t = np.asarray(dynamics["r_t"], dtype=float)[time_mask]
    t_transition = float(dynamics["t_transition"])
    D_fast = float(dynamics.get("D_fast", 0.0))
    r_fast = float(dynamics.get("r_fast", 0.0))
    D_slow = float(dynamics["D_slow"])
    r_slow = float(dynamics["r_slow"])

    sub = spec.subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.22)
    axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[1, 0])]

    # x 轴从 10^{-3} 开始，与 Extracted 数据实际起点对齐，消除左侧空白
    t_start = 1e-3
    t_line = np.logspace(np.log10(t_start), np.log10(times.max()), 500)
    panels = [
        (D_t, two_regime_values(t_line, t_transition, D_fast, D_slow), EPSILON, r"$D(t)$", (-0.001, 0.013)),
        (r_t, two_regime_values(t_line, t_transition, r_fast, r_slow), TRUE_R, r"$r(t)$", (-0.05, 1.2)),
    ]

    for i, (values, fit_values, true_value, ylabel, ylim) in enumerate(panels):
        ax = axes[i]
        ax.semilogx(times, values, color=COLORS["teacher"], lw=0.0, marker="o", ms=2.1, alpha=0.42, label="Extracted")
        ax.semilogx(t_line, fit_values, color=COLORS["fit"], lw=2.4, label="Two-regime")
        ax.axhline(true_value, color=COLORS["truth"], lw=1.6, ls="--", label="True")
        ax.axvline(t_transition, color=COLORS["student"], lw=1.1, ls=":", alpha=0.90)
        ax.text(
            t_transition * 1.08,
            ylim[0] + 0.55 * (ylim[1] - ylim[0]),
            r"$t^*$",
            color=COLORS["student"],
            fontsize=7.0,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="left",
        )
        ax.set_xlim(t_start, times.max())
        # 根据实际 x 范围重新设置刻度，避免 10^{-4} 把左边界拉回去
        ax.set_xticks([1e-3, 1e-2, 1e0, 1e1])
        ax.get_xaxis().set_major_formatter(mpl.ticker.LogFormatterMathtext())
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, labelpad=1.5)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        style_axis(ax)
        if i == 1:
            ax.set_xlabel(r"$t$", labelpad=1.5)
        else:
            ax.tick_params(labelbottom=False)

    handles = [
        Line2D([0], [0], marker="o", color=COLORS["teacher"], lw=0.0, ms=3.0, alpha=0.42, label="Extracted"),
        Line2D([0], [0], color=COLORS["fit"], lw=2.2, label="Two-regime"),
        Line2D([0], [0], color=COLORS["truth"], lw=1.6, ls="--", label="True"),
    ]
    # e 图：图例放到 D(t) 子图(axes[0])左上角，0.01 直线之下
    axes[0].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.72),
        ncol=1,
        frameon=False,
        handlelength=1.1,
        borderpad=0.0,
        labelspacing=0.12,
        columnspacing=0.38,
        fontsize=5.8,
    )


def save_panel_previews(fig: plt.Figure) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PANEL_DIR / "full_preview.png", facecolor="white", dpi=300)


def build_figure() -> None:
    warnings.filterwarnings("ignore")
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints...", flush=True)
    teacher_bundle = load_checkpoint_model(TEACHER_MODEL, is_student=False)
    student_bundle = load_checkpoint_model(STUDENT_MODEL, is_student=True)
    student_checkpoint = student_bundle[3]

    print("Solving BDF and evaluating teacher/student...", flush=True)
    trajectories = compute_trajectories(teacher_bundle, student_bundle)
    dynamics = dict(np.load(DYNAMICS_DATA))

    fig = plt.figure(figsize=(8.5, 10.8))
    outer = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.0, 1.05], width_ratios=[1.08, 0.92], hspace=0.22, wspace=0.20)

    plot_teacher_heatmaps(fig, outer[0, :], trajectories)
    ax_b = fig.add_subplot(outer[1, 0])
    plot_hmm_panel(ax_b)
    ax_c = fig.add_subplot(outer[1, 1])
    plot_pakd_loss(ax_c, student_checkpoint)
    plot_student_profiles(fig, outer[2, 0], trajectories, dynamics)
    plot_knowledge_discovery(fig, outer[2, 1], dynamics)

    # 统一放置 panel label 和标题（全局对齐，紧贴子图顶部）
    panel_info = [
        (outer[0, :], "a", "Surrogate validation & error analysis"),
        (outer[1, 0], "b", "HMM phase discovery"),
        (outer[1, 1], "c", "PAKD training"),
        (outer[2, 0], "d", "PAKD student validation"),
        (outer[2, 1], "e", "Equation discovery"),
    ]
    for cell, label, title in panel_info:
        bbox = cell.get_position(fig)
        y = bbox.y1 + 0.005
        x_label = bbox.x0 - 0.022
        x_title = bbox.x0
        fig.text(x_label, y, label, fontsize=13, fontweight="bold", ha="left", va="bottom")
        fig.text(x_title, y, title, fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"fisher_kpp_nature_main.{ext}", facecolor="white")
    save_panel_previews(fig)
    plt.close(fig)
    print(f"Saved main figure to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    build_figure()
