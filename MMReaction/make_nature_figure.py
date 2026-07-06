"""Build an A4-ready Nature-style main figure for the MMReaction example.

The figure is redrawn from project data/checkpoints with one shared visual
style. Existing result PDFs are not modified.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mmreaction_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from models import MLP, ResidualMLP


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "nature_figure"

TEACHER_MODEL = ROOT / "models" / "ResidualMLP_ResidualMLP_multi_50cond_normalized_best.pt"
STUDENT_MODEL = (
    ROOT
    / "models"
    / "students"
    / "student_PAKD_ResidualMLP_from_kd_complete_multiple_50_conditions_blocks1_wp7.0_lasthidden.pt"
)
STUDENT_MODEL_V2 = (
    ROOT
    / "models"
    / "students"
    / "student_PAKDv2_ResidualMLP_from_kd_complete_multiple_50_conditions_blocks1_wp5.0_exempt5_lasthidden.pt"
)
GAMMA_DATA = ROOT / "data" / "kd" / "kd_complete_multiple_50_conditions_with_gammas.npy"

SPECIES = ["E", "S", "ES", "P"]
TIME_POINTS = np.logspace(-8, 2, 500)
PARAMS = {"k1": 100.0, "km1": 10.0, "k2": 1.0}

COLORS = {
    "truth": "#111111",
    "teacher": "#1F78B4",
    "student": "#D62728",
    "qssa": "#4E9F50",
    "fast": "#E64B35",
    "slow": "#3C8DBC",
    "hidden": "#6A51A3",
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
        spine.set_visible(True)
        spine.set_linewidth(1.15)
    if grid:
        ax.grid(True, which="major", color=COLORS["grid"], lw=0.45, alpha=0.65)


def set_sparse_time_ticks(ax: plt.Axes, include_late: bool = False) -> None:
    ticks = [1e-8, 1e-5, 1e-2, 1e1]
    if include_late:
        ticks.append(1e2)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


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


def mm_reaction(_t: float, y: np.ndarray) -> list[float]:
    k1, km1, k2 = PARAMS["k1"], PARAMS["km1"], PARAMS["k2"]
    E, S, ES, _P = y
    return [
        -k1 * E * S + km1 * ES + k2 * ES,
        -k1 * E * S + km1 * ES,
        k1 * E * S - km1 * ES - k2 * ES,
        k2 * ES,
    ]


def analytical_solution(E0: float, S0: float) -> np.ndarray:
    sol = solve_ivp(
        mm_reaction,
        (TIME_POINTS[0], TIME_POINTS[-1]),
        [E0, S0, 0.0, 0.0],
        method="BDF",
        t_eval=TIME_POINTS,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Analytical solve failed: {sol.message}")
    return sol.y.T


def qssa_solution(E0: float, S0: float) -> np.ndarray:
    k1, km1, k2 = PARAMS["k1"], PARAMS["km1"], PARAMS["k2"]
    km = (km1 + k2) / k1
    et = E0
    S = np.zeros_like(TIME_POINTS)
    P = np.zeros_like(TIME_POINTS)
    E = np.zeros_like(TIME_POINTS)
    ES = np.zeros_like(TIME_POINTS)
    S[0] = S0

    for i in range(len(TIME_POINTS) - 1):
        dt = TIME_POINTS[i + 1] - TIME_POINTS[i]
        ES[i] = et * S[i] / (km + S[i])
        E[i] = et - ES[i]
        dsdt = -k2 * et * S[i] / (km + S[i])
        S[i + 1] = max(0.0, S[i] + dsdt * dt)
        P[i + 1] = P[i] - dsdt * dt

    ES[-1] = et * S[-1] / (km + S[-1])
    E[-1] = et - ES[-1]
    return np.column_stack([E, S, ES, P])


def make_model(checkpoint: dict, is_student: bool) -> torch.nn.Module:
    model_type = checkpoint.get("model_type", "ResidualMLP")
    hidden_dim = checkpoint.get("hidden_dim", 128)
    if is_student:
        num_blocks = checkpoint.get("num_blocks", checkpoint.get("training_args", {}).get("student_num_blocks", 1))
    else:
        num_blocks = checkpoint.get("num_layers", 3)
    dropout = checkpoint.get("dropout", 0.0)

    if model_type == "ResidualMLP":
        return ResidualMLP(input_size=5, output_size=4, hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout)
    if model_type == "MLP":
        return MLP(input_size=5, output_size=4, hidden_sizes=[hidden_dim] * num_blocks, dropout=dropout)
    raise ValueError(f"Unsupported model type: {model_type}")


def load_checkpoint_model(path: Path, is_student: bool) -> tuple[torch.nn.Module, object, object, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(checkpoint, is_student=is_student)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["X_scaler"], checkpoint.get("y_scaler"), checkpoint


def model_predict(model: torch.nn.Module, x_scaler, y_scaler, E0: float, S0: float) -> np.ndarray:
    X = np.zeros((len(TIME_POINTS), 5), dtype=np.float32)
    X[:, 0] = TIME_POINTS
    X[:, 1] = E0
    X[:, 2] = S0
    X[:, 0] = np.log10(X[:, 0] + 1e-12)
    Xn = x_scaler.transform(X)
    with torch.no_grad():
        pred = model(torch.tensor(Xn, dtype=torch.float32)).cpu().numpy()
    if y_scaler is not None:
        pred = y_scaler.inverse_transform(pred)
    return np.maximum(pred, 0.0)


def generate_test_conditions(n: int = 50) -> np.ndarray:
    rng = np.random.default_rng(42)
    E0 = 10 ** rng.uniform(np.log10(5e2), np.log10(1e3), n)
    S0 = 10 ** rng.uniform(np.log10(5e2), np.log10(1e3), n)
    return np.column_stack([E0, S0, np.zeros(n), np.zeros(n)])


def representative_index(conditions: np.ndarray) -> int:
    ratios = np.log10(conditions[:, 1] / conditions[:, 0])
    return int(np.argsort(np.abs(ratios - np.median(ratios)))[0])


def compute_all_trajectories(teacher_bundle, student_bundle, conditions: np.ndarray) -> dict[str, np.ndarray]:
    teacher_model, teacher_x, teacher_y, _ = teacher_bundle
    student_model, student_x, student_y, _ = student_bundle
    out = {"analytical": [], "qssa": [], "teacher": [], "student": []}
    for E0, S0, *_ in conditions:
        out["analytical"].append(analytical_solution(E0, S0))
        out["qssa"].append(qssa_solution(E0, S0))
        out["teacher"].append(model_predict(teacher_model, teacher_x, teacher_y, E0, S0))
        out["student"].append(model_predict(student_model, student_x, student_y, E0, S0))
    return {k: np.asarray(v) for k, v in out.items()}


def stiffness_metrics(trajectory: np.ndarray, method: str = "analytical",
                      time_points: np.ndarray | None = None) -> np.ndarray:
    """Compute a physically meaningful stiffness ratio for the MM reaction.

    The full 4D MM system has two conserved quantities:
        E + ES = E0   (total enzyme)
        S + ES + P = S0   (total substrate/product)
    These conservation laws give two zero eigenvalues in the 4D Jacobian,
    which makes the standard max/min eigenvalue ratio ill-defined or inflated
    by arbitrary numerical cutoffs.

    To avoid this, we eliminate the conserved modes and work with the
    intrinsic 2D dynamics in (S, P):
        ES = S0 - S - P
        E  = E0 - ES = E0 - S0 + S + P
        dS/dt = -k1 * E * S + km1 * ES
        dP/dt = k2 * ES

    The stiffness ratio is then max|Re(lambda)| / min|Re(lambda)| of the
    2x2 reduced Jacobian, which reflects the true timescale separation of
    the dynamical modes without artificial floors.

    Early time points (t < 1e-6) are excluded to avoid numerical artefacts
    from model predictions in the initial transient.
    """
    k1, km1, k2 = PARAMS["k1"], PARAMS["km1"], PARAMS["k2"]

    if method == "qssa":
        return np.ones(len(trajectory))

    # Exclude early time points where model predictions may be numerically unstable
    if time_points is not None:
        early_mask = time_points >= 1e-6
        trajectory = trajectory[early_mask]
        time_points = time_points[early_mask]

    E0 = trajectory[0, 0] + trajectory[0, 2]  # conserved total enzyme
    S0 = trajectory[0, 1] + trajectory[0, 2] + trajectory[0, 3]  # conserved S+ES+P

    ratios = []
    for _E, S, _ES, P in trajectory:
        # 2D reduced Jacobian of (dS/dt, dP/dt) with respect to (S, P)
        E = E0 - S0 + S + P
        J2 = np.array(
            [
                [-k1 * (S + E) - km1, -k1 * S - km1],
                [-k2, -k2],
            ]
        )
        eig = np.linalg.eigvals(J2)
        # For a stable system stiffness is governed by the magnitude of the
        # real parts of the eigenvalues (decay rates).
        decay_rates = -np.real(eig)
        decay_rates = decay_rates[decay_rates > 0]
        if len(decay_rates) == 0:
            ratios.append(1.0)
        else:
            ratios.append(np.max(decay_rates) / np.min(decay_rates))
    return np.asarray(ratios)


def plot_hmm_panel(fig: plt.Figure, spec) -> None:
    data = np.load(GAMMA_DATA)
    log_time = data[:, 0]
    time = 10**log_time
    conc = data[:, 5:9]
    gammas = data[:, 9:11]
    raw_phase = np.argmax(gammas, axis=1)

    phase_median = {p: np.median(log_time[raw_phase == p]) for p in (0, 1)}
    fast_phase = min(phase_median, key=phase_median.get)
    slow_phase = 1 - fast_phase

    rng = np.random.default_rng(7)
    plot_indices = []
    for phase in (fast_phase, slow_phase):
        idx = np.flatnonzero(raw_phase == phase)
        take = min(8000, len(idx))
        plot_indices.append(rng.choice(idx, take, replace=False))
    plot_idx = np.concatenate(plot_indices)
    phase_names = {fast_phase: "Fast", slow_phase: "Slow"}
    phase_colors = {fast_phase: COLORS["fast"], slow_phase: COLORS["slow"]}

    sub = spec.subgridspec(2, 2, hspace=0.14, wspace=0.12)
    axes = []
    for i, sp in enumerate(SPECIES):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        axes.append(ax)
        for phase in (fast_phase, slow_phase):
            mask = raw_phase[plot_idx] == phase
            ax.scatter(
                time[plot_idx][mask],
                conc[plot_idx, i][mask],
                s=5.2,
                c=phase_colors[phase],
                alpha=0.55,
                edgecolors="none",
                rasterized=True,
                label=phase_names[phase] if i == 0 else None,
            )
        ax.set_xscale("log")
        ax.set_xlim(1e-8, 1e2)
        set_sparse_time_ticks(ax)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        style_axis(ax)
        # Species label inside the plot
        # E and P at bottom-right, S and ES at top-right
        if sp in ("E", "P"):
            ax.text(
                0.96,
                0.08,
                rf"${sp}$",
                transform=ax.transAxes,
                ha="right",
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
                rf"${sp}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color=COLORS["text"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.3),
            )
        if i // 2 == 1:
            ax.set_xlabel(r"$t$ (s)", labelpad=1.5)
        else:
            ax.tick_params(labelbottom=False)
        if i % 2 == 0:
            ax.set_ylabel("Conc.", labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[3].legend(
        handles,
        labels,
        loc="upper left",
        ncol=1,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
        borderpad=0.0,
    )
    # Panel label 和 title 由 build_figure 统一放置


def plot_trajectory_grid(
    fig: plt.Figure,
    spec,
    trajectories: dict[str, np.ndarray],
    condition_idx: int,
    mode: str,
) -> None:
    sub = spec.subgridspec(2, 2, hspace=0.14, wspace=0.12)
    axes = []
    marker_idx = np.unique(np.linspace(0, len(TIME_POINTS) - 1, 18, dtype=int))

    if mode == "teacher":
        methods = [("analytical", "Ground truth"), ("teacher", "Teacher")]
        styles = {
            "analytical": dict(color=COLORS["truth"], marker="o", ms=4.0, mfc="white", mec=COLORS["truth"], mew=1.0, ls="none"),
            "teacher": dict(color=COLORS["teacher"], lw=2.1, ls="-"),
        }
        title = "Teacher surrogate"
        label = "a"
    else:
        methods = [("analytical", "Ground truth"), ("qssa", "QSSA"), ("teacher", "Teacher"), ("student", "Student")]
        styles = {
            "analytical": dict(color=COLORS["truth"], marker="o", ms=3.7, mfc="white", mec=COLORS["truth"], mew=0.95, ls="none"),
            "qssa": dict(color=COLORS["qssa"], lw=2.0, ls=":"),
            "teacher": dict(color=COLORS["teacher"], lw=2.0, ls="--"),
            "student": dict(color=COLORS["student"], lw=2.3, ls="-"),
        }
        title = "PAKD student vs QSSA"
        label = "d"

    for i, sp in enumerate(SPECIES):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        axes.append(ax)
        for key, leg_label in methods:
            y = trajectories[key][condition_idx, :, i]
            if key == "analytical":
                ax.semilogx(TIME_POINTS[marker_idx], y[marker_idx], label=leg_label if i == 0 else None, **styles[key])
            else:
                ax.semilogx(TIME_POINTS, y, label=leg_label if i == 0 else None, **styles[key])
        ax.set_xlim(1e-8, 1e2)
        set_sparse_time_ticks(ax)
        y_all = np.concatenate([trajectories[k][condition_idx, :, i] for k, _ in methods])
        pad = max(1.0, 0.08 * (np.max(y_all) - np.min(y_all)))
        ax.set_ylim(np.min(y_all) - pad, np.max(y_all) + pad)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        style_axis(ax)
        # Species label inside the plot
        # E and P at bottom-right, S and ES at top-right
        if sp in ("E", "P"):
            ax.text(
                0.96,
                0.08,
                rf"${sp}$",
                transform=ax.transAxes,
                ha="right",
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
                rf"${sp}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color=COLORS["text"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.3),
            )
        if i // 2 == 1:
            ax.set_xlabel(r"$t$ (s)", labelpad=1.5)
        else:
            ax.tick_params(labelbottom=False)
        if i % 2 == 0:
            ax.set_ylabel("Conc.", labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    if mode == "student":
        # d 图：图例放到 P 子图(axes[3])左上角，垂直排列
        axes[3].legend(
            handles,
            labels,
            loc="upper left",
            ncol=1,
            frameon=False,
            handlelength=0.95,
            borderpad=0.0,
            labelspacing=0.12,
            columnspacing=0.38,
            handletextpad=0.25,
            fontsize=5.7,
        )
    else:
        # a 图：图例放到 P 子图(axes[3])左上角，垂直排列
        axes[3].legend(
            handles,
            labels,
            loc="upper left",
            ncol=1,
            frameon=False,
            handlelength=1.4,
            borderpad=0.0,
            labelspacing=0.25,
            columnspacing=0.7,
        )
    # Panel label 和 title 由 build_figure 统一放置


def plot_pakd_loss(ax: plt.Axes, student_checkpoint: dict) -> None:
    losses = student_checkpoint["training_losses"]
    epochs = np.arange(1, len(losses["total"]) + 1)
    curves = [
        ("total", "Total", COLORS["teacher"]),
        ("output", "Output", "#F28E2B"),
        ("hidden", "Hidden", COLORS["hidden"]),
    ]
    for key, label, color in curves:
        ax.semilogy(epochs, losses[key], color=color, lw=2.2, label=label)
    ax.set_xlabel("Epoch", labelpad=1.5)
    ax.set_ylabel("Loss", labelpad=1.5)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)
    style_axis(ax)
    # Panel label 和 title 由 build_figure 统一放置


def plot_stiffness(ax: plt.Axes, trajectories: dict[str, np.ndarray]) -> None:
    methods = ["analytical", "teacher", "student"]
    labels = ["Truth", "Teacher", "Student"]
    colors = [COLORS["truth"], COLORS["teacher"], COLORS["student"]]
    stats = {}
    for method in methods:
        ratios = np.asarray([stiffness_metrics(traj, method=method, time_points=TIME_POINTS)
                             for traj in trajectories[method]])
        # Clip extreme stiffness ratios to the 99th percentile of the truth
        # to suppress numerical artefacts in model predictions while
        # preserving the genuine dynamic range.
        if method == "analytical":
            cap = np.nanpercentile(ratios, 99)
        ratios = np.clip(ratios, 0, cap)
        stats[method] = {
            "mean": np.nanmean(ratios, axis=1),
            "peak": np.nanmax(ratios, axis=1),
        }

    x = np.arange(2)
    width = 0.22
    for j, method in enumerate(methods):
        # Use median and inter-quartile range (IQR) because stiffness ratios
        # across conditions can be heavy-tailed; the median better represents
        # the typical case than the mean.
        def _mediqr(a):
            med = np.nanmedian(a)
            p25, p75 = np.nanpercentile(a, [25, 75])
            return med, [med - p25, p75 - med]

        vals, errs = zip(*[_mediqr(stats[method][k]) for k in ("mean", "peak")])
        errs = np.asarray(errs).T
        offset = (j - 1) * width
        edge = "black" if method == "student" else "none"
        lw = 1.2 if method == "student" else 0.0
        ax.bar(x + offset, vals, width, yerr=errs, color=colors[j], alpha=0.90, edgecolor=edge, linewidth=lw, capsize=2.0, label=labels[j])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Mean", "Peak"], fontweight="bold")
    ax.set_ylabel("Stiffness ratio", labelpad=1.5)
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=1,
        handlelength=1.0,
        labelspacing=0.2,
        columnspacing=0.6,
        borderpad=0.0,
    )
    style_axis(ax, grid=True)
    # Panel label 和 title 由 build_figure 统一放置


def plot_accuracy_vs_difficulty(ax: plt.Axes, trajectories: dict[str, np.ndarray], conditions: np.ndarray) -> None:
    """Plot RMSE vs log10(S0/E0) to show QSSA regime-dependent failure and PAKD robustness."""
    truth = trajectories["analytical"]
    log_ratio = np.log10(conditions[:, 1] / conditions[:, 0])

    # Compute per-condition RMSE
    rmse = {}
    for method in ["qssa", "teacher", "student"]:
        err = trajectories[method] - truth
        rmse[method] = np.sqrt(np.mean(err**2, axis=(1, 2)))

    # QSSA failure zone: |log10(S0/E0)| < 0.5  (S0/E0 between ~0.3 and ~3)
    failure_lo, failure_hi = -0.5, 0.5
    ax.axvspan(failure_lo, failure_hi, color="#FEF3C7", alpha=0.55, zorder=0)
    ax.text(
        (failure_lo + failure_hi) / 2,
        0.96,
        "QSSA failure zone",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.2,
        color="#92400E",
        fontweight="bold",
    )

    # Plot styles
    styles = {
        "qssa": dict(color=COLORS["qssa"], lw=2.0, ls=":", marker="o", ms=3.0, mfc="white", mec=COLORS["qssa"], mew=0.7, alpha=0.65),
        "teacher": dict(color=COLORS["teacher"], lw=2.0, ls="--", marker="s", ms=3.0, mfc="white", mec=COLORS["teacher"], mew=0.7, alpha=0.65),
        "student": dict(color=COLORS["student"], lw=2.2, ls="-", marker="D", ms=3.0, mfc=COLORS["student"], mec=COLORS["student"], mew=0.7, alpha=0.65),
    }
    labels = {"qssa": "QSSA", "teacher": "Teacher", "student": "PAKD student"}

    # Sort by difficulty and smooth with moving average
    order = np.argsort(log_ratio)
    log_ratio_sorted = log_ratio[order]
    window = 7
    for method in ["qssa", "teacher", "student"]:
        y_raw = rmse[method][order]
        if len(y_raw) >= window:
            pad = window // 2
            padded = np.pad(y_raw, (pad, pad), mode="edge")
            kernel = np.ones(window) / window
            y_smooth = np.convolve(padded, kernel, mode="valid")
        else:
            y_smooth = y_raw
        ax.semilogy(log_ratio_sorted, y_smooth, label=labels[method], **styles[method])

    ax.set_xlabel(r"$\log_{10}(S_0/E_0)$", labelpad=1.5)
    ax.set_ylabel("RMSE", labelpad=1.5)
    ax.set_title("Error vs condition difficulty", pad=2)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, labelspacing=0.2, borderpad=0.0)
    style_axis(ax, grid=True)

    # Annotation: improvement in failure zone
    mask_fail = (log_ratio >= failure_lo) & (log_ratio <= failure_hi)
    if mask_fail.any():
        qssa_fail = np.mean(rmse["qssa"][mask_fail])
        student_fail = np.mean(rmse["student"][mask_fail])
        ax.text(
            0.98,
            0.10,
            f"In failure zone:\n"
            f"QSSA = {qssa_fail:.0f}\n"
            f"Student = {student_fail:.0f}\n"
            f"({qssa_fail / student_fail:.1f}× better)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color=COLORS["student"],
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
        )

    add_panel_label(ax, "e", x=-0.16, y=1.18)


def build_figure() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints...")
    teacher_bundle = load_checkpoint_model(TEACHER_MODEL, is_student=False)
    student_bundle = load_checkpoint_model(STUDENT_MODEL, is_student=True)
    student_checkpoint = student_bundle[3]

    print("Generating trajectories...")
    conditions = generate_test_conditions(50)
    condition_idx = representative_index(conditions)
    trajectories = compute_all_trajectories(teacher_bundle, student_bundle, conditions)

    fig = plt.figure(figsize=(7.2, 9.4))
    outer = fig.add_gridspec(3, 2, height_ratios=[1.24, 1.0, 1.0], hspace=0.28, wspace=0.18)

    plot_trajectory_grid(fig, outer[0, :], trajectories, condition_idx, mode="teacher")
    plot_hmm_panel(fig, outer[1, 0])
    ax_c = fig.add_subplot(outer[1, 1])
    plot_pakd_loss(ax_c, student_checkpoint)
    plot_trajectory_grid(fig, outer[2, 0], trajectories, condition_idx, mode="student")
    ax_e = fig.add_subplot(outer[2, 1])
    plot_stiffness(ax_e, trajectories)

    # 统一放置 panel label 和标题（全局对齐，紧贴子图顶部）
    panel_info = [
        (outer[0, :], "a", "Teacher surrogate"),
        (outer[1, 0], "b", "HMM phase discovery"),
        (outer[1, 1], "c", "PAKD training"),
        (outer[2, 0], "d", "PAKD student vs QSSA"),
        (outer[2, 1], "e", "Stiffness reduction"),
    ]
    for cell, label, title in panel_info:
        bbox = cell.get_position(fig)
        y = bbox.y1 + 0.005
        x_label = bbox.x0 - 0.022
        x_title = bbox.x0
        fig.text(x_label, y, label, fontsize=13, fontweight="bold", ha="left", va="bottom")
        fig.text(x_title, y, title, fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"mmreaction_nature_main.{ext}", facecolor="white")
    plt.close(fig)
    print(f"Saved main figure to {OUT_DIR}")


if __name__ == "__main__":
    build_figure()
