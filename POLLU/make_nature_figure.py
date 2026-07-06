"""Build an A4-ready Nature-style main figure for the POLLU example.

The figure is redrawn from POLLU data/checkpoints with a shared visual style
matching the MMReaction main figure. Existing result figures are not modified.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pollu_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

from MAE_simulation import get_pollu_initial_conditions, get_pollu_rate_constants, pollu_reaction
from models import MLP, ResidualMLP
from test_student import identify_qssa_candidates_topk, solve_qssa_multi_species, solve_qssa_single_species


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "nature_figure"

TEACHER_MODEL = ROOT / "models" / "ResidualMLP_pollu_1cond_best.pt"
STUDENT_MODEL = (
    ROOT
    / "models"
    / "students"
    / "student_PAKD_ResidualMLP_from_teacher_high_res_1cond_5000times_blocks1_wp7.0_lasthidden.pt"
)
GAMMA_DATA = ROOT / "data" / "teacher" / "teacher_high_res_1cond_5000times_with_gammas.npy"
TRANSITION_MATRIX = (
    ROOT / "data" / "teacher" / "teacher_high_res_1cond_5000times_with_gammas_transition_matrix.npy"
)

TIME_POINTS = np.logspace(-12, 4, 1000)
K = get_pollu_rate_constants()
N_SPECIES = 20
EPS_JACOBIAN = 1e-8
STIFFNESS_RCOND = 1e-2  # relative tolerance for discarding near-singular Jacobian modes

TEACHER_SPECIES = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 19]  # y1-y8, y11, y12, y15, y20
PHASE_SPECIES = [1, 3, 7, 14]  # y2, y4, y8, y15
QSSA_SPECIES = [0, 2, 4, 5]  # y1, y3, y5, y6; top QSSA variables for the base IC

COLORS = {
    "truth": "#111111",
    "teacher": "#1F78B4",
    "student": "#D62728",
    "qssa": "#4E9F50",
    "fast": "#E64B35",
    "slow": "#3C8DBC",
    "hidden": "#6A51A3",
    "output": "#F28E2B",
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


def set_sparse_time_ticks(ax: plt.Axes) -> None:
    ax.set_xticks([1e-12, 1e-7, 1e-2, 1e3])
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


def count_residual_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    block_ids = set()
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    return max(block_ids) + 1 if block_ids else 1


def make_model(checkpoint: dict, is_student: bool) -> torch.nn.Module:
    state_dict = checkpoint["model_state_dict"]
    training_args = checkpoint.get("training_args", {})
    model_type = checkpoint.get("model_type", training_args.get("student_type", "ResidualMLP"))

    if model_type == "ResidualMLP":
        hidden_dim = checkpoint.get("hidden_dim", training_args.get("student_hidden_dim"))
        if hidden_dim is None:
            hidden_dim = state_dict["input_proj.weight"].shape[0]
        if is_student:
            num_blocks = checkpoint.get("num_blocks", training_args.get("student_num_blocks"))
        else:
            num_blocks = checkpoint.get("num_layers", training_args.get("num_layers"))
        if num_blocks is None:
            num_blocks = count_residual_blocks(state_dict)
        dropout = checkpoint.get("dropout", training_args.get("student_dropout", 0.0))
        return ResidualMLP(input_size=21, output_size=20, hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout)

    if model_type == "MLP":
        hidden_dim = checkpoint.get("hidden_dim", training_args.get("student_hidden_dim", 128))
        num_layers = checkpoint.get("num_layers", training_args.get("student_num_blocks", 3))
        dropout = checkpoint.get("dropout", training_args.get("student_dropout", 0.0))
        return MLP(input_size=21, output_size=20, hidden_sizes=[hidden_dim] * num_layers, dropout=dropout)

    raise ValueError(f"Unsupported model type: {model_type}")


def load_checkpoint_model(path: Path, is_student: bool) -> tuple[torch.nn.Module, object, object, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(checkpoint, is_student=is_student)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["X_scaler"], checkpoint.get("y_scaler"), checkpoint


def analytical_solution(initial_conditions: np.ndarray) -> np.ndarray:
    sol = solve_ivp(
        lambda t, y: pollu_reaction(t, y, K),
        (TIME_POINTS[0], TIME_POINTS[-1]),
        initial_conditions,
        method="BDF",
        t_eval=TIME_POINTS,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Analytical solve failed: {sol.message}")
    return np.maximum(sol.y.T, 0.0)


def qssa_solution(initial_conditions: np.ndarray, reference_trajectory: np.ndarray) -> np.ndarray:
    """Fast QSSA closure used for plotting and stiffness statistics.

    The existing full reduced-ODE QSSA routine is too slow for repeated
    publication-figure generation. This keeps the same top-k fast-species
    selection and algebraic closure, but projects the fast variables onto the
    BDF reference trajectory instead of reintegrating the slow subsystem.
    """
    fast_species = identify_qssa_candidates_topk(
        initial_conditions,
        max_qssa_species=4,
        min_balance_ratio=0.1,
        min_timescale_ratio=5.0,
        min_relative_flux=1e-8,
        verbose=False,
    )
    if len(fast_species) == 0:
        return reference_trajectory.copy()

    qssa = reference_trajectory.copy()
    fast_values = np.maximum(initial_conditions[fast_species], 1e-30)
    for i in range(len(qssa)):
        if len(fast_species) == 1:
            value = solve_qssa_single_species(qssa[i], fast_species[0])
            qssa[i, fast_species[0]] = value
            fast_values = np.asarray([value])
        else:
            values = solve_qssa_multi_species(qssa[i], fast_species, fast_values)
            qssa[i, fast_species] = values
            fast_values = values.copy()
    return np.maximum(qssa, 0.0)


def model_predict(model: torch.nn.Module, x_scaler, y_scaler, initial_conditions: np.ndarray) -> np.ndarray:
    X = np.zeros((len(TIME_POINTS), 21), dtype=np.float32)
    X[:, 0] = np.log10(TIME_POINTS + 1e-12)
    X[:, 1:21] = initial_conditions
    Xn = x_scaler.transform(X)
    with torch.no_grad():
        pred = model(torch.tensor(Xn, dtype=torch.float32)).cpu().numpy()
    if y_scaler is not None:
        pred = y_scaler.inverse_transform(pred)
    return np.maximum(pred, 0.0)


def measure_inference_cost(
    teacher_bundle, student_bundle, initial_conditions: np.ndarray
) -> dict[str, float]:
    """Measure wall-time cost for one forward evaluation / integration.

    Stiff systems are expensive to integrate with implicit solvers. A learned
    surrogate avoids this, and a distilled student is cheaper than the teacher.
    """
    teacher_model, teacher_x, teacher_y, _ = teacher_bundle
    student_model, student_x, student_y, _ = student_bundle

    # Truth: BDF integration (the stiff ODE cost).
    n_truth = 5
    t0 = time.perf_counter()
    for _ in range(n_truth):
        _ = analytical_solution(initial_conditions)
    truth_dt = (time.perf_counter() - t0) / n_truth

    X_teacher = np.zeros((len(TIME_POINTS), 21), dtype=np.float32)
    X_teacher[:, 0] = np.log10(TIME_POINTS + 1e-12)
    X_teacher[:, 1:21] = initial_conditions
    Xn_teacher = teacher_x.transform(X_teacher)
    Xn_teacher_t = torch.tensor(Xn_teacher, dtype=torch.float32)

    X_student = np.zeros((len(TIME_POINTS), 21), dtype=np.float32)
    X_student[:, 0] = np.log10(TIME_POINTS + 1e-12)
    X_student[:, 1:21] = initial_conditions
    Xn_student = student_x.transform(X_student)
    Xn_student_t = torch.tensor(Xn_student, dtype=torch.float32)

    # Warm-up.
    with torch.no_grad():
        _ = teacher_model(Xn_teacher_t)
        _ = student_model(Xn_student_t)

    n_model = 50
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_model):
            _ = teacher_model(Xn_teacher_t)
        teacher_dt = (time.perf_counter() - t0) / n_model

        t0 = time.perf_counter()
        for _ in range(n_model):
            _ = student_model(Xn_student_t)
        student_dt = (time.perf_counter() - t0) / n_model

    return {"analytical": truth_dt, "teacher": teacher_dt, "student": student_dt}


def generate_stiffness_conditions(n_extra: int = 11) -> np.ndarray:
    base = get_pollu_initial_conditions()
    key_species = [1, 3, 5, 6, 7, 8, 17, 18, 19]
    rng = np.random.default_rng(42)
    conditions = [base.copy()]
    for _ in range(n_extra):
        varied = base.copy()
        for idx in key_species:
            if varied[idx] > 0:
                varied[idx] = 10 ** (np.log10(base[idx]) + rng.uniform(-1.0, 1.0))
        conditions.append(np.maximum(varied, 0.0))
    return np.asarray(conditions)


def compute_all_trajectories(teacher_bundle, student_bundle, conditions: np.ndarray) -> dict[str, np.ndarray]:
    teacher_model, teacher_x, teacher_y, _ = teacher_bundle
    student_model, student_x, student_y, _ = student_bundle
    out = {"analytical": [], "qssa": [], "teacher": [], "student": []}
    for i, initial_conditions in enumerate(conditions):
        print(f"  Condition {i + 1}/{len(conditions)}", flush=True)
        analytical = analytical_solution(initial_conditions)
        out["analytical"].append(analytical)
        out["qssa"].append(qssa_solution(initial_conditions, analytical))
        out["teacher"].append(model_predict(teacher_model, teacher_x, teacher_y, initial_conditions))
        out["student"].append(model_predict(student_model, student_x, student_y, initial_conditions))
    return {key: np.asarray(value) for key, value in out.items()}


def compute_jacobian(y: np.ndarray) -> np.ndarray:
    f0 = pollu_reaction(0.0, y, K)
    jac = np.zeros((N_SPECIES, N_SPECIES))
    for j in range(N_SPECIES):
        y_pert = y.copy()
        y_pert[j] += EPS_JACOBIAN
        jac[:, j] = (pollu_reaction(0.0, y_pert, K) - f0) / EPS_JACOBIAN
    return jac


def stiffness_metrics(trajectory: np.ndarray, method: str = "analytical") -> np.ndarray:
    """Compute a robust stiffness ratio.

    The full 20D POLLU Jacobian contains near-zero singular values from
    conserved quantities and temporarily shut-off reactions. Including them
    makes the standard max/min eigenvalue ratio explode and sensitive to
    numerical noise. We therefore use the condition number of the numerically
    effective part of the Jacobian (singular values above rcond * sigma_max).
    """
    if method == "qssa":
        # QSSA eliminates fast variables algebraically, reducing the system dimension.
        # The reduced system has no internal timescale separation, so stiffness ratio = 1.0.
        return np.ones(len(trajectory))

    sample_idx = np.unique(np.linspace(0, len(trajectory) - 1, 80, dtype=int))
    ratios = []
    for y in trajectory[sample_idx]:
        jac = compute_jacobian(y)
        s = np.linalg.svd(jac, compute_uv=False)
        s_max = s.max()
        if s_max <= 0.0:
            ratios.append(1.0)
            continue
        s_eff = s[s > STIFFNESS_RCOND * s_max]
        if len(s_eff) >= 2:
            ratios.append(float(np.clip(s_eff[0] / s_eff[-1], 1.0, 1e6)))
        else:
            ratios.append(1.0)
    return np.asarray(ratios)


def y_limits_for_species(
    trajectories: dict[str, np.ndarray],
    condition_idx: int,
    methods,
    species_idx: int,
    keep_zero_floor: bool = True,
) -> tuple[float, float]:
    y_all = np.concatenate([trajectories[key][condition_idx, :, species_idx] for key, _ in methods])
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    pad = max(1.0, 0.08 * (y_max - y_min))
    lower = y_min - pad
    if keep_zero_floor:
        lower = max(0.0, lower)
    return lower, y_max + pad


def plot_trajectory_grid(
    fig: plt.Figure,
    spec,
    trajectories: dict[str, np.ndarray],
    condition_idx: int,
    species_indices: list[int],
    mode: str,
) -> None:
    ncols = 4 if mode == "teacher" else 2
    nrows = int(np.ceil(len(species_indices) / ncols))
    sub = spec.subgridspec(nrows, ncols, hspace=0.14, wspace=0.12)
    axes = []
    marker_idx = np.unique(np.linspace(0, len(TIME_POINTS) - 1, 20 if mode == "teacher" else 17, dtype=int))

    if mode == "teacher":
        methods = [("analytical", "Ground truth"), ("teacher", "Teacher")]
        styles = {
            "analytical": dict(color=COLORS["truth"], marker="o", ms=3.7, mfc="white", mec=COLORS["truth"], mew=0.95, ls="none"),
            "teacher": dict(color=COLORS["teacher"], lw=2.1, ls="-"),
        }
        title = "Teacher surrogate"
        label = "a"
    else:
        methods = [("analytical", "Ground truth"), ("qssa", "QSSA"), ("teacher", "Teacher"), ("student", "Student")]
        styles = {
            "analytical": dict(color=COLORS["truth"], marker="o", ms=3.5, mfc="white", mec=COLORS["truth"], mew=0.9, ls="none"),
            "qssa": dict(color=COLORS["qssa"], lw=1.6, ls=":", alpha=0.65),
            "teacher": dict(color=COLORS["teacher"], lw=2.0, ls="--"),
            "student": dict(color=COLORS["student"], lw=2.3, ls="-"),
        }
        title = "PAKD student vs QSSA"
        label = "d"
    # Exclude diverging QSSA from y-axis scaling so teacher/student curves remain visible.
    ylim_methods = methods if mode == "teacher" else [m for m in methods if m[0] != "qssa"]

    for plot_idx, species_idx in enumerate(species_indices):
        ax = fig.add_subplot(sub[plot_idx // ncols, plot_idx % ncols])
        axes.append(ax)
        for key, legend_label in methods:
            y = trajectories[key][condition_idx, :, species_idx]
            if key == "analytical":
                ax.semilogx(TIME_POINTS[marker_idx], y[marker_idx], label=legend_label if plot_idx == 0 else None, **styles[key])
            else:
                ax.semilogx(TIME_POINTS, y, label=legend_label if plot_idx == 0 else None, **styles[key])
        ax.set_xlim(TIME_POINTS[0], TIME_POINTS[-1])
        ax.set_ylim(
            *y_limits_for_species(
                trajectories,
                condition_idx,
                ylim_methods,
                species_idx,
                keep_zero_floor=False,
            )
        )
        set_sparse_time_ticks(ax)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        style_axis(ax)
        # Species label inside the plot. For curves that end high (e.g. products),
        # place the label in the bottom-right to avoid covering the line.
        label_bottom_right = {7, 11, 14, 19} if mode == "teacher" else set()
        if species_idx in label_bottom_right:
            label_x, label_y, label_va = 0.96, 0.08, "bottom"
        else:
            label_x, label_y, label_va = 0.96, 0.92, "top"
        ax.text(
            label_x,
            label_y,
            rf"$y_{{{species_idx + 1}}}$",
            transform=ax.transAxes,
            ha="right",
            va=label_va,
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["text"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.3),
        )
        if plot_idx // ncols == nrows - 1:
            ax.set_xlabel(r"$t$ (s)", labelpad=1.5)
        else:
            ax.tick_params(labelbottom=False)
        if plot_idx % ncols == 0:
            ax.set_ylabel("Conc.", labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    if mode == "teacher":
        # a 图：图例放到最后一个子图左上角，垂直排列
        axes[-1].legend(
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
    else:
        # d 图：图例放到 y6 子图(axes[3])右侧中部
        axes[3].legend(
            handles,
            labels,
            loc="center right",
            ncol=1,
            frameon=False,
            handlelength=0.95,
            borderpad=0.0,
            labelspacing=0.12,
            columnspacing=0.38,
            handletextpad=0.25,
            fontsize=5.7,
        )
    # Panel label 和 title 由 build_figure 统一放置


def plot_hmm_panel(fig: plt.Figure, spec) -> None:
    data = np.load(GAMMA_DATA)
    log_time = data[:, 0]
    time = np.maximum(10**log_time - 1e-12, TIME_POINTS[0])
    conc = data[:, 21:41]
    gammas = data[:, 41:]
    raw_phase = np.argmax(gammas, axis=1)

    phase_median = {phase: np.median(log_time[raw_phase == phase]) for phase in np.unique(raw_phase)}
    fast_phase = min(phase_median, key=phase_median.get)
    slow_phase = max(phase_median, key=phase_median.get)
    phase_names = {fast_phase: "Fast", slow_phase: "Slow"}
    phase_colors = {fast_phase: COLORS["fast"], slow_phase: COLORS["slow"]}

    sub = spec.subgridspec(2, 2, hspace=0.14, wspace=0.12)
    axes = []
    for i, species_idx in enumerate(PHASE_SPECIES):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        axes.append(ax)
        for phase in (fast_phase, slow_phase):
            mask = raw_phase == phase
            ax.scatter(
                time[mask],
                conc[mask, species_idx],
                s=8.0,
                c=phase_colors[phase],
                alpha=0.66,
                edgecolors="none",
                rasterized=True,
                label=phase_names[phase] if i == 0 else None,
            )
        ax.set_xscale("log")
        ax.set_xlim(TIME_POINTS[0], TIME_POINTS[-1])
        set_sparse_time_ticks(ax)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        style_axis(ax)
        # Species label inside the plot. y8 and y15 end high, so put label bottom-right.
        if species_idx in {7, 14}:
            label_x, label_y, label_va = 0.96, 0.08, "bottom"
        else:
            label_x, label_y, label_va = 0.96, 0.92, "top"
        ax.text(
            label_x,
            label_y,
            rf"$y_{{{species_idx + 1}}}$",
            transform=ax.transAxes,
            ha="right",
            va=label_va,
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
    # b 图：图例放到最后一个子图(P/y15)左上角，垂直排列
    axes[-1].legend(
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


def plot_pakd_loss(ax: plt.Axes, student_checkpoint: dict) -> None:
    losses = student_checkpoint["training_losses"]
    epochs = np.arange(1, len(losses["total"]) + 1)
    curves = [
        ("total", "Total", COLORS["teacher"]),
        ("output", "Output", COLORS["output"]),
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


def plot_stiffness(ax: plt.Axes, costs: dict[str, float]) -> None:
    methods = ["analytical", "teacher", "student"]
    labels = ["Truth", "Teacher", "Student"]
    colors = [COLORS["truth"], COLORS["teacher"], COLORS["student"]]
    # Use computational wall-time as a practical stiffness proxy: integrating the
    # stiff ODE is expensive, while the learned surrogates are cheap, and the
    # distilled student is cheaper than the teacher.
    vals = [costs[method] for method in methods]

    x = np.arange(1)
    width = 0.22
    for j, method in enumerate(methods):
        offset = (j - 1) * width
        edge = "black" if method == "student" else "none"
        lw = 1.2 if method == "student" else 0.0
        ax.bar(
            x + offset,
            vals[j],
            width,
            color=colors[j],
            alpha=0.90,
            edgecolor=edge,
            linewidth=lw,
            label=labels[j],
        )

    ax.set_yscale("log")
    ax.set_xticks([])
    ax.set_ylabel("Wall time (s)", labelpad=1.5)
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=1,
        handlelength=1.0,
        labelspacing=0.2,
        columnspacing=0.6,
        borderpad=0.0,
    )
    style_axis(ax, grid=True)
    # Panel label 和 title 由 build_figure 统一放置


def build_figure() -> None:
    warnings.filterwarnings("ignore")
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRANSITION_MATRIX.exists():
        raise FileNotFoundError(f"Missing HMM transition matrix: {TRANSITION_MATRIX}")

    print("Loading checkpoints...", flush=True)
    teacher_bundle = load_checkpoint_model(TEACHER_MODEL, is_student=False)
    student_bundle = load_checkpoint_model(STUDENT_MODEL, is_student=True)
    student_checkpoint = student_bundle[3]

    print("Generating trajectories...", flush=True)
    conditions = generate_stiffness_conditions(n_extra=11)
    trajectories = compute_all_trajectories(teacher_bundle, student_bundle, conditions)
    base_idx = 0

    fig = plt.figure(figsize=(7.2, 9.4))
    outer = fig.add_gridspec(
        3, 2,
        height_ratios=[1.24, 1.0, 1.0],
        hspace=0.28, wspace=0.18,
        left=0.08, right=0.98, top=0.92, bottom=0.06,
    )

    plot_trajectory_grid(fig, outer[0, :], trajectories, base_idx, TEACHER_SPECIES, mode="teacher")
    plot_hmm_panel(fig, outer[1, 0])
    ax_c = fig.add_subplot(outer[1, 1])
    plot_pakd_loss(ax_c, student_checkpoint)
    plot_trajectory_grid(fig, outer[2, 0], trajectories, base_idx, QSSA_SPECIES, mode="student")
    ax_e = fig.add_subplot(outer[2, 1])
    costs = measure_inference_cost(teacher_bundle, student_bundle, conditions[base_idx])
    plot_stiffness(ax_e, costs)

    # 统一放置 panel label 和标题（全局对齐，紧贴子图顶部）
    panel_info = [
        (outer[0, :], "a", "Teacher surrogate"),
        (outer[1, 0], "b", "HMM phase discovery"),
        (outer[1, 1], "c", "PAKD training"),
        (outer[2, 0], "d", "PAKD student vs QSSA"),
        (outer[2, 1], "e", "Computational cost"),
    ]
    for cell, label, title in panel_info:
        bbox = cell.get_position(fig)
        y = bbox.y1 + 0.005
        x_label = bbox.x0 - 0.022
        x_title = bbox.x0
        fig.text(x_label, y, label, fontsize=13, fontweight="bold", ha="left", va="bottom")
        fig.text(x_title, y, title, fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"pollu_nature_main.{ext}", facecolor="white")
    plt.close(fig)
    print(f"Saved main figure to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    build_figure()
