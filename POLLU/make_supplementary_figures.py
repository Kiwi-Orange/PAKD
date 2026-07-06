#!/usr/bin/env python3
"""Build a Nature-style supplementary diagnostics figure for POLLU.

The figure extends the POLLU main figure with phase, distillation, error, and
stiffness diagnostics. It redraws all panels from data/checkpoints and writes
new files under results/nature_figure/supplementary/.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path("/private/tmp/pollu_supp_mpl")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import make_nature_figure as main


OUT_DIR = ROOT / "results" / "nature_figure" / "supplementary"
PANEL_DIR = OUT_DIR / "panels"

FIGSIZE = (7.2, 9.4)
EPS = 1e-10

METHODS = ["analytical", "qssa", "teacher", "student"]
METHOD_LABELS = {
    "analytical": "Truth",
    "qssa": "QSSA",
    "teacher": "Teacher",
    "student": "Student",
}
METHOD_COLORS = {
    "analytical": main.COLORS["truth"],
    "qssa": main.COLORS["qssa"],
    "teacher": main.COLORS["teacher"],
    "student": main.COLORS["student"],
}
SPECIES_LABELS = [rf"$y_{{{i}}}$" for i in range(1, main.N_SPECIES + 1)]


def configure_style() -> None:
    main.configure_style()
    mpl.rcParams.update(
        {
            "font.size": 7.7,
            "axes.labelsize": 7.7,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.6,
            "figure.constrained_layout.use": False,
        }
    )



def style_colorbar(cbar, label_size: float = 7.2) -> None:
    cbar.outline.set_linewidth(0.75)
    cbar.ax.tick_params(width=0.8, length=2.5, pad=1.4, labelsize=label_size)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")


def log_time_ticks(ax: plt.Axes, values: list[int] | None = None) -> None:
    if values is None:
        values = [-12, -7, -2, 3]
    ax.set_xticks(values)
    ax.set_xticklabels([rf"$10^{{{v}}}$" for v in values])


def set_log_time_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(main.TIME_POINTS[0], main.TIME_POINTS[-1])
    main.set_sparse_time_ticks(ax)


def make_error_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "pollu_error",
        ["#F7FBFF", "#D8E8F4", "#8FC2DD", "#3F8FC1", "#084B83"],
        N=256,
    )


def load_gamma_bundle() -> dict[str, np.ndarray | int | float]:
    data = np.load(main.GAMMA_DATA)
    transition = np.load(main.TRANSITION_MATRIX)
    log_time = data[:, 0]
    gammas = data[:, 41:]
    raw_phase = np.argmax(gammas, axis=1)
    phase_median = {
        phase: np.median(log_time[raw_phase == phase])
        for phase in np.unique(raw_phase)
    }
    fast_phase = min(phase_median, key=phase_median.get)
    slow_phase = max(phase_median, key=phase_median.get)
    order = [fast_phase, slow_phase]
    transition = transition[np.ix_(order, order)]
    time = np.maximum(10**log_time - 1e-12, main.TIME_POINTS[0])
    sorted_idx = np.argsort(time)
    slow_cross = np.flatnonzero(gammas[sorted_idx, slow_phase] >= 0.5)
    transition_time = time[sorted_idx[slow_cross[0]]] if len(slow_cross) else np.nan

    return {
        "data": data,
        "time": time,
        "log_time": log_time,
        "gammas": gammas,
        "raw_phase": raw_phase,
        "fast_phase": fast_phase,
        "slow_phase": slow_phase,
        "transition": transition,
        "transition_time": transition_time,
    }


def moving_average(values: np.ndarray, window: int = 31) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def plot_phase_diagnostics(ax: plt.Axes, gamma_bundle: dict) -> None:
    """HMM phase posterior vs time (fast/slow)."""
    order = np.argsort(gamma_bundle["time"])
    time = gamma_bundle["time"][order]
    fast = gamma_bundle["gammas"][order, gamma_bundle["fast_phase"]]
    slow = gamma_bundle["gammas"][order, gamma_bundle["slow_phase"]]
    fast_s = moving_average(fast, window=35)
    slow_s = moving_average(slow, window=35)
    envelope = 0.035

    ax.plot(time, fast_s, color=main.COLORS["fast"], lw=2.35, label="Fast")
    ax.plot(time, slow_s, color=main.COLORS["slow"], lw=2.35, label="Slow")
    ax.fill_between(
        time,
        np.clip(fast_s - envelope, 0, 1),
        np.clip(fast_s + envelope, 0, 1),
        color=main.COLORS["fast"],
        alpha=0.12,
        lw=0,
    )
    ax.fill_between(
        time,
        np.clip(slow_s - envelope, 0, 1),
        np.clip(slow_s + envelope, 0, 1),
        color=main.COLORS["slow"],
        alpha=0.12,
        lw=0,
    )
    if np.isfinite(gamma_bundle["transition_time"]):
        ax.axvline(
            gamma_bundle["transition_time"],
            color=main.COLORS["text"],
            lw=1.15,
            ls="--",
            alpha=0.85,
        )
        ax.text(
            gamma_bundle["transition_time"],
            0.53,
            r"$t_{\mathrm{HMM}}$",
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            color=main.COLORS["text"],
        )
    set_log_time_axis(ax)
    ax.set_ylim(-0.04, 1.04)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel("Posterior")
    ax.legend(
        frameon=False,
        loc="center right",
        handlelength=1.2,
        borderpad=0.0,
        labelspacing=0.15,
    )
    main.style_axis(ax)
    # panel_label 和 title 由 build_figure 统一放置


def compute_error_maps(trajectories: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute mean log10 error across all conditions for teacher and student."""
    truth = trajectories["analytical"]
    maps = {}
    for method in ["teacher", "student"]:
        err = np.abs(trajectories[method] - truth)
        maps[method] = np.nanmean(np.log10(err + EPS), axis=0).T
    return maps


def species_rmse(trajectories: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute species-wise RMSE across all conditions."""
    truth = trajectories["analytical"]
    out = {}
    for method in ["qssa", "teacher", "student"]:
        err = trajectories[method] - truth
        out[method] = np.sqrt(np.nanmean(err**2, axis=(0, 1)))
    return out


def plot_species_rmse_ranking(ax: plt.Axes, rmse: dict[str, np.ndarray]) -> None:
    """Species-wise RMSE ranking (QSSA difficulty vs surrogate fidelity)."""
    order = np.argsort(rmse["qssa"])
    y = np.arange(main.N_SPECIES)
    offsets = {"qssa": -0.18, "teacher": 0.0, "student": 0.18}
    styles = {
        "qssa": ("QSSA", main.COLORS["qssa"], "D"),
        "teacher": ("Teacher", main.COLORS["teacher"], "o"),
        "student": ("Student", main.COLORS["student"], "s"),
    }
    log_rmse = {method: np.log10(np.maximum(values, EPS)) for method, values in rmse.items()}
    all_log = np.concatenate([log_rmse[method] for method in styles])
    finite = all_log[np.isfinite(all_log)]
    x_min = max(-10.0, np.floor(np.nanpercentile(finite, 1)) - 0.5)
    x_max = min(7.0, np.ceil(np.nanpercentile(finite, 95)) + 0.5)
    x_max = max(x_max, 5.0)

    for method, (label, color, marker) in styles.items():
        vals = log_rmse[method][order]
        clipped = vals > x_max
        plot_vals = np.minimum(vals, x_max)
        ax.scatter(
            plot_vals[~clipped],
            y[~clipped] + offsets[method],
            s=26 if method == "student" else 23,
            color=color,
            marker=marker,
            edgecolor="black" if method == "student" else "white",
            linewidth=0.55,
            label=label,
            zorder=3 if method == "student" else 2,
        )
        if np.any(clipped):
            ax.scatter(
                plot_vals[clipped],
                y[clipped] + offsets[method],
                s=34 if method == "student" else 31,
                color=color,
                marker=">",
                edgecolor="black" if method == "student" else "white",
                linewidth=0.55,
                zorder=4,
            )
    qssa_species_set = {idx + 1 for idx in main.QSSA_SPECIES}
    labels = []
    for idx in order:
        text = rf"$y_{{{idx + 1}}}$"
        labels.append(text + "*" if idx + 1 in qssa_species_set else text)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xlim(x_min, x_max + 0.35)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.axvline(0.0, color=main.COLORS["text"], lw=0.8, ls="--", alpha=0.45)
    ax.set_ylim(-0.7, main.N_SPECIES - 0.3)
    ax.set_xlabel(r"Species-wise $\log_{10}(\mathrm{RMSE})$")
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=3,
        handlelength=0.9,
        columnspacing=0.7,
        labelspacing=0.1,
        borderpad=0.0,
    )
    main.style_axis(ax)
    # panel_label 和 title 由 build_figure 统一放置


def plot_time_resolved_error(
    ax: plt.Axes,
    trajectories: dict[str, np.ndarray],
) -> None:
    """Mean absolute error vs time for all methods (base condition only)."""
    truth = trajectories["analytical"][0]
    methods = ["qssa", "teacher", "student"]
    labels = ["QSSA", "Teacher", "Student"]
    colors = [main.COLORS["qssa"], main.COLORS["teacher"], main.COLORS["student"]]
    styles = [":", "--", "-"]

    for method, label, color, ls in zip(methods, labels, colors, styles):
        err = np.abs(trajectories[method][0] - truth)
        mean_err = np.nanmean(err, axis=1)
        ax.semilogy(main.TIME_POINTS, mean_err, color=color, lw=2.0, ls=ls, label=label)

    ax.set_xscale("log")
    ax.set_xlim(main.TIME_POINTS[0], main.TIME_POINTS[-1])
    main.set_sparse_time_ticks(ax)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel("Mean absolute error")
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)
    ax.grid(True, alpha=0.3, lw=0.5)
    main.style_axis(ax)
    # panel_label 和 title 由 build_figure 统一放置


def plot_teacher_vs_student_errors(
    fig: plt.Figure,
    spec,
    trajectories: dict[str, np.ndarray],
) -> None:
    """Teacher vs Student error heatmaps side-by-side (both vs Truth)."""
    sub = spec.subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.06)
    ax_t = fig.add_subplot(sub[0, 0])
    ax_s = fig.add_subplot(sub[0, 1], sharey=ax_t)
    cax = fig.add_subplot(sub[0, 2])

    truth = trajectories["analytical"][0]
    teacher_err = np.log10(np.abs(trajectories["teacher"][0] - truth) + EPS)
    student_err = np.log10(np.abs(trajectories["student"][0] - truth) + EPS)

    all_vals = np.concatenate([teacher_err.ravel(), student_err.ravel()])
    finite = all_vals[np.isfinite(all_vals)]
    vmin, vmax = np.nanpercentile(finite, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1

    extent = [
        np.log10(main.TIME_POINTS[0]),
        np.log10(main.TIME_POINTS[-1]),
        0.5,
        main.N_SPECIES + 0.5,
    ]
    cmap = make_error_cmap()

    images = []
    for ax, data, title in [
        (ax_t, teacher_err.T, "Teacher"),
        (ax_s, student_err.T, "Student"),
    ]:
        img = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        images.append(img)
        ax.text(
            0.03,
            0.97,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            color=main.COLORS["text"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.3),
        )
        ax.set_xlabel(r"$t$ (s)")
        ax.set_yticks(np.arange(1, main.N_SPECIES + 1))
        ax.set_ylim(0.5, main.N_SPECIES + 0.5)
        log_time_ticks(ax, values=[-12, -7, -2, 3])
        ax.tick_params(axis="both", width=0.8, length=2.4, pad=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.95)

    ax_t.set_yticklabels(SPECIES_LABELS, fontsize=5.8)
    ax_t.set_ylabel("Species")
    ax_s.tick_params(labelleft=False)
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label(r"$\log_{10}(|e|+\epsilon)$", labelpad=1.8)
    style_colorbar(cbar, label_size=6.4)
    # panel_label 和 title 由 build_figure 统一放置


def trajectory_y_limits(trajectories: dict[str, np.ndarray], species_idx: int, condition_idx: int = 0) -> tuple[float, float]:
    values = np.concatenate(
        [
            trajectories[method][condition_idx, :, species_idx]
            for method in METHODS
        ]
    )
    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    span = max(y_max - y_min, max(abs(y_max), 1.0) * 0.02)
    lower = y_min - 0.08 * span
    upper = y_max + 0.10 * span
    if y_min >= 0.0 and lower >= 0.0:
        lower = -0.035 * max(y_max, 1.0)
    return lower, upper


def plot_trajectory_atlas(trajectories: dict[str, np.ndarray], condition_idx: int = 0) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(
        5,
        4,
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.98,
        hspace=0.14,
        wspace=0.15,
    )
    marker_idx = np.unique(np.linspace(0, len(main.TIME_POINTS) - 1, 15, dtype=int))
    styles = {
        "analytical": dict(color=METHOD_COLORS["analytical"], marker="o", ms=2.5, mfc="white", mec=METHOD_COLORS["analytical"], mew=0.7, ls="none"),
        "qssa": dict(color=METHOD_COLORS["qssa"], lw=1.45, ls=":", alpha=0.92),
        "teacher": dict(color=METHOD_COLORS["teacher"], lw=1.45, ls="--", alpha=0.96),
        "student": dict(color=METHOD_COLORS["student"], lw=1.75, ls="-", alpha=0.98),
    }
    axes = []
    for species_idx in range(main.N_SPECIES):
        row, col = divmod(species_idx, 4)
        ax = fig.add_subplot(grid[row, col])
        axes.append(ax)
        for method in METHODS:
            y = trajectories[method][condition_idx, :, species_idx]
            if method == "analytical":
                ax.semilogx(
                    main.TIME_POINTS[marker_idx],
                    y[marker_idx],
                    label=METHOD_LABELS[method] if species_idx == 0 else None,
                    **styles[method],
                )
            else:
                ax.semilogx(
                    main.TIME_POINTS,
                    y,
                    label=METHOD_LABELS[method] if species_idx == 0 else None,
                    **styles[method],
                )
        ax.set_xlim(main.TIME_POINTS[0], main.TIME_POINTS[-1])
        ax.set_ylim(*trajectory_y_limits(trajectories, species_idx, condition_idx))
        main.set_sparse_time_ticks(ax)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        # Species label inside the plot (top-right corner)
        ax.text(
            0.96,
            0.92,
            rf"$y_{{{species_idx + 1}}}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            color=main.COLORS["text"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.3),
        )
        main.style_axis(ax)
        if row == 4:
            ax.set_xlabel(r"$t$ (s)", labelpad=1.2)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel("Conc.", labelpad=1.4)
        else:
            ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    # 竖向图例放在左下角子图（y17）内部左下角
    axes[16].legend(
        handles,
        labels,
        loc="lower left",
        ncol=1,
        frameon=False,
        handlelength=1.35,
        columnspacing=0.95,
        handletextpad=0.35,
        borderpad=0.0,
        fontsize=7.3,
    )
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"pollu_supp_trajectory_atlas.{ext}", facecolor="white")
    fig.savefig(PANEL_DIR / "trajectory_atlas_preview.png", facecolor="white", dpi=300)
    plt.close(fig)


def build_figure() -> None:
    warnings.filterwarnings("ignore")
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading POLLU checkpoints...", flush=True)
    teacher_bundle = main.load_checkpoint_model(main.TEACHER_MODEL, is_student=False)
    student_bundle = main.load_checkpoint_model(main.STUDENT_MODEL, is_student=True)
    student_checkpoint = student_bundle[3]

    print("Loading HMM posterior data...", flush=True)
    gamma_bundle = load_gamma_bundle()

    print("Generating deterministic trajectories...", flush=True)
    conditions = main.generate_stiffness_conditions(n_extra=11)
    trajectories = main.compute_all_trajectories(teacher_bundle, student_bundle, conditions)

    print("Computing error maps...", flush=True)
    rmse = species_rmse(trajectories)

    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.06, 1.23, 1.18],
        hspace=0.28,
        wspace=0.18,
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.06,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    plot_species_rmse_ranking(ax_a, rmse)
    ax_b = fig.add_subplot(outer[0, 1])
    plot_time_resolved_error(ax_b, trajectories)
    plot_teacher_vs_student_errors(fig, outer[1, :], trajectories)
    ax_d = fig.add_subplot(outer[2, :])
    plot_phase_diagnostics(ax_d, gamma_bundle)

    # 统一放置 panel label 和标题（与正文大图一致）
    panel_info = [
        (outer[0, 0], "a", "QSSA difficulty and surrogate fidelity"),
        (outer[0, 1], "b", "Time-resolved error"),
        (outer[1, :], "c", "Teacher vs Student error"),
        (outer[2, :], "d", "HMM phase diagnostics"),
    ]
    for cell, label, title in panel_info:
        bbox = cell.get_position(fig)
        y = bbox.y1 + 0.005
        x_label = bbox.x0 - 0.022
        x_title = bbox.x0
        fig.text(x_label, y, label, fontsize=13, fontweight="bold", ha="left", va="bottom")
        fig.text(x_title, y, title, fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    fig.savefig(OUT_DIR / "pollu_supp_diagnostics.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "pollu_supp_diagnostics.png", facecolor="white")
    fig.savefig(OUT_DIR / "pollu_supp_diagnostics.svg", facecolor="white")
    fig.savefig(PANEL_DIR / "full_preview.png", facecolor="white", dpi=300)
    plt.close(fig)

    print("Building trajectory atlas...", flush=True)
    plot_trajectory_atlas(trajectories, condition_idx=0)
    print(f"Saved supplementary figures to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    build_figure()
