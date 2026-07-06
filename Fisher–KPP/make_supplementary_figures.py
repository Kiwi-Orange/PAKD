#!/usr/bin/env python3
"""Build Nature-style supplementary Fisher-KPP figures.

The figures extend the main Fisher-KPP figure with 2D trajectory, error,
distillation, and equation-discovery diagnostics. Existing main and 3D
supplementary figures are not modified.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path("/private/tmp/fisher_kpp_supp_mpl")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

import make_nature_figure as main


OUT_DIR = ROOT / "results" / "nature_figure" / "supplementary"
PANEL_DIR = OUT_DIR / "panels"

FIGSIZE = (7.2, 9.4)
EPS = 1e-10

METHODS = ["truth", "teacher", "student"]
METHOD_LABELS = {"truth": "Truth", "teacher": "Teacher", "student": "Student"}
METHOD_COLORS = {
    "truth": main.COLORS["truth"],
    "teacher": main.COLORS["teacher"],
    "student": main.COLORS["student"],
}


def configure_style() -> None:
    main.configure_style()
    mpl.rcParams.update(
        {
            "font.size": 7.7,
            "axes.labelsize": 7.7,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.5,
            "figure.constrained_layout.use": False,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.12) -> None:
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


def style_colorbar(cbar, label_size: float = 6.5) -> None:
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(width=0.8, length=2.5, pad=1.2, labelsize=label_size)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")


def set_log_time_y(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.set_ylim(main.TIME_POINTS[0], main.TIME_POINTS[-1])
    ax.set_yticks([1e-4, 1e-2, 1e0, 1e1])
    ax.get_yaxis().set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())


def set_log_time_x(ax: plt.Axes, xmin: float | None = None, xmax: float | None = None) -> None:
    ax.set_xscale("log")
    ax.set_xlim(main.TIME_POINTS[0] if xmin is None else xmin, main.TIME_POINTS[-1] if xmax is None else xmax)
    ax.set_xticks([1e-4, 1e-2, 1e0, 1e1])
    ax.get_xaxis().set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def make_error_cmap() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "fisher_error_light",
        ["#fffdf7", "#fee8c8", "#fdbb84", "#f46d43", "#a50026"],
    )


def load_gamma_bundle() -> dict[str, np.ndarray | int | float]:
    data = np.load(main.GAMMA_DATA)
    transition = np.load(main.GAMMA_DATA.with_name(main.GAMMA_DATA.stem + "_transition_matrix.npy"))
    time = np.maximum(data[:, 0], main.TIME_POINTS[0])
    gammas = data[:, -2:]
    raw_phase = np.argmax(gammas, axis=1)
    medians = {phase: np.median(time[raw_phase == phase]) for phase in np.unique(raw_phase)}
    fast_phase = min(medians, key=medians.get)
    slow_phase = max(medians, key=medians.get)
    order = [fast_phase, slow_phase]
    transition = transition[np.ix_(order, order)]
    sorted_idx = np.argsort(time)
    slow = gammas[sorted_idx, slow_phase]
    fast = gammas[sorted_idx, fast_phase]
    switch_idx = np.flatnonzero(np.diff((slow > fast).astype(int)) != 0)
    t_hmm = float(time[sorted_idx[switch_idx[0] + 1]]) if len(switch_idx) else float(np.median(time))
    return {
        "data": data,
        "time": time,
        "gammas": gammas,
        "raw_phase": raw_phase,
        "fast_phase": fast_phase,
        "slow_phase": slow_phase,
        "transition": transition,
        "t_hmm": t_hmm,
    }


def two_regime_values(times: np.ndarray, t_transition: float, fast: float, slow: float) -> np.ndarray:
    return np.where(times < t_transition, fast, slow)


def plot_solution_heatmaps(fig: plt.Figure, spec, trajectories: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.040], wspace=0.10)
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]
    cax = fig.add_subplot(sub[0, 3])
    panels = [
        ("truth", "BDF ground truth"),
        ("teacher", "Teacher surrogate"),
        ("student", "PAKD student"),
    ]
    mappable = None
    for i, (key, title) in enumerate(panels):
        ax = axes[i]
        mesh = ax.pcolormesh(
            trajectories["x"],
            main.TIME_POINTS,
            trajectories[key],
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        mappable = mesh
        ax.set_title(title, pad=2.0)
        ax.set_xlim(float(trajectories["x"][0]), float(trajectories["x"][-1]))
        ax.set_xticks([0.0, 0.5, 1.0])
        set_log_time_y(ax)
        main.style_axis(ax, grid=False)
        ax.set_xlabel(r"$x$")
        if i == 0:
            ax.set_ylabel(r"$t$")
        else:
            ax.tick_params(labelleft=False)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r"$u(x,t)$", labelpad=1.0)
    style_colorbar(cbar)


def plot_error_heatmaps(fig: plt.Figure, spec, trajectories: dict[str, np.ndarray]) -> None:
    """Student error heatmap only."""
    sub = spec.subgridspec(1, 2, width_ratios=[1.0, 0.040], wspace=0.10)
    ax = fig.add_subplot(sub[0, 0])
    cax = fig.add_subplot(sub[0, 1])
    student_err = np.abs(trajectories["student"] - trajectories["truth"])
    vmax = max(0.005, float(np.nanpercentile(student_err.ravel(), 99.0)))
    norm = mpl.colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)
    cmap = make_error_cmap()
    mesh = ax.pcolormesh(
        trajectories["x"],
        main.TIME_POINTS,
        student_err,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    ax.set_xlim(float(trajectories["x"][0]), float(trajectories["x"][-1]))
    ax.set_xticks([0.0, 0.5, 1.0])
    set_log_time_y(ax)
    main.style_axis(ax, grid=False)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$|e|$", labelpad=1.0)
    cbar.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    style_colorbar(cbar)


def key_times(t_hmm: float) -> list[float]:
    candidates = [1e-4, 4e-4, 2e-3, 5e-2, t_hmm, 0.7, 2.5, 10.0]
    clipped = [float(np.clip(t, main.TIME_POINTS[0], main.TIME_POINTS[-1])) for t in candidates]
    selected: list[float] = []
    for value in sorted(clipped):
        if not selected or abs(np.log10(value) - np.log10(selected[-1])) > 0.13:
            selected.append(value)
    fallbacks = [1e-4, 4e-4, 2e-3, 1e-2, 5e-2, 0.25, 0.7, 2.5, 10.0]
    for value in fallbacks:
        if len(selected) >= 8:
            break
        if all(abs(np.log10(value) - np.log10(old)) > 0.13 for old in selected):
            selected.append(value)
    return sorted(selected[:8])


def method_styles() -> dict[str, dict]:
    return {
        "truth": dict(color=METHOD_COLORS["truth"], marker="o", ms=2.7, mfc="white", mec=METHOD_COLORS["truth"], mew=0.75, ls="none"),
        "teacher": dict(color=METHOD_COLORS["teacher"], lw=1.75, ls="--"),
        "student": dict(color=METHOD_COLORS["student"], lw=2.05, ls="-"),
    }


def plot_spatial_profile_atlas(fig: plt.Figure, spec, trajectories: dict[str, np.ndarray], t_hmm: float) -> None:
    """Spatial profile snapshots at 4 key times."""
    sub = spec.subgridspec(1, 4, hspace=0.22, wspace=0.15)
    axes = []
    styles = method_styles()
    marker_idx = np.unique(np.linspace(0, len(trajectories["x"]) - 1, 24, dtype=int))
    targets = [1e-4, t_hmm, 1.0, 10.0]
    for i, target in enumerate(targets):
        ax = fig.add_subplot(sub[0, i])
        axes.append(ax)
        idx = int(np.argmin(np.abs(main.TIME_POINTS - target)))
        for method in METHODS:
            y = trajectories[method][idx]
            if method == "truth":
                ax.plot(
                    trajectories["x"][marker_idx],
                    y[marker_idx],
                    label=METHOD_LABELS[method] if i == 0 else None,
                    **styles[method],
                )
            else:
                ax.plot(trajectories["x"], y, label=METHOD_LABELS[method] if i == 0 else None, **styles[method])
        ax.set_xlim(float(trajectories["x"][0]), float(trajectories["x"][-1]))
        ax.set_ylim(-0.06, 1.06)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.text(
            0.5,
            0.97,
            rf"$t={main.TIME_POINTS[idx]:.2g}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.4),
        )
        main.style_axis(ax)
        ax.set_xlabel(r"$x$", labelpad=1.0)
        if i == 0:
            ax.set_ylabel(r"$u$", labelpad=1.2)
        else:
            ax.tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="center right",
        frameon=False,
        handlelength=1.1,
        columnspacing=0.55,
        labelspacing=0.1,
        borderpad=0.0,
        fontsize=6.2,
    )


def plot_temporal_trace_atlas(fig: plt.Figure, spec, trajectories: dict[str, np.ndarray]) -> None:
    """Temporal traces at 3 key spatial positions."""
    sub = spec.subgridspec(1, 3, hspace=0.22, wspace=0.18)
    axes = []
    styles = method_styles()
    x_targets = [0.10, 0.50, 0.90]
    marker_idx = np.unique(np.linspace(0, len(main.TIME_POINTS) - 1, 17, dtype=int))
    for i, target in enumerate(x_targets):
        ax = fig.add_subplot(sub[0, i])
        axes.append(ax)
        idx = int(np.argmin(np.abs(trajectories["x"] - target)))
        for method in METHODS:
            y = trajectories[method][:, idx]
            if method == "truth":
                ax.semilogx(
                    main.TIME_POINTS[marker_idx],
                    y[marker_idx],
                    label=METHOD_LABELS[method] if i == 0 else None,
                    **styles[method],
                )
            else:
                ax.semilogx(main.TIME_POINTS, y, label=METHOD_LABELS[method] if i == 0 else None, **styles[method])
        set_log_time_x(ax)
        ax.set_ylim(-0.06, 1.06)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.text(
            0.5,
            0.97,
            rf"$x={trajectories['x'][idx]:.2f}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.4),
        )
        main.style_axis(ax)
        ax.text(
            0.97,
            0.03,
            r"$t$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="black",
        )
        if i == 0:
            ax.set_ylabel(r"$u$", labelpad=1.2)
        else:
            ax.tick_params(labelleft=False)



def front_position(x_grid: np.ndarray, trajectory: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    positions = []
    for values in trajectory:
        shifted = values - threshold
        crossings = np.flatnonzero(np.diff(np.signbit(shifted)))
        if len(crossings):
            j = crossings[0]
            x0, x1 = x_grid[j], x_grid[j + 1]
            y0, y1 = values[j], values[j + 1]
            frac = (threshold - y0) / (y1 - y0 + EPS)
            positions.append(float(x0 + frac * (x1 - x0)))
        else:
            positions.append(float(x_grid[int(np.argmin(np.abs(shifted)))]))
    return np.asarray(positions)


def plot_front_mass(fig: plt.Figure, spec, trajectories: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(2, 1, hspace=0.24)
    axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[1, 0])]
    line_styles = {"truth": "-", "teacher": "--", "student": "-"}
    for method in METHODS:
        front = front_position(trajectories["x"], trajectories[method])
        mass = np.trapz(trajectories[method], trajectories["x"], axis=1)
        axes[0].semilogx(main.TIME_POINTS, front, color=METHOD_COLORS[method], lw=2.0 if method == "student" else 1.75, ls=line_styles[method], label=METHOD_LABELS[method])
        axes[1].semilogx(main.TIME_POINTS, mass, color=METHOD_COLORS[method], lw=2.0 if method == "student" else 1.75, ls=line_styles[method])
    for i, ax in enumerate(axes):
        set_log_time_x(ax)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        main.style_axis(ax)
        if i == 1:
            ax.set_xlabel(r"$t$", labelpad=1.0)
        else:
            ax.tick_params(labelbottom=False)
    axes[0].set_ylabel(r"$x_{u=0.5}$")
    axes[1].set_ylabel(r"$\int u\,dx$")



def build_trajectory_atlas(trajectories: dict[str, np.ndarray], gamma_bundle: dict) -> None:
    fig = plt.figure(figsize=(7.2, 7.0))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 1.0],
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.92,
        hspace=0.28,
    )
    plot_spatial_profile_atlas(fig, outer[0, 0], trajectories, float(gamma_bundle["t_hmm"]))
    plot_temporal_trace_atlas(fig, outer[1, 0], trajectories)
    plot_front_mass(fig, outer[2, 0], trajectories)

    # 统一放置 panel label 和标题
    panel_info = [
        (outer[0, 0], "a", "Spatial profile atlas"),
        (outer[1, 0], "b", "Temporal traces"),
        (outer[2, 0], "c", "Front/mass dynamics"),
    ]
    for cell, label, title in panel_info:
        bbox = cell.get_position(fig)
        y = bbox.y1 + 0.005
        x_label = bbox.x0 - 0.022
        x_title = bbox.x0
        fig.text(x_label, y, label, fontsize=13, fontweight="bold", ha="left", va="bottom")
        fig.text(x_title, y, title, fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"fisher_kpp_supp_trajectory_atlas.{ext}", facecolor="white")
    fig.savefig(PANEL_DIR / "fisher_kpp_supp_trajectory_atlas_preview.png", facecolor="white", dpi=300)
    plt.close(fig)


def plot_hmm_diagnostics(ax: plt.Axes, gamma_bundle: dict) -> None:
    """HMM phase posterior vs time (fast/slow)."""
    order = np.argsort(gamma_bundle["time"])
    time = gamma_bundle["time"][order]
    fast = gamma_bundle["gammas"][order, gamma_bundle["fast_phase"]]
    slow = gamma_bundle["gammas"][order, gamma_bundle["slow_phase"]]
    ax.fill_between(time, 0.0, 1.0, where=fast >= slow, color=main.COLORS["fast"], alpha=0.11, lw=0)
    ax.fill_between(time, 0.0, 1.0, where=slow > fast, color=main.COLORS["slow"], alpha=0.11, lw=0)
    ax.semilogx(time, fast, color=main.COLORS["fast"], lw=2.25, label="Fast")
    ax.semilogx(time, slow, color=main.COLORS["slow"], lw=2.25, label="Slow")
    ax.axvline(float(gamma_bundle["t_hmm"]), color=main.COLORS["text"], lw=1.05, ls=":", alpha=0.85)
    ax.text(
        float(gamma_bundle["t_hmm"]) * 1.08,
        0.52,
        r"$t_{\mathrm{HMM}}$",
        fontsize=6.5,
        fontweight="bold",
        rotation=90,
        ha="left",
        va="center",
    )
    set_log_time_x(ax)
    ax.set_ylim(-0.04, 1.04)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Posterior")
    ax.legend(frameon=False, loc="center right", handlelength=1.2, labelspacing=0.15, borderpad=0.0)
    main.style_axis(ax)



def plot_teacher_vs_student_errors(fig: plt.Figure, spec, trajectories: dict[str, np.ndarray]) -> None:
    """Teacher vs Student error heatmaps side-by-side (both vs Truth)."""
    sub = spec.subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.06)
    ax_t = fig.add_subplot(sub[0, 0])
    ax_s = fig.add_subplot(sub[0, 1], sharey=ax_t)
    cax = fig.add_subplot(sub[0, 2])

    teacher_err = np.abs(trajectories["teacher"] - trajectories["truth"])
    student_err = np.abs(trajectories["student"] - trajectories["truth"])

    all_vals = np.concatenate([teacher_err.ravel(), student_err.ravel()])
    finite = all_vals[np.isfinite(all_vals)]
    vmin, vmax = 0.0, max(0.005, float(np.nanpercentile(finite, 99.0)))
    if vmax <= vmin:
        vmax = vmin + 1

    cmap = make_error_cmap()
    norm = mpl.colors.PowerNorm(gamma=0.55, vmin=vmin, vmax=vmax)

    images = []
    for ax, data, title in [
        (ax_t, teacher_err, "Teacher"),
        (ax_s, student_err, "Student"),
    ]:
        img = ax.pcolormesh(
            trajectories["x"],
            main.TIME_POINTS,
            data,
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
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
        ax.set_xlim(float(trajectories["x"][0]), float(trajectories["x"][-1]))
        ax.set_xticks([0.0, 0.5, 1.0])
        set_log_time_y(ax)
        ax.tick_params(axis="both", width=0.8, length=2.4, pad=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.95)
        ax.set_xlabel(r"$x$")

    ax_t.set_ylabel(r"$t$")
    ax_s.tick_params(labelleft=False)
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label(r"$|e|$", labelpad=1.8)
    cbar.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    style_colorbar(cbar, label_size=6.4)
def plot_training_losses(ax: plt.Axes, student_checkpoint: dict) -> None:
    losses = student_checkpoint["training_losses"]
    epochs = np.arange(1, len(losses["total"]) + 1)
    curves = [
        ("total", "Total", main.COLORS["teacher"], 2.2),
        ("output", "Output", main.COLORS["output"], 2.0),
        ("hidden", "Hidden", main.COLORS["hidden"], 2.0),
        ("smoothness", "Smooth", "#8C6BB1", 1.75),
    ]
    for key, label, color, lw in curves:
        if key in losses:
            ax.semilogy(epochs, np.asarray(losses[key], dtype=float), color=color, lw=lw, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=2,
        handlelength=1.1,
        columnspacing=0.55,
        labelspacing=0.12,
        borderpad=0.0,
        fontsize=6.1,
    )
    main.style_axis(ax)



def plot_teacher_vs_student_errors(fig: plt.Figure, spec, trajectories: dict[str, np.ndarray]) -> None:
    """Teacher vs Student error heatmaps side-by-side (both vs Truth)."""
    sub = spec.subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.035], wspace=0.06)
    ax_t = fig.add_subplot(sub[0, 0])
    ax_s = fig.add_subplot(sub[0, 1], sharey=ax_t)
    cax = fig.add_subplot(sub[0, 2])

    teacher_err = np.abs(trajectories["teacher"] - trajectories["truth"])
    student_err = np.abs(trajectories["student"] - trajectories["truth"])

    all_vals = np.concatenate([teacher_err.ravel(), student_err.ravel()])
    finite = all_vals[np.isfinite(all_vals)]
    vmin, vmax = 0.0, max(0.005, float(np.nanpercentile(finite, 99.0)))
    if vmax <= vmin:
        vmax = vmin + 1

    cmap = make_error_cmap()
    norm = mpl.colors.PowerNorm(gamma=0.55, vmin=vmin, vmax=vmax)

    images = []
    for ax, data, title in [
        (ax_t, teacher_err, "Teacher"),
        (ax_s, student_err, "Student"),
    ]:
        img = ax.pcolormesh(
            trajectories["x"],
            main.TIME_POINTS,
            data,
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
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
        ax.set_xlim(float(trajectories["x"][0]), float(trajectories["x"][-1]))
        ax.set_xticks([0.0, 0.5, 1.0])
        set_log_time_y(ax)
        ax.tick_params(axis="both", width=0.8, length=2.4, pad=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.95)
        ax.set_xlabel(r"$x$")

    ax_t.set_ylabel(r"$t$")
    ax_s.tick_params(labelleft=False)
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label(r"$|e|$", labelpad=1.8)
    cbar.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    style_colorbar(cbar, label_size=6.4)
def get_hidden(model: torch.nn.Module, x_tensor: torch.Tensor, layer: str) -> torch.Tensor:
    if layer == "first":
        return model.get_first_hidden(x_tensor)
    return model.get_hidden_representation(x_tensor)


def compute_hidden_pca(teacher_bundle: tuple, student_bundle: tuple, gamma_bundle: dict, max_points: int = 950) -> dict[str, np.ndarray]:
    teacher_model, teacher_x, _, _ = teacher_bundle
    student_model, student_x, _, student_checkpoint = student_bundle
    data = gamma_bundle["data"]
    idx = np.linspace(0, len(data) - 1, min(max_points, len(data)), dtype=int)
    x_raw = np.zeros((len(idx), main.N_GRID + 1), dtype=np.float32)
    x_raw[:, 0] = np.log10(data[idx, 0] + 1.0)
    x_raw[:, 1:] = data[idx, 1 : main.N_GRID + 1]
    log_time = np.log10(np.maximum(data[idx, 0], main.TIME_POINTS[0]))
    layer = student_checkpoint.get("hidden_layer", "last")
    with torch.no_grad():
        xt = torch.tensor(teacher_x.transform(x_raw), dtype=torch.float32)
        xs = torch.tensor(student_x.transform(x_raw), dtype=torch.float32)
        teacher_hidden = get_hidden(teacher_model, xt, layer).cpu()
        student_hidden = get_hidden(student_model, xs, layer).cpu()
        proj_state = student_checkpoint.get("projection_state_dict")
        if proj_state is not None:
            out_dim, in_dim = proj_state["weight"].shape
            projection = torch.nn.Linear(in_dim, out_dim)
            projection.load_state_dict(proj_state)
            projection.eval()
            student_hidden = projection(student_hidden)
    teacher_np = teacher_hidden.numpy()
    student_np = student_hidden.numpy()
    center = teacher_np.mean(axis=0, keepdims=True)
    teacher_centered = teacher_np - center
    _, _, vt = np.linalg.svd(teacher_centered, full_matrices=False)
    components = vt[:2].T
    teacher_pc = teacher_centered @ components
    student_pc = (student_np - center) @ components
    scale = np.nanpercentile(np.abs(np.vstack([teacher_pc, student_pc])), 98)
    if np.isfinite(scale) and scale > 0:
        teacher_pc /= scale
        student_pc /= scale
    return {"teacher": teacher_pc, "student": student_pc, "log_time": log_time}


def plot_hidden_pca(ax: plt.Axes, hidden_pca: dict[str, np.ndarray]) -> None:
    norm = mpl.colors.Normalize(vmin=-4, vmax=1)
    cmap = "coolwarm"
    sc = ax.scatter(
        hidden_pca["teacher"][:, 0],
        hidden_pca["teacher"][:, 1],
        c=hidden_pca["log_time"],
        cmap=cmap,
        norm=norm,
        s=8,
        marker="o",
        alpha=0.34,
        linewidths=0,
        rasterized=True,
        label="Teacher",
    )
    ax.scatter(
        hidden_pca["student"][:, 0],
        hidden_pca["student"][:, 1],
        c=hidden_pca["log_time"],
        cmap=cmap,
        norm=norm,
        s=9,
        marker="x",
        alpha=0.62,
        linewidths=0.55,
        rasterized=True,
        label="Projected student",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.legend(frameon=False, loc="lower right", handlelength=0.8, borderpad=0.0, fontsize=6.2)
    main.style_axis(ax)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.018)
    cbar.set_label(r"$\log_{10}t$", labelpad=1.0)
    style_colorbar(cbar, label_size=6.1)


def plot_formula_panel(ax: plt.Axes, dynamics: dict[str, np.ndarray]) -> None:
    ax.axis("off")
    t_transition = float(dynamics["t_transition"])
    D_slow = float(dynamics["D_slow"])
    r_slow = float(dynamics["r_slow"])
    rmse_model = float(dynamics["rmse_model"])
    ax.text(
        0.02,
        0.72,
        r"$\frac{\partial u}{\partial t}=D(t)\frac{\partial^2u}{\partial x^2}+r(t)u(1-u)$",
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        0.02,
        0.43,
        rf"$D(t),r(t)=0$ for $t<t^*$;  $t^*={t_transition:.3g}$",
        transform=ax.transAxes,
        fontsize=7.4,
        fontweight="bold",
        color=main.COLORS["text"],
    )
    ax.text(
        0.02,
        0.24,
        rf"slow regime: $D={D_slow:.4f}$, $r={r_slow:.3f}$;  RMSE $={rmse_model:.3g}$",
        transform=ax.transAxes,
        fontsize=7.4,
        fontweight="bold",
        color=main.COLORS["text"],
    )
    ax.text(
        0.02,
        0.05,
        rf"true values: $D={main.EPSILON:.3f}$, $r={main.TRUE_R:.1f}$",
        transform=ax.transAxes,
        fontsize=7.2,
        fontweight="bold",
        color=main.COLORS["text"],
    )


def plot_discovered_dynamics(fig: plt.Figure, spec, dynamics: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(3, 1, hspace=0.22)
    axes = [fig.add_subplot(sub[i, 0]) for i in range(3)]
    times = np.asarray(dynamics["times"], dtype=float)
    mask = np.isfinite(times) & (times > 0)
    times = times[mask]
    D_t = np.asarray(dynamics["D_t"], dtype=float)[mask]
    r_t = np.asarray(dynamics["r_t"], dtype=float)[mask]
    residuals = np.asarray(dynamics["residuals"], dtype=float)[mask]
    t_transition = float(dynamics["t_transition"])
    # 真正从 1e-3 开始展示轨迹：过滤数据并重新对齐
    start_t = 1e-3
    keep = times >= start_t
    times = times[keep]
    D_t = D_t[keep]
    r_t = r_t[keep]
    residuals = residuals[keep]
    t_line = np.logspace(np.log10(start_t), np.log10(times.max()), 500)
    panels = [
        (D_t, two_regime_values(t_line, t_transition, float(dynamics["D_fast"]), float(dynamics["D_slow"])), main.EPSILON, r"$D(t)$", main.COLORS["teacher"]),
        (r_t, two_regime_values(t_line, t_transition, float(dynamics["r_fast"]), float(dynamics["r_slow"])), main.TRUE_R, r"$r(t)$", main.COLORS["fit"]),
        (residuals, None, None, "Residual", "#7B6D8D"),
    ]
    for i, (values, fit_values, true_value, ylabel, color) in enumerate(panels):
        ax = axes[i]
        ax.semilogx(times, values, color=color, marker="o", ms=2.0, lw=0.0, alpha=0.45, label="Extracted")
        if fit_values is not None:
            ax.semilogx(t_line, fit_values, color=main.COLORS["student"], lw=2.0, label="Two-regime")
            ax.axhline(true_value, color=main.COLORS["truth"], lw=1.35, ls="--", label="True")
        ax.axvline(t_transition, color=main.COLORS["text"], lw=0.9, ls=":", alpha=0.85)
        ax.set_ylabel(ylabel)
        set_log_time_x(ax, xmin=max(times.min(), 1e-3), xmax=times.max())
        ax.set_xticks([1e-3, 1e-2, 1e0, 1e1])
        ax.get_xaxis().set_major_formatter(mpl.ticker.LogFormatterMathtext())
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        main.style_axis(ax)
        # 紧凑 y 轴：去掉纵轴与曲线之间的空白
        y_arrays = [values]
        if fit_values is not None:
            y_arrays.append(fit_values)
        if true_value is not None:
            y_arrays.append(np.full_like(values, true_value))
        y_min = min(arr.min() for arr in y_arrays)
        y_max = max(arr.max() for arr in y_arrays)
        pad = max((y_max - y_min) * 0.03, 0.001)
        ax.set_ylim(y_min - pad, y_max + pad)
        if i == 2:
            ax.set_xlabel(r"$t$")
        else:
            ax.tick_params(labelbottom=False)
    axes[0].legend(
        frameon=False,
        loc="center left",
        ncol=1,
        handlelength=1.0,
        columnspacing=0.45,
        labelspacing=0.1,
        borderpad=0.0,
        fontsize=5.9,
    )


def plot_reduced_validation(fig: plt.Figure, spec, dynamics: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.050], wspace=0.09)
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]
    cax = fig.add_subplot(sub[0, 3])
    x_grid = np.asarray(dynamics["x_grid"], dtype=float)
    time_points = np.asarray(dynamics["time_points"], dtype=float)
    student = np.asarray(dynamics["traj_student"], dtype=float)
    model = np.asarray(dynamics["traj_model"], dtype=float)
    error = np.abs(student - model)
    err_vmax = max(0.003, float(np.nanpercentile(error, 99.0)))
    panels = [
        (student, "Student", "viridis", mpl.colors.Normalize(vmin=0.0, vmax=1.0)),
        (model, "Two-regime", "viridis", mpl.colors.Normalize(vmin=0.0, vmax=1.0)),
        (error, "Abs. error", make_error_cmap(), mpl.colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=err_vmax)),
    ]
    mappable = None
    for i, (values, title, cmap, norm) in enumerate(panels):
        ax = axes[i]
        mesh = ax.pcolormesh(x_grid, time_points, values, shading="auto", cmap=cmap, norm=norm, rasterized=True)
        if i == 2:
            mappable = mesh
        ax.set_title(title, pad=1.5, fontsize=7.2)
        ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yscale("log")
        ax.set_yticks([1e-3, 1e-1, 1e1])
        ax.get_yaxis().set_major_formatter(mpl.ticker.LogFormatterMathtext())
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlabel(r"$x$", labelpad=1.0)
        main.style_axis(ax, grid=False)
        if i == 0:
            ax.set_ylabel(r"$t$")
        else:
            ax.tick_params(labelleft=False)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r"$|e|$", labelpad=0.7)
    cbar.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    style_colorbar(cbar, label_size=5.9)


def build_diagnostics(teacher_bundle: tuple, student_bundle: tuple, student_checkpoint: dict, gamma_bundle: dict, dynamics: dict[str, np.ndarray], trajectories: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(7.2, 5.6))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.75],
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.92,
        hspace=0.22,
    )
    plot_solution_heatmaps(fig, outer[0, 0], trajectories)
    plot_error_heatmaps(fig, outer[1, 0], trajectories)

    # 统一放置 panel label 和标题（与正文大图一致）
    panel_info = [
        (outer[0, 0], "a", "Full space-time trajectories"),
        (outer[1, 0], "b", "Student error"),
    ]
    for cell, label, title in panel_info:
        bbox = cell.get_position(fig)
        y = bbox.y1 + 0.005
        x_label = bbox.x0 - 0.022
        x_title = bbox.x0
        fig.text(x_label, y, label, fontsize=13, fontweight="bold", ha="left", va="bottom")
        fig.text(x_title, y, title, fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"fisher_kpp_supp_diagnostics.{ext}", facecolor="white")
    fig.savefig(PANEL_DIR / "fisher_kpp_supp_diagnostics_preview.png", facecolor="white", dpi=300)
    plt.close(fig)


def build_figures() -> None:
    warnings.filterwarnings("ignore")
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints...", flush=True)
    teacher_bundle = main.load_checkpoint_model(main.TEACHER_MODEL, is_student=False)
    student_bundle = main.load_checkpoint_model(main.STUDENT_MODEL, is_student=True)
    student_checkpoint = student_bundle[3]

    print("Loading HMM and discovered dynamics...", flush=True)
    gamma_bundle = load_gamma_bundle()
    dynamics = dict(np.load(main.DYNAMICS_DATA))

    print("Solving BDF and evaluating teacher/student...", flush=True)
    trajectories = main.compute_trajectories(teacher_bundle, student_bundle)

    print("Building trajectory atlas...", flush=True)
    build_trajectory_atlas(trajectories, gamma_bundle)

    print("Building diagnostics figure...", flush=True)
    build_diagnostics(teacher_bundle, student_bundle, student_checkpoint, gamma_bundle, dynamics, trajectories)
    print(f"Saved supplementary figures to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    build_figures()
