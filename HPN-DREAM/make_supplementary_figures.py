#!/usr/bin/env python3
"""Build Nature-style supplementary figures for the HPN-DREAM cancer example."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path("/private/tmp/hpn_dream_supp_mpl")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator

import make_discovery_figure as disc
import make_nature_figure as base


OUT_DIR = ROOT / "results" / "nature_figure" / "supplementary"
PANEL_DIR = OUT_DIR / "panels"

FIGSIZE = (7.2, 9.4)
EPS = 1e-8

KEY_PROTEINS = [
    "EGFR_pY1068",
    "AKT_pS473",
    "AKT_pT308",
    "MAPK_pT202_Y204",
    "MEK1_pS217_S221",
    "mTOR_pS2448",
    "S6_pS235_S236",
    "S6_pS240_S244",
    "p70S6K_pT389",
    "4EBP1_pS65",
    "STAT3_pY705",
    "YB-1_PS102",
]

TRAJECTORY_PANELS = [
    ("EGFR_pY1068", "EGF|None"),
    ("EGFR_pY1173", "NRG1|None"),
    ("AKT_pS473", "IGF1|None"),
    ("AKT_pT308", "Insulin|None"),
    ("MAPK_pT202_Y204", "HGF|None"),
    ("MEK1_pS217_S221", "EGF|GSK690693"),
    ("mTOR_pS2448", "Serum|None"),
    ("S6_pS235_S236", "FGF1|None"),
    ("S6_pS240_S244", "FGF1|GSK690693_GSK1120212"),
    ("p70S6K_pT389", "Serum|PD173074"),
    ("STAT3_pY705", "HGF|PD173074"),
    ("YB-1_PS102", "Serum|PD173074"),
]

LIGAND_TRAJECTORY_CONDITIONS = [
    "Serum|None",
    "NRG1|None",
    "Insulin|None",
    "IGF1|None",
    "HGF|None",
    "FGF1|None",
    "EGF|None",
]

PERTURBATION_TRAJECTORY_CONDITIONS = [
    "Serum|PD173074",
    "NRG1|PD173074",
    "HGF|PD173074",
    "FGF1|GSK690693_GSK1120212",
    "EGF|GSK690693",
    "Insulin|GSK690693_GSK1120212",
    "IGF1|GSK690693",
]

CORE_TRAJECTORY_PROTEINS = [
    "EGFR_pY1068",
    "AKT_pS473",
    "AKT_pT308",
    "MAPK_pT202_Y204",
    "mTOR_pS2448",
    "S6_pS235_S236",
]

DOWNSTREAM_TRAJECTORY_PROTEINS = [
    "MEK1_pS217_S221",
    "S6_pS240_S244",
    "p70S6K_pT389",
    "4EBP1_pS65",
    "STAT3_pY705",
    "YB-1_PS102",
]

REP_CONDITIONS = [
    "None|None",
    "EGF|None",
    "NRG1|None",
    "HGF|None",
    "FGF1|GSK690693_GSK1120212",
    "Serum|PD173074",
]

EDGE_CLASS_COLORS = {
    "Common": "#303030",
    "Teacher-only": "#0B67A3",
    "Student-only": "#C81E1E",
}


def configure_style() -> None:
    base.configure_style()
    mpl.rcParams.update(
        {
            "font.size": 6.9,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.9,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )


def panel_label(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="black",
    )


def add_group_label(fig: plt.Figure, axes: list[plt.Axes], letter: str, xpad: float = 0.030, ypad: float = 0.006) -> None:
    boxes = [ax.get_position() for ax in axes]
    fig.text(
        min(box.x0 for box in boxes) - xpad,
        max(box.y1 for box in boxes) + ypad,
        letter,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def condition_to_underscore(condition: str) -> str:
    return condition.replace("|", "_")


def condition_to_pipe(condition: str) -> str:
    if "|" in condition:
        return condition
    parts = condition.split("_", 1)
    return f"{parts[0]}|{parts[1]}" if len(parts) == 2 else condition


def condition_short(condition: str) -> str:
    return base.condition_label(condition).replace(" + ", "\n+ ")


def style_cbar(cbar, label_size: float = 5.8) -> None:
    cbar.outline.set_linewidth(0.75)
    cbar.ax.tick_params(width=0.8, length=2.2, pad=1.0, labelsize=label_size)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")


def load_predictions() -> tuple[dict, np.ndarray, np.ndarray, tuple, tuple]:
    high = base.load_high_res()
    teacher_bundle = base.load_model(base.TEACHER_MODEL)
    student_bundle = base.load_model(base.STUDENT_MODEL)
    teacher_predictions = base.predict_model(teacher_bundle[0], teacher_bundle[1], high["x"])
    student_predictions = base.predict_model(student_bundle[0], student_bundle[1], high["x"])
    return high, teacher_predictions, student_predictions, teacher_bundle, student_bundle


def load_experimental() -> tuple[np.ndarray, np.ndarray, dict]:
    exp_path = base.resolve_experimental_path()
    _, x_exp, y_exp, info_exp = base.load_midas(exp_path)
    return x_exp, y_exp, info_exp


def all_condition_series(x_exp: np.ndarray, y_exp: np.ndarray, info_exp: dict) -> dict[str, dict]:
    series: dict[str, dict] = {}
    needed_conditions = (
        REP_CONDITIONS
        + [cond for _, cond in TRAJECTORY_PANELS]
        + LIGAND_TRAJECTORY_CONDITIONS
        + PERTURBATION_TRAJECTORY_CONDITIONS
    )
    for condition in sorted(set(needed_conditions)):
        stimulus, inhibitor = condition.split("|")
        try:
            series[condition] = base.aggregate_condition_series(
                x_exp,
                y_exp,
                info_exp,
                stimulus,
                inhibitor,
                max_time=240.0,
            )
        except Exception:
            continue
    return series


def dense_condition_trajectory(high: dict, predictions: np.ndarray, condition: str) -> tuple[np.ndarray, np.ndarray]:
    return base.condition_trajectory(high, predictions, condition)


def interpolate_dense(times_dense: np.ndarray, values_dense: np.ndarray, times: np.ndarray) -> np.ndarray:
    out = np.empty((len(times), values_dense.shape[1]), dtype=float)
    for j in range(values_dense.shape[1]):
        out[:, j] = np.interp(times, times_dense, values_dense[:, j])
    return out


def robust_range(values: np.ndarray, axis=0, floor: float = 0.05) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    spread = np.nanpercentile(values, 95, axis=axis) - np.nanpercentile(values, 5, axis=axis)
    std = np.nanstd(values, axis=axis)
    median_abs = np.nanmedian(np.abs(values), axis=axis)
    floor_val = np.maximum(floor, 0.10 * median_abs)
    return np.where(spread > floor_val, spread, np.maximum(std, floor_val))


def scaled_rmse(pred: np.ndarray, ref: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    rmse = np.sqrt(np.nanmean((pred - ref) ** 2, axis=0))
    return rmse / (np.asarray(scale, dtype=float) + EPS)


def normalized_rmse(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return scaled_rmse(pred, ref, robust_range(ref, axis=0))


def display_nrmse(values: np.ndarray) -> np.ndarray:
    return np.log10(1.0 + np.asarray(values, dtype=float))


def compute_fit_tables(
    high: dict,
    teacher_predictions: np.ndarray,
    student_predictions: np.ndarray,
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    info_exp: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teacher_rows = []
    student_rows = []
    distribution_rows = []
    teacher_proteins = high["protein_names"]
    exp_proteins = info_exp["protein_names"]
    exp_scale = {}
    teacher_scale = {}
    for protein in KEY_PROTEINS:
        if protein in exp_proteins:
            exp_idx = exp_proteins.index(protein)
            exp_scale[protein] = float(robust_range(y_exp[:, exp_idx], axis=None))
        if protein in teacher_proteins:
            high_idx = teacher_proteins.index(protein)
            teacher_scale[protein] = float(robust_range(teacher_predictions[:, high_idx], axis=None))
    for condition in high["condition_names"]:
        stim, inhib = condition.split("|")
        try:
            exp_series = base.aggregate_condition_series(x_exp, y_exp, info_exp, stim, inhib, max_time=240.0)
        except Exception:
            continue
        t_teacher, teacher_traj = dense_condition_trajectory(high, teacher_predictions, condition)
        _, student_traj = dense_condition_trajectory(high, student_predictions, condition)
        teacher_at_exp = interpolate_dense(t_teacher, teacher_traj, exp_series["time"])
        exp_vals = exp_series["median"]
        exp_rmse_values = []
        fid_rmse_values = []
        for protein in KEY_PROTEINS:
            if protein not in teacher_proteins or protein not in exp_proteins:
                continue
            t_idx = teacher_proteins.index(protein)
            e_idx = exp_proteins.index(protein)
            exp_rmse = scaled_rmse(teacher_at_exp[:, [t_idx]], exp_vals[:, [e_idx]], exp_scale[protein])[0]
            fid_rmse = scaled_rmse(student_traj[:, [t_idx]], teacher_traj[:, [t_idx]], teacher_scale[protein])[0]
            teacher_rows.append({"condition": condition, "protein": protein, "nrmse": exp_rmse})
            student_rows.append({"condition": condition, "protein": protein, "nrmse": fid_rmse})
            exp_rmse_values.append(exp_rmse)
            fid_rmse_values.append(fid_rmse)
        if exp_rmse_values:
            distribution_rows.append(
                {
                    "condition": condition,
                    "teacher_exp_nrmse": float(np.nanmedian(exp_rmse_values)),
                    "student_teacher_nrmse": float(np.nanmedian(fid_rmse_values)),
                }
            )
    return pd.DataFrame(teacher_rows), pd.DataFrame(student_rows), pd.DataFrame(distribution_rows)


def pivot_metric(df: pd.DataFrame, high: dict) -> tuple[np.ndarray, list[str]]:
    condition_order = [c for c in high["condition_names"] if c in set(df["condition"])]
    matrix = np.full((len(condition_order), len(KEY_PROTEINS)), np.nan)
    for i, condition in enumerate(condition_order):
        for j, protein in enumerate(KEY_PROTEINS):
            vals = df[(df["condition"] == condition) & (df["protein"] == protein)]["nrmse"]
            if len(vals):
                matrix[i, j] = float(vals.iloc[0])
    return matrix, condition_order


def plot_experimental_atlas(
    fig: plt.Figure,
    spec,
    high: dict,
    teacher_predictions: np.ndarray,
    student_predictions: np.ndarray,
    exp_series: dict[str, dict],
    exp_proteins: list[str],
) -> list[plt.Axes]:
    axes = []
    sub = spec.subgridspec(3, 4, hspace=0.35, wspace=0.22)
    teacher_proteins = high["protein_names"]
    for idx, (protein, condition) in enumerate(TRAJECTORY_PANELS):
        ax = fig.add_subplot(sub[idx // 4, idx % 4])
        axes.append(ax)
        if condition not in exp_series or protein not in exp_proteins or protein not in teacher_proteins:
            ax.axis("off")
            continue
        t_dense, teacher_traj = dense_condition_trajectory(high, teacher_predictions, condition)
        _, student_traj = dense_condition_trajectory(high, student_predictions, condition)
        exp = exp_series[condition]
        exp_idx = exp_proteins.index(protein)
        high_idx = teacher_proteins.index(protein)
        y = exp["median"][:, exp_idx]
        yerr = np.vstack([y - exp["lo"][:, exp_idx], exp["hi"][:, exp_idx] - y])
        ax.errorbar(
            exp["time"],
            y,
            yerr=yerr,
            fmt="o",
            color=base.COLORS["black"],
            mfc="white",
            mec=base.COLORS["black"],
            mew=0.8,
            ms=3.2,
            capsize=1.5,
            elinewidth=0.8,
            label="Experiment" if idx == 0 else None,
            zorder=3,
        )
        ax.plot(t_dense, teacher_traj[:, high_idx], color=base.COLORS["teacher"], lw=1.65, label="Teacher" if idx == 0 else None)
        ax.plot(t_dense, student_traj[:, high_idx], color=base.COLORS["student"], lw=1.85, label="Student" if idx == 0 else None)
        ax.set_title(base.display_name(protein).replace("\n", " "), pad=1.2, fontsize=6.8)
        ax.text(
            0.03,
            0.06,
            condition_short(condition),
            transform=ax.transAxes,
            fontsize=4.9,
            fontweight="bold",
            color=base.COLORS["mid"],
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.3),
        )
        base.format_time_axis(ax, label=idx // 4 == 2, xmax=250)
        if idx // 4 != 2:
            ax.tick_params(labelbottom=False)
        if idx % 4 == 0:
            ax.set_ylabel("Signal", labelpad=1.0)
        else:
            ax.tick_params(labelleft=False)
        base.set_sparse_y(ax, 3)
        base.padded_ylim(ax, [y, exp["lo"][:, exp_idx], exp["hi"][:, exp_idx], teacher_traj[:, high_idx], student_traj[:, high_idx]], frac=0.12)
        base.style_axis(ax, grid=True)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.06),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.55,
        labelspacing=0.1,
        borderpad=0.0,
    )
    axes[0].text(0.0, 1.32, "Experimental trajectory atlas", transform=axes[0].transAxes, fontsize=9.0, fontweight="bold")
    return axes


def protein_column_limits(
    high: dict,
    teacher_predictions: np.ndarray,
    student_predictions: np.ndarray,
    exp_series: dict[str, dict],
    exp_proteins: list[str],
    proteins: list[str],
    conditions: list[str],
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    teacher_proteins = high["protein_names"]
    for protein in proteins:
        values = []
        if protein not in teacher_proteins or protein not in exp_proteins:
            limits[protein] = (0.0, 1.0)
            continue
        high_idx = teacher_proteins.index(protein)
        exp_idx = exp_proteins.index(protein)
        for condition in conditions:
            if condition not in exp_series:
                continue
            _, teacher_traj = dense_condition_trajectory(high, teacher_predictions, condition)
            _, student_traj = dense_condition_trajectory(high, student_predictions, condition)
            exp = exp_series[condition]
            values.extend(
                [
                    teacher_traj[:, high_idx],
                    student_traj[:, high_idx],
                    exp["median"][:, exp_idx],
                    exp["lo"][:, exp_idx],
                    exp["hi"][:, exp_idx],
                ]
            )
        arr = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            limits[protein] = (0.0, 1.0)
            continue
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        pad = max(0.04, 0.10 * (hi - lo))
        limits[protein] = (lo - pad, hi + pad)
    return limits


def plot_condition_trajectory_grid(
    fig: plt.Figure,
    spec,
    high: dict,
    teacher_predictions: np.ndarray,
    student_predictions: np.ndarray,
    exp_series: dict[str, dict],
    exp_proteins: list[str],
    proteins: list[str],
    conditions: list[str],
    title: str,
    label: str,
) -> list[plt.Axes]:
    teacher_proteins = high["protein_names"]
    limits = protein_column_limits(high, teacher_predictions, student_predictions, exp_series, exp_proteins, proteins, conditions)
    sub = spec.subgridspec(len(conditions), len(proteins), hspace=0.16, wspace=0.16)
    axes: list[plt.Axes] = []
    for row, condition in enumerate(conditions):
        for col, protein in enumerate(proteins):
            ax = fig.add_subplot(sub[row, col])
            axes.append(ax)
            missing = condition not in exp_series or protein not in exp_proteins or protein not in teacher_proteins
            if missing:
                ax.axis("off")
                continue
            t_dense, teacher_traj = dense_condition_trajectory(high, teacher_predictions, condition)
            _, student_traj = dense_condition_trajectory(high, student_predictions, condition)
            exp = exp_series[condition]
            exp_idx = exp_proteins.index(protein)
            high_idx = teacher_proteins.index(protein)
            y = exp["median"][:, exp_idx]
            yerr = np.vstack([y - exp["lo"][:, exp_idx], exp["hi"][:, exp_idx] - y])
            ax.errorbar(
                exp["time"],
                y,
                yerr=yerr,
                fmt="o",
                color=base.COLORS["black"],
                mfc="white",
                mec=base.COLORS["black"],
                mew=0.65,
                ms=2.35,
                capsize=1.1,
                elinewidth=0.68,
                label="Experiment" if row == 0 and col == 0 else None,
                zorder=3,
            )
            ax.plot(
                t_dense,
                teacher_traj[:, high_idx],
                color=base.COLORS["teacher"],
                lw=1.28,
                label="Teacher" if row == 0 and col == 0 else None,
                zorder=2,
            )
            ax.plot(
                t_dense,
                student_traj[:, high_idx],
                color=base.COLORS["student"],
                lw=1.50,
                label="Student" if row == 0 and col == 0 else None,
                zorder=2,
            )
            ax.set_ylim(*limits[protein])
            ax.set_xlim(-4, 250)
            ax.xaxis.set_major_locator(FixedLocator([0, 60, 120, 240]))
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.set_xticklabels(["0", "60", "120", "240"] if row == len(conditions) - 1 else [])
            ax.yaxis.set_major_locator(MaxNLocator(2))
            if col == 0:
                ax.set_ylabel(
                    base.condition_label(condition),
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    fontweight="bold",
                    color="#444444",
                    labelpad=5.0,
                )
                ax.tick_params(labelleft=True, labelsize=5.0)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(
                    base.display_name(protein).replace("\n", " "),
                    fontsize=6.2,
                    fontweight="bold",
                    color="black",
                    pad=2.5,
                )

            ax.tick_params(width=0.75, length=1.8, pad=0.7, labelsize=5.0)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
            base.style_axis(ax, grid=True)
    handles, labels = axes[0].get_legend_handles_labels()
    # Place legend on the bottom-right subplot
    legend_ax = axes[-1]
    legend_ax.legend(
        handles,
        labels,
        loc="lower right",
        frameon=False,
        handlelength=1.0,
        columnspacing=0.55,
        labelspacing=0.1,
        borderpad=0.0,
        fontsize=5.7,
    )
    return axes


def build_trajectory_atlas(
    high: dict,
    teacher_predictions: np.ndarray,
    student_predictions: np.ndarray,
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    info_exp: dict,
    conditions: list[str],
    proteins: list[str],
    title: str,
    stem: str,
) -> None:
    exp_series = all_condition_series(x_exp, y_exp, info_exp)
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(
        1,
        1,
        left=0.055,
        right=0.985,
        bottom=0.062,
        top=0.90,
    )
    axes = plot_condition_trajectory_grid(
        fig,
        outer[0, 0],
        high,
        teacher_predictions,
        student_predictions,
        exp_series,
        info_exp["protein_names"],
        proteins,
        conditions,
        title,
        "a",
    )

    # 统一放置标题
    bbox = outer[0, 0].get_position(fig)
    y = bbox.y1 + 0.012
    x_title = bbox.x0
    fig.text(x_title, y, title.replace("Supplementary ", ""), fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    fig.canvas.draw()

    # Place a shared x-axis label "Time (min)" below the bottom row
    n_cols = len(proteins)
    n_rows = len(conditions)
    bottom_row_axes = axes[(n_rows - 1) * n_cols: n_rows * n_cols]
    left_x = min(ax.get_position().x0 for ax in bottom_row_axes)
    right_x = max(ax.get_position().x1 for ax in bottom_row_axes)
    bottom_y = min(ax.get_position().y0 for ax in bottom_row_axes)
    fig.text(
        (left_x + right_x) / 2,
        bottom_y - 0.012,
        "Time (min)",
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color="black",
    )

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", facecolor="white")
    fig.savefig(PANEL_DIR / f"{stem}_preview.png", facecolor="white", dpi=300)
    plt.close(fig)


def plot_metric_heatmap(fig: plt.Figure, ax: plt.Axes, matrix: np.ndarray, condition_order: list[str], title: str, label: str) -> None:
    plot_matrix = display_nrmse(matrix)
    vmax = min(1.05, max(0.18, float(np.nanpercentile(plot_matrix, 97))))
    img = ax.imshow(plot_matrix, aspect="auto", cmap="magma_r", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, pad=2.0)
    ax.set_xticks(np.arange(len(KEY_PROTEINS)))
    ax.set_xticklabels([base.short_name(p).replace("\n", " ") for p in KEY_PROTEINS], rotation=45, ha="right", fontsize=4.8)
    y_ticks = np.arange(0, len(condition_order), 4)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([condition_to_underscore(condition_order[i]) for i in y_ticks], fontsize=4.9)
    ax.tick_params(width=0.8, length=2.0, pad=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.016)
    cbar.set_label(r"$\log_{10}(1+\mathrm{nRMSE})$", labelpad=1.0)
    style_cbar(cbar)
    panel_label(ax, label, x=-0.12, y=1.14)


def plot_phase_map(fig: plt.Figure, spec) -> None:
    sub = spec.subgridspec(1, 2, width_ratios=[1.72, 0.90], wspace=0.22)
    ax = fig.add_subplot(sub[0, 0])
    side = sub[0, 1].subgridspec(2, 1, hspace=0.35)
    ax_post = fig.add_subplot(side[0, 0])
    ax_occ = fig.add_subplot(side[1, 0])
    data = np.load(base.HMM_DATA, allow_pickle=True)
    post = data["posteriors"].astype(float)
    cond_idx_all = data["condition_indices"].astype(int)
    time_idx_all = data["time_indices"].astype(int)
    time_points = data["time_points"].astype(float)
    valid_rows = np.flatnonzero(time_idx_all > 0)[: len(post)]
    cond_idx = cond_idx_all[valid_rows]
    time_idx = time_idx_all[valid_rows]
    times = time_points[time_idx]
    raw_phase = np.argmax(post, axis=1)
    medians = [np.median(times[raw_phase == k]) for k in range(post.shape[1])]
    fast = int(np.argmin(medians))
    slow = 1 - fast
    hard = np.full((int(data["treatment_conditions"].shape[0]), len(time_points) - 1), np.nan)
    for row, c, t in zip(raw_phase, cond_idx, time_idx):
        hard[int(c), int(t) - 1] = 0 if row == fast else 1
    cmap = mpl.colors.ListedColormap([base.COLORS["fast"], base.COLORS["slow"]])
    ax.imshow(hard, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("HMM phase map across conditions", pad=2.0)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Condition")
    ax.set_xticks([0, 24, 48, 72, 95], ["0", "24", "48", "72", "96"])
    ax.set_yticks([0, 8, 16, 24, 32])
    ax.tick_params(width=0.8, length=2.0, pad=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    df = pd.DataFrame({"time": times, "fast": post[:, fast], "slow": post[:, slow], "phase": raw_phase})
    mean = df.groupby("time")[["fast", "slow"]].mean().reset_index()
    ax_post.plot(mean["time"], mean["fast"], color=base.COLORS["fast"], lw=1.9, label="Early")
    ax_post.plot(mean["time"], mean["slow"], color=base.COLORS["slow"], lw=1.9, label="Late")
    base.format_time_axis(ax_post, label=False, log_only=True)
    ax_post.set_ylim(-0.04, 1.04)
    ax_post.set_ylabel("Posterior", labelpad=0.8)
    ax_post.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.0]))
    base.style_axis(ax_post, grid=True)
    ax_post.legend(frameon=False, loc="center right", handlelength=1.0, fontsize=5.2)

    counts = [int(np.sum(raw_phase == fast)), int(np.sum(raw_phase == slow))]
    ax_occ.bar([0, 1], counts, color=[base.COLORS["fast"], base.COLORS["slow"]], width=0.62)
    ax_occ.set_xticks([0, 1], ["Early", "Late"])
    ax_occ.set_ylabel("Samples", labelpad=0.8)
    ax_occ.yaxis.set_major_locator(MaxNLocator(3))
    base.style_axis(ax_occ, grid=True)
    panel_label(ax, "d", x=-0.13, y=1.14)


def get_hidden(model: torch.nn.Module, x_tensor: torch.Tensor, layer: str) -> torch.Tensor:
    if layer == "first":
        return model.get_first_hidden(x_tensor)
    return model.get_hidden_representation(x_tensor)


def compute_hidden_pca(teacher_bundle: tuple, student_bundle: tuple, high: dict, max_points: int = 1200) -> dict[str, np.ndarray]:
    teacher_model, _ = teacher_bundle
    student_model, student_ckpt = student_bundle
    idx = np.linspace(0, len(high["x"]) - 1, min(max_points, len(high["x"])), dtype=int)
    x_raw = high["x"][idx].astype(np.float32)
    time_color = x_raw[:, -1]
    layer = student_ckpt.get("hidden_layer", "last")
    with torch.no_grad():
        x_tensor = torch.tensor(x_raw, dtype=torch.float32)
        teacher_hidden = get_hidden(teacher_model, x_tensor, layer).cpu()
        student_hidden = get_hidden(student_model, x_tensor, layer).cpu()
        proj_state = student_ckpt.get("projection_state_dict")
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
    return {"teacher": teacher_pc, "student": student_pc, "time": time_color}


def plot_pakd_diagnostics(fig: plt.Figure, spec, student_ckpt: dict, hidden_pca: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.24)
    ax_loss = fig.add_subplot(sub[0, 0])
    ax_pca = fig.add_subplot(sub[0, 1])
    losses = student_ckpt["training_losses"]
    epochs = np.arange(1, len(losses["total"]) + 1)
    curves = [
        ("total", "Total", base.COLORS["black"]),
        ("output", "Output", base.COLORS["output"]),
        ("hidden", "Hidden", base.COLORS["hidden"]),
        ("smoothness", "Smooth", "#8C6BB1"),
    ]
    for key, label, color in curves:
        if key in losses:
            vals = np.maximum(np.asarray(losses[key], dtype=float), EPS)
            ax_loss.plot(epochs, base.moving_average(vals, 7), color=color, lw=1.8, label=label)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("PAKD losses", pad=1.5)
    ax_loss.xaxis.set_major_locator(MaxNLocator(4))
    ax_loss.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    base.style_axis(ax_loss, grid=True)
    ax_loss.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.0, columnspacing=0.45, fontsize=5.2)

    norm = mpl.colors.Normalize(vmin=0, vmax=240)
    sc = ax_pca.scatter(hidden_pca["teacher"][:, 0], hidden_pca["teacher"][:, 1], c=hidden_pca["time"], cmap="viridis", norm=norm, s=6, alpha=0.32, linewidths=0, rasterized=True, label="Teacher")
    ax_pca.scatter(hidden_pca["student"][:, 0], hidden_pca["student"][:, 1], c=hidden_pca["time"], cmap="viridis", norm=norm, s=7, alpha=0.56, marker="x", linewidths=0.45, rasterized=True, label="Student")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.set_title("Hidden PCA", pad=1.5)
    ax_pca.xaxis.set_major_locator(MaxNLocator(3))
    ax_pca.yaxis.set_major_locator(MaxNLocator(3))
    ax_pca.legend(frameon=False, loc="lower right", handlelength=0.8, fontsize=5.2)
    base.style_axis(ax_pca, grid=True)
    cbar = fig.colorbar(sc, ax=ax_pca, fraction=0.050, pad=0.018)
    cbar.set_label("Time", labelpad=0.8)
    style_cbar(cbar, label_size=5.1)
    panel_label(ax_loss, "e", x=-0.17, y=1.15)


def plot_performance_distribution(ax: plt.Axes, dist_df: pd.DataFrame) -> None:
    data = [
        display_nrmse(dist_df["teacher_exp_nrmse"].dropna().to_numpy()),
        display_nrmse(dist_df["student_teacher_nrmse"].dropna().to_numpy()),
    ]
    colors = [base.COLORS["teacher"], base.COLORS["student"]]
    labels = ["Teacher\nvs exp.", "Student\nvs teacher"]
    parts = ax.violinplot(data, positions=[0, 1], widths=0.58, showmeans=False, showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.18)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)
    rng = np.random.default_rng(9)
    for i, vals in enumerate(data):
        ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.035, len(vals)), vals, s=13, color=colors[i], alpha=0.72, edgecolor="white", linewidth=0.25)
        ax.plot([i - 0.20, i + 0.20], [np.nanmedian(vals), np.nanmedian(vals)], color="black", lw=1.2)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel(r"Median $\log_{10}(1+\mathrm{nRMSE})$")
    ax.set_title("Condition-level performance", pad=2.0)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    base.style_axis(ax, grid=True)
    panel_label(ax, "f", x=-0.12, y=1.14)


def build_experimental_atlas(
    high: dict,
    teacher_predictions: np.ndarray,
    student_predictions: np.ndarray,
    teacher_bundle: tuple,
    student_bundle: tuple,
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    info_exp: dict,
) -> None:
    exp_series = all_condition_series(x_exp, y_exp, info_exp)
    teacher_metric, student_metric, dist_df = compute_fit_tables(high, teacher_predictions, student_predictions, x_exp, y_exp, info_exp)
    teacher_matrix, condition_order = pivot_metric(teacher_metric, high)
    student_matrix, _ = pivot_metric(student_metric, high)
    hidden_pca = compute_hidden_pca(teacher_bundle, student_bundle, high)

    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(
        4,
        2,
        height_ratios=[2.55, 1.28, 1.20, 1.12],
        left=0.070,
        right=0.970,
        bottom=0.058,
        top=0.930,
        hspace=0.55,
        wspace=0.28,
    )
    axes_a = plot_experimental_atlas(fig, outer[0, :], high, teacher_predictions, student_predictions, exp_series, info_exp["protein_names"])
    ax_b = fig.add_subplot(outer[1, 0])
    ax_c = fig.add_subplot(outer[1, 1])
    plot_metric_heatmap(fig, ax_b, teacher_matrix, condition_order, "Teacher vs experimental median", "b")
    plot_metric_heatmap(fig, ax_c, student_matrix, condition_order, "Student vs teacher high-res", "c")
    plot_phase_map(fig, outer[2, 0])
    plot_pakd_diagnostics(fig, outer[2, 1], student_bundle[1], hidden_pca)
    ax_f = fig.add_subplot(outer[3, :])
    plot_performance_distribution(ax_f, dist_df)
    fig.suptitle("Supplementary HPN-DREAM experimental and distillation atlas", fontsize=12.2, fontweight="bold", y=0.985)
    fig.canvas.draw()
    add_group_label(fig, axes_a, "a")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"hpn_dream_supp_experimental_atlas.{ext}", facecolor="white")
    fig.savefig(PANEL_DIR / "hpn_dream_supp_experimental_atlas_preview.png", facecolor="white", dpi=300)
    plt.close(fig)


def aggregate_edge_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return (
        df.groupby("k_edges")["fit_quality"]
        .agg(mean="mean", q25=lambda x: np.percentile(x, 25), q75=lambda x: np.percentile(x, 75))
        .reset_index()
    )


def plot_edge_economy(ax: plt.Axes) -> None:
    teacher = aggregate_edge_curve(base.EDGE_TEACHER)
    student = aggregate_edge_curve(base.EDGE_STUDENT)
    for df, color, label in [(teacher, base.COLORS["teacher"], "Teacher"), (student, base.COLORS["student"], "Student")]:
        ax.plot(df["k_edges"], df["mean"], color=color, lw=2.1, label=label)
        ax.fill_between(df["k_edges"], df["q25"], df["q75"], color=color, alpha=0.13, lw=0)
    ax.set_xlim(0, 1640)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Edges kept")
    ax.set_ylabel("Fit quality")
    ax.set_title("Edge economy across conditions", pad=2.0)
    ax.xaxis.set_major_locator(FixedLocator([0, 400, 800, 1200, 1640]))
    ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 0.9, 1.0]))
    base.style_axis(ax, grid=True)
    ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    panel_label(ax, "a", x=-0.14, y=1.13)


def plot_edge_budget_summary(ax: plt.Axes) -> None:
    teacher = pd.read_csv(base.SUMMARY_TEACHER)
    student = pd.read_csv(base.SUMMARY_STUDENT)
    rows = []
    for source, df in [("Teacher", teacher), ("Student", student)]:
        for threshold in [0.90, 0.95]:
            vals = df[np.isclose(df["pct_threshold"], threshold)]["min_edges"].to_numpy(dtype=float)
            rows.append((source, threshold, vals))
    positions = [0.0, 0.34, 1.0, 1.34]
    colors = [base.COLORS["teacher"], base.COLORS["student"], base.COLORS["teacher"], base.COLORS["student"]]
    for pos, color, (_, _, vals) in zip(positions, colors, rows):
        box = ax.boxplot(vals, positions=[pos], widths=0.23, patch_artist=True, showfliers=False, manage_ticks=False)
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.1)
        for element in ["whiskers", "caps", "medians"]:
            for artist in box[element]:
                artist.set_color(color if element != "medians" else "black")
                artist.set_linewidth(1.0)
        ax.scatter(np.full(len(vals), pos) + np.linspace(-0.045, 0.045, len(vals)), vals, s=11, color=color, alpha=0.62, edgecolor="white", linewidth=0.25)
    ax.set_xticks([0.17, 1.17], ["90% peak", "95% peak"])
    ax.set_ylabel("Min edges")
    ax.set_title("Edge thresholds", pad=2.0)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    base.style_axis(ax, grid=True)
    handles = [
        Line2D([0], [0], color=base.COLORS["teacher"], lw=2, label="Teacher"),
        Line2D([0], [0], color=base.COLORS["student"], lw=2, label="Student"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", handlelength=1.0, fontsize=5.8)
    panel_label(ax, "b", x=-0.14, y=1.13)


def consensus_edge_matrix(edges_t: pd.DataFrame, edges_s: pd.DataFrame, max_nodes: int = 24) -> tuple[np.ndarray, list[str]]:
    combined = pd.concat([edges_t.assign(source_set="teacher"), edges_s.assign(source_set="student")], ignore_index=True)
    combined["score"] = combined["freq"].astype(float) * combined["gate_mean"].astype(float)
    top_edges = combined.sort_values("score", ascending=False).head(42)
    nodes = []
    protein_order = disc.load_protein_order()
    present = set(top_edges["source"]) | set(top_edges["target"])
    for protein in protein_order:
        if protein in present:
            nodes.append(protein)
    nodes = nodes[:max_nodes]
    node_to_i = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)))
    for _, row in top_edges.iterrows():
        if row["source"] not in node_to_i or row["target"] not in node_to_i:
            continue
        sign = 1.0 if row["sign"] == "activation" else -1.0
        value = sign * float(row["score"])
        i = node_to_i[row["source"]]
        j = node_to_i[row["target"]]
        if abs(value) > abs(matrix[i, j]):
            matrix[i, j] = value
    return matrix, nodes


def plot_consensus_matrix(fig: plt.Figure, ax: plt.Axes, edges_t: pd.DataFrame, edges_s: pd.DataFrame) -> None:
    matrix, nodes = consensus_edge_matrix(edges_t, edges_s)
    vmax = max(0.05, float(np.nanmax(np.abs(matrix))))
    img = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title("Consensus edge matrix", pad=2.0)
    labels = [base.short_name(node).replace("\n", " ") for node in nodes]
    ax.set_xticks(np.arange(len(nodes)), labels, rotation=90, fontsize=4.5)
    ax.set_yticks(np.arange(len(nodes)), labels, fontsize=4.5)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.tick_params(width=0.7, length=1.8, pad=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.016)
    cbar.ax.set_title("signed\nscore", fontsize=5.0, fontweight="bold", pad=2.0)
    style_cbar(cbar, label_size=5.3)
    panel_label(ax, "c", x=-0.14, y=1.13)


def plot_sensitivity_beeswarm(ax: plt.Axes, edges_s: pd.DataFrame) -> None:
    df_scores, _ = disc.collect_condition_sensitivity(edges_s, n_edges=10)
    edge_order = (
        df_scores.groupby("edge")["score"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=True)
        .index.tolist()
    )
    edge_to_y = {edge: i for i, edge in enumerate(edge_order)}
    rng = np.random.default_rng(11)
    y = np.array([edge_to_y[e] for e in df_scores["edge"]], dtype=float)
    y += rng.normal(0, 0.075, len(y))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("gate_low_high_supp", ["#1687D9", "#7E57C2", "#D81B60"])
    scatter = ax.scatter(
        df_scores["impact"],
        y,
        c=df_scores["gate"],
        cmap=cmap,
        vmin=0.0,
        vmax=max(0.6, float(df_scores["gate"].max())),
        s=17,
        alpha=0.92,
        linewidths=0,
        zorder=3,
    )
    ax.axvline(0, color="#7A7A7A", lw=1.0, alpha=0.85)
    labels = [
        edge.replace("->", " -> ")
        .replace("AKT_pS473", "AKT S473")
        .replace("AKT_pT308", "AKT T308")
        .replace("MAPK_pT202_Y204", "ERK")
        .replace("MEK1_pS217_S221", "MEK1")
        .replace("GSK3-alpha-beta_pS21_S9", "GSK3ab")
        .replace("GSK3-alpha-beta_pS9", "GSK3ab")
        .replace("PRAS40_pT246", "PRAS40")
        .replace("p70S6K_pT389", "p70S6K")
        .replace("PKC-alpha_pS657", "PKCa")
        .replace("Src_pY527", "Src Y527")
        .replace("4EBP1_pS65", "4EBP1")
        .replace("FOXO3a_pS318_S321", "FOXO3a")
        for edge in edge_order
    ]
    ax.set_yticks(np.arange(len(edge_order)), labels)
    ax.tick_params(axis="y", labelsize=4.9)
    ax.set_xlabel("Sensitivity impact")
    ax.set_title("Student sensitivity landscape", pad=2.0)
    x_abs = max(abs(float(df_scores["impact"].min())), abs(float(df_scores["impact"].max())))
    ax.set_xlim(-x_abs * 1.16, x_abs * 1.16)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    base.style_axis(ax, grid=True)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.018)
    cbar.set_ticks([0.0, max(0.6, float(df_scores["gate"].max()))])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Gate value", labelpad=1.0)
    style_cbar(cbar, label_size=5.2)
    panel_label(ax, "d", x=-0.14, y=1.13)


def top_network_edges(edges_t: pd.DataFrame, edges_s: pd.DataFrame, n_edges: int = 28) -> list[tuple]:
    edges_t = edges_t.copy()
    edges_s = edges_s.copy()
    edges_t["cls"] = "Teacher-only"
    edges_s["cls"] = "Student-only"
    common = set(zip(edges_t["source"], edges_t["target"])) & set(zip(edges_s["source"], edges_s["target"]))
    rows = []
    for df, default_cls in [(edges_t, "Teacher-only"), (edges_s, "Student-only")]:
        for _, row in df.iterrows():
            edge = (row["source"], row["target"])
            cls = "Common" if edge in common else default_cls
            score = float(row["freq"]) * float(row["gate_mean"])
            rows.append((edge, cls, row["sign"], score))
    rows = sorted(rows, key=lambda x: x[3], reverse=True)
    dedup = []
    seen = set()
    for edge, cls, sign, score in rows:
        if edge in seen:
            continue
        seen.add(edge)
        dedup.append((edge, cls, sign, score))
        if len(dedup) >= n_edges:
            break
    return dedup


def draw_rect_node(ax: plt.Axes, pos: np.ndarray, node: str) -> None:
    group = disc.pathway_of(node)
    color = disc.PATHWAY_COLORS.get(group, "#A7ADB8")
    label = base.short_name(node)
    width, height, fontsize = disc.node_box_dimensions(label, group)
    label_lines = label.split("\n")
    max_len = max(len(line) for line in label_lines)
    width = max(width * 1.12, 0.135 + 0.0145 * max_len)
    height = max(height * 1.08, 0.053 + 0.046 * len(label_lines))
    fontsize = min(fontsize, 5.0 if max_len >= 6 else 5.25)
    patch = FancyBboxPatch(
        (pos[0] - width / 2, pos[1] - height / 2),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="white",
        edgecolor=color,
        linewidth=1.15,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(pos[0], pos[1], label, ha="center", va="center", fontsize=fontsize, fontweight="bold", linespacing=0.86, zorder=4)


def plot_compact_network(ax: plt.Axes, edges_t: pd.DataFrame, edges_s: pd.DataFrame) -> None:
    edge_specs = top_network_edges(edges_t, edges_s, n_edges=28)
    nodes = []
    protein_order = disc.load_protein_order()
    present = {n for edge, *_ in edge_specs for n in edge}
    for protein in protein_order:
        if protein in present:
            nodes.append(protein)
    pos = disc.pathway_layered_layout(nodes)
    disc.draw_pathway_boxes(ax)
    scores = np.array([spec[3] for spec in edge_specs], dtype=float)
    s_min, s_max = float(scores.min()), float(scores.max())
    for idx, (edge, cls, sign, score) in enumerate(sorted(edge_specs, key=lambda x: x[3])):
        if edge[0] not in pos or edge[1] not in pos:
            continue
        norm = (score - s_min) / max(s_max - s_min, EPS)
        lw = 0.85 + 2.25 * (norm**1.1)
        alpha = 0.50 + 0.42 * norm
        offset = ((idx % 5) - 2) * 0.025
        disc.draw_routed_edge(
            ax,
            pos[edge[0]],
            pos[edge[1]],
            disc.pathway_of(edge[0]),
            disc.pathway_of(edge[1]),
            EDGE_CLASS_COLORS[cls],
            lw,
            "solid" if sign == "activation" else (0, (2.2, 1.4)),
            alpha,
            offset,
        )
    for node in nodes:
        draw_rect_node(ax, pos[node], node)
    ax.set_xlim(-1.96, 1.86)
    ax.set_ylim(-1.03, 1.08)
    ax.axis("off")
    ax.set_title("Top consensus pathway network", pad=2.0)
    handles = [
        Line2D([0], [0], color=EDGE_CLASS_COLORS["Common"], lw=2.1, label="Common"),
        Line2D([0], [0], color=EDGE_CLASS_COLORS["Teacher-only"], lw=2.1, label="Teacher-only"),
        Line2D([0], [0], color=EDGE_CLASS_COLORS["Student-only"], lw=2.1, label="Student-only"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, handlelength=1.1, fontsize=5.4)
    panel_label(ax, "e", x=-0.05, y=1.06)


def plot_hill_equation(ax: plt.Axes) -> None:
    disc.plot_equation_discovery(ax)
    panel_label(ax, "f", x=-0.09, y=1.07)


def build_discovery_robustness() -> None:
    edges_t = pd.read_csv(disc.CONSENSUS_TEACHER)
    edges_s = pd.read_csv(disc.CONSENSUS_STUDENT)
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.28, 1.55],
        left=0.075,
        right=0.970,
        bottom=0.060,
        top=0.930,
        hspace=0.44,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])
    ax_e = fig.add_subplot(outer[2, 0])
    ax_f = fig.add_subplot(outer[2, 1])
    plot_edge_economy(ax_a)
    plot_edge_budget_summary(ax_b)
    plot_consensus_matrix(fig, ax_c, edges_t, edges_s)
    plot_sensitivity_beeswarm(ax_d, edges_s)
    plot_compact_network(ax_e, edges_t, edges_s)
    plot_hill_equation(ax_f)
    fig.suptitle("Supplementary HPN-DREAM network and equation discovery robustness", fontsize=12.0, fontweight="bold", y=0.985)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"hpn_dream_supp_discovery_robustness.{ext}", facecolor="white")
    fig.savefig(PANEL_DIR / "hpn_dream_supp_discovery_robustness_preview.png", facecolor="white", dpi=300)
    plt.close(fig)


def build_figures() -> None:
    warnings.filterwarnings("ignore")
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading high-resolution predictions and checkpoints...", flush=True)
    high, teacher_predictions, student_predictions, teacher_bundle, student_bundle = load_predictions()
    print("Loading experimental MCF7 data...", flush=True)
    x_exp, y_exp, info_exp = load_experimental()

    print("Building ligand-response trajectory atlas...", flush=True)
    build_trajectory_atlas(
        high,
        teacher_predictions,
        student_predictions,
        x_exp,
        y_exp,
        info_exp,
        LIGAND_TRAJECTORY_CONDITIONS,
        CORE_TRAJECTORY_PROTEINS,
        "Supplementary HPN-DREAM ligand-response trajectory atlas",
        "hpn_dream_supp_ligand_trajectory_atlas",
    )

    print("Building inhibitor-response trajectory atlas...", flush=True)
    build_trajectory_atlas(
        high,
        teacher_predictions,
        student_predictions,
        x_exp,
        y_exp,
        info_exp,
        PERTURBATION_TRAJECTORY_CONDITIONS,
        DOWNSTREAM_TRAJECTORY_PROTEINS,
        "Supplementary HPN-DREAM inhibitor-response trajectory atlas",
        "hpn_dream_supp_inhibitor_trajectory_atlas",
    )
    print(f"Saved trajectory-focused supplementary figures to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    build_figures()
