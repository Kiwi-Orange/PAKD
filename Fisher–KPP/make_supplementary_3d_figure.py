"""Build a Nature-style supplementary 3D figure for Fisher-KPP.

The figure is redrawn from data/checkpoints and kept separate from the main
A4 figure. It combines HMM phase separation, teacher validation, and PAKD
student alignment in one compact six-panel supplementary figure.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fisher_kpp_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  Registers 3D projection.

from make_nature_figure import (
    COLORS,
    GAMMA_DATA,
    OUT_DIR,
    STUDENT_MODEL,
    TEACHER_MODEL,
    TIME_POINTS,
    compute_trajectories,
    configure_style,
    load_checkpoint_model,
)


OUTPUT_BASENAME = "fisher_kpp_3d_supplement"
PREVIEW_PATH = OUT_DIR / "panels" / f"{OUTPUT_BASENAME}_preview.png"

N_TIME_SURF = 190
N_SPACE_SURF = 82
CAMERA = {"elev": 26, "azim": -62}
LOG_TIME_TICKS = [-4, -2, 0, 1]
LOG_TIME_TICKLABELS = [r"$10^{-4}$", r"$10^{-2}$", r"$10^0$", r"$10^1$"]


def make_error_cmap() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "error_light_3d",
        ["#fffdf7", "#fee8c8", "#fdbb84", "#f46d43", "#a50026"],
    )


def surface_indices(n_time: int, n_space: int) -> tuple[np.ndarray, np.ndarray]:
    t_idx = np.unique(np.linspace(0, n_time - 1, N_TIME_SURF, dtype=int))
    x_idx = np.unique(np.linspace(0, n_space - 1, N_SPACE_SURF, dtype=int))
    return t_idx, x_idx


def prepare_surface_grid(trajectories: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_grid = trajectories["x"]
    t_idx, x_idx = surface_indices(len(TIME_POINTS), len(x_grid))
    x_s = x_grid[x_idx]
    log_t_s = np.log10(TIME_POINTS[t_idx])
    X, Y = np.meshgrid(x_s, log_t_s)
    return X, Y, t_idx, x_idx


def style_3d_axis(
    ax: plt.Axes,
    zlim: tuple[float, float],
    zticks: list[float],
    show_y_label: bool,
    show_z_label: bool,
    zlabel: str,
) -> None:
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#D7DDE6")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = "#D8DDE6"
        axis._axinfo["grid"]["linestyle"] = "--"
        axis._axinfo["grid"]["linewidth"] = 0.35

    ax.set_xlabel(r"$x$", labelpad=-1.0)
    ax.set_ylabel(r"$t$", labelpad=-1.0 if show_y_label else -4.0)
    ax.set_zlabel(zlabel if show_z_label else "", labelpad=-1.5)
    if not show_y_label:
        ax.set_ylabel("")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(LOG_TIME_TICKS[0], LOG_TIME_TICKS[-1])
    ax.set_zlim(*zlim)
    ax.set_xticks([0.0, 0.5, 1.0])
    if show_y_label:
        ax.set_xticklabels(["0.0", "0.5", ""])
    else:
        ax.set_xticklabels(["0.0", "", "1.0"])
    ax.set_yticks(LOG_TIME_TICKS)
    ax.set_yticklabels(LOG_TIME_TICKLABELS)
    ax.set_zticks(zticks)
    if not show_y_label:
        ax.set_yticklabels([])
    if not show_z_label:
        ax.set_zticklabels([])
    ax.tick_params(axis="both", which="major", labelsize=6.1, width=0.8, length=2.2, pad=-1.0)
    ax.zaxis.set_tick_params(labelsize=6.1, width=0.8, length=2.2, pad=-1.0)
    ax.view_init(**CAMERA)
    ax.set_box_aspect((1.28, 1.10, 0.72))



def plot_surface_panel(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cmap,
    norm,
    zlim: tuple[float, float],
    zticks: list[float],
    zlabel: str,
    show_y_label: bool,
    show_z_label: bool,
) -> mpl.cm.ScalarMappable:
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        linewidth=0.0,
        antialiased=True,
        alpha=0.97,
        shade=True,
    )
    surf.set_rasterized(True)
    style_3d_axis(ax, zlim, zticks, show_y_label, show_z_label, zlabel)
    return surf


def hmm_transition() -> tuple[float, float, dict[str, int]]:
    data = np.load(GAMMA_DATA)
    time = np.maximum(data[:, 0], TIME_POINTS[0])
    gammas = data[:, -2:]
    raw_phase = np.argmax(gammas, axis=1)
    medians = {phase: np.median(time[raw_phase == phase]) for phase in np.unique(raw_phase)}
    fast_phase = min(medians, key=medians.get)
    slow_phase = max(medians, key=medians.get)
    fast_gamma = gammas[:, fast_phase]
    slow_gamma = gammas[:, slow_phase]
    switch_idx = np.where(np.diff((slow_gamma > fast_gamma).astype(int)) != 0)[0]
    t_hmm = float(time[switch_idx[0] + 1]) if len(switch_idx) else float(np.median(time))
    return t_hmm, float(np.log10(t_hmm)), {"fast": fast_phase, "slow": slow_phase}


def add_phase_floor(ax: plt.Axes, log_t_hmm: float) -> None:
    z_floor = -0.075
    x0, x1 = 0.0, 1.0
    y_min, y_max = LOG_TIME_TICKS[0], LOG_TIME_TICKS[-1]

    for y0, y1, color in [
        (y_min, log_t_hmm, COLORS["fast"]),
        (log_t_hmm, y_max, COLORS["slow"]),
    ]:
        Xp, Yp = np.meshgrid([x0, x1], [y0, y1])
        Zp = np.full_like(Xp, z_floor)
        band = ax.plot_surface(Xp, Yp, Zp, color=color, alpha=0.20, edgecolor="none", shade=False)
        band.set_rasterized(True)

    Xp, Zp = np.meshgrid([x0, x1], [z_floor, 1.02])
    Yp = np.full_like(Xp, log_t_hmm)
    plane = ax.plot_surface(Xp, Yp, Zp, color="#111111", alpha=0.10, edgecolor="none", shade=False)
    plane.set_rasterized(True)
    ax.plot([x0, x1], [log_t_hmm, log_t_hmm], [z_floor, z_floor], color="#111111", lw=1.0, ls=":")
    ax.plot([x0, x1], [log_t_hmm, log_t_hmm], [1.02, 1.02], color="#111111", lw=1.1, ls=":")
    ax.plot([x0, x0], [log_t_hmm, log_t_hmm], [z_floor, 1.02], color="#111111", lw=0.9, ls=":")
    ax.plot([x1, x1], [log_t_hmm, log_t_hmm], [z_floor, 1.02], color="#111111", lw=0.9, ls=":")

    handles = [
        Line2D([0], [0], color=COLORS["fast"], lw=3.6, label="Fast"),
        Line2D([0], [0], color=COLORS["slow"], lw=3.6, label="Slow"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.93),
        frameon=False,
        fontsize=5.9,
        handlelength=1.0,
        borderpad=0.0,
        labelspacing=0.1,
    )


def format_colorbar(
    cb: mpl.colorbar.Colorbar,
    title: str,
    ticks: list[float] | None = None,
    tick_side: str = "right",
) -> None:
    cb.ax.set_title(title, fontsize=7.0, fontweight="bold", pad=3.0)
    cb.ax.yaxis.set_ticks_position(tick_side)
    cb.ax.tick_params(labelsize=6.2, width=0.8, length=2.2, pad=1.0)
    cb.outline.set_linewidth(0.9)
    if ticks is not None:
        cb.set_ticks(ticks)


def build_figure() -> None:
    warnings.filterwarnings("ignore")
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints...", flush=True)
    teacher_bundle = load_checkpoint_model(TEACHER_MODEL, is_student=False)
    student_bundle = load_checkpoint_model(STUDENT_MODEL, is_student=True)

    print("Solving BDF and evaluating teacher/student...", flush=True)
    trajectories = compute_trajectories(teacher_bundle, student_bundle)

    x_grid = trajectories["x"]
    truth = trajectories["truth"]
    teacher = trajectories["teacher"]
    student = trajectories["student"]
    teacher_error = np.abs(teacher - truth)
    student_error = np.abs(student - truth)
    combined_error = np.concatenate([teacher_error.ravel(), student_error.ravel()])
    err_vmax = max(0.015, float(np.nanpercentile(combined_error, 98.8)))
    teacher_error = np.clip(teacher_error, 0.0, err_vmax)
    student_error = np.clip(student_error, 0.0, err_vmax)

    u_cmap = mpl.colormaps["viridis"]
    u_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    e_cmap = make_error_cmap()
    e_norm = mpl.colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=err_vmax)

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.045],
        wspace=0.08,
        hspace=0.18,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    cax_u = fig.add_subplot(gs[0, 3])
    cax_e = fig.add_subplot(gs[1, 3])

    panels = [
        (truth, "a", "HMM phase surface", u_cmap, u_norm),
        (truth, "b", "BDF ground truth", u_cmap, u_norm),
        (teacher, "c", "Teacher surrogate", u_cmap, u_norm),
        (teacher_error, "d", "Teacher abs. error", e_cmap, e_norm),
        (student, "e", "PAKD student", u_cmap, u_norm),
        (student_error, "f", "Student abs. error", e_cmap, e_norm),
    ]

    for i, (values, label, title, cmap, norm) in enumerate(panels):
        ax = axes[i]
        im = ax.pcolormesh(
            x_grid,
            TIME_POINTS,
            values,
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        ax.set_yscale("log")
        ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
        ax.set_ylim(TIME_POINTS[0], TIME_POINTS[-1])
        ax.set_xticks([0.0, 0.5, 1.0])
        if i % 3 == 0:
            ax.set_xticklabels(["0.0", "0.5", ""])
        else:
            ax.set_xticklabels(["", "0.5", "1.0"])
        if i % 3 == 0:
            ax.set_ylabel(r"$t$", labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)
        if i >= 3:
            ax.set_xlabel(r"$x$", labelpad=1.5)
        else:
            ax.tick_params(labelbottom=False)
        ax.tick_params(axis="both", which="major", width=0.9, length=3.0, pad=1.5)
        ax.tick_params(axis="both", which="minor", width=0.6, length=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.05)
        ax.grid(True, which="major", color=COLORS["grid"], lw=0.45, alpha=0.65)
        text_color = "white" if cmap == u_cmap else "black"
        ax.text(
            0.04,
            0.96,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color=text_color,
            bbox=dict(facecolor="black" if text_color == "white" else "white", edgecolor="none", alpha=0.55, pad=0.3),
        )
        # a 图叠加 HMM phase 切换线
        if i == 0:
            data = np.load(GAMMA_DATA)
            time = np.maximum(data[:, 0], TIME_POINTS[0])
            gammas = data[:, -2:]
            raw_phase = np.argmax(gammas, axis=1)
            medians = {phase: np.median(time[raw_phase == phase]) for phase in np.unique(raw_phase)}
            fast_phase = min(medians, key=medians.get)
            slow_phase = max(medians, key=medians.get)
            fast_gamma = gammas[:, fast_phase]
            slow_gamma = gammas[:, slow_phase]
            switch_idx = np.where(np.diff((slow_gamma > fast_gamma).astype(int)) != 0)[0]
            if len(switch_idx):
                t_hmm = float(time[switch_idx[0] + 1])
                ax.axvline(t_hmm, color="white", lw=1.5, ls="--", alpha=0.9)
                ax.text(
                    t_hmm * 1.06,
                    TIME_POINTS[-1] * 0.05,
                    r"$t_{\mathrm{HMM}}$",
                    color="white",
                    fontsize=7.0,
                    fontweight="bold",
                    rotation=90,
                    va="center",
                    ha="left",
                )
            # 底部 phase 颜色带
            ax.axvspan(TIME_POINTS[0], t_hmm, ymin=0.0, ymax=0.06, color=COLORS["fast"], alpha=0.6, transform=ax.get_xaxis_transform())
            ax.axvspan(t_hmm, TIME_POINTS[-1], ymin=0.0, ymax=0.06, color=COLORS["slow"], alpha=0.6, transform=ax.get_xaxis_transform())

    cb_u = fig.colorbar(mpl.cm.ScalarMappable(norm=u_norm, cmap=u_cmap), cax=cax_u)
    format_colorbar(cb_u, r"$u$", ticks=[0.0, 0.5, 1.0], tick_side="right")
    cb_e = fig.colorbar(mpl.cm.ScalarMappable(norm=e_norm, cmap=e_cmap), cax=cax_e)
    format_colorbar(cb_e, r"$|e|$", ticks=[0.0, err_vmax / 2.0, err_vmax], tick_side="right")
    cb_e.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))

    # 统一放置 panel label 和标题
    panel_info = [
        (gs[0, 0], "a", "HMM phase surface"),
        (gs[0, 1], "b", "BDF ground truth"),
        (gs[0, 2], "c", "Teacher surrogate"),
        (gs[1, 0], "d", "Teacher abs. error"),
        (gs[1, 1], "e", "PAKD student"),
        (gs[1, 2], "f", "Student abs. error"),
    ]
    for cell, label, title in panel_info:
        bbox = cell.get_position(fig)
        y = bbox.y1 + 0.005
        x_label = bbox.x0 - 0.022
        x_title = bbox.x0
        fig.text(x_label, y, label, fontsize=13, fontweight="bold", ha="left", va="bottom")
        fig.text(x_title, y, title, fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{OUTPUT_BASENAME}.{ext}", facecolor="white")
    fig.savefig(PREVIEW_PATH, facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved supplementary 3D figure to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    build_figure()
