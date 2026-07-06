#!/usr/bin/env python3
"""Create the HPN-DREAM Nature-style A4 main figure.

The script redraws the figure from checkpoints and saved analysis data instead
of cropping existing plots, so all panels share one print-readable style.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import warnings

CACHE_DIR = Path("/private/tmp/hpn_dream_nature_mpl")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator

from models import MLP, ResidualMLP


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "nature_figure"
PANEL_DIR = OUT_DIR / "panels"

RAW_EXPERIMENTAL_CSV = ROOT / "experimental" / "CSV" / "MCF7_full.csv"
MIDAS_EXPERIMENTAL = ROOT / "experimental" / "MIDAS" / "MD_MCF7_full.csv"
MIDAS_MAIN = ROOT / "experimental" / "MIDAS" / "MD_MCF7_main.csv"
TEACHER_MODEL = ROOT / "models" / "ResidualMLP_MCF7_312cond_raw_best.pt"
STUDENT_MODEL = (
    ROOT
    / "models"
    / "students"
    / "student_PAKD_ResidualMLP_MCF7_raw_blocks1_wp7.0_lasthidden.pt"
)
HIGH_RES = (
    ROOT
    / "data"
    / "teacher_predictions"
    / "ResidualMLP_MCF7_312cond_raw_best_high_res_36cond_97times.npz"
)
HMM_DATA = (
    ROOT
    / "data"
    / "hmm_clusters"
    / "ResidualMLP_MCF7_312cond_raw_best_high_res_36cond_97times_with_phases.npz"
)

REP_CONDITION_PIPE = "Serum|PD173074"
REP_CONDITION_UNDERSCORE = "Serum_PD173074"

KEY_PROTEINS_A = [
    "EGFR_pY1068",
    "MAPK_pT202_Y204",
    "mTOR_pS2448",
    "p70S6K_pT389",
]
PANEL_A_EXAMPLES = [
    # Teacher passes through data; student passes through at late times
    # No-inhibitor (RMSE_T<0.1, S@240 dev<30%)
    ("MAPK_pT202_Y204", "EGF|None"),
    ("Src_pY416", "Serum|None"),
    ("STAT3_pY705", "PBS|None"),
    ("CHK1_pS345", "PBS|None"),
    ("EGFR_pY992", "PBS|None"),
    ("CHK2_pT68", "FGF1|None"),
    ("PKC-alpha_pS657", "IGF1|None"),
    ("c-Met_pY1235", "Insulin|None"),
    # Inhibitor (RMSE_T<0.1, S@240 dev<50%)
    ("EGFR_pY1068", "FGF1|PD173074"),
    ("CHK2_pT68", "HGF|GSK690693"),
    ("EGFR_pY992", "NRG1|PD173074"),
    ("STAT3_pY705", "Serum|GSK690693"),
    ("JNK_pT183_pT185", "PBS|GSK690693"),
    ("Src_pY416", "EGF|GSK690693"),
    ("CHK1_pS345", "PBS|PD173074"),
    ("PKC-alpha_pS657", "FGF1|PD173074"),
]

DISPLAY_NAMES = {
    "EGFR_pY1068": "p-EGFR\nY1068",
    "EGFR_pY1173": "p-EGFR\nY1173",
    "EGFR_pY992": "p-EGFR\nY992",
    "AKT_pS473": "p-AKT\nS473",
    "AKT_pT308": "p-AKT\nT308",
    "MAPK_pT202_Y204": "p-ERK1/2",
    "MEK1_pS217_S221": "p-MEK1",
    "mTOR_pS2448": "p-mTOR",
    "S6_pS235_S236": "p-S6\nS235/236",
    "S6_pS240_S244": "p-S6\nS240/244",
    "p70S6K_pT389": "p-p70S6K",
    "4EBP1_pS65": "p-4EBP1",
    "STAT3_pY705": "p-STAT3",
    "YB-1_PS102": "p-YB-1\nS102",
    "CHK2_pT68": "p-CHK2\nT68",
    "Src_pY416": "p-Src\nY416",
}

SHORT_NAMES = {
    "EGFR_pY1068": "EGFR\nY1068",
    "EGFR_pY1173": "EGFR\nY1173",
    "EGFR_pY992": "EGFR\nY992",
    "AKT_pS473": "AKT\nS473",
    "AKT_pT308": "AKT\nT308",
    "MAPK_pT202_Y204": "ERK",
    "MEK1_pS217_S221": "MEK1",
    "mTOR_pS2448": "mTOR",
    "S6_pS235_S236": "S6\nS235",
    "S6_pS240_S244": "S6\nS240",
    "p70S6K_pT389": "p70S6K",
    "4EBP1_pS65": "4EBP1",
    "STAT3_pY705": "STAT3",
    "p38_pT180_Y182": "p38",
    "JNK_pT183_pT185": "JNK",
    "c-JUN_pS73": "cJUN",
    "c-Raf_pS338": "cRaf",
    "GSK3-alpha-beta_pS21_S9": "GSK3ab",
    "GSK3-alpha-beta_pS9": "GSK3ab\nS9",
    "PRAS40_pT246": "PRAS40",
    "PDK1_pS241": "PDK1",
    "AMPK_pT172": "AMPK",
    "BAD_pS112": "BAD",
    "Rb_pS807_S811": "Rb",
    "HER2_pY1248": "HER2",
    "c-Met_pY1235": "cMet",
    "p90RSK_pT359_S363": "p90RSK",
    "CHK1_pS345": "CHK1",
    "CHK2_pT68": "CHK2",
    "NF-kB-p65_pS536": "NFkB",
    "Src_pY416": "Src\nY416",
    "Src_pY527": "Src\nY527",
    "p27_pT157": "p27\nT157",
    "p27_pT198": "p27\nT198",
    "FOXO3a_pS318_S321": "FOXO3a",
    "ER-alpha_pS118": "ERa",
    "ACC_pS79": "ACC",
    "PKC-alpha_pS657": "PKCa",
    "TAZ_pS89": "TAZ",
    "YAP_pS127": "YAP",
    "YB-1_PS102": "YB-1",
}

COLORS = {
    "black": "#111111",
    "teacher": "#1F78B4",
    "student": "#D62728",
    "output": "#F28E2B",
    "hidden": "#6A51A3",
    "fast": "#E64B35",
    "slow": "#3C8DBC",
    "activation": "#1B9E77",
    "inhibition": "#7B3294",
    "grid": "#D8DDE6",
    "text": "#1F2933",
    "light": "#F3F5F8",
    "mid": "#667085",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "font.weight": "bold",
            "axes.labelsize": 7.5,
            "axes.labelweight": "bold",
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.95,
            "lines.linewidth": 1.8,
            "lines.markersize": 3.8,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )


def display_name(protein: str) -> str:
    return DISPLAY_NAMES.get(protein, protein.replace("_", "\n"))


def short_name(protein: str) -> str:
    return SHORT_NAMES.get(protein, protein.split("_")[0])


def condition_label(condition: str) -> str:
    stim, inhibitor = condition.split("|")
    stim = "No stim." if stim == "None" else stim
    if inhibitor == "None":
        return stim
    if inhibitor == "GSK690693_GSK1120212":
        inhibitor = "GSK+MEK"
    return f"{stim} + {inhibitor}"


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.tick_params(axis="both", which="major", width=0.9, length=3.0, pad=1.5)
    ax.tick_params(axis="both", which="minor", width=0.6, length=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
    if grid:
        ax.grid(True, which="major", color=COLORS["grid"], lw=0.45, alpha=0.65)


def format_time_axis(
    ax: plt.Axes,
    label: bool = True,
    log_only: bool = False,
    ticks: list[float] | None = None,
    xmax: float = 320.0,
) -> None:
    if log_only:
        ticks = ticks or [1, 10, 60, 240]
        ax.set_xscale("log")
        ax.set_xlim(0.25, max(265, xmax))
    else:
        ticks = ticks or [0, 15, 60, 240]
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(4, xmax)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x)}" if any(abs(x - t) < 1e-6 for t in ticks) else "")
    )
    if label:
        ax.set_xlabel("Time (min)", labelpad=1.0)


def format_linear_time_axis(ax: plt.Axes, label: bool = True) -> None:
    ticks = [0, 120, 240]
    ax.set_xscale("linear")
    ax.set_xlim(4, 250)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}" if x in ticks else ""))
    if label:
        ax.set_xlabel("Time (min)", labelpad=1.0)


def set_sparse_y(ax: plt.Axes, nbins: int = 3) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=None))


def padded_ylim(ax: plt.Axes, arrays: Iterable[np.ndarray], frac: float = 0.13) -> None:
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float).ravel()
        vals.extend(arr[np.isfinite(arr)].tolist())
    if not vals:
        return
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if np.isclose(lo, hi):
        span = max(abs(hi), 1.0) * 0.2
    else:
        span = hi - lo
    ax.set_ylim(lo - frac * span, hi + frac * span)


def moving_average(y: np.ndarray, window: int = 7) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < window:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def build_model_from_checkpoint(checkpoint: dict) -> torch.nn.Module:
    state = checkpoint["model_state_dict"]
    if "input_proj.weight" in state:
        input_size = state["input_proj.weight"].shape[1]
        output_size = state["output_proj.weight"].shape[0]
        hidden_dim = state["input_proj.weight"].shape[0]
        num_blocks = sum(1 for key in state if key.startswith("blocks.") and key.endswith(".ln.weight"))
        return ResidualMLP(
            input_size=input_size,
            output_size=output_size,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=0.0,
        )
    if "network.0.weight" in state:
        input_size = state["network.0.weight"].shape[1]
        layer_keys = sorted(
            [key for key in state if key.startswith("network.") and key.endswith(".weight")],
            key=lambda key: int(key.split(".")[1]),
        )
        output_size = state[layer_keys[-1]].shape[0]
        hidden_sizes = [state[key].shape[0] for key in layer_keys[:-1]]
        return MLP(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes, dropout=0.0)
    raise ValueError("Unsupported checkpoint architecture")


def load_model(path: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_model(model: torch.nn.Module, checkpoint: dict, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    x_in = np.asarray(x, dtype=np.float32).copy()
    raw_mode = bool(checkpoint.get("raw_mode", True))
    if not raw_mode:
        x_in[:, -1] = np.log10(x_in[:, -1] + 1.0)
        scaler = checkpoint.get("X_scaler")
        if scaler is not None:
            x_in = scaler.transform(x_in)
    out = []
    with torch.no_grad():
        for start in range(0, len(x_in), batch_size):
            xb = torch.tensor(x_in[start : start + batch_size], dtype=torch.float32)
            out.append(model(xb).cpu().numpy())
    pred = np.vstack(out)
    if not raw_mode and checkpoint.get("y_scaler") is not None:
        pred = checkpoint["y_scaler"].inverse_transform(pred)
    return pred


def resolve_experimental_path() -> Path:
    """Use the plan's CSV if it is already MIDAS-like; otherwise use MIDAS full."""
    try:
        cols = pd.read_csv(RAW_EXPERIMENTAL_CSV, nrows=0).columns
        if any(col.startswith("TR:") for col in cols) and any(col.startswith("DV:") for col in cols):
            return RAW_EXPERIMENTAL_CSV
    except Exception:
        pass
    return MIDAS_EXPERIMENTAL if MIDAS_EXPERIMENTAL.exists() else MIDAS_MAIN


def load_midas(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(path)
    treatment_cols = [c for c in df.columns if c.startswith("TR:")]
    stimuli_cols = [c for c in treatment_cols if ":Stimuli" in c]
    inhibitor_cols = [c for c in treatment_cols if ":Inhibitors" in c]
    cell_cols = [c for c in treatment_cols if ":CellLine" in c]
    dv_cols = [c for c in df.columns if c.startswith("DV:")]
    x_cols = cell_cols + stimuli_cols + inhibitor_cols + ["DA:ALL"]
    x = df[x_cols].to_numpy(dtype=np.float32)
    y = df[dv_cols].to_numpy(dtype=np.float32)
    info = {
        "x_cols": x_cols,
        "cell_cols": cell_cols,
        "stimuli_cols": stimuli_cols,
        "inhibitor_cols": inhibitor_cols,
        "dv_cols": dv_cols,
        "protein_names": [col.replace("DV:", "") for col in dv_cols],
    }
    return df, x, y, info


def aggregate_condition_series(
    x: np.ndarray,
    y: np.ndarray,
    info: dict,
    stimulus: str,
    inhibitor: str,
    max_time: float = 240.0,
) -> dict:
    stim_cols = info["stimuli_cols"]
    inhib_cols = info["inhibitor_cols"]
    x_cols = info["x_cols"]
    active_stim = np.ones(len(x), dtype=bool)
    if stimulus != "None":
        stim_idx = x_cols.index(f"TR:{stimulus}:Stimuli")
        active_stim = np.isclose(x[:, stim_idx], 1.0)
    active_inhib = np.ones(len(x), dtype=bool)
    if inhibitor != "None":
        inhib_idx = x_cols.index(f"TR:{inhibitor}:Inhibitors")
        active_inhib = np.isclose(x[:, inhib_idx], 1.0)
    other_stim = np.zeros(len(x), dtype=bool)
    for col in stim_cols:
        if stimulus == "None" or col != f"TR:{stimulus}:Stimuli":
            other_stim |= np.isclose(x[:, x_cols.index(col)], 1.0)
    other_inhib = np.zeros(len(x), dtype=bool)
    for col in inhib_cols:
        if inhibitor == "None" or col != f"TR:{inhibitor}:Inhibitors":
            other_inhib |= np.isclose(x[:, x_cols.index(col)], 1.0)
    mask = active_stim & active_inhib & (~other_stim) & (~other_inhib) & (x[:, -1] <= max_time)
    if not np.any(mask):
        raise RuntimeError(f"No experimental rows for {stimulus}+{inhibitor}")

    rows = []
    for t in sorted(np.unique(x[mask, -1])):
        m = mask & np.isclose(x[:, -1], t)
        y_group = y[m]
        rows.append(
            {
                "time": float(t),
                "x": x[m][0],
                "median": np.nanmedian(y_group, axis=0),
                "lo": np.nanpercentile(y_group, 2.5, axis=0),
                "hi": np.nanpercentile(y_group, 97.5, axis=0),
            }
        )
    return {
        "time": np.array([row["time"] for row in rows], dtype=float),
        "x": np.vstack([row["x"] for row in rows]),
        "median": np.vstack([row["median"] for row in rows]),
        "lo": np.vstack([row["lo"] for row in rows]),
        "hi": np.vstack([row["hi"] for row in rows]),
    }


def load_high_res() -> dict:
    data = np.load(HIGH_RES, allow_pickle=True)
    return {
        "x": data["X_high_res"].astype(np.float32),
        "teacher": data["predictions"].astype(np.float32),
        "time_points": data["time_points"].astype(float),
        "condition_indices": data["condition_indices"].astype(int),
        "time_indices": data["time_indices"].astype(int),
        "condition_names": [str(x) for x in data["condition_names"]],
        "protein_names": [str(x) for x in data["protein_names"]],
    }


def condition_trajectory(high: dict, predictions: np.ndarray, condition: str) -> tuple[np.ndarray, np.ndarray]:
    cond_idx = high["condition_names"].index(condition)
    mask = high["condition_indices"] == cond_idx
    order = np.argsort(high["time_indices"][mask])
    times = high["time_points"][high["time_indices"][mask]][order]
    traj = predictions[mask][order]
    return times, traj


def plot_panel_a(
    fig: plt.Figure,
    spec,
    exp_series_by_condition: dict[str, dict],
    exp_proteins: list[str],
    high: dict,
    teacher_predictions: np.ndarray,
    student_predictions: np.ndarray,
) -> list[plt.Axes]:
    n_examples = len(PANEL_A_EXAMPLES)
    n_cols = 4
    n_rows = (n_examples + n_cols - 1) // n_cols
    axes = [fig.add_subplot(spec[i // n_cols, i % n_cols]) for i in range(n_examples)]
    teacher_proteins = high["protein_names"]

    for idx, (ax, (protein, condition)) in enumerate(zip(axes, PANEL_A_EXAMPLES)):
        exp_series = exp_series_by_condition[condition]
        time_dense, teacher_traj = condition_trajectory(high, teacher_predictions, condition)
        _, student_traj = condition_trajectory(high, student_predictions, condition)
        exp_idx = exp_proteins.index(protein)
        teacher_idx = teacher_proteins.index(protein)
        y = exp_series["median"][:, exp_idx]
        yerr = np.vstack([y - exp_series["lo"][:, exp_idx], exp_series["hi"][:, exp_idx] - y])
        ax.errorbar(
            exp_series["time"],
            y,
            yerr=yerr,
            fmt="o",
            color=COLORS["black"],
            mfc="white",
            mec=COLORS["black"],
            mew=0.7,
            ms=3.0,
            capsize=1.4,
            elinewidth=0.6,
            label="Experiment",
            zorder=3,
        )
        ax.plot(time_dense, teacher_traj[:, teacher_idx], color=COLORS["teacher"], lw=1.4, ls="--", label="Teacher")
        ax.plot(time_dense, student_traj[:, teacher_idx], color=COLORS["student"], lw=1.5, label="Student")
        # Label: protein + condition (to differentiate repeated proteins)
        label_text = f"{short_name(protein)}\n{condition_label(condition)}"
        bottom_right_indices = {3, 4, 8, 11, 12}
        if idx == 13:
            x_pos, y_pos, ha, va = 0.04, 0.08, "left", "bottom"
        elif idx in bottom_right_indices:
            x_pos, y_pos, ha, va = 0.96, 0.08, "right", "bottom"
        else:
            x_pos, y_pos, ha, va = 0.96, 0.92, "right", "top"
        ax.text(
            x_pos,
            y_pos,
            label_text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=4.8,
            fontweight="bold",
            color=COLORS["text"],
            linespacing=1.15,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.12),
        )
        format_time_axis(ax, label=idx >= n_examples - n_cols)
        if idx < n_examples - n_cols:
            ax.set_xticklabels([])
        if idx % n_cols == 0:
            ax.set_ylabel("Signal", labelpad=1.0, fontsize=6.5)
        ax.tick_params(labelsize=5.5, pad=1.0)
        set_sparse_y(ax, 3)
        padded_ylim(ax, [y, exp_series["lo"][:, exp_idx], exp_series["hi"][:, exp_idx],
                         teacher_traj[:, teacher_idx], student_traj[:, teacher_idx]])
        style_axis(ax, grid=True)

    handles = [
        Line2D([0], [0], marker="o", color=COLORS["black"], mfc="white", lw=0, ms=3.0, label="Experiment"),
        Line2D([0], [0], color=COLORS["teacher"], lw=1.4, ls="--", label="Teacher"),
        Line2D([0], [0], color=COLORS["student"], lw=1.5, label="Student"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.50, 0.98), handlelength=1.0, borderaxespad=0.10, ncol=1,
                   fontsize=5.8, labelspacing=0.10)
    return axes


def plot_panel_b(fig: plt.Figure, spec) -> plt.Axes:
    ax = fig.add_subplot(spec)
    data = np.load(HMM_DATA, allow_pickle=True)
    post = data["posteriors"].astype(float)
    time_points = data["time_points"].astype(float)
    time_indices = data["time_indices"].astype(int)

    valid_rows = np.flatnonzero(time_indices > 0)
    if len(valid_rows) < len(post):
        valid_rows = np.arange(len(post))
    valid_rows = valid_rows[: len(post)]
    times = time_points[time_indices[valid_rows]]
    phase_raw = np.argmax(post, axis=1)
    medians = [np.median(times[phase_raw == k]) for k in range(post.shape[1])]
    fast_idx = int(np.argmin(medians))
    slow_idx = 1 - fast_idx

    df = pd.DataFrame({"time": times, "fast": post[:, fast_idx], "slow": post[:, slow_idx], "phase": phase_raw})
    mean = df.groupby("time")[["fast", "slow"]].mean().reset_index()
    ax.plot(mean["time"], mean["fast"], color=COLORS["fast"], lw=2.4, label="Early phase")
    ax.plot(mean["time"], mean["slow"], color=COLORS["slow"], lw=2.4, label="Late phase")
    phase_switch = mean.loc[np.argmin(np.abs(mean["fast"] - mean["slow"])), "time"]
    ax.axvline(phase_switch, color=COLORS["mid"], lw=1.2, ls="--", alpha=0.8)
    ax.text(
        phase_switch * 1.08,
        0.52,
        f"$t_{{HMM}}$={phase_switch:.1f} min",
        fontsize=6.1,
        color=COLORS["mid"],
        rotation=90,
        va="center",
        ha="left",
    )
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("Posterior", labelpad=1.5)
    format_time_axis(ax, label=True, log_only=True)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.0]))
    style_axis(ax, grid=True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.88), handlelength=1.6, borderaxespad=0.25)

    inset = ax.inset_axes([0.08, 0.15, 0.27, 0.34])
    counts = [
        int(np.sum(phase_raw == fast_idx)),
        int(np.sum(phase_raw == slow_idx)),
    ]
    inset.bar([0, 1], counts, color=[COLORS["fast"], COLORS["slow"]], width=0.65)
    inset.set_xticks([0, 1], ["Early", "Late"])
    inset.set_yticks([])
    inset.tick_params(labelsize=5.3, width=0.8, length=2, pad=1)
    for i, count in enumerate(counts):
        inset.text(i, count * 0.55, f"{count}", ha="center", va="center", fontsize=5.2, fontweight="bold")
    inset.set_title("Samples", fontsize=5.6, pad=1.0, fontweight="bold")
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)
    return ax


def plot_panel_c(fig: plt.Figure, spec) -> plt.Axes:
    ax = fig.add_subplot(spec)
    checkpoint = torch.load(STUDENT_MODEL, map_location="cpu", weights_only=False)
    losses = checkpoint["training_losses"]
    epochs = np.arange(1, len(losses["total"]) + 1)
    for key, color, label in [
        ("total", COLORS["black"], "Total"),
        ("output", COLORS["output"], "Output"),
        ("hidden", COLORS["hidden"], "Hidden"),
    ]:
        vals = np.maximum(np.asarray(losses[key], dtype=float), 1e-8)
        ax.plot(epochs, moving_average(vals, 7), color=color, lw=2.2, label=label)
    ax.set_yscale("log")
    ax.set_xlim(1, len(epochs))
    ax.set_xlabel("Epoch", labelpad=1.0)
    ax.set_ylabel("Loss", labelpad=1.5)
    ax.xaxis.set_major_locator(FixedLocator([1, 250, 500]))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    style_axis(ax, grid=True)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6, borderaxespad=0.25)
    return ax


def save_outputs(fig: plt.Figure) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"hpn_dream_nature_main.{ext}", facecolor="white")
    fig.savefig(PANEL_DIR / "full_preview.png", facecolor="white", dpi=300)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    configure_style()

    exp_path = resolve_experimental_path()
    df_exp, x_exp, y_exp, info_exp = load_midas(exp_path)
    del df_exp
    exp_series_by_condition = {}
    for _, condition in PANEL_A_EXAMPLES:
        stimulus, inhibitor = condition.split("|")
        exp_series_by_condition[condition] = aggregate_condition_series(
            x_exp,
            y_exp,
            info_exp,
            stimulus,
            inhibitor,
            max_time=240.0,
        )

    high = load_high_res()
    teacher_model, teacher_ckpt = load_model(TEACHER_MODEL)
    student_model, student_ckpt = load_model(STUDENT_MODEL)
    teacher_predictions = predict_model(teacher_model, teacher_ckpt, high["x"])
    student_predictions = predict_model(student_model, student_ckpt, high["x"])

    fig = plt.figure(figsize=(9.5, 10.2), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.992,
        top=0.980,
        bottom=0.038,
        hspace=0.12,
        wspace=0.20,
        height_ratios=[4.5, 1.00],
    )
    n_rows_a = (len(PANEL_A_EXAMPLES) + 3) // 4
    panel_a_spec = outer[0, :].subgridspec(n_rows_a, 4, hspace=0.15, wspace=0.18)

    plot_panel_a(fig, panel_a_spec, exp_series_by_condition, info_exp["protein_names"], high,
                 teacher_predictions, student_predictions)
    plot_panel_b(fig, outer[1, 0])
    plot_panel_c(fig, outer[1, 1])

    panel_info = [
        (outer[0, :], "a", "Surrogate validation: Experiment, teacher & student"),
        (outer[1, 0], "b", "HMM signaling phase discovery"),
        (outer[1, 1], "c", "PAKD training"),
    ]
    for cell, label, title in panel_info:
        bbox = cell.get_position(fig)
        y = bbox.y1 + 0.002
        x_label = bbox.x0 - 0.018
        x_title = bbox.x0
        fig.text(x_label, y, label, fontsize=12, fontweight="bold", ha="left", va="bottom")
        fig.text(x_title, y, title, fontsize=8.5, fontweight="bold", ha="left", va="bottom")

    save_outputs(fig)
    plt.close(fig)

    print("Saved:")
    for ext in ["pdf", "png", "svg"]:
        path = OUT_DIR / f"hpn_dream_nature_main.{ext}"
        print(f"  {path} ({path.stat().st_size:,} bytes)")
    print(f"  {PANEL_DIR / 'full_preview.png'}")


if __name__ == "__main__":
    main()
