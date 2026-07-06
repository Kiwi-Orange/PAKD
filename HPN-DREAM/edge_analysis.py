"""
edge_analysis.py
----------------
Analyzes how edge count affects ODE fitting accuracy for teacher and student
models in the DARTS-Hill pipeline.

Strategy
--------
* Sort all edges by gate value (descending).
* For k = 0 … max_edges: keep the top-k gates at their original *soft* values,
  zero out the rest.  This preserves the trained dynamics for included edges
  and avoids the Vmax-dominance artifact of binary masking.
* Accuracy metric: derivative MSE between model-predicted dX/dt and
  cubic-spline-estimated dX/dt, evaluated at every observed time point.
  Normalized fit quality = 1 - MSE(k) / MSE(0)  (0 = no edges, 1 = full model).

Outputs (results/edge_analysis/)
---------------------------------
  accuracy_vs_edges_{source}.csv   — raw (condition, k_edges, mse, fit_quality)
  summary_{source}.csv             — min edges for 90 % / 95 % of peak quality
  plots/accuracy_vs_edges_{source}.png/.pdf
  plots/r2_heatmap_conditions_{source}.png/.pdf
  plots/accuracy_vs_edges_comparison.png/.pdf
"""

from __future__ import annotations

import os
import glob
import warnings
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hpn_dream_matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import CubicSpline
import networkx as nx
import torch
from test_teacher import (
    COLORS as PUB_COLORS,
    save_publication_figure,
    set_publication_style,
    style_publication_axes,
)

warnings.filterwarnings("ignore")

set_publication_style()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, "grn_ready_data")
RESULTS_DIR = os.path.join(ROOT, "results", "darts_hill")
OUT_DIR     = os.path.join(ROOT, "results", "edge_analysis")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

SOURCES = ["teacher", "student"]

# ── Data loading ───────────────────────────────────────────────────────────────

def load_ts(source: str, condition: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = os.path.join(DATA_DIR, f"ts_{source}_pred_{condition}.csv")
    df = pd.read_csv(path, index_col=0)
    proteins = list(df.index)
    times = np.array([float(c) for c in df.columns], dtype=float)
    return times, df.values.T.astype(np.float32), proteins   # (T, P)


def normalize_proteins(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins   = X.min(axis=0)
    scales = X.max(axis=0) - mins
    scales[scales < 1e-12] = 1.0
    return (X - mins) / scales, mins, scales


def discover_conditions(source: str) -> List[str]:
    pattern = os.path.join(DATA_DIR, f"ts_{source}_pred_*.csv")
    return [
        os.path.basename(f).replace(f"ts_{source}_pred_", "").replace(".csv", "")
        for f in sorted(glob.glob(pattern))
    ]


def load_mats(source: str, condition: str) -> Dict[str, np.ndarray]:
    """Load GATES, BETA, K, n and reconstruct sign_gate from edges.csv."""
    cond_dir = os.path.join(RESULTS_DIR, source, "per_condition", condition)
    mats: Dict[str, np.ndarray] = {}
    proteins: List[str] = []
    for name in ("GATES", "BETA", "K", "n"):
        df = pd.read_csv(os.path.join(cond_dir, f"{name}.csv"), index_col=0)
        mats[name.lower()] = df.values.astype(np.float32)
        if name == "GATES":
            proteins = list(df.index)
    mats["proteins"] = proteins  # type: ignore[assignment]

    # sign_gate: default from beta sign, override with edges.csv
    sign_gate = (mats["beta"] >= 0).astype(np.float32)
    edges_path = os.path.join(cond_dir, "edges.csv")
    if os.path.exists(edges_path):
        prot_idx = {p: i for i, p in enumerate(proteins)}
        for _, row in pd.read_csv(edges_path).iterrows():
            j = prot_idx.get(row["source"])
            i = prot_idx.get(row["target"])
            if j is not None and i is not None:
                sign_gate[j, i] = 1.0 if row["sign"] == "activation" else 0.0
    mats["sign_gate"] = sign_gate
    return mats


def load_node_params(
    source: str, condition: str, proteins: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse gamma, basal, Vmax from equations.txt."""
    P = len(proteins)
    gamma = np.ones(P,  dtype=np.float32)
    basal = np.zeros(P, dtype=np.float32)
    Vmax  = np.ones(P,  dtype=np.float32)
    prot_idx = {p: i for i, p in enumerate(proteins)}

    eq_path = os.path.join(RESULTS_DIR, source, "per_condition", condition, "equations.txt")
    with open(eq_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("gamma["):
                continue
            try:
                parts = line.split(",")
                pname = parts[0].split("[")[1].split("]")[0]
                idx = prot_idx.get(pname)
                if idx is None:
                    continue
                gamma[idx] = float(parts[0].split("=")[1])
                basal[idx] = float(parts[1].split("=")[1])
                Vmax[idx]  = float(parts[2].split("=")[1])
            except Exception:
                pass
    return gamma, basal, Vmax


# ── Hill ODE (stateless, no gradient) ─────────────────────────────────────────

def _hill(x: torch.Tensor, K: torch.Tensor, n: torch.Tensor, eps: float = 1e-12):
    x  = torch.clamp(x, min=0.0)
    K  = torch.clamp(K, min=eps)
    n  = torch.clamp(n, min=0.1, max=6.0)
    xn = torch.nan_to_num(torch.pow(x + eps, n), nan=0.0, posinf=1e8)
    Kn = torch.nan_to_num(torch.pow(K,       n), nan=0.0, posinf=1e8)
    d  = Kn + xn + eps
    return torch.clamp(xn / d, 0.0, 1.0), torch.clamp(Kn / d, 0.0, 1.0)


class HillODE(torch.nn.Module):
    """Forward-only Hill-ODE with pre-loaded (possibly masked) parameters."""

    def __init__(
        self,
        eff_gates: np.ndarray,   # (P,P) — already zeroed for removed edges
        beta:      np.ndarray,
        K:         np.ndarray,
        n:         np.ndarray,
        sign_gate: np.ndarray,
        gamma:     np.ndarray,
        basal:     np.ndarray,
        Vmax:      np.ndarray,
    ):
        super().__init__()
        def buf(a): return torch.tensor(a, dtype=torch.float32)
        self.register_buffer("g",    buf(eff_gates))
        self.register_buffer("beta", buf(beta))
        self.register_buffer("K",    buf(np.clip(K, 1e-6, None)))
        self.register_buffer("n",    buf(np.clip(n, 0.1, 6.0)))
        self.register_buffer("sg",   buf(sign_gate))
        self.register_buffer("gam",  buf(gamma))
        self.register_buffer("bas",  buf(basal))
        self.register_buffer("Vmax", buf(Vmax))

    @torch.no_grad()
    def forward(self, __t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        squeeze = X.ndim == 1
        Xb = X.unsqueeze(0) if squeeze else X          # (B, P)
        P  = Xb.shape[1]
        Xs = Xb[:, :, None].expand(-1, P, P)           # (B, P_src, P_tgt)
        Ha, Hi = _hill(Xs, self.K[None], self.n[None])
        H  = self.sg[None] * Ha + (1.0 - self.sg[None]) * Hi

        decay = -self.gam[None] * Xb
        add   = (self.g[None] * self.beta[None] * H).sum(dim=1) + self.bas[None]
        Hw    = self.g[None] * H + (1.0 - self.g[None])
        mult  = self.Vmax[None] * Hw.prod(dim=1)

        dX = torch.nan_to_num(decay + add + mult, nan=0.0, posinf=10.0, neginf=-10.0)
        dX = torch.clamp(dX, -100.0, 100.0)
        return dX.squeeze(0) if squeeze else dX


# ── Accuracy metric: derivative MSE ───────────────────────────────────────────

def spline_derivatives(t: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cubic-spline dY/dt at observed time points. Returns (T, P)."""
    dYdt = np.empty_like(Y)
    for p in range(Y.shape[1]):
        dYdt[:, p] = CubicSpline(t, Y[:, p])(t, 1)
    return dYdt


def derivative_mse(model: HillODE, t: np.ndarray, X_norm: np.ndarray) -> float:
    """MSE between model dX/dt and spline dX/dt, averaged over (T, P)."""
    target = spline_derivatives(t, X_norm)                 # (T, P)
    X_t    = torch.tensor(X_norm, dtype=torch.float32)
    t_dummy = torch.zeros(1)
    pred = np.stack([
        model(t_dummy, X_t[ti]).cpu().numpy() for ti in range(len(t))
    ])                                                      # (T, P)
    return float(np.mean((pred - target) ** 2))


# ── Soft top-k gate masking ────────────────────────────────────────────────────

def topk_soft_gates(gates: np.ndarray, k: int) -> np.ndarray:
    """
    Return a (P,P) matrix with the top-k off-diagonal gate values kept at
    their original soft values; all other entries set to 0.
    """
    g = gates.copy()
    np.fill_diagonal(g, 0.0)
    if k == 0:
        return np.zeros_like(g)
    flat = g.ravel()
    n_nonzero = int((flat > 0).sum())
    if k >= n_nonzero:
        return g
    thr = np.sort(flat)[-k]          # k-th largest value
    return np.where(g >= thr, g, 0.0)


def edge_counts(gates: np.ndarray) -> np.ndarray:
    """Sorted unique edge counts to sweep (0, 1, 2, …, max)."""
    g = gates.copy()
    np.fill_diagonal(g, 0.0)
    n_max = int((g > 0).sum())
    # Use a coarser grid for large networks to keep runtime manageable
    if n_max <= 50:
        return np.arange(0, n_max + 1)
    steps = np.unique(np.round(
        np.concatenate([
            np.arange(0, min(30, n_max) + 1),
            np.linspace(30, n_max, 40).astype(int),
        ])
    ).astype(int))
    return steps[steps <= n_max]


# ── Per-condition analysis ─────────────────────────────────────────────────────

def analyze_condition(source: str, condition: str) -> pd.DataFrame:
    """
    Sweep edge count for one condition.
    Returns DataFrame: condition, k_edges, mse, fit_quality, gate_thr_equiv.
    """
    times, X_raw, proteins = load_ts(source, condition)
    X_norm, _, _            = normalize_proteins(X_raw)
    mats                    = load_mats(source, condition)
    gamma, basal, Vmax      = load_node_params(source, condition, proteins)

    gates_mat = mats["gates"]
    beta_mat  = mats["beta"]
    K_mat     = mats["k"]
    n_mat     = mats["n"]
    sg_mat    = mats["sign_gate"]

    # Baseline: no edges (k=0)
    model0 = HillODE(np.zeros_like(gates_mat), beta_mat, K_mat, n_mat,
                     sg_mat, gamma, basal, Vmax)
    mse0   = derivative_mse(model0, times, X_norm)

    ks   = edge_counts(gates_mat)
    rows = []
    for k in ks:
        eff_g = topk_soft_gates(gates_mat, int(k))
        model = HillODE(eff_g, beta_mat, K_mat, n_mat, sg_mat, gamma, basal, Vmax)
        mse   = derivative_mse(model, times, X_norm)
        fq    = 1.0 - mse / (mse0 + 1e-12)   # normalized fit quality

        # Equivalent gate threshold: min gate value among kept edges
        kept = eff_g[eff_g > 0]
        gate_thr_equiv = float(kept.min()) if len(kept) > 0 else 1.0

        rows.append({
            "condition":      condition,
            "k_edges":        int(k),
            "mse":            round(mse, 6),
            "fit_quality":    round(fq, 4),
            "gate_thr_equiv": round(gate_thr_equiv, 4),
        })

    return pd.DataFrame(rows)


def analyze_source(source: str) -> pd.DataFrame:
    conditions = discover_conditions(source)
    print(f"\n{'='*60}")
    print(f"  Source: {source.upper()} | {len(conditions)} conditions")
    print(f"{'='*60}")

    all_dfs = []
    for i, cond in enumerate(conditions):
        print(f"  [{i+1:02d}/{len(conditions)}] {cond} ...", end=" ", flush=True)
        try:
            df = analyze_condition(source, cond)
            all_dfs.append(df)
            best_fq = df["fit_quality"].max()
            target  = 0.9 * best_fq
            good    = df[df["fit_quality"] >= target]
            min_e   = int(good["k_edges"].min()) if len(good) > 0 else int(df["k_edges"].max())
            print(f"peak_fq={best_fq:.3f}, min_edges(90%)={min_e}")
        except Exception as e:
            print(f"ERROR: {e}")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ── Summary ────────────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for cond, grp in df.groupby("condition"):
        grp     = grp.sort_values("k_edges")
        best_fq = grp["fit_quality"].max()
        max_e   = int(grp["k_edges"].max())
        for pct in (0.90, 0.95):
            good  = grp[grp["fit_quality"] >= pct * best_fq]
            min_e = int(good["k_edges"].min()) if len(good) > 0 else max_e
            rows.append({
                "source":          source,
                "condition":       cond,
                "peak_fit_quality": round(best_fq, 4),
                "pct_threshold":   pct,
                "min_edges":       min_e,
                "max_edges_total": max_e,
                "pct_of_max":      round(min_e / max(max_e, 1), 3),
            })
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

TEACHER_COLOR = PUB_COLORS["blue"]
STUDENT_COLOR = PUB_COLORS["prediction"]
COMMON_COLOR = PUB_COLORS["gray"]
LIGHT_GRID = PUB_COLORS["light_gray"]


def _save(fig: plt.Figure, path_no_ext: str):
    out_dir = os.path.dirname(path_no_ext)
    stem = os.path.basename(path_no_ext)
    save_publication_figure(fig, out_dir, stem)
    print(f"  Saved: {stem}.png/.pdf/.svg")


def _style_edge_axes(ax: plt.Axes, grid_axis: str | None = None):
    style_publication_axes(ax)
    if grid_axis:
        ax.grid(axis=grid_axis, color=LIGHT_GRID, linewidth=0.35)


def _short_condition_label(condition: str) -> str:
    label = condition.replace("_GSK690693_GSK1120212", "\nGSK+MEKi")
    label = label.replace("_GSK690693", "\nGSK")
    label = label.replace("_PD173074", "\nPD")
    label = label.replace("_None", "")
    label = label.replace("Insulin", "Ins")
    return label


def _condition_tick_label(condition: str) -> str:
    return _short_condition_label(condition).replace("\n", " ")


def plot_accuracy_vs_edges(df: pd.DataFrame, source: str):
    """Mean ± SD fit quality vs edge count across conditions."""
    agg = (
        df.groupby("k_edges")
        .agg(fq_mean=("fit_quality", "mean"),
             fq_std=("fit_quality", "std"),
             fq_med=("fit_quality", "median"))
        .reset_index()
        .sort_values("k_edges")
    )

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.fill_between(agg["k_edges"],
                    agg["fq_mean"] - agg["fq_std"],
                    agg["fq_mean"] + agg["fq_std"],
                    alpha=0.16, color=TEACHER_COLOR, linewidth=0, label="Mean ± SD")
    ax.plot(agg["k_edges"], agg["fq_mean"], "-o", ms=1.8, lw=0.9,
            color=TEACHER_COLOR, label="Mean")
    ax.plot(agg["k_edges"], agg["fq_med"], "--", color=STUDENT_COLOR,
            lw=0.8, label="Median")
    ax.axhline(0.90, ls=":", color=COMMON_COLOR, lw=0.65, label="90%")
    ax.axhline(0.95, ls="--", color=PUB_COLORS["good"], lw=0.65, label="95%")
    ax.set_xlabel("Edges (top-k gate)")
    ax.set_ylabel("Normalized fit quality")
    ax.set_title(source.capitalize(), pad=2.5)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    ax.set_ylim(-0.03, 1.03)
    _style_edge_axes(ax)
    fig.tight_layout()
    _save(fig, os.path.join(OUT_DIR, "plots", f"accuracy_vs_edges_{source}"))


def plot_per_condition_heatmap(df: pd.DataFrame, source: str):
    """Heatmap: fit quality per condition vs edge count."""
    # Downsample columns for readability
    all_k = sorted(df["k_edges"].unique())
    step  = max(1, len(all_k) // 30)
    k_sel = all_k[::step]
    sub   = df[df["k_edges"].isin(k_sel)]
    pivot = sub.pivot_table(index="condition", columns="k_edges", values="fit_quality")

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Fit quality")
    cbar.ax.tick_params(width=0.6, length=2)
    tick_step = max(1, len(pivot.columns) // 8)
    x_ticks = list(range(0, len(pivot.columns), tick_step))
    if x_ticks[-1] != len(pivot.columns) - 1:
        x_ticks.append(len(pivot.columns) - 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([pivot.columns[i] for i in x_ticks], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_condition_tick_label(c) for c in pivot.index])
    ax.set_xlabel("Edges (top-k)")
    ax.set_ylabel("Condition")
    ax.set_title(source.capitalize(), pad=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(width=0.6, length=2, pad=1.2)
    fig.tight_layout()
    _save(fig, os.path.join(OUT_DIR, "plots", f"fq_heatmap_conditions_{source}"))


def plot_joint_comparison(df_t: pd.DataFrame, df_s: pd.DataFrame):
    """Overlay teacher vs student mean fit quality vs edge count."""
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for df, color, label in [(df_t, TEACHER_COLOR, "Teacher"), (df_s, STUDENT_COLOR, "Student")]:
        agg = (df.groupby("k_edges")
               .agg(fq_mean=("fit_quality", "mean"), fq_std=("fit_quality", "std"))
               .reset_index().sort_values("k_edges"))
        ax.fill_between(agg["k_edges"],
                        agg["fq_mean"] - agg["fq_std"],
                        agg["fq_mean"] + agg["fq_std"],
                        alpha=0.12, color=color, linewidth=0)
        ax.plot(agg["k_edges"], agg["fq_mean"], "-o", ms=1.8, lw=0.9,
                color=color, label=label)

    ax.axhline(0.90, ls=":", color=COMMON_COLOR, lw=0.65, label="90%")
    ax.axhline(0.95, ls="--", color=COMMON_COLOR, lw=0.65, label="95%")
    ax.set_xlabel("Edges (top-k gate)")
    ax.set_ylabel("Normalized fit quality")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    ax.set_ylim(-0.03, 1.03)
    _style_edge_axes(ax)
    fig.tight_layout()
    _save(fig, os.path.join(OUT_DIR, "plots", "accuracy_vs_edges_comparison"))


def plot_per_condition_comparison(df_t: pd.DataFrame, df_s: pd.DataFrame,
                                  target_fq: float = 0.60):
    """
    6×6 grid: one subplot per condition, teacher (blue) vs student (red).
    A horizontal line marks target_fq; vertical ticks show where each crosses it.
    """
    conditions = sorted(df_t["condition"].unique())
    n_cols = 6
    n_rows = int(np.ceil(len(conditions) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.2, 7.0),
                             sharex=False, sharey=True)
    axes_flat = axes.ravel()

    def edges_at(df: pd.DataFrame, cond: str, fq: float) -> int:
        grp  = df[df["condition"] == cond].sort_values("k_edges")
        good = grp[grp["fit_quality"] >= fq]
        return int(good["k_edges"].min()) if len(good) else int(grp["k_edges"].max())

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]

        for df, color, label in [(df_t, TEACHER_COLOR, "Teacher"),
                                  (df_s, STUDENT_COLOR, "Student")]:
            grp = df[df["condition"] == cond].sort_values("k_edges")
            ax.plot(grp["k_edges"], grp["fit_quality"],
                    "-", lw=0.65, color=color, label=label)

            # vertical marker at crossing
            e = edges_at(df, cond, target_fq)
            ax.axvline(e, color=color, ls="--", lw=0.45, alpha=0.55)

        ax.axhline(target_fq, color=COMMON_COLOR, ls=":", lw=0.55)
        ax.set_title(_short_condition_label(cond), pad=1.5)
        ax.set_ylim(-0.03, 1.03)
        _style_edge_axes(ax)

        # axis labels only on edges
        if idx % n_cols == 0:
            ax.set_ylabel("Fit quality")
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Edges")

    # hide unused subplots
    for ax in axes_flat[len(conditions):]:
        ax.axis("off")

    # shared legend
    handles = [
        plt.Line2D([0], [0], color=TEACHER_COLOR, lw=1.0, label="Teacher"),
        plt.Line2D([0], [0], color=STUDENT_COLOR, lw=1.0, label="Student"),
        plt.Line2D([0], [0], color=COMMON_COLOR, lw=0.8, ls=":", label=f"FQ={target_fq}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6)
    fig.tight_layout(rect=[0, 0.035, 1, 1], h_pad=0.3, w_pad=0.25)
    _save(fig, os.path.join(OUT_DIR, "plots", "per_condition_comparison"))


def plot_edges_at_target_bar(df_t: pd.DataFrame, df_s: pd.DataFrame,
                              target_fq: float = 0.60):
    """
    Grouped bar chart: edges needed to reach target_fq, teacher vs student,
    sorted by the teacher–student difference so the gap is immediately visible.
    """
    def edges_at(df: pd.DataFrame, fq: float) -> pd.Series:
        out = {}
        for cond, grp in df.groupby("condition"):
            grp  = grp.sort_values("k_edges")
            good = grp[grp["fit_quality"] >= fq]
            out[cond] = int(good["k_edges"].min()) if len(good) else int(grp["k_edges"].max())
        return pd.Series(out)

    et = edges_at(df_t, target_fq).rename("teacher")
    es = edges_at(df_s, target_fq).rename("student")
    cmp = pd.concat([et, es], axis=1)
    cmp["diff"] = cmp["teacher"] - cmp["student"]
    cmp = cmp.sort_values("diff", ascending=False)

    x    = np.arange(len(cmp))
    w    = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0),
                              gridspec_kw={"height_ratios": [2.5, 1]})

    # ── top: grouped bars ──
    ax = axes[0]
    ax.bar(x - w/2, cmp["teacher"], w, color=TEACHER_COLOR, alpha=0.88, label="Teacher")
    ax.bar(x + w/2, cmp["student"], w, color=STUDENT_COLOR, alpha=0.88, label="Student")
    ax.set_xticks(x)
    ax.set_xticklabels([_condition_tick_label(c) for c in cmp.index], rotation=55, ha="right")
    ax.set_ylabel(f"Edges to reach FQ ≥ {target_fq}")
    ax.legend(frameon=False, loc="upper right", handlelength=1.5)
    _style_edge_axes(ax, grid_axis="y")
    ax.set_xlim(-0.6, len(cmp) - 0.4)

    # ── bottom: difference bar (teacher − student) ──
    ax2 = axes[1]
    colors = [TEACHER_COLOR if d >= 0 else STUDENT_COLOR for d in cmp["diff"]]
    ax2.bar(x, cmp["diff"], color=colors, alpha=0.8, edgecolor="white")
    ax2.axhline(0, color=PUB_COLORS["ground_truth"], lw=0.65)
    ax2.set_xticks(x)
    ax2.set_xticklabels([_condition_tick_label(c) for c in cmp.index], rotation=55, ha="right")
    ax2.set_ylabel("Teacher − Student\n(edges)")
    _style_edge_axes(ax2, grid_axis="y")
    ax2.set_xlim(-0.6, len(cmp) - 0.4)

    # annotation: median difference
    med = cmp["diff"].median()
    ax2.axhline(med, color=COMMON_COLOR, ls="--", lw=0.7,
                label=f"Median diff = {med:.0f}")
    ax2.legend(frameon=False, loc="best", handlelength=1.5)

    fig.tight_layout(h_pad=0.55)
    _save(fig, os.path.join(OUT_DIR, "plots",
                            f"edges_at_{int(target_fq*100)}pct_bar"))


def plot_min_edges_distribution(summary: pd.DataFrame, source: str):
    """Box/violin plot of min_edges distribution across conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.25))
    for ax, pct in zip(axes, [0.90, 0.95]):
        sub = summary[summary["pct_threshold"] == pct]
        ax.hist(sub["min_edges"], bins=14, color=PUB_COLORS["moderate"],
                edgecolor="white", linewidth=0.4, alpha=0.88)
        ax.axvline(sub["min_edges"].median(), color=PUB_COLORS["ground_truth"],
                   ls="--", lw=0.8,
                   label=f"Median={sub['min_edges'].median():.0f}")
        ax.set_xlabel("Min edges")
        ax.set_ylabel("Conditions")
        ax.set_title(f"{source.capitalize()} {int(pct*100)}%", pad=2.5)
        ax.legend(frameon=False, loc="best", handlelength=1.4)
        _style_edge_axes(ax, grid_axis="y")
    fig.tight_layout(w_pad=0.65)
    _save(fig, os.path.join(OUT_DIR, "plots", f"min_edges_distribution_{source}"))


# ── Network topology comparison ────────────────────────────────────────────────

# Short display names (single-line, compact)
_SHORT = {
    "EGFR_pY1068": "EGFR\npY1068",  "EGFR_pY1173": "EGFR\npY1173",
    "EGFR_pY992":  "EGFR\npY992",   "AKT_pS473":   "AKT\npS473",
    "AKT_pT308":   "AKT\npT308",    "MAPK_pT202_Y204": "MAPK\npT202",
    "MEK1_pS217_S221": "MEK1",      "mTOR_pS2448": "mTOR",
    "S6_pS235_S236": "S6\npS235",   "S6_pS240_S244": "S6\npS240",
    "p70S6K_pT389": "p70S6K",       "4EBP1_pS65":  "4EBP1",
    "STAT3_pY705": "STAT3",         "p38_pT180_Y182": "p38",
    "JNK_pT183_pT185": "JNK",       "c-JUN_pS73":  "cJUN",
    "c-Raf_pS338": "cRaf",          "GSK3-alpha-beta_pS21_S9": "GSK3ab\npS21",
    "GSK3-alpha-beta_pS9": "GSK3ab\npS9", "PRAS40_pT246": "PRAS40",
    "PDK1_pS241":  "PDK1",          "AMPK_pT172":  "AMPK",
    "BAD_pS112":   "BAD",           "Rb_pS807_S811": "Rb",
    "HER2_pY1248": "HER2",          "c-Met_pY1235": "cMet",
    "p90RSK_pT359_S363": "p90RSK",  "CHK1_pS345":  "CHK1",
    "CHK2_pT68":   "CHK2",          "NF-kB-p65_pS536": "NFkB",
    "Src_pY416":   "Src\npY416",    "Src_pY527":   "Src\npY527",
    "p27_pT157":   "p27\npT157",    "p27_pT198":   "p27\npT198",
    "FOXO3a_pS318_S321": "FOXO3a",  "ER-alpha_pS118": "ERα",
    "ACC_pS79":    "ACC",           "PKC-alpha_pS657": "PKCα",
    "TAZ_pS89":    "TAZ",           "YAP_pS127":   "YAP",
    "YB-1_PS102":  "YB-1",
}

def _short(name: str) -> str:
    return _SHORT.get(name, name.split("_")[0])


def _get_edges_at_fq(df: pd.DataFrame, condition: str,
                     mats: dict, target_fq: float) -> set:
    """
    Return the set of (source, target) edge tuples present in the top-k network
    that achieves target_fq fit quality for this condition.
    """
    grp  = df[df["condition"] == condition].sort_values("k_edges")
    good = grp[grp["fit_quality"] >= target_fq]
    k    = int(good["k_edges"].min()) if len(good) else int(grp["k_edges"].max())
    return _edges_from_topk(mats, k)


def _get_edges_at_gate_thr(df: pd.DataFrame, condition: str,
                            mats: dict, gate_thr: float) -> tuple[set, float, int]:
    """
    Return edges at the k closest to gate_thr equivalent, plus actual FQ and k.
    """
    grp = df[df["condition"] == condition].copy()
    grp["dist"] = (grp["gate_thr_equiv"] - gate_thr).abs()
    best = grp.loc[grp["dist"].idxmin()]
    k  = int(best["k_edges"])
    fq = float(best["fit_quality"])
    return _edges_from_topk(mats, k), fq, k


def _edges_from_topk(mats: dict, k: int) -> set:
    gates    = mats["gates"]
    proteins = mats["proteins"]
    eff_g    = topk_soft_gates(gates, k)
    edges: set = set()
    for j, src in enumerate(proteins):
        for i, tgt in enumerate(proteins):
            if i != j and eff_g[j, i] > 0:
                edges.add((src, tgt))
    return edges


def _fixed_layout(all_nodes: list, seed: int = 42) -> dict:
    """Spring layout on a complete shell so all 41 proteins fit nicely."""
    G_dummy = nx.DiGraph()
    G_dummy.add_nodes_from(all_nodes)
    # Add a few hub edges so spring layout spreads things out
    return nx.spring_layout(G_dummy, k=2.5 / max(len(all_nodes) ** 0.5, 1),
                            iterations=300, seed=seed)


# Edge / node colors
_C_COMMON  = COMMON_COLOR
_C_TEACHER = TEACHER_COLOR
_C_STUDENT = STUDENT_COLOR
_C_NODE    = "#F2F2F2"


def _draw_network_ax(
    ax: plt.Axes,
    edges_teacher: set,
    edges_student: set,
    pos: dict,
    title: str,
    show_labels: bool = True,
    node_size: int = 600,
    font_size: int = 6,
    arrow_size: int = 10,
):
    """Draw a single network panel with three edge classes."""
    common       = edges_teacher & edges_student
    teach_only   = edges_teacher - edges_student
    stud_only    = edges_student - edges_teacher

    all_nodes = sorted({n for e in (edges_teacher | edges_student)
                        for n in e})
    if not all_nodes:
        ax.text(0.5, 0.5, "no edges", ha="center", va="center",
                transform=ax.transAxes, color=COMMON_COLOR)
        ax.axis("off")
        ax.set_title(title, pad=2)
        return

    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    for e in common:     G.add_edge(*e, etype="common")
    for e in teach_only: G.add_edge(*e, etype="teacher")
    for e in stud_only:  G.add_edge(*e, etype="student")

    sub_pos = {n: pos[n] for n in all_nodes if n in pos}

    # Draw nodes
    nx.draw_networkx_nodes(G, sub_pos, ax=ax,
                           node_color=_C_NODE, node_size=node_size,
                           edgecolors="#BDBDBD", linewidths=0.45)

    # Draw edges by type
    for etype, color, style in [
        ("common",  _C_COMMON,  "solid"),
        ("teacher", _C_TEACHER, "solid"),
        ("student", _C_STUDENT, "solid"),
    ]:
        elist = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == etype]
        if elist:
            nx.draw_networkx_edges(
                G, sub_pos, edgelist=elist, ax=ax,
                edge_color=color, style=style,
                arrows=True, arrowsize=arrow_size,
                width=0.55, alpha=0.68,
                connectionstyle="arc3,rad=0.12",
                node_size=node_size,
            )

    if show_labels:
        labels = {n: _short(n) for n in all_nodes}
        nx.draw_networkx_labels(G, sub_pos, labels=labels, ax=ax,
                                font_size=font_size, font_weight="normal")

    ax.set_title(title, pad=2.5)
    ax.axis("off")


def _legend_handles() -> list:
    return [
        mpatches.Patch(color=_C_COMMON,  label="Common"),
        mpatches.Patch(color=_C_TEACHER, label="Teacher only"),
        mpatches.Patch(color=_C_STUDENT, label="Student only"),
    ]


def plot_network_per_condition(
    df_t: pd.DataFrame,
    df_s: pd.DataFrame,
    target_fq: float = 0.60,
    out_subdir: str = "networks_per_condition",
):
    """
    One figure per condition: single network with edges colored by type.
    Uses a shared fixed layout so all figures are spatially consistent.
    """
    out_dir = os.path.join(OUT_DIR, "plots", out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    conditions = sorted(df_t["condition"].unique())

    # Build shared layout from all proteins
    sample_mats = load_mats("teacher", conditions[0])
    all_proteins: list = sample_mats["proteins"]  # type: ignore[assignment]
    pos = _fixed_layout(all_proteins)

    for cond in conditions:
        mats_t = load_mats("teacher", cond)
        mats_s = load_mats("student", cond)

        et = _get_edges_at_fq(df_t, cond, mats_t, target_fq)
        es = _get_edges_at_fq(df_s, cond, mats_s, target_fq)

        common     = et & es
        teach_only = et - es
        stud_only  = es - et

        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        _draw_network_ax(
            ax, et, es, pos,
            title=(f"{_short_condition_label(cond).replace(chr(10), ' ')} | FQ ≥ {target_fq}\n"
                   f"Common={len(common)}  Teacher={len(teach_only)}"
                   f"  Student={len(stud_only)}"),
            show_labels=True, node_size=260, font_size=3.8, arrow_size=7,
        )
        fig.legend(handles=_legend_handles(), loc="lower center",
                   ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.01))
        fig.tight_layout(rect=[0, 0.045, 1, 1])
        _save(fig, os.path.join(out_dir, cond))
        print(f"    {cond}: common={len(common)} teach={len(teach_only)} stud={len(stud_only)}")


def plot_network_grid(
    df_t: pd.DataFrame,
    df_s: pd.DataFrame,
    target_fq: float = 0.60,
    n_cols: int = 6,
):
    """
    Grid of small network thumbnails — one cell per condition.
    Edges colored: gray=common, blue=teacher-only, red=student-only.
    """
    conditions = sorted(df_t["condition"].unique())
    n_rows = int(np.ceil(len(conditions) / n_cols))

    sample_mats = load_mats("teacher", conditions[0])
    all_proteins: list = sample_mats["proteins"]  # type: ignore[assignment]
    pos = _fixed_layout(all_proteins)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.2, 6.4))
    axes_flat = axes.ravel()

    for idx, cond in enumerate(conditions):
        mats_t = load_mats("teacher", cond)
        mats_s = load_mats("student", cond)
        et = _get_edges_at_fq(df_t, cond, mats_t, target_fq)
        es = _get_edges_at_fq(df_s, cond, mats_s, target_fq)

        common     = et & es
        teach_only = et - es
        stud_only  = es - et

        short_cond = _short_condition_label(cond)
        title = (f"{short_cond}\n"
                 f"C={len(common)} T={len(teach_only)} S={len(stud_only)}")

        _draw_network_ax(
            axes_flat[idx], et, es, pos,
            title=title,
            show_labels=False, node_size=16, font_size=3, arrow_size=4,
        )

    for ax in axes_flat[len(conditions):]:
        ax.axis("off")

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.4)
    fig.tight_layout(rect=[0, 0.035, 1, 1], h_pad=0.3, w_pad=0.2)
    _save(fig, os.path.join(OUT_DIR, "plots", f"network_grid_fq{int(target_fq*100)}pct"))


def plot_selected_at_gate_thr(
    df_t: pd.DataFrame,
    df_s: pd.DataFrame,
    conditions: list,
    gate_thr: float = 0.3,
    n_cols: int = 3,
):
    """
    Multi-panel figure showing selected conditions at a specific gate threshold.
    Each panel: single network with gray=common, blue=teacher-only, red=student-only.
    Panel title shows actual k_edges and FQ for teacher and student.
    Conditions are sorted so teacher-heavy ones come first.
    """
    # Pre-compute edge sets and sort by teacher-student edge diff (descending)
    sample_mats = load_mats("teacher", conditions[0])
    all_proteins: list = sample_mats["proteins"]  # type: ignore[assignment]
    pos = _fixed_layout(all_proteins)

    info = []
    for cond in conditions:
        mats_t = load_mats("teacher", cond)
        mats_s = load_mats("student", cond)
        et, fq_t, k_t = _get_edges_at_gate_thr(df_t, cond, mats_t, gate_thr)
        es, fq_s, k_s = _get_edges_at_gate_thr(df_s, cond, mats_s, gate_thr)
        info.append(dict(cond=cond, et=et, es=es,
                         k_t=k_t, k_s=k_s, fq_t=fq_t, fq_s=fq_s,
                         diff=k_t - k_s))

    # Sort: teacher-heavy first
    info.sort(key=lambda x: -x["diff"])

    n_rows = int(np.ceil(len(info) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.2, max(2.5, 2.4 * n_rows)))
    axes_flat = np.array(axes).ravel()

    for idx, d in enumerate(info):
        ax    = axes_flat[idx]
        et, es = d["et"], d["es"]
        common     = et & es
        teach_only = et - es
        stud_only  = es - et

        title = (
            f"{_short_condition_label(d['cond']).replace(chr(10), ' ')}\n"
            f"T: {d['k_t']} edges (FQ={d['fq_t']:.2f})  "
            f"S: {d['k_s']} edges (FQ={d['fq_s']:.2f})\n"
            f"Common={len(common)}  T-only={len(teach_only)}  S-only={len(stud_only)}"
        )
        _draw_network_ax(ax, et, es, pos, title=title,
                         show_labels=True, node_size=220,
                         font_size=3.7, arrow_size=6)

    for ax in axes_flat[len(info):]:
        ax.axis("off")

    fig.legend(handles=_legend_handles(), loc="lower center",
               ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1], h_pad=0.4, w_pad=0.25)

    fname = f"network_selected_gate{int(gate_thr*100)}pct"
    _save(fig, os.path.join(OUT_DIR, "plots", fname))


def select_gate_threshold_conditions(
    df_t: pd.DataFrame,
    df_s: pd.DataFrame,
    gate_thr: float = 0.30,
    n_conditions: int = 9,
) -> list[str]:
    """Choose compact network panels with the largest teacher-student edge gap."""
    conditions = sorted(set(df_t["condition"].unique()) & set(df_s["condition"].unique()))
    ranked: list[tuple[int, str]] = []
    for cond in conditions:
        try:
            mats_t = load_mats("teacher", cond)
            mats_s = load_mats("student", cond)
            _, _, k_t = _get_edges_at_gate_thr(df_t, cond, mats_t, gate_thr)
            _, _, k_s = _get_edges_at_gate_thr(df_s, cond, mats_s, gate_thr)
        except Exception as exc:
            print(f"  Skipping selected network condition {cond}: {exc}")
            continue
        ranked.append((k_t - k_s, cond))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [cond for _, cond in ranked[:n_conditions]]


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    results: Dict[str, pd.DataFrame] = {}

    for source in SOURCES:
        df = analyze_source(source)
        csv_path = os.path.join(OUT_DIR, f"accuracy_vs_edges_{source}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")
        results[source] = df

        if len(df) == 0:
            continue

        plot_accuracy_vs_edges(df, source)
        plot_per_condition_heatmap(df, source)

        summary = compute_summary(df, source)
        summary.to_csv(os.path.join(OUT_DIR, f"summary_{source}.csv"), index=False)
        plot_min_edges_distribution(summary, source)

        for pct in [0.90, 0.95]:
            sub = summary[summary["pct_threshold"] == pct]
            print(f"\n  [{source}] Min edges for {int(pct*100)}% of peak fit quality:")
            print(f"    Median={sub['min_edges'].median():.0f}  "
                  f"Mean={sub['min_edges'].mean():.1f}  "
                  f"Max={sub['min_edges'].max():.0f}  "
                  f"(out of ~{sub['max_edges_total'].median():.0f} total edges, "
                  f"{sub['pct_of_max'].median()*100:.0f}% of max)")

    if all(s in results and len(results[s]) > 0 for s in SOURCES):
        plot_joint_comparison(results["teacher"], results["student"])
        plot_per_condition_comparison(results["teacher"], results["student"])
        plot_edges_at_target_bar(results["teacher"], results["student"])

        print("\n── Network topology plots ─────────────────────────────────")
        print("  Grid summary …")
        plot_network_grid(results["teacher"], results["student"], target_fq=0.60)
        print("  Selected gate-threshold summary …")
        selected = select_gate_threshold_conditions(results["teacher"], results["student"],
                                                    gate_thr=0.30, n_conditions=9)
        if selected:
            plot_selected_at_gate_thr(results["teacher"], results["student"],
                                      selected, gate_thr=0.30, n_cols=3)
        print("  Per-condition figures …")
        plot_network_per_condition(results["teacher"], results["student"], target_fq=0.60)

    print(f"\n{'='*60}")
    print(f"  All outputs → {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
