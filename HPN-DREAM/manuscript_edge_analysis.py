"""
manuscript_edge_analysis.py
----------------------------
Publication-quality network figure comparing teacher and student consensus
signaling networks.

Strategy
--------
* Load all per-condition edges (gate ≥ 0.25) for teacher and student.
* For each model, keep edges that appear in ≥ 25% of conditions.
* Classify edges: common | teacher-only | student-only.
* Draw a single circular figure with proteins arranged by signaling pathway,
  edges colored by class (gray=common, blue=teacher-only, red=student-only).

Outputs
-------
  results/edge_analysis/publication/
    network_consensus_overlay.png/.pdf   — main overlay figure
    network_consensus_teacher.png/.pdf   — teacher only
    network_consensus_student.png/.pdf   — student only
    consensus_edges_teacher.csv
    consensus_edges_student.csv
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hpn_dream_matplotlib")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import networkx as nx

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results", "darts_hill")
OUT_DIR     = os.path.join(ROOT, "results", "edge_analysis", "publication")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Publication style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
})

# ── Pathway groupings (circular shell order) ───────────────────────────────────
PATHWAY_GROUPS: Dict[str, List[str]] = {
    "RTK": [
        "EGFR_pY1068", "EGFR_pY1173", "EGFR_pY992",
        "HER2_pY1248", "c-Met_pY1235",
    ],
    "PI3K/AKT": [
        "AKT_pS473", "AKT_pT308", "PDK1_pS241",
        "PRAS40_pT246", "FOXO3a_pS318_S321",
        "GSK3-alpha-beta_pS21_S9", "GSK3-alpha-beta_pS9",
    ],
    "mTOR": [
        "mTOR_pS2448", "p70S6K_pT389", "S6_pS235_S236",
        "S6_pS240_S244", "4EBP1_pS65",
    ],
    "MAPK/ERK": [
        "c-Raf_pS338", "MEK1_pS217_S221", "MAPK_pT202_Y204",
        "p90RSK_pT359_S363", "c-JUN_pS73",
    ],
    "Stress/SAPK": [
        "p38_pT180_Y182", "JNK_pT183_pT185",
        "AMPK_pT172", "ACC_pS79",
    ],
    "Cell cycle": [
        "Rb_pS807_S811", "CHK1_pS345", "CHK2_pT68",
        "p27_pT157", "p27_pT198",
    ],
    "Other": [
        "Src_pY416", "Src_pY527", "PKC-alpha_pS657",
        "BAD_pS112", "STAT3_pY705", "NF-kB-p65_pS536",
        "ER-alpha_pS118", "YAP_pS127", "TAZ_pS89",
        "YB-1_PS102",
    ],
}

PATHWAY_COLORS: Dict[str, str] = {
    "RTK":         "#F4A261",  # warm orange
    "PI3K/AKT":    "#2A9D8F",  # teal
    "mTOR":        "#E9C46A",  # yellow
    "MAPK/ERK":    "#E76F51",  # coral
    "Stress/SAPK": "#9B5DE5",  # purple
    "Cell cycle":  "#4D908E",  # blue-green
    "Other":       "#ADB5BD",  # gray
}

# Flat ordered list: all proteins in circular order
ALL_PROTEINS: List[str] = [p for ps in PATHWAY_GROUPS.values() for p in ps]

def pathway_of(protein: str) -> str:
    for pw, ps in PATHWAY_GROUPS.items():
        if protein in ps:
            return pw
    return "Other"

# Short display names (single line, compact for circular layout)
_SHORT: Dict[str, str] = {
    "EGFR_pY1068": "EGFR\npY1068",   "EGFR_pY1173": "EGFR\npY1173",
    "EGFR_pY992":  "EGFR\npY992",    "AKT_pS473":   "AKT\npS473",
    "AKT_pT308":   "AKT\npT308",     "MAPK_pT202_Y204": "MAPK\npT202",
    "MEK1_pS217_S221": "MEK1",       "mTOR_pS2448": "mTOR",
    "S6_pS235_S236": "S6\npS235",    "S6_pS240_S244": "S6\npS240",
    "p70S6K_pT389": "p70S6K",        "4EBP1_pS65":  "4EBP1",
    "STAT3_pY705": "STAT3",          "p38_pT180_Y182": "p38",
    "JNK_pT183_pT185": "JNK",        "c-JUN_pS73":  "cJUN",
    "c-Raf_pS338": "cRaf",           "GSK3-alpha-beta_pS21_S9": "GSK3ab\npS21",
    "GSK3-alpha-beta_pS9": "GSK3ab\npS9", "PRAS40_pT246": "PRAS40",
    "PDK1_pS241":  "PDK1",           "AMPK_pT172":  "AMPK",
    "BAD_pS112":   "BAD",            "Rb_pS807_S811": "Rb",
    "HER2_pY1248": "HER2",           "c-Met_pY1235": "cMet",
    "p90RSK_pT359_S363": "p90RSK",   "CHK1_pS345":  "CHK1",
    "CHK2_pT68":   "CHK2",           "NF-kB-p65_pS536": "NFkB",
    "Src_pY416":   "Src\npY416",     "Src_pY527":   "Src\npY527",
    "p27_pT157":   "p27\npT157",     "p27_pT198":   "p27\npT198",
    "FOXO3a_pS318_S321": "FOXO3a",   "ER-alpha_pS118": "ERα",
    "ACC_pS79":    "ACC",            "PKC-alpha_pS657": "PKCα",
    "TAZ_pS89":    "TAZ",            "YAP_pS127":   "YAP",
    "YB-1_PS102":  "YB-1",
}

def short(name: str) -> str:
    return _SHORT.get(name, name.split("_")[0])


# ── Consensus edge computation ─────────────────────────────────────────────────

def load_consensus_edges(source: str, min_freq: float = 0.25) -> pd.DataFrame:
    """
    Aggregate per-condition edge lists (gate ≥ 0.25).
    Return edges appearing in ≥ min_freq fraction of conditions,
    with columns: source, target, freq, gate_mean, gate_max, beta_mean, sign.
    """
    cond_dir = os.path.join(RESULTS_DIR, source, "per_condition")
    conditions = sorted(os.listdir(cond_dir))
    all_edges: List[pd.DataFrame] = []

    for cond in conditions:
        ep = os.path.join(cond_dir, cond, "edges.csv")
        if os.path.exists(ep):
            df = pd.read_csv(ep)
            df["condition"] = cond
            all_edges.append(df)

    if not all_edges:
        return pd.DataFrame()

    combined = pd.concat(all_edges, ignore_index=True)
    n_cond = len(conditions)

    freq_df = (
        combined
        .groupby(["source", "target"])
        .agg(
            count    = ("gate",  "count"),
            gate_mean = ("gate", "mean"),
            gate_max  = ("gate", "max"),
            beta_mean = ("beta", "mean"),
            sign      = ("sign", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )
    freq_df["freq"] = freq_df["count"] / n_cond

    consensus = freq_df[freq_df["freq"] >= min_freq].copy()
    consensus = consensus.sort_values("gate_mean", ascending=False).reset_index(drop=True)
    return consensus


# ── Circular layout ────────────────────────────────────────────────────────────

def circular_layout(proteins: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Place proteins on a circle in the order given by ALL_PROTEINS
    (grouped by pathway). Proteins not in ALL_PROTEINS go at the end.
    """
    ordered = [p for p in ALL_PROTEINS if p in proteins]
    remaining = [p for p in proteins if p not in ordered]
    ordered += remaining

    N = len(ordered)
    pos = {}
    for i, p in enumerate(ordered):
        angle = 2 * np.pi * i / N - np.pi / 2   # start at top
        pos[p] = (np.cos(angle), np.sin(angle))
    return pos


# ── Drawing ────────────────────────────────────────────────────────────────────

# Edge class colors
C_COMMON  = "#6E6E6E"   # neutral gray
C_TEACHER = "#0072B2"   # colorblind-safe blue
C_STUDENT = "#D55E00"   # colorblind-safe orange/red
C_ACT_ALPHA = 0.72
C_INH_ALPHA = 0.72


def _edge_style(sign: str, etype: str) -> dict:
    """Return arrowprops dict for annotate()."""
    color = {"common": C_COMMON, "teacher": C_TEACHER, "student": C_STUDENT}[etype]
    if sign == "activation":
        arrowstyle = "->,head_length=0.25,head_width=0.15"
        ls = "solid"
    else:
        arrowstyle = "-[,widthB=0.3,lengthB=0.12"
        ls = "dashed"
    return dict(
        arrowstyle=arrowstyle,
        color=color,
        linewidth=1.0,
        alpha=C_ACT_ALPHA,
        linestyle=ls,
        connectionstyle="arc3,rad=0.15",
        shrinkA=18, shrinkB=18,
    )


def draw_publication_network(
    ax: plt.Axes,
    edges_t: pd.DataFrame,
    edges_s: pd.DataFrame,
    pos: Dict[str, Tuple[float, float]],
    title: str,
    show_pathway_arcs: bool = True,
    node_radius: float = 0.055,
    axis_limit: float = 1.65,
    pathway_arc_radius: float = 1.28,
    pathway_label_radius: float = 1.42,
    pathway_arc_lw: float = 6.0,
    pathway_arc_alpha: float = 0.35,
    pathway_label_fontsize: float = 7.5,
    edge_alpha: float = 0.65,
    edge_lw_base: float = 0.9,
    edge_lw_scale: float = 1.2,
    edge_rad: float = 0.18,
    edge_shrink: float = 20,
    node_fontsize: float = 6.5,
    node_fontweight: str = "bold",
    node_pad: float = 0.30,
    node_edge_lw: float = 0.9,
    node_path_effect_lw: float = 1.5,
    title_fontsize: float = 12,
    title_pad: float = 10,
):
    """
    Draw the overlay network on ax.
      edges_t / edges_s: consensus edge DataFrames with columns
        source, target, sign, gate_mean, freq
    """
    # Build lookup: (src, tgt) -> row as plain dict
    set_t = {(r.source, r.target): r.to_dict() for _, r in edges_t.iterrows()}
    set_s = {(r.source, r.target): r.to_dict() for _, r in edges_s.iterrows()}

    keys_t = set(set_t.keys())
    keys_s = set(set_s.keys())
    common      = keys_t & keys_s
    teach_only  = keys_t - keys_s
    stud_only   = keys_s - keys_t

    all_nodes = sorted(
        {n for k in (keys_t | keys_s) for n in k},
        key=lambda p: ALL_PROTEINS.index(p) if p in ALL_PROTEINS else 9999,
    )

    # ── draw edges (common last so they sit on top) ──
    for group, color, etype in [
        (teach_only, C_TEACHER, "teacher"),
        (stud_only,  C_STUDENT, "student"),
        (common,     C_COMMON,  "common"),
    ]:
        for key in group:
            src, tgt = key
            if src not in pos or tgt not in pos:
                continue
            # pick sign from whichever edge record is available
            row  = set_t.get(key) or set_s.get(key)
            sign = row["sign"] if row is not None else "activation"
            lw   = edge_lw_base + edge_lw_scale * float(row["gate_mean"] if row is not None else 0.3)
            if sign == "activation":
                arrowstyle = "->,head_length=0.22,head_width=0.13"
                ls = "solid"
            else:
                arrowstyle = "-|>,head_length=0.22,head_width=0.13"
                ls = "dashed"
            ax.annotate(
                "",
                xy=pos[tgt], xycoords="data",
                xytext=pos[src], textcoords="data",
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    color=color,
                    linewidth=lw,
                    alpha=edge_alpha,
                    linestyle=ls,
                    connectionstyle=f"arc3,rad={edge_rad}",
                    shrinkA=edge_shrink, shrinkB=edge_shrink,
                ),
                zorder=2,
            )

    # ── pathway arc backgrounds ──
    if show_pathway_arcs:
        for pw, proteins in PATHWAY_GROUPS.items():
            in_net = [p for p in proteins if p in all_nodes]
            if len(in_net) < 2:
                continue
            angles = sorted(
                [np.arctan2(pos[p][1], pos[p][0]) for p in in_net]
            )
            # draw a thick arc
            a_start = angles[0] - 0.08
            a_end   = angles[-1] + 0.08
            arc_angles = np.linspace(a_start, a_end, 80)
            r = pathway_arc_radius
            ax.plot(
                r * np.cos(arc_angles),
                r * np.sin(arc_angles),
                lw=pathway_arc_lw, color=PATHWAY_COLORS[pw], alpha=pathway_arc_alpha,
                solid_capstyle="round", zorder=1,
            )
            # pathway label at midpoint
            mid = (a_start + a_end) / 2
            lx, ly = pathway_label_radius * np.cos(mid), pathway_label_radius * np.sin(mid)
            ax.text(
                lx, ly, pw,
                ha="center", va="center",
                fontsize=pathway_label_fontsize, fontweight="bold",
                color=PATHWAY_COLORS[pw],
                rotation=np.degrees(mid) if -np.pi/2 < mid < np.pi/2 else np.degrees(mid) + 180,
                zorder=6,
            )

    # ── nodes ──
    for node in all_nodes:
        if node not in pos:
            continue
        x, y = pos[node]
        pw    = pathway_of(node)
        fc    = PATHWAY_COLORS.get(pw, "#ADB5BD")
        ax.text(
            x, y, short(node),
            ha="center", va="center",
            fontsize=node_fontsize, fontweight=node_fontweight, color="#1a1a1a",
            bbox=dict(
                boxstyle=f"round,pad={node_pad}",
                facecolor=fc, edgecolor="#444444",
                linewidth=node_edge_lw, alpha=0.95,
            ),
            zorder=5,
            path_effects=[pe.withStroke(linewidth=node_path_effect_lw, foreground="white")],
        )

    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold",
                     pad=title_pad, color="#222222")


def _legend_handles(n_common: int, n_teach: int, n_stud: int) -> list:
    return [
        mpatches.Patch(color=C_COMMON,  label=f"Common ({n_common})"),
        mpatches.Patch(color=C_TEACHER, label=f"Teacher-only ({n_teach})"),
        mpatches.Patch(color=C_STUDENT, label=f"Student-only ({n_stud})"),
        mlines.Line2D([0],[0], color="#555", lw=1.5, ls="solid",  label="Activation"),
        mlines.Line2D([0],[0], color="#555", lw=1.5, ls="dashed", label="Inhibition"),
    ]


def _save(fig: plt.Figure, path_no_ext: str, exts: tuple[str, ...] = (".png", ".pdf")):
    for ext in exts:
        fig.savefig(path_no_ext + ext, dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    ext_summary = exts[0] + "".join(f"/{ext}" for ext in exts[1:])
    print(f"  Saved: {os.path.basename(path_no_ext)}{ext_summary}")


# ── Main figures ───────────────────────────────────────────────────────────────

def plot_overlay(
    edges_t: pd.DataFrame,
    edges_s: pd.DataFrame,
    min_freq: float,
):
    """Single overlay figure: all nodes on circle, edges colored by class."""
    all_nodes = sorted(
        set(edges_t["source"]) | set(edges_t["target"]) |
        set(edges_s["source"]) | set(edges_s["target"]),
        key=lambda p: ALL_PROTEINS.index(p) if p in ALL_PROTEINS else 9999,
    )
    pos = circular_layout(all_nodes)

    set_t = set(zip(edges_t["source"], edges_t["target"]))
    set_s = set(zip(edges_s["source"], edges_s["target"]))
    common     = set_t & set_s
    teach_only = set_t - set_s
    stud_only  = set_s - set_t

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    draw_publication_network(
        ax, edges_t, edges_s, pos,
        title="",
        axis_limit=1.38,
        pathway_arc_radius=1.19,
        pathway_label_radius=1.31,
        pathway_arc_lw=4.2,
        pathway_arc_alpha=0.32,
        pathway_label_fontsize=5.2,
        edge_alpha=0.58,
        edge_lw_base=0.55,
        edge_lw_scale=0.9,
        edge_rad=0.16,
        edge_shrink=14,
        node_fontsize=5.9,
        node_fontweight="bold",
        node_pad=0.22,
        node_edge_lw=0.55,
        node_path_effect_lw=0.85,
    )

    fig.text(
        0.5, 0.982,
        (f"Consensus network | freq ≥ {int(min_freq*100)}% | "
         f"T={len(set_t)}, S={len(set_s)}, shared={len(common)}"),
        ha="center", va="top", fontsize=7.0, fontweight="bold", color="#222222",
    )

    handles = _legend_handles(len(common), len(teach_only), len(stud_only))
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, 0.014),
               handlelength=1.25, columnspacing=1.0)

    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.07, top=0.955)
    _save(fig, os.path.join(OUT_DIR, "network_consensus_overlay"),
          exts=(".png", ".pdf", ".svg"))


def plot_single(
    edges: pd.DataFrame,
    source: str,
    min_freq: float,
    pos: dict,
):
    """Single-model publication network."""
    # For single-model plot: use a dummy empty df for the other model
    empty = pd.DataFrame(columns=edges.columns)
    if source == "teacher":
        et, es = edges, empty
    else:
        et, es = empty, edges

    n = len(edges)
    n_act = (edges["sign"] == "activation").sum()
    n_inh = n - n_act

    fig, ax = plt.subplots(figsize=(14, 14))
    draw_publication_network(
        ax, et, es, pos,
        title=(f"{source.capitalize()} consensus network  "
               f"(freq ≥ {int(min_freq*100)}% of conditions)\n"
               f"{n} edges  ({n_act} activation, {n_inh} inhibition)"),
    )

    color = C_TEACHER if source == "teacher" else C_STUDENT
    handles = [
        mpatches.Patch(color=color, label=f"{source.capitalize()} ({n} edges)"),
        mlines.Line2D([0],[0], color="#555", lw=1.5, ls="solid",  label="Activation"),
        mlines.Line2D([0],[0], color="#555", lw=1.5, ls="dashed", label="Inhibition"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, frameon=True, framealpha=0.95,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, os.path.join(OUT_DIR, f"network_consensus_{source}"))


def plot_side_by_side(
    edges_t: pd.DataFrame,
    edges_s: pd.DataFrame,
    min_freq: float,
    pos: dict,
):
    """Two-panel side-by-side figure for direct comparison."""
    empty = pd.DataFrame(columns=edges_t.columns)

    set_t = set(zip(edges_t["source"], edges_t["target"]))
    set_s = set(zip(edges_s["source"], edges_s["target"]))
    common = set_t & set_s

    fig, axes = plt.subplots(1, 2, figsize=(28, 14))

    for ax, src, et, es, color in [
        (axes[0], "teacher", edges_t, empty, C_TEACHER),
        (axes[1], "student", empty,   edges_s, C_STUDENT),
    ]:
        n = len(edges_t) if src == "teacher" else len(edges_s)
        n_act = ((edges_t if src=="teacher" else edges_s)["sign"] == "activation").sum()
        draw_publication_network(
            ax, et, es, pos,
            title=(f"{src.capitalize()}  —  {n} consensus edges\n"
                   f"({n_act} activation, {n - n_act} inhibition)"),
        )

    fig.suptitle(
        f"Consensus signaling networks  (freq ≥ {int(min_freq*100)}% of conditions)\n"
        f"Shared edges: {len(common)} / Teacher: {len(set_t)} / Student: {len(set_s)}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    handles = [
        mpatches.Patch(color=C_TEACHER, label=f"Teacher ({len(set_t)} edges)"),
        mpatches.Patch(color=C_STUDENT, label=f"Student ({len(set_s)} edges)"),
        mlines.Line2D([0],[0], color="#555", lw=1.5, ls="solid",  label="Activation"),
        mlines.Line2D([0],[0], color="#555", lw=1.5, ls="dashed", label="Inhibition"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=11, frameon=True, framealpha=0.95,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, os.path.join(OUT_DIR, "network_consensus_side_by_side"))


# ── Main ────────────────────────────────────────────────────────────────────────

def main(min_freq: float = 0.25):
    print(f"Loading consensus edges (freq ≥ {min_freq}) …")
    edges_t = load_consensus_edges("teacher", min_freq)
    edges_s = load_consensus_edges("student", min_freq)

    print(f"  Teacher: {len(edges_t)} consensus edges")
    print(f"  Student: {len(edges_s)} consensus edges")

    # Save edge tables
    edges_t.to_csv(os.path.join(OUT_DIR, "consensus_edges_teacher.csv"), index=False)
    edges_s.to_csv(os.path.join(OUT_DIR, "consensus_edges_student.csv"), index=False)
    print(f"  Saved consensus edge tables.")

    # Shared layout: union of all nodes in pathway order
    all_nodes = sorted(
        set(edges_t["source"]) | set(edges_t["target"]) |
        set(edges_s["source"]) | set(edges_s["target"]),
        key=lambda p: ALL_PROTEINS.index(p) if p in ALL_PROTEINS else 9999,
    )
    pos = circular_layout(all_nodes)

    print("\nGenerating figures …")
    plot_overlay(edges_t, edges_s, min_freq)
    plot_single(edges_t, "teacher", min_freq, pos)
    plot_single(edges_s, "student", min_freq, pos)
    plot_side_by_side(edges_t, edges_s, min_freq, pos)

    # Summary stats
    set_t = set(zip(edges_t["source"], edges_t["target"]))
    set_s = set(zip(edges_s["source"], edges_s["target"]))
    print(f"\nSummary (freq ≥ {min_freq}):")
    print(f"  Teacher edges:    {len(set_t)}")
    print(f"  Student edges:    {len(set_s)}")
    print(f"  Common:           {len(set_t & set_s)}")
    print(f"  Teacher-only:     {len(set_t - set_s)}")
    print(f"  Student-only:     {len(set_s - set_t)}")
    print(f"\n  Outputs → {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--min_freq", type=float, default=0.25,
                   help="Minimum fraction of conditions an edge must appear in")
    args = p.parse_args()
    main(args.min_freq)
