#!/usr/bin/env python3
"""Plot a single A4-portrait consensus network with pathway-grouped elliptical layout.

Edges are coloured by whether they appear in the teacher network, the student
network, or both. All union edges are drawn; activation/inhibition are not
visually distinguished.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path("/private/tmp/hpn_dream_network_mpl")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.lines import Line2D

sys.path.insert(0, str(ROOT))
import darts_hill_discovery as dhd

OUT_DIR = ROOT / "results" / "nature_figure" / "supplementary"
DATA_DIR = ROOT / "results" / "darts_hill_shared_topology"

STUDENT_EDGES = DATA_DIR / "student_shared" / "edges_shared.csv"
TEACHER_EDGES = DATA_DIR / "teacher_shared" / "edges_shared.csv"

EDGE_CLASS_COLORS = {
    "common":       "#333333",
    "teacher-only": "#1F78B4",
    "student-only": "#D62728",
}

PATHWAY_ORDER = ["RTK", "PI3K/AKT", "mTOR", "MAPK/ERK", "Stress", "Cell cycle", "Other"]

PATHWAY_MAP = {
    "EGFR_pY1068": "RTK", "EGFR_pY1173": "RTK", "EGFR_pY992": "RTK",
    "HER2_pY1248": "RTK", "c-Met_pY1235": "RTK",
    "PDK1_pS241": "PI3K/AKT", "AKT_pS473": "PI3K/AKT", "AKT_pT308": "PI3K/AKT",
    "PRAS40_pT246": "PI3K/AKT", "GSK3-alpha-beta_pS21_S9": "PI3K/AKT",
    "GSK3-alpha-beta_pS9": "PI3K/AKT", "FOXO3a_pS318_S321": "PI3K/AKT",
    "ACC_pS79": "PI3K/AKT", "PKC-alpha_pS657": "PI3K/AKT",
    "mTOR_pS2448": "mTOR", "S6_pS235_S236": "mTOR", "S6_pS240_S244": "mTOR",
    "p70S6K_pT389": "mTOR", "4EBP1_pS65": "mTOR",
    "MAPK_pT202_Y204": "MAPK/ERK", "MEK1_pS217_S221": "MAPK/ERK",
    "c-Raf_pS338": "MAPK/ERK", "p90RSK_pT359_S363": "MAPK/ERK",
    "c-JUN_pS73": "MAPK/ERK",
    "AMPK_pT172": "Stress", "p38_pT180_Y182": "Stress",
    "JNK_pT183_pT185": "Stress", "NF-kB-p65_pS536": "Stress",
    "CHK1_pS345": "Stress", "CHK2_pT68": "Stress", "BAD_pS112": "Stress",
    "Rb_pS807_S811": "Cell cycle", "p27_pT157": "Cell cycle",
    "p27_pT198": "Cell cycle",
    "STAT3_pY705": "Other", "YB-1_PS102": "Other",
    "Src_pY416": "Other", "Src_pY527": "Other",
    "ER-alpha_pS118": "Other", "TAZ_pS89": "Other", "YAP_pS127": "Other",
}

PATHWAY_COLORS = {
    "RTK":        "#D97706",
    "PI3K/AKT":   "#0B8071",
    "mTOR":       "#B8860B",
    "MAPK/ERK":   "#C0392B",
    "Stress":     "#7C3AED",
    "Cell cycle": "#24746F",
    "Other":      "#8B95A2",
}


def node_pathway(node: str) -> str:
    return PATHWAY_MAP.get(node, "Other")


def node_color(node: str) -> str:
    return PATHWAY_COLORS.get(node_pathway(node), PATHWAY_COLORS["Other"])


def _remove_overlaps(pos: dict, min_dist: float, iterations: int = 300) -> dict:
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes], dtype=float)
    for _ in range(iterations):
        moved = False
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                diff = coords[a] - coords[b]
                dist = np.linalg.norm(diff)
                if 1e-8 < dist < min_dist:
                    push = (min_dist - dist) / 2.0 * (diff / dist)
                    coords[a] += push
                    coords[b] -= push
                    moved = True
        if not moved:
            break
    return {nodes[k]: coords[k] for k in range(len(nodes))}


def _get_layout(G: nx.Graph, seed: int = 42) -> dict:
    """Pathway-grouped elliptical layout.

    Nodes are distributed along an ellipse, grouped by pathway into angular
    sectors. A short force-directed refinement reduces local overlaps while
    keeping the global elliptical structure.
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    groups: dict[str, list[str]] = {pw: [] for pw in PATHWAY_ORDER}
    for n in nodes:
        groups[node_pathway(n)].append(n)
    for pw in groups:
        groups[pw].sort()

    pos = {}
    n_nodes = len(nodes)
    a, b = 4.2, 3.0  # ellipse radii
    base_angle = -np.pi / 2
    used = 0
    for pw in PATHWAY_ORDER:
        members = groups[pw]
        if not members:
            continue
        sector = 2 * np.pi * len(members) / n_nodes
        gap = 0.06 * sector
        sector -= gap
        start = base_angle + 2 * np.pi * used / n_nodes + gap / 2
        for i, n in enumerate(members):
            theta = start + sector * i / max(len(members) - 1, 1)
            pos[n] = np.array([a * np.cos(theta), b * np.sin(theta)])
        used += len(members)

    pos = nx.spring_layout(G, pos=pos, k=0.9 / np.sqrt(max(len(G), 1)),
                           iterations=30, seed=seed)
    return _remove_overlaps(pos, min_dist=0.32, iterations=200)


def edge_weight(beta: float, gate: float) -> float:
    return abs(float(beta)) + 0.15 * float(gate)


def build_union_graph(teacher_df: pd.DataFrame, student_df: pd.DataFrame) -> nx.DiGraph:
    teacher = {
        (r["source"], r["target"]): edge_weight(r["beta"], r["gate"])
        for _, r in teacher_df.iterrows()
    }
    student = {
        (r["source"], r["target"]): edge_weight(r["beta"], r["gate"])
        for _, r in student_df.iterrows()
    }

    G = nx.DiGraph()
    for e in set(teacher.keys()) | set(student.keys()):
        in_t = e in teacher
        in_s = e in student
        if in_t and in_s:
            cls = "common"
            wt = max(teacher[e], student[e])
        elif in_t:
            cls = "teacher-only"
            wt = teacher[e]
        else:
            cls = "student-only"
            wt = student[e]
        G.add_edge(e[0], e[1], class_=cls, weight=wt)
    return G


def draw_combined_network(G: nx.DiGraph, title: str, out_stem: str) -> None:
    if len(G.nodes()) == 0 or len(G.edges()) == 0:
        print("  ⚠ Empty graph — skip")
        return

    pos = _get_layout(G)

    n_common = sum(1 for _, _, d in G.edges(data=True) if d["class_"] == "common")
    n_teacher = sum(1 for _, _, d in G.edges(data=True) if d["class_"] == "teacher-only")
    n_student = sum(1 for _, _, d in G.edges(data=True) if d["class_"] == "student-only")

    fig, ax = plt.subplots(figsize=(8.27, 11.69))

    ws = [d["weight"] for _, _, d in G.edges(data=True)]
    w_min, w_max = min(ws), max(ws)
    w_range = max(w_max - w_min, 1e-8)

    # Edges: light -> heavy
    for u, v, d in sorted(G.edges(data=True), key=lambda x: x[2]["weight"]):
        cls = d["class_"]
        norm = (d["weight"] - w_min) / w_range
        lw = 0.5 + 1.6 * (norm ** 0.8)
        alpha = 0.35 + 0.40 * norm
        color = EDGE_CLASS_COLORS[cls]
        ax.annotate(
            "",
            xy=pos[v], xycoords="data",
            xytext=pos[u], textcoords="data",
            arrowprops=dict(
                arrowstyle="->,head_length=0.30,head_width=0.18",
                color=color, linewidth=lw, alpha=alpha,
                linestyle="solid",
                connectionstyle="arc3,rad=0.05",
                shrinkA=18, shrinkB=18,
            ),
        )

    # Nodes
    for node in G.nodes():
        x, y = pos[node]
        ax.text(
            x, y, dhd.short_name_multiline(node),
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor=node_color(node),
                edgecolor="#555555",
                linewidth=1.2,
                alpha=0.95,
            ),
            zorder=5,
        )

    ax.axis("off")
    fig.tight_layout(pad=1.5, rect=[0, 0.04, 1, 0.95])

    # Title centered at top
    fig.text(
        0.5, 0.97, title,
        fontsize=15, fontweight="bold",
        ha="center", va="top", color="#222222",
    )

    # Legend at lower right (only edge classes)
    legend_elements = [
        Line2D([0], [0], color=EDGE_CLASS_COLORS["common"], lw=2.2,
               label=f"Common ({n_common})"),
        Line2D([0], [0], color=EDGE_CLASS_COLORS["teacher-only"], lw=2.2,
               label=f"Teacher-only ({n_teacher})"),
        Line2D([0], [0], color=EDGE_CLASS_COLORS["student-only"], lw=2.2,
               label=f"Student-only ({n_student})"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        ncol=1,
        fontsize=9.5,
        frameon=True,
        framealpha=0.95,
        edgecolor="none",
        fancybox=True,
        handlelength=2.0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{out_stem}.{ext}", facecolor="white", bbox_inches="tight")
    print(f"  ✓ Saved: {out_stem}.pdf / .png / .svg")
    plt.close(fig)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
    })

    teacher_df = pd.read_csv(TEACHER_EDGES)
    student_df = pd.read_csv(STUDENT_EDGES)

    print(f"Teacher edges: {len(teacher_df)}, Student edges: {len(student_df)}")

    G_union = build_union_graph(teacher_df, student_df)
    print(f"Combined graph: {len(G_union.edges())} edges, {len(G_union.nodes())} nodes")

    draw_combined_network(
        G_union,
        "HPN-DREAM shared-topology consensus network",
        "hpn_dream_consensus_networks",
    )


if __name__ == "__main__":
    main()
