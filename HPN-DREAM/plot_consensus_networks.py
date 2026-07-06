#!/usr/bin/env python3
"""Plot teacher and student consensus networks with edge-class coloring."""

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
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, str(ROOT))
import darts_hill_discovery as dhd

OUT_DIR = ROOT / "results" / "nature_figure" / "supplementary"
DATA_DIR = ROOT / "results" / "darts_hill_shared_topology"

STUDENT_EDGES = DATA_DIR / "student_shared" / "edges_shared.csv"
TEACHER_EDGES = DATA_DIR / "teacher_shared" / "edges_shared.csv"

# ── Nature-style palette ──────────────────────────────────────────────
PATHWAY_COLORS_NATURE = {
    "RTK":        "#D97706",
    "PI3K/AKT":   "#0B8071",
    "mTOR":       "#B8860B",
    "MAPK/ERK":   "#C0392B",
    "Stress":     "#7C3AED",
    "Cell cycle": "#24746F",
    "Other":      "#8B95A2",
}

PATHWAY_MAP = {
    "EGFR_pY1068": "RTK", "EGFR_pY1173": "RTK", "EGFR_pY992": "RTK",
    "HER2_pY1248": "RTK", "c-Met_pY1235": "RTK",
    "AKT_pS473": "PI3K/AKT", "AKT_pT308": "PI3K/AKT",
    "PDK1_pS241": "PI3K/AKT", "mTOR_pS2448": "mTOR",
    "PRAS40_pT246": "PI3K/AKT", "GSK3-alpha-beta_pS21_S9": "PI3K/AKT",
    "GSK3-alpha-beta_pS9": "PI3K/AKT", "AMPK_pT172": "PI3K/AKT",
    "S6_pS235_S236": "mTOR", "S6_pS240_S244": "mTOR",
    "p70S6K_pT389": "mTOR", "4EBP1_pS65": "mTOR",
    "ACC_pS79": "PI3K/AKT", "PKC-alpha_pS657": "PI3K/AKT",
    "MAPK_pT202_Y204": "MAPK/ERK", "MEK1_pS217_S221": "MAPK/ERK",
    "c-Raf_pS338": "MAPK/ERK", "p90RSK_pT359_S363": "MAPK/ERK",
    "p38_pT180_Y182": "Stress", "JNK_pT183_pT185": "Stress",
    "c-JUN_pS73": "Stress", "NF-kB-p65_pS536": "Stress",
    "CHK1_pS345": "Stress", "CHK2_pT68": "Stress",
    "Rb_pS807_S811": "Cell cycle", "p27_pT157": "Cell cycle",
    "p27_pT198": "Cell cycle",
    "STAT3_pY705": "Other", "YB-1_PS102": "Other",
    "Src_pY416": "Other", "Src_pY527": "Other",
    "BAD_pS112": "Other", "FOXO3a_pS318_S321": "Other",
    "ER-alpha_pS118": "Other", "TAZ_pS89": "Other", "YAP_pS127": "Other",
}

# Edge classification colours
EDGE_CLASS_COLORS = {
    "common":      "#333333",  # both teacher and student have it
    "teacher-only": "#1F78B4",  # only teacher
    "student-only": "#D62728",  # only student
}

PATHWAY_LAYERS = ["RTK", "PI3K/AKT", "MAPK/ERK", "mTOR", "Stress", "Cell cycle", "Other"]


def node_pathway(node: str) -> str:
    return PATHWAY_MAP.get(node, "Other")


def node_color(node: str) -> str:
    return PATHWAY_COLORS_NATURE.get(node_pathway(node), PATHWAY_COLORS_NATURE["Other"])


def spring_layout_from_graph(G: nx.DiGraph) -> dict:
    """Spring layout using actual edges, scaled to a fixed canvas."""
    pos = nx.spring_layout(G, k=1.8 / np.sqrt(max(len(G), 1)),
                           iterations=120, seed=42)
    # Remove overlaps
    node_list = list(pos.keys())
    coords = np.array([pos[n] for n in node_list], dtype=float)
    min_dist = 0.45
    for _ in range(300):
        moved = False
        for a in range(len(node_list)):
            for b in range(a + 1, len(node_list)):
                diff = coords[a] - coords[b]
                dist = np.linalg.norm(diff)
                if 1e-8 < dist < min_dist:
                    push = (min_dist - dist) / 2.0 * (diff / dist)
                    coords[a] += push
                    coords[b] -= push
                    moved = True
        if not moved:
            break
    for i, n in enumerate(node_list):
        pos[n] = coords[i]
    # Scale to canvas ~ 6 units
    coords = np.array(list(pos.values()))
    span = max(coords[:, 0].ptp(), coords[:, 1].ptp())
    scale = 6.0 / max(span, 1.0)
    for k in pos:
        pos[k] = pos[k] * scale
    return pos


def draw_nature_node(ax, x, y, node: str) -> None:
    """Nature-style node: white fill, pathway-coloured border."""
    label = dhd.short_name_multiline(node)
    color = node_color(node)
    lines = label.split("\n")
    max_len = max(len(l) for l in lines)
    w = 0.22 + 0.07 * max_len
    h = 0.15 + 0.11 * len(lines)

    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor="white", edgecolor=color, linewidth=1.6, zorder=4,
    )
    ax.add_patch(patch)
    fs = 7.5 if max_len <= 5 else 6.0
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#1F2933",
            linespacing=0.88, zorder=5)


def draw_classified_edge(ax, p_src, p_tgt, sign: str, lw: float, alpha: float,
                         cls: str) -> None:
    """Edge coloured by common/teacher-only/student-only; style for activation/inhibition."""
    color = EDGE_CLASS_COLORS[cls]
    ls = "solid" if sign == "activation" else (0, (3.0, 1.5))
    arrow = "->" if sign == "activation" else "-|>"
    ax.annotate(
        "", xy=p_tgt, xycoords="data", xytext=p_src, textcoords="data",
        arrowprops=dict(
            arrowstyle=f"{arrow},head_length=0.35,head_width=0.25",
            color=color, linewidth=lw, alpha=alpha,
            linestyle=ls, connectionstyle="arc3,rad=0.10",
            shrinkA=18, shrinkB=18,
        ),
        zorder=2,
    )


def build_graph(edge_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, r in edge_df.iterrows():
        G.add_edge(r["source"], r["target"],
                   gate=r["gate"], beta=r["beta"], sign=r["sign"])
    return G


def normalize_weights(edges_data: list) -> tuple[float, float, float]:
    if edges_data:
        ws = [e[2] for e in edges_data]
        return min(ws), max(ws), max(max(ws) - min(ws), 1e-8)
    return 1.0, 1.0, 1.0


def edge_weight(d: dict) -> float:
    return abs(float(d.get("beta", 0.0))) + 0.1 * float(d.get("gate", 0.0))


def plot_network_on_axis(ax: plt.Axes, edge_df: pd.DataFrame,
                         pos: dict, nodes: set[str],
                         edge_class_map: dict[tuple, str],
                         title: str, top_n: int = 80) -> None:
    """Draw one network panel; edge colours depend on common/unique classification.
    Only the top_n highest-weight edges are drawn for readability."""
    G = build_graph(edge_df)

    unique_key = "teacher-only" if "Teacher" in title else "student-only"

    edges_data = []
    for u, v, d in G.edges(data=True):
        cls = edge_class_map.get((u, v), "common")
        edges_data.append((u, v, edge_weight(d), d.get("sign", "activation"), cls))

    # Keep top edges by weight for visibility
    edges_data = sorted(edges_data, key=lambda x: x[2], reverse=True)[:top_n]

    w_min, w_max, w_range = normalize_weights(edges_data)

    # Draw edges light -> heavy
    for u, v, w, sign, cls in sorted(edges_data, key=lambda x: x[2]):
        norm = (w - w_min) / w_range
        lw = 1.0 + 3.2 * (norm ** 0.8)
        alpha = 0.60 + 0.38 * norm
        draw_classified_edge(ax, pos[u], pos[v], sign, lw, alpha, cls)

    # Draw nodes
    for node in nodes:
        draw_nature_node(ax, pos[node][0], pos[node][1], node)

    # Title + stats
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color="#1F2933")
    ax.text(0.97, 0.97, f"top {len(edges_data)} of {len(G.edges())} edges",
            transform=ax.transAxes, fontsize=9.0, fontweight="bold",
            ha="right", va="top", color="#667085")

    ax.axis("off")


def plot_combined_figure(teacher_df: pd.DataFrame, student_df: pd.DataFrame) -> None:
    """Side-by-side teacher/student networks with edge-class colours."""
    teacher_edges = set(zip(teacher_df["source"], teacher_df["target"]))
    student_edges = set(zip(student_df["source"], student_df["target"]))
    common_edges = teacher_edges & student_edges

    edge_class_map: dict[tuple, str] = {}
    for e in teacher_edges | student_edges:
        if e in common_edges:
            edge_class_map[e] = "common"
        elif e in teacher_edges:
            edge_class_map[e] = "teacher-only"
        else:
            edge_class_map[e] = "student-only"

    nodes = sorted(set(teacher_df["source"]) | set(teacher_df["target"]) |
                   set(student_df["source"]) | set(student_df["target"]))
    # Build union graph for layout (so spring uses all edges)
    union_G = nx.DiGraph()
    for df in (teacher_df, student_df):
        for _, r in df.iterrows():
            union_G.add_edge(r["source"], r["target"])
    pos = spring_layout_from_graph(union_G)

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(24, 12))

    plot_network_on_axis(ax_t, teacher_df, pos, set(nodes), edge_class_map,
                         "Teacher consensus network")
    plot_network_on_axis(ax_s, student_df, pos, set(nodes), edge_class_map,
                         "Student consensus network")

    # Shared legend
    legend_elements = [
        Patch(facecolor="white", edgecolor=EDGE_CLASS_COLORS["common"], lw=2.0,
              label=f"Common ({len(common_edges)})"),
        Patch(facecolor="white", edgecolor=EDGE_CLASS_COLORS["teacher-only"], lw=2.0,
              label=f"Teacher-only ({len(teacher_edges - common_edges)})"),
        Patch(facecolor="white", edgecolor=EDGE_CLASS_COLORS["student-only"], lw=2.0,
              label=f"Student-only ({len(student_edges - common_edges)})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=9.0,
               frameon=True, framealpha=0.95, edgecolor="#D8DDE6",
               fancybox=True, handlelength=2.0, borderpad=0.5)

    fig.suptitle("HPN-DREAM shared-topology consensus networks",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"hpn_dream_consensus_networks.{ext}",
                    facecolor="white", bbox_inches="tight")
    print(f"  ✓ Saved: hpn_dream_consensus_networks.pdf / .png / .svg")
    plt.close(fig)


def plot_single_network(edge_df: pd.DataFrame, title: str, stem: str,
                        top_n: int = 120) -> None:
    """Single-network standalone figure (activation/inhibition colours)."""
    G = build_graph(edge_df)
    if len(G) == 0:
        print(f"  ⚠ Empty graph — skip: {title}")
        return

    pos = spring_layout_from_graph(G)

    edges_data = [(u, v, edge_weight(d), d.get("sign", "activation"))
                  for u, v, d in G.edges(data=True)]
    edges_data = sorted(edges_data, key=lambda x: x[2], reverse=True)[:top_n]

    n_act = sum(1 for _, _, _, sign in edges_data if sign == "activation")
    n_inh = len(edges_data) - n_act
    w_min, w_max, w_range = normalize_weights(edges_data)

    fig, ax = plt.subplots(figsize=(20, 20))

    for u, v, w, sign in sorted(edges_data, key=lambda x: x[2]):
        norm = (w - w_min) / w_range
        lw = 1.0 + 3.2 * (norm ** 0.8)
        alpha = 0.60 + 0.38 * norm
        color = "#1B9E77" if sign == "activation" else "#7B3294"
        ls = "solid" if sign == "activation" else (0, (3.0, 1.5))
        arrow = "->" if sign == "activation" else "-|>"
        ax.annotate(
            "", xy=pos[v], xycoords="data", xytext=pos[u], textcoords="data",
            arrowprops=dict(
                arrowstyle=f"{arrow},head_length=0.45,head_width=0.30",
                color=color, linewidth=lw, alpha=alpha,
                linestyle=ls, connectionstyle="arc3,rad=0.10",
                shrinkA=22, shrinkB=22,
            ),
            zorder=2,
        )

    for node in G.nodes():
        draw_nature_node(ax, pos[node][0], pos[node][1], node)

    ax.text(0.97, 0.97, f"top {len(edges_data)} of {len(G.edges())} edges",
            transform=ax.transAxes, fontsize=10.0, color="#667085",
            ha="right", va="top")

    present_pws = sorted({node_pathway(n) for n in G.nodes()},
                         key=lambda pw: PATHWAY_LAYERS.index(pw) if pw in PATHWAY_LAYERS else 99)
    legend_elements = [
        Patch(facecolor="white", edgecolor=PATHWAY_COLORS_NATURE[pw], lw=2.4, label=pw)
        for pw in present_pws
    ] + [
        Line2D([0], [0], color="#1B9E77", lw=3.0, label=f"Activation ({n_act})"),
        Line2D([0], [0], color="#7B3294", lw=3.0, ls=(0, (3, 1.5)), label=f"Inhibition ({n_inh})"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.03), ncol=min(len(legend_elements), 5),
              fontsize=9.0, frameon=True, framealpha=0.95,
              edgecolor="#D8DDE6", fancybox=True, handlelength=2.2)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=18, color="#1F2933", loc="left")
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", facecolor="white", bbox_inches="tight")
    print(f"  ✓ Saved: {stem}.pdf / .png / .svg")
    plt.close(fig)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.linewidth": 1.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
    })

    student_df = pd.read_csv(STUDENT_EDGES)
    teacher_df = pd.read_csv(TEACHER_EDGES)

    print(f"Teacher: {len(teacher_df)} edges, {len(set(teacher_df['source']) | set(teacher_df['target']))} nodes")
    print(f"Student: {len(student_df)} edges, {len(set(student_df['source']) | set(student_df['target']))} nodes")

    print("Building combined edge-class consensus networks...")
    plot_combined_figure(teacher_df, student_df)

    print("Building standalone teacher network...")
    plot_single_network(teacher_df, "Teacher consensus network",
                        "hpn_dream_teacher_consensus_network")

    print("Building standalone student network...")
    plot_single_network(student_df, "Student consensus network",
                        "hpn_dream_student_consensus_network")

    print("Done.")


if __name__ == "__main__":
    main()
