#!/usr/bin/env python3
"""Create a standalone HPN-DREAM network/equation discovery figure.

Uses shared-topology results: a consensus network G shared across all 36
conditions, with condition-specific edge strengths beta.
Includes circular network topology panels in Nature style.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path("/private/tmp/hpn_dream_nature_mpl")
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
from matplotlib.ticker import MaxNLocator

import make_nature_figure as base


OUT_DIR = ROOT / "results" / "nature_figure"
PANEL_DIR = OUT_DIR / "panels"

SHARED_TEACHER_EDGES = ROOT / "results" / "darts_hill_shared_topology" / "teacher_shared" / "edges_shared.csv"
SHARED_STUDENT_EDGES = ROOT / "results" / "darts_hill_shared_topology" / "student_shared" / "edges_shared.csv"
SHARED_TEACHER_GATES = ROOT / "results" / "darts_hill_shared_topology" / "teacher_shared" / "GATES_shared.csv"
SHARED_STUDENT_GATES = ROOT / "results" / "darts_hill_shared_topology" / "student_shared" / "GATES_shared.csv"
SHARED_STUDENT_BETA_DIR = ROOT / "results" / "darts_hill_shared_topology" / "student_shared" / "per_condition"

PATHWAY_GROUPS = {
    "RTK": [
        "EGFR_pY1068", "EGFR_pY1173", "EGFR_pY992",
        "HER2_pY1248", "c-Met_pY1235",
    ],
    "PI3K-AKT": [
        "PDK1_pS241", "AKT_pT308", "AKT_pS473", "PRAS40_pT246",
        "FOXO3a_pS318_S321", "GSK3-alpha-beta_pS21_S9", "GSK3-alpha-beta_pS9",
    ],
    "MAPK-ERK": [
        "c-Raf_pS338", "MEK1_pS217_S221", "MAPK_pT202_Y204",
        "p90RSK_pT359_S363", "c-JUN_pS73",
    ],
    "mTOR": [
        "mTOR_pS2448", "p70S6K_pT389",
        "S6_pS235_S236", "S6_pS240_S244", "4EBP1_pS65",
    ],
    "Stress": [
        "AMPK_pT172", "ACC_pS79",
        "p38_pT180_Y182", "JNK_pT183_pT185", "BAD_pS112",
    ],
    "Cell cycle": [
        "CHK1_pS345", "CHK2_pT68",
        "p27_pT157", "p27_pT198", "Rb_pS807_S811",
    ],
    "Context": [
        "Src_pY416", "Src_pY527", "PKC-alpha_pS657",
        "NF-kB-p65_pS536", "ER-alpha_pS118",
        "YAP_pS127", "TAZ_pS89", "YB-1_PS102", "STAT3_pY705",
    ],
}

PATHWAY_COLORS = {
    "RTK": "#D97706",
    "PI3K-AKT": "#008B7A",
    "mTOR": "#C58B00",
    "MAPK-ERK": "#D54E2F",
    "Stress": "#7C3AED",
    "Cell cycle": "#24746F",
    "Context": "#6B7280",
}

PATHWAY_ORDER = ["RTK", "PI3K-AKT", "MAPK-ERK", "mTOR", "Stress", "Cell cycle", "Context"]

ACT_COLOR = "#1B9E77"
INH_COLOR = "#7B3294"


# ── Network topology helpers ──────────────────────────────────────────

# Signaling cascade layers (upstream → downstream)
SIGNALING_LAYERS = [
    # Layer 0: Membrane / receptors
    ["EGFR_pY1068", "EGFR_pY1173", "EGFR_pY992", "HER2_pY1248", "c-Met_pY1235"],
    # Layer 1: Membrane-associated / adaptors
    ["Src_pY416", "Src_pY527", "PDK1_pS241", "PKC-alpha_pS657", "c-Raf_pS338"],
    # Layer 2: Early kinases
    ["AMPK_pT172", "AKT_pS473", "AKT_pT308", "MEK1_pS217_S221",
     "p38_pT180_Y182", "JNK_pT183_pT185"],
    # Layer 3: Late kinases
    ["MAPK_pT202_Y204", "mTOR_pS2448", "GSK3-alpha-beta_pS21_S9",
     "GSK3-alpha-beta_pS9", "PRAS40_pT246"],
    # Layer 4: Downstream effectors (split into two for less crowding)
    ["p70S6K_pT389", "p90RSK_pT359_S363", "S6_pS235_S236",
     "S6_pS240_S244", "4EBP1_pS65"],
    # Layer 4b: More effectors
    ["ACC_pS79", "BAD_pS112", "FOXO3a_pS318_S321", "c-JUN_pS73"],
    # Layer 5: Cell cycle / DNA damage
    ["CHK1_pS345", "CHK2_pT68", "Rb_pS807_S811", "p27_pT157", "p27_pT198"],
    # Layer 6: Transcription / nuclear
    ["STAT3_pY705", "YB-1_PS102", "NF-kB-p65_pS536",
     "ER-alpha_pS118", "TAZ_pS89", "YAP_pS127"],
]


def node_pathway(node: str) -> str:
    for pw, members in PATHWAY_GROUPS.items():
        if node in members:
            return pw
    return "Context"


def node_layer(node: str) -> int:
    for i, layer in enumerate(SIGNALING_LAYERS):
        if node in layer:
            return i
    # Assign unlisted nodes heuristically by pathway
    pw = node_pathway(node)
    if pw == "RTK":
        return 0
    elif pw in ("Stress",):
        return 2
    elif pw in ("PI3K-AKT", "MAPK-ERK"):
        return 3
    elif pw == "mTOR":
        return 4
    elif pw == "Cell cycle":
        return 6
    return 3  # default mid


def build_graph(edge_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, r in edge_df.iterrows():
        G.add_edge(r["source"], r["target"],
                   gate=r["gate"], beta=r["beta"], sign=r["sign"])
    return G


def _remove_overlaps(pos: dict, min_dist: float, iterations: int = 300) -> dict:
    """Iteratively push apart nodes that are too close."""
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


def layered_rect_layout(G: nx.DiGraph) -> dict:
    """Upstream→downstream rectangular layout.
    
    Nodes are placed on a grid where the y-axis represents signaling depth
    (membrane → nucleus) and x-axis spreads nodes within each layer,
    grouped by pathway.
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    # Group nodes by layer
    layers: dict[int, list[str]] = {}
    for n in nodes:
        ly = node_layer(n)
        layers.setdefault(ly, []).append(n)

    sorted_layers = sorted(layers.keys())
    n_layers = len(sorted_layers)

    pos = {}
    x_total = 6.0
    y_total = 5.0
    y_start = 2.5
    y_step = y_total / max(n_layers - 1, 1)

    for li, ly in enumerate(sorted_layers):
        layer_nodes = layers[ly]
        layer_nodes.sort(key=lambda n: (PATHWAY_ORDER.index(node_pathway(n))
                                         if node_pathway(n) in PATHWAY_ORDER else 99))
        n_in_layer = len(layer_nodes)
        y = y_start - li * y_step
        x_step = x_total / max(n_in_layer + 1, 1)
        for i, n in enumerate(layer_nodes):
            x = -x_total / 2 + (i + 1) * x_step
            pos[n] = np.array([x, y])

    return _remove_overlaps(pos, min_dist=0.32, iterations=100)


def draw_network_node(ax, x, y, node: str) -> None:
    """Coloured pathway node."""
    label = base.short_name(node)
    color = PATHWAY_COLORS.get(node_pathway(node), PATHWAY_COLORS["Context"])
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=6.2, fontweight="bold", color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=color,
            edgecolor="#555555",
            linewidth=1.0,
            alpha=0.95,
        ),
        zorder=5,
    )


def draw_network_edge(ax, p_src, p_tgt, sign: str, lw: float, alpha: float,
                      same_layer: bool = False) -> None:
    """Edge with arrow; larger arc for same-layer edges to route around nodes."""
    color = ACT_COLOR if sign == "activation" else INH_COLOR
    ls = "solid" if sign == "activation" else (0, (3.5, 1.8))
    arrow = "->" if sign == "activation" else "-|>"
    rad = 0.30 if same_layer else 0.32
    ax.annotate(
        "", xy=p_tgt, xycoords="data", xytext=p_src, textcoords="data",
        arrowprops=dict(
            arrowstyle=f"{arrow},head_length=0.26,head_width=0.16",
            color=color, linewidth=lw, alpha=alpha,
            linestyle=ls, connectionstyle=f"arc3,rad={rad}",
            shrinkA=18, shrinkB=18,
        ),
        zorder=2,
    )


def plot_network_on_axis(ax: plt.Axes, edge_df: pd.DataFrame, title: str,
                         show_layer_labels: bool = True) -> None:
    """Draw consensus network with upstream→downstream layered layout."""
    G = build_graph(edge_df)
    if len(G) == 0:
        ax.axis("off")
        return
    pos = layered_rect_layout(G)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-2.8, 2.8)

    n_act = sum(1 for _, _, d in G.edges(data=True) if d.get("sign") == "activation")
    n_inh = len(G.edges()) - n_act

    edges_data = [(u, v, abs(float(d.get("beta", 0))) + 0.1 * float(d.get("gate", 0)),
                   d.get("sign", "activation"))
                  for u, v, d in G.edges(data=True)]
    if edges_data:
        ws = [e[2] for e in edges_data]
        w_min, w_max = min(ws), max(ws)
        w_range = max(w_max - w_min, 1e-8)
    else:
        w_min = w_max = w_range = 1.0

    for u, v, w, sign in sorted(edges_data, key=lambda x: x[2]):
        norm = (w - w_min) / w_range
        lw = 0.4 + 1.6 * (norm ** 0.7)
        alpha = 0.30 + 0.50 * norm
        same = node_layer(u) == node_layer(v)
        draw_network_edge(ax, pos[u], pos[v], sign, lw, alpha, same_layer=same)

    for node in G.nodes():
        x, y = pos[node]
        draw_network_node(ax, x, y, node)

    # Layer labels on left side — only on teacher panel
    if show_layer_labels:
        layer_names = {0: "Membrane", 1: "Adaptors", 2: "Early\nkinases",
                       3: "Late\nkinases", 4: "Effectors", 5: "Metabolic\n& stress",
                       6: "Cell\ncycle", 7: "Nuclear"}
        for n in G.nodes():
            ly = node_layer(n)
            if ly in layer_names:
                y_data = pos[n][1]
                y_ax = (y_data + 2.8) / 5.6  # ylim is [-2.8, 2.8]
                ax.text(0.97, y_ax, layer_names[ly],
                        transform=ax.transAxes, ha="left", va="center",
                        fontsize=5.5, fontweight="bold",
                        color=base.COLORS["mid"], linespacing=0.9)
                del layer_names[ly]

    # Legend
    legend_elements = [
        Line2D([0], [0], color=ACT_COLOR, lw=1.8, label=f"Activation ({n_act})"),
        Line2D([0], [0], color=INH_COLOR, lw=1.8, ls=(0, (3.5, 1.8)),
               label=f"Inhibition ({n_inh})"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=5.8,
              frameon=True, framealpha=0.90, edgecolor=base.COLORS["grid"],
              fancybox=True, handlelength=1.4, borderpad=0.4)

    ax.text(0.5, 0.99, title, transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=base.COLORS["text"],
            ha="center", va="bottom")
    ax.axis("off")


# ── Gate heatmaps ─────────────────────────────────────────────────────

def plot_gate_heatmaps(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Plot teacher and student gate matrices side-by-side, ordered by pathway."""
    teacher_gates = pd.read_csv(SHARED_TEACHER_GATES, index_col=0)
    student_gates = pd.read_csv(SHARED_STUDENT_GATES, index_col=0)

    ordered = []
    boundaries = []
    for pathway in ["RTK", "PI3K-AKT", "MAPK-ERK", "mTOR", "Stress", "Cell cycle", "Context"]:
        members = PATHWAY_GROUPS[pathway]
        present = [p for p in members if p in teacher_gates.columns]
        if present:
            s = len(ordered)
            ordered.extend(present)
            boundaries.append((s, s + len(present), pathway))

    teacher_mat = teacher_gates.loc[ordered, ordered].values
    student_mat = student_gates.loc[ordered, ordered].values
    P = len(ordered)

    short_labels = [base.short_name(p).replace("\n", " ") for p in ordered]

    for i, (ax, mat, label) in enumerate(zip(
        axes, [teacher_mat, student_mat], ["Teacher", "Student"],
    )):
        n_edges = int((mat > 0.5).sum())
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal",
                       interpolation="nearest", rasterized=True)
        ax.set_xticks(range(P))
        ax.set_yticks(range(P))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=2.8)
        ax.set_yticklabels(short_labels, fontsize=2.8)
        ax.tick_params(axis="both", which="major", width=0.3, length=1.5, pad=0.3)

        for s, e, name in boundaries:
            color = PATHWAY_COLORS[name]
            lw = 1.0
            ax.plot([s - 0.5, e - 0.5], [s - 0.5, s - 0.5], color=color, lw=lw, solid_capstyle="butt")
            ax.plot([s - 0.5, s - 0.5], [s - 0.5, e - 0.5], color=color, lw=lw, solid_capstyle="butt")
            ax.plot([e - 0.5, e - 0.5], [s - 0.5, e - 0.5], color=color, lw=lw, solid_capstyle="butt")
            ax.plot([s - 0.5, e - 0.5], [e - 0.5, e - 0.5], color=color, lw=lw, solid_capstyle="butt")

        ax.text(
            0.97, 0.97, f"{label}\n{n_edges} edges",
            transform=ax.transAxes, fontsize=5.5, fontweight="bold",
            va="top", ha="right", color="white",
            bbox=dict(facecolor="black", alpha=0.40, edgecolor="none", pad=0.25),
        )

        if i == 1:
            # Color bar on the right of student heatmap
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=30)
            cbar.set_label("Gate prob.", rotation=90, labelpad=2.0, fontsize=6.0, fontweight="bold")
            cbar.ax.tick_params(labelsize=5.0, width=0.4, length=2)
            cbar.outline.set_linewidth(0.6)


# ── Sensitivity plot ──────────────────────────────────────────────────

def plot_sensitivity(ax: plt.Axes, edges_df: pd.DataFrame) -> None:
    cond_dirs = sorted([p for p in SHARED_STUDENT_BETA_DIR.iterdir() if p.is_dir()])

    top_edges = edges_df.sort_values("gate", ascending=False).head(10)
    rows = []
    for cond_dir in cond_dirs:
        edge_path = cond_dir / "edges.csv"
        if not edge_path.exists():
            continue
        cond_df = pd.read_csv(edge_path).set_index(["source", "target"])
        for _, row in top_edges.iterrows():
            key = (row["source"], row["target"])
            if key in cond_df.index:
                val = cond_df.loc[key]
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[0]
                beta = float(val["beta"])
                sign = 1.0 if str(val["sign"]) == "activation" else -1.0
            else:
                beta = 0.0
                sign = 1.0
            rows.append({
                "edge": f"{row['source']}\u2192{row['target']}",
                "signed_beta": sign * beta,
                "gate": float(row["gate"]),
            })
    rdf = pd.DataFrame(rows)

    # de-mean per edge -> impact
    means = rdf.groupby("edge")["signed_beta"].transform("mean")
    rdf["impact"] = rdf["signed_beta"] - means

    # sort edges by mean |signed_beta| (ascending for bottom-up display)
    edge_order = (
        rdf.groupby("edge")["signed_beta"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values()
        .index.tolist()
    )
    edge_to_y = {edge: idx for idx, edge in enumerate(edge_order)}

    rng = np.random.default_rng(7)
    y = np.array([edge_to_y[e] for e in rdf["edge"]], dtype=float)
    y += rng.normal(0, 0.08, size=len(y))

    cmap = mpl.colors.LinearSegmentedColormap.from_list("gate", ["#1E88E5", "#7E57C2", "#D81B60"])
    scatter = ax.scatter(
        rdf["impact"], y,
        c=rdf["gate"], cmap=cmap,
        vmin=0.0, vmax=max(0.6, float(rdf["gate"].max())),
        s=18, alpha=0.88, linewidths=0, zorder=3,
    )

    # mean marker per edge
    for edge, y_pos in edge_to_y.items():
        mean_val = rdf.loc[rdf["edge"] == edge, "signed_beta"].mean()
        ax.scatter(
            [0], [y_pos], marker="D", s=28,
            facecolor="#222222", edgecolor="white", linewidths=0.6,
            zorder=4,
        )

    ax.axvline(0, color="#7A7A7A", lw=1.0, alpha=0.85, zorder=2)

    # y-axis labels: short names
    short_map = {}
    for _, row in top_edges.iterrows():
        src = base.short_name(row["source"]).replace("\n", " ")
        tgt = base.short_name(row["target"]).replace("\n", " ")
        short_map[f"{row['source']}\u2192{row['target']}"] = f"{src} \u2192 {tgt}"
    ax.set_yticks(
        np.arange(len(edge_order)),
        [short_map.get(e, e) for e in edge_order],
    )
    ax.tick_params(axis="y", labelsize=5.2)

    ax.set_xlabel("Sensitivity impact across conditions", labelpad=1.0)
    x_abs = max(abs(float(rdf["impact"].min())), abs(float(rdf["impact"].max())))
    ax.set_xlim(-x_abs * 1.15, x_abs * 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(True, axis="both", color=base.COLORS["grid"], linewidth=0.55, alpha=0.78)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(width=1.0, length=3.0, pad=1.5)

    cb = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.018)
    cb.set_label("Gate value", rotation=90, labelpad=2.0, fontweight="bold")
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cb.set_ticks([0.0, max(0.6, float(rdf["gate"].max()))])
    cb.set_ticklabels(["Low", "High"])
    cb.outline.set_linewidth(0.7)


def plot_sensitivity_horizontal(ax: plt.Axes, edges_df: pd.DataFrame) -> None:
    """Horizontal sensitivity plot: edges on x-axis, signed beta on y-axis."""
    cond_dirs = sorted([p for p in SHARED_STUDENT_BETA_DIR.iterdir() if p.is_dir()])

    top_edges = edges_df.sort_values("gate", ascending=False).head(20)
    rows = []
    for cond_dir in cond_dirs:
        edge_path = cond_dir / "edges.csv"
        if not edge_path.exists():
            continue
        cond_df = pd.read_csv(edge_path).set_index(["source", "target"])
        for _, row in top_edges.iterrows():
            key = (row["source"], row["target"])
            if key in cond_df.index:
                val = cond_df.loc[key]
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[0]
                beta = float(val["beta"])
                sign = 1.0 if str(val["sign"]) == "activation" else -1.0
            else:
                beta = 0.0
                sign = 1.0
            rows.append({
                "edge": f"{row['source']}\u2192{row['target']}",
                "signed_beta": sign * beta,
                "gate": float(row["gate"]),
            })
    rdf = pd.DataFrame(rows)

    edge_order = (
        rdf.groupby("edge")["signed_beta"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=False)
        .index.tolist()
    )
    edge_to_x = {edge: idx for idx, edge in enumerate(edge_order)}

    short_map = {}
    for _, row in top_edges.iterrows():
        src = base.short_name(row["source"]).replace("\n", " ")
        tgt = base.short_name(row["target"]).replace("\n", " ")
        short_map[f"{row['source']}\u2192{row['target']}"] = f"{src}\n{tgt}"

    rng = np.random.default_rng(7)
    x = np.array([edge_to_x[e] for e in rdf["edge"]], dtype=float)
    x += rng.normal(0, 0.08, size=len(x))

    cmap = mpl.colors.LinearSegmentedColormap.from_list("gate", ["#1E88E5", "#7E57C2", "#D81B60"])
    ax.scatter(
        x, rdf["signed_beta"],
        c=rdf["gate"], cmap=cmap,
        vmin=0.0, vmax=max(0.6, float(rdf["gate"].max())),
        s=18, alpha=0.88, linewidths=0, zorder=3,
    )

    for edge, x_pos in edge_to_x.items():
        mean_val = rdf.loc[rdf["edge"] == edge, "signed_beta"].mean()
        ax.scatter(
            [x_pos], [mean_val], marker="D", s=28,
            facecolor="#222222", edgecolor="white", linewidths=0.6, zorder=4,
        )

    ax.axhline(0, color="#7A7A7A", lw=1.0, alpha=0.85, zorder=2)
    ax.set_xticks(np.arange(len(edge_order)))
    ax.set_xticklabels(
        [short_map.get(e, e) for e in edge_order],
        rotation=90, fontsize=4.2, ha="center",
    )
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylabel(r"signed $\beta$ across conditions", labelpad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(True, axis="y", color=base.COLORS["grid"], linewidth=0.55, alpha=0.78)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(width=1.0, length=3.0, pad=1.5)


# ── Main figure assembly ──────────────────────────────────────────────

def save_outputs(fig: plt.Figure) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"hpn_dream_discovery_main.{ext}", facecolor="white")
    fig.savefig(PANEL_DIR / "hpn_dream_discovery_preview.png", facecolor="white", dpi=300)


def main() -> None:
    base.configure_style()

    teacher_edges = pd.read_csv(SHARED_TEACHER_EDGES)
    student_edges = pd.read_csv(SHARED_STUDENT_EDGES)

    fig = plt.figure(figsize=(7.8, 10.0))

    # 2-row layout: gate heatmaps / consensus networks
    outer = fig.add_gridspec(
        2, 1,
        left=0.050, right=0.992, top=0.978, bottom=0.032,
        height_ratios=[0.80, 1.50],
        hspace=0.12,
    )

    # ── Row 1: Gate heatmaps ──
    sub_a = outer[0, 0].subgridspec(1, 2, wspace=0.0)
    ax_teacher = fig.add_subplot(sub_a[0, 0])
    ax_student = fig.add_subplot(sub_a[0, 1])
    plot_gate_heatmaps(fig, [ax_teacher, ax_student])

    # ── Row 2: Consensus networks ──
    sub_b = outer[1, 0].subgridspec(1, 2, wspace=0.06)
    ax_t_net = fig.add_subplot(sub_b[0, 0])
    ax_s_net = fig.add_subplot(sub_b[0, 1])
    plot_network_on_axis(ax_t_net, teacher_edges,
                         f"Teacher consensus network", show_layer_labels=True)
    plot_network_on_axis(ax_s_net, student_edges,
                         f"Student consensus network", show_layer_labels=False)

    # ── Panel labels ──
    fig.canvas.draw()

    # Panel a: above heatmaps
    ax_boxes_a = [ax_teacher.get_position(), ax_student.get_position()]
    a_y1 = max(b.y1 for b in ax_boxes_a)
    a_x0 = min(b.x0 for b in ax_boxes_a)
    fig.text(a_x0 - 0.018, a_y1 + 0.008, "a", fontsize=13, fontweight="bold",
             ha="left", va="bottom")
    fig.text(a_x0, a_y1 + 0.008, "Consensus gate matrix", fontsize=9.5,
             fontweight="bold", ha="left", va="bottom")

    # Panel b: above network panels — align with panel a
    net_boxes = [ax_t_net.get_position(), ax_s_net.get_position()]
    b_y1 = max(b.y1 for b in net_boxes)
    fig.text(a_x0 - 0.018, b_y1 + 0.011, "b", fontsize=13, fontweight="bold",
             ha="left", va="bottom")
    fig.text(a_x0, b_y1 + 0.011, "Shared-topology consensus networks",
             fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    save_outputs(fig)
    plt.close(fig)

    print("Saved:")
    for ext in ["pdf", "png", "svg"]:
        path = OUT_DIR / f"hpn_dream_discovery_main.{ext}"
        if path.exists():
            print(f"  {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
