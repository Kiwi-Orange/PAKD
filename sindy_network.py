"""
SINDy + Elastic Net  —  Directed Causal Network Discovery for Phosphoproteomic Time Series.

Theory
------
We transform causal network discovery into a continuous‑time differential
equation inference problem (Sparse Identification of Nonlinear Dynamics):

    dX_i / dt  =  Σ_j  W_ji  X_j

*  W_ji ≠ 0   ⟹  directed edge  X_j → X_i  (time derivative gives strict direction)
*  |W_ji|     =  edge strength
*  W_ji > 0   ⟹  activation
*  W_ji < 0   ⟹  inhibition

Elastic Net (L1 + L2 regularisation) enforces sparsity while handling
correlated predictors — ideal for signalling cascades with co‑regulated arms.

Input
-----
  grn_ready_data/ts_{teacher,student}_pred_<stimulus>_<inhibitor>.csv
  Each CSV: protein_name × time points  (rows = proteins, columns = time)

Output
------
  results/sindy/
    W_consensus_{source}.csv            — consensus weight matrix
    W_consensus_{source}.{png,pdf}      — heatmap
    network_directed_{source}.{png,pdf} — directed graph with activation/inhibition
    network_directed_{source}_edges.csv — edge list
    W_per_condition/                    — per-condition weight matrices
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix as _coo_matrix
from scipy.sparse.linalg import expm as _sparse_expm
from sklearn.linear_model import ElasticNetCV

warnings.filterwarnings("ignore", category=UserWarning)

# ── matplotlib style ────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.linewidth": 0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "text.usetex": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ============================================================================
# Constants reused from infer_protein_network.py
# ============================================================================
DISPLAY_NAMES_MULTILINE = {
    "EGFR_pY1068": "EGFR\npY1068",
    "EGFR_pY1173": "EGFR\npY1173",
    "EGFR_pY992": "EGFR\npY992",
    "AKT_pS473": "AKT\npS473",
    "AKT_pT308": "AKT\npT308",
    "MAPK_pT202_Y204": "MAPK\npT202\nY204",
    "MEK1_pS217_S221": "MEK1\npS217\nS221",
    "mTOR_pS2448": "mTOR\npS2448",
    "S6_pS235_S236": "S6\npS235\nS236",
    "S6_pS240_S244": "S6\npS240\nS244",
    "p70S6K_pT389": "p70S6K\npT389",
    "4EBP1_pS65": "4EBP1\npS65",
    "STAT3_pY705": "STAT3\npY705",
    "p38_pT180_Y182": "p38\npT180\nY182",
    "JNK_pT183_pT185": "JNK\npT183\nT185",
    "c-JUN_pS73": "cJUN\npS73",
    "c-Raf_pS338": "cRaf\npS338",
    "GSK3-alpha-beta_pS21_S9": "GSK3ab\npS21S9",
    "GSK3-alpha-beta_pS9": "GSK3ab\npS9",
    "PRAS40_pT246": "PRAS40\npT246",
    "PDK1_pS241": "PDK1\npS241",
    "AMPK_pT172": "AMPK\npT172",
    "BAD_pS112": "BAD\npS112",
    "Rb_pS807_S811": "Rb\npS807\nS811",
    "HER2_pY1248": "HER2\npY1248",
    "c-Met_pY1235": "cMet\npY1235",
    "p90RSK_pT359_S363": "p90RSK\npT359\nS363",
    "CHK1_pS345": "CHK1\npS345",
    "CHK2_pT68": "CHK2\npT68",
    "NF-kB-p65_pS536": "NFkBp65\npS536",
    "Src_pY416": "SRC\npY416",
    "Src_pY527": "SRC\npY527",
    "p27_pT157": "p27\npT157",
    "p27_pT198": "p27\npT198",
    "FOXO3a_pS318_S321": "FOXO3a\npS318\nS321",
    "ER-alpha_pS118": "ERa\npS118",
    "ACC_pS79": "ACC\npS79",
    "PKC-alpha_pS657": "PKCa\npS657",
    "TAZ_pS89": "TAZ\npS89",
    "YAP_pS127": "YAP\npS127",
    "YB-1_PS102": "YB1\nPS102",
}

# ── Pathway-based node classification ───────────────────────────────────
PATHWAY_COLORS = {
    "RTK":          "#E8A838",  # amber  — Receptor Tyrosine Kinases
    "PI3K/AKT":     "#7CC68A",  # green  — PI3K / AKT / mTOR axis
    "MAPK/ERK":     "#E57373",  # salmon — Raf / MEK / ERK cascade
    "Stress/SAPK":  "#CE93D8",  # purple — p38, JNK, c-JUN
    "Cell cycle":   "#7EB6D9",  # blue   — cell-cycle & DNA-damage
    "Other":        "#C8C8C8",  # gray   — remaining / cross-pathway
}

NODE_PATHWAY: dict[str, str] = {
    # RTK
    "EGFR_pY1068": "RTK", "EGFR_pY1173": "RTK", "EGFR_pY992": "RTK",
    "HER2_pY1248": "RTK", "c-Met_pY1235": "RTK",
    # PI3K / AKT / mTOR
    "PDK1_pS241": "PI3K/AKT", "AKT_pS473": "PI3K/AKT", "AKT_pT308": "PI3K/AKT",
    "mTOR_pS2448": "PI3K/AKT", "p70S6K_pT389": "PI3K/AKT",
    "S6_pS235_S236": "PI3K/AKT", "S6_pS240_S244": "PI3K/AKT",
    "4EBP1_pS65": "PI3K/AKT", "PRAS40_pT246": "PI3K/AKT",
    "FOXO3a_pS318_S321": "PI3K/AKT",
    # MAPK / ERK
    "c-Raf_pS338": "MAPK/ERK", "MEK1_pS217_S221": "MAPK/ERK",
    "MAPK_pT202_Y204": "MAPK/ERK", "p90RSK_pT359_S363": "MAPK/ERK",
    # Stress / SAPK
    "p38_pT180_Y182": "Stress/SAPK", "JNK_pT183_pT185": "Stress/SAPK",
    "c-JUN_pS73": "Stress/SAPK",
    # Cell cycle / DNA damage
    "Rb_pS807_S811": "Cell cycle", "p27_pT157": "Cell cycle",
    "p27_pT198": "Cell cycle", "CHK1_pS345": "Cell cycle",
    "CHK2_pT68": "Cell cycle",
    # Other / cross-pathway
    "STAT3_pY705": "Other", "NF-kB-p65_pS536": "Other",
    "GSK3-alpha-beta_pS21_S9": "Other", "GSK3-alpha-beta_pS9": "Other",
    "BAD_pS112": "Other", "Src_pY416": "Other", "Src_pY527": "Other",
    "AMPK_pT172": "Other", "ACC_pS79": "Other",
    "ER-alpha_pS118": "Other", "PKC-alpha_pS657": "Other",
    "TAZ_pS89": "Other", "YAP_pS127": "Other", "YB-1_PS102": "Other",
}

AGGREGATE_PRIOR_NETWORK = {
    ("AKT_pT308", "AKT_pS473"),
    ("AKT_pS473", "mTOR_pS2448"),
    ("AKT_pT308", "mTOR_pS2448"),
    ("mTOR_pS2448", "p70S6K_pT389"),
    ("mTOR_pS2448", "4EBP1_pS65"),
    ("p70S6K_pT389", "S6_pS235_S236"),
    ("p70S6K_pT389", "S6_pS240_S244"),
    ("S6_pS235_S236", "S6_pS240_S244"),
    ("AKT_pS473", "GSK3-alpha-beta_pS21_S9"),
    ("AKT_pS473", "GSK3-alpha-beta_pS9"),
    ("AKT_pT308", "GSK3-alpha-beta_pS21_S9"),
    ("GSK3-alpha-beta_pS21_S9", "GSK3-alpha-beta_pS9"),
    ("AKT_pS473", "BAD_pS112"),
    ("AKT_pS473", "PRAS40_pT246"),
    ("AKT_pS473", "FOXO3a_pS318_S321"),
    ("PDK1_pS241", "AKT_pT308"),
    ("AMPK_pT172", "mTOR_pS2448"),
    ("mTOR_pS2448", "AKT_pS473"),
    ("c-Raf_pS338", "MEK1_pS217_S221"),
    ("MEK1_pS217_S221", "MAPK_pT202_Y204"),
    ("MAPK_pT202_Y204", "p90RSK_pT359_S363"),
    ("MAPK_pT202_Y204", "c-JUN_pS73"),
    ("JNK_pT183_pT185", "c-JUN_pS73"),
    ("p38_pT180_Y182", "MAPK_pT202_Y204"),
    ("MAPK_pT202_Y204", "Rb_pS807_S811"),
    ("p70S6K_pT389", "GSK3-alpha-beta_pS21_S9"),
    ("p70S6K_pT389", "BAD_pS112"),
    ("p90RSK_pT359_S363", "YB-1_PS102"),
    ("p90RSK_pT359_S363", "BAD_pS112"),
    ("EGFR_pY1068", "AKT_pT308"),
    ("EGFR_pY1173", "MAPK_pT202_Y204"),
    ("EGFR_pY1068", "EGFR_pY1173"),
    ("EGFR_pY1068", "EGFR_pY992"),
    ("EGFR_pY1173", "EGFR_pY992"),
    ("HER2_pY1248", "AKT_pT308"),
    ("HER2_pY1248", "AKT_pS473"),
    ("c-Met_pY1235", "AKT_pT308"),
    ("Src_pY416", "Src_pY527"),
    ("p27_pT157", "p27_pT198"),
    ("CHK1_pS345", "CHK2_pT68"),
    ("NF-kB-p65_pS536", "BAD_pS112"),
    ("p70S6K_pT389", "Rb_pS807_S811"),
    ("YAP_pS127", "TAZ_pS89"),
}

_PRIOR_EDGE_SET: set[tuple[str, str]] = set()
for _a, _b in AGGREGATE_PRIOR_NETWORK:
    _PRIOR_EDGE_SET.add((_a, _b))
    _PRIOR_EDGE_SET.add((_b, _a))


def _short_name_multiline(protein: str) -> str:
    if protein in DISPLAY_NAMES_MULTILINE:
        return DISPLAY_NAMES_MULTILINE[protein]
    parts = protein.split("_", 1)
    return f"{parts[0]}\n{parts[1]}" if len(parts) == 2 else protein


def _classify_node_color(node: str) -> str:
    pathway = NODE_PATHWAY.get(node, "Other")
    return PATHWAY_COLORS.get(pathway, "#C8C8C8")


def _save_fig(fig, output_path: str):
    """Save figure as PNG and PDF."""
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(output_path)}  +  {os.path.basename(pdf_path)}")


# ============================================================================
# Laplacian Diffusion Prior  (DREAM8 champion approach)
# ============================================================================
def _parse_pid_to_sif(pid_path: str) -> list[tuple[str, str]]:
    """Parse a NCI-PID .pid file and return edges as (source, target) pairs."""
    edges = []
    for line in open(pid_path):
        parts = line.rstrip().split("\t")
        if len(parts) == 3:
            # .pid format: source \t target \t interaction_type
            src, tgt = parts[0], parts[1]
            # Skip non-gene nodes (URLs, complexes with '/')
            if '/' in src or '/' in tgt:
                continue
            if src.startswith('http') or tgt.startswith('http'):
                continue
            if tgt == 'None' or src == 'None':
                continue
            edges.append((src, tgt))
    return edges


def _diffusion_kernel(edges: list[tuple[str, str]],
                      time_T: float = -0.1) -> tuple[np.ndarray, list[str]]:
    """Build graph Laplacian and compute diffusion kernel exp(time_T * L).

    Replicates the DREAM8 champion's kernel_scipy.py in Python 3.

    Returns
    -------
    K : (n, n) dense kernel matrix
    labels : node labels in sorted order
    """
    # Collect nodes and adjacency
    node_set: set[str] = set()
    edge_set: set[tuple[str, str]] = set()
    degrees: dict[str, int] = {}

    for src, tgt in edges:
        if (src, tgt) in edge_set:
            continue
        edge_set.add((src, tgt))
        edge_set.add((tgt, src))  # undirected
        node_set.add(src)
        node_set.add(tgt)
        degrees[src] = degrees.get(src, 0) + 1
        degrees[tgt] = degrees.get(tgt, 0) + 1

    if len(node_set) == 0:
        return np.zeros((0, 0)), []

    node_order = sorted(node_set)
    n = len(node_order)
    node2idx = {name: i for i, name in enumerate(node_order)}

    # Build sparse Laplacian  L = D - A
    from array import array as _array
    row = _array('i')
    col = _array('i')
    data = _array('f')

    # Diagonal: out-degree
    for i, name in enumerate(node_order):
        data.insert(len(data), degrees.get(name, 0))
        row.insert(len(row), i)
        col.insert(len(col), i)

    # Off-diagonal: -1 for each edge
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (node_order[i], node_order[j]) in edge_set:
                row.insert(len(row), i)
                col.insert(len(col), j)
                data.insert(len(data), -1)

    L = _coo_matrix((data, (row, col)), shape=(n, n)).tocsc()
    K = _sparse_expm(time_T * L)
    return K.toarray(), node_order


def build_laplacian_prior(prior_dir: str,
                          protein_names: list[str],
                          idmap_path: str | None = None) -> np.ndarray:
    """Build a (P, P) Laplacian diffusion prior matrix for our proteins.

    Pipeline (replicates DREAM8 champion's dream8-prior.R):
    1. For each .pid pathway file  →  compute diffusion kernel
    2. Aggregate all kernels into a HUGO-gene-level prior
    3. Map from HUGO genes to our antibody names via global-idmap.tab
    4. Normalize to [0, 1]

    Parameters
    ----------
    prior_dir : path to dream8-final-DCTDC directory
    protein_names : our 41 protein/antibody names
    idmap_path : path to global-idmap.tab  (auto-detected if None)

    Returns
    -------
    P_prior : (P, P) matrix with values in [0, 1]
    """
    if idmap_path is None:
        idmap_path = os.path.join(prior_dir, "global-idmap.tab")

    # ── 1. Load HUGO → antibody mapping ──
    hugo2ab: dict[str, set[str]] = {}   # HUGO gene → set of antibody names
    ab2hugo: dict[str, set[str]] = {}   # antibody → set of HUGO genes
    with open(idmap_path) as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) < 2:
                continue
            hugo, ab = parts[0], parts[1]
            hugo2ab.setdefault(hugo, set()).add(ab)
            ab2hugo.setdefault(ab, set()).add(hugo)

    # Collect the HUGO symbols relevant to our proteins
    our_hugos: set[str] = set()
    for pname in protein_names:
        if pname in ab2hugo:
            our_hugos.update(ab2hugo[pname])

    all_hugos = sorted({h for h in hugo2ab})  # all HUGO symbols in the map
    n_hugo = len(all_hugos)
    hugo2idx = {h: i for i, h in enumerate(all_hugos)}

    # ── 2. Process pathway files  →  aggregate kernels ──
    pid_dir = os.path.join(prior_dir, "prior-pid")
    pid_files = sorted(glob.glob(os.path.join(pid_dir, "*.pid")))

    prior_hugo = np.zeros((n_hugo, n_hugo))  # accumulates kernel values

    n_pathways_used = 0
    for pid_path in pid_files:
        edges = _parse_pid_to_sif(pid_path)
        if not edges:
            continue

        K, labels = _diffusion_kernel(edges)
        if len(labels) == 0:
            continue

        # Zero diagonal  (self-connections uninformative)
        np.fill_diagonal(K, 0.0)

        # Map kernel entries to HUGO-level prior
        for ki, lbl_i in enumerate(labels):
            if lbl_i not in hugo2idx:
                continue
            hi = hugo2idx[lbl_i]
            for kj, lbl_j in enumerate(labels):
                if lbl_j not in hugo2idx:
                    continue
                hj = hugo2idx[lbl_j]
                prior_hugo[hi, hj] += K[ki, kj]

        n_pathways_used += 1

    if n_pathways_used == 0:
        print("  ⚠ No pathway files found — returning zero prior")
        return np.zeros((len(protein_names), len(protein_names)))

    # Normalize HUGO-level prior
    max_val = np.max(np.abs(prior_hugo))
    if max_val > 0:
        prior_hugo /= max_val

    # ── 3. Map HUGO → our antibody protein names ──
    P = len(protein_names)
    P_prior = np.zeros((P, P))

    for pi, pname_i in enumerate(protein_names):
        hugos_i = ab2hugo.get(pname_i, set())
        valid_hi = [hugo2idx[h] for h in hugos_i if h in hugo2idx]
        if not valid_hi:
            continue
        for pj, pname_j in enumerate(protein_names):
            if pi == pj:
                continue
            hugos_j = ab2hugo.get(pname_j, set())
            valid_hj = [hugo2idx[h] for h in hugos_j if h in hugo2idx]
            if not valid_hj:
                continue
            # Average over all HUGO pairs (handles many-to-many mapping)
            vals = [prior_hugo[hi, hj] for hi in valid_hi for hj in valid_hj]
            P_prior[pi, pj] = np.mean(vals)

    # Final normalisation to [0, 1]
    max_p = np.max(P_prior)
    if max_p > 0:
        P_prior /= max_p

    print(f"  Laplacian prior: {n_pathways_used} pathways, "
          f"{np.count_nonzero(P_prior > 0.01)} / {P*(P-1)} pairs with P > 0.01")

    return P_prior


# ============================================================================
# Pathway Commons Prior  (bulk SIF-based)
# ============================================================================

# Relation types ranked by relevance to phospho-signaling
_PC_REL_WEIGHTS = {
    "controls-phosphorylation-of": 1.0,
    "controls-state-change-of": 0.8,
    "catalysis-precedes": 0.7,
    "controls-transport-of": 0.6,
    "controls-expression-of": 0.5,
    "in-complex-with": 0.3,
    "interacts-with": 0.2,
    "chemical-affects": 0.1,
}


def build_pathway_commons_prior(sif_path: str,
                                protein_names: list[str],
                                idmap_path: str | None = None) -> np.ndarray:
    """Build a (P, P) prior matrix from Pathway Commons SIF data.

    For each pair of proteins, the prior value is the maximum relation
    weight across all SIF relation types connecting their HUGO genes.
    The result is normalized to [0, 1].

    Parameters
    ----------
    sif_path : path to pc-hgnc.sif (uncompressed) or .sif.gz
    protein_names : our 41 antibody names
    idmap_path : path to global-idmap.tab (for antibody → HUGO mapping)

    Returns
    -------
    P_prior : (P, P) matrix with values in [0, 1]
    """
    import gzip

    # ── 1. Load HUGO → antibody mapping ──
    if idmap_path is None:
        idmap_path = os.path.join(os.path.dirname(sif_path), "..",
                                  "dream8-final-DCTDC", "global-idmap.tab")
        if not os.path.exists(idmap_path):
            idmap_path = os.path.join("dream8-final-DCTDC", "global-idmap.tab")

    ab2hugo: dict[str, set[str]] = {}
    with open(idmap_path) as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) < 2:
                continue
            hugo, ab = parts[0], parts[1]
            ab2hugo.setdefault(ab, set()).add(hugo)

    # Collect HUGO symbols for our proteins
    pname2hugos: dict[str, set[str]] = {}
    all_hugos: set[str] = set()
    for pname in protein_names:
        hugos = ab2hugo.get(pname, set())
        if hugos:
            pname2hugos[pname] = hugos
            all_hugos.update(hugos)

    if not all_hugos:
        print("  ⚠ No HUGO mapping found — returning zero PC prior")
        return np.zeros((len(protein_names), len(protein_names)))

    # ── 2. Scan SIF file for edges between our HUGO genes ──
    edge_weight: dict[tuple[str, str], float] = {}

    open_fn = gzip.open if sif_path.endswith(".gz") else open
    mode = "rt" if sif_path.endswith(".gz") else "r"
    with open_fn(sif_path, mode) as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) != 3:
                continue
            src, rel, tgt = parts
            if src not in all_hugos or tgt not in all_hugos or src == tgt:
                continue
            w = _PC_REL_WEIGHTS.get(rel, 0.1)
            key = (src, tgt)
            if w > edge_weight.get(key, 0.0):
                edge_weight[key] = w

    # ── 3. Map HUGO edges → antibody prior matrix ──
    P = len(protein_names)
    P_prior = np.zeros((P, P))

    for pi, pname_i in enumerate(protein_names):
        hugos_i = pname2hugos.get(pname_i, set())
        if not hugos_i:
            continue
        for pj, pname_j in enumerate(protein_names):
            if pi == pj:
                continue
            hugos_j = pname2hugos.get(pname_j, set())
            if not hugos_j:
                continue
            # Max weight across all HUGO gene pairs
            vals = [edge_weight.get((hi, hj), 0.0)
                    for hi in hugos_i for hj in hugos_j]
            P_prior[pi, pj] = max(vals) if vals else 0.0

    # Normalize to [0, 1]
    max_p = np.max(P_prior)
    if max_p > 0:
        P_prior /= max_p

    n_nonzero = np.count_nonzero(P_prior > 0.01)
    print(f"  Pathway Commons prior: {len(edge_weight)} HUGO-gene edges, "
          f"{n_nonzero} / {P*(P-1)} antibody pairs with P > 0.01")

    return P_prior


# ============================================================================
# Stage 0 — Data Loading
# ============================================================================
def load_ts_csv(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a single time-series CSV.

    Returns
    -------
    times : (T,)
    X : (T, P) — proteins along columns
    protein_names : list of str
    """
    df = pd.read_csv(path, index_col=0)  # rows=proteins, cols=time
    protein_names = list(df.index)
    times = np.array([float(c) for c in df.columns])
    X = df.values.T  # (T, P)
    return times, X, protein_names


def discover_conditions(data_dir: str, source: str = "teacher") -> list[tuple[str, str]]:
    """Return (condition_label, filepath) pairs for the given source."""
    pattern = os.path.join(data_dir, f"ts_{source}_pred_*.csv")
    files = sorted(glob.glob(pattern))
    results = []
    for fp in files:
        base = os.path.basename(fp).replace(f"ts_{source}_pred_", "").replace(".csv", "")
        results.append((base, fp))
    return results


# ============================================================================
# Stage 1 — Numerical Derivative  (Savitzky–Golay Smoothed)
# ============================================================================
def estimate_derivatives(times: np.ndarray, X: np.ndarray,
                         sg_window: int = 11, sg_poly: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate dX/dt using Savitzky–Golay differentiation.

    Savitzky–Golay simultaneously smooths noise and computes the derivative,
    avoiding the amplification of noise that plagues simple finite differences.

    Parameters
    ----------
    times : (T,)
    X : (T, P)
    sg_window : int
        Window length for the Savitzky–Golay filter (must be odd, > sg_poly).
    sg_poly : int
        Polynomial order for the filter.

    Returns
    -------
    times_d : (T,)   time points (same grid)
    X_smooth : (T, P)  smoothed X
    dXdt : (T, P)    estimated derivatives
    """
    T, P = X.shape
    # Ensure window is odd and ≤ T
    sg_window = min(sg_window, T if T % 2 == 1 else T - 1)
    if sg_window < sg_poly + 2:
        sg_window = sg_poly + 2
    if sg_window % 2 == 0:
        sg_window += 1

    # Average time step (non-uniform grid → we'll use uniform SG on index,
    # then rescale by dt)
    dt_avg = np.mean(np.diff(times))

    X_smooth = np.zeros_like(X)
    dXdt = np.zeros_like(X)

    for j in range(P):
        X_smooth[:, j] = savgol_filter(X[:, j], sg_window, sg_poly, deriv=0)
        # deriv=1 with delta=dt_avg gives dX/dt in physical units
        dXdt[:, j] = savgol_filter(X[:, j], sg_window, sg_poly, deriv=1, delta=dt_avg)

    return times, X_smooth, dXdt


# ============================================================================
# Stage 2 — SINDy + Elastic Net Regression
# ============================================================================
def sindy_elastic_net(X: np.ndarray, dXdt: np.ndarray,
                      l1_ratio_range: tuple[float, ...] = (0.1, 0.5, 0.7, 0.9, 0.95, 0.99),
                      n_alphas: int = 50,
                      cv: int = 5,
                      max_iter: int = 20000,
                      prior: np.ndarray | None = None,
                      prior_alpha: float = 0.9) -> np.ndarray:
    """Fit  dX_i/dt = Σ_j W_ji X_j  for every target protein i.

    Adaptive Laplacian-Prior Penalty
    --------------------------------
    When *prior* is supplied (a P×P matrix with values in [0,1]),
    the L1 penalty for edge j→i becomes:

        λ · (1 - α·P_ji) · |W_ji|

    This is implemented by **scaling** predictor column j when fitting
    target i:

        X̃_j = X_j / (1 - α·P_ji)

    ElasticNet sees X̃ and returns coefficients W̃.
    The true coefficient is recovered as  W_ji = W̃_ji / (1 - α·P_ji).

    A high prior P_ji → lower effective penalty → edge is easier to retain.
    A zero prior      → full penalty → edge needs strong data evidence.

    Parameters
    ----------
    X, dXdt : arrays  (T, P)
    prior : (P, P) or None — Laplacian diffusion prior matrix
    prior_alpha : float in (0, 1) — strength of prior relaxation
    (other params: same as before)

    Returns
    -------
    W : (P, P)  weight matrix.  W[j, i] ≠ 0 means X_j → X_i.
    """
    T, P = X.shape
    W = np.zeros((P, P))

    for i in range(P):
        y = dXdt[:, i]

        # Skip near-constant derivatives  (protein is inert)
        if np.std(y) < 1e-10:
            continue

        # ── Adaptive feature scaling ──
        if prior is not None:
            # penalty_scale[j] = 1 / (1 - α·P_ji)  for target i
            # Clamp denominator to avoid division by near-zero
            denom = np.maximum(1.0 - prior_alpha * prior[:, i], 0.05)
            scale = 1.0 / denom                    # (P,)
            X_scaled = X * scale[np.newaxis, :]     # (T, P) broadcast
        else:
            scale = None
            X_scaled = X

        model = ElasticNetCV(
            l1_ratio=list(l1_ratio_range),
            n_alphas=n_alphas,
            cv=cv,
            max_iter=max_iter,
            fit_intercept=True,
            n_jobs=-1,
        )
        model.fit(X_scaled, y)
        coefs = model.coef_

        # ── Recover true coefficients ──
        if scale is not None:
            coefs = coefs * scale   # W_ji = W̃_ji * scale[j] = W̃_ji / (1 - α·P_ji)

        W[:, i] = coefs

    return W


# ============================================================================
# Stage 3 — Multi-Condition Consensus
# ============================================================================
def consensus_weight_matrix(W_dict: dict[str, np.ndarray],
                            min_freq: float = 0.3,
                            min_abs_weight: float = 0.0) -> np.ndarray:
    """Aggregate per-condition W matrices into a consensus.

    Strategy
    --------
    1. For each (j, i) pair compute the frequency of non-zero entries across
       conditions  (= stability / reproducibility).
    2. Compute the mean weight across conditions **where it was non-zero**.
    3. Zero out entries that appear in fewer than *min_freq* fraction of conditions.

    Parameters
    ----------
    W_dict : {condition_label: W}   each W is (P, P)
    min_freq : float
        Minimum fraction of conditions where an edge is active to be retained.
    min_abs_weight : float
        Additional filter: drop edges with |mean weight| below this.

    Returns
    -------
    W_consensus : (P, P)
    """
    all_W = np.stack(list(W_dict.values()), axis=0)  # (C, P, P)
    C = all_W.shape[0]

    nonzero_mask = np.abs(all_W) > 1e-12  # (C, P, P)
    freq = nonzero_mask.sum(axis=0) / C     # (P, P)

    # Mean weight (only where non-zero) — avoid div by zero
    sum_w = np.where(nonzero_mask, all_W, 0.0).sum(axis=0)
    count_nz = nonzero_mask.sum(axis=0).astype(float)
    count_nz[count_nz == 0] = 1.0
    mean_w = sum_w / count_nz

    # Mask by frequency and magnitude
    keep = (freq >= min_freq) & (np.abs(mean_w) >= min_abs_weight)
    W_consensus = np.where(keep, mean_w, 0.0)

    # Zero the diagonal (self-regulation is unidentifiable from this ODE)
    np.fill_diagonal(W_consensus, 0.0)

    return W_consensus


# ============================================================================
# Stage 4 — Build NetworkX DiGraph from W
# ============================================================================
def weight_matrix_to_digraph(W: np.ndarray, protein_names: list[str],
                              weight_threshold: float = 0.0) -> nx.DiGraph:
    """Convert weight matrix to directed graph.

    Edge j → i exists when |W[j, i]| > weight_threshold.
    """
    P = W.shape[0]
    G = nx.DiGraph()

    for idx, name in enumerate(protein_names):
        G.add_node(name, index=idx)

    for j in range(P):
        for i in range(P):
            if j == i:
                continue
            w = W[j, i]
            if abs(w) <= weight_threshold:
                continue
            G.add_edge(protein_names[j], protein_names[i],
                       weight=abs(w),
                       sign=1 if w > 0 else -1,
                       raw_weight=w)

    return G


# ============================================================================
# Visualization
# ============================================================================
def _get_layout(G: nx.Graph, seed: int = 42) -> dict:
    """Compute spring layout with overlap removal."""
    pos = nx.spring_layout(G, k=2.2 / np.sqrt(max(len(G), 1)),
                           iterations=200, seed=seed)
    pos = _remove_overlaps(pos)
    return pos


def _remove_overlaps(pos: dict, min_dist: float = 0.18,
                     iterations: int = 300) -> dict:
    """Iteratively push overlapping nodes apart."""
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes], dtype=float)

    for _ in range(iterations):
        moved = False
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                diff = coords[a] - coords[b]
                dist = np.linalg.norm(diff)
                if dist < min_dist and dist > 1e-8:
                    push = (min_dist - dist) / 2.0 * (diff / dist)
                    coords[a] += push
                    coords[b] -= push
                    moved = True
        if not moved:
            break

    return {nodes[k]: coords[k] for k in range(len(nodes))}


def plot_sindy_network(G: nx.DiGraph, title: str, output_path: str):
    """Plot directed network with activation (green →) / inhibition (red ⊣) edges.

    Improvements
    ------------
    1) Put legend outside the axes on the right (no occlusion).
    2) Rescale layout coordinates to fill the plotting area (avoid huge blank space).
    3) Reserve a right margin for the legend via tight_layout(rect=...).
    """
    if len(G.nodes()) == 0 or len(G.edges()) == 0:
        print(f"  ⚠ Empty graph — skipping: {title}")
        return

    fig, ax = plt.subplots(figsize=(28, 24))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- layout ----
    pos = _get_layout(G)  # your spring_layout + overlap removal

    # Rescale pos to occupy most of the axes area (avoid cluster in a corner)
    xs = np.array([pos[n][0] for n in pos], dtype=float)
    ys = np.array([pos[n][1] for n in pos], dtype=float)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Prevent divide-by-zero if degenerate
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    # Map to [0.06, 0.94] × [0.06, 0.94]
    def _rescale(v, vmin, span, lo=0.06, hi=0.94):
        return lo + (v - vmin) / span * (hi - lo)

    pos = {
        n: (_rescale(pos[n][0], x_min, x_span), _rescale(pos[n][1], y_min, y_span))
        for n in pos
    }

    # ---- collect legend stats ----
    n_act, n_inh = 0, 0
    n_prior, n_novel = 0, 0

    # ---- edges ----
    for u, v, d in G.edges(data=True):
        sign = d.get("sign", 1)
        w = d.get("weight", 0.5)
        is_prior = (u, v) in _PRIOR_EDGE_SET or (v, u) in _PRIOR_EDGE_SET

        lw = max(1.2, 1.0 + 4.0 * min(w, 1.5))

        if sign > 0:
            color = "#2E7D32"  # activation
            n_act += 1
            arrow_style = "->,head_length=0.6,head_width=0.35"
        else:
            color = "#C62828"  # inhibition
            n_inh += 1
            arrow_style = "-|>,head_length=0.6,head_width=0.35"

        if is_prior:
            linestyle = "solid"
            alpha = 0.85
            n_prior += 1
        else:
            linestyle = (0, (5, 3))
            alpha = 0.55
            n_novel += 1

        ax.annotate(
            "",
            xy=pos[v], xycoords="axes fraction",
            xytext=pos[u], textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle=arrow_style,
                color=color,
                linewidth=lw,
                alpha=alpha,
                linestyle=linestyle,
                connectionstyle="arc3,rad=0.08",
                shrinkA=28, shrinkB=28,
            ),
        )

    # ---- nodes ----
    for node in G.nodes():
        x, y = pos[node]
        face_color = _classify_node_color(node)
        label = _short_name_multiline(node)

        ax.text(
            x, y, label,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="#222222",
            fontfamily="sans-serif",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=face_color,
                edgecolor="#666666",
                linewidth=1.5,
                alpha=0.95,
            ),
            zorder=5,
        )

    # ---- legend (outside right) ----
    legend_elements = [
        Patch(facecolor=c, edgecolor="#666666", lw=1.5, label=name)
        for name, c in PATHWAY_COLORS.items()
    ] + [
        Line2D([0], [0], color="#2E7D32", lw=3, marker=">", markersize=8,
               label=f"Activation ({n_act})"),
        Line2D([0], [0], color="#C62828", lw=3, marker=">", markersize=8,
               label=f"Inhibition ({n_inh})"),
        Line2D([0], [0], color="black", lw=2,
               label=f"In prior network ({n_prior})"),
        Line2D([0], [0], color="gray", lw=2, linestyle="--",
               label=f"Novel edge ({n_novel})"),
    ]

    # Put legend on the right, reserve space
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),  # outside the axes
        borderaxespad=0.0,
        ncol=1,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor="none",
        fancybox=True,
        handlelength=2.5,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="#333333")
    ax.axis("off")

    # Reserve right margin for legend
    plt.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    _save_fig(fig, output_path)


def plot_weight_heatmap(W: np.ndarray, protein_names: list[str],
                        title: str, output_path: str):
    """Plot the weight matrix as a red-blue heatmap (red=inhibition, blue=activation)."""
    P = len(protein_names)
    # Short names for tick labels
    short = [DISPLAY_NAMES_MULTILINE.get(p, p).replace("\n", " ") for p in protein_names]

    vmax = np.max(np.abs(W)) or 1.0

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(range(P))
    ax.set_xticklabels(short, rotation=90, fontsize=7, ha="center")
    ax.set_yticks(range(P))
    ax.set_yticklabels(short, fontsize=7)

    ax.set_xlabel("Target  (dX_i/dt)", fontsize=12)
    ax.set_ylabel("Source  (X_j)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("W_ji   (>0 activation, <0 inhibition)", fontsize=10)

    plt.tight_layout()
    _save_fig(fig, output_path)


def plot_edge_strength_distribution(W: np.ndarray, title: str, output_path: str):
    """Plot histogram of non-zero edge weights."""
    nonzero = W[np.abs(W) > 1e-12].ravel()
    if len(nonzero) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute values
    axes[0].hist(np.abs(nonzero), bins=50, color="#5C6BC0", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("|W_ji|")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Edge Strength Distribution (absolute)")

    # Signed values
    pos_w = nonzero[nonzero > 0]
    neg_w = nonzero[nonzero < 0]
    axes[1].hist(pos_w, bins=40, color="#2E7D32", alpha=0.7, label=f"Activation ({len(pos_w)})")
    axes[1].hist(neg_w, bins=40, color="#C62828", alpha=0.7, label=f"Inhibition ({len(neg_w)})")
    axes[1].set_xlabel("W_ji")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Signed Weight Distribution")
    axes[1].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, output_path)


# ============================================================================
# Export
# ============================================================================
def export_edge_list(G: nx.DiGraph, protein_names: list[str], output_path: str):
    """Export edge list with source, target, weight, sign, prior status."""
    rows = []
    for u, v, d in G.edges(data=True):
        is_prior = (u, v) in _PRIOR_EDGE_SET or (v, u) in _PRIOR_EDGE_SET
        rows.append({
            "source": u,
            "target": v,
            "raw_weight": d.get("raw_weight", 0.0),
            "abs_weight": d.get("weight", 0.0),
            "sign": "activation" if d.get("sign", 1) > 0 else "inhibition",
            "in_prior": is_prior,
        })
    df = pd.DataFrame(rows)
    df.sort_values("abs_weight", ascending=False, inplace=True)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Edge list saved: {os.path.basename(output_path)} ({len(df)} edges)")


def export_weight_matrix(W: np.ndarray, protein_names: list[str], output_path: str):
    """Export W as a labelled CSV."""
    df = pd.DataFrame(W, index=protein_names, columns=protein_names)
    df.index.name = "source_j"
    df.columns.name = "target_i"
    df.to_csv(output_path)
    print(f"  ✓ Weight matrix saved: {os.path.basename(output_path)}")


# ============================================================================
# Differential Network: Teacher vs Student
# ============================================================================
def compute_differential_sindy(W_teacher: np.ndarray, W_student: np.ndarray,
                                protein_names: list[str],
                                weight_threshold: float = 0.0) -> dict:
    """Compare teacher and student weight matrices edge-by-edge."""
    P = len(protein_names)
    shared, teacher_only, student_only = [], [], []

    for j in range(P):
        for i in range(P):
            if j == i:
                continue
            wt = W_teacher[j, i]
            ws = W_student[j, i]
            t_active = abs(wt) > weight_threshold
            s_active = abs(ws) > weight_threshold

            edge_info = {
                "source": protein_names[j],
                "target": protein_names[i],
                "teacher_weight": wt,
                "student_weight": ws,
            }

            if t_active and s_active:
                shared.append(edge_info)
            elif t_active:
                teacher_only.append(edge_info)
            elif s_active:
                student_only.append(edge_info)

    return {
        "shared": shared,
        "teacher_only": teacher_only,
        "student_only": student_only,
        "n_shared": len(shared),
        "n_teacher_only": len(teacher_only),
        "n_student_only": len(student_only),
    }


def plot_differential_sindy_network(diff_result: dict, title: str, output_path: str):
    """Plot differential network: shared / teacher-only / student-only edges.

    Improvements
    ------------
    1) Put legend outside the axes (right side) so it never occludes.
    2) Rescale layout coordinates to fill the axes (avoid huge blank areas).
    3) Use axes-fraction coordinates consistently for robust spacing.
    """
    G = nx.DiGraph()

    # Add all edges to a combined graph
    for edge in diff_result.get("shared", []):
        G.add_edge(edge["source"], edge["target"],
                   category="shared", raw_weight=edge.get("teacher_weight", 0.0))
    for edge in diff_result.get("teacher_only", []):
        G.add_edge(edge["source"], edge["target"],
                   category="teacher_only", raw_weight=edge.get("teacher_weight", 0.0))
    for edge in diff_result.get("student_only", []):
        G.add_edge(edge["source"], edge["target"],
                   category="student_only", raw_weight=edge.get("student_weight", 0.0))

    if len(G.nodes()) == 0 or len(G.edges()) == 0:
        print(f"  ⚠ Empty differential graph — skipping: {title}")
        return

    fig, ax = plt.subplots(figsize=(28, 24))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- layout ----
    pos = _get_layout(G)

    # Rescale pos to occupy most of the axes area
    xs = np.array([pos[n][0] for n in pos], dtype=float)
    ys = np.array([pos[n][1] for n in pos], dtype=float)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    def _rescale(v, vmin, span, lo=0.06, hi=0.94):
        return lo + (v - vmin) / span * (hi - lo)

    pos = {
        n: (_rescale(pos[n][0], x_min, x_span), _rescale(pos[n][1], y_min, y_span))
        for n in pos
    }

    cat_styles = {
        "shared":       {"color": "#333333", "lw_base": 2.0, "alpha": 0.85, "ls": "solid"},
        "teacher_only": {"color": "#1565C0", "lw_base": 1.8, "alpha": 0.75, "ls": (0, (5, 3))},
        "student_only": {"color": "#E65100", "lw_base": 1.8, "alpha": 0.75, "ls": (0, (3, 2))},
    }

    # ---- edges ----
    for u, v, d in G.edges(data=True):
        cat = d.get("category", "shared")
        style = cat_styles.get(cat, cat_styles["shared"])

        raw_w = float(d.get("raw_weight", 0.0))
        w = abs(raw_w)
        sign = 1 if raw_w >= 0 else -1
        arrow_style = "->,head_length=0.5,head_width=0.3" if sign > 0 else "-|>,head_length=0.5,head_width=0.3"

        ax.annotate(
            "",
            xy=pos[v], xycoords="axes fraction",
            xytext=pos[u], textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle=arrow_style,
                color=style["color"],
                linewidth=max(1.0, style["lw_base"] + 2.0 * min(w, 1.0)),
                alpha=style["alpha"],
                linestyle=style["ls"],
                connectionstyle="arc3,rad=0.08",
                shrinkA=28, shrinkB=28,
            ),
        )

    # ---- nodes ----
    for node in G.nodes():
        x, y = pos[node]
        ax.text(
            x, y, _short_name_multiline(node),
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="#222222",
            fontfamily="sans-serif",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=_classify_node_color(node),
                edgecolor="#666666",
                linewidth=1.5,
                alpha=0.95,
            ),
            zorder=5,
        )

    # ---- legend (outside right) ----
    n_sh = diff_result.get("n_shared", 0)
    n_to = diff_result.get("n_teacher_only", 0)
    n_so = diff_result.get("n_student_only", 0)

    legend_elements = [
        Line2D([0], [0], color="#333333", lw=3, label=f"Shared ({n_sh})"),
        Line2D([0], [0], color="#1565C0", lw=2.5, linestyle="--", label=f"Teacher only ({n_to})"),
        Line2D([0], [0], color="#E65100", lw=2.5, linestyle=":",  label=f"Student only ({n_so})"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        ncol=1,
        fontsize=11,
        framealpha=0.95,
        edgecolor="none",
        fancybox=True,
        handlelength=3,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="#333333")
    ax.axis("off")

    # Reserve right margin for legend
    plt.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    _save_fig(fig, output_path)

# ============================================================================
# Main Pipeline
# ============================================================================
def run_sindy_pipeline(data_dir: str, output_dir: str,
                       source: str = "teacher",
                       sg_window: int = 11,
                       sg_poly: int = 3,
                       l1_ratios: tuple[float, ...] = (0.1, 0.3, 0.5),
                       cv: int = 5,
                       consensus_freq: float = 0.1,
                       weight_threshold: float = 0.0,
                       prior_dir: str | None = None,
                       prior_alpha: float = 0.9,
                       prior_source: str = "pid",
                       pc_sif_path: str | None = None,
                       teacher_prior_matrix: np.ndarray | None = None,
                       teacher_prior_beta: float = 0.5) -> tuple[np.ndarray, list[str]]:
    """Run SINDy + Elastic Net pipeline on all conditions.

    Parameters
    ----------
    prior_source : 'pid' | 'pc' | 'both' | 'none'
        Which prior to use: NCI-PID Laplacian ('pid'), Pathway Commons SIF
        ('pc'), element-wise max of both ('both'), or no prior ('none').
    pc_sif_path : path to Pathway Commons pc-hgnc.sif file (for 'pc'/'both')
    teacher_prior_matrix : (P, P) array or None
        Normalized |W_teacher| consensus matrix to blend with the literature
        prior.  When provided the final prior becomes:
            P_combined = β · P_literature + (1-β) · teacher_prior_matrix
    teacher_prior_beta : float in (0, 1)
        Mixing weight β.  Higher = more weight on literature prior.

    Returns (W_consensus, protein_names).
    """
    os.makedirs(output_dir, exist_ok=True)
    per_cond_dir = os.path.join(output_dir, "W_per_condition", source)
    os.makedirs(per_cond_dir, exist_ok=True)

    conditions = discover_conditions(data_dir, source)
    if not conditions:
        raise FileNotFoundError(f"No files found for source={source!r} in {data_dir}")

    print(f"{'═' * 70}")
    print(f"  SINDy + Elastic Net  —  source: {source}")
    print(f"  Conditions found: {len(conditions)}")
    print(f"  SG filter: window={sg_window}, poly={sg_poly}")
    print(f"  Elastic Net l1_ratios: {l1_ratios}")
    print(f"  Consensus min frequency: {consensus_freq}")
    print(f"  Prior source: {prior_source}  α={prior_alpha}")
    print(f"{'═' * 70}")

    # ── Build prior (deferred until we know protein names) ──
    prior_matrix = None  # will be set after first CSV load

    W_dict: dict[str, np.ndarray] = {}
    protein_names = None

    for cond_label, filepath in conditions:
        print(f"\n── Condition: {cond_label} ──")

        times, X, pnames = load_ts_csv(filepath)
        if protein_names is None:
            protein_names = pnames
            # Build prior now that we know the protein names
            if prior_source in ("pid", "both") and prior_dir:
                print(f"\n  Building NCI-PID Laplacian diffusion prior ...")
                pid_prior = build_laplacian_prior(prior_dir, protein_names)
            else:
                pid_prior = None

            if prior_source in ("pc", "both") and pc_sif_path:
                print(f"\n  Building Pathway Commons prior ...")
                idmap = os.path.join(prior_dir, "global-idmap.tab") if prior_dir else None
                pc_prior = build_pathway_commons_prior(pc_sif_path, protein_names,
                                                       idmap_path=idmap)
            else:
                pc_prior = None

            # Combine priors
            if prior_source == "both" and pid_prior is not None and pc_prior is not None:
                prior_matrix = np.maximum(pid_prior, pc_prior)
                max_p = np.max(prior_matrix)
                if max_p > 0:
                    prior_matrix /= max_p
                n_nz = np.count_nonzero(prior_matrix > 0.01)
                P = len(protein_names)
                print(f"  Combined prior (element-wise max): "
                      f"{n_nz} / {P*(P-1)} pairs with P > 0.01")
            elif pid_prior is not None:
                prior_matrix = pid_prior
            elif pc_prior is not None:
                prior_matrix = pc_prior

            # ── Blend with teacher prior if provided ──
            if teacher_prior_matrix is not None:
                lit_prior = prior_matrix if prior_matrix is not None else np.zeros((len(protein_names), len(protein_names)))
                beta = teacher_prior_beta
                prior_matrix = beta * lit_prior + (1.0 - beta) * teacher_prior_matrix
                # Re-normalise to [0, 1]
                mx = np.max(prior_matrix)
                if mx > 0:
                    prior_matrix /= mx
                n_nz = np.count_nonzero(prior_matrix > 0.01)
                P = len(protein_names)
                print(f"  Combined prior (β={beta:.2f} lit + {1-beta:.2f} teacher): "
                      f"{n_nz} / {P*(P-1)} pairs with P > 0.01")

            if prior_matrix is not None:
                prior_path = os.path.join(output_dir, f"prior_{prior_source}.csv")
                pd.DataFrame(prior_matrix, index=protein_names,
                             columns=protein_names).to_csv(prior_path)
                print(f"  ✓ Prior saved: {os.path.basename(prior_path)}")
        else:
            assert pnames == protein_names, "Protein order mismatch across conditions"

        T, P = X.shape
        print(f"  Loaded: {T} time points × {P} proteins")

        # Estimate derivatives
        times_d, X_smooth, dXdt = estimate_derivatives(times, X, sg_window, sg_poly)
        print(f"  dX/dt estimated via Savitzky–Golay (window={sg_window}, poly={sg_poly})")

        # SINDy + Elastic Net  (with optional adaptive prior penalty)
        W = sindy_elastic_net(X_smooth, dXdt, l1_ratio_range=l1_ratios, cv=cv,
                              prior=prior_matrix, prior_alpha=prior_alpha)
        n_nonzero = np.count_nonzero(np.abs(W) > 1e-12) - P  # exclude diagonal
        np.fill_diagonal(W, 0.0)
        n_nonzero = np.count_nonzero(np.abs(W) > 1e-12)
        print(f"  Non-zero edges: {n_nonzero} / {P*(P-1)} possible "
              f"({100*n_nonzero/(P*(P-1)):.1f}% density)")

        W_dict[cond_label] = W

        # Save per-condition
        export_weight_matrix(W, protein_names,
                             os.path.join(per_cond_dir, f"W_{cond_label}.csv"))

    # ── Consensus ──
    print(f"\n{'─' * 50}")
    print(f"  Building consensus (min_freq={consensus_freq}) ...")
    W_consensus = consensus_weight_matrix(W_dict, min_freq=consensus_freq,
                                           min_abs_weight=weight_threshold)
    n_edges = np.count_nonzero(np.abs(W_consensus) > 1e-12)
    P = len(protein_names)
    print(f"  Consensus edges: {n_edges} / {P*(P-1)} "
          f"({100*n_edges/(P*(P-1)):.1f}% density)")

    pos_edges = np.count_nonzero(W_consensus > 1e-12)
    neg_edges = np.count_nonzero(W_consensus < -1e-12)
    print(f"  Activation: {pos_edges}  |  Inhibition: {neg_edges}")

    # ── Save ──
    prefix = os.path.join(output_dir, f"W_consensus_{source}")
    export_weight_matrix(W_consensus, protein_names, f"{prefix}.csv")
    plot_weight_heatmap(W_consensus, protein_names,
                        f"SINDy Consensus W  ({source})", f"{prefix}_heatmap.png")
    plot_edge_strength_distribution(W_consensus,
                                    f"Edge Weights ({source})",
                                    f"{prefix}_distribution.png")

    # ── Directed graph ──
    G = weight_matrix_to_digraph(W_consensus, protein_names, weight_threshold)
    plot_sindy_network(G,
                       f"SINDy + Elastic Net Directed Network ({source})",
                       os.path.join(output_dir, f"network_directed_{source}.png"))
    export_edge_list(G, protein_names,
                     os.path.join(output_dir, f"network_directed_{source}_edges.csv"))

    return W_consensus, protein_names


def compare_teacher_student(data_dir: str, output_dir: str, **kwargs):
    """Run SINDy pipeline on both teacher and student, then compare.

    When teacher_prior_beta is present in *kwargs*, the Teacher's consensus
    |W| is normalised to [0, 1] and blended with the literature prior to
    form a combined soft prior for the Student run:

        P_combined = β · P_literature + (1-β) · |W_teacher|_normalised
    """
    teacher_prior_beta = kwargs.pop("teacher_prior_beta", 0.5)

    print("\n" + "█" * 70)
    print("  TEACHER  network")
    print("█" * 70)
    W_teacher, proteins = run_sindy_pipeline(
        data_dir, output_dir, source="teacher",
        teacher_prior_beta=teacher_prior_beta, **kwargs,
    )

    # ── Build normalised teacher prior ──
    teacher_abs = np.abs(W_teacher)
    np.fill_diagonal(teacher_abs, 0.0)
    t_max = np.max(teacher_abs)
    teacher_prior = teacher_abs / t_max if t_max > 0 else teacher_abs
    n_nz = np.count_nonzero(teacher_prior > 0.01)
    P = len(proteins)
    print(f"\n  Teacher prior: |W_teacher| normalised — "
          f"{n_nz} / {P*(P-1)} pairs > 0.01")
    tp_path = os.path.join(output_dir, "teacher_prior.csv")
    pd.DataFrame(teacher_prior, index=proteins,
                 columns=proteins).to_csv(tp_path)
    print(f"  ✓ Saved: {os.path.basename(tp_path)}")

    print("\n" + "█" * 70)
    print(f"  STUDENT  network  (β={teacher_prior_beta:.2f} lit "
          f"+ {1-teacher_prior_beta:.2f} teacher)")
    print("█" * 70)
    W_student, _ = run_sindy_pipeline(
        data_dir, output_dir, source="student",
        teacher_prior_matrix=teacher_prior,
        teacher_prior_beta=teacher_prior_beta, **kwargs,
    )

    # ── Differential ──
    print("\n" + "─" * 70)
    print("  Comparing Teacher vs Student ...")
    diff = compute_differential_sindy(W_teacher, W_student, proteins)
    print(f"  Shared:       {diff['n_shared']}")
    print(f"  Teacher only: {diff['n_teacher_only']}")
    print(f"  Student only: {diff['n_student_only']}")

    plot_differential_sindy_network(
        diff,
        "Differential Network: Teacher vs Student  (SINDy)",
        os.path.join(output_dir, "network_differential_sindy.png"),
    )

    # Export differential edges
    diff_rows = []
    for cat_key, cat_label in [("shared", "shared"),
                                ("teacher_only", "teacher_only"),
                                ("student_only", "student_only")]:
        for e in diff[cat_key]:
            diff_rows.append({
                "source": e["source"],
                "target": e["target"],
                "teacher_weight": e["teacher_weight"],
                "student_weight": e["student_weight"],
                "category": cat_label,
            })
    df = pd.DataFrame(diff_rows)
    diff_path = os.path.join(output_dir, "differential_edges_sindy.csv")
    df.to_csv(diff_path, index=False)
    print(f"  ✓ Differential edges saved: {os.path.basename(diff_path)} ({len(df)} edges)")

    # ── Weight correlation ──
    t_flat = W_teacher.ravel()
    s_flat = W_student.ravel()
    mask = (np.abs(t_flat) > 1e-12) | (np.abs(s_flat) > 1e-12)
    if mask.any():
        from scipy.stats import pearsonr, spearmanr
        r_p, _ = pearsonr(t_flat[mask], s_flat[mask])
        r_s, _ = spearmanr(t_flat[mask], s_flat[mask])
        print(f"\n  Weight correlation (active edges):")
        print(f"    Pearson  r = {r_p:.4f}")
        print(f"    Spearman ρ = {r_s:.4f}")

    # Scatter plot of weights
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(t_flat[mask], s_flat[mask], alpha=0.4, s=12, color="#5C6BC0")
    lim = max(np.abs(t_flat[mask]).max(), np.abs(s_flat[mask]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Teacher W_ji", fontsize=12)
    ax.set_ylabel("Student W_ji", fontsize=12)
    ax.set_title("Teacher vs Student Edge Weights (SINDy)", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    plt.tight_layout()
    scat_path = os.path.join(output_dir, "teacher_vs_student_weights_sindy.png")
    _save_fig(fig, scat_path)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SINDy + Elastic Net directed causal network discovery "
                    "for phosphoproteomic time series.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Teacher network only
  python sindy_network.py --data_dir grn_ready_data --source teacher

  # Student network only
  python sindy_network.py --data_dir grn_ready_data --source student

  # Compare teacher vs student
  python sindy_network.py --data_dir grn_ready_data --source both

  # Custom parameters
  python sindy_network.py --data_dir grn_ready_data --source both \\
      --sg_window 15 --consensus_freq 0.4 --l1_ratios 0.5 0.9 0.99
""",
    )
    parser.add_argument("--data_dir", type=str, default="grn_ready_data",
                        help="Directory containing ts_{teacher,student}_pred_*.csv files")
    parser.add_argument("--output_dir", type=str, default="results/sindy",
                        help="Output directory")
    parser.add_argument("--source", type=str, default="both",
                        choices=["teacher", "student", "both"],
                        help="Which prediction source to use")
    parser.add_argument("--sg_window", type=int, default=11,
                        help="Savitzky-Golay window length (odd)")
    parser.add_argument("--sg_poly", type=int, default=3,
                        help="Savitzky-Golay polynomial order")
    parser.add_argument("--l1_ratios", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5],
                        help="Elastic Net L1 ratio grid")
    parser.add_argument("--cv", type=int, default=5,
                        help="Cross-validation folds")
    parser.add_argument("--consensus_freq", type=float, default=0.1,
                        help="Min fraction of conditions for consensus edge")
    parser.add_argument("--weight_threshold", type=float, default=0.0,
                        help="Min |W| to display an edge")
    parser.add_argument("--prior_dir", type=str, default="dream8-final-DCTDC",
                        help="Path to dream8-final-DCTDC dir for Laplacian prior "
                             "(omit to disable prior)")
    parser.add_argument("--prior_alpha", type=float, default=0.9,
                        help="Prior strength α in (0,1): higher = stronger relaxation "
                             "of penalty on prior-supported edges")
    parser.add_argument("--prior_source", type=str, default="pid",
                        choices=["pid", "pc", "both", "none"],
                        help="Prior source: 'pid' (NCI-PID Laplacian), "
                             "'pc' (Pathway Commons SIF), 'both' (max of both), "
                             "'none' (no prior)")
    parser.add_argument("--pc_sif", type=str, default="data/pc-hgnc.sif",
                        help="Path to Pathway Commons SIF file (for --prior_source pc/both)")
    parser.add_argument("--teacher_prior_beta", type=float, default=0.5,
                        help="Mixing weight β when --source both: "
                             "P = β·P_literature + (1-β)·|W_teacher|.  "
                             "Higher β trusts literature more. (default: 0.5)")

    args = parser.parse_args()

    kw = dict(
        sg_window=args.sg_window,
        sg_poly=args.sg_poly,
        l1_ratios=tuple(args.l1_ratios),
        cv=args.cv,
        consensus_freq=args.consensus_freq,
        weight_threshold=args.weight_threshold,
        prior_dir=args.prior_dir,
        prior_alpha=args.prior_alpha,
        prior_source=args.prior_source,
        pc_sif_path=args.pc_sif,
        teacher_prior_beta=args.teacher_prior_beta,
    )

    if args.source == "both":
        compare_teacher_student(args.data_dir, args.output_dir, **kw)
    else:
        run_sindy_pipeline(args.data_dir, args.output_dir, source=args.source, **kw)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
