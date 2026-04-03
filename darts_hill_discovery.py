# darts_hill_discovery.py
# ------------------------------------------------------------
# One-file pipeline: Differentiable Network Discovery + ODE Discovery
# for phosphoproteomic time series (MCF7-style).
#
# Hybrid Dual-Engine training:
#   Phase 1 — Topology Discovery (derivative matching via cubic splines)
#     No ODE integration; pointwise dX/dt matching. Ultra-fast on GPU.
#   Phase 2 — Dynamics Fine-tuning (torchode adaptive ODE integration)
#     Sparse network => low stiffness => smooth ODE fitting.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import glob
import os
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import time as _time
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchode

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Style
# ============================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

# ============================================================
# Display names (optional)
# ============================================================
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

# ============================================================
# Pathway colors (optional)
# ============================================================
PATHWAY_COLORS = {
    "RTK":          "#E8A838",
    "PI3K/AKT":     "#7CC68A",
    "MAPK/ERK":     "#E57373",
    "Stress/SAPK":  "#CE93D8",
    "Cell cycle":   "#7EB6D9",
    "Other":        "#C8C8C8",
}
NODE_PATHWAY: Dict[str, str] = {}  # fill if you want

def short_name_multiline(protein: str) -> str:
    if protein in DISPLAY_NAMES_MULTILINE:
        return DISPLAY_NAMES_MULTILINE[protein]
    parts = protein.split("_", 1)
    return f"{parts[0]}\n{parts[1]}" if len(parts) == 2 else protein

def classify_node_color(node: str) -> str:
    pathway = NODE_PATHWAY.get(node, "Other")
    return PATHWAY_COLORS.get(pathway, "#C8C8C8")

def save_fig(fig, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: {os.path.basename(output_path)} + {os.path.basename(pdf_path)}")

# ============================================================
# Data loading: rows=proteins, cols=time, values=levels
# ============================================================
def discover_conditions(data_dir: str, source: str) -> List[Tuple[str, str]]:
    pattern = os.path.join(data_dir, f"ts_{source}_pred_*.csv")
    files = sorted(glob.glob(pattern))
    out = []
    for fp in files:
        label = os.path.basename(fp).replace(f"ts_{source}_pred_", "").replace(".csv", "")
        out.append((label, fp))
    return out

def load_ts_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path, index_col=0)
    protein_names = list(df.index)
    times = np.array([float(c) for c in df.columns], dtype=float)
    X = df.values.T  # (T,P)
    return times, X, protein_names

# ============================================================
# Per-protein normalization
# ============================================================
def normalize_proteins(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize each protein to [0,1].
    X: (T, P) -> X_norm: (T, P), mins: (P,), scales: (P,)
    """
    mins = X.min(axis=0)       # (P,)
    maxs = X.max(axis=0)       # (P,)
    scales = maxs - mins
    scales[scales < 1e-12] = 1.0   # constant proteins stay at 0
    X_norm = (X - mins[None, :]) / scales[None, :]
    return X_norm, mins, scales

def denormalize_proteins(X_norm: np.ndarray, mins: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Inverse of normalize_proteins."""
    return X_norm * scales[None, :] + mins[None, :]

# ============================================================
# Hill building blocks
# ============================================================
def masked_fill_diagonal_(A: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    idx = torch.arange(A.shape[0], device=A.device)
    A[idx, idx] = value
    return A

def hill_function(x: torch.Tensor, K: torch.Tensor, n: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute both Hill activation and inhibition: (x^n/(K^n+x^n), K^n/(K^n+x^n))."""
    x = torch.clamp(x, min=0.0)
    K = torch.clamp(K, min=eps)
    n = torch.clamp(n, min=0.1, max=6.0)
    xn = torch.pow(x + eps, n)
    Kn = torch.pow(K, n)
    xn = torch.nan_to_num(xn, nan=0.0, posinf=1e8, neginf=0.0)
    Kn = torch.nan_to_num(Kn, nan=0.0, posinf=1e8, neginf=0.0)
    denom = Kn + xn + eps
    return torch.clamp(xn / denom, 0.0, 1.0), torch.clamp(Kn / denom, 0.0, 1.0)

class EdgeGates(nn.Module):
    """alpha -> gate in (0,1). alpha[j,i] = edge j->i"""
    def __init__(self, P: int, init_bias: float = -2.0, temperature: float = 1.0):
        super().__init__()
        self.P = P
        self.temperature = float(temperature)
        self.alpha = nn.Parameter(torch.full((P, P), float(init_bias)))
        masked_fill_diagonal_(self.alpha.data, -20.0)

    def set_temperature(self, t: float):
        self.temperature = float(t)

    def gates(self) -> torch.Tensor:
        g = torch.sigmoid(self.alpha / max(self.temperature, 1e-6))
        g = g.clone()
        masked_fill_diagonal_(g, 0.0)
        return g

    def l1(self) -> torch.Tensor:
        return self.gates().abs().sum()

class HillODEFuncDARTS(nn.Module):
    """
    Differentiable topology + Hill ODE params learned jointly.
    dX/dt = -gamma * X + basal + sum_j gate*beta*h + Vmax*prod_j (gate*h + (1-gate)*1)
    """
    def __init__(
        self,
        P: int,
        use_additive: bool = True,
        use_multiplicative: bool = True,
        init_gate_bias: float = -2.0,
        gate_temperature: float = 1.5,
    ):
        super().__init__()
        self.P = P
        self.use_add = use_additive
        self.use_mult = use_multiplicative

        self.gates_mod = EdgeGates(P, init_bias=init_gate_bias, temperature=gate_temperature)

        # edge params
        self.logK = nn.Parameter(torch.zeros(P, P))
        self.logn = nn.Parameter(torch.zeros(P, P))
        masked_fill_diagonal_(self.logK.data, -10.0)
        masked_fill_diagonal_(self.logn.data, -10.0)

        self.beta = nn.Parameter(torch.zeros(P, P))  # additive edge strength
        masked_fill_diagonal_(self.beta.data, 0.0)

        self.sign_alpha = nn.Parameter(torch.zeros(P, P))  # activation vs inhibition
        masked_fill_diagonal_(self.sign_alpha.data, 0.0)

        # node params
        self.basal = nn.Parameter(torch.zeros(P))
        self.logVmax = nn.Parameter(torch.zeros(P))
        self.loggamma = nn.Parameter(torch.zeros(P))

    def get_params(self) -> Dict[str, torch.Tensor]:
        g = self.gates_mod.gates()
        K = F.softplus(self.logK) + 1e-6
        n = F.softplus(self.logn) + 1e-6
        Vmax = F.softplus(self.logVmax)
        gamma = F.softplus(self.loggamma)
        sign_gate = torch.sigmoid(self.sign_alpha)  # 1 act, 0 inh
        return dict(gates=g, K=K, n=n, Vmax=Vmax, gamma=gamma, sign_gate=sign_gate)

    def _edge_h(self, X: torch.Tensor, K: torch.Tensor, n: torch.Tensor, sign_gate: torch.Tensor) -> torch.Tensor:
        B, P = X.shape
        X_src = X[:, :, None].expand(B, P, P)
        h_act, h_inh = hill_function(X_src, K[None, :, :], n[None, :, :])
        sg = sign_gate[None, :, :]
        return sg * h_act + (1.0 - sg) * h_inh

    def forward(self, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # accept (P,) or (B,P)
        if X.ndim == 1:
            Xb = X[None, :]
            squeeze = True
        else:
            Xb = X
            squeeze = False

        p = self.get_params()
        g = p["gates"]
        K = p["K"]
        n = p["n"]
        Vmax = p["Vmax"]
        gamma = p["gamma"]
        sign_gate = p["sign_gate"]

        H = self._edge_h(Xb, K, n, sign_gate)  # (B,P,P)

        decay = -gamma[None, :] * Xb

        add = 0.0
        if self.use_add:
            add = (g[None, :, :] * self.beta[None, :, :] * H).sum(dim=1) + self.basal[None, :]

        mult = 0.0
        if self.use_mult:
            H_weighted = g[None, :, :] * H + (1.0 - g[None, :, :]) * 1.0
            prod = H_weighted.prod(dim=1)
            mult = Vmax[None, :] * prod

        dX = decay + add + mult

        # stabilize (lightweight, keep same as your version)
        if torch.isnan(dX).any():
            dX = torch.where(torch.isnan(dX), torch.zeros_like(dX), dX)

        max_dX = 100.0
        dX = torch.clamp(dX, min=-max_dX, max=max_dX)

        return dX.squeeze(0) if squeeze else dX

# ============================================================
# Training config
# ============================================================
@dataclass
class TrainConfig:
    lr: float = 5e-4
    epochs: int = 1500
    lambda_sparse: float = 0.0
    lambda_beta: float = 0.0
    gate_temp_start: float = 2.0
    gate_temp_end: float = 0.4
    clip_grad: float = 2.0
    # Hybrid dual-engine
    phase1_frac: float = 0.8       # fraction of epochs for Phase 1 (derivative matching)
    # torchode integrator controls (Phase 2 only)
    ode_method: str = "tsit5"      # "tsit5" (5th order) or "dopri5"
    ode_atol: float = 1e-5
    ode_rtol: float = 1e-3
    # early stopping (per-phase)
    early_stop_patience: int = 200
    early_stop_min_delta: float = 1e-6
    # gate threshold for edge counting / export
    gate_thr: float = 0.01
    # temporal weighting: upweight later (stable) time points
    time_weight_alpha: float = 1.0    # ramp exponent (0=uniform, 1=linear, >1=stronger late-weighting)
    time_weight_min: float = 0.1      # floor weight for earliest time point

def cosine_anneal(start: float, end: float, step: int, total: int) -> float:
    if total <= 1:
        return end
    c = 0.5 * (1.0 + math.cos(math.pi * step / (total - 1)))
    return end + (start - end) * c

# ============================================================
# Temporal weighting
# ============================================================
def compute_temporal_weights(
    t: torch.Tensor, alpha: float = 1.0, w_min: float = 0.1,
) -> torch.Tensor:
    """Compute per-time-point weights that ramp from w_min to 1.0.
    w(t) = w_min + (1 - w_min) * ((t - t0) / (tmax - t0))^alpha
    Returns: (T,) tensor, normalized so mean == 1.0.
    """
    t0 = t[0]
    tmax = t[-1]
    span = tmax - t0
    if span < 1e-12 or alpha <= 0:
        return torch.ones_like(t)
    frac = (t - t0) / span                   # 0 → 1
    w = w_min + (1.0 - w_min) * frac.pow(alpha)
    w = w / w.mean()                          # normalize so mean==1
    return w

# ============================================================
# A. Spline-based derivative extraction (Phase 1)
# ============================================================
def compute_spline_derivatives(t: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fit cubic spline per protein, return dY/dt at each time point.
    t: (T,), Y: (T, P) -> dYdt: (T, P)
    """
    T, P = Y.shape
    dYdt = np.zeros_like(Y)
    for p in range(P):
        cs = CubicSpline(t, Y[:, p])
        dYdt[:, p] = cs(t, 1)  # 1st derivative
    return dYdt

# ============================================================
# B. torchode ODE solver wrapper (Phase 2)
# ============================================================
def torchode_solve(
    model: HillODEFuncDARTS,
    x0: torch.Tensor,          # (P,)
    t_eval: torch.Tensor,      # (T,)
    method: str = "tsit5",
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> torch.Tensor:
    """Solve ODE with torchode (adaptive Tsit5/Dopri5). Returns (T, P)."""
    y0 = x0.unsqueeze(0)        # (1, P)
    te = t_eval.unsqueeze(0)     # (1, T)
    term = torchode.ODETerm(model)
    step_cls = torchode.Tsit5 if method == "tsit5" else torchode.Dopri5
    step_method = step_cls(term=term)
    controller = torchode.IntegralController(
        atol=atol, rtol=rtol, term=term,
    )
    adjoint = torchode.AutoDiffAdjoint(step_method, controller)
    problem = torchode.InitialValueProblem(y0=y0, t_eval=te)
    sol = adjoint.solve(problem)
    return sol.ys[0]             # (T, P)

# ============================================================
# Training loop — Hybrid Dual-Engine
# ============================================================
def _run_phase(
    model: HillODEFuncDARTS,
    loss_fit_fn,
    cfg: TrainConfig,
    phase_epochs: int,
    epoch_offset: int,
    total_epochs: int,
    label: str,
) -> Tuple[Dict[str, float], float]:
    """Run one training phase (derivative matching or ODE fine-tuning).
    Returns (result_dict, elapsed_seconds).
    """
    t0 = _time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5,
        patience=max(cfg.early_stop_patience // 4, 50),
        min_lr=1e-7,
    )
    best_fit = float("inf")
    best_state: Dict[str, torch.Tensor] = {}
    no_improve = 0
    loss_val = 0.0
    fit_val = 0.0

    print(f"    ── {label} ({phase_epochs} ep) ──")
    for ep in range(phase_epochs):
        global_ep = epoch_offset + ep
        model.train()
        temp = cosine_anneal(cfg.gate_temp_start, cfg.gate_temp_end, global_ep, total_epochs)
        model.gates_mod.set_temperature(temp)

        opt.zero_grad(set_to_none=True)

        loss_fit = loss_fit_fn()
        loss_sparse = cfg.lambda_sparse * model.gates_mod.l1()
        loss_beta = cfg.lambda_beta * model.beta.abs().sum()
        loss = loss_fit + loss_sparse + loss_beta

        loss.backward()
        if cfg.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        opt.step()
        scheduler.step(loss_fit.item())

        fit_val = float(loss_fit.item())
        loss_val = float(loss.item())

        if fit_val < best_fit - cfg.early_stop_min_delta:
            best_fit = fit_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                elapsed = _time.perf_counter() - t0
                print(f"    [{label} ep {ep:04d}] Early stop — best_fit={best_fit:.3e}  elapsed={elapsed:.1f}s")
                break

        if ep % 10 == 0 or ep == phase_epochs - 1:
            elapsed = _time.perf_counter() - t0
            eps_done = ep + 1
            eta = elapsed / eps_done * (phase_epochs - eps_done)
            with torch.no_grad():
                g = model.gates_mod.gates()
                n_edges = (g > cfg.gate_thr).float().sum().item()
                density = n_edges / (model.P * (model.P - 1))
            print(
                f"    [{label} ep {ep:04d}/{phase_epochs}] "
                f"loss={loss_val:.3e} fit={fit_val:.3e} "
                f"sparse={loss_sparse.item():.3e} temp={temp:.2f} "
                f"edges={n_edges:.0f} dens={density:.3f} eta={eta:.0f}s"
            )

    if best_state:
        model.load_state_dict(best_state)

    return {"loss": loss_val, "fit": fit_val}, _time.perf_counter() - t0


def train_one_condition(
    model: HillODEFuncDARTS,
    t: torch.Tensor,      # (T,)
    y_true: torch.Tensor, # (T,P)
    cfg: TrainConfig,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    t = t.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.float32)
    x0 = y_true[0].detach()

    # Pre-compute spline derivatives for Phase 1
    t_np = t.detach().cpu().numpy()
    y_np = y_true.detach().cpu().numpy()
    dYdt_target = torch.tensor(
        compute_spline_derivatives(t_np, y_np), dtype=torch.float32, device=device,
    )

    # Temporal weights
    tw = compute_temporal_weights(t, alpha=cfg.time_weight_alpha, w_min=cfg.time_weight_min)
    tw_2d = tw[:, None]

    phase1_epochs = int(cfg.epochs * cfg.phase1_frac)
    phase2_epochs = cfg.epochs - phase1_epochs
    t0 = _time.perf_counter()

    # Phase 1: Derivative matching (no ODE solve)
    p1_fit = lambda: (tw_2d * (model(t, y_true) - dYdt_target) ** 2).mean()
    result, _ = _run_phase(model, p1_fit, cfg, phase1_epochs, 0, cfg.epochs, "P1 deriv")

    with torch.no_grad():
        g = model.gates_mod.gates()
        n_edges = int((g > cfg.gate_thr).float().sum().item())
    print(f"    ── Phase 1 done: {n_edges} edges (gate>{cfg.gate_thr}) out of {model.P*(model.P-1)} ──")

    if phase2_epochs <= 0:
        return result

    # Phase 2: ODE fine-tuning (torchode adaptive integration)
    p2_fit = lambda: (tw_2d * (torchode_solve(
        model, x0, t, method=cfg.ode_method, atol=cfg.ode_atol, rtol=cfg.ode_rtol,
    ) - y_true) ** 2).mean()
    result, _ = _run_phase(
        model, p2_fit, cfg, phase2_epochs, phase1_epochs, cfg.epochs,
        f"P2 ODE ({cfg.ode_method})",
    )

    print(f"    ── Training complete: total={_time.perf_counter() - t0:.1f}s ──")
    return result

# ============================================================
# Consensus across conditions
# ============================================================
def consensus_matrix(mats: Dict[str, np.ndarray], min_freq: float = 0.3) -> np.ndarray:
    all_W = np.stack(list(mats.values()), axis=0)  # (C,P,P)
    C = all_W.shape[0]
    nonzero = np.abs(all_W) > 1e-12
    freq = nonzero.sum(axis=0) / C

    sum_w = np.where(nonzero, all_W, 0.0).sum(axis=0)
    cnt = nonzero.sum(axis=0).astype(float)
    cnt[cnt == 0] = 1.0
    mean_w = sum_w / cnt

    keep = freq >= min_freq
    out = np.where(keep, mean_w, 0.0)
    np.fill_diagonal(out, 0.0)
    return out

# ============================================================
# Export: matrices + edges + readable equations
# ============================================================
@torch.no_grad()
def export_model_to_matrices(model: HillODEFuncDARTS) -> Dict[str, np.ndarray]:
    p = model.get_params()
    g = p["gates"].cpu().numpy()
    K = p["K"].cpu().numpy()
    n = p["n"].cpu().numpy()
    beta = model.beta.detach().cpu().numpy()
    sign_gate = p["sign_gate"].cpu().numpy()
    gamma = p["gamma"].cpu().numpy()
    basal = model.basal.detach().cpu().numpy()
    Vmax = p["Vmax"].cpu().numpy()
    return dict(gates=g, beta=beta, K=K, n=n, sign_gate=sign_gate, gamma=gamma, basal=basal, Vmax=Vmax)

def export_edges_from_matrices(
    mats: Dict[str, np.ndarray],
    names: List[str],
    thr_gate: float = 0.5,
    thr_abs_beta: float = 0.0,
) -> pd.DataFrame:
    g = mats["gates"]
    beta = mats["beta"]
    K = mats["K"]
    n = mats["n"]
    sign_gate = mats["sign_gate"]

    P = len(names)
    rows = []
    for j in range(P):
        for i in range(P):
            if i == j:
                continue
            if g[j, i] >= thr_gate and abs(beta[j, i]) >= thr_abs_beta:
                rows.append({
                    "source": names[j],
                    "target": names[i],
                    "gate": float(g[j, i]),
                    "beta": float(beta[j, i]),
                    "K": float(K[j, i]),
                    "n": float(n[j, i]),
                    "sign": "activation" if sign_gate[j, i] >= 0.5 else "inhibition",
                })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.sort_values(["gate", "beta"], ascending=[False, False], inplace=True)
    return df

def equations_text(
    mats: Dict[str, np.ndarray],
    names: List[str],
    topk: int = 6,
    gate_thr: float = 0.5,
) -> str:
    g = mats["gates"]
    beta = mats["beta"]
    K = mats["K"]
    n = mats["n"]
    sign_gate = mats["sign_gate"]
    gamma = mats["gamma"]
    basal = mats["basal"]
    Vmax = mats["Vmax"]

    P = len(names)
    lines = []
    lines.append("# Discovered Hill-ODE with differentiable topology\n")
    for i in range(P):
        tgt = names[i]
        lines.append(f"d({tgt})/dt = -gamma[{tgt}]*{tgt} + basal[{tgt}] + Add + Mult")
        lines.append(f"  gamma[{tgt}] = {gamma[i]:.6g}, basal[{tgt}] = {basal[i]:.6g}, Vmax[{tgt}] = {Vmax[i]:.6g}")

        inc = []
        for j in range(P):
            if j == i:
                continue
            if g[j, i] >= gate_thr:
                inc.append((g[j, i], abs(beta[j, i]), j))
        inc.sort(reverse=True)
        inc = inc[:topk]

        if len(inc) == 0:
            lines.append("  (no incoming edges above threshold)")
            lines.append("")
            continue

        lines.append("  Additive terms (top edges):")
        for gate_val, _, j in inc:
            src = names[j]
            sgn = "act" if sign_gate[j, i] >= 0.5 else "inh"
            if sgn == "act":
                h_form = f"H_act({src};K={K[j,i]:.3g},n={n[j,i]:.3g}) = {src}^n/(K^n+{src}^n)"
            else:
                h_form = f"H_inh({src};K={K[j,i]:.3g},n={n[j,i]:.3g}) = K^n/(K^n+{src}^n)"
            lines.append(f"    + gate({src}->{tgt})*beta({src}->{tgt})*{h_form}")
            lines.append(f"      gate={gate_val:.3f}, beta={beta[j,i]:.3g}, sign={sgn}")
        lines.append("")

    return "\n".join(lines)

# ============================================================
# Network visualization (legend outside axes)
# ============================================================
def _get_layout(G: nx.Graph, seed: int = 42) -> dict:
    pos = nx.spring_layout(G, k=2.2 / np.sqrt(max(len(G), 1)), iterations=200, seed=seed)
    return _remove_overlaps(pos)

def _remove_overlaps(pos: dict, min_dist: float = 0.18, iterations: int = 300) -> dict:
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

def build_graph_from_edges(edge_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, r in edge_df.iterrows():
        G.add_edge(r["source"], r["target"], gate=r["gate"], beta=r["beta"], sign=r["sign"])
    return G

def plot_network(G: nx.DiGraph, title: str, out_png: str):
    if len(G.nodes()) == 0 or len(G.edges()) == 0:
        print(f"  ⚠ Empty graph — skip plot: {title}")
        return

    fig, ax = plt.subplots(figsize=(28, 24))
    pos = _get_layout(G)

    n_act = sum(1 for _,_,d in G.edges(data=True) if d.get("sign","activation") == "activation")
    n_inh = len(G.edges()) - n_act

    for u, v, d in G.edges(data=True):
        sign = d.get("sign", "activation")
        w = float(abs(d.get("beta", 0.0))) + 0.1 * float(d.get("gate", 0.0))
        lw = max(1.2, 1.0 + 3.0 * min(w, 1.5))
        if sign == "activation":
            color = "#2E7D32"
            arrow_style = "->,head_length=0.6,head_width=0.35"
        else:
            color = "#C62828"
            arrow_style = "-|>,head_length=0.6,head_width=0.35"

        ax.annotate(
            "",
            xy=pos[v], xycoords="data",
            xytext=pos[u], textcoords="data",
            arrowprops=dict(
                arrowstyle=arrow_style,
                color=color,
                linewidth=lw,
                alpha=0.75,
                linestyle="solid",
                connectionstyle="arc3,rad=0.08",
                shrinkA=28, shrinkB=28,
            ),
        )

    for node in G.nodes():
        x, y = pos[node]
        ax.text(
            x, y, short_name_multiline(node),
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=classify_node_color(node),
                edgecolor="#666666",
                linewidth=1.5,
                alpha=0.95,
            ),
            zorder=5,
        )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, edgecolor="#666666", lw=1.2, label=name)
        for name, c in PATHWAY_COLORS.items()
    ] + [
        Line2D([0],[0], color="#2E7D32", lw=3, label=f"Activation ({n_act})"),
        Line2D([0],[0], color="#C62828", lw=3, label=f"Inhibition ({n_inh})"),
    ]

    fig.subplots_adjust(bottom=0.12)
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor="none",
        fancybox=True,
        handlelength=2.5,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="#333333")
    ax.axis("off")
    plt.tight_layout()
    save_fig(fig, out_png)

# ============================================================
# Trajectory plots (ODE rollout vs data)
# ============================================================
@torch.no_grad()
def plot_rollout_vs_data(
    model: HillODEFuncDARTS,
    t: np.ndarray,
    Y: np.ndarray,
    names: List[str],
    out_png: str,
    n_cols: int = 5,
    max_panels: int = 25,
    ode_method: str = "tsit5",
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-3,
    p_mins: np.ndarray | None = None,
    p_scales: np.ndarray | None = None,
):
    device = next(model.parameters()).device
    tt = torch.tensor(t, dtype=torch.float32, device=device)
    y_true_t = torch.tensor(Y, dtype=torch.float32, device=device)
    x0 = y_true_t[0]

    y_pred = torchode_solve(
        model, x0, tt,
        method=ode_method,
        atol=ode_atol,
        rtol=ode_rtol,
    ).cpu().numpy()

    # Denormalize for plotting in original scale
    Y_plot = Y
    y_pred_plot = y_pred
    if p_mins is not None and p_scales is not None:
        Y_plot = denormalize_proteins(Y, p_mins, p_scales)
        y_pred_plot = denormalize_proteins(y_pred, p_mins, p_scales)

    P = Y.shape[1]
    P_show = min(P, max_panels)
    n_rows = int(math.ceil(P_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)
    for k in range(P_show):
        r = k // n_cols
        c = k % n_cols
        ax = axes[r][c]
        ax.plot(t, Y_plot[:, k], "o", ms=4, mfc="white", mec="black", label="Data")
        ax.plot(t, y_pred_plot[:, k], "-", lw=2, color="#D84315", label="ODE")
        ax.set_title(names[k], fontsize=10, fontweight="bold")
        ax.set_xscale("log" if (t.min() > 0 and (t.max()/max(t.min(),1e-9) > 50)) else "linear")
        ax.grid(True, alpha=0.2)

    for k in range(P_show, n_rows*n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)
    fig.suptitle("ODE rollout vs data (selected proteins)", fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.06)
    plt.tight_layout()
    save_fig(fig, out_png)

# ============================================================
# Main pipeline per source
# ============================================================
def run_source(
    data_dir: str,
    output_dir: str,
    source: str,
    cfg: TrainConfig,
    device: str,
    use_additive: bool,
    use_multiplicative: bool,
    gate_thr: float,
    consensus_freq: float,
    normalize: bool = True,
):
    conds = discover_conditions(data_dir, source)
    if not conds:
        raise FileNotFoundError(f"No files for source={source} in {data_dir}")

    out_src = os.path.join(output_dir, source)
    os.makedirs(out_src, exist_ok=True)

    per_cond_mats: Dict[str, Dict[str, np.ndarray]] = {}
    proteins_ref: List[str] | None = None

    print(f"\n{'='*72}\nSOURCE={source} | conditions={len(conds)}\n{'='*72}")

    for ci, (label, fp) in enumerate(conds):
        print(f"\n-- Condition {ci+1}/{len(conds)}: {label}")
        t, X_raw, proteins = load_ts_csv(fp)
        if proteins_ref is None:
            proteins_ref = proteins
        else:
            assert proteins == proteins_ref, "Protein order mismatch across conditions"

        # Per-protein normalization
        if normalize:
            X, p_mins, p_scales = normalize_proteins(X_raw)
            print(f"    Normalized {len(proteins)} proteins to [0,1] (scale range: {p_scales.min():.3g} – {p_scales.max():.3g})")
        else:
            X = X_raw
            p_mins = np.zeros(X.shape[1])
            p_scales = np.ones(X.shape[1])

        tt = torch.tensor(t, dtype=torch.float32)
        YY = torch.tensor(X, dtype=torch.float32)

        P = X.shape[1]
        model = HillODEFuncDARTS(
            P=P,
            use_additive=use_additive,
            use_multiplicative=use_multiplicative,
            init_gate_bias=-2.0,
            gate_temperature=cfg.gate_temp_start,
        ).to(device=device, dtype=torch.float32)

        train_one_condition(model, tt, YY, cfg)

        mats = export_model_to_matrices(model)
        per_cond_mats[label] = mats

        # save per-condition outputs
        cond_dir = os.path.join(out_src, "per_condition", label)
        os.makedirs(cond_dir, exist_ok=True)

        pd.DataFrame(mats["gates"], index=proteins_ref, columns=proteins_ref).to_csv(os.path.join(cond_dir, "GATES.csv"))
        pd.DataFrame(mats["beta"], index=proteins_ref, columns=proteins_ref).to_csv(os.path.join(cond_dir, "BETA.csv"))
        pd.DataFrame(mats["K"], index=proteins_ref, columns=proteins_ref).to_csv(os.path.join(cond_dir, "K.csv"))
        pd.DataFrame(mats["n"], index=proteins_ref, columns=proteins_ref).to_csv(os.path.join(cond_dir, "n.csv"))

        edges = export_edges_from_matrices(mats, proteins_ref, thr_gate=gate_thr)
        edges.to_csv(os.path.join(cond_dir, "edges.csv"), index=False)

        eq_txt = equations_text(mats, proteins_ref, topk=6, gate_thr=gate_thr)
        with open(os.path.join(cond_dir, "equations.txt"), "w", encoding="utf-8") as f:
            f.write(eq_txt)

        if len(edges) > 0:
            G = build_graph_from_edges(edges)
            plot_network(G, f"DARTS-Hill Discovered Network ({source}) | {label}", os.path.join(cond_dir, "network.png"))

        plot_rollout_vs_data(
            model, t, X, proteins_ref,
            os.path.join(cond_dir, "rollout_vs_data.png"),
            ode_method=cfg.ode_method,
            ode_atol=cfg.ode_atol,
            ode_rtol=cfg.ode_rtol,
            p_mins=p_mins if normalize else None,
            p_scales=p_scales if normalize else None,
        )

    assert proteins_ref is not None

    # -------------------
    # consensus across conditions
    # -------------------
    W_eff = {}
    G_gate = {}
    for label, mats in per_cond_mats.items():
        W_eff[label] = mats["gates"] * mats["beta"]
        G_gate[label] = mats["gates"]

    W_cons = consensus_matrix(W_eff, min_freq=consensus_freq)
    G_cons = consensus_matrix(G_gate, min_freq=consensus_freq)

    pd.DataFrame(W_cons, index=proteins_ref, columns=proteins_ref).to_csv(os.path.join(out_src, "W_consensus.csv"))
    pd.DataFrame(G_cons, index=proteins_ref, columns=proteins_ref).to_csv(os.path.join(out_src, "GATES_consensus.csv"))

    # consensus edges (display only)
    mats_cons = {
        "gates": G_cons,
        "beta": W_cons / np.clip(G_cons, 1e-12, None),
        "K": np.zeros_like(G_cons),
        "n": np.zeros_like(G_cons),
        "sign_gate": np.ones_like(G_cons),
        "gamma": np.zeros(len(proteins_ref)),
        "basal": np.zeros(len(proteins_ref)),
        "Vmax": np.zeros(len(proteins_ref)),
    }
    edges_cons = export_edges_from_matrices(mats_cons, proteins_ref, thr_gate=gate_thr, thr_abs_beta=0.0)
    edges_cons.to_csv(os.path.join(out_src, "edges_consensus.csv"), index=False)

    if len(edges_cons) > 0:
        Gc = build_graph_from_edges(edges_cons)
        plot_network(Gc, f"DARTS-Hill Consensus Network ({source})", os.path.join(out_src, "network_consensus.png"))

    print(f"\n✓ {source} done. Consensus saved under: {out_src}")
    return out_src

# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser("DARTS-Hill Hybrid Dual-Engine Discovery")
    parser.add_argument("--data_dir", type=str, default="grn_ready_data",
                        help="Directory containing ts_{teacher,student}_pred_*.csv")
    parser.add_argument("--output_dir", type=str, default="results/darts_hill",
                        help="Output directory")
    parser.add_argument("--source", type=str, default="both", choices=["teacher","student","both"])
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")

    # training knobs
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--lambda_beta", type=float, default=0.0)
    parser.add_argument("--temp_start", type=float, default=2.0)
    parser.add_argument("--temp_end", type=float, default=0.01)

    # export knobs
    parser.add_argument("--gate_thr", type=float, default=0.3)
    parser.add_argument("--consensus_freq", type=float, default=0.0)

    # model form
    parser.add_argument("--use_add", action="store_true", help="Use additive Hill sum term")
    parser.add_argument("--use_mult", action="store_true", help="Use multiplicative Hill product term")

    # hybrid dual-engine
    parser.add_argument("--phase1_frac", type=float, default=0.2,
                        help="Fraction of epochs for Phase 1 (derivative matching)")

    # torchode integrator knobs (Phase 2)
    parser.add_argument("--ode_method", type=str, default="tsit5", choices=["tsit5", "dopri5"],
                        help="ODE solver method (tsit5=5th order, dopri5=Dormand-Prince 5th order)")
    parser.add_argument("--ode_atol", type=float, default=1e-5,
                        help="Absolute tolerance for adaptive step controller")
    parser.add_argument("--ode_rtol", type=float, default=1e-3,
                        help="Relative tolerance for adaptive step controller")

    # early stopping
    parser.add_argument("--early_stop_patience", type=int, default=100,
                        help="Epochs without improvement before early stop")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-6,
                        help="Minimum loss improvement to reset early-stop counter")

    # temporal weighting
    parser.add_argument("--time_weight_alpha", type=float, default=1.0,
                        help="Ramp exponent for temporal weighting (0=uniform, 1=linear, >1=stronger late-weighting)")
    parser.add_argument("--time_weight_min", type=float, default=0.1,
                        help="Floor weight for earliest time point (0-1)")

    # normalization
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable per-protein min-max normalization to [0,1]")

    args = parser.parse_args()

    # default both on if user didn't specify
    use_add = args.use_add or (not args.use_add and not args.use_mult)
    use_mult = args.use_mult or (not args.use_add and not args.use_mult)

    cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        lambda_sparse=args.lambda_sparse,
        lambda_beta=args.lambda_beta,
        gate_temp_start=args.temp_start,
        gate_temp_end=args.temp_end,
        phase1_frac=args.phase1_frac,
        ode_method=args.ode_method,
        ode_atol=args.ode_atol,
        ode_rtol=args.ode_rtol,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        gate_thr=args.gate_thr,
        time_weight_alpha=args.time_weight_alpha,
        time_weight_min=args.time_weight_min,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    out_teacher = None
    out_student = None

    if args.source in ["teacher", "both"]:
        out_teacher = run_source(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            source="teacher",
            cfg=cfg,
            device=args.device,
            use_additive=use_add,
            use_multiplicative=use_mult,
            gate_thr=args.gate_thr,
            consensus_freq=args.consensus_freq,
            normalize=not args.no_normalize,
        )
    if args.source in ["student", "both"]:
        out_student = run_source(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            source="student",
            cfg=cfg,
            device=args.device,
            use_additive=use_add,
            use_multiplicative=use_mult,
            gate_thr=args.gate_thr,
            consensus_freq=args.consensus_freq,
            normalize=not args.no_normalize,
        )

    print("\n✅ All done.")
    if out_teacher and out_student:
        print(f"Teacher outputs: {out_teacher}")
        print(f"Student outputs: {out_student}")

if __name__ == "__main__":
    main()