"""
Hill Function Network Fitting for Phosphoproteomic Signalling Networks.

Fitting modes
-------------
**DE_AUTO mode** (best-RMSE model selection — for teacher)::

    Fits all 15 non-empty subsets of {add, mult, ratio, ss}
    components per protein, selects the best by RMSE.
    No parsimony penalty — intentionally allows overfitting
    for maximum accuracy.

    Reports which combo won for each protein.

**DAE_AUTO mode** (mixed Hill ODE-auto + algebraic — for student)::

    Fast proteins → best-RMSE auto-select ODE (15 combos)
    Slow proteins → algebraic (multiplicative / ratio / additive)

Mode is determined automatically by source:
    teacher → de_auto
    student → dae_auto

Input
-----
    Network structure : results/sindy/W_consensus_{source}.csv
    Time series       : grn_ready_data/ts_{source}_pred_*.csv

Output
------
    results/hill/
        hill_params_{source}.csv              — fitted edge parameters
        fit_quality_{source}.csv              — per-protein R², RMSE
        fit_quality_{source}.png              — quality summary plots
        hill_params_heatmap_{source}.png      — w / K / n heatmaps
        hill_network_{source}.png             — directed network graph
"""

from __future__ import annotations

import argparse
import os
import warnings
from itertools import combinations as _combinations
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import numba as nb
import pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
from torchdiffeq import odeint

from sindy_network import (
    DISPLAY_NAMES_MULTILINE,
    PATHWAY_COLORS,
    NODE_PATHWAY,
    _PRIOR_EDGE_SET,
    _save_fig,
    load_ts_csv,
    discover_conditions,
    weight_matrix_to_digraph,
    plot_sindy_network,
)

warnings.filterwarnings("ignore", category=UserWarning)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ============================================================================
# Hill Functions  — vectorized + JIT-compiled
# ============================================================================
@nb.njit(cache=True, fastmath=True)
def hill_act_nb(x: np.ndarray, K: float, n: float) -> np.ndarray:
    """Activation Hill (numba):  x^n / (K^n + x^n). Fused loop, no alloc."""
    out = np.empty(x.shape[0])
    Kn = K ** n
    for i in range(x.shape[0]):
        xn = (x[i] if x[i] > 0.0 else 1e-12) ** n
        out[i] = xn / (Kn + xn)
    return out


@nb.njit(cache=True, fastmath=True)
def hill_inh_nb(x: np.ndarray, K: float, n: float) -> np.ndarray:
    """Inhibition Hill (numba):  K^n / (K^n + x^n)."""
    out = np.empty(x.shape[0])
    Kn = K ** n
    for i in range(x.shape[0]):
        xn = (x[i] if x[i] > 0.0 else 1e-12) ** n
        out[i] = Kn / (Kn + xn)
    return out


def hill_act(x: np.ndarray, K: float, n: float) -> np.ndarray:
    """Activation Hill:  x^n / (K^n + x^n)."""
    xn = np.power(np.maximum(x, 0.0) + 1e-12, n)
    Kn = np.power(np.maximum(K, 1e-12), n)
    return xn / (Kn + xn)


def hill_inh(x: np.ndarray, K: float, n: float) -> np.ndarray:
    """Inhibition Hill:  K^n / (K^n + x^n)."""
    xn = np.power(np.maximum(x, 0.0) + 1e-12, n)
    Kn = np.power(np.maximum(K, 1e-12), n)
    return Kn / (Kn + xn)


# ============================================================================
# Load Network Structure from SINDy Consensus
# ============================================================================
def load_sindy_network(consensus_csv: str):
    """Load W_consensus.  Returns (W, protein_names, edge_list).

    edge_list : [(source_idx, target_idx, weight, sign_str), …]
    """
    df = pd.read_csv(consensus_csv, index_col=0)
    proteins = list(df.index)
    W = df.values.astype(float)

    edges = []
    P = len(proteins)
    for j in range(P):
        for i in range(P):
            if j == i:
                continue
            if abs(W[j, i]) > 1e-12:
                sign_str = "activation" if W[j, i] > 0 else "inhibition"
                edges.append((j, i, W[j, i], sign_str))

    return W, proteins, edges


# ============================================================================
# Helper
# ============================================================================
def _compute_r2_rmse(residuals: np.ndarray, y: np.ndarray):
    # Use vectorized numpy — avoid redundant passes
    ss_res = np.dot(residuals, residuals)
    y_mean = y.mean()
    y_c = y - y_mean
    ss_tot = np.dot(y_c, y_c)
    r2   = 1.0 - ss_res / max(ss_tot, 1e-15)
    rmse = np.sqrt(ss_res / max(len(residuals), 1))
    return r2, rmse


# ============================================================================
# Algebraic Hill Fitting  (direct level fitting, no dX/dt)
# ============================================================================
def _fit_single_protein_algebraic(target_idx: int,
                                  reg_indices: list[int],
                                  reg_signs: list[int],
                                  sindy_weights: list[float],
                                  X_pool: np.ndarray,
                                  n_restarts: int = 3) -> dict:
    """Fit algebraic Hill model for one protein:

        X_i = Σ_j β_ji · f(X_j; K, n)  +  b

    Two-stage: Stage 1 fixes n=2, Stage 2 refines all.
    """
    M = len(reg_indices)
    N_full = X_pool.shape[0]

    # Subsample if dataset is very large  (>20× the number of params)
    n_params = 3 * M + 1
    max_samples = max(20 * n_params, 500)
    if N_full > max_samples:
        rng_sub = np.random.RandomState(target_idx)
        idx = rng_sub.choice(N_full, max_samples, replace=False)
        X_regs = X_pool[np.ix_(idx, reg_indices)]
        y_tgt  = X_pool[idx, target_idx]
    else:
        X_regs = X_pool[:, reg_indices]
        y_tgt  = X_pool[:, target_idx]

    N = len(y_tgt)

    # per-regulator stats
    med_regs = np.array([np.median(np.abs(X_regs[:, k])) + 1e-6
                         for k in range(M)])
    max_regs = np.array([np.max(np.abs(X_regs[:, k])) + 1e-6
                         for k in range(M)])
    y_med = np.median(np.abs(y_tgt)) + 1e-6

    # ── residual: full (β, K, n, b) ──
    # param layout: [β0, K0, n0, β1, K1, n1, ..., b]
    def residuals_full(params):
        pred = np.full(N, params[3 * M])              # basal
        for k in range(M):
            beta, K, n_h = params[3*k], params[3*k+1], params[3*k+2]
            if reg_signs[k] > 0:
                pred += beta * hill_act(X_regs[:, k], K, n_h)
            else:
                pred += beta * hill_inh(X_regs[:, k], K, n_h)
        return pred - y_tgt

    # ── Stage 1: fix n, param layout: [β0, K0, β1, K1, ..., b] ──
    def residuals_fixed_n(params, n_fixed):
        pred = np.full(N, params[2 * M])              # basal
        for k in range(M):
            beta, K = params[2*k], params[2*k+1]
            if reg_signs[k] > 0:
                pred += beta * hill_act(X_regs[:, k], K, n_fixed)
            else:
                pred += beta * hill_inh(X_regs[:, k], K, n_fixed)
        return pred - y_tgt

    # ── bounds ──
    K_ub = np.clip(max_regs * 3, 0.01, None)
    beta_ub = np.abs(y_tgt).max() * 3 + 1e-3  # β at most a few × max level

    def _bounds_full():
        lb = np.zeros(3 * M + 1)
        ub = np.full(3 * M + 1, np.inf)
        for k in range(M):
            lb[3*k]     = 0.0;    ub[3*k]     = beta_ub     # β
            lb[3*k + 1] = 1e-4;   ub[3*k + 1] = K_ub[k]    # K
            lb[3*k + 2] = 0.5;    ub[3*k + 2] = 5.0        # n
        lb[3*M] = -beta_ub;  ub[3*M] = beta_ub              # basal (may be neg)
        return lb, ub

    def _bounds_s1():
        lb = np.zeros(2 * M + 1)
        ub = np.full(2 * M + 1, np.inf)
        for k in range(M):
            lb[2*k]     = 0.0;    ub[2*k]     = beta_ub
            lb[2*k + 1] = 1e-4;   ub[2*k + 1] = K_ub[k]
        lb[2*M] = -beta_ub;  ub[2*M] = beta_ub
        return lb, ub

    rng = np.random.RandomState(42)
    best_cost = np.inf
    best_popt = None

    lb_full, ub_full = _bounds_full()
    lb_s1,   ub_s1   = _bounds_s1()

    for restart in range(n_restarts):
        jitter = 1.0 + 0.3 * rng.randn(M)
        jitter = np.clip(jitter, 0.5, 2.0)

        K_init = med_regs * jitter
        K_init = np.clip(K_init, lb_s1[1::2][:M] + 1e-6, ub_s1[1::2][:M] - 1e-6)
        n_stage1 = 2.0

        # Rough init: at half-max, Hill = 0.5, so  β ≈ 2 × (fraction of y explained)
        beta_init = np.full(M, y_med / max(M, 1))
        beta_init = np.clip(beta_init, 1e-5, beta_ub - 1e-3)

        # Stage 1
        p0_s1 = np.zeros(2 * M + 1)
        for k in range(M):
            p0_s1[2*k]     = beta_init[k]
            p0_s1[2*k + 1] = K_init[k]
        p0_s1[2*M] = np.mean(y_tgt) * 0.5  # basal ~ half mean
        p0_s1 = np.clip(p0_s1, lb_s1 + 1e-8, ub_s1 - 1e-8)

        try:
            res1 = least_squares(lambda p: residuals_fixed_n(p, n_stage1),
                                 p0_s1, bounds=(lb_s1, ub_s1),
                                 method="trf", max_nfev=10000,
                                 ftol=1e-10, xtol=1e-10)
        except Exception:
            continue

        # Stage 2
        p0_full = np.zeros(3 * M + 1)
        for k in range(M):
            p0_full[3*k]     = res1.x[2*k]
            p0_full[3*k + 1] = res1.x[2*k + 1]
            p0_full[3*k + 2] = n_stage1
        p0_full[3*M] = res1.x[2*M]
        p0_full = np.clip(p0_full, lb_full + 1e-8, ub_full - 1e-8)

        try:
            res2 = least_squares(residuals_full, p0_full,
                                 bounds=(lb_full, ub_full),
                                 method="trf", max_nfev=20000,
                                 ftol=1e-12, xtol=1e-12)
        except Exception:
            continue

        if res2.cost < best_cost:
            best_cost = res2.cost
            best_popt = res2.x

    if best_popt is None:
        raise RuntimeError("All restarts failed")

    popt = best_popt

    # Evaluate on FULL data for accurate R² (fitting may have used subsample)
    X_regs_full = X_pool[:, reg_indices]
    y_full = X_pool[:, target_idx]
    pred_full = np.full(N_full, popt[3 * M])
    for k in range(M):
        beta, K, n_h = popt[3*k], popt[3*k+1], popt[3*k+2]
        if reg_signs[k] > 0:
            pred_full += beta * hill_act(X_regs_full[:, k], K, n_h)
        else:
            pred_full += beta * hill_inh(X_regs_full[:, k], K, n_h)
    r2, rmse = _compute_r2_rmse(pred_full - y_full, y_full)

    edge_params = []
    for k in range(M):
        edge_params.append({
            "w":    popt[3*k],       # β stored under key "w" for compat
            "K":    popt[3*k + 1],
            "n":    popt[3*k + 2],
            "sign": "activation" if reg_signs[k] > 0 else "inhibition",
        })

    return {
        "edge_params": edge_params,
        "gamma": 0.0,               # no degradation in algebraic mode
        "basal": popt[3 * M],
        "r2": r2,
        "rmse": rmse,
    }


# ============================================================================
# Multiplicative Fractional Occupancy Hill Fitting
# ============================================================================
def _fit_single_protein_multiplicative(target_idx: int,
                                       reg_indices: list[int],
                                       reg_signs: list[int],
                                       sindy_weights: list[float],
                                       X_pool: np.ndarray,
                                       n_restarts: int = 3) -> dict:
    """Fit multiplicative fractional occupancy Hill model:

        X_i = basal + Vmax × Π_{act} hill_act(X_j) × Π_{inh} hill_inh(X_k)

    All Hill functions are probability multipliers in [0, 1].
    A single strong inhibitor (Hill ≈ 0) vetoes the entire product.

    Parameters: 2M + 2  (vs 3M + 1 for additive algebraic).
    """
    M = len(reg_indices)
    N_full = X_pool.shape[0]

    # Subsample if dataset is very large
    n_params = 2 * M + 2
    max_samples = max(20 * n_params, 500)
    if N_full > max_samples:
        rng_sub = np.random.RandomState(target_idx)
        idx = rng_sub.choice(N_full, max_samples, replace=False)
        X_regs = X_pool[np.ix_(idx, reg_indices)]
        y_tgt = X_pool[idx, target_idx]
    else:
        X_regs = X_pool[:, reg_indices]
        y_tgt = X_pool[:, target_idx]

    N = len(y_tgt)

    med_regs = np.array([np.median(np.abs(X_regs[:, k])) + 1e-6
                         for k in range(M)])
    max_regs = np.array([np.max(np.abs(X_regs[:, k])) + 1e-6
                         for k in range(M)])
    y_max = np.max(np.abs(y_tgt)) + 1e-6

    # ── full residual: params = [K0, n0, K1, n1, ..., Vmax, basal] ──
    def residuals_full(params):
        product = np.ones(N)
        for k in range(M):
            K, n_h = params[2 * k], params[2 * k + 1]
            if reg_signs[k] > 0:
                product *= hill_act(X_regs[:, k], K, n_h)
            else:
                product *= hill_inh(X_regs[:, k], K, n_h)
        return params[2 * M + 1] + params[2 * M] * product - y_tgt

    # ── Stage 1: n fixed, params = [K0, K1, ..., Vmax, basal] ──
    def residuals_fixed_n(params, n_fixed):
        product = np.ones(N)
        for k in range(M):
            K = params[k]
            if reg_signs[k] > 0:
                product *= hill_act(X_regs[:, k], K, n_fixed)
            else:
                product *= hill_inh(X_regs[:, k], K, n_fixed)
        return params[M + 1] + params[M] * product - y_tgt

    # ── bounds ──
    K_ub = np.clip(max_regs * 3, 0.01, None)
    Vmax_ub = y_max * 5

    def _bounds_full():
        lb = np.zeros(2 * M + 2)
        ub = np.full(2 * M + 2, np.inf)
        for k in range(M):
            lb[2 * k] = 1e-4;      ub[2 * k] = K_ub[k]       # K
            lb[2 * k + 1] = 0.5;   ub[2 * k + 1] = 5.0       # n
        lb[2 * M] = 0.0;           ub[2 * M] = Vmax_ub        # Vmax >= 0
        lb[2 * M + 1] = -Vmax_ub;  ub[2 * M + 1] = Vmax_ub   # basal
        return lb, ub

    def _bounds_s1():
        lb = np.zeros(M + 2)
        ub = np.full(M + 2, np.inf)
        for k in range(M):
            lb[k] = 1e-4;  ub[k] = K_ub[k]
        lb[M] = 0.0;           ub[M] = Vmax_ub
        lb[M + 1] = -Vmax_ub;  ub[M + 1] = Vmax_ub
        return lb, ub

    rng = np.random.RandomState(42)
    best_cost = np.inf
    best_popt = None

    lb_full, ub_full = _bounds_full()
    lb_s1, ub_s1 = _bounds_s1()

    for restart in range(n_restarts):
        jitter = 1.0 + 0.3 * rng.randn(M)
        jitter = np.clip(jitter, 0.5, 2.0)

        K_init = med_regs * jitter
        K_init = np.clip(K_init, lb_s1[:M] + 1e-6, ub_s1[:M] - 1e-6)
        n_stage1 = 2.0

        Vmax_init = np.max(y_tgt) - np.min(y_tgt) + 1e-3
        Vmax_init = np.clip(Vmax_init, 1e-3, Vmax_ub - 1e-3)
        basal_init = np.clip(np.min(y_tgt), -Vmax_ub + 1e-3, Vmax_ub - 1e-3)

        # Stage 1: fix n=2
        p0_s1 = np.zeros(M + 2)
        p0_s1[:M] = K_init
        p0_s1[M] = Vmax_init
        p0_s1[M + 1] = basal_init
        p0_s1 = np.clip(p0_s1, lb_s1 + 1e-8, ub_s1 - 1e-8)

        try:
            res1 = least_squares(lambda p: residuals_fixed_n(p, n_stage1),
                                 p0_s1, bounds=(lb_s1, ub_s1),
                                 method="trf", max_nfev=10000,
                                 ftol=1e-10, xtol=1e-10)
        except Exception:
            continue

        # Stage 2: refine all including n
        p0_full = np.zeros(2 * M + 2)
        for k in range(M):
            p0_full[2 * k] = res1.x[k]          # K from stage 1
            p0_full[2 * k + 1] = n_stage1       # n = 2 to start
        p0_full[2 * M] = res1.x[M]              # Vmax
        p0_full[2 * M + 1] = res1.x[M + 1]      # basal
        p0_full = np.clip(p0_full, lb_full + 1e-8, ub_full - 1e-8)

        try:
            res2 = least_squares(residuals_full, p0_full,
                                 bounds=(lb_full, ub_full),
                                 method="trf", max_nfev=20000,
                                 ftol=1e-12, xtol=1e-12)
        except Exception:
            continue

        if res2.cost < best_cost:
            best_cost = res2.cost
            best_popt = res2.x

    if best_popt is None:
        raise RuntimeError("All restarts failed")

    popt = best_popt

    # Evaluate on FULL data for accurate R²
    X_regs_full = X_pool[:, reg_indices]
    y_full = X_pool[:, target_idx]
    product = np.ones(N_full)
    for k in range(M):
        K, n_h = popt[2 * k], popt[2 * k + 1]
        if reg_signs[k] > 0:
            product *= hill_act(X_regs_full[:, k], K, n_h)
        else:
            product *= hill_inh(X_regs_full[:, k], K, n_h)
    pred_full = popt[2 * M + 1] + popt[2 * M] * product
    r2, rmse = _compute_r2_rmse(pred_full - y_full, y_full)

    Vmax = popt[2 * M]
    edge_params = []
    for k in range(M):
        edge_params.append({
            "w": Vmax,                 # Vmax stored as "w" for compat
            "K": popt[2 * k],
            "n": popt[2 * k + 1],
            "sign": "activation" if reg_signs[k] > 0 else "inhibition",
        })

    return {
        "edge_params": edge_params,
        "vmax": Vmax,
        "gamma": 0.0,
        "basal": popt[2 * M + 1],
        "r2": r2,
        "rmse": rmse,
    }


# ============================================================================
# Kinase-Phosphatase Ratio Hill Model
# ============================================================================
def _fit_single_protein_ratio(target_idx: int,
                              reg_indices: list[int],
                              reg_signs: list[int],
                              sindy_weights: list[float],
                              X_pool: np.ndarray,
                              n_restarts: int = 3,
                              w_pool: np.ndarray | None = None) -> dict:
    """Fit kinase-phosphatase ratio Hill model:

        X_target = Vmax × V_kin / (V_kin + V_phos)

    where:
        V_kin  = w0 + Σ_{j∈Act} w_j × X_j^n / (K_j^n + X_j^n)
        V_phos = γ0 + Σ_{k∈Inh} v_k × X_k^n / (K_k^n + X_k^n)

    Activators contribute to phosphorylation rate (numerator).
    Inhibitors contribute to dephosphorylation rate (denominator).
    Output is naturally bounded in [0, Vmax].

    Parameters: 3M + 3  (w_k, K_k, n_k per regulator + Vmax, w0, γ0).
    """
    M = len(reg_indices)
    N_full = X_pool.shape[0]
    act_mask = np.array([s > 0 for s in reg_signs])  # (M,)

    # Subsample if dataset is very large
    n_params = 3 * M + 3
    max_samples = max(20 * n_params, 500)
    if N_full > max_samples:
        rng_sub = np.random.RandomState(target_idx)
        idx = rng_sub.choice(N_full, max_samples, replace=False)
        X_regs = X_pool[np.ix_(idx, reg_indices)]
        y_tgt = X_pool[idx, target_idx]
        w_sub = w_pool[idx] if w_pool is not None else np.ones(max_samples)
    else:
        X_regs = X_pool[:, reg_indices]
        y_tgt = X_pool[:, target_idx]
        w_sub = w_pool if w_pool is not None else np.ones(N_full)

    N = len(y_tgt)
    sqrt_w = np.sqrt(w_sub)
    X_regs = np.maximum(X_regs, 0.0) + 1e-12

    med_regs = np.array([np.median(X_regs[:, k]) for k in range(M)])
    max_regs = np.array([np.max(X_regs[:, k]) + 1e-6 for k in range(M)])
    y_max = np.max(np.abs(y_tgt)) + 1e-6

    # ── Full residual: [w_0, K_0, n_0, ..., Vmax, w0_leak, γ0] ──
    def residuals_full(params):
        Vmax_p = params[3 * M]
        w0_p = params[3 * M + 1]
        g0_p = params[3 * M + 2]
        V_kin = np.full(N, w0_p)
        V_phos = np.full(N, g0_p)
        for k in range(M):
            h = hill_act(X_regs[:, k], params[3*k+1], params[3*k+2])
            if act_mask[k]:
                V_kin += params[3*k] * h
            else:
                V_phos += params[3*k] * h
        pred = Vmax_p * V_kin / (V_kin + V_phos + 1e-15)
        return sqrt_w * (pred - y_tgt)

    # ── Stage 1: n=2, weights=1, fit [K_0, ..., K_{M-1}, Vmax, w0, γ0] ──
    def residuals_s1(params_s1):
        Vmax_p = params_s1[M]
        w0_p = params_s1[M + 1]
        g0_p = params_s1[M + 2]
        V_kin = np.full(N, w0_p)
        V_phos = np.full(N, g0_p)
        for k in range(M):
            h = hill_act(X_regs[:, k], params_s1[k], 2.0)
            if act_mask[k]:
                V_kin += h
            else:
                V_phos += h
        pred = Vmax_p * V_kin / (V_kin + V_phos + 1e-15)
        return sqrt_w * (pred - y_tgt)

    # ── Stage 2: n=2, free weights, [w_0, K_0, ..., Vmax, w0, γ0] ──
    def residuals_s2(params_s2):
        Vmax_p = params_s2[2 * M]
        w0_p = params_s2[2 * M + 1]
        g0_p = params_s2[2 * M + 2]
        V_kin = np.full(N, w0_p)
        V_phos = np.full(N, g0_p)
        for k in range(M):
            h = hill_act(X_regs[:, k], params_s2[2*k+1], 2.0)
            if act_mask[k]:
                V_kin += params_s2[2*k] * h
            else:
                V_phos += params_s2[2*k] * h
        pred = Vmax_p * V_kin / (V_kin + V_phos + 1e-15)
        return sqrt_w * (pred - y_tgt)

    # ── Bounds ──
    K_ub = np.clip(max_regs * 3, 0.01, None)

    def _bounds_s1():
        lb = np.zeros(M + 3);  ub = np.full(M + 3, np.inf)
        for k in range(M):
            lb[k] = 1e-4;  ub[k] = K_ub[k]
        lb[M] = 1e-4;    ub[M] = y_max * 5      # Vmax
        lb[M+1] = 1e-6;  ub[M+1] = 10.0          # w0
        lb[M+2] = 1e-6;  ub[M+2] = 10.0          # γ0
        return lb, ub

    def _bounds_s2():
        lb = np.zeros(2*M + 3);  ub = np.full(2*M + 3, np.inf)
        for k in range(M):
            lb[2*k] = 1e-4;    ub[2*k] = 20.0        # w_k
            lb[2*k+1] = 1e-4;  ub[2*k+1] = K_ub[k]   # K_k
        lb[2*M] = 1e-4;    ub[2*M] = y_max * 5       # Vmax
        lb[2*M+1] = 1e-6;  ub[2*M+1] = 10.0          # w0
        lb[2*M+2] = 1e-6;  ub[2*M+2] = 10.0          # γ0
        return lb, ub

    def _bounds_full():
        lb = np.zeros(3*M + 3);  ub = np.full(3*M + 3, np.inf)
        for k in range(M):
            lb[3*k] = 1e-4;    ub[3*k] = 20.0        # w_k
            lb[3*k+1] = 1e-4;  ub[3*k+1] = K_ub[k]   # K_k
            lb[3*k+2] = 0.5;   ub[3*k+2] = 5.0        # n_k
        lb[3*M] = 1e-4;    ub[3*M] = y_max * 5       # Vmax
        lb[3*M+1] = 1e-6;  ub[3*M+1] = 10.0          # w0
        lb[3*M+2] = 1e-6;  ub[3*M+2] = 10.0          # γ0
        return lb, ub

    rng = np.random.RandomState(42)
    best_cost = np.inf
    best_popt = None

    lb_s1, ub_s1 = _bounds_s1()
    lb_s2, ub_s2 = _bounds_s2()
    lb_full, ub_full = _bounds_full()

    for restart in range(n_restarts):
        jitter = 1.0 + 0.3 * rng.randn(M)
        jitter = np.clip(jitter, 0.5, 2.0)

        K_init = med_regs * jitter
        K_init = np.clip(K_init, lb_s1[:M] + 1e-6, ub_s1[:M] - 1e-6)
        Vmax_init = np.clip(np.max(y_tgt), 1e-3, y_max * 5 - 1e-3)

        # Stage 1: fix n=2, weights=1
        p0_s1 = np.zeros(M + 3)
        p0_s1[:M] = K_init
        p0_s1[M] = Vmax_init
        p0_s1[M+1] = 0.1   # w0
        p0_s1[M+2] = 0.1   # γ0
        p0_s1 = np.clip(p0_s1, lb_s1 + 1e-8, ub_s1 - 1e-8)

        try:
            res1 = least_squares(residuals_s1, p0_s1, bounds=(lb_s1, ub_s1),
                                 method="trf", max_nfev=10000,
                                 ftol=1e-10, xtol=1e-10)
        except Exception:
            continue

        # Stage 2: free weights, still n=2
        p0_s2 = np.zeros(2*M + 3)
        for k in range(M):
            p0_s2[2*k] = 1.0           # w_k init
            p0_s2[2*k+1] = res1.x[k]   # K_k from stage 1
        p0_s2[2*M] = res1.x[M]
        p0_s2[2*M+1] = res1.x[M+1]
        p0_s2[2*M+2] = res1.x[M+2]
        p0_s2 = np.clip(p0_s2, lb_s2 + 1e-8, ub_s2 - 1e-8)

        try:
            res2 = least_squares(residuals_s2, p0_s2, bounds=(lb_s2, ub_s2),
                                 method="trf", max_nfev=15000,
                                 ftol=1e-10, xtol=1e-10)
        except Exception:
            continue

        # Stage 3: full refinement including n
        p0_full = np.zeros(3*M + 3)
        for k in range(M):
            p0_full[3*k] = res2.x[2*k]       # w_k
            p0_full[3*k+1] = res2.x[2*k+1]   # K_k
            p0_full[3*k+2] = 2.0              # n_k = 2 to start
        p0_full[3*M] = res2.x[2*M]
        p0_full[3*M+1] = res2.x[2*M+1]
        p0_full[3*M+2] = res2.x[2*M+2]
        p0_full = np.clip(p0_full, lb_full + 1e-8, ub_full - 1e-8)

        try:
            res3 = least_squares(residuals_full, p0_full,
                                 bounds=(lb_full, ub_full),
                                 method="trf", max_nfev=20000,
                                 ftol=1e-12, xtol=1e-12)
        except Exception:
            continue

        if res3.cost < best_cost:
            best_cost = res3.cost
            best_popt = res3.x

    if best_popt is None:
        raise RuntimeError("All restarts failed")

    popt = best_popt

    # Evaluate on FULL data for accurate R²
    X_regs_full = np.maximum(X_pool[:, reg_indices], 0.0) + 1e-12
    y_full = X_pool[:, target_idx]

    Vmax = popt[3 * M]
    w0 = popt[3 * M + 1]
    g0 = popt[3 * M + 2]

    V_kin = np.full(N_full, w0)
    V_phos = np.full(N_full, g0)
    for k in range(M):
        h = hill_act(X_regs_full[:, k], popt[3*k+1], popt[3*k+2])
        if act_mask[k]:
            V_kin += popt[3*k] * h
        else:
            V_phos += popt[3*k] * h

    pred_full = Vmax * V_kin / (V_kin + V_phos + 1e-15)
    r2, rmse = _compute_r2_rmse(pred_full - y_full, y_full)

    edge_params = []
    for k in range(M):
        edge_params.append({
            "w": popt[3 * k],
            "K": popt[3 * k + 1],
            "n": popt[3 * k + 2],
            "sign": "activation" if reg_signs[k] > 0 else "inhibition",
        })

    return {
        "edge_params": edge_params,
        "model": "ratio",
        "vmax": Vmax,
        "w0": w0,
        "gamma0": g0,
        "gamma": 0.0,
        "basal": 0.0,
        "r2": r2,
        "rmse": rmse,
    }



# ============================================================================
# DAE (Mixed ODE + Algebraic) Student Model
# ============================================================================
def classify_fast_slow(all_X_raw: list[np.ndarray],
                       proteins: list[str],
                       fast_frac: float = 0.5
                       ) -> tuple[list[int], list[int], np.ndarray]:
    """Classify proteins as fast (ODE) or slow (algebraic) for DAE model.

    Dynamism score per protein: average CV of step-to-step changes across
    conditions.  High score → fast / intense → needs ODE.
    Low score  → slow / quasi-static → algebraic suffices.

    Returns (fast_indices, slow_indices, scores).
    """
    P = len(proteins)
    scores = np.zeros(P)

    for i in range(P):
        cv_sum = 0.0
        n_cond = 0
        for X_c in all_X_raw:
            traj = X_c[:, i]
            mean_val = np.mean(np.abs(traj)) + 1e-12
            diffs = np.diff(traj)
            cv_sum += np.sqrt(np.mean(diffs ** 2)) / mean_val
            n_cond += 1
        scores[i] = cv_sum / max(n_cond, 1)

    n_fast = max(1, int(round(fast_frac * P)))
    n_fast = min(n_fast, P - 1)  # keep at least 1 slow
    ranked = np.argsort(scores)[::-1]
    fast_set = set(ranked[:n_fast].tolist())

    fast_indices = sorted(i for i in range(P) if i in fast_set)
    slow_indices = sorted(i for i in range(P) if i not in fast_set)

    return fast_indices, slow_indices, scores


def fit_all_proteins_dae(W: np.ndarray,
                         proteins: list[str],
                         all_X_raw: list[np.ndarray],
                         all_times_raw: list[np.ndarray],
                         fast_frac: float = 0.5,
                         dae_alg_mode: str = "multiplicative",
                         early_weight_tau: float = 30.0,
                         cond_W_list: list[np.ndarray] | None = None,
                         n_restarts: int = 3) -> dict:
    """Fit DAE model: Hill ODE for fast, algebraic for slow.

    Parameters
    ----------
    fast_frac : fraction of proteins classified as fast (ODE)
    dae_alg_mode : algebraic model for slow variables
        "multiplicative", "ratio", or "algebraic"
    early_weight_tau : exponential decay τ for early-dynamics weighting
    cond_W_list : optional list of per-condition W matrices for causal masking
    n_restarts : number of random restarts for Hill ODE fitting
    """
    P = len(proteins)

    # ── classify ──
    fast_idx, slow_idx, scores = classify_fast_slow(all_X_raw, proteins,
                                                     fast_frac)
    fast_names = [proteins[i] for i in fast_idx]
    slow_names = [proteins[i] for i in slow_idx]

    print(f"\n  DAE classification (fast_frac={fast_frac:.2f}):")
    print(f"    Fast (Hill ODE):  {len(fast_idx)} — "
          + ", ".join(fast_names[:8]) + ("..." if len(fast_idx) > 8 else ""))
    print(f"    Slow (algebraic): {len(slow_idx)} — "
          + ", ".join(slow_names[:8]) + ("..." if len(slow_idx) > 8 else ""))

    results: dict[str, dict] = {}

    # ── fit SLOW proteins (algebraic) ──
    print(f"\n{'─' * 50}")
    print(f"  Fitting SLOW proteins ({dae_alg_mode}):")
    print(f"{'─' * 50}")

    X_pool = np.vstack(all_X_raw)

    for i in slow_idx:
        pname = proteins[i]
        reg_indices = [j for j in range(P) if j != i and abs(W[j, i]) > 1e-12]
        reg_signs = [1 if W[j, i] > 0 else -1 for j in reg_indices]
        M = len(reg_indices)
        sindy_w = [W[j, i] for j in reg_indices]

        if M == 0:
            results[pname] = dict(
                edge_params=[], reg_indices=[], reg_names=[],
                gamma=0.0, basal=np.nan, r2=np.nan, rmse=np.nan,
                dae_type="slow",
            )
            print(f"  [{i+1:2d}/{P}] {pname:30s}  — no regulators, skip")
            continue

        try:
            if dae_alg_mode == "ratio":
                fit = _fit_single_protein_ratio(i, reg_indices, reg_signs,
                                                sindy_w, X_pool)
            elif dae_alg_mode == "multiplicative":
                fit = _fit_single_protein_multiplicative(i, reg_indices,
                                                         reg_signs,
                                                         sindy_w, X_pool)
            else:
                fit = _fit_single_protein_algebraic(i, reg_indices,
                                                     reg_signs,
                                                     sindy_w, X_pool)
        except Exception as e:
            print(f"  [{i+1:2d}/{P}] {pname:30s}  — FAILED: {e}")
            results[pname] = dict(
                edge_params=[], reg_indices=reg_indices,
                reg_names=[proteins[j] for j in reg_indices],
                gamma=0.0, basal=np.nan, r2=np.nan, rmse=np.nan,
                dae_type="slow",
            )
            continue

        fit["reg_indices"] = reg_indices
        fit["reg_names"] = [proteins[j] for j in reg_indices]
        fit["dae_type"] = "slow"
        results[pname] = fit

        n_act = sum(1 for s in reg_signs if s > 0)
        print(f"  [{i+1:2d}/{P}] {pname:30s}  M={M:2d} (+{n_act}/-{M-n_act})  "
              f"R²={fit['r2']:.4f}  RMSE={fit['rmse']:.6f}")

    # ── fit FAST proteins (Hill ODE) ──
    print(f"\n{'─' * 50}")
    print(f"  Fitting FAST proteins (Hill ODE):")
    print(f"{'─' * 50}")

    for i in fast_idx:
        pname = proteins[i]
        reg_indices = [j for j in range(P) if j != i and abs(W[j, i]) > 1e-12]
        reg_signs = [1 if W[j, i] > 0 else -1 for j in reg_indices]
        M = len(reg_indices)

        if M == 0:
            results[pname] = dict(
                edge_params=[], reg_indices=[], reg_names=[],
                gamma=np.nan, basal=np.nan, vmax=0.0,
                r2=np.nan, rmse=np.nan, dae_type="fast",
            )
            print(f"  [{i+1:2d}/{P}] {pname:30s}  — no regulators, skip")
            continue

        common_kw = dict(
            target_idx=i, reg_indices=reg_indices, reg_signs=reg_signs,
            all_X=all_X_raw, all_times=all_times_raw,
            early_weight_tau=early_weight_tau,
            cond_W_list=cond_W_list, n_restarts=n_restarts)

        # ── fit all 15 combos, pick best RMSE ──
        fits: dict[str, dict | None] = {}
        rmse_map: dict[str, float] = {}
        for combo in ALL_COMBOS:
            lbl = _combo_label(combo)
            try:
                fit = _fit_single_protein_combined(combo=combo, **common_kw)
                fits[lbl] = fit
                rmse_map[lbl] = fit["rmse"] if np.isfinite(fit.get("rmse", np.nan)) else np.inf
            except Exception:
                fits[lbl] = None
                rmse_map[lbl] = np.inf

        combo_labels_local = [_combo_label(c) for c in ALL_COMBOS]
        best_label = min(rmse_map, key=rmse_map.get)
        best_fit = fits[best_label]

        if best_fit is None or not np.isfinite(rmse_map[best_label]):
            for lbl in combo_labels_local:
                if fits[lbl] is not None:
                    best_label = lbl
                    best_fit = fits[lbl]
                    break
            if best_fit is None:
                results[pname] = dict(
                    edge_params=[], reg_indices=reg_indices,
                    reg_names=[proteins[j] for j in reg_indices],
                    gamma=np.nan, basal=np.nan, vmax=0.0,
                    r2=np.nan, rmse=np.nan, dae_type="fast",
                    model="failed",
                )
                print(f"  [{i+1:2d}/{P}] {pname:30s}  — ALL COMBOS FAILED")
                continue

        best_fit["rmse_all"] = dict(rmse_map)
        best_fit["reg_names"] = [proteins[j] for j in
                                  best_fit.get("reg_indices", reg_indices)]
        best_fit["dae_type"] = "fast"
        results[pname] = best_fit

        r2_val = best_fit.get("r2", float("nan"))
        rmse_val = best_fit.get("rmse", float("nan"))
        n_edges = len(best_fit.get("edge_params", []))
        print(f"  [{i+1:2d}/{P}] {pname:30s}  "
              f"→ {best_label:20s}  RMSE={rmse_val:.4f}  R²={r2_val:.4f}  "
              f"edges={n_edges}  γ={best_fit.get('gamma', 0):.4f}")

    # ── store DAE metadata ──
    results["_dae_meta"] = {
        "fast_indices": fast_idx,
        "slow_indices": slow_idx,
        "fast_names": fast_names,
        "slow_names": slow_names,
        "scores": scores.tolist(),
        "fast_frac": fast_frac,
        "dae_alg_mode": dae_alg_mode,
    }

    return results


# ============================================================================
# ODE Data Preprocessing  (shared by all ODE fitting functions)
# ============================================================================
def _precompute_ode_data(target_idx, reg_indices, reg_signs, all_X, all_times,
                         early_weight_tau=0.0, cond_W_list=None,
                         neutral_fill=1e-12):
    """Build concatenated arrays for numba ODE kernels.

    Returns dict with keys:
      all_xp, xp_max, med_regs, K_ub,
      _all_reg, _all_obs, _all_w, _all_dt,
      _seg_lens, _seg_starts, _act_mask
    """
    M = len(reg_indices)
    C = len(all_X)

    cond_data = []
    for c in range(C):
        X_c = all_X[c]
        t_c = all_times[c]
        T_c = len(t_c)
        xp_obs = X_c[:, target_idx].copy()
        dt_arr = np.diff(t_c)

        if M > 0:
            reg_data = np.column_stack(
                [np.maximum(X_c[:, reg_indices[k]], 0.0) + 1e-12
                 for k in range(M)])
        else:
            reg_data = np.empty((T_c, 0))

        if cond_W_list is not None:
            Wc = cond_W_list[c]
            for k in range(M):
                if abs(Wc[reg_indices[k], target_idx]) < 1e-12:
                    reg_data[:, k] = neutral_fill

        if early_weight_tau > 0:
            w = np.exp(-t_c / early_weight_tau)
            w = np.clip(w, 0.1, 1.0)
        else:
            w = np.ones(T_c)
        w /= w.mean()
        cond_data.append((xp_obs, reg_data, dt_arr, w, T_c))

    all_xp = np.concatenate([cd[0] for cd in cond_data])
    xp_max = np.max(np.abs(all_xp)) + 1e-6

    if M > 0:
        all_regs_arr = np.vstack([cd[1] for cd in cond_data])
        med_regs = np.array([np.median(all_regs_arr[:, k]) + 1e-6
                             for k in range(M)])
        max_regs = np.array([np.max(all_regs_arr[:, k]) + 1e-6
                             for k in range(M)])
        K_ub = np.clip(max_regs * 3, 0.01, None)
    else:
        med_regs = np.empty(0)
        K_ub = np.empty(0)

    _all_reg = np.ascontiguousarray(np.vstack([cd[1] for cd in cond_data]))
    _all_obs = np.ascontiguousarray(np.concatenate([cd[0] for cd in cond_data]))
    _all_w = np.ascontiguousarray(np.concatenate([cd[3] for cd in cond_data]))
    _all_dt = np.ascontiguousarray(np.concatenate([cd[2] for cd in cond_data]))
    _seg_lens = np.array([cd[4] for cd in cond_data], dtype=np.int64)
    _seg_starts = np.zeros(len(cond_data), dtype=np.int64)
    _seg_starts[1:] = np.cumsum(_seg_lens[:-1])
    _act_mask = np.array([s > 0 for s in reg_signs], dtype=np.bool_)

    return {
        "all_xp": all_xp, "xp_max": xp_max,
        "med_regs": med_regs, "K_ub": K_ub,
        "_all_reg": _all_reg, "_all_obs": _all_obs,
        "_all_w": _all_w, "_all_dt": _all_dt,
        "_seg_lens": _seg_lens, "_seg_starts": _seg_starts,
        "_act_mask": _act_mask,
    }


# ============================================================================
# Combined ODE  —  Unified kernel for all 2^4-1 component combinations
# ============================================================================

_COMBO_COMPONENTS = ("add", "mult", "ratio", "ss")

# Pre-compute all 15 non-empty subsets
ALL_COMBOS: list[frozenset] = []
for _r in range(1, 5):
    for _c in _combinations(_COMBO_COMPONENTS, _r):
        ALL_COMBOS.append(frozenset(_c))


def _combo_label(combo) -> str:
    """Human-readable label for a component combination."""
    return "+".join(c for c in _COMBO_COMPONENTS if c in combo)


@nb.njit(cache=True)
def _combined_ode_residuals_nb(
        K_arr, n_arr,
        beta_arr, basal,
        Vmax_mult,
        w_ratio, Vmax_ratio, w0_ratio, g0_ratio,
        g_arr, h_arr, alpha_ss, beta_ss,
        gamma,
        use_add, use_mult, use_ratio, use_ss,
        all_reg, all_obs, all_w, all_dt,
        seg_starts, seg_lens, act_mask):
    """Unified ODE residual for any combination of add/mult/ratio/ss.

    dX/dt = -gamma*X + F_add + F_mult + F_ratio + F_ss  (whichever active)

    Hill-based components (add, mult, ratio) share K, n parameters.
    S-system has its own g, h parameters.
    All arrays must have length M = all_reg.shape[1].
    """
    N_total = all_obs.shape[0]
    M = all_reg.shape[1]
    use_hill = use_add or use_mult or use_ratio

    # Pre-compute activator-style Hill values  h_act = x^n / (K^n + x^n)
    if use_hill and M > 0:
        hill_act_vals = np.empty((N_total, M))
        for k in range(M):
            Kn = K_arr[k] ** n_arr[k]
            for t in range(N_total):
                xrn = all_reg[t, k] ** n_arr[k]
                hill_act_vals[t, k] = xrn / (Kn + xrn)
    else:
        hill_act_vals = np.empty((N_total, 0))

    # Compute total forcing at each time point
    F = np.empty(N_total)
    for t in range(N_total):
        f_val = 0.0

        # Basal offset (when add or mult active)
        if use_add or use_mult:
            f_val += basal

        # Additive:  sum_j beta_j * h_j  (sign-dependent Hill)
        if use_add:
            for k in range(M):
                if act_mask[k]:
                    f_val += beta_arr[k] * hill_act_vals[t, k]
                else:
                    f_val += beta_arr[k] * (1.0 - hill_act_vals[t, k])

        # Multiplicative:  Vmax * prod_j h_j  (sign-dependent Hill)
        if use_mult:
            prod = 1.0
            for k in range(M):
                if act_mask[k]:
                    prod *= hill_act_vals[t, k]
                else:
                    prod *= (1.0 - hill_act_vals[t, k])
            f_val += Vmax_mult * prod

        # Ratio:  Vmax_r * V_kin / (V_kin + V_phos)
        # All edges use activator-style Hill; sign picks kin vs phos arm
        if use_ratio:
            v_kin = w0_ratio
            v_phos = g0_ratio
            for k in range(M):
                if act_mask[k]:
                    v_kin += w_ratio[k] * hill_act_vals[t, k]
                else:
                    v_phos += w_ratio[k] * hill_act_vals[t, k]
            f_val += Vmax_ratio * v_kin / (v_kin + v_phos + 1e-15)

        # S-system:  alpha * prod X_j^g_j  -  beta_s * prod X_j^h_j
        if use_ss:
            prod_gen = 1.0
            prod_deg = 1.0
            for k in range(M):
                x_val = all_reg[t, k]
                prod_gen *= x_val ** g_arr[k]
                prod_deg *= x_val ** h_arr[k]
            f_val += alpha_ss * prod_gen - beta_ss * prod_deg

        F[t] = f_val

    # Analytical integration per condition segment
    xi = np.empty(N_total)
    dt_offset = 0
    C = seg_starts.shape[0]

    for seg_i in range(C):
        start = seg_starts[seg_i]
        T_c = seg_lens[seg_i]
        xi[start] = all_obs[start]

        if gamma > 1e-12:
            for t_local in range(T_c - 1):
                dt_val = all_dt[dt_offset + t_local]
                d = np.exp(-gamma * dt_val)
                g = (1.0 - d) / gamma
                xi[start + t_local + 1] = (d * xi[start + t_local]
                                            + F[start + t_local] * g)
        else:
            for t_local in range(T_c - 1):
                dt_val = all_dt[dt_offset + t_local]
                xi[start + t_local + 1] = (xi[start + t_local]
                                            + F[start + t_local] * dt_val)
        dt_offset += T_c - 1

    # Weighted residuals
    residuals = np.empty(N_total)
    for t in range(N_total):
        residuals[t] = np.sqrt(all_w[t]) * (xi[t] - all_obs[t])

    return residuals


def _fit_single_protein_combined(target_idx: int,
                                 reg_indices: list[int],
                                 reg_signs: list[int],
                                 all_X: list[np.ndarray],
                                 all_times: list[np.ndarray],
                                 combo: frozenset,
                                 early_weight_tau: float = 0.0,
                                 cond_W_list: list[np.ndarray] | None = None,
                                 n_restarts: int = 3) -> dict:
    """Fit a combined ODE with any subset of {add, mult, ratio, ss}.

    Two-stage fitting:
      Stage 1: fix Hill n=2, ratio w=1
      Stage 2: free all parameters
    """
    use_add = "add" in combo
    use_mult = "mult" in combo
    use_ratio = "ratio" in combo
    use_ss = "ss" in combo
    use_hill = use_add or use_mult or use_ratio
    use_basal = use_add or use_mult

    M = len(reg_indices)
    C = len(all_X)

    # ── Adaptive complexity budget ──
    n_components = sum([use_add, use_mult, use_ratio, use_ss])
    if n_components >= 3:
        eff_restarts = max(n_restarts - 2, 1)
        nfev_s1 = 2000
        nfev_s2 = 3000
    elif n_components == 2:
        eff_restarts = max(n_restarts - 1, 1)
        nfev_s1 = 2500
        nfev_s2 = 4000
    else:
        eff_restarts = n_restarts
        nfev_s1 = 3000
        nfev_s2 = 5000

    # ── Precompute per-condition data ──
    neutral = 1.0 if (use_ss and not use_hill) else 1e-12
    _od = _precompute_ode_data(
        target_idx, reg_indices, reg_signs, all_X, all_times,
        early_weight_tau=early_weight_tau, cond_W_list=cond_W_list,
        neutral_fill=neutral)
    all_xp = _od['all_xp']
    xp_max = _od['xp_max']
    med_regs = _od['med_regs']
    K_ub = _od['K_ub']
    _all_reg = _od['_all_reg']
    _all_obs = _od['_all_obs']
    _all_w = _od['_all_w']
    _all_dt = _od['_all_dt']
    _seg_lens = _od['_seg_lens']
    _seg_starts = _od['_seg_starts']
    _act_mask = _od['_act_mask']
    Vmax_ub = xp_max * 10
    basal_ub = xp_max * 5
    beta_ub = xp_max * 5
    alpha_ub = xp_max * 20

    # ── Parameter layout builders ──
    def _layout(stage1: bool):
        off = {}; idx = 0
        if use_hill:
            off["K"] = idx; idx += M
            if not stage1:
                off["n"] = idx; idx += M
        if use_add:
            off["beta"] = idx; idx += M
        if use_mult:
            off["Vmax"] = idx; idx += 1
        if use_basal:
            off["basal"] = idx; idx += 1
        if use_ratio:
            if not stage1:
                off["w_r"] = idx; idx += M
            off["Vmax_r"] = idx; idx += 1
            off["w0"] = idx; idx += 1
            off["g0"] = idx; idx += 1
        if use_ss:
            off["g"] = idx; idx += M
            off["h"] = idx; idx += M
            off["alpha"] = idx; idx += 1
            off["beta_s"] = idx; idx += 1
        off["gamma"] = idx; idx += 1
        return idx, off

    np_s1, off_s1 = _layout(stage1=True)
    np_full, off_full = _layout(stage1=False)

    # ── Unpack params → numba kernel args ──
    def _unpack(params, offsets, stage1=False):
        K = params[offsets["K"]:offsets["K"]+M] if use_hill else np.zeros(M)
        n = np.full(M, 2.0) if (stage1 or not use_hill) else params[offsets["n"]:offsets["n"]+M]
        beta = params[offsets["beta"]:offsets["beta"]+M] if use_add else np.zeros(M)
        basal_v = float(params[offsets["basal"]]) if use_basal else 0.0
        Vmax_v = float(params[offsets["Vmax"]]) if use_mult else 0.0
        if use_ratio:
            w_r = np.ones(M) if stage1 else params[offsets["w_r"]:offsets["w_r"]+M]
            Vmax_r = float(params[offsets["Vmax_r"]])
            w0_v = float(params[offsets["w0"]])
            g0_v = float(params[offsets["g0"]])
        else:
            w_r = np.zeros(M); Vmax_r = 0.0; w0_v = 0.0; g0_v = 0.0
        if use_ss:
            g = params[offsets["g"]:offsets["g"]+M]
            h = params[offsets["h"]:offsets["h"]+M]
            alpha_v = float(params[offsets["alpha"]])
            beta_s = float(params[offsets["beta_s"]])
        else:
            g = np.zeros(M); h = np.zeros(M); alpha_v = 0.0; beta_s = 0.0
        gamma_v = float(params[offsets["gamma"]])
        return (K, n, beta, basal_v, Vmax_v,
                w_r, Vmax_r, w0_v, g0_v,
                g, h, alpha_v, beta_s, gamma_v)

    # ── Residual closures ──
    def _res_s1(params):
        args = _unpack(params, off_s1, stage1=True)
        return _combined_ode_residuals_nb(
            *args, use_add, use_mult, use_ratio, use_ss,
            _all_reg, _all_obs, _all_w, _all_dt,
            _seg_starts, _seg_lens, _act_mask)

    def _res_full(params):
        args = _unpack(params, off_full, stage1=False)
        return _combined_ode_residuals_nb(
            *args, use_add, use_mult, use_ratio, use_ss,
            _all_reg, _all_obs, _all_w, _all_dt,
            _seg_starts, _seg_lens, _act_mask)

    # ── Bounds ──
    def _make_bounds(n_params, offsets, stage1=False):
        lb = np.empty(n_params); ub = np.empty(n_params)
        if use_hill:
            o = offsets["K"]
            for k in range(M):
                lb[o+k] = 1e-4; ub[o+k] = K_ub[k]
            if not stage1:
                o = offsets["n"]
                for k in range(M):
                    lb[o+k] = 0.5; ub[o+k] = 5.0
        if use_add:
            o = offsets["beta"]
            for k in range(M):
                lb[o+k] = 0.0; ub[o+k] = beta_ub
        if use_mult:
            o = offsets["Vmax"]
            lb[o] = 0.0; ub[o] = Vmax_ub
        if use_basal:
            o = offsets["basal"]
            lb[o] = -basal_ub; ub[o] = basal_ub
        if use_ratio:
            if not stage1:
                o = offsets["w_r"]
                for k in range(M):
                    lb[o+k] = 1e-4; ub[o+k] = 20.0
            o = offsets["Vmax_r"]
            lb[o] = 1e-4; ub[o] = Vmax_ub
            lb[o+1] = 1e-6; ub[o+1] = 10.0   # w0
            lb[o+2] = 1e-6; ub[o+2] = 10.0   # g0
        if use_ss:
            o = offsets["g"]
            for k in range(M):
                lb[o+k] = -3.0; ub[o+k] = 3.0
            o = offsets["h"]
            for k in range(M):
                lb[o+k] = -3.0; ub[o+k] = 3.0
            o = offsets["alpha"]
            lb[o] = 1e-6; ub[o] = alpha_ub
            lb[o+1] = 1e-6; ub[o+1] = alpha_ub   # beta_s
        o = offsets["gamma"]
        lb[o] = 1e-4; ub[o] = 2.0
        return lb, ub

    lb_s1, ub_s1 = _make_bounds(np_s1, off_s1, stage1=True)
    lb_full, ub_full = _make_bounds(np_full, off_full, stage1=False)

    rng = np.random.RandomState(42 + target_idx)
    best_cost = np.inf
    best_popt = None

    for _restart in range(eff_restarts):
        # ── Stage 1 initial guess ──
        p0 = np.empty(np_s1)
        if use_hill and M > 0:
            o = off_s1["K"]
            jitter = np.clip(1.0 + 0.3 * rng.randn(M), 0.5, 2.0)
            p0[o:o+M] = np.clip(med_regs * jitter,
                                lb_s1[o:o+M] + 1e-6, ub_s1[o:o+M] - 1e-6)
        if use_add:
            o = off_s1["beta"]
            p0[o:o+M] = 0.1 * xp_max
        if use_mult:
            p0[off_s1["Vmax"]] = np.clip(np.ptp(all_xp) + 1e-3,
                                          1e-3, Vmax_ub - 1e-3)
        if use_basal:
            p0[off_s1["basal"]] = np.clip(np.mean(all_xp) * 0.1,
                                           -basal_ub + 1e-3, basal_ub - 1e-3)
        if use_ratio:
            o = off_s1["Vmax_r"]
            p0[o] = np.clip(np.max(all_xp), 1e-3, Vmax_ub - 1e-3)
            p0[o+1] = 0.1   # w0
            p0[o+2] = 0.1   # g0
        if use_ss:
            og = off_s1["g"]; oh = off_s1["h"]
            for k in range(M):
                if reg_signs[k] > 0:
                    p0[og+k] = np.clip(0.5 + 0.3 * rng.randn(), -2.9, 2.9)
                    p0[oh+k] = np.clip(0.1 * rng.randn(), -2.9, 2.9)
                else:
                    p0[og+k] = np.clip(0.1 * rng.randn(), -2.9, 2.9)
                    p0[oh+k] = np.clip(0.5 + 0.3 * rng.randn(), -2.9, 2.9)
            oa = off_s1["alpha"]
            p0[oa] = np.clip(xp_max * 0.5 * (1 + 0.3 * rng.randn()),
                             1e-5, alpha_ub - 1e-3)
            p0[oa+1] = np.clip(xp_max * 0.5 * (1 + 0.3 * rng.randn()),
                               1e-5, alpha_ub - 1e-3)
        p0[off_s1["gamma"]] = np.clip(0.01 + 0.02 * rng.rand(), 1e-3, 1.9)
        p0 = np.clip(p0, lb_s1 + 1e-8, ub_s1 - 1e-8)

        try:
            res1 = least_squares(_res_s1, p0, bounds=(lb_s1, ub_s1),
                                 method="trf", max_nfev=nfev_s1,
                                 ftol=1e-8, xtol=1e-8)
        except Exception:
            continue

        # ── Stage 2 initial from Stage 1 ──
        p0f = np.empty(np_full)
        if use_hill:
            o1 = off_s1["K"]; o2 = off_full["K"]
            p0f[o2:o2+M] = res1.x[o1:o1+M]
            p0f[off_full["n"]:off_full["n"]+M] = 2.0
        if use_add:
            o1 = off_s1["beta"]; o2 = off_full["beta"]
            p0f[o2:o2+M] = res1.x[o1:o1+M]
        if use_mult:
            p0f[off_full["Vmax"]] = res1.x[off_s1["Vmax"]]
        if use_basal:
            p0f[off_full["basal"]] = res1.x[off_s1["basal"]]
        if use_ratio:
            p0f[off_full["w_r"]:off_full["w_r"]+M] = 1.0
            p0f[off_full["Vmax_r"]] = res1.x[off_s1["Vmax_r"]]
            p0f[off_full["w0"]] = res1.x[off_s1["w0"]]
            p0f[off_full["g0"]] = res1.x[off_s1["g0"]]
        if use_ss:
            for key in ("g", "h", "alpha", "beta_s"):
                sz = M if key in ("g", "h") else 1
                p0f[off_full[key]:off_full[key]+sz] = \
                    res1.x[off_s1[key]:off_s1[key]+sz]
        p0f[off_full["gamma"]] = res1.x[off_s1["gamma"]]
        p0f = np.clip(p0f, lb_full + 1e-8, ub_full - 1e-8)

        try:
            res2 = least_squares(_res_full, p0f, bounds=(lb_full, ub_full),
                                 method="trf", max_nfev=nfev_s2,
                                 ftol=1e-8, xtol=1e-8)
        except Exception:
            continue

        if res2.cost < best_cost:
            best_cost = res2.cost
            best_popt = res2.x.copy()

    if best_popt is None:
        raise RuntimeError(f"Combined ODE ({_combo_label(combo)}): all restarts failed")

    # ── Unpack best parameters ──
    p = best_popt
    K_opt = p[off_full["K"]:off_full["K"]+M] if use_hill else np.zeros(M)
    n_opt = p[off_full["n"]:off_full["n"]+M] if use_hill else np.full(M, 2.0)
    beta_opt = p[off_full["beta"]:off_full["beta"]+M] if use_add else np.zeros(M)
    Vmax_opt = float(p[off_full["Vmax"]]) if use_mult else 0.0
    basal_opt = float(p[off_full["basal"]]) if use_basal else 0.0
    if use_ratio:
        wr_opt = p[off_full["w_r"]:off_full["w_r"]+M]
        Vmax_r_opt = float(p[off_full["Vmax_r"]])
        w0_opt = float(p[off_full["w0"]])
        g0_opt = float(p[off_full["g0"]])
    else:
        wr_opt = np.zeros(M); Vmax_r_opt = 0.0; w0_opt = 0.0; g0_opt = 0.0
    if use_ss:
        g_opt = p[off_full["g"]:off_full["g"]+M]
        h_opt = p[off_full["h"]:off_full["h"]+M]
        alpha_opt = float(p[off_full["alpha"]])
        beta_s_opt = float(p[off_full["beta_s"]])
    else:
        g_opt = np.zeros(M); h_opt = np.zeros(M)
        alpha_opt = 0.0; beta_s_opt = 0.0
    gamma_opt = float(p[off_full["gamma"]])

    # ── Final R² / RMSE ──
    residuals = _combined_ode_residuals_nb(
        K_opt, n_opt, beta_opt, basal_opt, Vmax_opt,
        wr_opt, Vmax_r_opt, w0_opt, g0_opt,
        g_opt, h_opt, alpha_opt, beta_s_opt, gamma_opt,
        use_add, use_mult, use_ratio, use_ss,
        _all_reg, _all_obs, _all_w, _all_dt,
        _seg_starts, _seg_lens, _act_mask)
    pred_err = residuals / np.sqrt(_all_w)
    r2, rmse = _compute_r2_rmse(pred_err, _all_obs)

    # ── Build edge_params ──
    combo_label = _combo_label(combo)
    edge_params: list[dict] = []
    for k in range(M):
        ep: dict = {
            "K": float(K_opt[k]),
            "n": float(n_opt[k]),
            "sign": "activation" if reg_signs[k] > 0 else "inhibition",
            "w": Vmax_opt if use_mult else (
                 float(beta_opt[k]) if use_add else (
                 float(wr_opt[k]) if use_ratio else alpha_opt)),
        }
        if use_add:
            ep["beta"] = float(beta_opt[k])
        if use_ratio:
            ep["w_ratio"] = float(wr_opt[k])
        if use_ss:
            ep["g"] = float(g_opt[k])
            ep["h"] = float(h_opt[k])
        edge_params.append(ep)

    result: dict = {
        "edge_params": edge_params,
        "reg_indices": list(reg_indices),
        "model": combo_label,
        "gamma": gamma_opt,
        "basal": basal_opt,
        "vmax": Vmax_opt,
        "r2": r2,
        "rmse": rmse,
        "de_cost": float(best_cost),
        "combo_flags": {"add": use_add, "mult": use_mult,
                        "ratio": use_ratio, "ss": use_ss},
    }
    if use_add:
        result["coex"] = True
    if use_ratio:
        result["vmax_ratio"] = Vmax_r_opt
        result["w0"] = w0_opt
        result["gamma0"] = g0_opt
    if use_ss:
        result["alpha"] = alpha_opt
        result["beta_s"] = beta_s_opt

    return result


# ============================================================================
# BIC Model Selection  (de_auto)
# ============================================================================

def _compute_bic(rss: float, n_obs: int, n_params: int) -> float:
    """Bayesian Information Criterion: BIC = n*ln(RSS/n) + k*ln(n)."""
    if rss <= 0 or n_obs <= 0:
        return np.inf
    return n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)


def _count_n_obs(all_X: list[np.ndarray]) -> int:
    """Total number of time-series observations across conditions."""
    return sum(x.shape[0] for x in all_X)


def fit_all_proteins_auto(W: np.ndarray,
                          proteins: list[str],
                          all_X: list[np.ndarray],
                          all_times: list[np.ndarray],
                          early_weight_tau: float = 0.0,
                          cond_W_list: list[np.ndarray] | None = None,
                          n_restarts: int = 3) -> dict[str, dict]:
    """Best-RMSE model selection across all 15 ODE component combinations.

    Components:
      add   — additive Hill:      sum_j beta_j * h_j
      mult  — multiplicative:     Vmax * prod_j h_j
      ratio — push-pull ratio:    Vmax_r * V_kin / (V_kin + V_phos)
      ss    — S-system power-law: alpha * prod X^g - beta * prod X^h

    Tries all 2^4 - 1 = 15 non-empty subsets per protein, picks best RMSE.
    """
    P = len(proteins)
    results: dict[str, dict] = {}

    combo_labels = [_combo_label(c) for c in ALL_COMBOS]
    model_wins: dict[str, int] = {lbl: 0 for lbl in combo_labels}

    print(f"\n  Auto-selection (best RMSE): fitting {len(ALL_COMBOS)} "
          f"combo models per protein ...")
    print(f"  Combos: {', '.join(combo_labels)}")

    for i in range(P):
        pname = proteins[i]
        reg_indices = [j for j in range(P) if j != i and abs(W[j, i]) > 1e-12]
        reg_signs = [1 if W[j, i] > 0 else -1 for j in reg_indices]
        M = len(reg_indices)

        if M == 0:
            results[pname] = dict(
                edge_params=[], reg_indices=[], reg_names=[],
                gamma=0.0, basal=np.nan, vmax=0.0,
                r2=np.nan, rmse=np.nan, model="none",
            )
            print(f"  [{i+1:2d}/{P}] {pname:30s}  — no regulators, skip")
            continue

        common_kw = dict(
            target_idx=i, reg_indices=reg_indices, reg_signs=reg_signs,
            all_X=all_X, all_times=all_times,
            early_weight_tau=early_weight_tau,
            cond_W_list=cond_W_list, n_restarts=n_restarts)

        # ── Fit all 15 combos ──
        fits: dict[str, dict | None] = {}
        rmse_map: dict[str, float] = {}

        for combo in ALL_COMBOS:
            lbl = _combo_label(combo)
            try:
                fit = _fit_single_protein_combined(combo=combo, **common_kw)
                fits[lbl] = fit
                rmse_map[lbl] = fit["rmse"] if np.isfinite(fit.get("rmse", np.nan)) else np.inf
            except Exception:
                fits[lbl] = None
                rmse_map[lbl] = np.inf

        # ── Select best by RMSE ──
        best_label = min(rmse_map, key=rmse_map.get)
        best_fit = fits[best_label]

        if best_fit is None or not np.isfinite(rmse_map[best_label]):
            for lbl in combo_labels:
                if fits[lbl] is not None:
                    best_label = lbl
                    best_fit = fits[lbl]
                    break
            if best_fit is None:
                results[pname] = dict(
                    edge_params=[], reg_indices=reg_indices,
                    reg_names=[proteins[j] for j in reg_indices],
                    gamma=np.nan, basal=np.nan, vmax=0.0,
                    r2=np.nan, rmse=np.nan, model="failed",
                )
                print(f"  [{i+1:2d}/{P}] {pname:30s}  — ALL COMBOS FAILED")
                continue

        best_fit["rmse_all"] = dict(rmse_map)
        best_fit["reg_names"] = [proteins[j] for j in
                                  best_fit.get("reg_indices", reg_indices)]
        results[pname] = best_fit
        model_wins[best_label] = model_wins.get(best_label, 0) + 1

        r2_val = best_fit.get("r2", float("nan"))
        rmse_val = best_fit.get("rmse", float("nan"))
        n_edges = len(best_fit.get("edge_params", []))

        # Top-5 combos by RMSE for compact display
        sorted_combos = sorted(rmse_map.items(), key=lambda x: x[1])[:5]
        top_str = " | ".join(
            f"{m}={v:.4f}" if np.isfinite(v) else f"{m}=—"
            for m, v in sorted_combos)

        print(f"  [{i+1:2d}/{P}] {pname:30s}  "
              f"→ {best_label:20s}  RMSE={rmse_val:.4f}  R²={r2_val:.4f}  "
              f"edges={n_edges}  γ={best_fit.get('gamma', 0):.4f}")
        print(f"          top5: {top_str}")

    # ── Summary ──
    active_models = {m: c for m, c in model_wins.items() if c > 0}
    print(f"\n  Model selection summary ({len(active_models)} combos used):")
    for lbl in combo_labels:
        if model_wins[lbl] > 0:
            print(f"    {lbl:20s} : {model_wins[lbl]:3d} proteins")

    return results


# ============================================================================
# Algebraic Hill — Predicted Levels (for visualization)
# ============================================================================
def predict_algebraic(fit_results: dict, proteins: list[str],
                      X_obs: np.ndarray, times: np.ndarray = None
                      ) -> np.ndarray:
    """Predict protein levels using fitted algebraic/multiplicative/ratio Hill model.

    Automatically detects model type from fit results.
    When *times* is provided, DE-fitted (ODE) proteins are integrated via
    forward Euler with observed regulators.  Otherwise they fall back to the
    multiplicative algebraic approximation.

    Returns (T, P) prediction array.
    """
    T, P = X_obs.shape
    X_pred = np.full((T, P), np.nan)
    dt = np.diff(times) if times is not None else None

    for i, pname in enumerate(proteins):
        fit = fit_results.get(pname)
        if fit is None:
            X_pred[:, i] = X_obs[:, i]
            continue

        if not fit.get("edge_params"):
            X_pred[:, i] = X_obs[:, i]
            continue

        # ── DE / ODE-fitted protein: analytical integration ──
        if "de_cost" in fit and dt is not None:
            model_type = fit.get("model", "")
            xi = np.empty(T)
            xi[0] = X_obs[0, i]
            gamma_p = fit["gamma"]

            cflags = fit.get("combo_flags")
            if cflags:
                # ── Combined model: compute F from active components ──
                c_add = cflags.get("add", False)
                c_mult = cflags.get("mult", False)
                c_ratio = cflags.get("ratio", False)
                c_ss = cflags.get("ss", False)
                c_hill = c_add or c_mult or c_ratio

                # Pre-compute activator-style Hill values
                if c_hill:
                    h_act_arr = np.empty((T, len(fit["edge_params"])))
                    for k, ep in enumerate(fit["edge_params"]):
                        j = fit["reg_indices"][k]
                        h_act_arr[:, k] = hill_act(
                            np.maximum(X_obs[:, j], 0.0) + 1e-12,
                            ep["K"], ep["n"])

                F = np.zeros(T)
                # Basal
                if c_add or c_mult:
                    F += fit["basal"]
                # Additive
                if c_add:
                    for k, ep in enumerate(fit["edge_params"]):
                        hv = h_act_arr[:, k] if ep["sign"] == "activation" \
                             else (1.0 - h_act_arr[:, k])
                        F += ep.get("beta", 0.0) * hv
                # Multiplicative
                if c_mult:
                    product = np.ones(T)
                    for k, ep in enumerate(fit["edge_params"]):
                        hv = h_act_arr[:, k] if ep["sign"] == "activation" \
                             else (1.0 - h_act_arr[:, k])
                        product *= hv
                    F += fit["vmax"] * product
                # Ratio
                if c_ratio:
                    V_kin = np.full(T, fit.get("w0", 0.1))
                    V_phos = np.full(T, fit.get("gamma0", 0.1))
                    for k, ep in enumerate(fit["edge_params"]):
                        wr = ep.get("w_ratio", 1.0)
                        if ep["sign"] == "activation":
                            V_kin += wr * h_act_arr[:, k]
                        else:
                            V_phos += wr * h_act_arr[:, k]
                    F += fit.get("vmax_ratio", 1.0) * V_kin / (V_kin + V_phos + 1e-15)
                # S-system
                if c_ss:
                    prod_gen = np.ones(T)
                    prod_deg = np.ones(T)
                    for k, ep in enumerate(fit["edge_params"]):
                        j = fit["reg_indices"][k]
                        x_reg = np.maximum(X_obs[:, j], 0.0) + 1e-12
                        prod_gen *= np.power(x_reg, ep.get("g", 0.0))
                        prod_deg *= np.power(x_reg, ep.get("h", 0.0))
                    F += fit.get("alpha", 0.0) * prod_gen \
                         - fit.get("beta_s", 0.0) * prod_deg

            elif model_type == "ratio_ode":
                # Push-pull ratio ODE: F = Vmax · V_kin/(V_kin+V_phos)
                Vmax_p = fit["vmax"]
                w0_p = fit.get("w0", 0.1)
                g0_p = fit.get("gamma0", 0.1)
                V_kin = np.full(T, w0_p)
                V_phos = np.full(T, g0_p)
                for k, ep in enumerate(fit["edge_params"]):
                    j = fit["reg_indices"][k]
                    h = hill_act(np.maximum(X_obs[:, j], 0.0) + 1e-12,
                                 ep["K"], ep["n"])
                    if ep["sign"] == "activation":
                        V_kin += ep["w"] * h
                    else:
                        V_phos += ep["w"] * h
                F = Vmax_p * V_kin / (V_kin + V_phos + 1e-15)

            elif model_type == "ssystem":
                # S-system ODE: F = α·Π X_j^g_j − β·Π X_j^h_j
                alpha_val = fit["alpha"]
                beta_val = fit["beta_s"]
                prod_gen = np.ones(T)
                prod_deg = np.ones(T)
                for k, ep in enumerate(fit["edge_params"]):
                    j = fit["reg_indices"][k]
                    x_reg = np.maximum(X_obs[:, j], 0.0) + 1e-12
                    prod_gen *= np.power(x_reg, ep["g"])
                    prod_deg *= np.power(x_reg, ep["h"])
                F = alpha_val * prod_gen - beta_val * prod_deg

            else:
                # Multiplicative / coexistence Hill ODE
                hill_vals = np.empty((T, len(fit["edge_params"])))
                product = np.ones(T)
                for k, ep in enumerate(fit["edge_params"]):
                    j = fit["reg_indices"][k]
                    if ep["sign"] == "activation":
                        h = hill_act(X_obs[:, j], ep["K"], ep["n"])
                    else:
                        h = hill_inh(X_obs[:, j], ep["K"], ep["n"])
                    hill_vals[:, k] = h
                    product *= h

                Vmax_p = fit["vmax"]
                basal_p = fit["basal"]
                F = basal_p + Vmax_p * product
                if fit.get("coex"):
                    for k, ep in enumerate(fit["edge_params"]):
                        F += ep.get("beta", 0.0) * hill_vals[:, k]

            if gamma_p > 1e-12:
                decay = np.exp(-gamma_p * dt)
                gain = (1.0 - decay) / gamma_p
            else:
                decay = np.ones_like(dt)
                gain = dt.copy()
            for t in range(T - 1):
                xi[t + 1] = decay[t] * xi[t] + F[t] * gain[t]
            X_pred[:, i] = xi

        # ── Kinase-Phosphatase Ratio model ──
        elif fit.get("model") == "ratio":
            # Kinase-phosphatase ratio model
            V_kin = np.full(T, fit["w0"])
            V_phos = np.full(T, fit["gamma0"])
            for k, ep in enumerate(fit["edge_params"]):
                j = fit["reg_indices"][k]
                h = hill_act(np.maximum(X_obs[:, j], 0.0) + 1e-12,
                             ep["K"], ep["n"])
                if ep["sign"] == "activation":
                    V_kin += ep["w"] * h
                else:
                    V_phos += ep["w"] * h
            X_pred[:, i] = fit["vmax"] * V_kin / (V_kin + V_phos + 1e-15)
        elif "vmax" in fit:
            # Multiplicative fractional occupancy model
            product = np.ones(T)
            for k, ep in enumerate(fit["edge_params"]):
                j = fit["reg_indices"][k]
                if ep["sign"] == "activation":
                    product *= hill_act(X_obs[:, j], ep["K"], ep["n"])
                else:
                    product *= hill_inh(X_obs[:, j], ep["K"], ep["n"])
            X_pred[:, i] = fit["basal"] + fit["vmax"] * product
        else:
            # Additive model (original)
            pred = np.full(T, fit["basal"])
            for k, ep in enumerate(fit["edge_params"]):
                j = fit["reg_indices"][k]
                if ep["sign"] == "activation":
                    pred += ep["w"] * hill_act(X_obs[:, j], ep["K"], ep["n"])
                else:
                    pred += ep["w"] * hill_inh(X_obs[:, j], ep["K"], ep["n"])
            X_pred[:, i] = pred

    return X_pred


# ============================================================================
# Visualization
# ============================================================================
def plot_fit_quality(fit_results: dict, proteins: list[str], output_path: str):
    """Histogram + bar chart of per-protein R²."""
    r2_vals, names = [], []
    for p in proteins:
        fit = fit_results.get(p)
        if fit and np.isfinite(fit.get("r2", np.nan)):
            r2_vals.append(fit["r2"])
            names.append(DISPLAY_NAMES_MULTILINE.get(p, p).replace("\n", " "))
    if not r2_vals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── histogram ──
    ax = axes[0]
    ax.hist(r2_vals, bins=20, color="#5C6BC0", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(r2_vals), color="#C62828", ls="--", lw=2,
               label=f"Median = {np.median(r2_vals):.3f}")
    ax.set_xlabel("R²", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Hill Function Fit Quality", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    # ── sorted bar chart ──
    ax = axes[1]
    order = np.argsort(r2_vals)[::-1]
    colors = ["#4CAF50" if r2_vals[k] >= 0.5
              else "#FF9800" if r2_vals[k] >= 0
              else "#F44336" for k in order]
    ax.barh(range(len(order)), [r2_vals[k] for k in order],
            color=colors, edgecolor="white")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in order], fontsize=7)
    ax.set_xlabel("R²", fontsize=12)
    ax.set_title("Per-Protein Fit Quality (sorted)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    _save_fig(fig, output_path)


# ============================================================================
# ★ NEW: Time-Series Data vs Equation Dynamics Comparison
# ============================================================================
def _compute_equation_dynamics(fit: dict, X_obs: np.ndarray,
                                proteins: list[str], times: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (dX_obs/dt_numeric, dX_eq/dt_from_ODE) for one protein's fit.

    Uses centred finite differences on the observed data (dX_obs) and the
    analytical RHS of the fitted ODE equation (dX_eq).
    Both arrays have length T-2 (interior points, avoiding edge effects).
    """
    T = X_obs.shape[0]
    i = proteins.index(fit["_pname"])

    # Centred finite difference of observations
    dt_fwd = times[2:] - times[1:-1]
    dt_bwd = times[1:-1] - times[:-2]
    dX_obs = (X_obs[2:, i] - X_obs[:-2, i]) / (dt_fwd + dt_bwd)

    # Analytical RHS of the fitted ODE:  dX/dt = -γX + F(t)
    gamma_p = fit.get("gamma", 0.0)
    cflags = fit.get("combo_flags", {})
    use_add   = cflags.get("add",   False) or (fit.get("model", "") == "add")
    use_mult  = cflags.get("mult",  False) or ("vmax" in fit and not cflags)
    use_ratio = cflags.get("ratio", False) or fit.get("model") == "ratio"
    use_ss    = cflags.get("ss",    False)
    use_hill  = use_add or use_mult or use_ratio

    F = np.zeros(T)

    if use_hill and fit.get("edge_params"):
        h_act_arr = np.empty((T, len(fit["edge_params"])))
        for k, ep in enumerate(fit["edge_params"]):
            j = fit["reg_indices"][k]
            h_act_arr[:, k] = hill_act(
                np.maximum(X_obs[:, j], 0.0) + 1e-12, ep["K"], ep["n"])

        if use_add or use_mult:
            F += fit.get("basal", 0.0)
        if use_add:
            for k, ep in enumerate(fit["edge_params"]):
                hv = h_act_arr[:, k] if ep["sign"] == "activation" \
                     else (1.0 - h_act_arr[:, k])
                F += ep.get("beta", 0.0) * hv
        if use_mult:
            product = np.ones(T)
            for k, ep in enumerate(fit["edge_params"]):
                hv = h_act_arr[:, k] if ep["sign"] == "activation" \
                     else (1.0 - h_act_arr[:, k])
                product *= hv
            F += fit.get("vmax", 0.0) * product
        if use_ratio:
            V_kin  = np.full(T, fit.get("w0", 0.1))
            V_phos = np.full(T, fit.get("gamma0", 0.1))
            for k, ep in enumerate(fit["edge_params"]):
                wr = ep.get("w_ratio", 1.0)
                if ep["sign"] == "activation":
                    V_kin  += wr * h_act_arr[:, k]
                else:
                    V_phos += wr * h_act_arr[:, k]
            F += fit.get("vmax_ratio", fit.get("vmax", 1.0)) * \
                 V_kin / (V_kin + V_phos + 1e-15)

    if use_ss and fit.get("edge_params"):
        prod_gen = np.ones(T)
        prod_deg = np.ones(T)
        for k, ep in enumerate(fit["edge_params"]):
            j = fit["reg_indices"][k]
            x_reg = np.maximum(X_obs[:, j], 0.0) + 1e-12
            prod_gen *= np.power(x_reg, ep.get("g", 0.0))
            prod_deg *= np.power(x_reg, ep.get("h", 0.0))
        F += fit.get("alpha", 0.0) * prod_gen - fit.get("beta_s", 0.0) * prod_deg

    dX_eq = -gamma_p * X_obs[1:-1, i] + F[1:-1]
    return dX_obs, dX_eq


def plot_timeseries_vs_dynamics(fit_results: dict,
                                proteins: list[str],
                                all_X: list[np.ndarray],
                                all_times: list[np.ndarray],
                                cond_labels: list[str],
                                output_path: str,
                                n_proteins: int = 9,
                                cond_idx: int = 0):
    """★ Compare measured time-series data against equation dynamics.

    For each selected protein produces a 3-panel comparison:
      Left  : Observed trajectory + Hill-model prediction (time domain)
      Middle : dX/dt from data  vs  dX/dt from equation  (derivative domain)
      Right  : Phase-plane scatter — observed X(t) vs dX/dt (data & equation)

    Proteins are selected by descending R² so the best-fit examples appear.

    Parameters
    ----------
    fit_results : dict of per-protein fit results (from any fitting mode)
    proteins    : ordered list of protein names
    all_X       : list of (T, P) observed arrays per condition
    all_times   : list of (T,) time arrays per condition
    cond_labels : list of condition labels
    output_path : PNG output path
    n_proteins  : number of proteins to display (default 9)
    cond_idx    : which condition to visualise (default 0)
    """
    times = all_times[cond_idx]
    X_obs = all_X[cond_idx]
    T = X_obs.shape[0]

    # Select proteins with valid fits, ranked by R²
    ranked = sorted(
        [(p, fit_results[p].get("r2", -np.inf))
         for p in proteins
         if p in fit_results
         and fit_results[p].get("edge_params")
         and np.isfinite(fit_results[p].get("r2", np.nan))],
        key=lambda x: -x[1]
    )[:n_proteins]

    if not ranked:
        print("  [timeseries_vs_dynamics] No valid fits to plot.")
        return

    # Compute predicted trajectory for all selected proteins at once
    sel_proteins = [p for p, _ in ranked]
    X_pred = predict_algebraic(fit_results, proteins, X_obs, times)

    n_show = len(ranked)
    n_cols = 3   # Left=ts, Middle=deriv, Right=phase
    n_rows = n_show

    fig = plt.figure(figsize=(17, 4.2 * n_rows), constrained_layout=True)
    fig.suptitle(
        f"Time-Series Data vs Equation Dynamics  —  {cond_labels[cond_idx]}",
        fontsize=15, fontweight="bold", y=1.01)

    # Colour palette: observed = dark, equation = blue, deriv = teal/orange
    C_OBS  = "#263238"   # dark slate
    C_PRED = "#1E88E5"   # blue
    C_DOBS = "#00897B"   # teal  (data derivative)
    C_DEQ  = "#F4511E"   # orange (equation derivative)
    C_ZERO = "#90A4AE"   # light grey

    outer = gridspec.GridSpec(n_rows, 1, figure=fig,
                              hspace=0.55)

    for row_idx, (pname, r2) in enumerate(ranked):
        idx   = proteins.index(pname)
        short = DISPLAY_NAMES_MULTILINE.get(pname, pname).replace("\n", " ")
        fit   = fit_results[pname]
        fit["_pname"] = pname

        # Interior time grid for derivatives (T-2 points)
        t_mid = times[1:-1]
        try:
            dX_obs_dt, dX_eq_dt = _compute_equation_dynamics(
                fit, X_obs, proteins, times)
        except Exception:
            dX_obs_dt = np.gradient(X_obs[:, idx], times)[1:-1]
            dX_eq_dt  = np.zeros(T - 2)

        # Normalise for display (each axis is independent)
        d_obs_std = np.std(dX_obs_dt) + 1e-12
        d_eq_std  = np.std(dX_eq_dt)  + 1e-12

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[row_idx],
            wspace=0.38, width_ratios=[1.6, 1.4, 1.0])

        # ── Panel A: time-series ──────────────────────────────────────────
        ax_ts = fig.add_subplot(inner[0])
        ax_ts.plot(times, X_obs[:, idx], "o-",
                   color=C_OBS, ms=3.5, lw=1.4, alpha=0.85, label="Observed")
        ax_ts.plot(times, X_pred[:, idx], "-",
                   color=C_PRED, lw=2.2, alpha=0.9, label=f"Hill fit (R²={r2:.3f})")
        ax_ts.set_title(f"A  {short}", fontsize=9.5, fontweight="bold",
                        color="#1A237E")
        ax_ts.set_xlabel("Time (min)", fontsize=8)
        ax_ts.set_ylabel("Protein level", fontsize=8)
        ax_ts.legend(fontsize=7.5, loc="best")
        ax_ts.tick_params(labelsize=7)
        ax_ts.set_facecolor("#FAFAFA")

        # ── Panel B: derivative comparison ───────────────────────────────
        ax_dv = fig.add_subplot(inner[1])
        ax_dv.axhline(0, color=C_ZERO, lw=0.8, ls="--")
        ax_dv.plot(t_mid, dX_obs_dt, "-", color=C_DOBS, lw=1.6, alpha=0.85,
                   label="dX/dt  (data)")
        ax_dv.plot(t_mid, dX_eq_dt,  "-", color=C_DEQ,  lw=1.6, alpha=0.85,
                   ls="--", label="dX/dt  (equation)")

        # Fill agreement / disagreement regions
        agreement = np.sign(dX_obs_dt) == np.sign(dX_eq_dt)
        for k in range(len(t_mid) - 1):
            fc = "#B2DFDB" if agreement[k] else "#FFCCBC"
            ax_dv.axvspan(t_mid[k], t_mid[k + 1], alpha=0.25,
                          color=fc, linewidth=0)

        ax_dv.set_title("B  dX/dt: data vs equation", fontsize=9.5,
                        fontweight="bold", color="#1A237E")
        ax_dv.set_xlabel("Time (min)", fontsize=8)
        ax_dv.set_ylabel("dX/dt", fontsize=8)
        ax_dv.legend(fontsize=7, loc="best")
        ax_dv.tick_params(labelsize=7)
        ax_dv.set_facecolor("#FAFAFA")

        # Annotation: fraction of time equation & data agree in sign
        frac_agree = agreement.mean()
        ax_dv.text(0.97, 0.05, f"sign-agree {frac_agree:.0%}",
                   transform=ax_dv.transAxes, ha="right", fontsize=7.5,
                   color="#004D40",
                   bbox=dict(boxstyle="round,pad=0.25",
                             fc="#E0F2F1", ec="#80CBC4", alpha=0.85))

        # ── Panel C: phase-plane (X vs dX/dt) ────────────────────────────
        ax_ph = fig.add_subplot(inner[2])
        x_int = X_obs[1:-1, idx]     # X values at interior time points

        # Scatter: data
        sc_obs = ax_ph.scatter(x_int, dX_obs_dt, c=t_mid,
                               cmap="viridis", s=16, alpha=0.75,
                               zorder=3, label="Data", linewidths=0)
        # Scatter: equation
        ax_ph.scatter(x_int, dX_eq_dt, c=t_mid,
                      cmap="plasma", s=16, alpha=0.55, marker="^",
                      zorder=2, label="Equation", linewidths=0)
        ax_ph.axhline(0, color=C_ZERO, lw=0.7, ls="--")
        ax_ph.axvline(0, color=C_ZERO, lw=0.7, ls="--")

        ax_ph.set_title("C  Phase-plane", fontsize=9.5,
                        fontweight="bold", color="#1A237E")
        ax_ph.set_xlabel("X (level)", fontsize=8)
        ax_ph.set_ylabel("dX/dt", fontsize=8)
        ax_ph.legend(fontsize=7, loc="upper right",
                     markerscale=1.2, framealpha=0.8)
        ax_ph.tick_params(labelsize=7)
        ax_ph.set_facecolor("#FAFAFA")

        # Pearson r between data-derivative and equation-derivative
        if len(dX_obs_dt) > 2:
            corr = np.corrcoef(dX_obs_dt, dX_eq_dt)[0, 1]
            ax_ph.text(0.03, 0.95, f"r = {corr:.3f}",
                       transform=ax_ph.transAxes, fontsize=8,
                       va="top", color="#4A148C",
                       bbox=dict(boxstyle="round,pad=0.25",
                                 fc="#EDE7F6", ec="#CE93D8", alpha=0.85))

        # Cleanup hidden _pname key
        fit.pop("_pname", None)

    _save_fig(fig, output_path)
    print(f"  ✓ Time-series vs dynamics: {os.path.basename(output_path)}")


def plot_dynamics_summary_heatmap(fit_results: dict,
                                  proteins: list[str],
                                  all_X: list[np.ndarray],
                                  all_times: list[np.ndarray],
                                  output_path: str,
                                  cond_idx: int = 0):
    """Heatmap of derivative correlation (data dX/dt vs equation dX/dt).

    Rows = proteins, columns = conditions.  Colour = Pearson r.
    Highlights proteins where the fitted equation captures dynamics well
    (warm) vs. poorly (cool).
    """
    times = all_times[cond_idx]
    X_obs = all_X[cond_idx]

    valid = [(p, fit_results[p])
             for p in proteins
             if p in fit_results
             and fit_results[p].get("edge_params")
             and np.isfinite(fit_results[p].get("r2", np.nan))]
    if not valid:
        return

    n_show = len(valid)
    corr_vals  = np.full(n_show, np.nan)
    agree_vals = np.full(n_show, np.nan)
    r2_vals    = np.full(n_show, np.nan)
    short_names = []

    for row_i, (pname, fit) in enumerate(valid):
        fit["_pname"] = pname
        try:
            dX_obs_dt, dX_eq_dt = _compute_equation_dynamics(
                fit, X_obs, proteins, times)
            if len(dX_obs_dt) > 2:
                corr_vals[row_i]  = np.corrcoef(dX_obs_dt, dX_eq_dt)[0, 1]
                agree_vals[row_i] = (np.sign(dX_obs_dt) == np.sign(dX_eq_dt)).mean()
        except Exception:
            pass
        r2_vals[row_i] = fit.get("r2", np.nan)
        short_names.append(DISPLAY_NAMES_MULTILINE.get(pname, pname).replace("\n", " "))
        fit.pop("_pname", None)

    # Sort by correlation (descending)
    order = np.argsort(corr_vals)[::-1]
    corr_sorted  = corr_vals[order]
    agree_sorted = agree_vals[order]
    r2_sorted    = r2_vals[order]
    names_sorted = [short_names[k] for k in order]

    fig, axes = plt.subplots(1, 3, figsize=(20, max(4, n_show * 0.35 + 2)),
                             constrained_layout=True)

    for ax, vals, title, cmap, vmin, vmax in [
        (axes[0], corr_sorted[:, np.newaxis],
         "Derivative Pearson r\n(data vs equation)", "RdYlGn", -1, 1),
        (axes[1], agree_sorted[:, np.newaxis],
         "Sign-agreement fraction\n(data vs equation)", "RdYlGn", 0, 1),
        (axes[2], r2_sorted[:, np.newaxis],
         "Trajectory R²\n(Hill fit)", "Blues", 0, 1),
    ]:
        masked = np.ma.masked_invalid(vals)
        im = ax.imshow(masked, cmap=cmap, aspect="auto",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(names_sorted, fontsize=6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

        # Annotate cells
        for row_i in range(n_show):
            v = vals[row_i, 0]
            if np.isfinite(v):
                ax.text(0, row_i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="black",
                        fontweight="bold" if abs(v) > 0.7 else "normal")

    fig.suptitle("Dynamics Summary Heatmap  —  Data vs Equation",
                 fontsize=13, fontweight="bold")
    _save_fig(fig, output_path)
    print(f"  ✓ Dynamics heatmap: {os.path.basename(output_path)}")


def plot_algebraic_fits(fit_results: dict,
                        proteins: list[str],
                        all_X: list[np.ndarray],
                        all_times: list[np.ndarray],
                        cond_labels: list[str],
                        output_path: str,
                        n_examples: int = 6):
    """Overlay observed vs algebraic-Hill-predicted levels (best-R² proteins)."""
    r2_list = [(p, fit_results[p]["r2"])
               for p in proteins
               if p in fit_results and np.isfinite(fit_results[p].get("r2", np.nan))]
    r2_list.sort(key=lambda x: -x[1])
    examples = [p for p, _ in r2_list[:n_examples]]
    if not examples:
        return

    cond_idx = 0
    times = all_times[cond_idx]
    X_obs = all_X[cond_idx]
    X_pred = predict_algebraic(fit_results, proteins, X_obs, times)

    n_cols = 3
    n_rows = (len(examples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for k, pname in enumerate(examples):
        row, col = k // n_cols, k % n_cols
        ax = axes[row, col]
        idx = proteins.index(pname)
        short = DISPLAY_NAMES_MULTILINE.get(pname, pname).replace("\n", " ")

        ax.plot(times, X_obs[:, idx], "o-", color="#333333", ms=3, lw=1.5,
                label="Observed", alpha=0.8)
        ax.plot(times, X_pred[:, idx], "-", color="#1E88E5", lw=2,
                label="Hill fit", alpha=0.9)

        r2 = fit_results[pname]["r2"]
        ax.set_title(f"{short}  (R²={r2:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (min)", fontsize=9)
        ax.set_ylabel("Level", fontsize=9)
        ax.legend(fontsize=8)

    for k in range(len(examples), n_rows * n_cols):
        row, col = k // n_cols, k % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"Algebraic Hill Fits — {cond_labels[cond_idx]}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, output_path)


def plot_algebraic_all(fit_results: dict,
                       proteins: list[str],
                       all_X: list[np.ndarray],
                       all_times: list[np.ndarray],
                       cond_labels: list[str],
                       output_path: str,
                       cond_idx: int = 0):
    """Full grid: all proteins, one condition — observed vs algebraic pred."""
    times = all_times[cond_idx]
    X_obs = all_X[cond_idx]
    X_pred = predict_algebraic(fit_results, proteins, X_obs, times)

    P = len(proteins)
    n_cols = 6
    n_rows = (P + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.ravel()

    for i, pname in enumerate(proteins):
        ax = axes[i]
        short = DISPLAY_NAMES_MULTILINE.get(pname, pname).replace("\n", " ")
        ax.plot(times, X_obs[:, i], "-", color="#333333", lw=1.2, alpha=0.8)
        ax.plot(times, X_pred[:, i], "-", color="#1E88E5", lw=1.5, alpha=0.85)
        r2 = fit_results.get(pname, {}).get("r2", np.nan)
        r2_str = f"{r2:.2f}" if np.isfinite(r2) else "n/a"
        ax.set_title(f"{short}\nR²={r2_str}", fontsize=7, pad=2)
        ax.tick_params(labelsize=5)

    for i in range(P, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"All Proteins — {cond_labels[cond_idx]}  "
                 "(black=obs, blue=algebraic Hill)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, output_path)


def plot_hill_params_heatmap(fit_results: dict, proteins: list[str],
                             output_path: str):
    """Heatmaps of w, K, n across the edge matrix."""
    P = len(proteins)
    w_mat = np.full((P, P), np.nan)
    K_mat = np.full((P, P), np.nan)
    n_mat = np.full((P, P), np.nan)

    for i, target in enumerate(proteins):
        fit = fit_results.get(target)
        if fit is None or not fit.get("edge_params"):
            continue
        for k, ep in enumerate(fit["edge_params"]):
            j = fit["reg_indices"][k]
            w_mat[j, i] = ep["w"]
            K_mat[j, i] = ep["K"]
            n_mat[j, i] = ep["n"]

    short = [DISPLAY_NAMES_MULTILINE.get(p, p).replace("\n", " ") for p in proteins]

    fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    for ax, mat, label, cmap in [
        (axes[0], w_mat, "w  (max rate)",    "YlOrRd"),
        (axes[1], K_mat, "K  (EC50)",        "YlGnBu"),
        (axes[2], n_mat, "n  (Hill coeff)",  "Purples"),
    ]:
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(P));  ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.set_yticks(range(P));  ax.set_yticklabels(short, fontsize=6)
        ax.set_xlabel("Target i", fontsize=10)
        ax.set_ylabel("Source j", fontsize=10)
        ax.set_title(label, fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    _save_fig(fig, output_path)


# ============================================================================
# Export
# ============================================================================
def export_hill_params(fit_results: dict, proteins: list[str],
                       output_path: str) -> pd.DataFrame:
    """Export fitted Hill parameters as an edge-table CSV."""
    rows = []
    for target in proteins:
        fit = fit_results.get(target)
        if fit is None or not fit.get("edge_params"):
            continue
        for k, ep in enumerate(fit["edge_params"]):
            source = fit["reg_names"][k]
            in_prior = (source, target) in _PRIOR_EDGE_SET or \
                       (target, source) in _PRIOR_EDGE_SET
            row = {
                "source": source,
                "target": target,
                "model":  fit.get("model", ""),
                "sign":   ep["sign"],
                "w":      ep["w"],
                "K":      ep["K"],
                "n":      ep["n"],
                "target_gamma": fit["gamma"],
                "target_basal": fit["basal"],
                "target_r2":    fit["r2"],
                "in_prior":     in_prior,
            }
            # Optional fields from specific models
            if "beta" in ep:
                row["beta"] = ep["beta"]
            if "g" in ep:
                row["g"] = ep["g"]
                row["h"] = ep["h"]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Hill parameters: {os.path.basename(output_path)} "
          f"({len(df)} edges)")
    return df


def export_fit_quality(fit_results: dict, proteins: list[str],
                       output_path: str) -> pd.DataFrame:
    """Export per-protein fit quality summary."""
    rows = []
    for p in proteins:
        fit = fit_results.get(p)
        if fit is None:
            continue
        row = {
            "protein":       p,
            "model":         fit.get("model", ""),
            "n_regulators":  len(fit.get("edge_params", [])),
            "r_squared":     fit.get("r2", np.nan),
            "rmse":          fit.get("rmse", np.nan),
            "gamma":         fit.get("gamma", np.nan),
            "basal":         fit.get("basal", np.nan),
            "dae_type":      fit.get("dae_type", ""),
        }
        if "bic" in fit:
            row["bic"] = fit["bic"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Fit quality: {os.path.basename(output_path)}")
    return df


# ============================================================================
# Main Pipeline
# ============================================================================
def run_hill_pipeline(data_dir: str, sindy_dir: str, output_dir: str,
                      source: str = "teacher",
                      early_weight_tau: float = 30.0,
                      dae_fast_frac: float = 0.5,
                      dae_alg_mode: str = "multiplicative",
                      n_restarts: int = 3,
                      **_ignored):
    """Run Hill function fitting pipeline.

    Mode is determined automatically by source:
        teacher → de_auto  (best-RMSE auto-select across 15 ODE combos)
        student → dae_auto  (fast→auto-select ODE, slow→algebraic)

    Parameters
    ----------
    early_weight_tau : exponential decay τ for early-dynamics weighting
    dae_fast_frac : fraction of proteins treated as fast (for student/dae_auto)
    dae_alg_mode : algebraic model for slow proteins in DAE mode
    n_restarts : number of random restarts for Hill ODE fitting
    """
    mode = "dae_auto" if source == "student" else "de_auto"
    os.makedirs(output_dir, exist_ok=True)

    # ── load network ──
    consensus_path = os.path.join(sindy_dir, f"W_consensus_{source}.csv")
    if not os.path.exists(consensus_path):
        raise FileNotFoundError(f"Consensus matrix not found: {consensus_path}")

    W, proteins, edges = load_sindy_network(consensus_path)
    P = len(proteins)
    n_edges = len(edges)

    mode_label = {"de_auto": "Best-RMSE Auto-Select (15 ODE combos)",
                  "dae_auto": "DAE (Auto-Select ODE + Algebraic)"}.get(mode, mode)
    print(f"{'═' * 70}")
    print(f"  Hill Function Fitting ({mode_label})  —  source: {source}")
    print(f"  Network: {P} proteins, {n_edges} directed edges")
    print(f"{'═' * 70}")

    # ── load data ──
    conditions = discover_conditions(data_dir, source)
    if not conditions:
        raise FileNotFoundError(f"No files for source={source!r} in {data_dir}")

    print(f"  Loading {len(conditions)} conditions ...")

    all_X_raw: list[np.ndarray] = []
    all_times_raw: list[np.ndarray] = []
    cond_labels: list[str] = []

    for label, filepath in conditions:
        times, X, pnames = load_ts_csv(filepath)
        assert pnames == proteins, f"Protein order mismatch in {label}"
        all_X_raw.append(X)
        all_times_raw.append(times)
        cond_labels.append(label)

    print(f"  Data: {len(conditions)} conditions × "
          f"{all_X_raw[0].shape[0]} time points × {P} proteins")

    # ── load per-condition W matrices (for condition-specific causal masking) ──
    cond_W_list: list[np.ndarray] | None = None
    wpc_dir = os.path.join(sindy_dir, "W_per_condition", source)
    if os.path.isdir(wpc_dir):
        cond_W_list = []
        n_loaded = 0
        for label in cond_labels:
            wpc_path = os.path.join(wpc_dir, f"W_{label}.csv")
            if os.path.exists(wpc_path):
                Wc, _, _ = load_sindy_network(wpc_path)
                cond_W_list.append(Wc)
                n_loaded += 1
            else:
                cond_W_list.append(W)  # fallback to consensus
        print(f"  Per-condition W: loaded {n_loaded}/{len(cond_labels)} "
              f"causal matrices from {wpc_dir}")
    else:
        print(f"  Per-condition W: not found, using consensus W for all conditions")

    # ── fit Hill parameters ──
    print(f"\n{'─' * 70}")
    if mode == "dae_auto":
        print(f"  DAE-Auto: fast → auto-select ODE (15 combos), slow → {dae_alg_mode}")
        print(f"  fast_frac={dae_fast_frac:.2f}, n_restarts={n_restarts}, "
              f"early_weight_tau={early_weight_tau:.1f}")
    else:
        print(f"  Best-RMSE Auto-Select:  fitting 15 ODE combos per protein ...")
        print(f"  n_restarts={n_restarts}, "
              f"early_weight_tau={early_weight_tau:.1f}")
    print(f"{'─' * 70}")

    _ode_kw = dict(early_weight_tau=early_weight_tau,
                   cond_W_list=cond_W_list, n_restarts=n_restarts)

    if mode == "dae_auto":
        fit_results = fit_all_proteins_dae(W, proteins, all_X_raw, all_times_raw,
                                           fast_frac=dae_fast_frac,
                                           dae_alg_mode=dae_alg_mode,
                                           early_weight_tau=early_weight_tau,
                                           cond_W_list=cond_W_list,
                                           n_restarts=n_restarts)
    else:  # de_auto
        fit_results = fit_all_proteins_auto(
            W, proteins, all_X_raw, all_times_raw, **_ode_kw)

    # summary
    r2_vals = [fit_results[p]["r2"] for p in proteins
               if p in fit_results and np.isfinite(fit_results[p].get("r2", np.nan))]
    if r2_vals:
        print(f"\n  R² summary — median: {np.median(r2_vals):.4f}  "
              f"mean: {np.mean(r2_vals):.4f}  "
              f"min: {np.min(r2_vals):.4f}  max: {np.max(r2_vals):.4f}")

    # ── export ──
    print(f"\n{'─' * 70}")
    print(f"  Exporting results ...")
    print(f"{'─' * 70}")

    export_hill_params(fit_results, proteins,
                       os.path.join(output_dir, f"hill_params_{source}.csv"))
    export_fit_quality(fit_results, proteins,
                       os.path.join(output_dir, f"fit_quality_{source}.csv"))

    # ── plots ──
    print(f"\n{'─' * 70}")
    print(f"  Generating plots ...")
    print(f"{'─' * 70}")

    plot_fit_quality(fit_results, proteins,
                     os.path.join(output_dir, f"fit_quality_{source}.png"))

    plot_algebraic_fits(fit_results, proteins, all_X_raw, all_times_raw,
                        cond_labels,
                        os.path.join(output_dir,
                                     f"algebraic_fits_{source}.png"))
    plot_algebraic_all(fit_results, proteins, all_X_raw, all_times_raw,
                       cond_labels,
                       os.path.join(output_dir,
                                    f"algebraic_all_{source}.png"))

    plot_hill_params_heatmap(fit_results, proteins,
                             os.path.join(output_dir,
                                          f"hill_params_heatmap_{source}.png"))

    # ★ NEW: time-series vs equation dynamics comparison
    plot_timeseries_vs_dynamics(
        fit_results, proteins, all_X_raw, all_times_raw, cond_labels,
        os.path.join(output_dir, f"timeseries_vs_dynamics_{source}.png"),
        n_proteins=min(9, len(proteins)))

    plot_dynamics_summary_heatmap(
        fit_results, proteins, all_X_raw, all_times_raw,
        os.path.join(output_dir, f"dynamics_heatmap_{source}.png"))

    # ── network plot (reusing sindy visualisation with Hill-derived weights) ──
    W_hill = np.zeros((P, P))
    for target in proteins:
        fit = fit_results.get(target)
        if fit is None or not fit.get("edge_params"):
            continue
        i = proteins.index(target)
        for k, ep in enumerate(fit["edge_params"]):
            j = fit["reg_indices"][k]
            sign = 1.0 if ep["sign"] == "activation" else -1.0
            W_hill[j, i] = sign * ep["w"]

    G = weight_matrix_to_digraph(W_hill, proteins)
    plot_sindy_network(G, f"Hill Function Network ({source})",
                       os.path.join(output_dir, f"hill_network_{source}.png"))

    return fit_results, proteins


def compare_teacher_student_hill(data_dir: str, sindy_dir: str,
                                 output_dir: str, **kwargs):
    """Run Hill fitting on both teacher and student, then compare.

    Teacher uses de_auto, student uses dae_auto (determined automatically).
    """
    print("\n" + "█" * 70)
    print("  TEACHER — Hill Function Fitting (de_auto)")
    print("█" * 70)
    fit_teacher, proteins = run_hill_pipeline(
        data_dir, sindy_dir, output_dir, source="teacher", **kwargs)

    print("\n" + "█" * 70)
    print("  STUDENT — Hill Function Fitting (dae_auto)")
    print("█" * 70)
    fit_student, _ = run_hill_pipeline(
        data_dir, sindy_dir, output_dir, source="student", **kwargs)

    # ── comparison table ──
    print(f"\n{'─' * 70}")
    print(f"  Comparing Teacher vs Student Hill parameters ...")
    print(f"{'─' * 70}")

    rows = []
    for p in proteins:
        ft = fit_teacher.get(p, {})
        fs = fit_student.get(p, {})

        t_edges = {}
        if ft and ft.get("edge_params"):
            for k, ep in enumerate(ft["edge_params"]):
                t_edges[ft["reg_names"][k]] = ep
        s_edges = {}
        if fs and fs.get("edge_params"):
            for k, ep in enumerate(fs["edge_params"]):
                s_edges[fs["reg_names"][k]] = ep

        for src in set(t_edges) | set(s_edges):
            te = t_edges.get(src)
            se = s_edges.get(src)
            rows.append({
                "source":    src,
                "target":    p,
                "teacher_w": te["w"] if te else 0.0,
                "teacher_K": te["K"] if te else np.nan,
                "teacher_n": te["n"] if te else np.nan,
                "student_w": se["w"] if se else 0.0,
                "student_K": se["K"] if se else np.nan,
                "student_n": se["n"] if se else np.nan,
                "category":  ("shared" if te and se else
                              "teacher_only" if te else "student_only"),
            })

    df = pd.DataFrame(rows)
    comp_path = os.path.join(output_dir, "hill_comparison_teacher_student.csv")
    df.to_csv(comp_path, index=False)

    n_shared  = (df["category"] == "shared").sum()
    n_t_only  = (df["category"] == "teacher_only").sum()
    n_s_only  = (df["category"] == "student_only").sum()
    print(f"  Shared:       {n_shared}")
    print(f"  Teacher only: {n_t_only}")
    print(f"  Student only: {n_s_only}")
    print(f"  ✓ Comparison: {os.path.basename(comp_path)}")

    # ── scatter plot: teacher vs student w ──
    mask = df["category"] == "shared"
    if mask.any():
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(df.loc[mask, "teacher_w"], df.loc[mask, "student_w"],
                   alpha=0.5, s=18, color="#5C6BC0")
        lim = max(df.loc[mask, "teacher_w"].max(),
                  df.loc[mask, "student_w"].max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
        ax.set_xlabel("Teacher w", fontsize=12)
        ax.set_ylabel("Student w", fontsize=12)
        ax.set_title("Hill w: Teacher vs Student (shared edges)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig, os.path.join(output_dir,
                                    "hill_w_teacher_vs_student.png"))


# ============================================================================
# Global Network Fitting  —  PyTorch + torchdiffeq on GPU/MPS
# ============================================================================

def _select_device() -> torch.device:
    """Pick best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HillODEFunc(nn.Module):
    """Full-network Hill ODE with per-protein auto-selected component combos.

    dX_i/dt = −γ_i · X_i + F_add_i + F_mult_i + F_ratio_i + F_ss_i

    where each component is enabled/disabled per protein via static combo flags
    compiled from the per-protein auto-selection results:

      F_add   = basal + Σ_j β_j · h_j(X_j)         (additive Hill)
      F_mult  = Vmax  · ∏_j h_j(X_j)                (multiplicative Hill)
      F_ratio = Vmax_r · V_kin / (V_kin + V_phos)   (push-pull ratio)
      F_ss    = α · ∏_j X_j^{g_j} − β_s · ∏_j X_j^{h_j}  (S-system)

    Shared per edge:  K, n  (Hill half-max & coefficient)
    Additive:         β per edge, basal per target
    Multiplicative:   Vmax per target
    Ratio:            w_ratio per edge, Vmax_ratio/w0/g0 per target
    S-system:         g, h per edge, alpha/beta_s per target

    All positivity constraints via log-space or softplus.
    Combo flags are boolean buffers (no gradient), making the computation
    graph static per protein — unused components are zero-masked away.
    """

    def __init__(self, P: int, edge_indices: list[tuple[int, int]],
                 edge_signs: list[int],
                 combo_flags: np.ndarray,
                 K_init: np.ndarray, n_init: np.ndarray,
                 beta_init: np.ndarray,
                 gamma_init: np.ndarray, basal_init: np.ndarray,
                 Vmax_init: np.ndarray,
                 w_ratio_init: np.ndarray,
                 Vmax_ratio_init: np.ndarray,
                 w0_ratio_init: np.ndarray, g0_ratio_init: np.ndarray,
                 g_init: np.ndarray, h_init: np.ndarray,
                 alpha_init: np.ndarray, beta_s_init: np.ndarray):
        super().__init__()
        self.P = P
        E = len(edge_indices)

        # ── Edge topology (fixed) ──
        src_idx = [e[0] for e in edge_indices]
        tgt_idx = [e[1] for e in edge_indices]
        self.register_buffer("src", torch.tensor(src_idx, dtype=torch.long))
        self.register_buffer("tgt", torch.tensor(tgt_idx, dtype=torch.long))
        self.register_buffer("sign", torch.tensor(edge_signs, dtype=torch.float32))

        # ── Per-protein combo flags: (P, 4) bool — [add, mult, ratio, ss] ──
        self.register_buffer("combo_flags",
                             torch.tensor(combo_flags, dtype=torch.bool))
        # Derived convenience masks: (P,)
        self.register_buffer("use_add",   self.combo_flags[:, 0])
        self.register_buffer("use_mult",  self.combo_flags[:, 1])
        self.register_buffer("use_ratio", self.combo_flags[:, 2])
        self.register_buffer("use_ss",    self.combo_flags[:, 3])
        # Which proteins need Hill values at all (add / mult / ratio)
        self.register_buffer("use_hill",
                             self.combo_flags[:, :3].any(dim=1))
        # Basal is active when add or mult is active
        self.register_buffer("use_basal",
                             self.combo_flags[:, 0] | self.combo_flags[:, 1])

        # ── Per-target edge grouping ──
        tgt_edge_map: dict[int, list[int]] = {}
        for eidx, (_, ti) in enumerate(edge_indices):
            tgt_edge_map.setdefault(ti, []).append(eidx)
        max_reg = max((len(v) for v in tgt_edge_map.values()), default=0)

        _edge_idx_pad = torch.zeros(P, max_reg, dtype=torch.long)
        _edge_mask    = torch.zeros(P, max_reg, dtype=torch.bool)
        for ti, eidxs in tgt_edge_map.items():
            for k, eidx in enumerate(eidxs):
                _edge_idx_pad[ti, k] = eidx
                _edge_mask[ti, k] = True
        self.register_buffer("_edge_idx_pad", _edge_idx_pad)
        self.register_buffer("_edge_mask", _edge_mask)
        self._max_reg = max_reg

        # ── Shared per-edge parameters ──
        self.log_K = nn.Parameter(torch.tensor(
            np.log(K_init + 1e-8), dtype=torch.float32))        # (E,)
        self.log_n = nn.Parameter(torch.tensor(
            np.log(n_init), dtype=torch.float32))                # (E,)

        # ── Additive component: β per edge, basal per target ──
        self._raw_beta = nn.Parameter(torch.tensor(
            beta_init, dtype=torch.float32))                     # (E,)
        self.basal = nn.Parameter(torch.tensor(
            basal_init, dtype=torch.float32))                    # (P,)

        # ── Multiplicative component: Vmax per target ──
        self._raw_Vmax = nn.Parameter(torch.tensor(
            Vmax_init, dtype=torch.float32))                     # (P,)

        # ── Ratio component: w_ratio per edge, Vmax_r/w0/g0 per target ──
        self._raw_w_ratio = nn.Parameter(torch.tensor(
            w_ratio_init, dtype=torch.float32))                  # (E,)
        self._raw_Vmax_ratio = nn.Parameter(torch.tensor(
            Vmax_ratio_init, dtype=torch.float32))               # (P,)
        self._raw_w0 = nn.Parameter(torch.tensor(
            w0_ratio_init, dtype=torch.float32))                 # (P,)
        self._raw_g0 = nn.Parameter(torch.tensor(
            g0_ratio_init, dtype=torch.float32))                 # (P,)

        # ── S-system component: g, h per edge, alpha/beta_s per target ──
        self.ss_g = nn.Parameter(torch.tensor(
            g_init, dtype=torch.float32))                        # (E,)
        self.ss_h = nn.Parameter(torch.tensor(
            h_init, dtype=torch.float32))                        # (E,)
        self._raw_alpha = nn.Parameter(torch.tensor(
            alpha_init, dtype=torch.float32))                    # (P,)
        self._raw_beta_s = nn.Parameter(torch.tensor(
            beta_s_init, dtype=torch.float32))                   # (P,)

        # ── Degradation ──
        self._raw_gamma = nn.Parameter(torch.tensor(
            gamma_init, dtype=torch.float32))                    # (P,)

    # ── Property accessors with positivity constraints ──
    @property
    def K(self):
        return torch.exp(self.log_K)

    @property
    def n(self):
        return torch.clamp(torch.exp(self.log_n), 0.5, 5.0)

    @property
    def gamma(self):
        return nn.functional.softplus(self._raw_gamma)

    @property
    def Vmax(self):
        return nn.functional.softplus(self._raw_Vmax)

    @property
    def beta(self):
        return nn.functional.softplus(self._raw_beta)

    @property
    def w_ratio(self):
        return nn.functional.softplus(self._raw_w_ratio)

    @property
    def Vmax_ratio(self):
        return nn.functional.softplus(self._raw_Vmax_ratio)

    @property
    def w0(self):
        return nn.functional.softplus(self._raw_w0)

    @property
    def g0(self):
        return nn.functional.softplus(self._raw_g0)

    @property
    def alpha_ss(self):
        return nn.functional.softplus(self._raw_alpha)

    @property
    def beta_ss(self):
        return nn.functional.softplus(self._raw_beta_s)

    def forward(self, t, X_flat):
        """ODE right-hand side:  dX/dt for all proteins × all conditions.

        X_flat: (C*P,) flattened.  Set self._C before odeint.
        Returns: (C*P,).
        """
        C = getattr(self, "_C", 1)
        P = self.P
        X = X_flat.view(C, P)                               # (C, P)

        K = self.K                                           # (E,)
        n = self.n                                           # (E,)

        # ── Hill values for all edges: h(X_src; K, n, sign) ──
        x_src = torch.clamp(X[:, self.src], min=1e-12)      # (C, E)
        xn = torch.pow(x_src, n.unsqueeze(0))                # (C, E)
        Kn = torch.pow(K.unsqueeze(0), n.unsqueeze(0))       # (1, E)
        h_act = xn / (Kn + xn)                               # (C, E)
        is_act = (self.sign > 0).unsqueeze(0)                 # (1, E)
        # Sign-dependent: activator → h_act, inhibitor → 1-h_act
        h = torch.where(is_act, h_act, 1.0 - h_act)         # (C, E)

        dXdt = -self.gamma * X                                # (C, P)

        # ────────── F_add: basal + Σ β_j · h_j ──────────
        if self.use_add.any():
            # Gather β·h per edge, then scatter-add to targets
            beta_h = self.beta.unsqueeze(0) * h               # (C, E)
            # Sum per target using padded gather
            beta_h_pad = torch.zeros(C, beta_h.shape[1] + 1,
                                      device=X.device)
            beta_h_pad[:, :beta_h.shape[1]] = beta_h
            beta_h_gathered = beta_h_pad[:, self._edge_idx_pad]  # (C, P, max_reg)
            beta_h_gathered = torch.where(self._edge_mask.unsqueeze(0),
                                           beta_h_gathered,
                                           torch.zeros_like(beta_h_gathered))
            sum_beta_h = beta_h_gathered.sum(dim=2)            # (C, P)
            F_add = self.basal + sum_beta_h                    # (C, P)
            dXdt = dXdt + F_add * self.use_add.float()

        # ────────── F_mult: Vmax · ∏ h_j ──────────
        if self.use_mult.any():
            h_pad = torch.ones(C, h.shape[1] + 1, device=X.device)
            h_pad[:, :h.shape[1]] = h
            h_gathered = h_pad[:, self._edge_idx_pad]          # (C, P, max_reg)
            h_gathered = torch.where(self._edge_mask.unsqueeze(0),
                                      h_gathered,
                                      torch.ones_like(h_gathered))
            prod_h = h_gathered.prod(dim=2)                    # (C, P)
            # Add basal for mult-only proteins (those with mult but not add)
            mult_basal = self.basal * self.use_basal.float() * (
                ~self.use_add & self.use_mult).float()
            F_mult = mult_basal + self.Vmax * prod_h
            dXdt = dXdt + F_mult * self.use_mult.float()

        # ────────── F_ratio: Vmax_r · V_kin / (V_kin + V_phos) ──────────
        if self.use_ratio.any():
            # w_ratio · h_act (always activator-style for ratio)
            wr_h = self.w_ratio.unsqueeze(0) * h_act           # (C, E)
            wr_h_pad = torch.zeros(C, wr_h.shape[1] + 1,
                                    device=X.device)
            wr_h_pad[:, :wr_h.shape[1]] = wr_h
            wr_h_gathered = wr_h_pad[:, self._edge_idx_pad]    # (C, P, max_reg)

            # Build act/inh edge masks per target
            edge_is_act = is_act[:, :].expand(C, -1)           # (C, E)
            act_pad = torch.zeros(C, edge_is_act.shape[1] + 1,
                                   dtype=torch.bool, device=X.device)
            act_pad[:, :edge_is_act.shape[1]] = edge_is_act
            act_gathered = act_pad[:, self._edge_idx_pad]      # (C, P, max_reg)

            kin_vals = torch.where(
                self._edge_mask.unsqueeze(0) & act_gathered,
                wr_h_gathered, torch.zeros_like(wr_h_gathered))
            phos_vals = torch.where(
                self._edge_mask.unsqueeze(0) & ~act_gathered,
                wr_h_gathered, torch.zeros_like(wr_h_gathered))

            V_kin  = self.w0 + kin_vals.sum(dim=2)             # (C, P)
            V_phos = self.g0 + phos_vals.sum(dim=2)            # (C, P)
            F_ratio = self.Vmax_ratio * V_kin / (V_kin + V_phos + 1e-15)
            dXdt = dXdt + F_ratio * self.use_ratio.float()

        # ────────── F_ss: α · ∏ X^g − β_s · ∏ X^h ──────────
        if self.use_ss.any():
            x_src_ss = torch.clamp(X[:, self.src], min=1e-12)  # (C, E)
            # Generation: ∏ X_j^{g_j}
            xg = torch.pow(x_src_ss, self.ss_g.unsqueeze(0))   # (C, E)
            xg_pad = torch.ones(C, xg.shape[1] + 1, device=X.device)
            xg_pad[:, :xg.shape[1]] = xg
            xg_gathered = xg_pad[:, self._edge_idx_pad]        # (C, P, max_reg)
            xg_gathered = torch.where(self._edge_mask.unsqueeze(0),
                                       xg_gathered,
                                       torch.ones_like(xg_gathered))
            prod_g = xg_gathered.prod(dim=2)                   # (C, P)

            # Degradation: ∏ X_j^{h_j}
            xh = torch.pow(x_src_ss, self.ss_h.unsqueeze(0))
            xh_pad = torch.ones(C, xh.shape[1] + 1, device=X.device)
            xh_pad[:, :xh.shape[1]] = xh
            xh_gathered = xh_pad[:, self._edge_idx_pad]
            xh_gathered = torch.where(self._edge_mask.unsqueeze(0),
                                       xh_gathered,
                                       torch.ones_like(xh_gathered))
            prod_h_ss = xh_gathered.prod(dim=2)

            F_ss = self.alpha_ss * prod_g - self.beta_ss * prod_h_ss
            dXdt = dXdt + F_ss * self.use_ss.float()

        return dXdt.reshape(-1)


def _init_hill_ode_params(W: np.ndarray, proteins: list[str],
                          all_X_raw: list[np.ndarray],
                          fit_results: dict | None = None) -> dict:
    """Initialise all Hill ODE parameters from SINDy network + data statistics.

    If fit_results is provided (from per-protein auto-select fitting), use
    those as warm-start initial values and compile the per-protein combo flags.
    Otherwise default to mult-only for all proteins.

    Returns dict with keys:
        edge_indices, edge_signs, combo_flags (P,4),
        K_init, n_init, beta_init, gamma_init, basal_init, Vmax_init,
        w_ratio_init, Vmax_ratio_init, w0_ratio_init, g0_ratio_init,
        g_init, h_init, alpha_init, beta_s_init
    """
    P = len(proteins)
    X_pool = np.vstack(all_X_raw)
    medians = np.median(X_pool, axis=0)
    stds    = np.std(X_pool, axis=0)

    edge_indices: list[tuple[int, int]] = []
    edge_signs:   list[int] = []
    for j in range(P):
        for i in range(P):
            if j != i and abs(W[j, i]) > 1e-12:
                edge_indices.append((j, i))
                edge_signs.append(1 if W[j, i] > 0 else -1)

    E = len(edge_indices)

    # Build reverse lookup: (j, i) → edge index
    edge_lookup: dict[tuple[int, int], int] = {
        (j, i): eidx for eidx, (j, i) in enumerate(edge_indices)
    }

    # Shared per-edge
    K_init      = np.full(E, 0.0, dtype=np.float64)
    n_init      = np.full(E, 2.0, dtype=np.float64)
    # Additive per-edge
    beta_init   = np.full(E, 0.1, dtype=np.float64)
    # Ratio per-edge
    w_ratio_init = np.full(E, 1.0, dtype=np.float64)
    # S-system per-edge
    g_init      = np.full(E, 0.5, dtype=np.float64)
    h_init      = np.full(E, 0.5, dtype=np.float64)

    # Per-protein
    gamma_init      = np.full(P, 0.05, dtype=np.float64)
    basal_init      = np.zeros(P, dtype=np.float64)
    Vmax_init       = np.zeros(P, dtype=np.float64)
    Vmax_ratio_init = np.full(P, 1.0, dtype=np.float64)
    w0_ratio_init   = np.full(P, 0.1, dtype=np.float64)
    g0_ratio_init   = np.full(P, 0.1, dtype=np.float64)
    alpha_init      = np.full(P, 1.0, dtype=np.float64)
    beta_s_init     = np.full(P, 1.0, dtype=np.float64)

    # Combo flags: (P, 4) — [add, mult, ratio, ss]
    combo_flags = np.zeros((P, 4), dtype=np.bool_)

    if fit_results is not None:
        for i, pname in enumerate(proteins):
            fit = fit_results.get(pname)
            if fit is None:
                combo_flags[i, 1] = True  # default: mult
                Vmax_init[i] = stds[i] + 1e-3
                basal_init[i] = medians[i] * 0.1
                continue

            # Extract combo flags from per-protein auto-select results
            cf = fit.get("combo_flags")
            if cf:
                combo_flags[i, 0] = cf.get("add", False)
                combo_flags[i, 1] = cf.get("mult", False)
                combo_flags[i, 2] = cf.get("ratio", False)
                combo_flags[i, 3] = cf.get("ss", False)
            else:
                combo_flags[i, 1] = True  # fallback: mult

            gamma_init[i] = max(fit.get("gamma", 0.05), 1e-4)
            basal_init[i] = fit.get("basal", 0.0)
            Vmax_init[i]  = fit.get("vmax", stds[i] + 1e-3)

            # Ratio per-protein params
            if combo_flags[i, 2]:
                Vmax_ratio_init[i] = fit.get("vmax_ratio", 1.0)
                w0_ratio_init[i]   = fit.get("w0", 0.1)
                g0_ratio_init[i]   = fit.get("gamma0", 0.1)

            # S-system per-protein params
            if combo_flags[i, 3]:
                alpha_init[i]  = fit.get("alpha", 1.0)
                beta_s_init[i] = fit.get("beta_s", 1.0)

            # Per-edge params
            if fit.get("edge_params"):
                reg_idx = fit.get("reg_indices", [])
                for k, ep in enumerate(fit["edge_params"]):
                    j = reg_idx[k]
                    eidx = edge_lookup.get((j, i))
                    if eidx is None:
                        continue
                    K_init[eidx] = max(ep.get("K", medians[j] + 1e-6), 1e-6)
                    n_init[eidx] = np.clip(ep.get("n", 2.0), 0.5, 5.0)
                    if "beta" in ep:
                        beta_init[eidx] = ep["beta"]
                    if "w_ratio" in ep:
                        w_ratio_init[eidx] = ep["w_ratio"]
                    if "g" in ep:
                        g_init[eidx] = ep["g"]
                    if "h" in ep:
                        h_init[eidx] = ep["h"]
    else:
        # Cold start: default all proteins to mult-only
        combo_flags[:, 1] = True
        for eidx, (j, i) in enumerate(edge_indices):
            K_init[eidx] = medians[j] + 1e-6
        for i in range(P):
            Vmax_init[i] = stds[i] + 1e-3
            basal_init[i] = medians[i] * 0.1

    return dict(
        edge_indices=edge_indices, edge_signs=edge_signs,
        combo_flags=combo_flags,
        K_init=K_init, n_init=n_init, beta_init=beta_init,
        gamma_init=gamma_init, basal_init=basal_init, Vmax_init=Vmax_init,
        w_ratio_init=w_ratio_init, Vmax_ratio_init=Vmax_ratio_init,
        w0_ratio_init=w0_ratio_init, g0_ratio_init=g0_ratio_init,
        g_init=g_init, h_init=h_init,
        alpha_init=alpha_init, beta_s_init=beta_s_init,
    )


def global_hill_fit(W: np.ndarray, proteins: list[str],
                    all_X_raw: list[np.ndarray],
                    all_times_raw: list[np.ndarray],
                    fit_results: dict | None = None,
                    lr: float = 1e-3,
                    n_epochs: int = 2000,
                    patience: int = 200,
                    l1_lambda: float = 1e-4,
                    early_weight_tau: float = 0.0,
                    device: torch.device | None = None,
                    ) -> tuple[dict, HillODEFunc]:
    """Global network fitting: train all Hill ODE parameters simultaneously.

    Uses torchdiffeq.odeint for differentiable ODE integration and
    Adam optimiser for gradient descent on GPU/MPS.

    Parameters
    ----------
    W : (P, P) SINDy consensus weight matrix
    proteins : list of protein names
    all_X_raw : list of (T_c, P) observed trajectories, one per condition
    all_times_raw : list of (T_c,) time arrays
    fit_results : optional per-protein fit results for warm-start
    lr : Adam learning rate
    n_epochs : max training epochs
    patience : early stopping patience (epochs without improvement)
    l1_lambda : L1 regularisation on Vmax (sparsity)
    early_weight_tau : exponential decay τ for early time weighting
    device : torch device (auto-detected if None)

    Returns
    -------
    fit_results_dict : dict mapping protein → fit result (same format as
                       per-protein fitting for compatibility with export)
    ode_func : trained HillODEFunc module
    """
    if device is None:
        device = _select_device()
    print(f"  Device: {device}")

    P = len(proteins)
    C = len(all_X_raw)

    # ── initialise parameters ──
    init = _init_hill_ode_params(W, proteins, all_X_raw, fit_results)
    edge_indices = init["edge_indices"]
    edge_signs   = init["edge_signs"]
    E = len(edge_indices)

    # Count active components
    cf = init["combo_flags"]
    n_add   = cf[:, 0].sum()
    n_mult  = cf[:, 1].sum()
    n_ratio = cf[:, 2].sum()
    n_ss    = cf[:, 3].sum()
    n_params = 2 * E + 3 * P   # K, n per edge + gamma, basal, Vmax per protein
    if n_add:   n_params += E + P         # beta per edge + basal per protein
    if n_ratio: n_params += E + 3 * P     # w_ratio per edge + Vmax_r/w0/g0
    if n_ss:    n_params += 2 * E + 2 * P # g,h per edge + alpha,beta_s

    print(f"  Network: {P} proteins, {E} edges, ~{n_params} active parameters")
    print(f"  Combos: add={n_add} mult={n_mult} ratio={n_ratio} ss={n_ss}")

    # ── build ODE function ──
    ode_func = HillODEFunc(
        P, edge_indices, edge_signs,
        combo_flags=init["combo_flags"],
        K_init=init["K_init"], n_init=init["n_init"],
        beta_init=init["beta_init"],
        gamma_init=init["gamma_init"], basal_init=init["basal_init"],
        Vmax_init=init["Vmax_init"],
        w_ratio_init=init["w_ratio_init"],
        Vmax_ratio_init=init["Vmax_ratio_init"],
        w0_ratio_init=init["w0_ratio_init"],
        g0_ratio_init=init["g0_ratio_init"],
        g_init=init["g_init"], h_init=init["h_init"],
        alpha_init=init["alpha_init"], beta_s_init=init["beta_s_init"],
    )
    ode_func = ode_func.to(device)

    # ── prepare batched data (all conditions share the same time grid) ──
    # Stack all conditions:  X_all: (C, T, P),  t_shared: (T,)
    t_shared = torch.tensor(all_times_raw[0], dtype=torch.float32, device=device)
    T = len(all_times_raw[0])
    X_all = torch.stack([
        torch.tensor(X_c, dtype=torch.float32, device=device)
        for X_c in all_X_raw
    ])  # (C, T, P)

    # Time-based weights: (T,)
    if early_weight_tau > 0:
        w_t = torch.exp(-t_shared / early_weight_tau)
        w_t = torch.clamp(w_t, 0.1, 1.0)
        w_t = w_t / w_t.mean()
    else:
        w_t = torch.ones(T, device=device)

    # Initial conditions for all conditions: (C, P) → flatten to (C*P,)
    x0_batch = X_all[:, 0, :]  # (C, P)

    # Tell ODE func how many conditions we're batching
    ode_func._C = C

    # ── optimiser ──
    optimiser = torch.optim.Adam(ode_func.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=patience // 4,
        min_lr=1e-6)

    # ── training loop ──
    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    total_points = C * T * P

    print(f"\n{'─' * 70}")
    print(f"  Global Fitting: {n_epochs} max epochs, patience={patience}, "
          f"lr={lr}, L1={l1_lambda}")
    print(f"  Batched: {C} conditions × {T} time steps × {P} proteins"
          f" = {total_points} data points")
    print(f"{'─' * 70}")

    for epoch in range(n_epochs):
        optimiser.zero_grad()

        # Single batched odeint call: y0=(C*P,), result=(T, C*P)
        X_pred_flat = odeint(ode_func, x0_batch.reshape(-1), t_shared,
                             method="dopri5", rtol=1e-4, atol=1e-6,
                             options={"dtype": torch.float32})  # (T, C*P)
        X_pred = X_pred_flat.view(T, C, P)                      # (T, C, P)
        X_pred = X_pred.permute(1, 0, 2)                        # (C, T, P)

        # Weighted MSE across all conditions
        resid = (X_pred - X_all) ** 2                    # (C, T, P)
        weighted = resid * w_t.unsqueeze(0).unsqueeze(2) # broadcast (1,T,1)
        mse = weighted.sum() / total_points

        # L1 regularisation on Vmax
        l1 = l1_lambda * ode_func.Vmax.abs().mean()
        loss = mse + l1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=10.0)
        optimiser.step()
        scheduler.step(loss.item())

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in ode_func.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"  epoch {epoch:5d}  loss={loss_val:.6f}  "
                  f"mse={mse.item():.6f}  lr={lr_now:.2e}")

        if epochs_no_improve >= patience:
            print(f"  → Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # ── restore best ──
    if best_state is not None:
        ode_func.load_state_dict(best_state)
    ode_func.eval()
    print(f"\n  Best loss: {best_loss:.6f}")

    # ── extract results into per-protein format for export compatibility ──
    fit_out = _extract_global_results(ode_func, proteins, edge_indices,
                                       edge_signs, all_X_raw, all_times_raw,
                                       device)
    return fit_out, ode_func


@torch.no_grad()
def _extract_global_results(ode_func: HillODEFunc,
                             proteins: list[str],
                             edge_indices: list[tuple[int, int]],
                             edge_signs: list[int],
                             all_X_raw: list[np.ndarray],
                             all_times_raw: list[np.ndarray],
                             device: torch.device) -> dict:
    """Convert trained global ODE parameters → per-protein fit_results dict.

    This makes the output compatible with export_hill_params /
    export_fit_quality / plot functions.
    """
    P = ode_func.P
    K_np       = ode_func.K.cpu().numpy()
    n_np       = ode_func.n.cpu().numpy()
    gamma_np   = ode_func.gamma.cpu().numpy()
    basal_np   = ode_func.basal.cpu().numpy()
    Vmax_np    = ode_func.Vmax.cpu().numpy()
    beta_np    = ode_func.beta.cpu().numpy()
    wr_np      = ode_func.w_ratio.cpu().numpy()
    g_np       = ode_func.ss_g.cpu().detach().numpy()
    h_np       = ode_func.ss_h.cpu().detach().numpy()
    combo_np   = ode_func.combo_flags.cpu().numpy()   # (P, 4)

    # Per-protein ratio/ss params
    Vmax_r_np  = ode_func.Vmax_ratio.cpu().numpy()
    w0_np      = ode_func.w0.cpu().numpy()
    g0_np      = ode_func.g0.cpu().numpy()
    alpha_np   = ode_func.alpha_ss.cpu().numpy()
    beta_s_np  = ode_func.beta_ss.cpu().numpy()

    # Build per-target edge lists
    tgt_edges: dict[int, list[int]] = {}
    for eidx, (j, i) in enumerate(edge_indices):
        tgt_edges.setdefault(i, []).append(eidx)

    # Per-condition predictions for R²/RMSE  (batched)
    C = len(all_X_raw)
    t_shared = torch.tensor(all_times_raw[0], dtype=torch.float32, device=device)
    x0_batch = torch.stack([
        torch.tensor(X_c[0], dtype=torch.float32, device=device)
        for X_c in all_X_raw
    ])  # (C, P)
    ode_func._C = C
    X_pred_flat = odeint(ode_func, x0_batch.reshape(-1), t_shared,
                         method="dopri5", rtol=1e-4, atol=1e-6,
                         options={"dtype": torch.float32})  # (T, C*P)
    T = len(all_times_raw[0])
    X_pred_all = X_pred_flat.view(T, C, P).permute(1, 0, 2).cpu().numpy()  # (C, T, P)

    X_obs_cat  = np.vstack(all_X_raw)
    X_pred_cat = X_pred_all.reshape(C * T, P)

    _COMP_NAMES = ("add", "mult", "ratio", "ss")

    fit_results: dict[str, dict] = {}
    for i, pname in enumerate(proteins):
        obs  = X_obs_cat[:, i]
        pred = X_pred_cat[:, i]

        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rmse = np.sqrt(np.mean((obs - pred) ** 2))

        # Combo label for this protein
        active = [c for c, flag in zip(_COMP_NAMES, combo_np[i]) if flag]
        combo_label = "+".join(active) if active else "none"

        edge_params = []
        reg_indices = []
        reg_names   = []
        for eidx in tgt_edges.get(i, []):
            j, _ = edge_indices[eidx]
            sign_val = edge_signs[eidx]
            ep: dict = {
                "K":    float(K_np[eidx]),
                "n":    float(n_np[eidx]),
                "sign": "activation" if sign_val > 0 else "inhibition",
                "w":    float(Vmax_np[i]),
            }
            if combo_np[i, 0]:  # add
                ep["beta"] = float(beta_np[eidx])
            if combo_np[i, 2]:  # ratio
                ep["w_ratio"] = float(wr_np[eidx])
            if combo_np[i, 3]:  # ss
                ep["g"] = float(g_np[eidx])
                ep["h"] = float(h_np[eidx])
            edge_params.append(ep)
            reg_indices.append(j)
            reg_names.append(proteins[j])

        result: dict = {
            "edge_params": edge_params,
            "reg_indices": reg_indices,
            "reg_names":   reg_names,
            "gamma":  float(gamma_np[i]),
            "basal":  float(basal_np[i]),
            "vmax":   float(Vmax_np[i]),
            "r2":     float(r2),
            "rmse":   float(rmse),
            "model":  f"global_{combo_label}",
            "combo_flags": {c: bool(combo_np[i, k])
                            for k, c in enumerate(_COMP_NAMES)},
        }
        if combo_np[i, 2]:
            result["vmax_ratio"] = float(Vmax_r_np[i])
            result["w0"] = float(w0_np[i])
            result["gamma0"] = float(g0_np[i])
        if combo_np[i, 3]:
            result["alpha"] = float(alpha_np[i])
            result["beta_s"] = float(beta_s_np[i])

        fit_results[pname] = result

    return fit_results


def plot_global_fit(ode_func: HillODEFunc, proteins: list[str],
                    all_X_raw: list[np.ndarray],
                    all_times_raw: list[np.ndarray],
                    cond_labels: list[str],
                    output_path: str,
                    device: torch.device | None = None):
    """Plot observed vs predicted trajectories for all proteins & conditions."""
    if device is None:
        device = next(ode_func.parameters()).device
    P = len(proteins)
    C = len(all_X_raw)
    n_cols = min(6, P)
    n_rows = (P + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows),
                              squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, max(C, 10)))

    with torch.no_grad():
        # Batched prediction for all conditions at once
        t_shared = torch.tensor(all_times_raw[0], dtype=torch.float32, device=device)
        x0_batch = torch.stack([
            torch.tensor(X_c[0], dtype=torch.float32, device=device)
            for X_c in all_X_raw
        ])  # (C, P)
        ode_func._C = C
        X_pred_flat = odeint(ode_func, x0_batch.reshape(-1), t_shared,
                             method="dopri5", rtol=1e-4, atol=1e-6,
                             options={"dtype": torch.float32})  # (T, C*P)
        T = len(all_times_raw[0])
        X_pred_all = X_pred_flat.view(T, C, P).permute(1, 0, 2).cpu().numpy()  # (C, T, P)

        for c in range(C):
            X_obs = all_X_raw[c]
            t_obs = all_times_raw[c]
            X_pred = X_pred_all[c]

            for i in range(P):
                ax = axes[i // n_cols, i % n_cols]
                ax.plot(t_obs, X_obs[:, i], "o", color=colors[c % len(colors)],
                        ms=3, alpha=0.5)
                ax.plot(t_obs, X_pred[:, i], "-", color=colors[c % len(colors)],
                        alpha=0.7, lw=1.2)

    for i in range(P):
        ax = axes[i // n_cols, i % n_cols]
        dname = DISPLAY_NAMES_MULTILINE.get(proteins[i], proteins[i])
        ax.set_title(dname, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)

    for i in range(P, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle("Global Hill ODE Fit: Observed (dots) vs Predicted (lines)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, output_path)


# ── Global fitting pipeline ──

def run_global_hill_pipeline(data_dir: str, sindy_dir: str, output_dir: str,
                             source: str = "teacher",
                             lr: float = 1e-3,
                             n_epochs: int = 2000,
                             patience: int = 200,
                             l1_lambda: float = 1e-4,
                             early_weight_tau: float = 0.0,
                             warmstart: bool = True,
                             warmstart_restarts: int = 1,
                             **_ignored):
    """Run the global Hill ODE fitting pipeline on GPU/MPS.

    1. Load network + data
    2. Optionally run quick per-protein fit for warm-start
    3. Global fit with torchdiffeq on GPU/MPS
    4. Export results + plots

    Parameters
    ----------
    warmstart : if True, run a fast per-protein fit first and use those
                parameters as initial values for global optimisation.
    warmstart_restarts : n_restarts for the warmstart per-protein fit
    """
    os.makedirs(output_dir, exist_ok=True)
    device = _select_device()

    # ── load network ──
    consensus_path = os.path.join(sindy_dir, f"W_consensus_{source}.csv")
    if not os.path.exists(consensus_path):
        raise FileNotFoundError(f"Consensus matrix not found: {consensus_path}")

    W, proteins, edges = load_sindy_network(consensus_path)
    P = len(proteins)

    print(f"{'═' * 70}")
    print(f"  GLOBAL Hill ODE Fitting  —  source: {source}  —  device: {device}")
    print(f"  Network: {P} proteins, {len(edges)} directed edges")
    print(f"{'═' * 70}")

    # ── load data ──
    conditions = discover_conditions(data_dir, source)
    if not conditions:
        raise FileNotFoundError(f"No files for source={source!r} in {data_dir}")

    all_X_raw: list[np.ndarray] = []
    all_times_raw: list[np.ndarray] = []
    cond_labels: list[str] = []

    for label, filepath in conditions:
        times, X, pnames = load_ts_csv(filepath)
        assert pnames == proteins, f"Protein order mismatch in {label}"
        all_X_raw.append(X)
        all_times_raw.append(times)
        cond_labels.append(label)

    print(f"  Data: {len(conditions)} conditions × "
          f"{all_X_raw[0].shape[0]} time points × {P} proteins")

    # ── optional warm-start from per-protein fitting ──
    warmstart_results = None
    if warmstart:
        print(f"\n{'─' * 70}")
        print(f"  Warm-start: quick per-protein fit (n_restarts={warmstart_restarts}) ...")
        print(f"{'─' * 70}")
        warmstart_results = fit_all_proteins_auto(
            W, proteins, all_X_raw, all_times_raw,
            early_weight_tau=early_weight_tau,
            n_restarts=warmstart_restarts)

        r2_ws = [warmstart_results[p]["r2"] for p in proteins
                 if p in warmstart_results
                 and np.isfinite(warmstart_results[p].get("r2", np.nan))]
        if r2_ws:
            print(f"  Warm-start R² — median: {np.median(r2_ws):.4f}")

    # ── global fit ──
    print(f"\n{'█' * 70}")
    print(f"  GLOBAL NETWORK FITTING (BPTT + {device})")
    print(f"{'█' * 70}")

    fit_results, ode_func = global_hill_fit(
        W, proteins, all_X_raw, all_times_raw,
        fit_results=warmstart_results,
        lr=lr, n_epochs=n_epochs, patience=patience,
        l1_lambda=l1_lambda, early_weight_tau=early_weight_tau,
        device=device)

    # ── summary ──
    r2_vals = [fit_results[p]["r2"] for p in proteins
               if np.isfinite(fit_results[p].get("r2", np.nan))]
    if r2_vals:
        print(f"\n  Global R² — median: {np.median(r2_vals):.4f}  "
              f"mean: {np.mean(r2_vals):.4f}  "
              f"min: {np.min(r2_vals):.4f}  max: {np.max(r2_vals):.4f}")

    # ── export ──
    print(f"\n{'─' * 70}")
    print(f"  Exporting global results ...")
    print(f"{'─' * 70}")

    export_hill_params(fit_results, proteins,
                       os.path.join(output_dir, f"hill_params_{source}_global.csv"))
    export_fit_quality(fit_results, proteins,
                       os.path.join(output_dir, f"fit_quality_{source}_global.csv"))

    # ── save trained model ──
    model_path = os.path.join(output_dir, f"hill_ode_{source}_global.pt")
    torch.save({
        "state_dict": ode_func.state_dict(),
        "proteins": proteins,
        "edge_indices": [(j, i) for j, i in zip(ode_func.src.cpu().tolist(),
                                                  ode_func.tgt.cpu().tolist())],
        "edge_signs": ode_func.sign.cpu().tolist(),
    }, model_path)
    print(f"  ✓ Model checkpoint: {os.path.basename(model_path)}")

    # ── plots ──
    print(f"\n{'─' * 70}")
    print(f"  Generating plots ...")
    print(f"{'─' * 70}")

    plot_global_fit(ode_func, proteins, all_X_raw, all_times_raw,
                    cond_labels,
                    os.path.join(output_dir, f"global_fit_{source}.png"),
                    device=device)

    plot_fit_quality(fit_results, proteins,
                     os.path.join(output_dir, f"fit_quality_{source}_global.png"))

    plot_hill_params_heatmap(fit_results, proteins,
                             os.path.join(output_dir,
                                          f"hill_params_heatmap_{source}_global.png"))

    # ★ NEW: time-series vs equation dynamics comparison
    plot_timeseries_vs_dynamics(
        fit_results, proteins, all_X_raw, all_times_raw, cond_labels,
        os.path.join(output_dir,
                     f"timeseries_vs_dynamics_{source}_global.png"),
        n_proteins=min(9, len(proteins)))

    plot_dynamics_summary_heatmap(
        fit_results, proteins, all_X_raw, all_times_raw,
        os.path.join(output_dir,
                     f"dynamics_heatmap_{source}_global.png"))

    # ── network plot ──
    W_hill = np.zeros((P, P))
    for target in proteins:
        fit = fit_results.get(target)
        if fit is None or not fit.get("edge_params"):
            continue
        i = proteins.index(target)
        for k, ep in enumerate(fit["edge_params"]):
            j = fit["reg_indices"][k]
            sign = 1.0 if ep["sign"] == "activation" else -1.0
            W_hill[j, i] = sign * ep["w"]

    G = weight_matrix_to_digraph(W_hill, proteins)
    plot_sindy_network(G, f"Global Hill Network ({source})",
                       os.path.join(output_dir,
                                    f"hill_network_{source}_global.png"))

    return fit_results, ode_func, proteins


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hill function ODE fitting for phosphoproteomic "
                    "signalling networks.\n\n"
                    "Mode is determined automatically by source:\n"
                    "  teacher → de_auto  (best-RMSE auto-select, 15 ODE combos)\n"
                    "  student → dae_auto (fast→auto-select ODE, slow→algebraic)\n\n"
                    "Use --fit global for GPU/MPS-accelerated global network fitting\n"
                    "(all proteins jointly via BPTT + torchdiffeq).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hill_network.py --source teacher
  python hill_network.py --source student --fit global
  python hill_network.py --source both    --fit global
""",
    )
    parser.add_argument("--data_dir",   default="grn_ready_data")
    parser.add_argument("--sindy_dir",  default="results/sindy")
    parser.add_argument("--output_dir", default="results/hill")
    parser.add_argument("--source",     default="both",
                        choices=["teacher", "student", "both"])
    parser.add_argument("--fit",        default="global",
                        choices=["individual", "global"],
                        help="'individual' per-protein fitting (CPU); "
                             "'global' joint network fitting (GPU/MPS)")
    parser.add_argument("--early_weight_tau", type=float, default=0.0,
                        help="Exponential decay τ (minutes) for early-dynamics "
                             "weighting.  0 = uniform weights (default).")
    # Individual-mode options
    parser.add_argument("--dae_fast_frac", type=float, default=0.5,
                        help="Fraction of proteins treated as fast/ODE in DAE mode")
    parser.add_argument("--dae_alg_mode", default="multiplicative",
                        choices=["multiplicative", "ratio", "algebraic"],
                        help="Algebraic model for slow proteins in DAE mode")
    parser.add_argument("--n_restarts", type=int, default=3,
                        help="Number of random restarts for Hill ODE fitting")
    # Global-mode options
    parser.add_argument("--lr",         type=float, default=1e-2,
                        help="Learning rate for global fitting (Adam)")
    parser.add_argument("--n_epochs",   type=int, default=8000,
                        help="Max epochs for global fitting")
    parser.add_argument("--patience",   type=int, default=500,
                        help="Early stopping patience for global fitting")
    parser.add_argument("--l1_lambda",  type=float, default=1e-4,
                        help="L1 regularisation on Vmax for global fitting")
    parser.add_argument("--no_warmstart", action="store_true",
                        help="Skip per-protein warm-start for global fitting")

    args = parser.parse_args()

    if args.fit == "global":
        gkw = dict(lr=args.lr, n_epochs=args.n_epochs,
                    patience=args.patience, l1_lambda=args.l1_lambda,
                    early_weight_tau=args.early_weight_tau,
                    warmstart=not args.no_warmstart)
        if args.source == "both":
            for src in ("teacher", "student"):
                run_global_hill_pipeline(args.data_dir, args.sindy_dir,
                                         args.output_dir, source=src, **gkw)
        else:
            run_global_hill_pipeline(args.data_dir, args.sindy_dir,
                                     args.output_dir, source=args.source, **gkw)
    else:
        kw = dict(early_weight_tau=args.early_weight_tau,
                  dae_fast_frac=args.dae_fast_frac,
                  dae_alg_mode=args.dae_alg_mode,
                  n_restarts=args.n_restarts)
        if args.source == "both":
            compare_teacher_student_hill(args.data_dir, args.sindy_dir,
                                         args.output_dir, **kw)
        else:
            run_hill_pipeline(args.data_dir, args.sindy_dir, args.output_dir,
                              source=args.source, **kw)

    print("\n✅ Hill function fitting complete.")


if __name__ == "__main__":
    main()