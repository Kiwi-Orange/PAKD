"""
Analyze the slow manifold dynamics learned by the PAKD student model.

The student learns to:
- FREEZE dynamics in the fast regime (D≈0, r≈0) - preserves initial condition
- ACTIVATE dynamics in the slow regime (D→D_true, r→r_true)

This script discovers the transition time and effective coefficients.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import argparse
import os
from tqdm import tqdm

from models import MLP, ResidualMLP
from test_student import (load_model, get_device, create_time_features,
                          get_fisher_kpp_initial_condition)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

COMPACT_ERROR_VMAX = 2.0e-2


def _savefig_pair(fig, filepath_base):
    """Save matching PDF and PNG files for compact publication figures."""
    fig.savefig(f'{filepath_base}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{filepath_base}.png', dpi=300, bbox_inches='tight')


def _positive_time_min(time_points):
    """Return the first strictly positive time for log-scale axes."""
    positive_times = np.asarray(time_points)[np.asarray(time_points) > 0]
    return positive_times[0] if positive_times.size else time_points[0]


def _style_compact_2d_axes(ax, labelsize=9, grid=True):
    """Consistent compact 2D styling for NC subfigure exports."""
    ax.tick_params(axis='both', which='major', labelsize=labelsize,
                   width=1.3, length=4, direction='out')
    ax.tick_params(axis='both', which='minor', width=1.0, length=2.5,
                   direction='out')
    if grid:
        ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def _select_transition_slices(time_points, t_transition, n_fast=3, n_slow=3):
    """Select unique representative time indices around the fast/slow transition."""
    target = n_fast + n_slow
    if len(time_points) <= target:
        return np.arange(len(time_points), dtype=int)

    idx_trans = int(np.argmin(np.abs(time_points - t_transition)))
    fast_stop = max(0, idx_trans - 1)
    fast_indices = np.linspace(0, fast_stop, n_fast).round().astype(int)
    slow_indices = np.linspace(idx_trans, len(time_points) - 1, n_slow).round().astype(int)

    selected = []
    for idx in np.concatenate([fast_indices, slow_indices]):
        idx = int(np.clip(idx, 0, len(time_points) - 1))
        if idx not in selected:
            selected.append(idx)

    if len(selected) < target:
        fallback = np.linspace(0, len(time_points) - 1, target * 2).round().astype(int)
        for idx in fallback:
            idx = int(np.clip(idx, 0, len(time_points) - 1))
            if idx not in selected:
                selected.append(idx)
            if len(selected) == target:
                break

    return np.array(sorted(selected[:target]), dtype=int)


# ============================================================================
# Two-Regime Model: Fast (frozen) + Slow (active)
# ============================================================================
class TwoRegimeCoefficients:
    """
    Two-regime model for slow manifold reduction.
    
    Fast regime (t < t_transition): D ≈ 0, r ≈ 0 (frozen)
    Slow regime (t >= t_transition): D = D_slow, r = r_slow (active)
    """
    
    def __init__(self, t_transition, D_slow, r_slow, D_fast=0.0, r_fast=0.0):
        self.t_transition = t_transition
        self.D_fast = D_fast
        self.r_fast = r_fast
        self.D_slow = D_slow
        self.r_slow = r_slow
    
    def eval_D(self, t):
        t = np.atleast_1d(t)
        result = np.where(t < self.t_transition, self.D_fast, self.D_slow)
        return result.item() if result.size == 1 else result
    
    def eval_r(self, t):
        t = np.atleast_1d(t)
        result = np.where(t < self.t_transition, self.r_fast, self.r_slow)
        return result.item() if result.size == 1 else result
    
    def __repr__(self):
        return (f"TwoRegime(t*={self.t_transition:.4f}, "
                f"fast=[D={self.D_fast:.4f}, r={self.r_fast:.4f}], "
                f"slow=[D={self.D_slow:.4f}, r={self.r_slow:.4f}])")


def detect_transition_time(times, D_t, r_t, threshold_frac=0.5):
    """Detect transition time where coefficients activate."""
    n_final = max(1, len(times) // 5)
    D_final = np.median(D_t[-n_final:])
    r_final = np.median(r_t[-n_final:])
    
    D_threshold = threshold_frac * D_final
    r_threshold = threshold_frac * r_final
    
    idx_D = np.argmax(D_t > D_threshold) if np.any(D_t > D_threshold) else len(times) - 1
    idx_r = np.argmax(r_t > r_threshold) if np.any(r_t > r_threshold) else len(times) - 1
    
    return times[min(idx_D, idx_r)], min(idx_D, idx_r)


def fit_two_regime(times, D_t, r_t):
    """Fit two-regime model to extracted coefficients."""
    print("\nFitting two-regime model (fast/slow)...")
    
    valid = np.isfinite(D_t) & np.isfinite(r_t)
    t_valid, D_valid, r_valid = times[valid], D_t[valid], r_t[valid]
    
    t_trans, idx_trans = detect_transition_time(t_valid, D_valid, r_valid)
    idx_trans = max(2, min(idx_trans, len(t_valid) - 2))
    
    D_fast = np.median(D_valid[:idx_trans]) if idx_trans > 0 else 0.0
    r_fast = np.median(r_valid[:idx_trans]) if idx_trans > 0 else 0.0
    D_slow = np.median(D_valid[idx_trans:])
    r_slow = np.median(r_valid[idx_trans:])
    
    # Force fast regime to near-zero if small
    if D_fast < 0.3 * D_slow:
        D_fast = 0.0
    if r_fast < 0.3 * r_slow:
        r_fast = 0.0
    
    model = TwoRegimeCoefficients(t_trans, D_slow, r_slow, D_fast, r_fast)
    
    # R² scores
    D_pred = model.eval_D(t_valid)
    r_pred = model.eval_r(t_valid)
    ss_res_D = np.sum((D_valid - D_pred) ** 2)
    ss_tot_D = np.sum((D_valid - np.mean(D_valid)) ** 2)
    ss_res_r = np.sum((r_valid - r_pred) ** 2)
    ss_tot_r = np.sum((r_valid - np.mean(r_valid)) ** 2)
    
    r2_D = 1 - ss_res_D / ss_tot_D if ss_tot_D > 0 else 0
    r2_r = 1 - ss_res_r / ss_tot_r if ss_tot_r > 0 else 0
    
    print(f"  Transition time t* = {t_trans:.4f}")
    print(f"  Fast regime: D = {D_fast:.6f}, r = {r_fast:.6f}")
    print(f"  Slow regime: D = {D_slow:.6f}, r = {r_slow:.6f}")
    print(f"  R² scores: D = {r2_D:.4f}, r = {r2_r:.4f}")
    
    return model, {'D_r2': r2_D, 'r_r2': r2_r, 't_transition': t_trans}


def optimize_transition_time(times, D_t, r_t, traj_student, time_points, n_grid, h, u0_student):
    """Optimize transition time to minimize trajectory error."""
    print("\nOptimizing transition time...")
    
    valid = np.isfinite(D_t) & np.isfinite(r_t)
    D_valid, r_valid = D_t[valid], r_t[valid]
    
    n_final = max(1, len(D_valid) // 5)
    D_slow = np.median(D_valid[-n_final:])
    r_slow = np.median(r_valid[-n_final:])
    
    def objective(log_t_trans):
        t_trans = 10 ** log_t_trans
        model = TwoRegimeCoefficients(t_trans, D_slow, r_slow, 0.0, 0.0)
        traj_model = integrate_two_regime_ode(model, u0_student, time_points, n_grid, h)
        if traj_model is None:
            return np.inf
        return np.sqrt(np.mean((traj_student - traj_model) ** 2))
    
    log_t_min = np.log10(times.min())
    log_t_max = np.log10(times.max()) - 0.5
    result = minimize_scalar(objective, bounds=(log_t_min, log_t_max), method='bounded')
    
    t_trans_opt = 10 ** result.x
    print(f"  Optimal t* = {t_trans_opt:.4f} (RMSE = {result.fun:.4e})")
    
    return t_trans_opt


# ============================================================================
# Extract Pointwise Coefficients
# ============================================================================
def extract_coefficients_pointwise(U, dUdt, h, time_points):
    """Extract D(t) and r(t) for each time slice."""
    n_samples, n_grid = U.shape
    times = time_points[1:n_samples + 1]
    
    margin = max(2, n_grid // 10)
    interior = slice(margin, n_grid - margin)
    
    D_t = np.zeros(n_samples)
    r_t = np.zeros(n_samples)
    residuals = np.zeros(n_samples)
    
    print("\nExtracting pointwise D(t), r(t)...")
    
    for ti in tqdm(range(n_samples), desc="Time slices"):
        u = U[ti]
        dudt = dUdt[ti]
        
        laplacian = np.zeros(n_grid)
        laplacian[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / h**2
        reaction = u * (1 - u)
        
        A = np.column_stack([laplacian[interior], reaction[interior]])
        b = dudt[interior]
        AtA = A.T @ A + 1e-8 * np.eye(2)
        coeffs = np.linalg.solve(AtA, A.T @ b)
        
        D_t[ti] = max(coeffs[0], 0.0)
        r_t[ti] = max(coeffs[1], 0.0)
        residuals[ti] = np.linalg.norm(b - A @ coeffs) / (np.linalg.norm(b) + 1e-10)
    
    return times, D_t, r_t, residuals


# ============================================================================
# ODE Integration
# ============================================================================
def integrate_two_regime_ode(model, u0, time_points, n_grid, h):
    """Integrate ODE with two-regime coefficients."""
    def f(t, u):
        D_t = model.eval_D(t)
        r_t = model.eval_r(t)
        
        dudt = np.zeros(n_grid)
        dudt[1:-1] = (D_t * (u[2:] - 2*u[1:-1] + u[:-2]) / h**2 
                      + r_t * u[1:-1] * (1 - u[1:-1]))
        dudt[0] = D_t * (u[1] - 2*u[0]) / h**2 + r_t * u[0] * (1 - u[0])
        dudt[-1] = D_t * (-2*u[-1] + u[-2]) / h**2 + r_t * u[-1] * (1 - u[-1])
        return dudt
    
    try:
        sol = solve_ivp(f, (time_points[0], time_points[-1]), u0,
                       t_eval=time_points, method='BDF', max_step=0.01)
        return sol.y.T if sol.success else None
    except Exception as e:
        print(f"Warning: ODE integration failed: {e}")
        return None


# ============================================================================
# Dynamics Analyzer
# ============================================================================
class DynamicsAnalyzer:
    """Analyze learned dynamics from student model."""
    
    def __init__(self, model, X_scaler, y_scaler, n_grid, device):
        self.model = model
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.n_grid = n_grid
        self.device = device
        self.h = 1.0 / (n_grid + 1)
    
    def predict(self, t, u0):
        """Get model prediction at time t."""
        X = np.zeros((1, self.n_grid + 1), dtype=np.float32)
        X[0, 0] = t
        X[0, 1:] = u0
        
        time_feat = create_time_features(X[:, 0])
        X_aug = np.column_stack([time_feat, X[:, 1:]])
        X_norm = self.X_scaler.transform(X_aug)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy()
        
        if self.y_scaler is not None:
            pred = self.y_scaler.inverse_transform(pred)
        
        return np.clip(pred[0], 0.0, 1.0)
    
    def compute_dudt(self, t, u0, dt=1e-3):
        """Compute du/dt via central differences."""
        dt_actual = min(dt, t * 0.1) if t > 0 else dt
        u_plus = self.predict(t + dt_actual, u0)
        u_minus = self.predict(max(t - dt_actual, 1e-8), u0)
        return (u_plus - u_minus) / (2 * dt_actual)
    
    def collect_data(self, u0, time_points):
        """Collect (u, du/dt) data pairs."""
        print("\nCollecting dynamics data from student model...")
        U_list, dUdt_list = [], []
        
        for t in tqdm(time_points[1:], desc="Time points"):
            U_list.append(self.predict(t, u0))
            dUdt_list.append(self.compute_dudt(t, u0))
        
        return np.array(U_list), np.array(dUdt_list)
    
    def generate_trajectory(self, u0, time_points):
        """Generate trajectory from student model."""
        return np.array([self.predict(t, u0) for t in tqdm(time_points, desc="Student")])
    
    def get_student_initial_condition(self, u0, t_min):
        """Get the student's effective initial condition at t_min."""
        return self.predict(t_min, u0)


# ============================================================================
# Visualization
# ============================================================================
def save_figure(fig, filepath_base):
    """Save figure in PNG and PDF formats."""
    for ext in ['png', 'pdf']:
        fig.savefig(f'{filepath_base}.{ext}', dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath_base}.png/pdf")


def plot_initial_condition_comparison(x_grid, u0_original, u0_student, t_min, output_dir):
    """Plot comparison of original IC vs student's effective IC."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(x_grid, u0_original, 'g--', linewidth=2.5, label='Original IC (step function)')
    ax.plot(x_grid, u0_student, 'b-', linewidth=2.5, label=f'Student IC at t={t_min:.4f}')
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$u$', fontsize=14)
    ax.set_title('Initial Condition: Original vs Student', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    
    ax.text(0.5, 0.02, 
            f'The two-regime ODE starts from Student IC (blue), not Original IC (green)',
            ha='center', transform=ax.transAxes, fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/initial_condition_comparison')
    plt.close()

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(x_grid, u0_original, color='#33A02C', linestyle='--',
            linewidth=2.0, label='Original IC')
    ax.plot(x_grid, u0_student, color='#1F78B4', linestyle='-',
            linewidth=2.2, label=rf'Student IC, $t={t_min:.1e}$')

    ax.set_xlabel(r'$x$', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'$u$', fontsize=11, fontweight='bold')
    ax.set_title('Initial condition', fontsize=11, fontweight='bold', pad=5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(frameon=True, fancybox=False, shadow=False,
              loc='best', fontsize=8, framealpha=0.9)
    _style_compact_2d_axes(ax, labelsize=9)

    fig.tight_layout(pad=0.8)
    _savefig_pair(fig, f'{output_dir}/initial_condition_comparison_compact')
    plt.close(fig)


def plot_coefficients(times, D_t, r_t, model, output_dir):
    """Plot extracted coefficients with two-regime fit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    t_dense = np.logspace(np.log10(times.min()), np.log10(times.max()), 500)
    
    # Plot D(t)
    ax = axes[0]
    ax.semilogx(times, D_t, 'b.', alpha=0.4, markersize=4, label='Extracted')
    ax.semilogx(t_dense, model.eval_D(t_dense), 'b-', linewidth=2.5, label='Two-regime fit')
    ax.axhline(0.01, color='r', linestyle='--', linewidth=2, label='True D=0.01')
    ax.axvline(model.t_transition, color='k', linestyle=':', linewidth=2, 
               label=f'$t^*$={model.t_transition:.4f}')
    ax.axvspan(times.min(), model.t_transition, alpha=0.1, color='blue', label='Fast (frozen)')
    ax.axvspan(model.t_transition, times.max(), alpha=0.1, color='green', label='Slow (active)')
    ax.set_xlabel('Time $t$', fontsize=14)
    ax.set_ylabel('$D(t)$', fontsize=14)
    ax.set_title('Diffusion Coefficient', fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot r(t)
    ax = axes[1]
    ax.semilogx(times, r_t, 'g.', alpha=0.4, markersize=4, label='Extracted')
    ax.semilogx(t_dense, model.eval_r(t_dense), 'g-', linewidth=2.5, label='Two-regime fit')
    ax.axhline(1.0, color='r', linestyle='--', linewidth=2, label='True r=1.0')
    ax.axvline(model.t_transition, color='k', linestyle=':', linewidth=2,
               label=f'$t^*$={model.t_transition:.4f}')
    ax.axvspan(times.min(), model.t_transition, alpha=0.1, color='blue')
    ax.axvspan(model.t_transition, times.max(), alpha=0.1, color='green')
    ax.set_xlabel('Time $t$', fontsize=14)
    ax.set_ylabel('$r(t)$', fontsize=14)
    ax.set_title('Reaction Coefficient', fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/two_regime_coefficients')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharex=True)
    compact_specs = [
        (axes[0], D_t, model.eval_D(t_dense), 0.01, '#1F78B4',
         r'$D(t)$', 'Diffusion', 'True $D=0.01$'),
        (axes[1], r_t, model.eval_r(t_dense), 1.0, '#33A02C',
         r'$r(t)$', 'Reaction', 'True $r=1$'),
    ]

    for ax, extracted, fitted, true_value, color, ylabel, title, true_label in compact_specs:
        ax.axvspan(times.min(), model.t_transition, alpha=0.10, color='#5167d6')
        ax.axvspan(model.t_transition, times.max(), alpha=0.10, color='#5fac6f')
        ax.semilogx(times, extracted, '.', color=color, alpha=0.35,
                    markersize=3.0, label='Extracted')
        ax.semilogx(t_dense, fitted, color=color, linewidth=2.1,
                    label='Two-regime')
        ax.axhline(true_value, color='#E31A1C', linestyle='--',
                   linewidth=1.7, label=true_label)
        ax.axvline(model.t_transition, color='black', linestyle=':',
                   linewidth=1.8, label=rf'$t^*={model.t_transition:.2g}$')
        ax.set_xlim(times.min(), times.max())
        ax.set_xlabel(r'$t$', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=5)
        ax.legend(frameon=True, fancybox=False, shadow=False,
                  loc='best', fontsize=7.2, framealpha=0.9)
        _style_compact_2d_axes(ax, labelsize=8.5)

    fig.tight_layout(pad=0.8, w_pad=0.7)
    _savefig_pair(fig, f'{output_dir}/two_regime_coefficients_compact')
    plt.close(fig)


def plot_trajectories(x_grid, time_points, traj_student, traj_model, t_transition, output_dir):
    """Plot trajectory comparison heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ['Student NN', 'Two-Regime ODE', '|Student - Two-Regime|']
    data = [traj_student, traj_model, np.abs(traj_student - traj_model)]
    cmaps = ['viridis', 'viridis', 'Reds']
    
    for ax, title, d, cmap in zip(axes, titles, data, cmaps):
        vmax = 1 if '|' not in title else np.percentile(d, 99)
        im = ax.pcolormesh(x_grid, time_points, d, cmap=cmap, vmin=0, vmax=vmax, shading='nearest')
        ax.axhline(t_transition, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.set_yscale('log')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_title(title, fontweight='bold')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/trajectories_heatmap')
    plt.close()

    fig = plt.figure(figsize=(11.0, 3.1))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(
        1, 7,
        width_ratios=[1.0, 1.0, 0.035, 0.10, 1.0, 0.035, 0.02],
        wspace=0.08
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 4]),
    ]
    cax_u = fig.add_subplot(gs[0, 2])
    cax_e = fig.add_subplot(gs[0, 5])

    compact_titles = ['Student NN', 'Two-Regime ODE', 'Abs. error']
    compact_data = [
        np.clip(traj_student, 0.0, 1.0),
        np.clip(traj_model, 0.0, 1.0),
        np.clip(np.abs(traj_student - traj_model), 0.0, COMPACT_ERROR_VMAX),
    ]
    compact_cmaps = ['RdBu_r', 'RdBu_r', 'YlOrRd']
    compact_vmax = [1.0, 1.0, COMPACT_ERROR_VMAX]

    ims = []
    for idx, (ax, title, d, cmap, vmax) in enumerate(
            zip(axes, compact_titles, compact_data, compact_cmaps, compact_vmax)):
        im = ax.pcolormesh(
            x_grid, time_points, d, cmap=cmap, vmin=0.0, vmax=vmax,
            shading='nearest'
        )
        ims.append(im)
        ax.axhline(t_transition, color='white', linestyle='--',
                   linewidth=1.2, alpha=0.9)
        ax.set_yscale('log')
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(_positive_time_min(time_points), time_points[-1])
        ax.set_xlabel(r'$x$', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel(r'$t$', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=5)
        _style_compact_2d_axes(ax, labelsize=8.5, grid=False)

    cb_u = fig.colorbar(ims[0], cax=cax_u)
    cax_u.set_title(r'$u$', fontsize=9, fontweight='bold', pad=3)
    cb_u.ax.tick_params(labelsize=8.5, width=1.0, length=3)
    cb_u.outline.set_linewidth(1.0)

    cb_e = fig.colorbar(ims[2], cax=cax_e)
    cb_e.set_label(r'$|u_{NN}-u_{ODE}|$', fontsize=9,
                   fontweight='bold', labelpad=3)
    cb_e.ax.tick_params(labelsize=8.5, width=1.0, length=3)
    cb_e.outline.set_linewidth(1.0)

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.86)
    _savefig_pair(fig, f'{output_dir}/trajectories_heatmap_compact')
    plt.close(fig)


def plot_time_slices(x_grid, time_points, traj_student, traj_model, t_transition, output_dir):
    """Plot solution profiles at selected times."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    idx_trans = np.argmin(np.abs(time_points - t_transition))
    fast_indices = np.linspace(0, max(0, idx_trans-1), 4).astype(int)
    slow_indices = np.linspace(idx_trans, len(time_points)-1, 4).astype(int)
    time_indices = np.concatenate([fast_indices, slow_indices])
    
    for ax, ti in zip(axes.flatten(), time_indices):
        t = time_points[ti]
        regime = "FAST" if t < t_transition else "SLOW"
        
        ax.plot(x_grid, traj_student[ti], 'b-', linewidth=2.5, label='Student')
        ax.plot(x_grid, traj_model[ti], 'r--', linewidth=2, label='Two-Regime')
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u$')
        ax.set_title(f't = {t:.4f} [{regime}]', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/time_slices')
    plt.close()

    compact_indices = _select_transition_slices(time_points, t_transition, n_fast=3, n_slow=3)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0), sharex=True, sharey=True)

    for plot_idx, (ax, ti) in enumerate(zip(axes.flatten(), compact_indices)):
        t = time_points[ti]
        regime = 'Fast' if t < t_transition else 'Slow'

        ax.plot(x_grid, traj_student[ti], color='#1F78B4', linewidth=2.0,
                label='Student')
        ax.plot(x_grid, traj_model[ti], color='#E31A1C', linestyle='--',
                linewidth=1.9, label='Two-Regime')

        ax.set_title(rf'$t={t:.2g}$  {regime}', fontsize=10,
                     fontweight='bold', pad=4)
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(-0.05, 1.08)
        if plot_idx >= 3:
            ax.set_xlabel(r'$x$', fontsize=10.5, fontweight='bold')
        if plot_idx % 3 == 0:
            ax.set_ylabel(r'$u$', fontsize=10.5, fontweight='bold')
        if plot_idx == 0:
            ax.legend(frameon=True, fancybox=False, shadow=False,
                      loc='best', fontsize=7.5, framealpha=0.9)
        _style_compact_2d_axes(ax, labelsize=8.5)

    fig.tight_layout(pad=0.7, w_pad=0.35, h_pad=0.55)
    _savefig_pair(fig, f'{output_dir}/time_slices_compact')
    plt.close(fig)


def plot_regime_diagram(model, times, output_dir):
    """Plot regime diagram showing fast/slow separation."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    t_trans = model.t_transition
    
    ax.axvspan(times.min(), t_trans, alpha=0.3, color='blue')
    ax.text(np.sqrt(times.min() * t_trans), 0.7, 
            f'FAST REGIME\n(Frozen)\nD ≈ {model.D_fast:.4f}\nr ≈ {model.r_fast:.4f}',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.axvspan(t_trans, times.max(), alpha=0.3, color='green')
    ax.text(np.sqrt(t_trans * times.max()), 0.7,
            f'SLOW REGIME\n(Active)\nD = {model.D_slow:.4f}\nr = {model.r_slow:.4f}',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.axvline(t_trans, color='red', linewidth=3, linestyle='-', label=f'$t^*$ = {t_trans:.4f}')
    
    ax.set_xscale('log')
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time $t$', fontsize=14)
    ax.set_title('Slow Manifold Reduction: Two-Regime Model', fontweight='bold', fontsize=14)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=12)
    
    ax.text(0.5, -0.15, 
            'Student FREEZES dynamics in fast regime, then ACTIVATES in slow regime',
            ha='center', transform=ax.transAxes, fontsize=11, style='italic')
    
    plt.tight_layout()
    save_figure(fig, f'{output_dir}/regime_diagram')
    plt.close()

    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    ax.axvspan(times.min(), t_trans, alpha=0.16, color='#5167d6')
    ax.axvspan(t_trans, times.max(), alpha=0.16, color='#5fac6f')
    ax.axvline(t_trans, color='black', linewidth=1.8, linestyle=':',
               label=rf'$t^*={t_trans:.2g}$')

    fast_x = np.sqrt(times.min() * t_trans)
    slow_x = np.sqrt(t_trans * times.max())
    ax.text(fast_x, 0.55,
            rf'Fast: $D={model.D_fast:.2g}$, $r={model.r_fast:.2g}$',
            ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(slow_x, 0.55,
            rf'Slow: $D={model.D_slow:.2g}$, $r={model.r_slow:.2g}$',
            ha='center', va='center', fontsize=8.5, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel(r'$t$', fontsize=11, fontweight='bold')
    ax.set_title('Two-regime dynamics', fontsize=11, fontweight='bold', pad=5)
    ax.legend(frameon=True, fancybox=False, shadow=False,
              loc='upper right', fontsize=7.5, framealpha=0.9)
    _style_compact_2d_axes(ax, labelsize=8.5, grid=False)

    fig.tight_layout(pad=0.6)
    _savefig_pair(fig, f'{output_dir}/regime_diagram_compact')
    plt.close(fig)


def print_summary(model, metrics, rmse_model, n_grid, u0_original, u0_student, t_min):
    """Print analysis summary."""
    h = 1.0 / (n_grid + 1)
    ic_diff = np.sqrt(np.mean((u0_original - u0_student) ** 2))
    
    print("\n" + "=" * 70)
    print("SLOW MANIFOLD ANALYSIS: TWO-REGIME MODEL")
    print("=" * 70)
    
    print(f"\n  INITIAL CONDITION:")
    print(f"      Original IC: step function")
    print(f"      Student IC at t={t_min:.4f}: smoothed by network")
    print(f"      RMSE(Original, Student IC): {ic_diff:.4e}")
    print(f"      → Two-Regime ODE starts from Student IC")
    
    print(f"\n  DISCOVERED DYNAMICS:")
    print(f"  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  FAST REGIME (t < {model.t_transition:.4f}):                                    │")
    print(f"  │      ∂u/∂t ≈ 0  (dynamics frozen)                               │")
    print(f"  │      D = {model.D_fast:.6f}, r = {model.r_fast:.6f}                            │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  SLOW REGIME (t ≥ {model.t_transition:.4f}):                                    │")
    print(f"  │      ∂u/∂t = D·Δu + r·u(1-u)                                    │")
    print(f"  │      D = {model.D_slow:.6f}, r = {model.r_slow:.6f}                            │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")
    
    print(f"\n  TRANSITION TIME: t* = {model.t_transition:.4f}")
    
    print(f"\n  COMPARISON TO TRUE KPP (D=0.01, r=1.0):")
    print(f"      D_slow / D_true = {model.D_slow/0.01:.2%}")
    print(f"      r_slow / r_true = {model.r_slow/1.0:.2%}")
    
    print(f"\n  FIT QUALITY:")
    print(f"      D(t) R² = {metrics['D_r2']:.4f}")
    print(f"      r(t) R² = {metrics['r_r2']:.4f}")
    
    print(f"\n  TRAJECTORY RMSE (Student vs Two-Regime): {rmse_model:.4e}")
    
    print(f"\n  PHYSICAL INTERPRETATION:")
    print(f"      • Fast time scale: τ_fast ~ h²/D ≈ {h**2/0.01:.2e}")
    print(f"      • Student maps original IC → smoothed IC at t_min")
    print(f"      • Freezes dynamics until t* = {model.t_transition:.4f}")
    print(f"      • Slow manifold dynamics activate after t*")
    
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Analyze Student Slow Manifold Dynamics')
    parser.add_argument('--student_model', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--n_time_points', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='results/student_dynamics')
    parser.add_argument('--ic_type', type=str, default='step',
                        choices=['step', 'gaussian', 'sine'])
    parser.add_argument('--optimize_transition', action='store_true',
                        help='Optimize transition time to minimize trajectory error')
    args = parser.parse_args()
    
    # Setup
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    print(f"\nLoading: {args.student_model}")
    model, X_scaler, y_scaler, model_type, n_grid = load_model(args.student_model, device, is_student=True)
    
    analyzer = DynamicsAnalyzer(model, X_scaler, y_scaler, n_grid, device)
    x_grid = np.linspace(0, 1, n_grid + 2)[1:-1]
    u0_original = get_fisher_kpp_initial_condition(x_grid, args.ic_type)
    time_points = np.logspace(-3, 1, args.n_time_points)
    t_min = time_points[0]
    h = 1.0 / (n_grid + 1)
    
    # Get student's effective initial condition
    u0_student = analyzer.get_student_initial_condition(u0_original, t_min)
    print(f"\n  Student IC at t={t_min:.4f} differs from original by RMSE = {np.sqrt(np.mean((u0_original - u0_student)**2)):.4e}")
    
    output_dir = os.path.join(args.output_dir, f"{model_type}_{args.ic_type}_two_regime")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract coefficients
    U, dUdt = analyzer.collect_data(u0_original, time_points)
    times, D_t, r_t, residuals = extract_coefficients_pointwise(U, dUdt, h, time_points)
    
    # Fit two-regime model
    regime_model, metrics = fit_two_regime(times, D_t, r_t)
    
    # Generate student trajectory
    print("\nGenerating trajectories...")
    traj_student = analyzer.generate_trajectory(u0_original, time_points)
    
    # Optionally optimize transition time
    if args.optimize_transition:
        t_trans_opt = optimize_transition_time(times, D_t, r_t, traj_student, time_points, 
                                                n_grid, h, u0_student)
        n_final = max(1, len(times) // 5)
        D_slow = np.median(D_t[-n_final:])
        r_slow = np.median(r_t[-n_final:])
        regime_model = TwoRegimeCoefficients(t_trans_opt, D_slow, r_slow, 0.0, 0.0)
        metrics['t_transition'] = t_trans_opt
    
    # Integrate ODE starting from student's IC
    print("  Integrating Two-Regime ODE from Student IC...")
    traj_model = integrate_two_regime_ode(regime_model, u0_student, time_points, n_grid, h)
    
    if traj_model is None:
        traj_model = np.zeros_like(traj_student)
    
    # Metrics
    rmse_model = np.sqrt(np.mean((traj_student - traj_model) ** 2))
    
    # Visualize
    plot_initial_condition_comparison(x_grid, u0_original, u0_student, t_min, output_dir)
    plot_coefficients(times, D_t, r_t, regime_model, output_dir)
    plot_trajectories(x_grid, time_points, traj_student, traj_model, regime_model.t_transition, output_dir)
    plot_time_slices(x_grid, time_points, traj_student, traj_model, regime_model.t_transition, output_dir)
    plot_regime_diagram(regime_model, times, output_dir)
    
    # Summary
    print_summary(regime_model, metrics, rmse_model, n_grid, u0_original, u0_student, t_min)
    
    # Save results
    np.savez(f'{output_dir}/discovered_dynamics.npz',
             times=times, D_t=D_t, r_t=r_t, residuals=residuals,
             t_transition=regime_model.t_transition,
             D_fast=regime_model.D_fast, r_fast=regime_model.r_fast,
             D_slow=regime_model.D_slow, r_slow=regime_model.r_slow,
             u0_original=u0_original, u0_student=u0_student,
             traj_student=traj_student, traj_model=traj_model,
             rmse_model=rmse_model,
             x_grid=x_grid, time_points=time_points)
    
    print(f"\n✓ Results saved to: {output_dir}/")
    print("Compact NC subfigure outputs include:")
    print("  - initial_condition_comparison_compact.pdf/.png")
    print("  - two_regime_coefficients_compact.pdf/.png")
    print("  - trajectories_heatmap_compact.pdf/.png")
    print("  - time_slices_compact.pdf/.png")
    print("  - regime_diagram_compact.pdf/.png")


if __name__ == "__main__":
    main()
