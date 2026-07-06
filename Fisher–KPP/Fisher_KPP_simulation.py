import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import diags, csr_matrix, issparse
from scipy.sparse.linalg import splu
import os
import argparse
from tqdm import tqdm
import warnings
import matplotlib as mpl
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

# Suppress overflow warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Publication-ready matplotlib settings
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
})

@dataclass
class SimulationConfig:
    """Configuration for Fisher-KPP simulation."""
    # Spatial discretization
    n_interior: int = 100  # Number of interior grid points
    x_left: float = 0.0
    x_right: float = 1.0
    
    # Diffusion coefficient (small = stiff problem)
    epsilon: float = 0.01
    
    # Boundary condition type: 'dirichlet' or 'neumann'
    bc_type: str = 'dirichlet'
    
    # Time parameters
    t_span: Tuple[float, float] = (0.0, 2.0)
    n_time_points: int = 500
    
    # Solver settings
    atol: float = 1e-10
    rtol: float = 1e-8
    solver_method: str = 'BDF'  # BDF is good for stiff problems
    
    @property
    def h(self) -> float:
        """Grid spacing."""
        return (self.x_right - self.x_left) / (self.n_interior + 1)
    
    @property
    def x_interior(self) -> np.ndarray:
        """Interior grid points."""
        return np.linspace(self.x_left + self.h, self.x_right - self.h, self.n_interior)
    
    @property
    def x_full(self) -> np.ndarray:
        """Full grid including boundaries."""
        return np.linspace(self.x_left, self.x_right, self.n_interior + 2)
    
    @property
    def stiffness_ratio(self) -> float:
        """Estimate of stiffness: ratio of fast to slow timescales."""
        # Fast mode: O(epsilon/h^2), Slow mode: O(1)
        return self.epsilon / (self.h ** 2)


def build_laplacian_matrix(config: SimulationConfig) -> csr_matrix:
    """
    Build the discrete Laplacian matrix for second-order central differences.
    
    Returns a SPARSE matrix in CSR format for efficient matrix-vector products.
    
    For Dirichlet BC: Standard tridiagonal [-1, 2, -1] / h^2
    For Neumann BC: Modified at boundaries to enforce zero flux
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
        
    Returns
    -------
    csr_matrix
        Sparse discrete Laplacian matrix (n_interior × n_interior)
    """
    n = config.n_interior
    h = config.h
    h2 = h ** 2
    
    # Build diagonals for tridiagonal Laplacian: (1/h^2) * [-1, 2, -1]
    main_diag = 2.0 * np.ones(n) / h2
    off_diag = -np.ones(n - 1) / h2
    
    # Modify for Neumann BC if needed
    if config.bc_type == 'neumann':
        # Use ghost points: u_{-1} = u_1 and u_{n+1} = u_{n-1}
        # This modifies the corner entries
        main_diag[0] = 1.0 / h2
        main_diag[-1] = 1.0 / h2
    
    # Create sparse matrix directly in CSR format
    A = diags(
        [off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format='csr'
    )
    
    return A


def fisher_kpp_rhs(t: float, u: np.ndarray, 
                   A: csr_matrix, epsilon: float) -> np.ndarray:
    """
    Right-hand side of the Fisher-KPP equation (method of lines).
    
    du/dt = -ε * A @ u + u * (1 - u)
    
    where A is the discrete Laplacian matrix (sparse).
    
    Parameters
    ----------
    t : float
        Current time (not used, but required by solver)
    u : np.ndarray
        Current solution at interior grid points
    A : csr_matrix
        Sparse discrete Laplacian matrix
    epsilon : float
        Diffusion coefficient
        
    Returns
    -------
    np.ndarray
        Time derivative du/dt
    """
    # Ensure non-negative (numerical stability)
    u = np.maximum(u, 0.0)
    
    # Diffusion term: -ε * A @ u (sparse matrix-vector product, O(n))
    diffusion = -epsilon * A.dot(u)
    
    # Reaction term: u * (1 - u) (logistic growth)
    reaction = u * (1 - u)
    
    return diffusion + reaction


def fisher_kpp_jacobian(t: float, u: np.ndarray,
                        A: csr_matrix, epsilon: float) -> csr_matrix:
    """
    Analytical Jacobian of the Fisher-KPP equation.
    
    J(u) = -ε * A + diag(1 - 2u)
    
    The Jacobian is sparse (tridiagonal from diffusion + diagonal from reaction).
    
    Parameters
    ----------
    t : float
        Current time (not used, but required by solver)
    u : np.ndarray
        Current solution at interior grid points
    A : csr_matrix
        Sparse discrete Laplacian matrix
    epsilon : float
        Diffusion coefficient
        
    Returns
    -------
    csr_matrix
        Sparse Jacobian matrix
    """
    n = len(u)
    
    # Reaction term Jacobian: d/du [u(1-u)] = 1 - 2u
    reaction_jacobian = diags(1.0 - 2.0 * u, offsets=0, shape=(n, n), format='csr')
    
    # Full Jacobian: J = -ε*A + diag(1-2u)
    J = -epsilon * A + reaction_jacobian
    
    return J


def solve_fisher_kpp(config: SimulationConfig, 
                     u0: np.ndarray,
                     t_eval: Optional[np.ndarray] = None,
                     use_jacobian: bool = True) -> Dict:
    """
    Solve the Fisher-KPP equation using method of lines.
    
    Uses sparse Laplacian and analytical Jacobian for efficiency.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    u0 : np.ndarray
        Initial condition at interior grid points
    t_eval : np.ndarray, optional
        Time points for output
    use_jacobian : bool
        Whether to provide analytical Jacobian (recommended for stiff problems)
        
    Returns
    -------
    dict
        Solution dictionary with 't', 'u', 'x', and 'success' keys
    """
    # Build sparse Laplacian matrix
    A = build_laplacian_matrix(config)
    
    # Default time evaluation points
    if t_eval is None:
        t_eval = np.linspace(config.t_span[0], config.t_span[1], config.n_time_points)
    
    # Define RHS and Jacobian functions with closure over A and epsilon
    def rhs(t, u):
        return fisher_kpp_rhs(t, u, A, config.epsilon)
    
    def jac(t, u):
        return fisher_kpp_jacobian(t, u, A, config.epsilon)
    
    try:
        # Solve with or without analytical Jacobian
        solve_kwargs = {
            'fun': rhs,
            't_span': config.t_span,
            'y0': u0,
            'method': config.solver_method,
            't_eval': t_eval,
            'atol': config.atol,
            'rtol': config.rtol,
        }
        
        # Add Jacobian for implicit solvers (BDF, Radau)
        if use_jacobian and config.solver_method in ['BDF', 'Radau']:
            solve_kwargs['jac'] = jac
            # Specify sparsity structure for efficiency
            solve_kwargs['jac_sparsity'] = A  # Same sparsity pattern as Laplacian
        
        solution = solve_ivp(**solve_kwargs)
        
        return {
            't': solution.t,
            'u': solution.y,  # Shape: (n_interior, n_time_points)
            'x': config.x_interior,
            'success': solution.success,
            'nfev': solution.nfev,
            'njev': getattr(solution, 'njev', 0),
            'nlu': getattr(solution, 'nlu', 0),
        }
        
    except Exception as e:
        print(f"Error solving Fisher-KPP: {str(e)}")
        return {
            't': np.array([]),
            'u': np.empty((config.n_interior, 0)),  # Proper 2D array shape
            'x': config.x_interior,
            'success': False,
            'nfev': 0,
            'njev': 0,
            'nlu': 0,
        }


def get_initial_condition(x: np.ndarray, ic_type: str = 'gaussian', 
                          params: Optional[Dict] = None) -> np.ndarray:
    """
    Generate initial condition for Fisher-KPP equation.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial grid points
    ic_type : str
        Type of initial condition:
        - 'gaussian': Gaussian bump
        - 'step': Step function (traveling wave initial)
        - 'sine': Sine wave
        - 'random': Random perturbation
    params : dict, optional
        Parameters for the initial condition
        
    Returns
    -------
    np.ndarray
        Initial concentration profile (clipped to [0, 1])
    """
    if params is None:
        params = {}
    
    if ic_type == 'gaussian':
        center = params.get('center', 0.5)
        width = params.get('width', 0.1)
        amplitude = params.get('amplitude', 0.8)
        u0 = amplitude * np.exp(-((x - center) / width) ** 2)
        
    elif ic_type == 'step':
        # Step function for traveling wave
        x_step = params.get('x_step', 0.3)
        # Smooth transition using tanh
        transition_width = params.get('transition_width', 0.05)
        u0 = 0.5 * (1 - np.tanh((x - x_step) / transition_width))
        
    elif ic_type == 'sine':
        n_modes = params.get('n_modes', 1)
        u0 = 0.5 * (1 + 0.5 * np.sin(n_modes * np.pi * x))
        
    elif ic_type == 'random':
        np.random.seed(params.get('seed', 42))
        base = params.get('base', 0.3)
        noise_level = params.get('noise_level', 0.2)
        u0 = base + noise_level * np.random.rand(len(x))
        
    elif ic_type == 'double_gaussian':
        # Two Gaussian bumps
        u0 = (0.6 * np.exp(-((x - 0.3) / 0.08) ** 2) + 
              0.4 * np.exp(-((x - 0.7) / 0.08) ** 2))
        
    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")
    
    # Ensure u ∈ [0, 1] (physical constraint)
    return np.clip(u0, 0.0, 1.0)


def add_boundary_values(u_interior: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """
    Add boundary values to interior solution for plotting.
    
    Parameters
    ----------
    u_interior : np.ndarray
        Solution at interior points, shape (n_interior,) or (n_interior, n_times)
    config : SimulationConfig
        Simulation configuration
        
    Returns
    -------
    np.ndarray
        Full solution including boundaries
    """
    if u_interior.ndim == 1:
        if config.bc_type == 'dirichlet':
            return np.concatenate([[0.0], u_interior, [0.0]])
        else:  # Neumann: copy boundary values
            return np.concatenate([[u_interior[0]], u_interior, [u_interior[-1]]])
    else:
        n_times = u_interior.shape[1]
        if config.bc_type == 'dirichlet':
            zeros = np.zeros((1, n_times))
            return np.vstack([zeros, u_interior, zeros])
        else:
            return np.vstack([u_interior[0:1, :], u_interior, u_interior[-1:, :]])


def compute_metrics(u: np.ndarray, config: SimulationConfig) -> Dict:
    """
    Compute evaluation metrics for Fisher-KPP solution.
    
    Parameters
    ----------
    u : np.ndarray
        Solution array, shape (n_interior, n_times)
    config : SimulationConfig
        Simulation configuration
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {}
    
    # Maximum principle: u should stay in [0, 1]
    metrics['u_min'] = np.min(u)
    metrics['u_max'] = np.max(u)
    metrics['max_principle_satisfied'] = (metrics['u_min'] >= -1e-10 and 
                                           metrics['u_max'] <= 1.0 + 1e-10)
    
    # Mass (integral of u)
    h = config.h
    mass = np.sum(u, axis=0) * h
    metrics['initial_mass'] = mass[0]
    metrics['final_mass'] = mass[-1]
    
    # Steady state check (for Neumann BC, should approach u=1)
    metrics['final_mean'] = np.mean(u[:, -1])
    
    # L2 norm evolution
    metrics['l2_norm_initial'] = np.sqrt(np.sum(u[:, 0]**2) * h)
    metrics['l2_norm_final'] = np.sqrt(np.sum(u[:, -1]**2) * h)
    
    return metrics


def _choose_snapshot_indices(t, n_snapshots=10, prefer_log=True):
    """
    Robustly choose exactly n_snapshots indices.
    - If t contains 0, include it explicitly.
    - Use monotone interpolation in log(t) or linear(t) to get evenly-spaced samples.
    - Avoid collapsing to only a few unique indices.
    """
    t = np.asarray(t)
    n = len(t)
    if n_snapshots >= n:
        return np.arange(n)

    # include t=0 if present
    has_zero = (t[0] == 0.0)
    idx0 = [0] if has_zero else []
    start = 1 if has_zero else 0

    tt = t[start:]
    if len(tt) == 0:
        return np.array(idx0, dtype=int)

    # choose target values in log or linear space
    if prefer_log:
        tt_safe = np.clip(tt, 1e-300, None)
        s = np.log10(tt_safe)
    else:
        s = tt

    # map evenly-spaced s-targets -> fractional indices by interpolation
    m = n_snapshots - len(idx0)
    s_targets = np.linspace(s[0], s[-1], m)

    # indices in [0, len(tt)-1] corresponding to s_targets
    base = np.arange(len(tt), dtype=float)
    frac = np.interp(s_targets, s, base)
    idx = np.rint(frac).astype(int)

    # enforce uniqueness while keeping count m
    idx = np.clip(idx, 0, len(tt)-1)
    chosen = []
    used = set()
    # greedy: if duplicate, push to nearest available neighbor
    for k in idx:
        if k not in used:
            chosen.append(k); used.add(k)
        else:
            # search outward for nearest unused
            d = 1
            found = None
            while (k-d >= 0) or (k+d < len(tt)):
                if k-d >= 0 and (k-d) not in used:
                    found = k-d; break
                if k+d < len(tt) and (k+d) not in used:
                    found = k+d; break
                d += 1
            if found is None:
                # fallback: pick any unused
                for cand in range(len(tt)):
                    if cand not in used:
                        found = cand; break
            chosen.append(found); used.add(found)

    chosen = np.array(chosen, dtype=int) + start
    out = np.array(idx0 + chosen.tolist(), dtype=int)

    # final sanity: exactly n_snapshots
    if len(out) > n_snapshots:
        out = out[:n_snapshots]
    elif len(out) < n_snapshots:
        # pad with evenly spaced indices
        pad = np.linspace(0, n-1, n_snapshots, dtype=int)
        for p in pad:
            if len(out) >= n_snapshots: break
            if p not in out:
                out = np.append(out, p)

    return np.sort(out)


def plot_snapshots(t, x, u, config, sim_idx,
                   output_dir='plots/fisher_kpp',
                   n_snapshots=12,
                   n_background=60,
                   prefer_log=True):
    """
    Direction-enhanced snapshots:
    - draw many thin background curves (time "trajectory cloud")
    - highlight selected snapshots with thicker lines + inline time labels
    - add colorbar showing time with early->late cues
    """
    os.makedirs(output_dir, exist_ok=True)

    t = np.asarray(t)
    x = np.asarray(x)
    u = np.asarray(u)  # shape (n_x, n_t)

    # background indices: many curves to show continuous evolution
    if n_background >= len(t):
        bg_idx = np.arange(len(t))
    else:
        bg_idx = _choose_snapshot_indices(t, n_snapshots=n_background, prefer_log=prefer_log)

    # highlighted indices: fewer, emphasized
    hi_idx = _choose_snapshot_indices(t, n_snapshots=n_snapshots, prefer_log=prefer_log)

    # color mapping by time (log works well when t is log-spaced)
    if prefer_log:
        vmin = max(np.min(t[t > 0]) if np.any(t > 0) else 1e-12, 1e-12)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=np.max(t))
    else:
        norm = mpl.colors.Normalize(vmin=np.min(t), vmax=np.max(t))
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1) background "trajectory cloud"
    for j in bg_idx:
        ax.plot(x, u[:, j], color=cmap(norm(max(t[j], 1e-12))), linewidth=1.0, alpha=0.18)

    # 2) highlight snapshots (thicker + label at right edge)
    for j in hi_idx:
        col = cmap(norm(max(t[j], 1e-12)))
        ax.plot(x, u[:, j], color=col, linewidth=2.8, alpha=0.95)
        # inline label near right boundary (avoid clutter: label only some)
        ax.text(x[-1] + 0.01*(x[-1]-x[0]), u[-1, j],
                f'{t[j]:.3g}', color=col, fontsize=9, fontweight='bold',
                va='center', ha='left', clip_on=False)

    # 3) colorbar + explicit direction cue
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'time $t$', fontsize=12, fontweight='bold')
    cbar.ax.text(0.5, -0.08, 'early', transform=cbar.ax.transAxes,
                 ha='center', va='top', fontsize=10, fontweight='bold')
    cbar.ax.text(0.5, 1.04, 'late', transform=cbar.ax.transAxes,
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # small arrow in-axes for redundancy (works in print)
    ax.annotate('time increases', xy=(0.96, 0.92), xytext=(0.72, 0.92),
                xycoords='axes fraction', textcoords='axes fraction',
                ha='left', va='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.8, color='black'))

    ax.set_xlabel(r'$x$', fontweight='bold', fontsize=14)
    ax.set_ylabel(r'$u(x,t)$', fontweight='bold', fontsize=14)
    ax.set_title(f'Fisher-KPP Evolution ($\\varepsilon={config.epsilon}$)',
                 fontweight='bold', fontsize=13)

    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.05, 1.1)
    # leave room for right-side time labels
    ax.set_xlim(x[0], x[-1] + 0.08*(x[-1]-x[0]))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/snapshots_{sim_idx}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/snapshots_{sim_idx}.pdf', dpi=600, bbox_inches='tight')
    plt.close()


def plot_spacetime(t, x, u, config, sim_idx,
                   output_dir='plots/fisher_kpp'):
    """
    Spacetime visualization of Fisher-KPP solution.
    - Heatmap showing solution evolution
    - Contour lines marking key levels
    - Time direction arrow
    """
    os.makedirs(output_dir, exist_ok=True)

    t = np.asarray(t)
    x = np.asarray(x)
    u = np.asarray(u)  # shape (n_x, n_t)

    T, X = np.meshgrid(t, x)
    fig, ax = plt.subplots(figsize=(11, 6))

    # Main heatmap with 100 levels
    levels = np.linspace(np.min(u), np.max(u), 100)
    cf = ax.contourf(X, T, u, levels=levels, cmap='RdYlBu_r', extend='both')
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(r'$u(x,t)$', fontsize=12, fontweight='bold')

    # Contour lines at key levels
    cs = ax.contour(X, T, u, levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
                    colors='white', linewidths=1, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # Time direction indicator
    ax.annotate('', xy=(0.5, 0.9), xytext=(0.5, 0.1),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2.5, color='darkblue'))
    ax.text(0.53, 0.5, 'time', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='darkblue', va='center')

    # Labels
    ax.set_xlabel(r'$x$', fontweight='bold', fontsize=12)
    ax.set_ylabel(r'$t$', fontweight='bold', fontsize=12)
    ax.set_title(f'Fisher-KPP Spacetime ($\\varepsilon={config.epsilon}$)',
                 fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/spacetime_{sim_idx}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/spacetime_{sim_idx}.pdf', dpi=600, bbox_inches='tight')
    plt.close()


def plot_metrics_evolution(t: np.ndarray, u: np.ndarray,
                           config: SimulationConfig, sim_idx: int,
                           output_dir: str = 'plots/fisher_kpp'):
    """
    Plot evolution of solution metrics over time.
    """
    h = config.h
    
    # Compute time-dependent metrics
    mass = np.sum(u, axis=0) * h
    l2_norm = np.sqrt(np.sum(u**2, axis=0) * h)
    u_max = np.max(u, axis=0)
    u_min = np.min(u, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mass evolution
    ax = axes[0, 0]
    ax.plot(t, mass, 'b-', linewidth=2)
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('Mass $\\int u\\, dx$', fontsize=12)
    ax.set_title('Mass Evolution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # L2 norm evolution
    ax = axes[0, 1]
    ax.plot(t, l2_norm, 'r-', linewidth=2)
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('$\\|u\\|_{L^2}$', fontsize=12)
    ax.set_title('$L^2$ Norm Evolution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Max/Min evolution (maximum principle check)
    ax = axes[1, 0]
    ax.fill_between(t, u_min, u_max, alpha=0.3, color='green')
    ax.plot(t, u_max, 'g-', linewidth=2, label='max $u$')
    ax.plot(t, u_min, 'g--', linewidth=2, label='min $u$')
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='$u=1$')
    ax.axhline(y=0.0, color='k', linestyle=':', alpha=0.5, label='$u=0$')
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('$u$', fontsize=12)
    ax.set_title('Maximum Principle Check', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Stiffness info
    ax = axes[1, 1]
    ax.text(0.5, 0.7, f'Diffusion: $\\varepsilon = {config.epsilon}$',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.55, f'Grid spacing: $h = {config.h:.4f}$',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.4, f'Stiffness ratio: $\\varepsilon/h^2 = {config.stiffness_ratio:.1f}$',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.25, f'Interior points: $N = {config.n_interior}$',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.text(0.5, 0.1, f'BC type: {config.bc_type}',
            transform=ax.transAxes, fontsize=14, ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Simulation Parameters', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_{sim_idx}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/metrics_{sim_idx}.pdf', dpi=600, bbox_inches='tight')
    plt.close()


def benchmark_jacobian(config: SimulationConfig, n_trials: int = 5):
    """
    Benchmark solver performance with and without analytical Jacobian.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    n_trials : int
        Number of trials for timing
    """
    import time
    
    print("\n=== Jacobian Benchmark ===")
    print(f"Grid points: {config.n_interior}")
    print(f"Stiffness ratio ε/h² = {config.stiffness_ratio:.2f}")
    
    u0 = get_initial_condition(config.x_interior, 'gaussian')
    
    results = {}
    
    for use_jac in [False, True]:
        jac_str = "with" if use_jac else "without"
        times = []
        nfev_list = []
        njev_list = []
        
        for _ in range(n_trials):
            start = time.perf_counter()
            sol = solve_fisher_kpp(config, u0, use_jacobian=use_jac)
            elapsed = time.perf_counter() - start
            
            if sol['success']:
                times.append(elapsed)
                nfev_list.append(sol['nfev'])
                njev_list.append(sol['njev'])
        
        if times:
            results[jac_str] = {
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'nfev_mean': np.mean(nfev_list),
                'njev_mean': np.mean(njev_list),
            }
            print(f"\n{jac_str.capitalize()} analytical Jacobian:")
            print(f"  Time: {results[jac_str]['time_mean']:.4f} ± {results[jac_str]['time_std']:.4f} s")
            print(f"  Function evals: {results[jac_str]['nfev_mean']:.0f}")
            print(f"  Jacobian evals: {results[jac_str]['njev_mean']:.0f}")
    
    if 'with' in results and 'without' in results:
        speedup = results['without']['time_mean'] / results['with']['time_mean']
        print(f"\nSpeedup with analytical Jacobian: {speedup:.2f}x")
    
    return results


def run_teacher_pretraining_simulation(
    initial_conditions_list: List[Tuple[np.ndarray, str]],
    config: SimulationConfig,
    plot_every: Optional[int] = None
) -> np.ndarray:
    """
    Generate comprehensive dataset for teacher model pre-training.
    
    Parameters
    ----------
    initial_conditions_list : list
        List of (u0, ic_name) tuples
    config : SimulationConfig
        Simulation configuration
    plot_every : int, optional
        If provided, plot every nth simulation
        
    Returns
    -------
    np.ndarray
        Teacher dataset with columns: [t, u0_1, ..., u0_n, u_1, ..., u_n]
    """
    print("\n=== FISHER-KPP TEACHER DATASET GENERATION ===")
    
    # Create directories
    for dir_path in ['plots/fisher_kpp', 'plots/fisher_kpp/individual', 
                     'data/fisher_kpp']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Log-spaced time evaluation points to capture fast dynamics
    t_start, t_end = config.t_span
    
    if t_start <= 0:
        # Handle t=0 case: use small epsilon and prepend 0
        # Use fast timescale as reference: tau_fast ~ h^2 / epsilon
        tau_fast = config.h**2 / config.epsilon
        t_min = tau_fast / 100  # Start at 1% of fast timescale
        t_log = np.logspace(np.log10(t_min), np.log10(t_end), config.n_time_points - 1)
        t_eval = np.concatenate([[0.0], t_log])
    else:
        t_eval = np.logspace(np.log10(t_start), np.log10(t_end), config.n_time_points)
    
    # Build Laplacian once to show sparsity info
    A = build_laplacian_matrix(config)
    nnz = A.nnz
    total = config.n_interior ** 2
    
    print(f"\nConfiguration:")
    print(f"  Spatial points (interior): {config.n_interior}")
    print(f"  Grid spacing h: {config.h:.6f}")
    print(f"  Diffusion ε: {config.epsilon}")
    print(f"  Stiffness ratio ε/h²: {config.stiffness_ratio:.2f}")
    print(f"  Fast timescale h²/ε: {config.h**2/config.epsilon:.2e}")
    print(f"  Time span: [{config.t_span[0]}, {config.t_span[1]}]")
    print(f"  Time points: {config.n_time_points}")
    print(f"  Time sampling: LOG-SPACED (captures fast dynamics)")
    print(f"    First 5 times: {t_eval[:5]}")
    print(f"    Last 5 times: {t_eval[-5:]}")
    print(f"  Boundary condition: {config.bc_type}")
    print(f"  Solver: {config.solver_method}")
    print(f"  Laplacian sparsity: {nnz}/{total} ({100*nnz/total:.2f}% non-zero)")
    print(f"  Using analytical Jacobian: Yes")
    
    teacher_results = []
    
    print(f"\nSimulating {len(initial_conditions_list)} initial conditions...")
    for i, (u0, ic_name) in enumerate(tqdm(initial_conditions_list, desc="Simulations", ncols=100)):
        # Solve Fisher-KPP with analytical Jacobian
        sol = solve_fisher_kpp(config, u0, t_eval, use_jacobian=True)
        
        if not sol['success']:
            print(f"\nWarning: Simulation {i} ({ic_name}) failed")
            continue
        
        # Compute metrics
        metrics = compute_metrics(sol['u'], config)
        if not metrics['max_principle_satisfied']:
            print(f"\nWarning: Maximum principle violated in simulation {i}")
        
        # Store data for each time point
        # Format: [t, u0_1, ..., u0_n, u_1(t), ..., u_n(t)]
        for j, t in enumerate(sol['t']):
            row = np.concatenate([
                [t],           # time
                u0,            # initial condition (n_interior values)
                sol['u'][:, j] # solution at time t (n_interior values)
            ])
            teacher_results.append(row)
        
        # Plot if requested
        if plot_every is not None and i % plot_every == 0:
            u_full = add_boundary_values(sol['u'], config)
            x_full = config.x_full
            
            plot_spacetime(sol['t'], x_full, u_full, config, i)
            plot_snapshots(sol['t'], x_full, u_full, config, i)
            plot_metrics_evolution(sol['t'], sol['u'], config, i)
            plot_spacetime_3d(sol['t'], x_full, u_full, config, i)  # NEW 3D plot
    
    # Combine results
    teacher_data = np.array(teacher_results)
    
    print(f"\nTeacher dataset generated:")
    print(f"  Shape: {teacher_data.shape}")
    print(f"  Columns: time (1) + u0 ({config.n_interior}) + u ({config.n_interior})")
    print(f"  Total simulations: {len(initial_conditions_list)}")
    print(f"  Total data points: {len(teacher_data)}")
    
    return teacher_data


def plot_spacetime_3d(t, x, u, config, sim_idx,
                      output_dir='plots/fisher_kpp',
                      azim=45, elev=25):
    """
    3D surface plot of Fisher-KPP solution u(x,t).
    
    Parameters
    ----------
    t : np.ndarray
        Time points
    x : np.ndarray
        Spatial grid points
    u : np.ndarray
        Solution array, shape (n_x, n_t)
    config : SimulationConfig
        Simulation configuration
    sim_idx : int
        Simulation index for file naming
    output_dir : str
        Output directory path
    azim : float
        Azimuth angle (degrees)
    elev : float
        Elevation angle (degrees)
    """
    os.makedirs(output_dir, exist_ok=True)

    t = np.asarray(t)
    x = np.asarray(x)
    u = np.asarray(u)

    T, X = np.meshgrid(t, x)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot with cool colormap
    surf = ax.plot_surface(X, T, u, cmap='RdYlBu_r', 
                          linewidth=0, antialiased=True, 
                          alpha=0.95, shade=True,
                          vmin=np.min(u), vmax=np.max(u))

    # Add contour projections on bottom for structure
    contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    ax.contourf(X, T, u, levels=contour_levels, 
               zdir='z', offset=np.min(u)-0.1, 
               cmap='RdYlBu_r', alpha=0.4)

    # Colorbar
    cbar = plt.colorbar(surf, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(r'$u(x,t)$', fontsize=12, fontweight='bold', rotation=270, labelpad=20)

    # Labels with enhanced styling
    ax.set_xlabel(r'$x$ (space)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel(r'$t$ (time)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_zlabel(r'$u(x,t)$', fontsize=13, fontweight='bold', labelpad=10)

    # Title
    ax.set_title(f'Fisher-KPP 3D Spacetime ($\\varepsilon={config.epsilon}$)',
                fontweight='bold', fontsize=14, pad=20)

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Grid styling
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(f'{output_dir}/spacetime_3d_{sim_idx}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/spacetime_3d_{sim_idx}.pdf', dpi=600, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate datasets for Fisher-KPP reaction-diffusion equation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', choices=['teacher', 'verify', 'single', 'benchmark'], 
                       default='teacher', help='Operation mode')
    parser.add_argument('--n_interior', type=int, default=100,
                       help='Number of interior grid points')
    parser.add_argument('--epsilon', type=float, default=0.01,
                       help='Diffusion coefficient')
    parser.add_argument('--bc_type', choices=['dirichlet', 'neumann'], default='dirichlet',
                       help='Boundary condition type')
    parser.add_argument('--t_end', type=float, default=10.0,
                       help='End time for simulation')
    parser.add_argument('--n_time_points', type=int, default=1000,
                       help='Number of time points')
    parser.add_argument('--num_ics', type=int, default=1,
                       help='Number of different initial conditions (1 is often sufficient)')
    parser.add_argument('--plot_every', type=int, default=1,
                       help='Plot every N simulations')
    parser.add_argument('--ic_type', type=str, default='step',
                       help='Initial condition type: gaussian, step, sine, random, double_gaussian')
    args = parser.parse_args()
    
    # Configuration
    config = SimulationConfig(
        n_interior=args.n_interior,
        epsilon=args.epsilon,
        bc_type=args.bc_type,
        t_span=(0.0, args.t_end),
        n_time_points=args.n_time_points
    )
    
    if args.mode == 'benchmark':
        benchmark_jacobian(config)
        
    elif args.mode == 'verify':
        print("\n=== Solver Verification Mode ===")
        print(f"Stiffness ratio ε/h² = {config.stiffness_ratio:.2f}")
        
        A = build_laplacian_matrix(config)
        print(f"Laplacian: sparse matrix with {A.nnz} non-zeros ({A.format} format)")
        
        u0 = get_initial_condition(config.x_interior, 'gaussian')
        
        for solver in ['BDF', 'Radau', 'LSODA']:
            config.solver_method = solver
            print(f"\nTesting {solver} solver...")
            try:
                sol = solve_fisher_kpp(config, u0, use_jacobian=True)
                metrics = compute_metrics(sol['u'], config)
                print(f"  Success: {sol['success']}")
                print(f"  Function evaluations: {sol['nfev']}")
                print(f"  Jacobian evaluations: {sol['njev']}")
                print(f"  Max principle satisfied: {metrics['max_principle_satisfied']}")
                print(f"  u range: [{metrics['u_min']:.6f}, {metrics['u_max']:.6f}]")
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    elif args.mode == 'single':
        print("\n=== Single Simulation Mode ===")
        os.makedirs('plots/fisher_kpp', exist_ok=True)
        
        u0 = get_initial_condition(config.x_interior, args.ic_type)
        sol = solve_fisher_kpp(config, u0, use_jacobian=True)
        
        if sol['success']:
            u_full = add_boundary_values(sol['u'], config)
            x_full = config.x_full
            
            plot_spacetime(sol['t'], x_full, u_full, config, 0)
            plot_snapshots(sol['t'], x_full, u_full, config, 0)
            plot_metrics_evolution(sol['t'], sol['u'], config, 0)
            
            metrics = compute_metrics(sol['u'], config)
            print("\nMetrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            print(f"\nSolver stats:")
            print(f"  Function evaluations: {sol['nfev']}")
            print(f"  Jacobian evaluations: {sol['njev']}")
        else:
            print("Simulation failed!")
    
    elif args.mode == 'teacher':
        print("\n=== FISHER-KPP TEACHER DATASET GENERATION ===")
        print(f"\nNote: Using {args.num_ics} initial condition(s).")
        print(f"For high-dimensional systems (n={args.n_interior}), a single trajectory")
        print(f"with {args.n_time_points} time points often provides sufficient coverage")
        print(f"for learning the dynamics operator F(u).\n")
        
        # Create directories
        for dir_path in ['plots/fisher_kpp', 'data/fisher_kpp']:
            os.makedirs(dir_path, exist_ok=True)
        
        # Generate initial conditions
        initial_conditions_list = []
        np.random.seed(42)
        
        if args.num_ics == 1:
            # Single IC: use a representative one
            u0 = get_initial_condition(config.x_interior, args.ic_type)
            initial_conditions_list.append((u0, f'{args.ic_type}_0'))
        else:
            # Multiple ICs: cycle through types for diversity
            ic_types = ['gaussian', 'step', 'sine', 'double_gaussian']
            for i in range(args.num_ics):
                ic_type = ic_types[i % len(ic_types)]
                
                if ic_type == 'gaussian':
                    params = {
                        'center': np.random.uniform(0.3, 0.7),
                        'width': np.random.uniform(0.05, 0.15),
                        'amplitude': np.random.uniform(0.5, 1.0)
                    }
                elif ic_type == 'step':
                    params = {
                        'x_step': np.random.uniform(0.2, 0.5),
                        'transition_width': np.random.uniform(0.03, 0.08)
                    }
                elif ic_type == 'sine':
                    params = {'n_modes': np.random.randint(1, 4)}
                else:
                    params = {}
                
                u0 = get_initial_condition(config.x_interior, ic_type, params)
                initial_conditions_list.append((u0, f'{ic_type}_{i}'))
        
        # Generate dataset
        teacher_data = run_teacher_pretraining_simulation(
            initial_conditions_list, config, plot_every=args.plot_every
        )
        
        # Save dataset
        n = config.n_interior
        header = 'time,' + ','.join([f'u0_{i+1}' for i in range(n)]) + ',' + \
                 ','.join([f'u_{i+1}' for i in range(n)])
        
        filename = f'teacher_fisher_kpp_eps{config.epsilon}_n{n}_ics{args.num_ics}'
        np.save(f'data/fisher_kpp/{filename}.npy', teacher_data)
        np.savetxt(f'data/fisher_kpp/{filename}.csv',
                   teacher_data, delimiter=',', header=header, comments='')
        
        print(f"\n✓ Dataset saved:")
        print(f"  {filename}.npy/.csv")
        print(f"  Data shape: {teacher_data.shape}")
        print(f"  Samples per IC: {args.n_time_points}")
        print(f"  Total samples: {teacher_data.shape[0]}")
        
        # Print coverage analysis
        print(f"\n=== Coverage Analysis ===")
        print(f"  State dimension: {n}")
        print(f"  Time points per trajectory: {args.n_time_points}")
        print(f"  Number of trajectories: {args.num_ics}")
        print(f"  Total (u, F(u)) pairs: {teacher_data.shape[0]}")
        print(f"  Ratio samples/dimension: {teacher_data.shape[0] / n:.1f}")

if __name__ == '__main__':
    main()