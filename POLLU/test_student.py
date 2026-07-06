import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.integrate import solve_ivp
from scipy.linalg import schur
from scipy.optimize import root
from tqdm import tqdm

# Import models
from models import MLP, ResidualMLP
from MAE_simulation import pollu_reaction, get_pollu_rate_constants, get_pollu_initial_conditions

# Constants
SPECIES_NAMES = [f'y{i+1}' for i in range(20)]
N_SPECIES = 20
TIME_POINTS = np.logspace(-12, 4, 1000)
EPS_JACOBIAN = 1e-8

# Rate constants
K = get_pollu_rate_constants()

# Color palette for all methods
COLORS = {
    'analytical': '#000000',  # Black
    'qssa': '#1F78B4',        # Blue
    'teacher': '#33A02C',     # Green
    'student': '#E31A1C',     # Red
}

# Publication-quality matplotlib settings — matches test_teacher.py
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05


class ModelConfig:
    """Configuration class for student model parameters"""
    def __init__(self, checkpoint: dict):
        self.is_kd_student = any(key in checkpoint for key in 
                                  ['student_type', 'teacher_model_path', 'training_args', 'model_type'])
        
        if not self.is_kd_student:
            raise ValueError("This script is for student models only. Use test_teacher.py for teacher models.")
        
        training_args = checkpoint.get('training_args', {})
        self.model_type = checkpoint.get('model_type', training_args.get('student_type', 'MLP'))
        self.hidden_dim = checkpoint.get('hidden_dim', training_args.get('student_hidden_dim', 128))
        self.num_blocks = checkpoint.get('num_blocks', training_args.get('student_num_blocks', 1))
        self.dropout = checkpoint.get('dropout', training_args.get('student_dropout', 0.0))


def create_model(model_type: str, hidden_dim: int, num_blocks: int, dropout: float) -> nn.Module:
    """Create model based on type"""
    if model_type == 'MLP':
        return MLP(input_size=21, output_size=20, hidden_sizes=[hidden_dim]*num_blocks, dropout=dropout)
    elif model_type == 'ResidualMLP':
        return ResidualMLP(input_size=21, output_size=20, hidden_dim=hidden_dim, 
                          num_blocks=num_blocks, dropout=dropout)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using NVIDIA GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device, is_student: bool = True):
    """Load a model from checkpoint"""
    print(f"Loading {'student' if is_student else 'teacher'} model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if is_student:
        config = ModelConfig(checkpoint)
        model_type = config.model_type
        hidden_dim = config.hidden_dim
        num_blocks = config.num_blocks
        dropout = config.dropout
    else:
        model_type = checkpoint.get('model_type', 'ResidualMLP')
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_blocks = checkpoint.get('num_layers', 3)
        dropout = checkpoint.get('dropout', 0.0)
        
        # Try to infer from state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'input_proj.weight' in state_dict:
                hidden_dim = state_dict['input_proj.weight'].shape[0]
                num_blocks = sum(1 for k in state_dict.keys() if 'blocks.' in k and '.ln.weight' in k)
    
    print(f"  Model type: {model_type}")
    print(f"  Hidden dim: {hidden_dim}, Blocks/Layers: {num_blocks}")
    
    model = create_model(model_type, hidden_dim, num_blocks, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    
    return model, checkpoint['X_scaler'], checkpoint.get('y_scaler'), model_type


def generate_analytical_solution(initial_conditions: np.ndarray):
    """Generate analytical solution for POLLU kinetics"""
    sol = solve_ivp(
        lambda t, y: pollu_reaction(t, y, K),
        [TIME_POINTS[0], TIME_POINTS[-1]], 
        initial_conditions,
        t_eval=TIME_POINTS, 
        method='BDF',
        rtol=1e-8, 
        atol=1e-10
    )
    
    if not sol.success:
        raise RuntimeError(f"Analytical solution failed: {sol.message}")
    return TIME_POINTS, sol.y.T


def compute_jacobian(y: np.ndarray) -> np.ndarray:
    """Compute the Jacobian matrix at state y using finite differences."""
    f0 = pollu_reaction(0, y, K)
    J = np.zeros((N_SPECIES, N_SPECIES))
    
    for j in range(N_SPECIES):
        y_pert = y.copy()
        y_pert[j] += EPS_JACOBIAN
        J[:, j] = (pollu_reaction(0, y_pert, K) - f0) / EPS_JACOBIAN
    
    return J


def compute_production_consumption(y: np.ndarray):
    """
    Compute production and consumption rates for each species.
    """
    dydt = pollu_reaction(0, y, K)
    
    # Get individual reaction rates
    r = np.zeros(25)
    r[0] = K[0] * y[0]
    r[1] = K[1] * y[1] * y[3]
    r[2] = K[2] * y[4] * y[1]
    r[3] = K[3] * y[6]
    r[4] = K[4] * y[6]
    r[5] = K[5] * y[6] * y[5]
    r[6] = K[6] * y[8]
    r[7] = K[7] * y[8] * y[5]
    r[8] = K[8] * y[10] * y[1]
    r[9] = K[9] * y[10] * y[0]
    r[10] = K[10] * y[12]
    r[11] = K[11] * y[9] * y[1]
    r[12] = K[12] * y[13]
    r[13] = K[13] * y[0] * y[5]
    r[14] = K[14] * y[2]
    r[15] = K[15] * y[3]
    r[16] = K[16] * y[3]
    r[17] = K[17] * y[15]
    r[18] = K[18] * y[15]
    r[19] = K[19] * y[16] * y[5]
    r[20] = K[20] * y[18]
    r[21] = K[21] * y[18]
    r[22] = K[22] * y[0] * y[3]
    r[23] = K[23] * y[18] * y[0]
    r[24] = K[24] * y[19]
    
    species_reactions = {
        0:  ([1, 2, 8, 10, 11, 21, 24], [0, 9, 13, 22, 23]),
        1:  ([0, 20], [1, 2, 8, 11]),
        2:  ([0, 16, 18, 21], [14]),
        3:  ([14], [1, 15, 16, 22]),
        4:  ([3, 5, 6, 12, 19], [2]),
        5:  ([2, 17], [5, 7, 13, 19]),
        6:  ([12], [3, 4, 5]),
        7:  ([3, 4, 5, 6], []),
        8:  ([], [6, 7]),
        9:  ([6, 8], [11]),
        10: ([7, 10], [8, 9]),
        11: ([8], []),
        12: ([9], [10]),
        13: ([11], [12]),
        14: ([13], []),
        15: ([15], [17, 18]),
        16: ([], [19]),
        17: ([19], []),
        18: ([22, 24], [20, 21, 23]),
        19: ([23], [24]),
    }
    
    species_info = {}
    
    for sp_idx in range(N_SPECIES):
        prod_rxns, cons_rxns = species_reactions[sp_idx]
        
        production = sum(r[rxn_idx] for rxn_idx in prod_rxns)
        consumption = sum(r[rxn_idx] for rxn_idx in cons_rxns)
        
        # Handle special cases (stoichiometry > 1)
        if sp_idx == 4:  # y5 has 2*r4
            production += r[3]
        if sp_idx == 5:  # y6 has 2*r18
            production += r[17]
        
        # Calculate balance ratio
        max_flux = max(production, consumption)
        if max_flux > 1e-30:
            balance_ratio = min(production, consumption) / max_flux
        else:
            balance_ratio = 0.0
        
        species_info[sp_idx] = {
            'production': production,
            'consumption': consumption,
            'balance_ratio': balance_ratio,
            'net_rate': dydt[sp_idx],
            'concentration': y[sp_idx],
            'is_cumulative': consumption < 1e-30 and production > 1e-20
        }
    
    return species_info


def identify_qssa_candidates_topk(y: np.ndarray, 
                                   max_qssa_species: int = 3,
                                   min_balance_ratio: float = 0.1,
                                   min_timescale_ratio: float = 5.0,
                                   min_relative_flux: float = 1e-8,
                                   verbose: bool = False) -> list:
    """
    Identify top-k species suitable for QSSA using simplified timescale analysis.
    """
    species_info = compute_production_consumption(y)
    
    # Compute Jacobian
    J = compute_jacobian(y)
    
    # Estimate species timescales from diagonal elements
    # τ_i ≈ 1 / |J_ii| (characteristic relaxation time)
    diag_J = np.abs(np.diag(J))
    
    # Use flux-based timescale as fallback when diagonal is near-zero
    # τ_i ≈ concentration / total_flux
    flux_timescales = np.zeros(N_SPECIES)
    for i in range(N_SPECIES):
        total_flux = species_info[i]['production'] + species_info[i]['consumption']
        conc = max(species_info[i]['concentration'], 1e-30)
        if total_flux > 1e-20:
            flux_timescales[i] = conc / total_flux
        else:
            flux_timescales[i] = 1e30  # Very slow if no flux
    
    # Combine: use Jacobian diagonal if valid, otherwise use flux-based
    timescales = np.zeros(N_SPECIES)
    for i in range(N_SPECIES):
        if diag_J[i] > 1e-20:
            timescales[i] = 1.0 / diag_J[i]
        else:
            timescales[i] = flux_timescales[i]
    
    # Compute relative timescale (smaller = faster = better QSSA candidate)
    # Use a reasonable cap on max timescale
    valid_timescales = timescales[timescales < 1e20]
    if len(valid_timescales) > 0:
        max_timescale = np.max(valid_timescales)
    else:
        max_timescale = 1.0  # Fallback
    
    # Cap infinite timescales
    timescales = np.minimum(timescales, max_timescale * 1e6)
    relative_timescale = timescales / max_timescale
    
    # Compute max flux across all species for relative comparison
    all_fluxes = [species_info[i]['production'] + species_info[i]['consumption'] 
                  for i in range(N_SPECIES)]
    max_flux_global = max(all_fluxes) if max(all_fluxes) > 0 else 1.0
    
    # Build candidate list
    candidates = []
    
    if verbose:
        print("\n" + "="*80)
        print("QSSA TOP-K CANDIDATE ANALYSIS (Simplified Jacobian)")
        print("="*80)
        print(f"Max timescale (capped): {max_timescale:.2e} s")
        print(f"Max flux (global): {max_flux_global:.2e}")
        print("-"*80)
        print(f"{'Species':>8} {'τ_rel':>10} {'Balance':>8} {'Conc':>10} "
              f"{'Prod':>10} {'Cons':>10} {'FluxRel':>10} {'RankScore':>10} {'Status':>10}")
        print("-"*80)
    
    for sp_idx in range(N_SPECIES):
        info = species_info[sp_idx]
        tau_rel = relative_timescale[sp_idx]
        
        # Compute relative flux
        total_flux = info['production'] + info['consumption']
        relative_flux = total_flux / max_flux_global
        
        # Exclusion criteria
        reasons = []
        
        # Check 1: Not cumulative (no consumption pathway)
        if info['is_cumulative']:
            reasons.append("cumulative")
        
        # Check 2: Minimum balance ratio
        if info['balance_ratio'] < min_balance_ratio:
            reasons.append(f"imbalanced({info['balance_ratio']:.2f})")
        
        # Check 3: Must be fast (small relative timescale)
        if tau_rel > 1.0 / min_timescale_ratio:
            reasons.append(f"slow(τ_rel={tau_rel:.2f})")
        
        # Check 4: Must have significant flux relative to system scale
        if relative_flux < min_relative_flux:
            reasons.append(f"low_flux({relative_flux:.1e})")
        
        # Check 5: Exclude zero/negligible concentration species
        if info['concentration'] < 1e-30:
            reasons.append("zero_conc")
        
        # Compute composite ranking score (higher = better QSSA candidate)
        if not reasons:
            speed_score = 1.0 - min(tau_rel, 1.0)  # Clamp to [0, 1]
            rank_score = (
                0.5 * speed_score +
                0.3 * info['balance_ratio'] +
                0.2 * min(1.0, -np.log10(info['concentration'] + 1e-30) / 10)
            )
            candidates.append((sp_idx, rank_score, tau_rel, info['balance_ratio']))
        else:
            rank_score = 0.0
        
        status = "CANDIDATE" if not reasons else "EXCLUDED"
        
        if verbose:
            print(f"y{sp_idx+1:>7} {tau_rel:>10.4f} {info['balance_ratio']:>8.3f} "
                  f"{info['concentration']:>10.2e} {info['production']:>10.2e} "
                  f"{info['consumption']:>10.2e} {relative_flux:>10.2e} {rank_score:>10.3f} {status:>10}")
            if reasons:
                print(f"         → Excluded: {', '.join(reasons)}")
    
    # Sort by composite rank score and select top-k
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [sp_idx for sp_idx, _, _, _ in candidates[:max_qssa_species]]
    
    if verbose:
        print("-"*80)
        if selected:
            print(f"TOP-{max_qssa_species} QSSA SPECIES: {[f'y{i+1}' for i in selected]}")
            print(f"ODE REDUCTION: {N_SPECIES} → {N_SPECIES - len(selected)} differential equations")
            for sp_idx, rank, tau, bal in candidates[:max_qssa_species]:
                print(f"  y{sp_idx+1}: rank={rank:.3f}, τ_rel={tau:.4f}, balance={bal:.3f}")
        else:
            print("No suitable QSSA candidates found - using full ODE system")
        print("="*80 + "\n")
    
    return sorted(selected)


def solve_qssa_single_species(y_full: np.ndarray, fast_idx: int, 
                               max_iter: int = 50, tol: float = 1e-10) -> float:
    """
    Solve QSSA for a single fast species using bisection method.
    Robust fallback for single-species case.
    """
    y = y_full.copy()
    
    y_low = 1e-30
    y_high = max(y[fast_idx] * 100, 1e-5)
    
    y[fast_idx] = y_low
    f_low = pollu_reaction(0, y, K)[fast_idx]
    
    y[fast_idx] = y_high
    f_high = pollu_reaction(0, y, K)[fast_idx]
    
    # Try to bracket the root
    if f_low * f_high > 0:
        for scale in [1e3, 1e6, 1e9]:
            y_high *= scale
            y[fast_idx] = y_high
            f_high = pollu_reaction(0, y, K)[fast_idx]
            if f_low * f_high <= 0:
                break
        else:
            return y_full[fast_idx]
    
    # Bisection
    for _ in range(max_iter):
        y_mid = (y_low + y_high) / 2
        y[fast_idx] = y_mid
        f_mid = pollu_reaction(0, y, K)[fast_idx]
        
        if abs(f_mid) < tol or (y_high - y_low) / 2 < tol * max(y_mid, 1e-30):
            return y_mid
        
        if f_low * f_mid < 0:
            y_high = y_mid
            f_high = f_mid
        else:
            y_low = y_mid
            f_low = f_mid
    
    return (y_low + y_high) / 2


def solve_qssa_multi_species(y_full: np.ndarray, fast_indices: list, 
                              y_fast_prev: np.ndarray = None,
                              max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """
    Solve QSSA algebraic equations for multiple fast species simultaneously.
    
    Uses scipy.optimize.root with warm starting from previous solution.
    Falls back to sequential single-species solving if root finding fails.
    
    Args:
        y_full: Full state vector (slow species values are fixed)
        fast_indices: List of indices for fast (QSSA) species
        y_fast_prev: Previous solution for warm starting (optional)
        max_iter: Maximum iterations for root solver
        tol: Convergence tolerance
    
    Returns:
        Array of QSSA species concentrations
    """
    n_fast = len(fast_indices)
    fast_indices_arr = np.array(fast_indices)
    
    # Initial guess: use previous solution or current values
    if y_fast_prev is not None and len(y_fast_prev) == n_fast:
        y_fast_init = np.maximum(y_fast_prev, 1e-30)
    else:
        y_fast_init = np.maximum(y_full[fast_indices_arr], 1e-30)
    
    def residual(y_fast_log):
        """Residual function in log-space for better conditioning"""
        y_fast = np.exp(y_fast_log)
        y_temp = y_full.copy()
        y_temp[fast_indices_arr] = y_fast
        dydt = pollu_reaction(0, y_temp, K)
        return dydt[fast_indices_arr]
    
    def jacobian(y_fast_log):
        """Analytical Jacobian of residual w.r.t. log(y_fast)"""
        y_fast = np.exp(y_fast_log)
        y_temp = y_full.copy()
        y_temp[fast_indices_arr] = y_fast
        
        J_sub = np.zeros((n_fast, n_fast))
        f0 = pollu_reaction(0, y_temp, K)[fast_indices_arr]
        
        for j in range(n_fast):
            y_pert = y_temp.copy()
            dy = max(EPS_JACOBIAN * y_fast[j], 1e-20)
            y_pert[fast_indices_arr[j]] += dy
            f1 = pollu_reaction(0, y_pert, K)[fast_indices_arr]
            # Chain rule: d/d(log y) = y * d/dy
            J_sub[:, j] = (f1 - f0) / dy * y_fast[j]
        
        return J_sub
    
    # Try root finding in log-space
    y_fast_log_init = np.log(y_fast_init)
    
    try:
        result = root(
            residual,
            y_fast_log_init,
            jac=jacobian,
            method='hybr',
            tol=tol,
            options={'maxfev': max_iter * n_fast}
        )
        
        if result.success or np.max(np.abs(result.fun)) < tol * 10:
            y_fast_solution = np.exp(result.x)
            return np.maximum(y_fast_solution, 1e-30)
    except Exception:
        pass
    
    # Fallback: solve sequentially
    y_fast_solution = np.zeros(n_fast)
    y_temp = y_full.copy()
    
    for i, fast_idx in enumerate(fast_indices):
        y_fast_solution[i] = solve_qssa_single_species(y_temp, fast_idx)
        y_temp[fast_idx] = y_fast_solution[i]
    
    return y_fast_solution


def qssa_solution(initial_conditions: np.ndarray, 
                  max_qssa_species: int = 3,
                  min_balance_ratio: float = 0.1,
                  min_timescale_ratio: float = 5.0,
                  min_relative_flux: float = 1e-8,
                  verbose: bool = False):
    """
    Generate QSSA solution with top-k fast species selection.
    """
    # Identify top-k QSSA candidates
    fast_species = identify_qssa_candidates_topk(
        initial_conditions, 
        max_qssa_species=max_qssa_species,
        min_balance_ratio=min_balance_ratio,
        min_timescale_ratio=min_timescale_ratio,
        min_relative_flux=min_relative_flux,
        verbose=verbose
    )
    
    # If no suitable species, fall back to analytical
    if len(fast_species) == 0:
        if verbose:
            print("No suitable QSSA species. Falling back to analytical solution.")
        return generate_analytical_solution(initial_conditions)
    
    # Setup indices
    fast_indices = list(fast_species)
    slow_indices = [i for i in range(N_SPECIES) if i not in fast_species]
    n_fast = len(fast_indices)
    
    if verbose:
        print(f"\nQSSA Configuration:")
        print(f"  Fast species (algebraic): {[f'y{i+1}' for i in fast_indices]}")
        print(f"  Slow species (differential): {[f'y{i+1}' for i in slow_indices]}")
        print(f"  System: {N_SPECIES} ODEs → {len(slow_indices)} ODEs + {n_fast} algebraic\n")
    
    y0_slow = initial_conditions[slow_indices]
    
    # Cache for warm starting and state tracking
    y_full_cache = [initial_conditions.copy()]
    y_fast_prev = [initial_conditions[fast_indices].copy()]
    qssa_solve_failures = [0]  # Track QSSA solver failures
    
    def qssa_rhs(t, y_slow):
        """RHS for reduced ODE system with QSSA closure"""
        # Reconstruct full state
        y_full_cache[0][slow_indices] = np.maximum(y_slow, 1e-30)  # Ensure positivity
        
        # Solve QSSA equations
        try:
            if n_fast == 1:
                y_fast = solve_qssa_single_species(y_full_cache[0], fast_indices[0])
                y_full_cache[0][fast_indices[0]] = y_fast
                y_fast_prev[0] = np.array([y_fast])
            else:
                y_fast = solve_qssa_multi_species(
                    y_full_cache[0], fast_indices, y_fast_prev[0]
                )
                y_full_cache[0][fast_indices] = y_fast
                y_fast_prev[0] = y_fast.copy()
        except Exception as e:
            qssa_solve_failures[0] += 1
            # Use previous values if QSSA solve fails
            pass
        
        # Compute RHS for slow species only
        dydt_full = pollu_reaction(t, y_full_cache[0], K)
        return dydt_full[slow_indices]
    
    # Integrate reduced system with more robust settings
    try:
        sol = solve_ivp(
            qssa_rhs,
            [TIME_POINTS[0], TIME_POINTS[-1]], 
            y0_slow,
            t_eval=TIME_POINTS, 
            method='BDF',
            rtol=1e-4,      # Relaxed from 1e-6
            atol=1e-6,      # Relaxed from 1e-9
            max_step=1.0,   # Increased from 0.5
            first_step=1e-12  # Start with small step for stiff early phase
        )
        
        if not sol.success:
            print(f"  [QSSA] Integration failed: {sol.message}")
            print(f"  [QSSA] QSSA solver failures during integration: {qssa_solve_failures[0]}")
            print(f"  [QSSA] Falling back to analytical solution.")
            return generate_analytical_solution(initial_conditions)
            
    except Exception as e:
        print(f"  [QSSA] Exception during integration: {e}")
        print(f"  [QSSA] Falling back to analytical solution.")
        return generate_analytical_solution(initial_conditions)
    
    if verbose or qssa_solve_failures[0] > 0:
        print(f"  [QSSA] Integration succeeded with {qssa_solve_failures[0]} QSSA solver failures")
    
    # Reconstruct full trajectory
    y_full = np.zeros((len(TIME_POINTS), N_SPECIES))
    y_full[:, slow_indices] = sol.y.T
    
    # Recover fast species values at each time point
    y_fast_running = initial_conditions[fast_indices].copy()
    
    for i in range(len(TIME_POINTS)):
        y_temp = y_full[i].copy()
        
        if n_fast == 1:
            y_fast_val = solve_qssa_single_species(y_temp, fast_indices[0])
            y_full[i, fast_indices[0]] = y_fast_val
            y_fast_running = np.array([y_fast_val])
        else:
            y_fast_vals = solve_qssa_multi_species(y_temp, fast_indices, y_fast_running)
            y_full[i, fast_indices] = y_fast_vals
            y_fast_running = y_fast_vals.copy()
    
    return TIME_POINTS, y_full


def generate_model_predictions(model: nn.Module, initial_conditions: np.ndarray, 
                               X_scaler, device: torch.device, y_scaler=None):
    """Generate predictions from model"""
    X_pred = np.zeros((len(TIME_POINTS), 21), dtype=np.float32)
    X_pred[:, 0] = TIME_POINTS
    X_pred[:, 1:21] = initial_conditions
    
    X_transformed = X_pred.copy()
    X_transformed[:, 0] = np.log10(X_transformed[:, 0] + 1e-12)
    X_pred_norm = X_scaler.transform(X_transformed)
    
    X_tensor = torch.tensor(X_pred_norm, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    if y_scaler is not None:
        predictions = y_scaler.inverse_transform(predictions)
    
    return TIME_POINTS, np.maximum(predictions, 0)


def generate_test_conditions(num_conditions=50, variation_scale='standard'):
    """Generate test conditions for POLLU"""
    np.random.seed(42)
    
    base_ic = get_pollu_initial_conditions()
    
    if num_conditions == 1:
        return base_ic.reshape(1, -1)
    
    key_species = [1, 3, 5, 6, 7, 8, 17, 18, 19]
    scale_factors = {'low': 0.5, 'standard': 1.0, 'high': 2.0}
    scale_factor = scale_factors.get(variation_scale, 1.0)
    
    conditions = []
    for _ in range(num_conditions):
        new_condition = base_ic.copy()
        for idx in key_species:
            if base_ic[idx] > 0:
                log_base = np.log10(base_ic[idx])
                log_var = np.random.uniform(-scale_factor, scale_factor)
                new_condition[idx] = 10 ** (log_base + log_var)
        conditions.append(new_condition)
    
    return np.array(conditions)


def batch_generate_trajectories(generator_func, test_conditions, desc="Generating", **kwargs):
    """Batch generate trajectories with progress bar"""
    trajectories = []
    failed = 0
    
    for ic in tqdm(test_conditions, desc=desc):
        try:
            _, traj = generator_func(ic, **kwargs)
            trajectories.append(traj)
        except Exception as e:
            _, traj = generate_analytical_solution(ic)
            trajectories.append(traj)
            failed += 1
    
    if failed > 0:
        print(f"  Warning: {failed}/{len(test_conditions)} conditions fell back to analytical")
    
    return np.array(trajectories)


def evaluate_all_methods(model, X_scaler, y_scaler, device, test_conditions,
                        teacher_model=None, teacher_X_scaler=None, teacher_y_scaler=None,
                        include_qssa=True, max_qssa_species: int = 3):
    """Unified evaluation for all methods"""
    results = {
        'test_conditions': test_conditions,
        'time_points': TIME_POINTS
    }
    
    print("Generating analytical solutions...")
    results['analytical'] = batch_generate_trajectories(
        generate_analytical_solution, test_conditions, desc="Analytical"
    )
    
    if include_qssa:
        print(f"\nGenerating QSSA predictions (top-{max_qssa_species} fast species)...")
        # Show analysis for first condition
        candidates = identify_qssa_candidates_topk(
            test_conditions[0], max_qssa_species=max_qssa_species, verbose=True
        )
        print(f"\n>>> QSSA candidates found: {len(candidates)} species")
        if len(candidates) == 0:
            print(">>> WARNING: No QSSA candidates found! QSSA will fall back to analytical solution.")
            print(">>> This is why QSSA and Ground Truth trajectories overlap exactly.")
        else:
            print(f">>> Selected QSSA species: {[f'y{i+1}' for i in candidates]}")
        
        results['qssa'] = batch_generate_trajectories(
            lambda ic: qssa_solution(ic, max_qssa_species=max_qssa_species, verbose=False), 
            test_conditions, desc="QSSA"
        )
        
        # Check if QSSA actually differs from analytical
        qssa_diff = np.max(np.abs(results['qssa'] - results['analytical']))
        print(f"\n>>> Max difference between QSSA and Analytical: {qssa_diff:.2e}")
        if qssa_diff < 1e-6:
            print(">>> QSSA is identical to Analytical - likely falling back due to no candidates.")
    
    if teacher_model is not None:
        print("\nGenerating teacher predictions...")
        results['teacher'] = batch_generate_trajectories(
            lambda ic: generate_model_predictions(teacher_model, ic, 
                                                   teacher_X_scaler, device, teacher_y_scaler),
            test_conditions, desc="Teacher"
        )
    
    print("\nGenerating student predictions...")
    results['student'] = batch_generate_trajectories(
        lambda ic: generate_model_predictions(model, ic, X_scaler, device, y_scaler),
        test_conditions, desc="Student"
    )
    
    return results


def compute_stiffness_metrics(trajectory):
    """Compute stiffness ratio for a trajectory"""
    stiffness_ratios = np.zeros(len(trajectory))
    
    for i, y in enumerate(trajectory):
        J = compute_jacobian(y)
        eig_magnitudes = np.abs(np.linalg.eigvals(J))
        eig_magnitudes = eig_magnitudes[eig_magnitudes > 1e-10]
        
        if len(eig_magnitudes) > 0:
            stiffness_ratios[i] = np.max(eig_magnitudes) / (np.min(eig_magnitudes) + 1e-10)
        else:
            stiffness_ratios[i] = 1.0
    
    return stiffness_ratios


def analyze_stiffness(results):
    """Analyze stiffness for all available methods"""
    stiffness_data = {}
    
    for method in ['analytical', 'qssa', 'teacher', 'student']:
        if method in results:
            print(f"Computing stiffness for {method}...")
            stiffness_list = []
            for traj in tqdm(results[method], desc=f"Stiffness ({method})"):
                stiffness_list.append(compute_stiffness_metrics(traj))
            stiffness_data[method] = np.array(stiffness_list)
    
    return stiffness_data


def plot_stiffness_comparison(stiffness_data, model_type, output_dir):
    """Create publication-quality stiffness comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}
    for method, data in stiffness_data.items():
        mean_stiffness = np.nanmean(data, axis=1)
        peak_stiffness = np.nanmax(data, axis=1)
        stats[method] = {
            'mean': np.nanmean(mean_stiffness),
            'mean_std': np.nanstd(mean_stiffness),
            'peak': np.nanmean(peak_stiffness),
            'peak_std': np.nanstd(peak_stiffness),
            'mean_values': mean_stiffness,
            'peak_values': peak_stiffness
        }
    
    if 'analytical' in stats:
        gt_mean = stats['analytical']['mean_values']
        gt_peak = stats['analytical']['peak_values']
        for method in stats:
            if method != 'analytical':
                mean_err = np.abs(stats[method]['mean_values'] - gt_mean) / (gt_mean + 1e-10)
                peak_err = np.abs(stats[method]['peak_values'] - gt_peak) / (gt_peak + 1e-10)
                stats[method]['mean_error'] = np.nanmean(mean_err) * 100
                stats[method]['peak_error'] = np.nanmean(peak_err) * 100
    
    method_order = ['analytical', 'qssa', 'teacher', 'student']
    available_methods = [m for m in method_order if m in stats]
    labels = [m.capitalize() for m in available_methods]
    colors_list = [COLORS[m] for m in available_methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    means = [stats[m]['mean'] for m in available_methods]
    stds = [stats[m]['mean_std'] for m in available_methods]
    ax.bar(range(len(means)), means, color=colors_list, alpha=0.8, 
           edgecolor='black', linewidth=1.5, width=0.6,
           yerr=stds, capsize=8, error_kw={'linewidth': 2})
    ax.set_ylabel('Mean Stiffness Ratio', fontsize=14, fontweight='bold')
    ax.set_title('Average Mean Stiffness\n(Across All ICs)', fontsize=15, fontweight='bold', pad=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1]
    peaks = [stats[m]['peak'] for m in available_methods]
    peak_stds = [stats[m]['peak_std'] for m in available_methods]
    ax.bar(range(len(peaks)), peaks, color=colors_list, alpha=0.8,
           edgecolor='black', linewidth=1.5, width=0.6,
           yerr=peak_stds, capsize=8, error_kw={'linewidth': 2})
    ax.set_ylabel('Peak Stiffness Ratio', fontsize=14, fontweight='bold')
    ax.set_title('Average Peak Stiffness\n(Across All ICs)', fontsize=15, fontweight='bold', pad=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stiffness_comparison.pdf')
    plt.savefig(f'{output_dir}/stiffness_comparison.png', dpi=300)
    plt.close()
    
    print(f"\n{'='*70}")
    print("STIFFNESS COMPARISON")
    print(f"{'='*70}")
    
    print(f"\nMEAN STIFFNESS RATIO:")
    for method in available_methods:
        s = stats[method]
        line = f"  {method.capitalize():12s}: {s['mean']:.4e} ± {s['mean_std']:.4e}"
        if 'mean_error' in s:
            line += f"  (error: {s['mean_error']:.2f}%)"
        print(line)
    
    print(f"\nPEAK STIFFNESS RATIO:")
    for method in available_methods:
        s = stats[method]
        line = f"  {method.capitalize():12s}: {s['peak']:.4e} ± {s['peak_std']:.4e}"
        if 'peak_error' in s:
            line += f"  (error: {s['peak_error']:.2f}%)"
        print(line)
    
    print(f"\n{'='*70}")
    print(f"✓ Stiffness plots saved to: {output_dir}")
    
    return stats


def plot_trajectories_comparison(results, model_type, output_dir, methods_to_plot=None):
    """Create publication-quality trajectory comparison plots"""
    os.makedirs(output_dir, exist_ok=True)

    if methods_to_plot is None:
        methods_to_plot = ['analytical', 'qssa', 'teacher', 'student']

    available_methods = [m for m in methods_to_plot if m in results]
    test_conditions = results['test_conditions']
    time_points = results['time_points']

    np.random.seed(42)
    num_conditions = len(test_conditions)
    num_to_plot = min(3, num_conditions)
    representative_indices = list(range(num_to_plot)) if num_conditions <= 3 else \
                             np.random.choice(num_conditions, size=num_to_plot, replace=False)

    def marker_indices_uniform_logt(time_points, n_markers=25, include_endpoints=True):
        t = np.asarray(time_points)
        t = np.clip(t, 1e-300, None)
        logt = np.log10(t)
        logt_u = np.linspace(logt[0], logt[-1], n_markers)
        t_u = 10 ** logt_u
        idx = np.searchsorted(t, t_u)
        idx = np.clip(idx, 0, len(t) - 1)
        idx = np.unique(idx)
        if include_endpoints:
            idx = np.unique(np.concatenate(([0], idx, [len(t) - 1])))
        return idx

    marker_idx_gt = marker_indices_uniform_logt(time_points, n_markers=28, include_endpoints=True)

    linestyles = {'analytical': 'none', 'qssa': ':', 'teacher': '-.', 'student': '-'}
    linewidths = {'analytical': 0, 'qssa': 2.5, 'teacher': 2.5, 'student': 3.0}
    labels_display = {'analytical': 'Ground Truth', 'qssa': 'QSSA',
                      'teacher': 'Teacher', 'student': 'Student'}

    key_species_idx = [0, 1, 3, 6, 7, 8]
    key_titles = [rf'$y_{{{i+1}}}$' for i in key_species_idx]

    # Compute y-limits for key species
    y_limits_key = []
    for s_idx in key_species_idx:
        all_data = []
        for idx in representative_indices:
            for method in available_methods:
                all_data.append(results[method][idx, :, s_idx])
        all_data = np.concatenate(all_data)
        y_min, y_max = np.min(all_data), np.max(all_data)
        y_range = y_max - y_min if (y_max - y_min) > 1e-12 else 1.0
        y_limits_key.append((y_min - 0.10 * y_range, y_max + 0.10 * y_range))

    # ========================================
    # Figure 1: Key species (2x3) per condition
    #   - GT: filled stars, others: lines only
    #   - shared legend above the grid
    # ========================================
    for plot_id, idx in enumerate(representative_indices):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()

        for ax_i, (species_idx, title_str) in enumerate(zip(key_species_idx, key_titles)):
            ax = axes[ax_i]

            for method in available_methods:
                data = results[method][idx, :, species_idx]
                label = labels_display[method] if ax_i == 0 else ""

                if method == 'analytical':
                    ax.semilogx(time_points[marker_idx_gt], data[marker_idx_gt],
                                marker='*', markersize=11.0, linestyle='none',
                                markeredgewidth=0.5, markeredgecolor=COLORS[method],
                                markerfacecolor=COLORS[method], color=COLORS[method],
                                zorder=10, label=label)
                else:
                    ax.semilogx(time_points, data, color=COLORS[method],
                                linewidth=linewidths[method], linestyle=linestyles[method],
                                label=label, zorder=5 - list(available_methods).index(method))

            ax.set_xlim(time_points[0], time_points[-1])
            ax.set_ylim(y_limits_key[ax_i])

            ax.set_title(title_str, fontsize=15, fontweight='bold', pad=8)
            ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Conc. (M)', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12,
                           width=1.5, length=5, direction='in')
            ax.tick_params(axis='both', which='minor', width=1.0, length=3, direction='in')
            ax.grid(True, alpha=0.25, which='both', linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

        # Shared legend above the subplot grid
        handles, labels_leg = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_leg,
                   loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(available_methods), frameon=True,
                   fontsize=13, framealpha=0.95, edgecolor='gray',
                   fancybox=False, handlelength=2.0, columnspacing=1.5)

        plt.tight_layout(rect=[0, 0, 1, 0.93], pad=1.2)
        plt.savefig(f'{output_dir}/trajectories_key_species_cond{plot_id+1}.pdf', bbox_inches='tight', transparent=True)
        plt.savefig(f'{output_dir}/trajectories_key_species_cond{plot_id+1}.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

    # ========================================
    # Figure 2: All 20 species (complete grid) — teacher-consistent style
    # Temporarily use sans-serif to match teacher's visual weight (DejaVu Sans default)
    # ========================================
    ncols = 5
    nrows_species = int(np.ceil(N_SPECIES / ncols))   # 4 species-rows per condition

    # Global y-limits for all species
    y_limits_all = []
    for s_idx in range(N_SPECIES):
        all_data = []
        for idx in representative_indices:
            for method in available_methods:
                all_data.append(results[method][idx, :, s_idx])
        all_data = np.concatenate(all_data)
        y_min, y_max = np.min(all_data), np.max(all_data)
        y_range = y_max - y_min if (y_max - y_min) > 1e-12 else 1.0
        y_limits_all.append((y_min - 0.10 * y_range, y_max + 0.10 * y_range))

    fig_c = plt.figure(
        figsize=(ncols * 3.2, num_to_plot * nrows_species * 2.2 + 0.6),
        constrained_layout=False,
    )
    gs_top = fig_c.add_gridspec(
        num_to_plot, 1,
        hspace=0.55,
        left=0.07, right=0.98,
        top=0.93, bottom=0.06,
    )

    legend_handles = []
    legend_labels_c = []

    for r, cond_idx in enumerate(representative_indices):
        gs_cond = gs_top[r].subgridspec(nrows_species, ncols, hspace=0.55, wspace=0.30)

        for species_idx in range(N_SPECIES):
            sr = species_idx // ncols
            sc = species_idx % ncols
            ax = fig_c.add_subplot(gs_cond[sr, sc])

            for method in available_methods:
                data = results[method][cond_idx, :, species_idx]
                if method == 'analytical':
                    h = ax.semilogx(
                        time_points[marker_idx_gt], data[marker_idx_gt],
                        marker='*', markersize=8, linestyle='none',
                        markeredgewidth=1.5, markeredgecolor=COLORS[method],
                        markerfacecolor='white', color=COLORS[method],
                        zorder=5,
                    )
                else:
                    h = ax.semilogx(
                        time_points, data,
                        color=COLORS[method], linewidth=linewidths[method],
                        linestyle=linestyles[method],
                        zorder=4 - list(available_methods).index(method),
                    )
                if r == 0 and species_idx == 0:
                    legend_handles.append(h[0])
                    legend_labels_c.append(labels_display[method])

            ax.set_title(rf'$y_{{{species_idx+1}}}$', fontsize=13, fontweight='bold', pad=3)

            # x-label and tick labels only on bottom species-row of each condition block
            if sr == nrows_species - 1:
                ax.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=11, fontweight='bold')
            else:
                ax.tick_params(labelbottom=False)

            if sc == 0:
                ax.set_ylabel('Conc.', fontsize=11, fontweight='bold')

            ax.set_ylim(y_limits_all[species_idx])
            ax.set_xlim(time_points[0], time_points[-1])
            ax.tick_params(axis='both', which='major', labelsize=9,
                           pad=2, width=1.2, length=4, direction='out')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.spines['left'].set_linewidth(1.2)
            ax.xaxis.set_tick_params(direction='out')
            ax.yaxis.set_tick_params(direction='out')

    # Shared legend — top-right, no frame (matches teacher)
    fig_c.legend(
        legend_handles, legend_labels_c,
        loc='upper right', bbox_to_anchor=(0.98, 0.99),
        fontsize=12, frameon=False,
        ncol=len(available_methods), handlelength=2.0, handletextpad=0.5,
    )

    fig_c.savefig(f'{output_dir}/trajectories_all_species.pdf', bbox_inches='tight', transparent=True)
    fig_c.savefig(f'{output_dir}/trajectories_all_species.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_c)

    # ========================================
    # Figure 3: Stacked view (first condition, all species)
    #   - Titles not used; y-label already y_i
    # ========================================
    first_idx = representative_indices[0]
    fig, axes = plt.subplots(20, 1, figsize=(10, 28), sharex=True)

    for s, ax in enumerate(axes):
        for method in available_methods:
            data = results[method][first_idx, :, s]
            label = labels_display[method] if s == 0 else ""

            if method == 'analytical':
                ax.semilogx(time_points[marker_idx_gt], data[marker_idx_gt],
                            marker='*', markersize=9.0, linestyle='none',
                            markeredgewidth=0.4, markeredgecolor=COLORS[method],
                            markerfacecolor=COLORS[method], color=COLORS[method],
                            zorder=10, label=label)
            else:
                ax.semilogx(time_points, data, color=COLORS[method],
                            linewidth=linewidths[method], linestyle=linestyles[method],
                            label=label, zorder=5 - list(available_methods).index(method))

        ax.set_ylabel(rf'$y_{{{s+1}}}$', fontsize=13, fontweight='bold')
        ax.set_ylim(y_limits_all[s])
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.tick_params(axis='y', labelsize=11, width=1.3, length=4, direction='in')
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

    axes[-1].set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=14, fontweight='bold')
    axes[-1].set_xlim(time_points[0], time_points[-1])
    axes[-1].tick_params(axis='x', labelsize=12, width=1.3, length=4, direction='in')

    # Shared legend to the right of the stacked panels
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg,
               loc='center left', bbox_to_anchor=(1.01, 0.5),
               ncol=1, frameon=True,
               fontsize=12, framealpha=0.95, edgecolor='gray',
               fancybox=False, handlelength=2.0)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(f'{output_dir}/trajectories_stacked.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'{output_dir}/trajectories_stacked.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print(f"✓ Trajectory plots saved to: {output_dir}")


def compute_metrics(results):
    """Compute error metrics for all methods vs analytical"""
    metrics = {}
    gt = results['analytical']
    
    for method in ['qssa', 'teacher', 'student']:
        if method in results:
            pred = results[method]
            metrics[method] = {
                'mae_overall': mean_absolute_error(gt.ravel(), pred.ravel()),
                'rmse_overall': np.sqrt(mean_squared_error(gt.ravel(), pred.ravel())),
                'r2_overall': r2_score(gt.ravel(), pred.ravel()),
                'species': {}
            }
            
            for i, species in enumerate(SPECIES_NAMES):
                gt_s = gt[:, :, i].ravel()
                pred_s = pred[:, :, i].ravel()
                metrics[method]['species'][species] = {
                    'MAE': mean_absolute_error(gt_s, pred_s),
                    'RMSE': np.sqrt(mean_squared_error(gt_s, pred_s)),
                    'R2': r2_score(gt_s, pred_s)
                }
    
    return metrics


def print_metrics(metrics):
    """Print formatted metrics"""
    print(f"\n{'='*70}")
    print("PREDICTION ACCURACY COMPARISON")
    print(f"{'='*70}")
    
    for method, data in metrics.items():
        print(f"\n{method.upper()}:")
        print(f"  Overall: MAE={data['mae_overall']:.4e}, RMSE={data['rmse_overall']:.4e}, R²={data['r2_overall']:.4f}")
        
        species_r2 = [(s, d['R2']) for s, d in data['species'].items()]
        species_r2.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Best 5 species (by R²):")
        for species, r2 in species_r2[:5]:
            errors = data['species'][species]
            print(f"    {species}: MAE={errors['MAE']:.4e}, R²={r2:.4f}")
        
        print(f"  Worst 5 species (by R²):")
        for species, r2 in species_r2[-5:]:
            errors = data['species'][species]
            print(f"    {species}: MAE={errors['MAE']:.4e}, R²={r2:.4f}")
    
    if 'student' in metrics and 'qssa' in metrics:
        print(f"\n{'='*70}")
        print("STUDENT vs QSSA COMPARISON")
        print(f"{'='*70}")
        
        s = metrics['student']
        q = metrics['qssa']
        
        mae_imp = (1 - s['mae_overall'] / q['mae_overall']) * 100
        rmse_imp = (1 - s['rmse_overall'] / q['rmse_overall']) * 100
        r2_diff = s['r2_overall'] - q['r2_overall']
        
        print(f"  MAE improvement: {mae_imp:+.2f}%")
        print(f"  RMSE improvement: {rmse_imp:+.2f}%")
        print(f"  ΔR²: {r2_diff:+.4f}")
    
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate student models for POLLU')
    parser.add_argument('--student_model', type=str, required=True)
    parser.add_argument('--teacher_model', type=str, default=None)
    parser.add_argument('--compare_qssa', action='store_true')
    parser.add_argument('--max_qssa_species', type=int, default=4,
                       help='Maximum number of QSSA species (top-k selection)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--num_test_conditions', type=int, default=50)
    parser.add_argument('--use_base_ic', action='store_true', 
                       help='Test on base initial condition only')
    parser.add_argument('--output_dir', type=str, default='results/student_evaluation')
    parser.add_argument('--analyze_stiffness', action='store_true')
    
    args = parser.parse_args()
    
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    
    if args.use_base_ic:
        print("Using base initial condition only...")
        test_conditions = generate_test_conditions(num_conditions=1)
    else:
        print(f"Generating {args.num_test_conditions} test conditions...")
        test_conditions = generate_test_conditions(args.num_test_conditions)
    
    try:
        model, X_scaler, y_scaler, model_type = load_model(args.student_model, device, is_student=True)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    teacher_model = teacher_X_scaler = teacher_y_scaler = None
    if args.teacher_model:
        teacher_model, teacher_X_scaler, teacher_y_scaler, _ = load_model(
            args.teacher_model, device, is_student=False
        )
    
    output_dir = os.path.join(args.output_dir, f"Student_{model_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EVALUATING STUDENT MODEL ({model_type}) - POLLU")
    print(f"{'='*60}")
    
    results = evaluate_all_methods(
        model, X_scaler, y_scaler, device, test_conditions,
        teacher_model, teacher_X_scaler, teacher_y_scaler,
        include_qssa=args.compare_qssa,
        max_qssa_species=args.max_qssa_species
    )
    
    metrics = compute_metrics(results)
    print_metrics(metrics)
    
    print("\nGenerating trajectory plots...")
    plot_trajectories_comparison(results, model_type, output_dir)
    
    if args.analyze_stiffness:
        print("\nPerforming stiffness analysis...")
        stiffness_data = analyze_stiffness(results)
        plot_stiffness_comparison(stiffness_data, model_type, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()