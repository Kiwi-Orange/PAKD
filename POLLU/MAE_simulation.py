import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import argparse
from tqdm import tqdm
import warnings
import matplotlib as mpl
from typing import List, Tuple, Dict, Optional
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
    """Configuration for POLLU simulation."""
    n_species: int = 20
    n_reactions: int = 25
    n_points_per_phase: int = 1000
    atol: float = 1e-10
    rtol: float = 1e-8
    solver_method: str = 'BDF'
    
    # Single time span for entire simulation
    t_span: Tuple[float, float] = (1e-12, 1e4)

def get_pollu_rate_constants() -> np.ndarray:
    """
    Return the rate constants for the POLLU model.
    
    Returns
    -------
    np.ndarray
        Array of 25 rate constants with physical units [1/s] or [1/(M·s)]
    """
    return np.array([
        3.5e+0,   # k1
        5.0e+1,   # k2     (medium+)
        1.0e+4,   # k3     (fast+)
        5.0e-2,   # k4     (slow)
        5.0e-2,   # k5     (slow)
        2.0e+4,   # k6     (fast+)
        5.0e-3,   # k7     (very slow)
        5.0e+4,   # k8     (very fast)
        2.0e+4,   # k9     (very fast)
        1.0e+5,   # k10    (ultra fast)
        5.0e-1,   # k11    (medium-)
        2.0e+4,   # k12    (very fast)
        3.0e-1,   # k13    (medium-)
        1.0e+4,   # k14    (very fast)
        1.0e+6,   # k15    (ultra fast)
        1.0e-2,   # k16    (slow)
        2.0e+0,   # k17    (medium)
        5.0e+5,   # k18    (ultra fast)
        2.0e+6,   # k19    (ultra fast)
        3.0e+3,   # k20    (fast)
        3.0e+0,   # k21    (medium)
        8.0e+0,   # k22    (medium+)
        5.0e+0,   # k23    (medium+)
        3.0e+3,   # k24    (fast)
        5.0e+0    # k25    (medium+)
    ], dtype=np.float64)

def get_pollu_initial_conditions() -> np.ndarray:
    """
    Return the initial conditions for the POLLU model.
    
    Designed to activate multiple reaction pathways:
    - High NO₂ (y2) and O₃ (y4) for photochemistry
    - Moderate NO (y1) for NOₓ cycling
    - Substantial radical precursors (y6, y7, y9)
    - Non-zero intermediates to test full reaction network
    
    Returns
    -------
    np.ndarray
        Initial concentrations of 20 species [M]
    """
    return np.array([
        5e2,    # y1  - NO (moderate, enables r1, r9, r13, r22, r23)
        2e3,    # y2  - NO₂ (high, key pollutant, drives r1, r2, r8, r11)
        1e2,    # y3  - NO₃ (present, enables r14)
        5e2,    # y4  - O₃ (substantial, enables r1, r15, r16, r22)
        2e2,    # y5  - Alkene (moderate, enables r2)
        1e2,    # y6  - OH radical (moderate, enables r5, r7, r13, r19)
        1e3,    # y7  - RO₂ radical (high, drives r3, r4, r5)
        5e2,    # y8  - Product (present)
        1e2,    # y9  - HO₂ radical (moderate, enables r6, r7)
        5e1,    # y10 - Intermediate (non-zero, enables r11)
        2e2,    # y11 - Reactive species (moderate, enables r8, r9)
        5e1,    # y12 - Product
        1e2,    # y13 - Species (enables r10)
        8e1,    # y14 - Intermediate (enables r12)
        3e1,    # y15 - Product
        1e2,    # y16 - Precursor (enables r17, r18)
        5e1,    # y17 - Species (enables r19)
        2e2,    # y18 - Reservoir
        1e2,    # y19 - Key intermediate (enables r20, r21, r23)
        5e1     # y20 - Species (enables r24)
    ], dtype=np.float64)

def pollu_reaction(t: float, y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    POLLU air pollution model with 20 species and 25 reactions.
    
    Parameters
    ----------
    t : float
        Time point (required for ODE solvers but not used)
    y : np.ndarray
        Current state of the system (20 species concentrations)
    k : np.ndarray
        Reaction rate constants (25 values)
        
    Returns
    -------
    np.ndarray
        Derivatives dy/dt for each species
    """
    # Ensure non-negative concentrations for physical validity
    y = np.maximum(y, 0.0)
    
    # Unpack species concentrations for clarity
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20 = y
    
    # Pre-allocate reaction rates array
    r = np.empty(25, dtype=np.float64)
    
    # Calculate reaction rates
    r[0] = k[0] * y1
    r[1] = k[1] * y2 * y4
    r[2] = k[2] * y5 * y2
    r[3] = k[3] * y7
    r[4] = k[4] * y7
    r[5] = k[5] * y7 * y6
    r[6] = k[6] * y9
    r[7] = k[7] * y9 * y6
    r[8] = k[8] * y11 * y2
    r[9] = k[9] * y11 * y1
    r[10] = k[10] * y13
    r[11] = k[11] * y10 * y2
    r[12] = k[12] * y14
    r[13] = k[13] * y1 * y6
    r[14] = k[14] * y3
    r[15] = k[15] * y4
    r[16] = k[16] * y4
    r[17] = k[17] * y16
    r[18] = k[18] * y16
    r[19] = k[19] * y17 * y6
    r[20] = k[20] * y19
    r[21] = k[21] * y19
    r[22] = k[22] * y1 * y4
    r[23] = k[23] * y19 * y1
    r[24] = k[24] * y20
    
    # Calculate derivatives based on stoichiometry
    dy = np.empty(20, dtype=np.float64)
    dy[0] = -r[0] - r[9] - r[13] - r[22] - r[23] + r[1] + r[2] + r[8] + r[10] + r[11] + r[21] + r[24]
    dy[1] = -r[1] - r[2] - r[8] - r[11] + r[0] + r[20]
    dy[2] = -r[14] + r[0] + r[16] + r[18] + r[21]
    dy[3] = -r[1] - r[15] - r[16] - r[22] + r[14]
    dy[4] = -r[2] + 2*r[3] + r[5] + r[6] + r[12] + r[19]
    dy[5] = -r[5] - r[7] - r[13] - r[19] + r[2] + 2*r[17]
    dy[6] = -r[3] - r[4] - r[5] + r[12]
    dy[7] = r[3] + r[4] + r[5] + r[6]
    dy[8] = -r[6] - r[7]
    dy[9] = -r[11] + r[6] + r[8]
    dy[10] = -r[8] - r[9] + r[7] + r[10]
    dy[11] = r[8]
    dy[12] = -r[10] + r[9]
    dy[13] = -r[12] + r[11]
    dy[14] = r[13]
    dy[15] = -r[17] - r[18] + r[15]
    dy[16] = -r[19]
    dy[17] = r[19]
    dy[18] = -r[20] - r[21] - r[23] + r[22] + r[24]
    dy[19] = -r[24] + r[23]
    
    return dy

def solve_phase(
    t_span: Tuple[float, float],
    y0: np.ndarray,
    k: np.ndarray,
    t_eval: np.ndarray,
    config: SimulationConfig
) -> Dict:
    """
    Solve one phase of the POLLU simulation.
    
    Parameters
    ----------
    t_span : tuple
        Start and end time
    y0 : np.ndarray
        Initial conditions
    k : np.ndarray
        Rate constants
    t_eval : np.ndarray
        Time points for evaluation
    config : SimulationConfig
        Simulation configuration
        
    Returns
    -------
    dict
        Solution dictionary with 't', 'y', and 'success' keys
    """
    try:
        solution = solve_ivp(
            fun=lambda t, y: pollu_reaction(t, y, k),
            t_span=t_span,
            y0=y0,
            method=config.solver_method,
            t_eval=t_eval,
            atol=config.atol,
            rtol=config.rtol,
            max_step=np.inf
        )
        return {'t': solution.t, 'y': solution.y, 'success': solution.success}
    except Exception as e:
        print(f"Error solving phase {t_span}: {str(e)}")
        return {'t': np.array([]), 'y': np.array([]), 'success': False}

def generate_time_points(config: SimulationConfig) -> np.ndarray:
    """
    Generate time evaluation points uniformly on log scale.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
        
    Returns
    -------
    np.ndarray
        Array of 1000 time points from 1e-12 to 1e4
    """
    t_eval = np.logspace(
        np.log10(config.t_span[0]), 
        np.log10(config.t_span[1]), 
        config.n_points_per_phase
    )
    
    return t_eval

def plot_combined_species(
    combined_t: np.ndarray,
    combined_y: np.ndarray,
    config: SimulationConfig,
    sim_idx: int,
    output_dir: str = 'plots/teacher/combined'
):
    """
    Plot combined species evolution.
    
    Parameters
    ----------
    combined_t : np.ndarray
        Combined time array
    combined_y : np.ndarray
        Combined concentration array
    config : SimulationConfig
        Simulation configuration
    sim_idx : int
        Simulation index
    output_dir : str
        Output directory for plots
    """
    colors = plt.cm.tab20(np.linspace(0, 1, config.n_species))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all species on semi-log scale
    for species_idx in range(config.n_species):
        ax.semilogx(combined_t, combined_y[species_idx], 
                    alpha=0.8, color=colors[species_idx], linewidth=1.8,
                    label=f'$y_{{{species_idx+1}}}$')
    
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Concentration (M)', fontweight='bold', fontsize=14)
    ax.set_title(f'All Species Evolution', 
                 fontweight='bold', fontsize=15, pad=12)
    ax.legend(loc='best', fontsize=9, ncol=2, 
              frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/teacher_pollu_{sim_idx}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/teacher_pollu_{sim_idx}.pdf', dpi=600, bbox_inches='tight')
    plt.close()

def plot_individual_species(
    combined_t: np.ndarray,
    combined_y: np.ndarray,
    config: SimulationConfig,
    sim_idx: int,
    output_dir: str = 'plots/teacher/individual'
):
    """
    Plot individual species in a grid layout.
    
    Parameters
    ----------
    combined_t : np.ndarray
        Combined time array
    combined_y : np.ndarray
        Combined concentration array
    config : SimulationConfig
        Simulation configuration
    sim_idx : int
        Simulation index
    output_dir : str
        Output directory for plots
    """
    colors = plt.cm.tab20(np.linspace(0, 1, config.n_species))
    
    fig, axes = plt.subplots(5, 4, figsize=(16, 18))
    fig.suptitle(f'Individual Species Evolution', 
                fontsize=14, fontweight='bold', y=0.995)
    
    for species_idx in range(config.n_species):
        row = species_idx // 4
        col = species_idx % 4
        ax = axes[row, col]
        
        ax.semilogx(combined_t, combined_y[species_idx], 
                   linewidth=2, color=colors[species_idx])
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Concentration (M)', fontsize=10)
        ax.set_title(f'$y_{{{species_idx+1}}}$', fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
        ax.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/teacher_pollu_{sim_idx}_individual.png', 
                dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/teacher_pollu_{sim_idx}_individual.pdf', 
                dpi=600, bbox_inches='tight')
    plt.close()

def run_teacher_pretraining_simulation_pollu(
    initial_conditions_list: List[np.ndarray],
    k: np.ndarray,
    config: Optional[SimulationConfig] = None,
    plot_every: Optional[int] = None
) -> np.ndarray:
    """
    Generate comprehensive dataset for teacher model pre-training (POLLU).
    
    Parameters
    ----------
    initial_conditions_list : list
        List of initial condition arrays (each with 20 species)
    k : np.ndarray
        Reaction rate constants
    config : SimulationConfig, optional
        Simulation configuration
    plot_every : int, optional
        If provided, plot every nth simulation
        
    Returns
    -------
    np.ndarray
        Teacher dataset
    """
    if config is None:
        config = SimulationConfig()
    
    print("\n=== TEACHER MODEL PRE-TRAINING DATASET GENERATION (POLLU) ===")
    
    # Create directories
    for dir_path in ['plots/teacher', 'plots/teacher/individual', 
                     'plots/teacher/combined', 'data/teacher']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Generate time points
    time_points = generate_time_points(config)
    
    print(f"\nTime discretization:")
    print(f"  Total points: {len(time_points)}")
    print(f"  Time range: [{config.t_span[0]:.2e}, {config.t_span[1]:.2e}] seconds")
    print(f"  Sampling: uniform on log scale")
    
    teacher_results = []
    
    print(f"\nSimulating {len(initial_conditions_list)} initial conditions...")
    for i, y0 in enumerate(tqdm(initial_conditions_list, desc="Simulations", ncols=100)):
        # Single solve across entire time span
        sol = solve_phase(
            t_span=config.t_span,
            y0=y0,
            k=k,
            t_eval=time_points,
            config=config
        )
        
        if not sol['success']:
            print(f"\nWarning: Simulation {i} failed")
            continue
        
        combined_t = sol['t']
        combined_y = sol['y']
        
        # Store data: [time, y0[0]...y0[19], y[0]...y[19]]
        combined_data = np.column_stack((
            combined_t,
            np.tile(y0, (len(combined_t), 1)),
            combined_y.T
        ))
        
        teacher_results.append(combined_data)
        
        # Plot if requested
        if (plot_every is not None and i % plot_every == 0) or len(initial_conditions_list) == 1:
            plot_combined_species(combined_t, combined_y, config, i)
            plot_individual_species(combined_t, combined_y, config, i)
    
    # Combine all results
    teacher_data = np.vstack(teacher_results)
    
    print(f"\nTeacher dataset generated:")
    print(f"  Shape: {teacher_data.shape}")
    print(f"  Columns: time (1) + initial_conditions (20) + concentrations (20) = 41")
    print(f"  Time range: [{np.min(teacher_data[:, 0]):.2e}, "
          f"{np.max(teacher_data[:, 0]):.2e}]")
    print(f"  Total simulations: {len(teacher_results)}")
    print(f"  Total time points: {len(teacher_data)}")
    
    # Concentration statistics
    concentrations = teacher_data[:, 21:41]
    print(f"\nConcentration statistics:")
    print(f"  Range: [{np.min(concentrations):.3e}, {np.max(concentrations):.3e}]")
    print(f"  Mean: {np.mean(concentrations):.3e}")
    print(f"  Std: {np.std(concentrations):.3e}")
    
    return teacher_data

def main():
    parser = argparse.ArgumentParser(
        description='Generate datasets for POLLU air pollution model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', choices=['teacher', 'verify'], default='teacher',
                       help='Operation mode')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of different initial conditions')
    parser.add_argument('--plot_every', type=int, default=1,
                       help='Plot every N simulations')
    parser.add_argument('--single_ic', action='store_true',
                       help='Use only base initial condition')
    parser.add_argument('--ic_variation', type=float, default=0.2,
                       help='Initial condition variation range (±20%% by default)')
    parser.add_argument('--n_points', type=int, default=1000,
                       help='Number of points per phase')
    args = parser.parse_args()
    
    # Configuration
    config = SimulationConfig(n_points_per_phase=args.n_points)
    
    # Get rate constants and initial conditions
    k = get_pollu_rate_constants()
    y0_base = get_pollu_initial_conditions()
    
    if args.mode == 'verify':
        print("\n=== Solver Verification Mode ===")
        print("Testing with base initial conditions...")
        
        for solver in ['BDF', 'Radau']:
            print(f"\nTesting {solver} solver...")
            try:
                sol = solve_ivp(
                    fun=lambda t, y: pollu_reaction(t, y, k),
                    t_span=(1e-12, 1e4),
                    y0=y0_base,
                    method=solver,
                    t_eval=np.logspace(-12, 4, 1000),
                    atol=1e-10,
                    rtol=1e-8
                )
                print(f"  Success: {sol.success}")
                print(f"  Points: {len(sol.t)}")
                print(f"  Final concentrations range: "
                      f"[{np.min(sol.y[:, -1]):.2e}, {np.max(sol.y[:, -1]):.2e}]")
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    elif args.mode == 'teacher':
        # Generate initial conditions
        if args.single_ic:
            print("\nUsing single base initial condition")
            initial_conditions_list = [y0_base]
        else:
            np.random.seed(42)
            initial_conditions_list = []
            
            lower_bound = 1.0 - args.ic_variation
            upper_bound = 1.0 + args.ic_variation
            
            print(f"\nGenerating {args.num_samples} varied initial conditions")
            print(f"Variation range: [{lower_bound:.2f}, {upper_bound:.2f}] × base values")
            
            for _ in range(args.num_samples):
                variation = np.random.uniform(lower_bound, upper_bound, size=20)
                y0_varied = y0_base * variation
                y0_varied = np.maximum(y0_varied, 0.0)
                initial_conditions_list.append(y0_varied)
        
        # Generate teacher dataset
        teacher_data = run_teacher_pretraining_simulation_pollu(
            initial_conditions_list, k,
            config=config,
            plot_every=args.plot_every
        )
        
        # Save dataset (updated header - no phase column)
        header = 'time,' + ','.join([f'y0_{i+1}' for i in range(20)]) + ',' + \
                 ','.join([f'y{i+1}' for i in range(20)])
        
        filename_base = 'teacher_pollu_single_ic' if args.single_ic else f'teacher_pollu_{args.num_samples}_samples'
        
        np.save(f'data/teacher/{filename_base}.npy', teacher_data)
        np.savetxt(f'data/teacher/{filename_base}.csv',
                   teacher_data, delimiter=',', header=header, comments='')
        
        print(f"\n✓ Dataset saved:")
        print(f"  {filename_base}.npy/.csv")
        print(f"  Data shape: {teacher_data.shape} (time + 20 ICs + 20 concentrations)")

if __name__ == '__main__':
    main()