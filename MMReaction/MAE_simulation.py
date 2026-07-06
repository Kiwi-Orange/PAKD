import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import argparse
from tqdm import tqdm  # For progress bars
import torch
import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Define the ODE system for the Michaelis-Menten reaction
def mm_reaction(t, y, k1, km1, k2):
    """
    Michaelis-Menten enzyme kinetics reaction model.

    This function implements the system of ordinary differential equations (ODEs)
    for the Michaelis-Menten enzyme kinetics mechanism:
    E + S <-> ES -> E + P

    Parameters
    ----------
    t : float
        Time point (required for ODE solvers, not used in the function)
    y : list or array
        Current state of the system as [E, S, ES, P], where:
        - E: enzyme concentration
        - S: substrate concentration
        - ES: enzyme-substrate complex concentration
        - P: product concentration
    k1 : float
        Forward rate constant for enzyme-substrate binding (E + S -> ES)
    km1 : float
        Reverse rate constant for enzyme-substrate dissociation (ES -> E + S)
    k2 : float
        Catalytic rate constant for product formation (ES -> E + P)

    Returns
    -------
    list
        Derivatives of each species with respect to time [dE_dt, dS_dt, dES_dt, dP_dt]
    """
    E, S, ES, P = y
    dE_dt = -k1 * E * S + km1 * ES + k2 * ES
    dS_dt = -k1 * E * S + km1 * ES
    dES_dt = k1 * E * S - km1 * ES - k2 * ES
    dP_dt = k2 * ES
    return [dE_dt, dS_dt, dES_dt, dP_dt]

def verify_ode_solvers(initial_conditions, k1=100.0, km1=10.0, k2=1.0):
    """
    Verify that different ODE solvers produce consistent results for the MM reaction.
    
    Parameters:
    ----------
    initial_conditions : list
        Initial condition [E, S, ES, P] to test
    k1, km1, k2 : float
        Reaction rate constants
        
    Returns:
    -------
    dict
        Dictionary of solver results for comparison
    """
    print("\n=== ODE Solver Verification ===")
    print(f"Testing with initial condition: E0={initial_conditions[0]:.1e}, S0={initial_conditions[1]:.1e}")
    
    # Create directories for plots
    os.makedirs('plots/verification', exist_ok=True)
    
    # Define solvers to test - select stable solvers for stiff systems
    solvers = ['BDF', 'Radau']
    print(f"Comparing solvers: {', '.join(solvers)}")
    
    # Define time spans - short and long
    t_span_short = (0, 1e-5)  # Changed from 1e-6 to 1e-5
    t_span_long = (0, 10)
    print(f"Short time span: {t_span_short}")
    print(f"Long time span: {t_span_long}")
    
    # Create time evaluation points with safe spacing to avoid numerical issues
    t_eval_short = np.logspace(-12, -5, 50)  # Updated upper bound to -5 (from -6)

    # Create long-term evaluation points with careful spacing
    t_eval_long = np.concatenate([
        np.logspace(-12, -2, 500),       # Updated upper bound to -2 (from -5)
        np.linspace(1e-2, 10, 500)      # More separation from fast phase boundary
    ])
    
    # Explicitly sort to ensure time points are in ascending order
    t_eval_short = np.sort(t_eval_short)
    t_eval_long = np.sort(t_eval_long)
    
    # Remove any duplicates that might exist
    t_eval_short = np.unique(t_eval_short)
    t_eval_long = np.unique(t_eval_long)
    
    print(f"Using {len(t_eval_short)} time points for short-term verification")
    print(f"Using {len(t_eval_long)} time points for long-term verification")
    
    # Store results for each solver
    short_results = {}
    long_results = {}
    
    # Use tqdm to show progress through solvers
    print("\nSolving ODEs with different solvers:")
    for solver in tqdm(solvers, desc="Testing solvers", ncols=100):
        try:
            # Short-term solution with tighter error control
            solution_short = solve_ivp(
                fun=lambda t, y: mm_reaction(t, y, k1, km1, k2),
                t_span=t_span_short,
                y0=initial_conditions,
                method=solver,
                t_eval=t_eval_short,
                atol=1e-8,
                rtol=1e-6,
                max_step=1e-7  # Limit maximum step size for better stability
            )
            
            # Long-term solution with tighter error control
            solution_long = solve_ivp(
                fun=lambda t, y: mm_reaction(t, y, k1, km1, k2),
                t_span=t_span_long,
                y0=initial_conditions,
                method=solver,
                t_eval=t_eval_long,
                atol=1e-8,
                rtol=1e-6
            )
            
            # Store results
            short_results[solver] = solution_short
            long_results[solver] = solution_long
            
        except Exception as e:
            print(f"\nError with {solver} solver: {str(e)}")
            # Continue with other solvers even if one fails
    
    # Compare results between solvers that completed successfully
    if len(short_results) > 1:
        reference_solver = list(short_results.keys())[0]  # Use first successful solver as reference
        print(f"\nComparing solvers against {reference_solver}:")
        
        # Use tqdm for the comparison process
        comparison_solvers = [s for s in short_results.keys() if s != reference_solver]
        for solver in tqdm(comparison_solvers, desc="Comparing results", ncols=100):
            # Compare short-term results
            max_diff_short = np.max(np.abs(short_results[solver].y - short_results[reference_solver].y))
            
            # Compare long-term results if available
            if solver in long_results and reference_solver in long_results:
                max_diff_long = np.max(np.abs(long_results[solver].y - long_results[reference_solver].y))
                print(f"\n{solver} vs {reference_solver}:")
                print(f"  Short-term max absolute difference: {max_diff_short:.3e}")
                print(f"  Long-term max absolute difference: {max_diff_long:.3e}")
            else:
                print(f"\n{solver} vs {reference_solver}:")
                print(f"  Short-term max absolute difference: {max_diff_short:.3e}")
                print(f"  Long-term comparison not available")
    else:
        print("\nNot enough successful solvers for comparison.")
    
    # Plot comparison if we have results
    print("\nGenerating comparison plots...")
    with tqdm(total=2, desc="Creating plots", ncols=100) as pbar:
        if len(short_results) > 0:
            plt.figure(figsize=(15, 10))
            
            # Compare species concentrations for short timespan
            for i, species in enumerate(['E', 'S', 'ES', 'P']):
                plt.subplot(2, 2, i+1)
                for solver in short_results.keys():
                    plt.plot(short_results[solver].t, short_results[solver].y[i], label=f'{solver}')
                plt.xlabel('Time')
                plt.ylabel(f'[{species}]')
                plt.title(f'Solver Comparison - Short-term [{species}]')
                plt.xscale('log')
                if i == 0:  # Only add legend to the first subplot
                    plt.legend()
            
            plt.tight_layout()
            plt.savefig('plots/verification/solver_comparison_short.pdf')
            pbar.update(1)
            
            # Long-term comparison if available
            if len(long_results) > 0:
                plt.figure(figsize=(15, 10))
                
                for i, species in enumerate(['E', 'S', 'ES', 'P']):
                    plt.subplot(2, 2, i+1)
                    for solver in long_results.keys():
                        plt.semilogx(long_results[solver].t, long_results[solver].y[i], label=f'{solver}')
                    plt.xlabel('Time (log scale)')
                    plt.ylabel(f'[{species}]')
                    plt.title(f'Solver Comparison - Long-term [{species}]')
                    if i == 0:  # Only add legend to the first subplot
                        plt.legend()
                
                plt.tight_layout()
                plt.savefig('plots/verification/solver_comparison_long.pdf')
            pbar.update(1)
    
    # Create a detailed error analysis
    print("\nPerforming detailed error analysis...")
    if len(short_results) > 1 and 'BDF' in short_results:
        reference_solver = 'BDF'  # Use BDF as reference
        
        # For each species, compare all solvers to BDF
        species_labels = ['E', 'S', 'ES', 'P']
        
        # Create error summary plot
        plt.figure(figsize=(15, 10))
        
        # Use tqdm for the error analysis process
        for i, species in enumerate(tqdm(species_labels, desc="Analyzing errors", ncols=100)):
            plt.subplot(2, 2, i+1)
            
            for solver in [s for s in short_results.keys() if s != reference_solver]:
                # Calculate relative error for this species
                rel_error = np.abs(short_results[solver].y[i] - short_results[reference_solver].y[i])
                rel_error /= (np.abs(short_results[reference_solver].y[i]) + 1e-10)  # Avoid division by zero
                
                plt.loglog(short_results[reference_solver].t, rel_error, label=f'{solver}')
                
            plt.xlabel('Time (log scale)')
            plt.ylabel(f'Relative error in [{species}]')
            plt.title(f'Solver Accuracy vs {reference_solver} - [{species}]')
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/verification/solver_error_analysis.pdf')
    
    plt.close('all')
    print("\nVerification complete. Comparison plots saved to plots/verification/ directory.")
    
    return short_results, long_results

def run_teacher_pretraining_simulation(initial_conditions, k1=100.0, km1=10.0, k2=1.0, plot_every=None):
    """
    Generate comprehensive dataset for teacher model pre-training.
    Dense sampling across the entire time range with a single unified phase.
    
    Parameters:
    ----------
    initial_conditions : list
        List of [E, S, ES, P] initial conditions to simulate
    k1, km1, k2 : float
        Reaction rate constants
    plot_every : int or None
        If provided, plot every nth simulation
    
    Returns:
    -------
    np.ndarray
        Combined dataset for comprehensive teacher training
    """
    print("\n=== TEACHER MODEL PRE-TRAINING DATASET GENERATION ===")
    print("Generating comprehensive dataset with unified time sampling for teacher training")
    
    # Create directories
    os.makedirs('plots/teacher', exist_ok=True)
    os.makedirs('data/teacher', exist_ok=True)
    
    # Define single time span - MODIFIED TO 1e2
    t_span = (1e-8, 1e2)

    # Dense sampling for comprehensive teacher training
    n_points = 1000 

    # Generate time points
    t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), n_points)

    print(f"Time points: {len(t_eval)} (t ∈ [{t_span[0]}, {t_span[1]}])")
    
    # Store results
    teacher_results = []
    
    print(f"Simulating {len(initial_conditions)} initial conditions for teacher pre-training...")
    for i, y0 in enumerate(tqdm(initial_conditions)):
        # Solve ODE across entire time range
        solution = solve_ivp(
            fun=lambda t, y: mm_reaction(t, y, k1, km1, k2),
            t_span=t_span,
            y0=y0,
            method='BDF',
            t_eval=t_eval,
            atol=1e-10,
            rtol=1e-8
        )
        
        # Extract solution
        t_result = solution.t
        E_result = solution.y[0]
        S_result = solution.y[1]
        ES_result = solution.y[2]
        P_result = solution.y[3]
        
        # Store data with original initial conditions
        result_data = np.column_stack((
            t_result,
            np.tile(y0, (len(t_result), 1)),  # Original initial conditions
            np.column_stack([E_result, S_result, ES_result, P_result])
        ))
        
        teacher_results.append(result_data)
        
        # Plot if requested - UPDATE PLOT RANGE TO 1e5
        if (plot_every is not None and i % plot_every == 0) or len(initial_conditions) == 1:
            plt.figure(figsize=(8, 6))
            colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
            linestyles = ['-', '--', '-.', ':']

            mask = (t_result >= 1e-8) & (t_result <= 1e5)  # MODIFIED
            t_plot = t_result[mask]
            E_plot = E_result[mask]
            S_plot = S_result[mask]
            ES_plot = ES_result[mask]
            P_plot = P_result[mask]

            # Plot with different line styles
            plt.semilogx(t_plot, E_plot, color=colors[0], linestyle=linestyles[0], label='E')
            plt.semilogx(t_plot, S_plot, color=colors[1], linestyle=linestyles[1], label='S')
            plt.semilogx(t_plot, ES_plot, color=colors[2], linestyle=linestyles[2], label='ES')
            plt.semilogx(t_plot, P_plot, color=colors[3], linestyle=linestyles[3], label='P')

            plt.xlabel('Time (s)')
            plt.ylabel('Concentration')
            plt.xlim(1e-8, 1e2)  # MODIFIED
            plt.ylim(-50, 1000)
            plt.margins(y=0.08)
            plt.tight_layout(pad=1.5)
            plt.legend(loc='upper right', frameon=False, ncol=2)
            plt.savefig(f'plots/teacher/teacher_publish_E{y0[0]:.1e}_S{y0[1]:.1e}.pdf', bbox_inches='tight')
            plt.close()

    # Combine all results into single dataset
    teacher_combined = np.vstack(teacher_results)
    
    print(f"Teacher pre-training dataset generated:")
    print(f"  Combined dataset shape: {teacher_combined.shape}")
    print(f"  Total time points per condition: {n_points}")
    
    # After the main simulation loop
    # Select 9 representative conditions for paper plots
    selected_indices = np.linspace(0, len(initial_conditions)-1, 9, dtype=int)
    print(f"Plotting 9 representative conditions for paper: indices {selected_indices}")

    for idx in selected_indices:
        y0 = initial_conditions[idx]
        result_data = teacher_results[idx]
        t_plot = result_data[:, 0]
        E_plot = result_data[:, 5]
        S_plot = result_data[:, 6]
        ES_plot = result_data[:, 7]
        P_plot = result_data[:, 8]

        plt.figure(figsize=(8, 6))
        colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
        linestyles = ['-', '--', '-.', ':']

        plt.semilogx(t_plot, E_plot, color=colors[0], linestyle=linestyles[0], label='E')
        plt.semilogx(t_plot, S_plot, color=colors[1], linestyle=linestyles[1], label='S')
        plt.semilogx(t_plot, ES_plot, color=colors[2], linestyle=linestyles[2], label='ES')
        plt.semilogx(t_plot, P_plot, color=colors[3], linestyle=linestyles[3], label='P')

        plt.xlabel('Time (s)')
        plt.ylabel('Concentration')
        plt.xlim(1e-8, 1e2)  # MODIFIED
        plt.ylim(-50, 1000)
        plt.margins(y=0.08)
        plt.tight_layout(pad=1.5)
        plt.legend(loc='upper right', frameon=False, ncol=2)
        plt.savefig(f'plots/teacher/paper_plot_E{y0[0]:.1e}_S{y0[1]:.1e}.pdf', bbox_inches='tight')
        plt.close()

    return teacher_combined

def generate_interpolated_conditions(teacher_conditions, expansion_factor=10):
    """
    Generate more initial conditions by interpolating between teacher conditions
    for knowledge distillation.
    
    Parameters:
    ----------
    teacher_conditions : list
        Original conditions used for teacher training
    expansion_factor : int
        How many times more conditions to generate for KD
        
    Returns:
    -------
    list
        Expanded set of initial conditions for KD
    """
    print(f"\nGenerating {expansion_factor}x more conditions for KD...")
    
    # Convert to numpy for easier manipulation
    teacher_array = np.array(teacher_conditions)
    E_range = [teacher_array[:, 0].min(), teacher_array[:, 0].max()]
    S_range = [teacher_array[:, 1].min(), teacher_array[:, 1].max()]
    
    # Generate more diverse conditions
    n_kd_conditions = len(teacher_conditions) * expansion_factor
    
    # Sample uniformly within the ranges of E and S
    E_expanded = np.random.uniform(E_range[0], E_range[1], n_kd_conditions)
    S_expanded = np.random.uniform(S_range[0], S_range[1], n_kd_conditions)
    
    kd_conditions = [[E, S, 0.0, 0.0] for E, S in zip(E_expanded, S_expanded)]
    
    print(f"Teacher E0 range: [{E_range[0]:.2e}, {E_range[1]:.2e}]")
    print(f"KD E0 range: [{min(E_expanded):.2e}, {max(E_expanded):.2e}]")
    print(f"Teacher S0 range: [{S_range[0]:.2e}, {S_range[1]:.2e}]")
    print(f"KD S0 range: [{min(S_expanded):.2e}, {max(S_expanded):.2e}]")
    
    return kd_conditions

def run_kd_simulation_from_teacher(initial_conditions, teacher_model_path, 
                                   k1=100.0, km1=10.0, k2=1.0, 
                                   plot_every=None, device='cpu'):
    """
    Generate dataset for knowledge distillation using the trained teacher model.
    Uses teacher's predictions instead of numerical ODE solver.
    
    Parameters:
    ----------
    initial_conditions : list
        List of [E, S, ES, P] initial conditions to simulate
    teacher_model_path : str
        Path to the trained teacher model checkpoint
    k1, km1, k2 : float
        Reaction rate constants (not used, kept for API compatibility)
    plot_every : int or None
        If provided, plot every nth simulation
    device : str or torch.device
        Device to run teacher model on
    
    Returns:
    -------
    np.ndarray
        Complete dataset with teacher predictions for KD
    """
    print("\n=== KNOWLEDGE DISTILLATION DATASET GENERATION (FROM TEACHER) ===")
    print("Generating dataset using trained teacher model predictions")
    
    # Create directories
    os.makedirs('plots/kd', exist_ok=True)
    os.makedirs('data/kd', exist_ok=True)
    
    # Load teacher model
    print(f"Loading teacher model from {teacher_model_path}...")
    checkpoint = torch.load(teacher_model_path, map_location=device)
    
    # Import models module to reconstruct the model
    from models import MLP, ResidualMLP
    
    # Get model architecture parameters from checkpoint
    model_type = checkpoint.get('model_type', 'MLP')
    hidden_dim = checkpoint.get('hidden_dim', 128)
    num_layers = checkpoint.get('num_layers', 3)
    dropout = checkpoint.get('dropout', 0.0)
    
    # Reconstruct model based on saved type and parameters
    if model_type == 'ResidualMLP':
        teacher_model = ResidualMLP(input_size=5, output_size=4, 
                                   hidden_dim=hidden_dim, 
                                   num_blocks=num_layers, 
                                   dropout=dropout)
    else:
        teacher_model = MLP(input_size=5, output_size=4, 
                          hidden_sizes=[hidden_dim] * num_layers, 
                          dropout=dropout)
    
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    print(f"Teacher model loaded: {model_type}")
    print(f"Architecture: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")
    
    # Load preprocessing info
    X_scaler = checkpoint['X_scaler']
    y_scaler = checkpoint.get('y_scaler', None)
    
    if y_scaler is not None:
        print("Using global y normalization")
    else:
        print("No y normalization detected")
    
    # Define single time span - MODIFIED TO 1e2
    t_span = (1e-8, 1e2)
    
    # Dense sampling for KD
    n_points = 5000

    # Generate time points
    t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), n_points)
    
    print(f"Time points: {len(t_eval)} (t ∈ [{t_span[0]}, {t_span[1]}]) - LOGSPACE")
    print(f"Total time points per condition: {len(t_eval)}")
    
    # Store results
    kd_results = []
    
    print(f"Generating predictions for {len(initial_conditions)} initial conditions using teacher...")
    
    with torch.no_grad():
        for i, y0 in enumerate(tqdm(initial_conditions)):
            # Create input matrix: [time, E0, S0, ES0, P0] for all time points
            X_batch = np.column_stack([
                t_eval,
                np.tile(y0, (len(t_eval), 1))
            ])
            
            # Preprocess inputs same way as teacher training
            X_copy = X_batch.copy()
            X_copy[:, 0] = np.log10(X_copy[:, 0] + 1e-12)  # Log transform time
            X_normalized = X_scaler.transform(X_copy)
            
            # Convert to tensor and get predictions
            X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(device)
            y_pred = teacher_model(X_tensor).cpu().numpy()
            
            # Inverse transform predictions using global scaler
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred)
            
            # Ensure non-negative concentrations
            y_pred = np.maximum(y_pred, 0)
            
            # Store data: [time, E0, S0, ES0, P0, E_pred, S_pred, ES_pred, P_pred]
            kd_data = np.column_stack([
                t_eval,
                np.tile(y0, (len(t_eval), 1)),
                y_pred
            ])
            
            kd_results.append(kd_data)
            
            # Plot if requested - UPDATE PLOT RANGE TO 1e5
            if (plot_every is not None and i % plot_every == 0) or len(initial_conditions) == 1:
                plt.figure(figsize=(8, 6))
                colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
                linestyles = ['-', '--', '-.', ':']

                # Use updated mask for 1e5
                mask = (t_eval >= 1e-8) & (t_eval <= 1e5)  # MODIFIED
                t_plot = t_eval[mask]
                E_plot = y_pred[mask, 0]
                S_plot = y_pred[mask, 1]
                ES_plot = y_pred[mask, 2]
                P_plot = y_pred[mask, 3]

                # Plot with same style as teacher
                plt.semilogx(t_plot, E_plot, color=colors[0], linestyle=linestyles[0], label='E')
                plt.semilogx(t_plot, S_plot, color=colors[1], linestyle=linestyles[1], label='S')
                plt.semilogx(t_plot, ES_plot, color=colors[2], linestyle=linestyles[2], label='ES')
                plt.semilogx(t_plot, P_plot, color=colors[3], linestyle=linestyles[3], label='P')

                plt.xlabel('Time (s)')
                plt.ylabel('Concentration')
                plt.xlim(1e-8, 1e2)  # MODIFIED
                plt.ylim(-50, 1000)
                plt.margins(y=0.08)
                plt.tight_layout(pad=1.5)
                plt.legend(loc='upper right', frameon=False, ncol=2)
                plt.savefig(f'plots/kd/kd_publish_E{y0[0]:.1e}_S{y0[1]:.1e}.pdf', bbox_inches='tight')
                plt.close()
    
    # Combine results
    kd_combined = np.vstack(kd_results)
    
    print(f"KD dataset generated from teacher:")
    print(f"  Combined dataset shape: {kd_combined.shape}")
    print(f"  Total points per condition: {len(t_eval)}")
    
    # Select 9 representative conditions for paper plots (consistent with teacher)
    selected_indices = np.linspace(0, len(initial_conditions)-1, 9, dtype=int)
    print(f"Plotting 9 representative conditions for paper: indices {selected_indices}")

    for idx in selected_indices:
        y0 = initial_conditions[idx]
        kd_data = kd_results[idx]
        t_plot = kd_data[:, 0]
        E_plot = kd_data[:, 5]
        S_plot = kd_data[:, 6]
        ES_plot = kd_data[:, 7]
        P_plot = kd_data[:, 8]

        # Apply updated mask for 1e5
        mask = (t_plot >= 1e-8) & (t_plot <= 1e5)  # MODIFIED
        t_plot = t_plot[mask]
        E_plot = E_plot[mask]
        S_plot = S_plot[mask]
        ES_plot = ES_plot[mask]
        P_plot = P_plot[mask]

        plt.figure(figsize=(8, 6))
        colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
        linestyles = ['-', '--', '-.', ':']

        plt.semilogx(t_plot, E_plot, color=colors[0], linestyle=linestyles[0], label='E')
        plt.semilogx(t_plot, S_plot, color=colors[1], linestyle=linestyles[1], label='S')
        plt.semilogx(t_plot, ES_plot, color=colors[2], linestyle=linestyles[2], label='ES')
        plt.semilogx(t_plot, P_plot, color=colors[3], linestyle=linestyles[3], label='P')

        plt.xlabel('Time (s)')
        plt.ylabel('Concentration')
        plt.xlim(1e-8, 1e2)  # MODIFIED
        plt.ylim(-50, 1000)
        plt.margins(y=0.08)
        plt.tight_layout(pad=1.5)
        plt.legend(loc='upper right', frameon=False, ncol=2)
        plt.savefig(f'plots/kd/paper_plot_E{y0[0]:.1e}_S{y0[1]:.1e}.pdf', bbox_inches='tight')
        plt.close()
    
    return kd_combined

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate datasets for teacher pre-training and knowledge distillation')
    parser.add_argument('--mode', choices=['teacher', 'kd', 'both'], default='both',
                       help='Dataset generation mode: teacher (pre-training), kd (knowledge distillation), or both')
    parser.add_argument('--single', action='store_true', help='Run with a single initial condition')
    parser.add_argument('--E0', type=float, default=1e3, help='Initial enzyme concentration')
    parser.add_argument('--S0', type=float, default=1e3, help='Initial substrate concentration')
    parser.add_argument('--num_samples', type=int, default=50, 
                       help='Number of different initial conditions for multiple simulations')
    parser.add_argument('--plot_every', type=int, default=5, 
                       help='Plot every N simulations when running multiple')
    parser.add_argument('--verify', action='store_true', help='Verify ODE solvers for consistency')
    parser.add_argument('--teacher_model', type=str, default=None,
                       help='Path to trained teacher model for KD data generation (required for KD mode)')
    args = parser.parse_args()

    # Parameters for the reaction
    k1 = 100.0
    km1 = 10.0
    k2 = 1.0
    
    if args.single:
        # Single initial condition
        print(f"Running single simulation with E0={args.E0}, S0={args.S0}")
        initial_conditions = [[args.E0, args.S0, 0.0, 0.0]]
    else:
        # Multiple initial conditions with random sampling
        # Sample 0.5-1 times the single situation values equally
        print(f"Generating {args.num_samples} random initial conditions")
        print(f"E0 range: [0.5 × {args.E0:.8e}, 1.0 × {args.E0:.8e}] = [{0.5*args.E0:.8e}, {args.E0:.8e}]")
        print(f"S0 range: [0.5 × {args.S0:.8e}, 1.0 × {args.S0:.8e}] = [{0.5*args.S0:.8e}, {args.S0:.8e}]")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate random multipliers uniformly distributed between 0.5 and 1.0
        E_multipliers = np.random.uniform(0.5, 1.0, args.num_samples)
        S_multipliers = np.random.uniform(0.5, 1.0, args.num_samples)

        # Generate all initial conditions
        initial_conditions = []
        for i in range(args.num_samples):
            E_init = E_multipliers[i] * args.E0
            S_init = S_multipliers[i] * args.S0
            initial_conditions.append([E_init, S_init, 0.0, 0.0])
        
        print(f"Generated {len(initial_conditions)} different initial conditions")
        print(f"E0 actual range: [{min(ic[0] for ic in initial_conditions):.8e}, {max(ic[0] for ic in initial_conditions):.8e}]")
        print(f"S0 actual range: [{min(ic[1] for ic in initial_conditions):.8e}, {max(ic[1] for ic in initial_conditions):.8e}]")

    # Run solver verification if requested
    if args.verify:
        short_results, long_results = verify_ode_solvers(initial_conditions[0], k1, km1, k2)
    
    # Generate file suffix
    if args.single:
        file_suffix = f"_single_E{args.E0:.6e}_S{args.S0:.6e}"
    else:
        file_suffix = f"_multiple_{len(initial_conditions)}_conditions"
    
    # Generate datasets based on mode
    if args.mode in ['teacher', 'both']:
        print("\n" + "="*60)
        print("GENERATING TEACHER PRE-TRAINING DATASET")
        print("="*60)
        
        teacher_data = run_teacher_pretraining_simulation(
            initial_conditions, k1, km1, k2,
            plot_every=None if args.single else args.plot_every
        )
        
        # Save teacher dataset (no longer needs phase indicator)
        header_teacher = 'time,E0,S0,ES0,P0,E,S,ES,P'
        
        np.save(f'data/teacher/teacher_combined{file_suffix}.npy', teacher_data)
        np.savetxt(f'data/teacher/teacher_combined{file_suffix}.csv', teacher_data,
                   delimiter=',', header=header_teacher, comments='')
        
        print(f"\nTeacher pre-training dataset saved:")
        print(f"  Combined (fast + slow) dataset: {teacher_data.shape}")
    
    if args.mode in ['kd', 'both']:
        print("\n" + "="*60)
        print("GENERATING KNOWLEDGE DISTILLATION DATASET")
        print("="*60)
        
        # Generate more diverse conditions for KD
        if args.mode == 'both':
            kd_conditions = generate_interpolated_conditions(initial_conditions, expansion_factor=5)
        else:
            kd_conditions = initial_conditions
        
        # Verify teacher model is provided
        if not args.teacher_model:
            raise ValueError("KD data generation requires --teacher_model path. "
                           "Please provide a trained teacher model.")
        
        # Use teacher model to generate KD data
        print("Using trained teacher model for KD data generation...")
        
        # Select device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        print(f"Using device: {device}")
        
        kd_data = run_kd_simulation_from_teacher(
            kd_conditions, args.teacher_model,
            k1=k1, km1=km1, k2=k2,
            plot_every=None if args.single else args.plot_every,
            device=device
        )
        
        # Save KD dataset (complete time range)
        header_kd = 'time,E0,S0,ES0,P0,E,S,ES,P'
        
        np.save(f'data/kd/kd_complete{file_suffix}.npy', kd_data)
        np.savetxt(f'data/kd/kd_complete{file_suffix}.csv', kd_data,
                   delimiter=',', header=header_kd, comments='')
        
        print(f"\nKD dataset saved:")
        print(f"  Complete time range [0, 50]: {kd_data.shape}")

    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Generated datasets for {len(initial_conditions)} initial conditions")
    
    if args.mode in ['teacher', 'both']:
        print("Teacher dataset: Combined fast + slow phases for comprehensive learning")
    if args.mode in ['kd', 'both']:
        print("KD dataset: Dense sampling across complete time range [0, 50]")

    print("All datasets saved for deep learning usage.")

if __name__ == "__main__":
    main()