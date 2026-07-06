import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import argparse
from scipy.integrate import solve_ivp
from tqdm import tqdm
from models import MLP, ResidualMLP
import matplotlib as mpl

# Configure matplotlib for publication quality (NC-level, subplot-ready)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Roman']
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titlesize'] = 17
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['legend.fontsize'] = 13
mpl.rcParams['figure.titlesize'] = 17
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

# Species names for MM reaction
SPECIES_NAMES = ['E', 'S', 'ES', 'P']

def mm_reaction(t, y, k1, km1, k2):
    E, S, ES, P = y
    dE_dt = -k1 * E * S + km1 * ES + k2 * ES
    dS_dt = -k1 * E * S + km1 * ES
    dES_dt = k1 * E * S - km1 * ES - k2 * ES
    dP_dt = k2 * ES
    return [dE_dt, dS_dt, dES_dt, dP_dt]

def generate_ground_truth(initial_conditions, time_points, k1=100.0, km1=10.0, k2=1.0):
    results = []
    for ic in tqdm(initial_conditions, desc="Generating ground truth"):
        solution = solve_ivp(
            fun=lambda t, y: mm_reaction(t, y, k1, km1, k2),
            t_span=(time_points[0], time_points[-1]),
            y0=ic,
            method='BDF',
            t_eval=time_points,
            atol=1e-10,
            rtol=1e-8
        )
        results.append(solution.y.T)
    return np.array(results)

class TeacherModelEvaluator:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.load_model(model_path)
    
    def __getattr__(self, name):
        if self.model is not None and not name.startswith('_'):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def load_model(self, model_path):
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_type = checkpoint.get('model_type', 'MLP')
        
        # Get model architecture parameters
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_layers = checkpoint.get('num_layers', 3)
        dropout = checkpoint.get('dropout', 0.0)
        
        # Support MLP and ResidualMLP only
        if model_type == 'ResidualMLP':
            self.model = ResidualMLP(input_size=5, output_size=4, 
                                    hidden_dim=hidden_dim, 
                                    num_blocks=num_layers,
                                    dropout=dropout)
        else:
            self.model = MLP(input_size=5, output_size=4, 
                           hidden_sizes=[hidden_dim] * num_layers,
                           dropout=dropout)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessing info
        self.X_scaler = checkpoint['X_scaler']
        self.y_scaler = checkpoint.get('y_scaler', None)
        
        print(f"Model loaded successfully: {model_type}")
        print(f"Architecture: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")
        
        if self.y_scaler is not None:
            print(f"Using global y normalization")
        else:
            print(f"No y normalization detected")

    def predict(self, X):
        """Predict concentrations for given inputs"""
        X_copy = X.copy()
        X_copy[:, 0] = np.log10(X_copy[:, 0] + 1e-12)  # Log transform time
        X_norm = self.X_scaler.transform(X_copy)
        
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        if self.y_scaler is not None:
            predictions = self.y_scaler.inverse_transform(predictions)
        
        return predictions

def generate_many_test_conditions(num_conditions=50, E0_base=1e3, S0_base=1e3):
    """Generate test conditions with same range as training data"""
    np.random.seed(42)
    
    E_multipliers = np.random.uniform(0.5, 1.0, num_conditions)
    S_multipliers = np.random.uniform(0.5, 1.0, num_conditions)
    
    conditions = []
    for i in range(num_conditions):
        E0 = E_multipliers[i] * E0_base
        S0 = S_multipliers[i] * S0_base
        conditions.append([E0, S0, 0.0, 0.0])
    
    print(f"Generated {num_conditions} test conditions:")
    print(f"  E0 range: [{0.5*E0_base:.2e}, {1.0*E0_base:.2e}]")
    print(f"  S0 range: [{0.5*S0_base:.2e}, {1.0*S0_base:.2e}]")
    
    return np.array(conditions)

def evaluate_model(evaluator, test_conditions, time_points, k1=100.0, km1=10.0, k2=1.0):
    print("\n" + "="*60)
    print("EVALUATING PERFORMANCE")
    print("="*60)
    
    ground_truth = generate_ground_truth(test_conditions, time_points, k1, km1, k2)
    n_conditions, n_times = len(test_conditions), len(time_points)
    
    # Create input array
    X_test = np.array([[t] + list(ic) for ic in test_conditions for t in time_points])
    
    # Get predictions
    predictions = evaluator.predict(X_test).reshape(n_conditions, n_times, 4)
    
    results = {
        'ground_truth': ground_truth,
        'predictions': predictions,
        'mae_overall': mean_absolute_error(ground_truth.flatten(), predictions.flatten()),
        'rmse_overall': np.sqrt(mean_squared_error(ground_truth.flatten(), predictions.flatten())),
    }
    
    # Compute relative error
    gt_flat = ground_truth.flatten()
    pred_flat = predictions.flatten()
    relative_error = np.mean(np.abs(gt_flat - pred_flat) / (np.abs(gt_flat) + 1e-12))
    results['relative_error_overall'] = relative_error
    
    species_errors = {}
    for i, species in enumerate(SPECIES_NAMES):
        gt_species = ground_truth[:, :, i].flatten()
        pred_species = predictions[:, :, i].flatten()
        species_errors[species] = {
            'MAE': mean_absolute_error(gt_species, pred_species),
            'RMSE': np.sqrt(mean_squared_error(gt_species, pred_species)),
            'R2': r2_score(gt_species, pred_species),
            'Relative_Error': np.mean(np.abs(gt_species - pred_species) / (np.abs(gt_species) + 1e-12))
        }
    results['species_errors'] = species_errors
    
    print(f"\nRESULTS:")
    print(f"  Overall MAE: {results['mae_overall']:.4e}")
    print(f"  Overall RMSE: {results['rmse_overall']:.4e}")
    print(f"  Overall Relative Error: {results['relative_error_overall']:.4f}")
    for species, errors in results['species_errors'].items():
        print(f"  {species}: MAE={errors['MAE']:.4e}, RMSE={errors['RMSE']:.4e}, R²={errors['R2']:.4f}")
    
    return results

def analyze_error_over_time(evaluator, test_conditions, time_points, k1=100.0, km1=10.0, k2=1.0):
    ground_truth = generate_ground_truth(test_conditions, time_points, k1, km1, k2)
    n_conditions, n_times = len(test_conditions), len(time_points)
    
    # Create input array
    X_test = np.array([[t] + list(ic) for ic in test_conditions for t in time_points])
    
    # Get predictions
    predictions = evaluator.predict(X_test).reshape(n_conditions, n_times, 4)
    
    time_errors = {
        'mae_by_time': np.zeros(n_times),
        'rmse_by_time': np.zeros(n_times),
        'mae_by_time_by_species': np.zeros((n_times, 4)),
        'rmse_by_time_by_species': np.zeros((n_times, 4)),
        'relative_error_by_time': np.zeros(n_times),
        'r2_by_time': np.zeros(n_times)
    }
    
    for t_idx in range(n_times):
        gt_t = ground_truth[:, t_idx, :].flatten()
        pred_t = predictions[:, t_idx, :].flatten()
        time_errors['mae_by_time'][t_idx] = mean_absolute_error(gt_t, pred_t)
        time_errors['rmse_by_time'][t_idx] = np.sqrt(mean_squared_error(gt_t, pred_t))
        time_errors['relative_error_by_time'][t_idx] = np.mean(np.abs(gt_t - pred_t) / (np.abs(gt_t) + 1e-12))
        time_errors['r2_by_time'][t_idx] = r2_score(gt_t, pred_t)
        
        for s_idx in range(4):
            gt_species_t = ground_truth[:, t_idx, s_idx]
            pred_species_t = predictions[:, t_idx, s_idx]
            time_errors['mae_by_time_by_species'][t_idx, s_idx] = mean_absolute_error(gt_species_t, pred_species_t)
            time_errors['rmse_by_time_by_species'][t_idx, s_idx] = np.sqrt(mean_squared_error(gt_species_t, pred_species_t))
    
    condition_errors = {
        'mae_by_condition': np.zeros(n_conditions),
        'rmse_by_condition': np.zeros(n_conditions),
        'r2_by_condition': np.zeros(n_conditions),
        'relative_error_by_condition': np.zeros(n_conditions)
    }
    
    for c_idx in range(n_conditions):
        gt_c = ground_truth[c_idx, :, :].flatten()
        pred_c = predictions[c_idx, :, :].flatten()
        condition_errors['mae_by_condition'][c_idx] = mean_absolute_error(gt_c, pred_c)
        condition_errors['rmse_by_condition'][c_idx] = np.sqrt(mean_squared_error(gt_c, pred_c))
        condition_errors['r2_by_condition'][c_idx] = r2_score(gt_c, pred_c)
        condition_errors['relative_error_by_condition'][c_idx] = np.mean(np.abs(gt_c - pred_c) / (np.abs(gt_c) + 1e-12))
    
    return {
        'time_errors': time_errors,
        'condition_errors': condition_errors,
        'ground_truth': ground_truth,
        'predictions': predictions,
        'test_conditions': test_conditions,
        'time_points': time_points
    }

def plot_trajectories(results, analysis, model_name, output_dir='results/teacher_evaluation'):
    """Create publication-quality trajectory plots matching reference style"""
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    time_points = analysis['time_points']
    test_conditions = analysis['test_conditions']
    
    np.random.seed(42)
    num_conditions = len(test_conditions)
    num_to_plot = min(3, num_conditions)
    if num_conditions <= 3:
        representative_indices = list(range(num_conditions))
    else:
        representative_indices = np.random.choice(num_conditions, size=num_to_plot, replace=False)
    
    ground_truth = results['ground_truth']
    predictions = results['predictions']
    
    colors = {
        'ground_truth': '#000000',
        'prediction': '#E84A27',  # Coral/orange-red like in reference
    }
    
    # Subsample for discrete ground truth markers - use LINEAR spacing in log-space index
    # This gives more evenly distributed points on a log-scale x-axis
    n_markers = 20  # Number of discrete points for ground truth
    marker_indices = np.unique(np.round(np.linspace(0, len(time_points)-1, n_markers)).astype(int))
    marker_indices = np.clip(marker_indices, 0, len(time_points)-1)

    # Compute global y-axis limits for each species across all plotted conditions
    y_limits = []
    for col in range(4):
        all_gt = np.concatenate([ground_truth[idx, :, col] for idx in representative_indices])
        all_pred = np.concatenate([predictions[idx, :, col] for idx in representative_indices])
        all_data = np.concatenate([all_gt, all_pred])
        y_min, y_max = np.min(all_data), np.max(all_data)
        y_range = y_max - y_min if y_max - y_min > 1e-10 else 1.0
        y_limits.append((y_min - 0.1 * y_range, y_max + 0.1 * y_range))

    # Figure 1: Complete trajectories - subplot-ready style
    fig, axes = plt.subplots(num_to_plot, 4, figsize=(10, 2.4 * num_to_plot),
                             sharex=True, sharey='col')

    if num_to_plot == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(representative_indices):
        for col, species in enumerate(SPECIES_NAMES):
            ax = axes[row, col]

            # Ground truth as discrete star markers
            ax.semilogx(time_points[marker_indices], ground_truth[idx, marker_indices, col],
                       marker='*', markersize=8, color=colors['ground_truth'],
                       linestyle='none', markeredgewidth=0.8, markeredgecolor=colors['ground_truth'],
                       markerfacecolor=colors['ground_truth'], zorder=5,
                       label='Ground truth' if row == 0 and col == 0 else "")

            # Prediction as bold solid line
            ax.semilogx(time_points, predictions[idx, :, col],
                       color=colors['prediction'], linewidth=2.5,
                       label='Prediction' if row == 0 and col == 0 else "")

            # Column header (species name) only on top row
            if row == 0:
                ax.set_title(f'${species}$', fontsize=16, fontweight='bold', pad=4)

            # x-label only on bottom row
            if row == num_to_plot - 1:
                ax.set_xlabel(r'$t$ (s)', fontsize=15, fontweight='bold')

            # y-label only on leftmost column, as condition index
            if col == 0:
                ax.set_ylabel(f'Cond. {row+1}', fontsize=14, fontweight='bold', labelpad=3)

            ax.set_ylim(y_limits[col])
            ax.set_xlim(time_points[0], time_points[-1])
            ax.tick_params(axis='both', which='major', labelsize=12, length=4)
            ax.tick_params(axis='both', which='minor', length=2)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.4)

            # Suppress redundant tick labels on shared axes
            if row < num_to_plot - 1:
                ax.tick_params(labelbottom=False)
            if col > 0:
                ax.tick_params(labelleft=False)

    # Single compact legend outside the grid
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=13,
               frameon=True, fancybox=False, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.08), borderpad=0.5, handlelength=1.8)

    plt.tight_layout(pad=0.8, h_pad=0.4, w_pad=0.4)
    plt.savefig(f'{model_output_dir}/trajectories_complete.pdf', bbox_inches='tight')
    plt.savefig(f'{model_output_dir}/trajectories_complete.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Stacked subplots style (like reference image exactly)
    for idx in representative_indices[:1]:  # Just first condition for this style
        fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        
        for col, (ax, species) in enumerate(zip(axes, SPECIES_NAMES)):
            # Ground truth as discrete open circles - evenly distributed
            ax.semilogx(time_points[marker_indices], ground_truth[idx, marker_indices, col],
                       marker='o', markersize=8, color=colors['ground_truth'],
                       linestyle='none', markeredgewidth=1.8, markeredgecolor=colors['ground_truth'],
                       markerfacecolor='white', zorder=5,
                       label='Ground Truth' if col == 0 else "")
            
            # Prediction as solid line
            ax.semilogx(time_points, predictions[idx, :, col], 
                       color=colors['prediction'], linewidth=2.2,
                       label='Prediction' if col == 0 else "")
            
            ax.set_ylabel(f'$y_{col+1}$', fontsize=16, fontweight='bold')

            # Use global y-axis limits for this species
            ax.set_ylim(y_limits[col])

            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.grid(False)

            if col == 0:
                ax.legend(frameon=True, loc='upper right', fontsize=13)

        axes[-1].set_xlabel(r'$t$ [sec]', fontsize=16, fontweight='bold')
        axes[-1].set_xlim(time_points[0], time_points[-1])
        
        plt.tight_layout()
        plt.savefig(f'{model_output_dir}/trajectories_stacked.pdf')
        plt.savefig(f'{model_output_dir}/trajectories_stacked.png', dpi=300)
        plt.close()

    # Figure 3: Fast dynamics (0-1 ms) - same style
    fast_mask = time_points <= 1e-3
    if np.any(fast_mask):
        fast_time_points = time_points[fast_mask]
        n_fast = len(fast_time_points)
        
        # Evenly distributed markers for fast dynamics
        n_fast_markers = min(25, n_fast // 2)
        fast_marker_indices = np.unique(np.linspace(0, n_fast-1, n_fast_markers, dtype=int))
        
        # Compute global y-axis limits for fast dynamics
        y_limits_fast = []
        for col in range(4):
            all_gt = np.concatenate([ground_truth[idx, :n_fast, col] for idx in representative_indices])
            all_pred = np.concatenate([predictions[idx, :n_fast, col] for idx in representative_indices])
            all_data = np.concatenate([all_gt, all_pred])
            y_min, y_max = np.min(all_data), np.max(all_data)
            y_range = y_max - y_min if y_max - y_min > 1e-10 else 1.0
            y_limits_fast.append((y_min - 0.1 * y_range, y_max + 0.1 * y_range))
        
        fig, axes = plt.subplots(num_to_plot, 4, figsize=(16, 3.5 * num_to_plot))
        
        if num_to_plot == 1:
            axes = axes.reshape(1, -1)
        
        for row, idx in enumerate(representative_indices):
            for col, species in enumerate(SPECIES_NAMES):
                ax = axes[row, col]
                
                # Ground truth as discrete open circles - evenly distributed
                ax.plot(fast_time_points[fast_marker_indices] * 1000, 
                       ground_truth[idx, fast_marker_indices, col],
                       marker='o', markersize=7, color=colors['ground_truth'],
                       linestyle='none', markeredgewidth=1.5, markeredgecolor=colors['ground_truth'],
                       markerfacecolor='white', zorder=5,
                       label='Ground Truth' if row == 0 and col == 0 else "")
                
                # Prediction as solid line
                ax.plot(fast_time_points * 1000, predictions[idx, :n_fast, col], 
                       color=colors['prediction'], linewidth=2.0,
                       label='Prediction' if row == 0 and col == 0 else "")
                
                if row == num_to_plot - 1:
                    ax.set_xlabel('Time (ms)', fontsize=15, fontweight='bold')

                ax.set_ylabel(f'${species}$', fontsize=16, fontweight='bold')

                # Use global y-axis limits for fast dynamics
                ax.set_ylim(y_limits_fast[col])

                if row == 0 and col == 0:
                    ax.legend(frameon=True, fancybox=False, shadow=False,
                             loc='best', fontsize=13, framealpha=0.9)

                ax.set_xlim(0, fast_time_points[-1] * 1000)
                ax.tick_params(axis='both', which='major', labelsize=13)
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                
        plt.tight_layout(pad=1.5)
        plt.savefig(f'{model_output_dir}/trajectories_fast.pdf')
        plt.savefig(f'{model_output_dir}/trajectories_fast.png', dpi=300)
        plt.close()
    
    print(f"Trajectory plots saved to {model_output_dir}")

def plot_error_analysis(analysis, model_name, output_dir='results/teacher_evaluation'):
    """
    Publication-quality error analysis visualization:
      1) Error over time (RMSE/MAE)
      2) Error by species (bar chart) - NEW
      3) Quantile band + tail index over time
      4) Phase-aligned error
      5) Condition difficulty ranking
      6) Phase portrait with error vectors
      7) Error heatmap over time and species
    """
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'serif'

    time_points = analysis['time_points']
    gt = analysis['ground_truth']
    pred = analysis['predictions']
    conds = analysis['test_conditions']

    eps = 1e-12
    C, T, D = gt.shape
    assert D == 4

    # Handle single condition case
    if C == 1:
        print("  Single condition detected - generating simplified error analysis plots...")
        
        diff = pred - gt
        abs_error = np.abs(diff[0])
        rmse_by_time = np.sqrt(np.mean(diff[0]**2, axis=1))
        mae_by_time = np.mean(np.abs(diff[0]), axis=1)
        
        # Figure 1: Error over time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(rf'{model_name}: Error over time (single condition)', fontsize=13, fontweight='bold')
        
        ax.loglog(time_points, rmse_by_time + eps, linewidth=2.5, label=r'$\mathrm{RMSE}$')
        ax.loglog(time_points, mae_by_time + eps, linewidth=2.0, linestyle='--', label=r'$\mathrm{MAE}$')
        
        ax.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'Error', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        
        plt.tight_layout()
        plt.savefig(f'{model_output_dir}/error_over_time.pdf')
        plt.savefig(f'{model_output_dir}/error_over_time.png', dpi=300)
        plt.close()
        
        # Figure 2: Error by species (bar chart)
        species_mae = np.mean(abs_error, axis=0)
        species_rmse = np.sqrt(np.mean(diff[0]**2, axis=0))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(4)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, species_mae + eps, width, label='MAE', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, species_rmse + eps, width, label='RMSE', color='#d62728', alpha=0.8)
        
        ax.set_xlabel('Species', fontsize=14, fontweight='bold')
        ax.set_ylabel('Error', fontsize=14, fontweight='bold')
        ax.set_title(rf'{model_name}: Error by species', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(SPECIES_NAMES, fontsize=13, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.25, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{model_output_dir}/error_by_species.pdf')
        plt.savefig(f'{model_output_dir}/error_by_species.png', dpi=300)
        plt.close()
        
        # Figure 3: Phase portrait (S vs P)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        Sg, Pg = gt[0, :, 1], gt[0, :, 3]
        Sp, Pp = pred[0, :, 1], pred[0, :, 3]
        
        ax.plot(Sg, Pg, linewidth=2.2, label=r'Ground Truth', color='#000000')
        ax.plot(Sp, Pp, linewidth=2.0, linestyle='--', label=r'Prediction', color='#E31A1C')
        
        arrow_n = 25
        arrow_idx = np.unique(np.round(np.logspace(0, np.log10(T-1), arrow_n)).astype(int))
        arrow_idx = np.clip(arrow_idx, 0, T-1)
        
        dx = (Sp[arrow_idx] - Sg[arrow_idx])
        dy = (Pp[arrow_idx] - Pg[arrow_idx])
        
        if np.any(np.abs(dx) > eps) or np.any(np.abs(dy) > eps):
            ax.quiver(Sg[arrow_idx], Pg[arrow_idx], dx, dy,
                      angles='xy', scale_units='xy', scale=1.0, width=0.004, alpha=0.6)
        
        ax.set_title(rf'Phase portrait $(S, P)$', fontsize=13, fontweight='bold')
        ax.set_xlabel(r'$S$', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'$P$', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=12, frameon=True)
        
        plt.tight_layout()
        plt.savefig(f'{model_output_dir}/error_phase_portrait_vectors.pdf')
        plt.savefig(f'{model_output_dir}/error_phase_portrait_vectors.png', dpi=300)
        plt.close()
        
        # Figure 4: Error heatmap
        fig, ax = plt.subplots(figsize=(7, 3.5))

        log_error = np.log10(abs_error.T + eps)
        vmin, vmax = np.nanpercentile(log_error, 2), np.nanpercentile(log_error, 98)

        im = ax.imshow(log_error, aspect='auto', cmap='RdYlBu_r',
                       vmin=vmin, vmax=vmax,
                       extent=[np.log10(time_points[0]), np.log10(time_points[-1]), 3.5, -0.5],
                       interpolation='nearest')

        # Add contour lines so structure is visible at small sizes
        t_log = np.linspace(np.log10(time_points[0]), np.log10(time_points[-1]), log_error.shape[1])
        y_pos = np.linspace(-0.5, 3.5, log_error.shape[0])
        CS = ax.contour(t_log, y_pos, log_error, levels=5, colors='white',
                        linewidths=0.6, alpha=0.5)

        # Replace numeric x-ticks with actual time values
        t_ticks_log = np.arange(np.ceil(np.log10(time_points[0])),
                                np.floor(np.log10(time_points[-1])) + 1, 2).astype(int)
        ax.set_xticks(t_ticks_log)
        ax.set_xticklabels([rf'$10^{{{v}}}$' for v in t_ticks_log], fontsize=9)

        ax.set_xlabel(r'$t$ (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Species', fontsize=11, fontweight='bold')
        ax.set_yticks(range(4))
        ax.set_yticklabels([r'$E$', r'$S$', r'$ES$', r'$P$'], fontsize=11)
        ax.set_title(r'$\log_{10}|\mathrm{error}|$', fontsize=11, fontweight='bold', pad=4)

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(r'$\log_{10}|\varepsilon|$', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        plt.savefig(f'{model_output_dir}/error_heatmap.pdf')
        plt.savefig(f'{model_output_dir}/error_heatmap.png', dpi=300)
        plt.close()
        
        print(f"Error analysis plots (single condition) saved to {model_output_dir}")
        return

    # -----------------------------
    # Multiple conditions case
    # -----------------------------
    
    # Sort conditions by S0/E0 ratio
    ratio = conds[:, 1] / (conds[:, 0] + eps)
    order = np.argsort(ratio)
    gt = gt[order]
    pred = pred[order]
    ratio = ratio[order]
    conds = conds[order]

    diff = pred - gt
    rmse_ct = np.sqrt(np.mean(diff**2, axis=2))
    mae_ct = np.mean(np.abs(diff), axis=2)

    qs = [0.1, 0.5, 0.9]
    rmse_q = np.quantile(rmse_ct, qs, axis=0)
    mae_q = np.quantile(mae_ct, qs, axis=0)
    tail_rmse = (rmse_q[2] + eps) / (rmse_q[1] + eps)

    # Figure 1: Quantile band + tail index
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(rf'{model_name}: Error distribution over time', fontsize=13, fontweight='bold')

    rmse_q_safe = np.maximum(rmse_q, eps)
    mae_q_safe = np.maximum(mae_q, eps)

    ax.loglog(time_points, rmse_q_safe[1], linewidth=2.5, label=r'$\mathrm{RMSE}$ median')
    ax.fill_between(time_points, rmse_q_safe[0], rmse_q_safe[2], alpha=0.25, label=r'$\mathrm{RMSE}$ 10–90%')

    ax.loglog(time_points, mae_q_safe[1], linewidth=2.0, linestyle='--', label=r'$\mathrm{MAE}$ median')
    ax.fill_between(time_points, mae_q_safe[0], mae_q_safe[2], alpha=0.15, label=r'$\mathrm{MAE}$ 10–90%')

    ax.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Error', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.semilogx(time_points, tail_rmse, linewidth=1.8, linestyle=':', color='green',
                 label=r'Tail index $q_{0.9}/q_{0.5}$')
    ax2.set_ylabel(r'$q_{0.9}/q_{0.5}$', fontsize=11, fontweight='bold')
    ax2.tick_params(labelsize=9)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/error_quantile_tail.pdf')
    plt.savefig(f'{model_output_dir}/error_quantile_tail.png', dpi=300)
    plt.close()

    # Figure 2: Error by species (bar chart) - NEW
    species_mae = np.zeros(4)
    species_rmse = np.zeros(4)
    species_mae_std = np.zeros(4)
    species_rmse_std = np.zeros(4)
    
    for i in range(4):
        abs_err = np.abs(diff[:, :, i])
        species_mae[i] = np.mean(abs_err)
        species_mae_std[i] = np.std(np.mean(abs_err, axis=1))
        species_rmse[i] = np.sqrt(np.mean(diff[:, :, i]**2))
        species_rmse_std[i] = np.std(np.sqrt(np.mean(diff[:, :, i]**2, axis=1)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, species_mae + eps, width, yerr=species_mae_std,
                   label='MAE', color='#1f77b4', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, species_rmse + eps, width, yerr=species_rmse_std,
                   label='RMSE', color='#d62728', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Species', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error', fontsize=14, fontweight='bold')
    ax.set_title(rf'{model_name}: Error by species (across {C} conditions)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SPECIES_NAMES, fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.25, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/error_by_species.pdf')
    plt.savefig(f'{model_output_dir}/error_by_species.png', dpi=300)
    plt.close()

    # Figure 3: Phase-aligned error
    P_gt = gt[:, :, 3]
    P_final = P_gt[:, -1] + eps
    target = 0.5 * P_final

    t_star = np.full(C, time_points[-1])
    for i in range(C):
        idx = np.where(P_gt[i] >= target[i])[0]
        if len(idx) > 0:
            t_star[i] = time_points[idx[0]]

    tau = np.log10((time_points[None, :] + eps) / (t_star[:, None] + eps))
    rmse = rmse_ct

    tau_min, tau_max = -4.0, 4.0
    n_bins = 120
    tau_edges = np.linspace(tau_min, tau_max, n_bins + 1)
    tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])

    rmse_tau_q = np.zeros((3, n_bins))
    for b in range(n_bins):
        mask = (tau >= tau_edges[b]) & (tau < tau_edges[b+1])
        vals = rmse[mask]
        if vals.size < 20:
            rmse_tau_q[:, b] = np.nan
        else:
            rmse_tau_q[:, b] = np.quantile(vals, qs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(r'Phase-aligned error: $\tau=\log_{10}(t/t_{1/2})$', fontsize=13, fontweight='bold')

    ax.plot(tau_centers, rmse_tau_q[1], linewidth=2.5, label=r'$\mathrm{RMSE}$ median')
    ax.fill_between(tau_centers, rmse_tau_q[0], rmse_tau_q[2], alpha=0.25, label=r'10–90%')
    ax.axvline(0.0, linewidth=1.5, linestyle='--', color='gray', alpha=0.7)
    ax.text(0.02, 0.92, r'$\tau=0$ at $t=t_{1/2}$', transform=ax.transAxes, fontsize=10)

    ax.set_xlabel(r'$\tau=\log_{10}(t/t_{1/2})$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\mathrm{RMSE}$', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/error_phase_aligned.pdf')
    plt.savefig(f'{model_output_dir}/error_phase_aligned.png', dpi=300)
    plt.close()

    # Figure 4: Condition difficulty ranking
    difficulty = np.median(rmse_ct, axis=1)
    x_log = np.log10(ratio + eps)
    y_log = np.log10(difficulty + eps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    idx_sorted = np.argsort(difficulty)
    ax.plot(np.arange(C), difficulty[idx_sorted] + eps, linewidth=2.5)
    ax.set_yscale('log')
    ax.set_title(r'Condition difficulty ranking', fontsize=13, fontweight='bold')
    ax.set_xlabel(r'Rank (easy $\rightarrow$ hard)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Median $\mathrm{RMSE}$ over time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.scatter(x_log, y_log, s=30, alpha=0.75, c='#1f77b4')
    ax.set_title(r'What explains difficulty? (proxy: $S_0/E_0$)', fontsize=13, fontweight='bold')
    ax.set_xlabel(r'$\log_{10}(S_0/E_0)$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(\mathrm{median\ RMSE})$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/error_condition_difficulty.pdf')
    plt.savefig(f'{model_output_dir}/error_condition_difficulty.png', dpi=300)
    plt.close()

    # Figure 5: Phase portrait with error vectors
    rep = [idx_sorted[0], idx_sorted[C//2], idx_sorted[-1]]
    labels_plot = [r'Easy', r'Median', r'Hard']

    arrow_n = 35
    arrow_idx = np.unique(np.round(np.logspace(0, np.log10(T-1), arrow_n)).astype(int))
    arrow_idx = np.clip(arrow_idx, 0, T-1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, ci, lab in zip(axes, rep, labels_plot):
        Sg, Pg = gt[ci, :, 1], gt[ci, :, 3]
        Sp, Pp = pred[ci, :, 1], pred[ci, :, 3]

        ax.plot(Sg, Pg, linewidth=2.2, label=r'Ground Truth', color='#000000')
        ax.plot(Sp, Pp, linewidth=2.0, linestyle='--', label=r'Prediction', color='#E31A1C')

        dx = (Sp[arrow_idx] - Sg[arrow_idx])
        dy = (Pp[arrow_idx] - Pg[arrow_idx])
        ax.quiver(Sg[arrow_idx], Pg[arrow_idx], dx, dy,
                  angles='xy', scale_units='xy', scale=1.0, width=0.003, alpha=0.6)

        ax.set_title(rf'Phase portrait $(S, P)$ — {lab}', fontsize=13, fontweight='bold')
        ax.set_xlabel(r'$S$', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$P$', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)

        if ax is axes[0]:
            ax.legend(fontsize=10, frameon=True)

    plt.savefig(f'{model_output_dir}/error_phase_portrait_vectors.pdf')
    plt.savefig(f'{model_output_dir}/error_phase_portrait_vectors.png', dpi=300)
    plt.close()

    # Figure 6: Error heatmap (averaged across conditions)
    fig, ax = plt.subplots(figsize=(7, 3.5))

    mean_abs_error = np.mean(np.abs(diff), axis=0).T  # (4, T)
    log_error = np.log10(mean_abs_error + eps)
    vmin, vmax = np.nanpercentile(log_error, 2), np.nanpercentile(log_error, 98)

    im = ax.imshow(log_error, aspect='auto', cmap='RdYlBu_r',
                   vmin=vmin, vmax=vmax,
                   extent=[np.log10(time_points[0]), np.log10(time_points[-1]), 3.5, -0.5],
                   interpolation='nearest')

    # Add contour lines so structure is visible at small sizes
    t_log = np.linspace(np.log10(time_points[0]), np.log10(time_points[-1]), log_error.shape[1])
    y_pos = np.linspace(-0.5, 3.5, log_error.shape[0])
    ax.contour(t_log, y_pos, log_error, levels=5, colors='white',
               linewidths=0.6, alpha=0.5)

    # Replace numeric x-ticks with actual time values
    t_ticks_log = np.arange(np.ceil(np.log10(time_points[0])),
                            np.floor(np.log10(time_points[-1])) + 1, 2).astype(int)
    ax.set_xticks(t_ticks_log)
    ax.set_xticklabels([rf'$10^{{{v}}}$' for v in t_ticks_log], fontsize=9)

    ax.set_xlabel(r'$t$ (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Species', fontsize=11, fontweight='bold')
    ax.set_yticks(range(4))
    ax.set_yticklabels([r'$E$', r'$S$', r'$ES$', r'$P$'], fontsize=11)
    ax.set_title(r'Mean $\log_{10}|\mathrm{error}|$', fontsize=11, fontweight='bold', pad=4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r'$\log_{10}|\varepsilon|$', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/error_heatmap.pdf')
    plt.savefig(f'{model_output_dir}/error_heatmap.png', dpi=300)
    plt.close()

    print(f"Error analysis plots saved to {model_output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of teacher model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--num_test_conditions', type=int, default=100, help='Number of test conditions')
    parser.add_argument('--E0', type=float, default=1e3, help='Base enzyme concentration')
    parser.add_argument('--S0', type=float, default=1e3, help='Base substrate concentration')
    args = parser.parse_args()
    
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using NVIDIA GPU")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(args.device)
        print(f"Using {device}")
    
    evaluator = TeacherModelEvaluator(args.model_path, device)
    model_name = evaluator.model.__class__.__name__
    
    print(f"Evaluating {model_name} model...")
    
    time_points = np.logspace(-8, np.log10(1e2), 1000)
    print(f"Time points: {len(time_points)} (from {time_points[0]:.2e} to {time_points[-1]:.2e})")
    print(f"\nTesting on {args.num_test_conditions} diverse conditions...")
    test_conditions = generate_many_test_conditions(args.num_test_conditions, args.E0, args.S0)
    
    results = evaluate_model(evaluator, test_conditions, time_points)
    analysis = analyze_error_over_time(evaluator, test_conditions, time_points)
    
    print("\nGenerating publication-quality plots...")
    plot_trajectories(results, analysis, model_name)
    print("Generating error analysis plots...")
    plot_error_analysis(analysis, model_name)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*60)
    print(f"\nPERFORMANCE:")
    print(f"  MAE: {results['mae_overall']:.4e}")
    print(f"  RMSE: {results['rmse_overall']:.4e}")
    print(f"  Relative Error: {results['relative_error_overall']:.4f}")
    
    condition_errors = analysis['condition_errors']
    print(f"\nERROR STATISTICS ACROSS {args.num_test_conditions} CONDITIONS:")
    print(f"  MAE - Mean: {np.mean(condition_errors['mae_by_condition']):.4e}")
    print(f"  MAE - Std:  {np.std(condition_errors['mae_by_condition']):.4e}")
    print(f"  MAE - Min:  {np.min(condition_errors['mae_by_condition']):.4e}")
    print(f"  MAE - Max:  {np.max(condition_errors['mae_by_condition']):.4e}")
    print(f"  RMSE - Mean: {np.mean(condition_errors['rmse_by_condition']):.4e}")
    print(f"  RMSE - Std:  {np.std(condition_errors['rmse_by_condition']):.4e}")
    print(f"  R² - Mean:  {np.mean(condition_errors['r2_by_condition']):.4f}")
    print(f"  R² - Std:   {np.std(condition_errors['r2_by_condition']):.4f}")
    print(f"  R² - Min:   {np.min(condition_errors['r2_by_condition']):.4f}")
    print(f"  Relative Error - Mean: {np.mean(condition_errors['relative_error_by_condition']):.4f}")
    
    time_errors = analysis['time_errors']
    early_time_idx = len(time_points) // 10
    late_time_idx = len(time_points) * 9 // 10
    print(f"\nTIME-DEPENDENT ERROR ANALYSIS:")
    print(f"  Early time RMSE (t < {time_points[early_time_idx]:.2e}): {np.mean(time_errors['rmse_by_time'][:early_time_idx]):.4e}")
    print(f"  Late time RMSE  (t > {time_points[late_time_idx]:.2e}): {np.mean(time_errors['rmse_by_time'][late_time_idx:]):.4e}")
    
    print(f"\nPublication-quality figures saved to: results/teacher_evaluation/{model_name}/")
    print("Files generated:")
    print("  - trajectories_complete.pdf/.png - Complete trajectory comparisons")
    print("  - trajectories_fast.pdf/.png - Fast dynamics (0-1 ms)")
    print("  - error_by_species.pdf/.png - Error by species (bar chart)")
    print("  - error_quantile_tail.pdf/.png - Error quantile bands over time")
    print("  - error_phase_aligned.pdf/.png - Phase-aligned error analysis")
    print("  - error_condition_difficulty.pdf/.png - Condition difficulty ranking")
    print("  - error_phase_portrait_vectors.pdf/.png - Phase portrait with error vectors")
    print("  - error_heatmap.pdf/.png - Error heatmap over time and species")

if __name__ == "__main__":
    main()