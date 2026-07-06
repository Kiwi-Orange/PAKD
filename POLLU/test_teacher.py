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
from MAE_simulation import get_pollu_initial_conditions

# Configure matplotlib for publication quality (Nature style)
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

def get_pollu_rate_constants() -> np.ndarray:
    """Return the rate constants for the POLLU model."""
    return np.array([
        3.5e+0, 5.0e+1, 1.0e+4, 5.0e-2, 5.0e-2, 2.0e+4, 5.0e-3, 5.0e+4, 2.0e+4,
        1.0e+5, 5.0e-1, 2.0e+4, 3.0e-1, 1.0e+4, 1.0e+6, 1.0e-2, 2.0e+0, 5.0e+5,
        2.0e+6, 3.0e+3, 3.0e+0, 8.0e+0, 5.0e+0, 3.0e+3, 5.0e+0
    ], dtype=np.float64)

def pollu_reaction(t: float, y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """POLLU air pollution model with 20 species and 25 reactions."""
    y = np.maximum(y, 0.0)
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20 = y
    
    r = np.empty(25, dtype=np.float64)
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

def generate_ground_truth(initial_conditions, time_points, k):
    """Generate ground truth solutions for POLLU model."""
    results = []
    for ic in tqdm(initial_conditions, desc="Generating ground truth"):
        solution = solve_ivp(
            fun=lambda t, y: pollu_reaction(t, y, k),
            t_span=(time_points[0], time_points[-1]),
            y0=ic,
            method='BDF',
            t_eval=time_points,
            atol=1e-10,
            rtol=1e-8
        )
        results.append(solution.y.T)
    return np.array(results)

def create_time_features(time_array):
    """Create log10 time feature (must match training)"""
    t = time_array
    
    # Single time transformation - log scale (same as training)
    t1 = np.log10(t + 1e-12)
    
    return t1.reshape(-1, 1)  # Shape: (n_samples, 1)

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
        model_type = checkpoint.get('model_type', 'ResidualMLP')
        
        # Get input size from checkpoint or infer from weights
        input_size = checkpoint.get('input_size', 21)
        
        # Try to infer from weights if not in checkpoint
        if 'model_state_dict' in checkpoint:
            first_layer_key = None
            for key in checkpoint['model_state_dict'].keys():
                if 'input_proj.weight' in key or 'network.0.weight' in key:
                    first_layer_key = key
                    break
            
            if first_layer_key is not None:
                input_size = checkpoint['model_state_dict'][first_layer_key].shape[1]
                print(f"  ✓ Detected input size from weights: {input_size}")
        
        print(f"  Model type: {model_type}")
        print(f"  Input size: {input_size}")
    
        # Create model matching training configuration
        if model_type == 'ResidualMLP':
            # Try to infer architecture
            hidden_dim = 128
            num_blocks = 3
            
            # Try to get from state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'input_proj.weight' in state_dict:
                    hidden_dim = state_dict['input_proj.weight'].shape[0]
                # Count blocks
                num_blocks = sum(1 for k in state_dict.keys() if 'blocks.' in k and '.ln.weight' in k)
            
            self.model = ResidualMLP(
                input_size=input_size,
                output_size=20, 
                hidden_dim=hidden_dim, 
                num_blocks=num_blocks, 
                dropout=0.0
            )
            print(f"  Architecture: {num_blocks} blocks, hidden_dim={hidden_dim}")
        
        else:  # MLP
            hidden_dim = 128
            num_layers = 3
            
            # Try to infer from state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Count layers (excluding final output layer)
                num_layers = sum(1 for k in state_dict.keys() if 'network.' in k and '.weight' in k) - 1
                # Get hidden dim from first layer
                if 'network.0.weight' in state_dict:
                    hidden_dim = state_dict['network.0.weight'].shape[0]
            
            self.model = MLP(
                input_size=input_size,
                output_size=20, 
                hidden_sizes=[hidden_dim] * num_layers,
                dropout=0.0
            )
            print(f"  Architecture: {num_layers} layers, hidden_dim={hidden_dim}")
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessing info
        self.X_scaler = checkpoint['X_scaler']
        self.y_scaler = checkpoint.get('y_scaler', None)  # NEW: Load y_scaler
        self.input_size = input_size
        
        if self.y_scaler is not None:
            print(f"  ✓ Using normalized targets (y_scaler loaded)")
        else:
            print(f"  ⚠️  No y_scaler found (old model format)")
        
        print(f"✓ Model loaded successfully")

    def predict(self, X):
        """
        Predict POLLU species concentrations.
        
        Parameters
        ----------
        X : np.ndarray
            Input features [time, y0_1, ..., y0_20] with shape (n_samples, 21)
            
        Returns
        -------
        np.ndarray
            Predicted concentrations [y1, ..., y20] with shape (n_samples, 20)
        """
        # Apply same preprocessing as training
        X_copy = X.copy()
        
        # Create single log10 time feature (same as training)
        time_features = create_time_features(X_copy[:, 0])  # Shape: (n_samples, 1)
        
        # Combine time features with initial conditions
        X_augmented = np.column_stack([
            time_features,           # 1 time feature
            X_copy[:, 1:21]         # 20 initial conditions (unchanged)
        ])  # Total: 21 features
        
        X_norm = self.X_scaler.transform(X_augmented)
    
        # Predict (in normalized space)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
            
            # Inverse transform if y_scaler is available
            if self.y_scaler is not None:
                predictions = self.y_scaler.inverse_transform(predictions)
            
            # Ensure non-negative concentrations (physical constraint)
            predictions = np.maximum(predictions, 0.0)
            
            return predictions

def generate_test_conditions_pollu(num_conditions=50, variation_scale='standard', use_base_ic=False):
    """
    Generate test conditions for POLLU model.
    
    Parameters
    ----------
    num_conditions : int
        Number of test conditions to generate
    variation_scale : str
        'standard', 'low', or 'high' - controls concentration ranges
    use_base_ic : bool
        If True, return only the base initial condition (ignore num_conditions)
        
    Returns
    -------
    np.ndarray
        Array of initial conditions (num_conditions, 20) or (1, 20) if use_base_ic
    """
    np.random.seed(42)
    
    # Base POLLU initial conditions
    base_ic = get_pollu_initial_conditions()  # Use consistent base IC
    
    # If testing on base IC only, return it directly
    if use_base_ic:
        print("Using base initial condition only")
        return base_ic.reshape(1, -1)
    
    # Key species indices (non-zero in base condition)
    key_species = [1, 3, 5, 6, 7, 8, 17, 18, 19]
    
    # Define variation ranges based on scale
    if variation_scale == 'low':
        scale_factor = 0.5  # ±50%
    elif variation_scale == 'high':
        scale_factor = 2.0  # ±200%
    else:  # standard
        scale_factor = 1.0  # ±100%
    
    conditions = []
    for _ in range(num_conditions):
        new_condition = base_ic.copy()
        # Vary key species
        for idx in key_species:
            if base_ic[idx] > 0:
                # Log-uniform variation
                log_base = np.log10(base_ic[idx])
                log_var = np.random.uniform(-scale_factor, scale_factor)
                new_condition[idx] = 10 ** (log_base + log_var)
        conditions.append(new_condition)
    
    return np.array(conditions)

def evaluate_model(evaluator, test_conditions, time_points, k):
    """
    Evaluate model performance on POLLU test conditions.
    
    Parameters
    ----------
    evaluator : TeacherModelEvaluator
        Trained model evaluator
    test_conditions : np.ndarray
        Test initial conditions (n_conditions, 20)
    time_points : np.ndarray
        Time points for evaluation
    k : np.ndarray
        Rate constants (25 values)
        
    Returns
    -------
    dict
        Evaluation results
    """
    print("\n" + "="*60)
    print(f"EVALUATING PERFORMANCE ON POLLU (20 SPECIES)")
    if evaluator.y_scaler is not None:
        print(f"Model trained with: normalized targets (StandardScaler)")
    else:
        print(f"Model trained with: original scale targets")
    print("="*60)
    
    # Generate ground truth
    ground_truth = generate_ground_truth(test_conditions, time_points, k)
    n_conditions, n_times = len(test_conditions), len(time_points)
    
    # Prepare input: [time, y0_1, ..., y0_20]
    X_test = np.array([[t] + list(ic) for ic in test_conditions for t in time_points])
    
    # Predict (predictions are in original scale after inverse transform)
    predictions = evaluator.predict(X_test).reshape(n_conditions, n_times, 20)
    
    # Compute errors
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
    
    # Species-specific errors
    species_names = [f'y{i+1}' for i in range(20)]
    species_errors = {}
    for i, species in enumerate(species_names):
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
    print(f"\nKey species errors:")
    key_species = ['y1', 'y2', 'y4', 'y7', 'y8']
    for species in key_species:
        errors = results['species_errors'][species]
        print(f"  {species}: MAE={errors['MAE']:.4e}, R²={errors['R2']:.4f}, RelErr={errors['Relative_Error']:.4f}")
    
    return results

def analyze_error_over_time(evaluator, test_conditions, time_points, k):
    """Analyze how errors evolve over time for POLLU model."""
    ground_truth = generate_ground_truth(test_conditions, time_points, k)
    n_conditions, n_times = len(test_conditions), len(time_points)
    X_test = np.array([[t] + list(ic) for ic in test_conditions for t in time_points])
    predictions = evaluator.predict(X_test).reshape(n_conditions, n_times, 20)
    
    time_errors = {
        'mae_by_time': np.zeros(n_times),
        'rmse_by_time': np.zeros(n_times),
        'mae_by_time_by_species': np.zeros((n_times, 20)),
        'rmse_by_time_by_species': np.zeros((n_times, 20)),
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
        
        for s_idx in range(20):
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
    """
    Publication-quality trajectory plots for POLLU (20 species),
    styled to match MM example:
      - Ground truth: sparse open-circle markers
      - Prediction: solid line
      - Global y-limits per species across plotted conditions
      - (A) Grid: 20 species x up to 3 conditions (rows = conditions, cols = 5)
      - (B) Stacked: 20x1 for first condition (like MM stacked)
      - (C) Fast window: optional, early-time zoom if desired
    """
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # --- Basic setup ---
    time_points = analysis['time_points']
    test_conditions = analysis['test_conditions']
    ground_truth = results['ground_truth']      # (C,T,20)
    predictions  = results['predictions']       # (C,T,20)

    species_names = [f'y{i+1}' for i in range(20)]

    # Select representative conditions (up to 3)
    np.random.seed(42)
    num_conditions = len(test_conditions)
    num_to_plot = min(3, num_conditions)
    if num_conditions <= 3:
        representative_indices = list(range(num_conditions))
    else:
        representative_indices = np.random.choice(num_conditions, size=num_to_plot, replace=False)

    # Colors consistent with your MM style
    colors = {
        'ground_truth': '#000000',
        'prediction':   '#E84A27',   # orange-red like MM reference
    }

    # ---- Sparse markers: use linear spacing in index (good on log-x) ----
    n_markers = 30
    marker_indices = np.unique(np.round(np.linspace(0, len(time_points)-1, n_markers)).astype(int))
    marker_indices = np.clip(marker_indices, 0, len(time_points)-1)

    # ---- Global y-limits per species across plotted conditions ----
    y_limits = []
    for s in range(20):
        all_gt = np.concatenate([ground_truth[idx, :, s] for idx in representative_indices])
        all_pd = np.concatenate([predictions[idx, :, s]  for idx in representative_indices])
        all_data = np.concatenate([all_gt, all_pd])
        y_min, y_max = np.min(all_data), np.max(all_data)
        y_range = y_max - y_min if (y_max - y_min) > 1e-12 else 1.0
        y_limits.append((y_min - 0.10 * y_range, y_max + 0.10 * y_range))

    # ============================================================
    # (A) Grid figure: rows=conditions × 4 species-rows, cols=5
    #     Nature-quality: constrained_layout, LaTeX titles,
    #     per-condition row labels, shared top legend
    # ============================================================
    ncols = 5
    nrows_species = int(np.ceil(20 / ncols))          # 4 species-rows per condition
    total_rows    = num_to_plot * nrows_species        # e.g. 12 for 3 conditions

    # Separator height between condition blocks (empty row fraction)
    sep = 0.015  # relative hspace separator handled via gridspec

    fig_c = plt.figure(
        figsize=(ncols * 3.2, total_rows * 2.2 + 0.6),
        constrained_layout=False,
    )

    # Outer gridspec: one row per condition block, with small gaps between blocks
    gs_top = fig_c.add_gridspec(
        num_to_plot, 1,
        hspace=0.55,                  # gap between condition blocks
        left=0.07, right=0.98,
        top=0.93, bottom=0.06,
    )

    # Collect all subplot axes for shared legend construction
    all_axes = []
    panel_labels = ['a', 'b', 'c']

    for r, cond_idx in enumerate(representative_indices):
        # Inner gridspec: nrows_species × ncols for this condition block
        gs_cond = gs_top[r].subgridspec(nrows_species, ncols, hspace=0.55, wspace=0.30)

        for s in range(20):
            sr = s // ncols   # species row within this condition block
            sc = s % ncols
            ax = fig_c.add_subplot(gs_cond[sr, sc])
            all_axes.append(ax)

            # Ground truth — stars, sparse
            h_gt = ax.semilogx(
                time_points[marker_indices],
                ground_truth[cond_idx, marker_indices, s],
                marker='*', markersize=8, linestyle='none',
                markeredgewidth=1.5, markeredgecolor=colors['ground_truth'],
                markerfacecolor='white', color=colors['ground_truth'],
                zorder=5,
            )
            # Prediction — solid line
            h_pred = ax.semilogx(
                time_points, predictions[cond_idx, :, s],
                color=colors['prediction'], linewidth=3.0,
            )

            # Species title with LaTeX
            ax.set_title(rf'$y_{{{s+1}}}$', fontsize=13, fontweight='bold', pad=3)

            # x-label only on the last species-row of this condition block
            if sr == nrows_species - 1:
                ax.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=11, fontweight='bold')
            else:
                ax.tick_params(labelbottom=False)

            # y-label only on first column
            if sc == 0:
                ax.set_ylabel('Conc.', fontsize=11, fontweight='bold')

            ax.set_ylim(y_limits[s])
            ax.set_xlim(time_points[0], time_points[-1])
            ax.tick_params(axis='both', which='major', labelsize=9, pad=2, width=1.2, length=4)

            # Nature-style aesthetics
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.spines['left'].set_linewidth(1.2)
            ax.xaxis.set_tick_params(direction='out')
            ax.yaxis.set_tick_params(direction='out')

        # Optional: Condition block label on the left edge
        # Removed per user request to delete the "a Condition X" words.
        pass

    # Shared legend at the very top of the figure
    fig_c.legend(
        [h_gt[0], h_pred[0]],
        ['Ground truth', 'Prediction'],
        loc='upper right',
        bbox_to_anchor=(0.98, 0.99),
        fontsize=12, frameon=False,
        ncol=2, handlelength=2.0, handletextpad=0.5,
    )

    # Optional: Suptitle is commented out as it is usually not desired in subpanels of large figures
    # fig_c.suptitle(
    #     rf'{model_name}: All-species trajectories',
    #     fontsize=12, fontweight='bold', y=0.975,
    # )

    plt.savefig(f'{model_output_dir}/trajectories_complete.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'{model_output_dir}/trajectories_complete.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_c)

    # ============================================================
    # (A2) Nature-style combined figure:
    #   Top row  — hero panels: 5 key species, large & readable
    #   Bottom   — compact overview grid of all 20 species (1 condition)
    # ============================================================
    key_species = [1, 4, 6, 9, 14]   # y2, y5, y7, y10, y15 — chemically diverse
    hero_cond = representative_indices[0]

    # gridspec: 2 rows, top row = hero (height 3), bottom = compact grid (height 2)
    fig_n = plt.figure(figsize=(18, 10), constrained_layout=False)
    gs_outer = fig_n.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.38)

    # --- Top: hero panels (1 row × 5 cols) ---
    gs_hero = gs_outer[0].subgridspec(1, len(key_species), wspace=0.32)
    for hi, s in enumerate(key_species):
        ax_h = fig_n.add_subplot(gs_hero[0, hi])
        ax_h.semilogx(time_points[marker_indices],
                      ground_truth[hero_cond, marker_indices, s],
                      marker='*', markersize=10, linestyle='none',
                      markeredgewidth=1.5, markeredgecolor=colors['ground_truth'],
                      markerfacecolor='white', color=colors['ground_truth'],
                      zorder=5, label='Ground truth' if hi == 0 else '')
        ax_h.semilogx(time_points, predictions[hero_cond, :, s],
                      color=colors['prediction'], linewidth=3.5,
                      label='Prediction' if hi == 0 else '')
        ax_h.set_title(rf'$y_{{{s+1}}}$', fontsize=15, fontweight='bold')
        ax_h.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=14, fontweight='bold')
        if hi == 0:
            ax_h.set_ylabel('Concentration', fontsize=14, fontweight='bold')
            ax_h.legend(fontsize=12, frameon=True, framealpha=0.9,
                        loc='best', handlelength=1.4)
        ax_h.set_ylim(y_limits[s])
        ax_h.set_xlim(time_points[0], time_points[-1])
        ax_h.tick_params(labelsize=12)
        ax_h.grid(True, alpha=0.2, linewidth=0.5)

    # panel label
    fig_n.text(0.01, 0.97, 'a', fontsize=20, fontweight='bold', va='top')

    # --- Bottom: compact all-20-species grid (4 rows × 5 cols) ---
    gs_bot = gs_outer[1].subgridspec(4, 5, wspace=0.28, hspace=0.55)
    fig_n.text(0.01, 0.47, 'b', fontsize=20, fontweight='bold', va='top')

    for s in range(20):
        r_b, c_b = divmod(s, 5)
        ax_b = fig_n.add_subplot(gs_bot[r_b, c_b])
        ax_b.semilogx(time_points[marker_indices],
                      ground_truth[hero_cond, marker_indices, s],
                      marker='*', markersize=6, linestyle='none',
                      markeredgewidth=1.2, markeredgecolor=colors['ground_truth'],
                      markerfacecolor='white', color=colors['ground_truth'],
                      zorder=5)
        ax_b.semilogx(time_points, predictions[hero_cond, :, s],
                      color=colors['prediction'], linewidth=2.0)
        ax_b.set_title(rf'$y_{{{s+1}}}$', fontsize=11, fontweight='bold', pad=2)
        ax_b.set_ylim(y_limits[s])
        ax_b.set_xlim(time_points[0], time_points[-1])
        ax_b.tick_params(labelsize=9, pad=1)
        ax_b.grid(True, alpha=0.2, linewidth=0.4)
        # x-label only on bottom row; y-label only on first col
        if r_b == 3:
            ax_b.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=10, fontweight='bold')
        if c_b == 0:
            ax_b.set_ylabel('Conc.', fontsize=10, fontweight='bold')

    fig_n.suptitle(rf'{model_name}: Predicted vs. ground-truth trajectories (POLLU, 20 species)',
                   fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(f'{model_output_dir}/trajectories_nature.pdf', bbox_inches='tight')
    plt.savefig(f'{model_output_dir}/trajectories_nature.png', dpi=300, bbox_inches='tight')
    plt.close(fig_n)

    # ============================================================
    # (B) Stacked figure: 20x1 for first representative condition
    # ============================================================
    first_idx = representative_indices[0]
    fig, axes = plt.subplots(20, 1, figsize=(8.2, 24.0), sharex=True)

    for s, ax in enumerate(axes):
        ax.semilogx(time_points[marker_indices],
                    ground_truth[first_idx, marker_indices, s],
                    marker='*', markersize=10,
                    linestyle='none',
                    markeredgewidth=1.8,
                    markeredgecolor=colors['ground_truth'],
                    markerfacecolor='white',
                    color=colors['ground_truth'],
                    zorder=5,
                    label='Ground Truth' if s == 0 else "")

        ax.semilogx(time_points,
                    predictions[first_idx, :, s],
                    color=colors['prediction'],
                    linewidth=3.0,
                    label='Prediction' if s == 0 else "")

        ax.set_ylabel(rf'$y_{{{s+1}}}$', fontsize=14, fontweight='bold')
        ax.set_ylim(y_limits[s])
        ax.grid(False)

        if s == 0:
            ax.legend(frameon=True, loc='upper right', fontsize=12)

    axes[-1].set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=16, fontweight='bold')
    axes[-1].set_xlim(time_points[0], time_points[-1])

    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/trajectories_stacked.pdf')
    plt.savefig(f'{model_output_dir}/trajectories_stacked.png', dpi=300)
    plt.close()

    # ============================================================
    # (C) Optional: early-time "fast dynamics" zoom (set threshold)
    # ============================================================
    fast_t_max = 1e-6
    fast_mask = time_points <= fast_t_max
    if np.any(fast_mask):
        fast_time = time_points[fast_mask]
        n_fast = len(fast_time)

        # marker indices in fast window (index relative to fast_time)
        n_fast_markers = min(20, max(6, n_fast // 10))
        fast_marker_idx = np.unique(np.linspace(0, n_fast - 1, n_fast_markers, dtype=int))

        # y-limits in fast window (per species, across plotted conditions)
        y_limits_fast = []
        for s in range(20):
            all_gt = np.concatenate([ground_truth[idx, :n_fast, s] for idx in representative_indices])
            all_pd = np.concatenate([predictions[idx, :n_fast, s]  for idx in representative_indices])
            all_data = np.concatenate([all_gt, all_pd])
            y_min, y_max = np.min(all_data), np.max(all_data)
            y_range = y_max - y_min if (y_max - y_min) > 1e-12 else 1.0

            # 关键：fast 段变化可能很“尖”，padding 建议更大一些，避免顶到边界
            pad = 0.20
            y_limits_fast.append((y_min - pad * y_range, y_max + pad * y_range))

        # Plot only key species for fast view (avoid 20x figure too large)
        key_species_idx = [0, 1, 3, 6, 7, 8]
        key_names = [species_names[i] for i in key_species_idx]

        fig, axes = plt.subplots(2, 3, figsize=(3.8 * 3, 3.2 * 2))
        axes = axes.flatten()

        for plot_idx, s in enumerate(key_species_idx):
            ax = axes[plot_idx]

            # Ground truth markers: USE fast_time / fast_marker_idx
            ax.semilogx(
                fast_time[fast_marker_idx],
                ground_truth[representative_indices[0], fast_marker_idx, s],
                marker='*', markersize=12,
                linestyle='none',
                markeredgewidth=1.8,
                markeredgecolor=colors['ground_truth'],
                markerfacecolor='white',
                color=colors['ground_truth'],
                zorder=5,
                label='Ground Truth' if plot_idx == 0 else ""
            )

            # Prediction line: USE fast_time and slice :n_fast
            ax.semilogx(
                fast_time,
                predictions[representative_indices[0], :n_fast, s],
                color=colors['prediction'],
                linewidth=3.5,
                label='Prediction' if plot_idx == 0 else ""
            )

            ax.set_title(key_names[plot_idx], fontsize=14, fontweight='bold', pad=5)
            ax.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=14, fontweight='bold')
            ax.set_ylabel('Conc.', fontsize=14, fontweight='bold')

            # USE fast y-limits
            ax.set_ylim(y_limits_fast[s])

            # 关键：xlim 也必须是 fast window
            ax.set_xlim(fast_time[0], fast_time[-1])

            ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)

            if plot_idx == 0:
                ax.legend(frameon=True, fancybox=False, shadow=False,
                        loc='best', fontsize=13, framealpha=0.9)

        plt.tight_layout(pad=1.2)
        plt.savefig(f'{model_output_dir}/trajectories_fast.pdf')
        plt.savefig(f'{model_output_dir}/trajectories_fast.png', dpi=300)
        plt.close()

    print(f"Trajectory plots saved to {model_output_dir}")

def plot_error_analysis(analysis, model_name, output_dir='results/teacher_evaluation'):
    """
    Better-than-heatmap visualization set for POLLU:
      1) Quantile band + tail index over time (log-time)
      2) Phase-aligned error (t normalized by event time t*)
      3) Condition difficulty ranking + explained by key species ratio
      4) Phase portrait (y2 vs y5) with error vectors (sparse arrows)

    LaTeX:
      - Uses mathtext by default (works without external LaTeX).
      - If you have a LaTeX installation and want full usetex, uncomment the line below.
    """
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # ---- LaTeX-like rendering (mathtext; no external TeX needed) ----
    plt.rcParams['text.usetex'] = False     # set True if you have LaTeX installed
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'serif'

    time_points   = analysis['time_points']        # (T,)
    gt            = analysis['ground_truth']       # (C, T, 20)
    pred          = analysis['predictions']        # (C, T, 20)
    conds         = analysis['test_conditions']    # (C, 20) initial conditions

    species_names = [rf'$y_{{{i+1}}}$' for i in range(20)]
    eps = 1e-12

    C, T, D = gt.shape
    assert D == 20

    # Handle single condition case - simplified plots
    if C == 1:
        print("  Single condition detected - generating simplified error analysis plots...")
        
        # Single condition: Plot error over time for each species
        diff = pred - gt  # (1, T, 20)
        abs_error = np.abs(diff[0])  # (T, 20)
        rmse_by_time = np.sqrt(np.mean(diff[0]**2, axis=1))  # (T,)
        mae_by_time = np.mean(np.abs(diff[0]), axis=1)  # (T,)
        
        # Figure 1: error over time
        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

        ax.loglog(time_points, rmse_by_time + eps, linewidth=1.5, label=r'RMSE')
        ax.loglog(time_points, mae_by_time + eps, linewidth=1.5, linestyle='--', label=r'MAE')
        ax.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=8)
        ax.set_ylabel(r'Error', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7, direction='out', length=3, width=0.8)
        ax.tick_params(axis='both', which='minor', direction='out', length=2, width=0.6)
        ax.legend(fontsize=7, frameon=False, loc='upper right')
        
        # Nature-style aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)

        plt.savefig(f'{model_output_dir}/error_over_time.pdf', bbox_inches='tight', transparent=True)
        plt.savefig(f'{model_output_dir}/error_over_time.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

        # Figure 2: error by species
        species_mae = np.mean(abs_error, axis=0)  # (20,)
        species_rmse = np.sqrt(np.mean(diff[0]**2, axis=0))  # (20,)

        fig, ax_sp = plt.subplots(figsize=(6.0, 2.5), constrained_layout=True)

        x = np.arange(20)
        width = 0.35
        ax_sp.bar(x - width/2, species_mae + eps, width, label='MAE', alpha=0.85, color='#E84A27')
        ax_sp.bar(x + width/2, species_rmse + eps, width, label='RMSE', alpha=0.85, color='#000000')
        ax_sp.set_yscale('log')
        ax_sp.set_xticks(x)
        ax_sp.set_xticklabels([rf'$y_{{{i+1}}}$' for i in range(20)], rotation=90, fontsize=6)
        ax_sp.set_xlabel('Species', fontsize=8)
        ax_sp.set_ylabel('Error', fontsize=8)
        ax_sp.tick_params(axis='both', which='major', labelsize=7, direction='out', length=3, width=0.8)
        ax_sp.tick_params(axis='x', pad=2)
        ax_sp.legend(fontsize=7, frameon=False, loc='best')
        
        # Nature-style aesthetics
        ax_sp.spines['top'].set_visible(False)
        ax_sp.spines['right'].set_visible(False)
        ax_sp.spines['bottom'].set_linewidth(0.8)
        ax_sp.spines['left'].set_linewidth(0.8)

        plt.savefig(f'{model_output_dir}/error_by_species.pdf', bbox_inches='tight', transparent=True)
        plt.savefig(f'{model_output_dir}/error_by_species.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        
        # Figure 3: Phase portrait (y2 vs y5) - single condition
        fig, ax = plt.subplots(figsize=(3, 3))
        
        y2g, y5g = gt[0, :, 1], gt[0, :, 4]
        y2p, y5p = pred[0, :, 1], pred[0, :, 4]
        
        ax.plot(y2g, y5g, linewidth=1.5, label=r'GT', color='#000000')
        ax.plot(y2p, y5p, linewidth=1.5, linestyle='--', label=r'Pred', color='#E84A27')
        
        # sparse arrows
        arrow_n = 25
        arrow_idx = np.unique(np.round(np.logspace(0, np.log10(T-1), arrow_n)).astype(int))
        arrow_idx = np.clip(arrow_idx, 0, T-1)
        
        dx = (y2p[arrow_idx] - y2g[arrow_idx])
        dy = (y5p[arrow_idx] - y5g[arrow_idx])
        
        # Only plot arrows if there are non-zero errors
        if np.any(np.abs(dx) > eps) or np.any(np.abs(dy) > eps):
            ax.quiver(y2g[arrow_idx], y5g[arrow_idx], dx, dy,
                      angles='xy', scale_units='xy', scale=1.0, width=0.005, alpha=0.6, color='#E84A27')
        
        ax.set_xlabel(r'$y_2$', fontsize=8)
        ax.set_ylabel(r'$y_5$', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7, direction='out', length=3, width=0.8)
        ax.legend(fontsize=7, frameon=False, loc='best')
        
        # Nature-style aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        
        plt.tight_layout()
        plt.savefig(f'{model_output_dir}/error_phase_portrait_vectors.pdf', bbox_inches='tight', transparent=True)
        plt.savefig(f'{model_output_dir}/error_phase_portrait_vectors.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        
        # Figure 4: Error heatmap over time and species
        fig, ax = plt.subplots(figsize=(6, 3))
        
        # Log-transform errors for better visualization
        log_error = np.log10(abs_error.T + eps)  # (20, T)
        
        im = ax.imshow(log_error, aspect='auto', cmap='viridis',
                       extent=[np.log10(time_points[0]), np.log10(time_points[-1]), 19.5, -0.5])
        
        ax.set_xlabel(r'$\log_{10}(t)$', fontsize=8)
        ax.set_ylabel('Species', fontsize=8)
        ax.set_yticks(range(20))
        ax.set_yticklabels([rf'$y_{{{i+1}}}$' for i in range(20)], fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=7, direction='out', length=3, width=0.8)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'$\log_{10}$(|Error|)', fontsize=8)
        cbar.ax.tick_params(labelsize=7, direction='out', length=3, width=0.8)
        cbar.outline.set_linewidth(0.8)
        
        # Nature-style aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        
        plt.tight_layout()
        plt.savefig(f'{model_output_dir}/error_heatmap.pdf', bbox_inches='tight', transparent=True)
        plt.savefig(f'{model_output_dir}/error_heatmap.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        
        print(f"Error analysis plots (single condition) saved to {model_output_dir}")
        return

    # -----------------------------
    # Multiple conditions case - original code
    # -----------------------------
    
    # 0) Sort conditions for structure (by y2_0/y6_0 ratio - key species)
    # Use y2 (index 1) and y6 (index 5) as key species ratio
    ratio = (conds[:, 1] + eps) / (conds[:, 5] + eps)  # y2_0/y6_0
    order = np.argsort(ratio)
    gt   = gt[order]
    pred = pred[order]
    ratio = ratio[order]
    conds = conds[order]

    # -----------------------------
    # 1) Vectorized errors
    # -----------------------------
    diff = pred - gt                           # (C,T,20)
    rmse_ct = np.sqrt(np.mean(diff**2, axis=2))# (C,T) overall (across species)
    mae_ct  = np.mean(np.abs(diff), axis=2)    # (C,T)

    # --- Quantiles over conditions for each time ---
    qs = [0.1, 0.5, 0.9]
    rmse_q = np.quantile(rmse_ct, qs, axis=0)  # (3,T)
    mae_q  = np.quantile(mae_ct,  qs, axis=0)

    # A small "tail index": 90th / 50th ratio (how heavy the tail is)
    tail_rmse = (rmse_q[2] + eps) / (rmse_q[1] + eps)

    # -----------------------------
    # Figure 1: Quantile band + tail index
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_title(rf'{model_name}: error distribution over time', fontsize=13, fontweight='bold')

    # Ensure positive values for log scale
    rmse_q_safe = np.maximum(rmse_q, eps)
    mae_q_safe = np.maximum(mae_q, eps)

    ax.loglog(time_points, rmse_q_safe[1], linewidth=2.5, label=r'$\mathrm{RMSE}$ median')
    ax.fill_between(time_points, rmse_q_safe[0], rmse_q_safe[2], alpha=0.25, label=r'$\mathrm{RMSE}$ 10–90\%')

    ax.loglog(time_points, mae_q_safe[1], linewidth=2.0, linestyle='--', label=r'$\mathrm{MAE}$ median')
    ax.fill_between(time_points, mae_q_safe[0], mae_q_safe[2], alpha=0.15, label=r'$\mathrm{MAE}$ 10–90\%')

    ax.set_xlabel(r'$t\ \mathrm{(s)}$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Second y-axis: tail index
    ax2 = ax.twinx()
    ax2.semilogx(time_points, tail_rmse, linewidth=1.8, linestyle=':', label=r'$\mathrm{tail} = q_{0.9}/q_{0.5}$')
    ax2.set_ylabel(r'$q_{0.9}/q_{0.5}$', fontsize=11, fontweight='bold')
    ax2.tick_params(labelsize=9)

    # combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/error_quantile_tail.pdf')
    plt.savefig(f'{model_output_dir}/error_quantile_tail.png', dpi=300)
    plt.close()

    # -----------------------------
    # 2) Phase-aligned error: align by event time t*
    #    Here: t* = time when y5 (index 4) reaches half of its max value (per condition)
    # -----------------------------
    y5_gt = gt[:, :, 4]  # (C,T) - y5 is a key reactive species
    y5_max = np.max(y5_gt, axis=1) + eps
    target = 0.5 * y5_max

    t_star = np.full(C, time_points[-1])
    for i in range(C):
        idx = np.where(y5_gt[i] >= target[i])[0]
        if len(idx) > 0:
            t_star[i] = time_points[idx[0]]

    # normalized time tau = log10(t/t*)
    # We will bin tau to get a clean plot.
    tau = np.log10((time_points[None, :] + eps) / (t_star[:, None] + eps))  # (C,T)
    rmse = rmse_ct  # (C,T)

    # Define tau bins (shared across conditions)
    tau_min, tau_max = -4.0, 4.0
    n_bins = 120
    tau_edges = np.linspace(tau_min, tau_max, n_bins + 1)
    tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])

    # Aggregate: for each bin, collect all rmse values and compute quantiles
    rmse_tau_q = np.zeros((3, n_bins))
    for b in range(n_bins):
        mask = (tau >= tau_edges[b]) & (tau < tau_edges[b+1])
        vals = rmse[mask]
        if vals.size < 20:
            rmse_tau_q[:, b] = np.nan
        else:
            rmse_tau_q[:, b] = np.quantile(vals, qs)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_title(r'Phase-aligned error: $\tau=\log_{10}(t/t_{1/2})$', fontsize=13, fontweight='bold')

    ax.plot(tau_centers, rmse_tau_q[1], linewidth=2.5, label=r'$\mathrm{RMSE}$ median')
    ax.fill_between(tau_centers, rmse_tau_q[0], rmse_tau_q[2], alpha=0.25, label=r'10–90\%')
    ax.axvline(0.0, linewidth=1.5, linestyle='--', alpha=0.7)
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

    # -----------------------------
    # 3) Condition difficulty ranking (+ explain with log(y2_0/y6_0))
    # -----------------------------
    # Define difficulty per condition = median RMSE over time
    difficulty = np.median(rmse_ct, axis=1)  # (C,)
    x = np.log10(ratio + eps)
    y = np.log10(difficulty + eps)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # 3a) Ranking plot
    ax = axes[0]
    idx_sorted = np.argsort(difficulty)
    ax.plot(np.arange(C), difficulty[idx_sorted] + eps, linewidth=2.5)
    ax.set_yscale('log')
    ax.set_title(r'Condition difficulty ranking', fontsize=13, fontweight='bold')
    ax.set_xlabel(r'rank (easy $\rightarrow$ hard)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'median $\mathrm{RMSE}$ over time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)

    # 3b) Explain by y2_0/y6_0
    ax = axes[1]
    ax.scatter(x, y, s=22, alpha=0.75)
    ax.set_title(r'What explains difficulty? (proxy: $y_2^0/y_6^0$)', fontsize=13, fontweight='bold')
    ax.set_xlabel(r'$\log_{10}(y_2^0/y_6^0)$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(\mathrm{median\ RMSE})$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/error_condition_difficulty.pdf')
    plt.savefig(f'{model_output_dir}/error_condition_difficulty.png', dpi=300)
    plt.close()

    # -----------------------------
    # 4) Phase portrait with error vectors (y2 vs y5) — sparse arrows
    # -----------------------------
    # pick a few representative conditions: easy/median/hard
    rep = [idx_sorted[0], idx_sorted[C//2], idx_sorted[-1]]
    labels_plot = [r'easy', r'median', r'hard']

    # sparse time indices for arrows
    arrow_n = 35
    arrow_idx = np.unique(np.round(np.logspace(0, np.log10(T-1), arrow_n)).astype(int))
    arrow_idx = np.clip(arrow_idx, 0, T-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, ci, lab in zip(axes, rep, labels_plot):
        # Use y2 (index 1) and y5 (index 4) for phase portrait
        y2g, y5g = gt[ci, :, 1], gt[ci, :, 4]
        y2p, y5p = pred[ci, :, 1], pred[ci, :, 4]

        ax.plot(y2g, y5g, linewidth=2.2, label=r'GT')
        ax.plot(y2p, y5p, linewidth=2.0, linestyle='--', label=r'Pred')

        # error vectors
        dx = (y2p[arrow_idx] - y2g[arrow_idx])
        dy = (y5p[arrow_idx] - y5g[arrow_idx])
        ax.quiver(y2g[arrow_idx], y5g[arrow_idx], dx, dy,
                  angles='xy', scale_units='xy', scale=1.0, width=0.003, alpha=0.6)

        ax.set_title(rf'Phase portrait $(y_2, y_5)$ — {lab}', fontsize=12, fontweight='bold')
        ax.set_xlabel(r'$y_2$', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$y_5$', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)

        if ax is axes[0]:
            ax.legend(fontsize=10, frameon=True)

    plt.savefig(f'{model_output_dir}/error_phase_portrait_vectors.pdf')
    plt.savefig(f'{model_output_dir}/error_phase_portrait_vectors.png', dpi=300)
    plt.close()
    
    print(f"Error analysis plots saved to {model_output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate teacher model on POLLU')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--num_test_conditions', type=int, default=100, 
                       help='Number of test conditions')
    parser.add_argument('--use_base_ic', action='store_true',
                       help='Test on base initial condition only')
    parser.add_argument('--variation_scale', type=str, default='standard',
                       choices=['low', 'standard', 'high'],
                       help='Variation scale for test conditions')
    args = parser.parse_args()
    
    # Device selection
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
    
    # Load model
    evaluator = TeacherModelEvaluator(args.model_path, device)
    model_name = evaluator.model.__class__.__name__
    
    print(f"\nEvaluating {model_name} model on POLLU...")
    
    # Setup
    k = get_pollu_rate_constants()
    time_points = np.logspace(-12, 4, 1000)  # Match training: 1000 points
    print(f"Time points: {len(time_points)} (from {time_points[0]:.2e} to {time_points[-1]:.2e})")
    
    # Generate test conditions
    if args.use_base_ic:
        print("\nTesting on base initial condition only...")
        test_conditions = generate_test_conditions_pollu(num_conditions=1, use_base_ic=True)
    else:
        print(f"\nGenerating {args.num_test_conditions} test conditions (variation: {args.variation_scale})...")
        test_conditions = generate_test_conditions_pollu(
            args.num_test_conditions, 
            variation_scale=args.variation_scale,
            use_base_ic=False
        )
    
    print(f"Test conditions shape: {test_conditions.shape}")
    
    # Evaluate
    results = evaluate_model(evaluator, test_conditions, time_points, k)
    analysis = analyze_error_over_time(evaluator, test_conditions, time_points, k)
    
    # Plot results
    print("\nGenerating publication-quality plots...")
    model_name_suffix = "_base_ic" if args.use_base_ic else f"_{args.variation_scale}"
    plot_trajectories(results, analysis, model_name + model_name_suffix)
    print("Generating error analysis plots...")
    plot_error_analysis(analysis, model_name + model_name_suffix)
    
    # Summary
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION SUMMARY (POLLU - 20 SPECIES)")
    print("="*60)
    
    if args.use_base_ic:
        print(f"\nTesting Mode: Base Initial Condition Only")
    else:
        print(f"\nTesting Mode: {args.num_test_conditions} conditions (variation: {args.variation_scale})")
    
    print(f"\nPERFORMANCE:")
    print(f"  MAE:  {results['mae_overall']:.4e}")
    print(f"  RMSE: {results['rmse_overall']:.4e}")
    print(f"  Relative Error: {results['relative_error_overall']:.4f}")
    
    condition_errors = analysis['condition_errors']
    num_conditions = len(test_conditions)
    print(f"\nERROR STATISTICS ACROSS {num_conditions} CONDITION(S):")
    print(f"  MAE - Mean: {np.mean(condition_errors['mae_by_condition']):.4e}")
    if num_conditions > 1:
        print(f"  MAE - Std:  {np.std(condition_errors['mae_by_condition']):.4e}")
        print(f"  MAE - Min:  {np.min(condition_errors['mae_by_condition']):.4e}")
        print(f"  MAE - Max:  {np.max(condition_errors['mae_by_condition']):.4e}")
    print(f"  RMSE - Mean: {np.mean(condition_errors['rmse_by_condition']):.4e}")
    if num_conditions > 1:
        print(f"  RMSE - Std:  {np.std(condition_errors['rmse_by_condition']):.4e}")
    print(f"  R² - Mean:  {np.mean(condition_errors['r2_by_condition']):.4f}")
    if num_conditions > 1:
        print(f"  R² - Std:   {np.std(condition_errors['r2_by_condition']):.4f}")
        print(f"  R² - Min:   {np.min(condition_errors['r2_by_condition']):.4f}")
    print(f"  Relative Error - Mean: {np.mean(condition_errors['relative_error_by_condition']):.4f}")
    
    time_errors = analysis['time_errors']
    early_time_idx = len(time_points) // 10
    late_time_idx = len(time_points) * 9 // 10
    print(f"\nTIME-DEPENDENT ERROR ANALYSIS:")
    print(f"  Early time RMSE (t < {time_points[early_time_idx]:.2e}): {np.mean(time_errors['rmse_by_time'][:early_time_idx]):.4e}")
    print(f"  Late time RMSE  (t > {time_points[late_time_idx]:.2e}): {np.mean(time_errors['rmse_by_time'][late_time_idx:]):.4e}")
    
    print(f"\nPublication-quality figures saved to: results/teacher_evaluation/{model_name + model_name_suffix}/")
    print("Files generated:")
    print("  - trajectories_complete.pdf/.png - Complete trajectory comparisons (all 20 species)")
    print("  - trajectories_key_species.pdf/.png - Key species trajectories")
    print("  - error_over_time.pdf/.png - Error over time")
    print("  - error_by_species.pdf/.png - Error by species (bar chart)")
    print("  - error_heatmap.pdf/.png - Error heatmap over time and species")
    print("  - error_quantile_tail.pdf/.png - Error quantile bands over time")
    print("  - error_phase_aligned.pdf/.png - Phase-aligned error analysis")
    print("  - error_condition_difficulty.pdf/.png - Condition difficulty ranking")
    print("  - error_phase_portrait_vectors.pdf/.png - Phase portrait with error vectors")

if __name__ == "__main__":
    main()