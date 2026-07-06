import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import argparse
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from tqdm import tqdm
from models import MLP, ResidualMLP
import matplotlib as mpl

# Configure matplotlib for publication quality
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Roman']
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.linewidth'] = 1.8
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.15

COMPACT_3D_ERROR_VMAX = 2.0e-2


def build_laplacian_matrix(n_interior, h):
    """Build discrete Laplacian matrix for Fisher-KPP."""
    h2 = h ** 2
    main_diag = 2.0 * np.ones(n_interior) / h2
    off_diag = -np.ones(n_interior - 1) / h2
    A = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr')
    return A


def fisher_kpp_rhs(t, u, A, epsilon):
    """Right-hand side of Fisher-KPP equation."""
    u = np.maximum(u, 0.0)
    diffusion = -epsilon * A.dot(u)
    reaction = u * (1 - u)
    return diffusion + reaction


def get_fisher_kpp_initial_condition(x, ic_type='step', params=None):
    """
    Generate initial condition for Fisher-KPP equation.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial grid points
    ic_type : str
        Type: 'step', 'gaussian', 'sine', 'double_gaussian'
    params : dict, optional
        Parameters for the initial condition
    """
    if params is None:
        params = {}
    
    if ic_type == 'step':
        x_step = params.get('x_step', 0.3)
        transition_width = params.get('transition_width', 0.05)
        u0 = 0.5 * (1 - np.tanh((x - x_step) / transition_width))
    elif ic_type == 'gaussian':
        center = params.get('center', 0.5)
        width = params.get('width', 0.1)
        amplitude = params.get('amplitude', 0.8)
        u0 = amplitude * np.exp(-((x - center) / width) ** 2)
    elif ic_type == 'sine':
        n_modes = params.get('n_modes', 1)
        u0 = 0.5 * (1 + 0.5 * np.sin(n_modes * np.pi * x))
    elif ic_type == 'double_gaussian':
        u0 = (0.6 * np.exp(-((x - 0.3) / 0.08) ** 2) + 
              0.4 * np.exp(-((x - 0.7) / 0.08) ** 2))
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")
    
    return np.clip(u0, 0.0, 1.0)


def generate_ground_truth(initial_conditions, time_points, n_grid, epsilon=0.01):
    """Generate ground truth solutions for Fisher-KPP model."""
    h = 1.0 / (n_grid + 1)
    A = build_laplacian_matrix(n_grid, h)
    
    results = []
    for ic in tqdm(initial_conditions, desc="Generating ground truth"):
        solution = solve_ivp(
            fun=lambda t, y: fisher_kpp_rhs(t, y, A, epsilon),
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
    t1 = np.log10(t + 1.0)
    return t1.reshape(-1, 1)


class TeacherModelEvaluator:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.n_grid = 100  # Default
        self.load_model(model_path)
    
    def __getattr__(self, name):
        if self.model is not None and not name.startswith('_'):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def load_model(self, model_path):
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_type = checkpoint.get('model_type', 'ResidualMLP')
        
        # Get n_grid from checkpoint
        self.n_grid = checkpoint.get('n_grid', 100)
        input_size = checkpoint.get('input_size', self.n_grid + 1)
        output_size = checkpoint.get('output_size', self.n_grid)
        
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
        print(f"  Output size: {output_size}")
        print(f"  Grid points: {self.n_grid}")
    
        # Create model matching training configuration
        if model_type == 'ResidualMLP':
            hidden_dim = 128
            num_blocks = 3
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'input_proj.weight' in state_dict:
                    hidden_dim = state_dict['input_proj.weight'].shape[0]
                num_blocks = sum(1 for k in state_dict.keys() if 'blocks.' in k and '.ln.weight' in k)
            
            self.model = ResidualMLP(
                input_size=input_size,
                output_size=output_size, 
                hidden_dim=hidden_dim, 
                num_blocks=num_blocks, 
                dropout=0.0
            )
            print(f"  Architecture: {num_blocks} blocks, hidden_dim={hidden_dim}")
        
        else:  # MLP
            hidden_dim = 128
            num_layers = 3
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                num_layers = sum(1 for k in state_dict.keys() if 'network.' in k and '.weight' in k) - 1
                if 'network.0.weight' in state_dict:
                    hidden_dim = state_dict['network.0.weight'].shape[0]
            
            self.model = MLP(
                input_size=input_size,
                output_size=output_size, 
                hidden_sizes=[hidden_dim] * num_layers,
                dropout=0.0
            )
            print(f"  Architecture: {num_layers} layers, hidden_dim={hidden_dim}")
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessing info
        self.X_scaler = checkpoint['X_scaler']
        self.y_scaler = checkpoint.get('y_scaler', None)
        self.input_size = input_size
        
        if self.y_scaler is not None:
            print(f"  ✓ Using normalized targets (y_scaler loaded)")
        else:
            print(f"  ⚠️  No y_scaler found (old model format)")
        
        print(f"✓ Model loaded successfully")

    def predict(self, X):
        """
        Predict Fisher-KPP solution at grid points.
        
        Parameters
        ----------
        X : np.ndarray
            Input features [time, u0_1, ..., u0_n] with shape (n_samples, n_grid+1)
            
        Returns
        -------
        np.ndarray
            Predicted solution [u_1, ..., u_n] with shape (n_samples, n_grid)
        """
        X_copy = X.copy()
        n_grid = self.n_grid
        
        # Create single log10 time feature (same as training)
        time_features = create_time_features(X_copy[:, 0])
        
        # Combine time features with initial conditions
        X_augmented = np.column_stack([
            time_features,
            X_copy[:, 1:n_grid+1]
        ])
        
        X_norm = self.X_scaler.transform(X_augmented)
    
        # Predict (in normalized space)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
            
            # Inverse transform if y_scaler is available
            if self.y_scaler is not None:
                predictions = self.y_scaler.inverse_transform(predictions)
            
            # Ensure non-negative (physical constraint for Fisher-KPP)
            predictions = np.clip(predictions, 0.0, 1.0)
            
            return predictions


def generate_test_conditions_fisher_kpp(n_grid, num_conditions=10, ic_type='step', use_base_ic=False):
    """
    Generate test conditions for Fisher-KPP model.
    
    Parameters
    ----------
    n_grid : int
        Number of interior grid points
    num_conditions : int
        Number of test conditions to generate
    ic_type : str
        Initial condition type
    use_base_ic : bool
        If True, return only the base initial condition
        
    Returns
    -------
    np.ndarray
        Array of initial conditions (num_conditions, n_grid) or (1, n_grid)
    """
    np.random.seed(42)
    
    x = np.linspace(0, 1, n_grid + 2)[1:-1]  # Interior points
    
    # Base initial condition
    base_ic = get_fisher_kpp_initial_condition(x, ic_type)
    
    if use_base_ic:
        print("Using base initial condition only")
        return base_ic.reshape(1, -1)
    
    conditions = [base_ic]
    
    # Generate varied conditions
    ic_types = ['step', 'gaussian', 'sine', 'double_gaussian']
    
    for i in range(num_conditions - 1):
        ic_t = ic_types[i % len(ic_types)]
        
        if ic_t == 'step':
            params = {
                'x_step': np.random.uniform(0.2, 0.5),
                'transition_width': np.random.uniform(0.03, 0.08)
            }
        elif ic_t == 'gaussian':
            params = {
                'center': np.random.uniform(0.3, 0.7),
                'width': np.random.uniform(0.05, 0.15),
                'amplitude': np.random.uniform(0.5, 1.0)
            }
        elif ic_t == 'sine':
            params = {'n_modes': np.random.randint(1, 4)}
        else:
            params = {}
        
        u0 = get_fisher_kpp_initial_condition(x, ic_t, params)
        conditions.append(u0)
    
    return np.array(conditions)


def evaluate_model(evaluator, test_conditions, time_points, n_grid, epsilon=0.01):
    """
    Evaluate model performance on Fisher-KPP test conditions.
    """
    print("\n" + "="*60)
    print(f"EVALUATING PERFORMANCE ON FISHER-KPP ({n_grid} GRID POINTS)")
    if evaluator.y_scaler is not None:
        print(f"Model trained with: normalized targets (StandardScaler)")
    else:
        print(f"Model trained with: original scale targets")
    print("="*60)
    
    # Generate ground truth
    ground_truth = generate_ground_truth(test_conditions, time_points, n_grid, epsilon)
    n_conditions, n_times = len(test_conditions), len(time_points)
    
    # Prepare input: [time, u0_1, ..., u0_n]
    X_test = np.array([[t] + list(ic) for ic in test_conditions for t in time_points])
    
    # Predict
    predictions = evaluator.predict(X_test).reshape(n_conditions, n_times, n_grid)
    
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
    
    # Grid point-specific errors
    grid_errors = {}
    for i in range(n_grid):
        gt_grid = ground_truth[:, :, i].flatten()
        pred_grid = predictions[:, :, i].flatten()
        grid_errors[f'x_{i+1}'] = {
            'MAE': mean_absolute_error(gt_grid, pred_grid),
            'RMSE': np.sqrt(mean_squared_error(gt_grid, pred_grid)),
            'R2': r2_score(gt_grid, pred_grid),
            'Relative_Error': np.mean(np.abs(gt_grid - pred_grid) / (np.abs(gt_grid) + 1e-12))
        }
    results['grid_errors'] = grid_errors
    
    print(f"\nRESULTS:")
    print(f"  Overall MAE: {results['mae_overall']:.4e}")
    print(f"  Overall RMSE: {results['rmse_overall']:.4e}")
    print(f"  Overall Relative Error: {results['relative_error_overall']:.4f}")
    
    # Show errors at key spatial locations
    key_indices = [0, n_grid//4, n_grid//2, 3*n_grid//4, n_grid-1]
    print(f"\nErrors at key grid points:")
    for idx in key_indices:
        key = f'x_{idx+1}'
        errors = results['grid_errors'][key]
        print(f"  {key}: MAE={errors['MAE']:.4e}, R²={errors['R2']:.4f}")
    
    return results


def analyze_error_over_time(evaluator, test_conditions, time_points, n_grid, epsilon=0.01):
    """Analyze how errors evolve over time for Fisher-KPP model."""
    ground_truth = generate_ground_truth(test_conditions, time_points, n_grid, epsilon)
    n_conditions, n_times = len(test_conditions), len(time_points)
    X_test = np.array([[t] + list(ic) for ic in test_conditions for t in time_points])
    predictions = evaluator.predict(X_test).reshape(n_conditions, n_times, n_grid)
    
    time_errors = {
        'mae_by_time': np.zeros(n_times),
        'rmse_by_time': np.zeros(n_times),
        'mae_by_time_by_grid': np.zeros((n_times, n_grid)),
        'rmse_by_time_by_grid': np.zeros((n_times, n_grid)),
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
        
        for g_idx in range(n_grid):
            gt_grid_t = ground_truth[:, t_idx, g_idx]
            pred_grid_t = predictions[:, t_idx, g_idx]
            time_errors['mae_by_time_by_grid'][t_idx, g_idx] = mean_absolute_error(gt_grid_t, pred_grid_t)
            time_errors['rmse_by_time_by_grid'][t_idx, g_idx] = np.sqrt(mean_squared_error(gt_grid_t, pred_grid_t))
    
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


def plot_trajectories(results, analysis, model_name, n_grid, output_dir='results/teacher_evaluation'):
    """Create publication-quality trajectory plots for Fisher-KPP."""
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
        'prediction': '#E31A1C',
    }
    
    # MORE markers for temporal plot (open circles)
    n_markers_temporal = 20
    marker_indices_temporal = np.unique(np.round(np.logspace(0, np.log10(len(time_points)-1), n_markers_temporal)).astype(int))
    marker_indices_temporal = np.clip(marker_indices_temporal, 0, len(time_points)-1)
    positive_time_min = time_points[1] if len(time_points) > 1 and time_points[0] <= 0 else time_points[0]

    # Select key grid points to plot (10 evenly spaced)
    n_plot_grids = 10
    grid_indices = np.linspace(0, n_grid-1, n_plot_grids, dtype=int)
    x_full = np.linspace(0, 1, n_grid + 2)[1:-1]
    
    # Figure: Solution at key grid points over time
    ncols = 5
    nrows_per_condition = int(np.ceil(n_plot_grids / ncols))
    total_rows = num_to_plot * nrows_per_condition
    
    fig, axes = plt.subplots(total_rows, ncols, figsize=(ncols * 5, total_rows * 4))
    
    if total_rows == 1:
        axes = axes.reshape(1, -1)
    
    for cond_plot_idx, cond_idx in enumerate(representative_indices):
        for plot_idx, grid_idx in enumerate(grid_indices):
            row = cond_plot_idx * nrows_per_condition + plot_idx // ncols
            col = plot_idx % ncols
            ax = axes[row, col]
            
            # Ground truth as open circles (MORE in temporal)
            ax.semilogx(time_points[marker_indices_temporal], ground_truth[cond_idx, marker_indices_temporal, grid_idx],
                       marker='o', markersize=8, linestyle='none',
                       markeredgewidth=1.8, markeredgecolor=colors['ground_truth'],
                       markerfacecolor='white', color=colors['ground_truth'],
                       zorder=5, label='Ground Truth' if plot_idx == 0 and cond_plot_idx == 0 else "")

            # Prediction line
            ax.semilogx(time_points, predictions[cond_idx, :, grid_idx],
                       color=colors['prediction'], linewidth=2.5, linestyle='-',
                       label='Prediction' if plot_idx == 0 and cond_plot_idx == 0 else "")

            if row == (cond_plot_idx + 1) * nrows_per_condition - 1 or row == total_rows - 1:
                ax.set_xlabel('Time', fontsize=14, fontweight='bold')

            if col == 0:
                ax.set_ylabel('$u(x,t)$', fontsize=14, fontweight='bold')
            
            gt_grid = ground_truth[cond_idx, :, grid_idx]
            pred_grid = predictions[cond_idx, :, grid_idx]
            rmse = np.sqrt(mean_squared_error(gt_grid, pred_grid))
            
            x_val = x_full[grid_idx]
            if plot_idx == 0:
                if num_to_plot == 1:
                    title = f'$x={x_val:.2f}$ (Base IC)\nRMSE={rmse:.2e}'
                else:
                    title = f'$x={x_val:.2f}$ (Cond {cond_plot_idx+1})\nRMSE={rmse:.2e}'
            else:
                title = f'$x={x_val:.2f}$\nRMSE={rmse:.2e}'

            ax.set_title(title, fontsize=14, fontweight='bold', pad=8)

            if plot_idx == 0 and cond_plot_idx == 0:
                ax.legend(frameon=True, fancybox=False, shadow=False,
                         loc='best', fontsize=12, framealpha=0.9)

            ax.set_xlim(positive_time_min, time_points[-1])
            ax.set_ylim(-0.05, 1.1)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True, alpha=0.3, which='both')
    
    # Hide empty subplots
    total_needed = num_to_plot * n_plot_grids
    total_subplots = total_rows * ncols
    for idx in range(total_needed, total_subplots):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{model_output_dir}/trajectories_temporal.pdf')
    plt.savefig(f'{model_output_dir}/trajectories_temporal.png', dpi=300)
    plt.close()

    # 3D Spatial profiles — NC-quality: GT | Prediction | Absolute Error
    from mpl_toolkits.mplot3d import Axes3D

    x_plot = x_full

    # Log-spaced downsampling keeps early-time dynamics without overloading 3D renderer
    n_t_surf_full, n_x_surf_full = 150, 60
    t_idx_surf_full = np.unique(
        np.round(np.logspace(0, np.log10(len(time_points) - 1), n_t_surf_full)).astype(int)
    )
    t_idx_surf_full = np.clip(t_idx_surf_full, 0, len(time_points) - 1)
    x_idx_surf_full = np.linspace(0, len(x_plot) - 1, n_x_surf_full, dtype=int)
    T_mesh_full, X_mesh_full = np.meshgrid(time_points[t_idx_surf_full], x_plot[x_idx_surf_full])

    # Compact version uses lighter sampling so the reduced subfigure keeps clean silhouettes.
    n_t_surf_compact, n_x_surf_compact = 110, 42
    t_idx_surf_compact = np.unique(
        np.round(np.logspace(0, np.log10(len(time_points) - 1), n_t_surf_compact)).astype(int)
    )
    t_idx_surf_compact = np.clip(t_idx_surf_compact, 0, len(time_points) - 1)
    x_idx_surf_compact = np.linspace(0, len(x_plot) - 1, n_x_surf_compact, dtype=int)
    T_mesh_compact, X_mesh_compact = np.meshgrid(
        time_points[t_idx_surf_compact],
        x_plot[x_idx_surf_compact]
    )

    def _style_3d(ax, labelsize=12, grid_color='#cccccc', grid_width=0.6,
                  pane_edge='#cccccc', tick_pad=4):
        """Transparent panes + light grey grid for publication-style 3D axes."""
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor(pane_edge)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo['grid']['color'] = grid_color
            axis._axinfo['grid']['linestyle'] = '--'
            axis._axinfo['grid']['linewidth'] = grid_width
        ax.tick_params(axis='both', labelsize=labelsize, pad=tick_pad)
        ax.zaxis.set_tick_params(labelsize=labelsize, pad=tick_pad)

    def _format_3d_axes(ax, zlabel, zlim, labelsize=14, titlesize=15,
                        compact=False, zticks=None, show_ylabel=True,
                        show_zlabel=True, ztick_formatter=None):
        """Apply consistent camera, aspect, and axis styling for 3D surfaces."""
        _style_3d(
            ax,
            labelsize=10 if compact else 12,
            grid_color='#d7d7d7' if compact else '#cccccc',
            grid_width=0.4 if compact else 0.6,
            pane_edge='#dddddd' if compact else '#cccccc',
            tick_pad=2 if compact else 4
        )
        ax.set_xlabel(r'$t$', fontsize=11 if compact else labelsize,
                      fontweight='bold', labelpad=6 if compact else 12)
        if show_ylabel:
            ax.set_ylabel(r'$x$', fontsize=11 if compact else labelsize,
                          fontweight='bold', labelpad=6 if compact else 12)
        else:
            ax.set_ylabel('')
        if show_zlabel:
            ax.set_zlabel(zlabel, fontsize=11 if compact else labelsize,
                          fontweight='bold', labelpad=6 if compact else 12)
        else:
            ax.set_zlabel('')
        ax.set_xlim(time_points[0], time_points[-1])
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(*zlim)
        ax.set_xticks([0.0, 5.0, 10.0] if time_points[-1] >= 10.0 else np.linspace(time_points[0], time_points[-1], 3))
        ax.set_yticks([0.0, 0.5, 1.0])
        if zticks is not None:
            ax.set_zticks(zticks)
        elif zlim[1] <= 0.03:
            ax.set_zticks(np.linspace(zlim[0], zlim[1], 4))
        else:
            ax.set_zticks([0.0, 0.5, 1.0])
        if ztick_formatter is not None:
            ax.zaxis.set_major_formatter(ztick_formatter)
        ax.view_init(elev=24, azim=48)
        ax.set_box_aspect((1.35, 1.0, 0.82 if compact else 0.9))

    for cond_plot_idx, cond_idx in enumerate(representative_indices):
        # Subsample surfaces
        gt_s  = ground_truth[cond_idx][np.ix_(t_idx_surf_full, x_idx_surf_full)].T  # (n_x, n_t)
        pr_s  = predictions[cond_idx][np.ix_(t_idx_surf_full, x_idx_surf_full)].T
        er_s  = np.abs(gt_s - pr_s)

        vmin_u, vmax_u = 0.0, 1.0
        vmax_e = float(np.nanpercentile(er_s, 99.5))   # robust colour ceiling

        fig = plt.figure(figsize=(24, 7.5))
        fig.patch.set_facecolor('white')

        gt_title = ('Ground Truth (Base IC)' if num_to_plot == 1
                    else f'Ground Truth (Cond {cond_plot_idx + 1})')

        # ── Panel A: Ground Truth ────────────────────────────────────────────
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(T_mesh_full, X_mesh_full, gt_s, cmap='RdBu_r',
                                 vmin=vmin_u, vmax=vmax_u,
                                 alpha=0.93, edgecolor='none', antialiased=True)
        _format_3d_axes(ax1, r'$u(x,t)$', (0.0, 1.0))
        ax1.set_title(gt_title, fontsize=15, fontweight='bold', pad=14)
        cb1 = fig.colorbar(surf1, ax=ax1, fraction=0.025, pad=0.08, shrink=0.70)
        cb1.set_label(r'$u(x,t)$', fontsize=13, fontweight='bold', labelpad=8)
        cb1.ax.tick_params(labelsize=11)
        cb1.outline.set_linewidth(1.5)

        # ── Panel B: Prediction ──────────────────────────────────────────────
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(T_mesh_full, X_mesh_full, pr_s, cmap='RdBu_r',
                                 vmin=vmin_u, vmax=vmax_u,
                                 alpha=0.93, edgecolor='none', antialiased=True)
        _format_3d_axes(ax2, r'$u(x,t)$', (0.0, 1.0))
        ax2.set_title('Prediction', fontsize=15, fontweight='bold', pad=14)
        cb2 = fig.colorbar(surf2, ax=ax2, fraction=0.025, pad=0.08, shrink=0.70)
        cb2.set_label(r'$u(x,t)$', fontsize=13, fontweight='bold', labelpad=8)
        cb2.ax.tick_params(labelsize=11)
        cb2.outline.set_linewidth(1.5)

        # ── Panel C: Absolute Error ──────────────────────────────────────────
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(T_mesh_full, X_mesh_full, er_s, cmap='YlOrRd',
                                 vmin=0, vmax=vmax_e,
                                 alpha=0.93, edgecolor='none', antialiased=True)
        _format_3d_axes(ax3, r'$|\hat{u}-u|$', (0.0, vmax_e * 1.1))
        ax3.set_title(r'Absolute Error  $|\hat{u}-u|$', fontsize=15, fontweight='bold', pad=14)
        cb3 = fig.colorbar(surf3, ax=ax3, fraction=0.025, pad=0.08, shrink=0.70)
        cb3.set_label(r'$|\hat{u}-u|$', fontsize=13, fontweight='bold', labelpad=8)
        cb3.ax.tick_params(labelsize=11)
        cb3.outline.set_linewidth(1.5)

        plt.tight_layout(pad=2.0)

        cond_suffix = f"_cond{cond_plot_idx+1}" if num_to_plot > 1 else ""
        plt.savefig(f'{model_output_dir}/trajectories_spatial_3d{cond_suffix}.pdf', bbox_inches='tight')
        plt.savefig(f'{model_output_dir}/trajectories_spatial_3d{cond_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Compact 3D version for embedding as a subfigure in a larger NC figure.
        gt_compact = ground_truth[cond_idx][np.ix_(t_idx_surf_compact, x_idx_surf_compact)].T
        pr_compact = predictions[cond_idx][np.ix_(t_idx_surf_compact, x_idx_surf_compact)].T
        er_compact = np.abs(gt_compact - pr_compact)

        fig = plt.figure(figsize=(11.0, 3.3))
        fig.patch.set_facecolor('white')
        gs = fig.add_gridspec(
            1, 7,
            width_ratios=[1.0, 1.0, 0.04, 0.12, 1.0, 0.04, 0.02],
            wspace=0.03
        )

        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        cax_u = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 4], projection='3d')
        cax_e = fig.add_subplot(gs[0, 5])

        surf1 = ax1.plot_surface(
            T_mesh_compact, X_mesh_compact, gt_compact,
            cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u,
            alpha=0.96, edgecolor='none', antialiased=True
        )
        surf2 = ax2.plot_surface(
            T_mesh_compact, X_mesh_compact, pr_compact,
            cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u,
            alpha=0.96, edgecolor='none', antialiased=True
        )
        surf3 = ax3.plot_surface(
            T_mesh_compact, X_mesh_compact, er_compact,
            cmap='YlOrRd', vmin=0.0, vmax=COMPACT_3D_ERROR_VMAX,
            alpha=0.96, edgecolor='none', antialiased=True
        )

        _format_3d_axes(ax1, r'$u$', (0.0, 1.0), compact=True, show_zlabel=False)
        _format_3d_axes(ax2, r'$u$', (0.0, 1.0), compact=True,
                        show_ylabel=False, show_zlabel=False)
        _format_3d_axes(
            ax3, r'$|u-\hat{u}|$', (0.0, COMPACT_3D_ERROR_VMAX),
            compact=True, zticks=[0.0, 0.01, 0.02], show_ylabel=False,
            show_zlabel=False,
            ztick_formatter=mpl.ticker.FormatStrFormatter('%.2f')
        )

        ax1.set_title('Ground truth', fontsize=11, fontweight='bold', pad=5)
        ax2.set_title('Prediction', fontsize=11, fontweight='bold', pad=5)
        ax3.set_title('Abs. error', fontsize=11, fontweight='bold', pad=5)

        cb_u = fig.colorbar(surf1, cax=cax_u)
        cax_u.set_title(r'$u$', fontsize=9, fontweight='bold', pad=3)
        cb_u.ax.tick_params(labelsize=9, width=1.0, length=3)
        cb_u.outline.set_linewidth(1.0)

        cb_e = fig.colorbar(surf3, cax=cax_e)
        cb_e.set_label(r'$|u-\hat{u}|$', fontsize=9, fontweight='bold', labelpad=3)
        cb_e.ax.tick_params(labelsize=9, width=1.0, length=3)
        cb_e.outline.set_linewidth(1.0)

        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.90)
        plt.savefig(f'{model_output_dir}/trajectories_spatial_3d_compact{cond_suffix}.pdf', bbox_inches='tight')
        plt.savefig(f'{model_output_dir}/trajectories_spatial_3d_compact{cond_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Original 2D spatial profiles at different times
    n_time_snapshots = 6
    time_indices = np.linspace(0, len(time_points) - 1, n_time_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration
    
    x_plot = x_full
    
    # FEWER markers for spatial plot (open circles)
    n_markers_spatial = 30
    marker_indices_spatial = np.linspace(0, len(x_plot) - 1, n_markers_spatial, dtype=int)
    
    for plot_idx, t_idx in enumerate(time_indices):
        ax = axes[plot_idx]
        cond_idx = representative_indices[0]  # Use first condition for all spatial plots
        
        # Ground truth as open circles (FEWER in spatial)
        ax.plot(x_plot[marker_indices_spatial], ground_truth[cond_idx, t_idx, marker_indices_spatial],
               marker='o', markersize=7, linestyle='none',
               markeredgewidth=1.8, markeredgecolor=colors['ground_truth'],
               markerfacecolor='white', color=colors['ground_truth'],
               zorder=5, label='Ground Truth' if plot_idx == 0 else "")

        # Prediction line
        ax.plot(x_plot, predictions[cond_idx, t_idx, :],
               color=colors['prediction'], linewidth=2.5, linestyle='-',
               label='Prediction' if plot_idx == 0 else "")

        ax.set_xlabel('$x$', fontsize=14, fontweight='bold')
        ax.set_ylabel('$u(x,t)$', fontsize=14, fontweight='bold')

        rmse = np.sqrt(mean_squared_error(ground_truth[cond_idx, t_idx, :],
                                           predictions[cond_idx, t_idx, :]))
        ax.set_title(f'$t={time_points[t_idx]:.3f}$\nRMSE={rmse:.2e}', fontsize=14, fontweight='bold')

        if plot_idx == 0:
            ax.legend(loc='best', fontsize=12)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(f'{model_output_dir}/trajectories_spatial.pdf')
    plt.savefig(f'{model_output_dir}/trajectories_spatial.png', dpi=300)
    plt.close()
    
    print(f"Trajectory plots saved to {model_output_dir}")


def plot_error_analysis(analysis, model_name, n_grid, output_dir='results/teacher_evaluation'):
    """Error analysis visualization for Fisher-KPP."""
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    time_points = analysis['time_points']
    gt = analysis['ground_truth']
    pred = analysis['predictions']
    conds = analysis['test_conditions']

    eps = 1e-12
    C, T, N = gt.shape

    # Handle single condition case
    if C == 1:
        print("  Single condition detected - generating simplified error analysis plots...")
        
        diff = pred - gt
        abs_error = np.abs(diff[0])
        rmse_by_time = np.sqrt(np.mean(diff[0]**2, axis=1))
        mae_by_time = np.mean(np.abs(diff[0]), axis=1)
        
        # Figure 1: Error over time
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.set_title(rf'{model_name}: Error over time', fontsize=15, fontweight='bold')

        ax.loglog(time_points, rmse_by_time + eps, linewidth=3.0, label=r'RMSE')
        ax.loglog(time_points, mae_by_time + eps, linewidth=2.5, linestyle='--', label=r'MAE')

        ax.set_xlabel(r'$t$', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'Error', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.25)
        
        plt.tight_layout()
        plt.savefig(f'{model_output_dir}/error_over_time.pdf')
        plt.savefig(f'{model_output_dir}/error_over_time.png', dpi=300)
        plt.close()
        
        # Figure 2: Error heatmap (space vs time) — NC-quality
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.grid(False)

        log_error = np.log10(abs_error.T + eps)

        x_interior = np.linspace(0, 1, N + 2)[1:-1]
        log_t_min = np.log10(time_points[0] + eps)
        log_t_max = np.log10(time_points[-1])

        # origin='lower': row 0 (x≈0) at bottom, high-x at top — matches y-axis direction
        im = ax.imshow(log_error, aspect='auto', cmap='RdYlBu_r',
                       extent=[log_t_min, log_t_max, x_interior[0], x_interior[-1]],
                       origin='lower', interpolation='bilinear')

        ax.set_xlabel(r'$\log_{10}(t)$', fontsize=16, fontweight='bold', labelpad=8)
        ax.set_ylabel(r'$x$', fontsize=16, fontweight='bold', labelpad=8)
        ax.set_title(
            rf'{model_name}: $\log_{{10}}|\hat{{u}} - u|$',
            fontsize=16, fontweight='bold', pad=10
        )
        ax.tick_params(axis='both', which='major', labelsize=13,
                       width=1.8, length=5, direction='out')
        ax.tick_params(axis='both', which='minor', width=1.2, length=3, direction='out')
        ax.minorticks_on()

        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
        cbar.set_label(r'$\log_{10}|\hat{u} - u|$', fontsize=14,
                       fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=12, width=1.5, length=4)
        cbar.outline.set_linewidth(1.8)

        plt.tight_layout(pad=1.8)
        plt.savefig(f'{model_output_dir}/error_heatmap.pdf')
        plt.savefig(f'{model_output_dir}/error_heatmap.png', dpi=300)
        plt.close()
        
        # Figure 3: Error at different spatial locations
        fig, ax = plt.subplots(figsize=(13, 8))

        x_full = np.linspace(0, 1, N + 2)[1:-1]
        key_indices = [0, N//4, N//2, 3*N//4, N-1]
        colors_line = plt.cm.viridis(np.linspace(0, 1, len(key_indices)))

        for i, idx in enumerate(key_indices):
            ax.loglog(time_points, abs_error[:, idx] + eps,
                     color=colors_line[i], linewidth=2.5, label=f'$x={x_full[idx]:.2f}$')

        ax.set_xlabel(r'$t$', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'|Error|', fontsize=14, fontweight='bold')
        ax.set_title('Error at key spatial locations', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.25)
        
        plt.tight_layout()
        plt.savefig(f'{model_output_dir}/error_by_location.pdf')
        plt.savefig(f'{model_output_dir}/error_by_location.png', dpi=300)
        plt.close()
        
        print(f"Error analysis plots (single condition) saved to {model_output_dir}")
        return

    # Multiple conditions case
    diff = pred - gt
    rmse_ct = np.sqrt(np.mean(diff**2, axis=2))
    mae_ct = np.mean(np.abs(diff), axis=2)

    qs = [0.1, 0.5, 0.9]
    rmse_q = np.quantile(rmse_ct, qs, axis=0)
    mae_q = np.quantile(mae_ct, qs, axis=0)

    tail_rmse = (rmse_q[2] + eps) / (rmse_q[1] + eps)

    # Figure 1: Quantile band + tail index
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(rf'{model_name}: error distribution over time', fontsize=15, fontweight='bold')

    rmse_q_safe = np.maximum(rmse_q, eps)
    mae_q_safe = np.maximum(mae_q, eps)

    ax.loglog(time_points, rmse_q_safe[1], linewidth=3.0, label=r'RMSE median')
    ax.fill_between(time_points, rmse_q_safe[0], rmse_q_safe[2], alpha=0.25, label=r'RMSE 10–90%')

    ax.loglog(time_points, mae_q_safe[1], linewidth=2.5, linestyle='--', label=r'MAE median')
    ax.fill_between(time_points, mae_q_safe[0], mae_q_safe[2], alpha=0.15, label=r'MAE 10–90%')

    ax.set_xlabel(r'$t$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Error', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)

    ax2 = ax.twinx()
    ax2.semilogx(time_points, tail_rmse, linewidth=2.2, linestyle=':', color='gray', label=r'tail = q$_{0.9}$/q$_{0.5}$')
    ax2.set_ylabel(r'q$_{0.9}$/q$_{0.5}$', fontsize=13, fontweight='bold')
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/error_quantile_tail.pdf')
    plt.savefig(f'{model_output_dir}/error_quantile_tail.png', dpi=300)
    plt.close()

    # Figure 2: Condition difficulty ranking
    difficulty = np.median(rmse_ct, axis=1)

    fig, ax = plt.subplots(figsize=(13, 8))
    idx_sorted = np.argsort(difficulty)
    ax.plot(np.arange(C), difficulty[idx_sorted] + eps, linewidth=3.0)
    ax.set_yscale('log')
    ax.set_title(r'Condition difficulty ranking', fontsize=15, fontweight='bold')
    ax.set_xlabel(r'Rank (easy → hard)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Median RMSE over time', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(f'{model_output_dir}/error_condition_difficulty.pdf')
    plt.savefig(f'{model_output_dir}/error_condition_difficulty.png', dpi=300)
    plt.close()
    
    print(f"Error analysis plots saved to {model_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate teacher model on Fisher-KPP')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--num_test_conditions', type=int, default=10, 
                       help='Number of test conditions')
    parser.add_argument('--use_base_ic', action='store_true',
                       help='Test on base initial condition only')
    parser.add_argument('--ic_type', type=str, default='step',
                       choices=['step', 'gaussian', 'sine', 'double_gaussian'],
                       help='Initial condition type')
    parser.add_argument('--epsilon', type=float, default=0.01,
                       help='Diffusion coefficient')
    parser.add_argument('--t_end', type=float, default=10.0,
                       help='End time for evaluation')
    parser.add_argument('--n_time_points', type=int, default=1000,
                       help='Number of time points')
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
    n_grid = evaluator.n_grid
    
    print(f"\nEvaluating {model_name} model on Fisher-KPP (n_grid={n_grid})...")
    
    # Time points (log-spaced to capture fast dynamics)
    t_min = 1e-4
    time_points = np.logspace(np.log10(t_min), np.log10(args.t_end), args.n_time_points)
    time_points = np.concatenate([[0.0], time_points])
    print(f"Time points: {len(time_points)} (from {time_points[0]:.2e} to {time_points[-1]:.2e})")
    
    # Generate test conditions
    if args.use_base_ic:
        print("\nTesting on base initial condition only...")
        test_conditions = generate_test_conditions_fisher_kpp(n_grid, num_conditions=1, 
                                                               ic_type=args.ic_type, use_base_ic=True)
    else:
        print(f"\nGenerating {args.num_test_conditions} test conditions (type: {args.ic_type})...")
        test_conditions = generate_test_conditions_fisher_kpp(n_grid, args.num_test_conditions, 
                                                               ic_type=args.ic_type, use_base_ic=False)
    
    print(f"Test conditions shape: {test_conditions.shape}")
    
    # Evaluate
    results = evaluate_model(evaluator, test_conditions, time_points, n_grid, args.epsilon)
    analysis = analyze_error_over_time(evaluator, test_conditions, time_points, n_grid, args.epsilon)
    
    # Plot results
    print("\nGenerating publication-quality plots...")
    model_name_suffix = "_base_ic" if args.use_base_ic else f"_{args.ic_type}"
    plot_trajectories(results, analysis, model_name + model_name_suffix, n_grid)
    print("Generating error analysis plots...")
    plot_error_analysis(analysis, model_name + model_name_suffix, n_grid)
    
    # Summary
    print("\n" + "="*60)
    print(f"COMPREHENSIVE EVALUATION SUMMARY (FISHER-KPP - {n_grid} GRID POINTS)")
    print("="*60)
    
    if args.use_base_ic:
        print(f"\nTesting Mode: Base Initial Condition Only")
    else:
        print(f"\nTesting Mode: {args.num_test_conditions} conditions (type: {args.ic_type})")
    
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
    print("  - trajectories_temporal.pdf/.png - Solution at key grid points over time")
    print("  - trajectories_spatial_3d.pdf/.png - Full-size 3D GT / prediction / error")
    print("  - trajectories_spatial_3d_compact.pdf/.png - Compact 3D subfigure for NC layouts")
    print("  - trajectories_spatial.pdf/.png - Spatial profiles at different times")
    print("  - error_over_time.pdf/.png - Error evolution")
    print("  - error_heatmap.pdf/.png - Space-time error heatmap")
    print("  - error_by_location.pdf/.png - Error at key spatial locations")
    if num_conditions > 1:
        print("  - error_quantile_tail.pdf/.png - Error quantile bands")
        print("  - error_condition_difficulty.pdf/.png - Condition difficulty ranking")


if __name__ == "__main__":
    main()
