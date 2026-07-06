import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from tqdm import tqdm

# Import models
from models import MLP, ResidualMLP

# Configure matplotlib for publication quality
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Color palette for all methods
COLORS = {
    'analytical': '#000000',  # Black
    'teacher': '#33A02C',     # Green
    'student': '#E31A1C',     # Red
}

COMPACT_3D_ERROR_VMAX = 2.0e-2


def _positive_time_min(time_points):
    """Return the first positive time for log-scale axes."""
    positive_times = np.asarray(time_points)[np.asarray(time_points) > 0]
    return positive_times[0] if positive_times.size else time_points[0]


def _savefig_pair(fig, base_path):
    """Save matching PDF and PNG outputs with publication defaults."""
    fig.savefig(f'{base_path}.pdf', bbox_inches='tight')
    fig.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')


def _style_3d(ax, labelsize=10, grid_color='#d7d7d7',
              grid_width=0.4, pane_edge='#dddddd', tick_pad=2):
    """Transparent panes and light grid, matching test_teacher compact 3D style."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(pane_edge)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['grid']['color'] = grid_color
        axis._axinfo['grid']['linestyle'] = '--'
        axis._axinfo['grid']['linewidth'] = grid_width
    ax.tick_params(axis='both', labelsize=labelsize, pad=tick_pad)
    ax.zaxis.set_tick_params(labelsize=labelsize, pad=tick_pad)


def _format_3d_axes(ax, time_points, zlim, zticks=None, show_ylabel=True,
                    show_zlabel=False, ztick_formatter=None):
    """Apply the compact NC 3D axis contract used by teacher/student subfigures."""
    _style_3d(ax)
    ax.set_xlabel(r'$t$', fontsize=11, fontweight='bold', labelpad=6)
    ax.set_ylabel(r'$x$' if show_ylabel else '', fontsize=11,
                  fontweight='bold', labelpad=6)
    ax.set_zlabel(r'$u$' if show_zlabel else '', fontsize=11,
                  fontweight='bold', labelpad=6)
    ax.set_xlim(time_points[0], time_points[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(*zlim)
    ax.set_xticks([0.0, 5.0, 10.0] if time_points[-1] >= 10.0
                  else np.linspace(time_points[0], time_points[-1], 3))
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks(zticks if zticks is not None else [0.0, 0.5, 1.0])
    if ztick_formatter is not None:
        ax.zaxis.set_major_formatter(ztick_formatter)
    ax.view_init(elev=24, azim=48)
    ax.set_box_aspect((1.35, 1.0, 0.82))


def _style_compact_2d_axes(ax, labelsize=10):
    """Consistent compact 2D styling for NC subfigure exports."""
    ax.tick_params(axis='both', which='major', labelsize=labelsize,
                   width=1.3, length=4, direction='out')
    ax.tick_params(axis='both', which='minor', width=1.0, length=2.5,
                   direction='out')
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def _plot_compact_3d_triplet(output_dir, base_name, cond_suffix, time_points,
                             x_plot, left_data, middle_data, error_data,
                             titles):
    """Create an NC-ready compact 3D triplet with shared u colorbar and fixed error scale."""
    n_t_surf, n_x_surf = 110, 42
    t_idx = np.unique(
        np.round(np.logspace(0, np.log10(len(time_points) - 1), n_t_surf)).astype(int)
    )
    t_idx = np.clip(t_idx, 0, len(time_points) - 1)
    x_idx = np.linspace(0, len(x_plot) - 1, n_x_surf, dtype=int)
    T_mesh, X_mesh = np.meshgrid(time_points[t_idx], x_plot[x_idx])

    left_s = left_data[np.ix_(t_idx, x_idx)].T
    middle_s = middle_data[np.ix_(t_idx, x_idx)].T
    error_s = np.clip(
        np.abs(error_data[np.ix_(t_idx, x_idx)]).T,
        0.0,
        COMPACT_3D_ERROR_VMAX
    )

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
        T_mesh, X_mesh, left_s, cmap='RdBu_r', vmin=0.0, vmax=1.0,
        alpha=0.96, edgecolor='none', antialiased=True
    )
    ax2.plot_surface(
        T_mesh, X_mesh, middle_s, cmap='RdBu_r', vmin=0.0, vmax=1.0,
        alpha=0.96, edgecolor='none', antialiased=True
    )
    surf3 = ax3.plot_surface(
        T_mesh, X_mesh, error_s, cmap='YlOrRd',
        vmin=0.0, vmax=COMPACT_3D_ERROR_VMAX,
        alpha=0.96, edgecolor='none', antialiased=True
    )

    _format_3d_axes(ax1, time_points, (0.0, 1.0), show_zlabel=False)
    _format_3d_axes(ax2, time_points, (0.0, 1.0),
                    show_ylabel=False, show_zlabel=False)
    _format_3d_axes(
        ax3, time_points, (0.0, COMPACT_3D_ERROR_VMAX),
        zticks=[0.0, 0.01, 0.02], show_ylabel=False, show_zlabel=False,
        ztick_formatter=mpl.ticker.FormatStrFormatter('%.2f')
    )

    ax1.set_title(titles[0], fontsize=11, fontweight='bold', pad=5)
    ax2.set_title(titles[1], fontsize=11, fontweight='bold', pad=5)
    ax3.set_title(titles[2], fontsize=11, fontweight='bold', pad=5)

    cb_u = fig.colorbar(surf1, cax=cax_u)
    cax_u.set_title(r'$u$', fontsize=9, fontweight='bold', pad=3)
    cb_u.ax.tick_params(labelsize=9, width=1.0, length=3)
    cb_u.outline.set_linewidth(1.0)

    cb_e = fig.colorbar(surf3, cax=cax_e)
    cb_e.set_label(r'$|u-\hat{u}|$', fontsize=9, fontweight='bold', labelpad=3)
    cb_e.ax.tick_params(labelsize=9, width=1.0, length=3)
    cb_e.outline.set_linewidth(1.0)

    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.90)
    _savefig_pair(fig, f'{output_dir}/{base_name}{cond_suffix}')
    plt.close(fig)


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
        self.n_grid = checkpoint.get('n_grid', training_args.get('n_grid', 100))


def create_model(model_type: str, input_size: int, output_size: int, 
                 hidden_dim: int, num_blocks: int, dropout: float) -> nn.Module:
    """Create model based on type"""
    if model_type == 'MLP':
        return MLP(input_size=input_size, output_size=output_size, 
                   hidden_sizes=[hidden_dim]*num_blocks, dropout=dropout)
    elif model_type == 'ResidualMLP':
        return ResidualMLP(input_size=input_size, output_size=output_size, 
                          hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout)
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
    """Generate initial condition for Fisher-KPP equation."""
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


def create_time_features(time_array):
    """Create log10 time feature (must match training)"""
    t = time_array
    t1 = np.log10(t + 1.0)
    return t1.reshape(-1, 1)


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
        n_grid = config.n_grid
    else:
        model_type = checkpoint.get('model_type', 'ResidualMLP')
        n_grid = checkpoint.get('n_grid', 100)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_blocks = checkpoint.get('num_layers', 3)
        dropout = checkpoint.get('dropout', 0.0)
        
        # Try to infer from state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'input_proj.weight' in state_dict:
                hidden_dim = state_dict['input_proj.weight'].shape[0]
                num_blocks = sum(1 for k in state_dict.keys() if 'blocks.' in k and '.ln.weight' in k)
    
    input_size = n_grid + 1  # time + n_grid initial conditions
    output_size = n_grid     # n_grid solution values
    
    print(f"  Model type: {model_type}")
    print(f"  Input size: {input_size}, Output size: {output_size}")
    print(f"  Hidden dim: {hidden_dim}, Blocks/Layers: {num_blocks}")
    print(f"  Grid points: {n_grid}")
    
    model = create_model(model_type, input_size, output_size, hidden_dim, num_blocks, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    
    return model, checkpoint['X_scaler'], checkpoint.get('y_scaler'), model_type, n_grid


def generate_analytical_solution(initial_condition: np.ndarray, time_points: np.ndarray,
                                  n_grid: int, epsilon: float = 0.01):
    """Generate analytical solution for Fisher-KPP equation."""
    h = 1.0 / (n_grid + 1)
    A = build_laplacian_matrix(n_grid, h)
    
    sol = solve_ivp(
        fun=lambda t, y: fisher_kpp_rhs(t, y, A, epsilon),
        t_span=(time_points[0], time_points[-1]),
        y0=initial_condition,
        method='BDF',
        t_eval=time_points,
        atol=1e-10,
        rtol=1e-8
    )
    
    if not sol.success:
        raise RuntimeError(f"Analytical solution failed: {sol.message}")
    
    return time_points, sol.y.T


def generate_model_predictions(model: nn.Module, initial_condition: np.ndarray,
                                time_points: np.ndarray, X_scaler, device: torch.device, 
                                y_scaler=None, n_grid: int = 100):
    """Generate predictions from model"""
    n_times = len(time_points)
    
    # Create input features: [time, u0_1, ..., u0_n]
    X_pred = np.zeros((n_times, n_grid + 1), dtype=np.float32)
    X_pred[:, 0] = time_points
    X_pred[:, 1:n_grid+1] = initial_condition
    
    # Apply time feature engineering (log10)
    X_copy = X_pred.copy()
    time_features = create_time_features(X_copy[:, 0])
    X_augmented = np.column_stack([time_features, X_copy[:, 1:n_grid+1]])
    
    # Normalize
    X_norm = X_scaler.transform(X_augmented)
    
    X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    # Inverse transform if y_scaler exists
    if y_scaler is not None:
        predictions = y_scaler.inverse_transform(predictions)
    
    # Ensure physical constraints
    predictions = np.clip(predictions, 0.0, 1.0)
    
    return time_points, predictions


def generate_test_conditions(n_grid: int, num_conditions: int = 10, 
                              ic_type: str = 'step', use_base_ic: bool = False):
    """Generate test conditions for Fisher-KPP"""
    np.random.seed(42)
    
    x = np.linspace(0, 1, n_grid + 2)[1:-1]  # Interior points
    
    # Base initial condition
    base_ic = get_fisher_kpp_initial_condition(x, ic_type)
    
    if use_base_ic or num_conditions == 1:
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


def batch_generate_trajectories(generator_func, test_conditions, time_points, 
                                 n_grid, desc="Generating", **kwargs):
    """Batch generate trajectories with progress bar"""
    trajectories = []
    failed = 0
    
    for ic in tqdm(test_conditions, desc=desc):
        try:
            _, traj = generator_func(ic, time_points, n_grid, **kwargs)
            trajectories.append(traj)
        except Exception as e:
            print(f"  Warning: {e}")
            # Fallback to zeros
            trajectories.append(np.zeros((len(time_points), n_grid)))
            failed += 1
    
    if failed > 0:
        print(f"  Warning: {failed}/{len(test_conditions)} conditions failed")
    
    return np.array(trajectories)


def evaluate_all_methods(model, X_scaler, y_scaler, device, test_conditions,
                         time_points, n_grid, epsilon=0.01,
                         teacher_model=None, teacher_X_scaler=None, teacher_y_scaler=None):
    """Unified evaluation for all methods"""
    results = {
        'test_conditions': test_conditions,
        'time_points': time_points,
        'n_grid': n_grid
    }
    
    print("Generating analytical solutions...")
    results['analytical'] = batch_generate_trajectories(
        generate_analytical_solution, test_conditions, time_points, n_grid,
        desc="Analytical", epsilon=epsilon
    )
    
    if teacher_model is not None:
        print("\nGenerating teacher predictions...")
        teacher_trajectories = []
        for ic in tqdm(test_conditions, desc="Teacher"):
            _, traj = generate_model_predictions(
                teacher_model, ic, time_points, teacher_X_scaler, device, 
                teacher_y_scaler, n_grid
            )
            teacher_trajectories.append(traj)
        results['teacher'] = np.array(teacher_trajectories)
    
    print("\nGenerating student predictions...")
    student_trajectories = []
    for ic in tqdm(test_conditions, desc="Student"):
        _, traj = generate_model_predictions(
            model, ic, time_points, X_scaler, device, y_scaler, n_grid
        )
        student_trajectories.append(traj)
    results['student'] = np.array(student_trajectories)
    
    return results


def compute_metrics(results):
    """Compute error metrics for all methods vs analytical"""
    metrics = {}
    gt = results['analytical']
    n_grid = results['n_grid']
    
    for method in ['teacher', 'student']:
        if method in results:
            pred = results[method]
            metrics[method] = {
                'mae_overall': mean_absolute_error(gt.flatten(), pred.flatten()),
                'rmse_overall': np.sqrt(mean_squared_error(gt.flatten(), pred.flatten())),
                'r2_overall': r2_score(gt.flatten(), pred.flatten()),
                'grid_points': {}
            }
            
            # Per grid point metrics
            for i in range(n_grid):
                gt_grid = gt[:, :, i].flatten()
                pred_grid = pred[:, :, i].flatten()
                metrics[method]['grid_points'][f'x_{i+1}'] = {
                    'MAE': mean_absolute_error(gt_grid, pred_grid),
                    'RMSE': np.sqrt(mean_squared_error(gt_grid, pred_grid)),
                    'R2': r2_score(gt_grid, pred_grid)
                }
    
    return metrics


def print_metrics(metrics, n_grid):
    """Print formatted metrics"""
    print(f"\n{'='*70}")
    print("PREDICTION ACCURACY COMPARISON")
    print(f"{'='*70}")
    
    for method, data in metrics.items():
        print(f"\n{method.upper()}:")
        print(f"  Overall: MAE={data['mae_overall']:.4e}, RMSE={data['rmse_overall']:.4e}, R²={data['r2_overall']:.4f}")
        
        # Show key grid points
        key_indices = [0, n_grid//4, n_grid//2, 3*n_grid//4, n_grid-1]
        print(f"  Key grid points (by R²):")
        for idx in key_indices:
            key = f'x_{idx+1}'
            errors = data['grid_points'][key]
            print(f"    {key}: MAE={errors['MAE']:.4e}, R²={errors['R2']:.4f}")
    
    if 'student' in metrics and 'teacher' in metrics:
        print(f"\n{'='*70}")
        print("STUDENT vs TEACHER COMPARISON")
        print(f"{'='*70}")
        
        s = metrics['student']
        t = metrics['teacher']
        
        mae_imp = (1 - s['mae_overall'] / t['mae_overall']) * 100
        rmse_imp = (1 - s['rmse_overall'] / t['rmse_overall']) * 100
        r2_diff = s['r2_overall'] - t['r2_overall']
        
        print(f"  MAE change: {mae_imp:+.2f}%")
        print(f"  RMSE change: {rmse_imp:+.2f}%")
        print(f"  ΔR²: {r2_diff:+.4f}")
    
    print(f"\n{'='*70}")


def plot_trajectories_comparison(results, model_type, output_dir, methods_to_plot=None):
    """Create publication-quality trajectory comparison plots for Fisher-KPP"""
    os.makedirs(output_dir, exist_ok=True)
    
    if methods_to_plot is None:
        methods_to_plot = ['analytical', 'teacher', 'student']
    
    available_methods = [m for m in methods_to_plot if m in results]
    test_conditions = results['test_conditions']
    time_points = results['time_points']
    n_grid = results['n_grid']
    
    np.random.seed(42)
    num_conditions = len(test_conditions)
    num_to_plot = min(3, num_conditions)
    representative_indices = list(range(num_to_plot)) if num_conditions <= 3 else \
                             np.random.choice(num_conditions, size=num_to_plot, replace=False)
    
    # Colors and styles (consistent with test_teacher.py)
    colors = {
        'analytical': '#000000',  # Black - ground truth
        'teacher': '#33A02C',     # Green
        'student': '#E31A1C',     # Red
    }
    linestyles = {'analytical': 'none', 'teacher': '--', 'student': '-'}
    
    # MORE markers for temporal plot (consistent with test_teacher.py)
    n_markers_temporal = 20
    marker_indices_temporal = np.unique(np.round(np.logspace(0, np.log10(len(time_points)-1), n_markers_temporal)).astype(int))
    marker_indices_temporal = np.clip(marker_indices_temporal, 0, len(time_points)-1)
    positive_time_min = _positive_time_min(time_points)
    
    # Select key grid points to plot (10 evenly spaced)
    n_plot_grids = 10
    grid_indices = np.linspace(0, n_grid-1, n_plot_grids, dtype=int)
    x_full = np.linspace(0, 1, n_grid + 2)[1:-1]
    
    # Figure: Solution at key grid points over time
    ncols = 5
    nrows_per_condition = int(np.ceil(n_plot_grids / ncols))
    total_rows = num_to_plot * nrows_per_condition
    
    fig, axes = plt.subplots(total_rows, ncols, figsize=(ncols * 4, total_rows * 3))
    
    if total_rows == 1:
        axes = axes.reshape(1, -1)
    
    for cond_plot_idx, cond_idx in enumerate(representative_indices):
        for plot_idx, grid_idx in enumerate(grid_indices):
            row = cond_plot_idx * nrows_per_condition + plot_idx // ncols
            col = plot_idx % ncols
            ax = axes[row, col]
            
            for method in available_methods:
                data = results[method][cond_idx, :, grid_idx]
                label = method.capitalize() if plot_idx == 0 and cond_plot_idx == 0 else ""
                
                if method == 'analytical':
                    # Ground truth as open circles (consistent with test_teacher.py)
                    ax.semilogx(time_points[marker_indices_temporal], data[marker_indices_temporal],
                               marker='o', markersize=6.5, linestyle='none',
                               markeredgewidth=1.4, markeredgecolor=colors[method],
                               markerfacecolor='white', color=colors[method],
                               zorder=5, label=label)
                else:
                    # Teacher and student as lines
                    ax.semilogx(time_points, data, color=colors[method], 
                               linewidth=2.0, linestyle=linestyles[method], label=label)
            
            if row == (cond_plot_idx + 1) * nrows_per_condition - 1 or row == total_rows - 1:
                ax.set_xlabel('Time', fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel('$u(x,t)$', fontsize=11, fontweight='bold')
            
            gt = results['analytical'][cond_idx, :, grid_idx]
            student_pred = results['student'][cond_idx, :, grid_idx]
            rmse = np.sqrt(mean_squared_error(gt, student_pred))
            r2 = r2_score(gt, student_pred)
            
            x_val = x_full[grid_idx]
            if plot_idx == 0:
                if num_to_plot == 1:
                    title = f'$x={x_val:.2f}$ (Base IC)\nRMSE={rmse:.2e}, R²={r2:.3f}'
                else:
                    title = f'$x={x_val:.2f}$ (Cond {cond_plot_idx+1})\nRMSE={rmse:.2e}, R²={r2:.3f}'
            else:
                title = f'$x={x_val:.2f}$\nRMSE={rmse:.2e}, R²={r2:.3f}'
            ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
            
            if plot_idx == 0 and cond_plot_idx == 0:
                ax.legend(frameon=True, fancybox=False, shadow=False,
                         loc='best', fontsize=9, framealpha=0.9)
            
            ax.set_xlim(positive_time_min, time_points[-1])
            ax.set_ylim(-0.05, 1.1)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.grid(True, alpha=0.3, which='both')
    
    # Hide unused subplots
    total_needed = num_to_plot * n_plot_grids
    total_subplots = total_rows * ncols
    for idx in range(total_needed, total_subplots):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout(pad=1.5)
    plt.savefig(f'{output_dir}/trajectories_temporal.pdf')
    plt.savefig(f'{output_dir}/trajectories_temporal.png', dpi=300)
    plt.close()

    # Compact temporal panel for NC multi-panel figures.
    compact_grid_indices = np.linspace(0, n_grid - 1, 5, dtype=int)
    compact_cond_idx = representative_indices[0]
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.7), sharey=True)

    for plot_idx, grid_idx in enumerate(compact_grid_indices):
        ax = axes[plot_idx]

        for method in available_methods:
            data = results[method][compact_cond_idx, :, grid_idx]
            label = method.capitalize() if plot_idx == 0 else ""

            if method == 'analytical':
                ax.semilogx(
                    time_points[marker_indices_temporal],
                    data[marker_indices_temporal],
                    marker='o', markersize=4.6, linestyle='none',
                    markeredgewidth=1.1, markeredgecolor=colors[method],
                    markerfacecolor='white', color=colors[method],
                    zorder=5, label='Analytical' if plot_idx == 0 else ""
                )
            else:
                ax.semilogx(
                    time_points, data, color=colors[method],
                    linewidth=1.8, linestyle=linestyles[method], label=label
                )

        gt = results['analytical'][compact_cond_idx, :, grid_idx]
        student_pred = results['student'][compact_cond_idx, :, grid_idx]
        rmse = np.sqrt(mean_squared_error(gt, student_pred))

        ax.set_title(f'$x={x_full[grid_idx]:.2f}$\nRMSE={rmse:.1e}',
                     fontsize=9.5, fontweight='bold', pad=4)
        ax.set_xlim(positive_time_min, time_points[-1])
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel(r'$t$', fontsize=10, fontweight='bold')
        if plot_idx == 0:
            ax.set_ylabel(r'$u(x,t)$', fontsize=10, fontweight='bold')
            ax.legend(frameon=True, fancybox=False, shadow=False,
                      loc='best', fontsize=7.5, framealpha=0.9)
        _style_compact_2d_axes(ax, labelsize=8.5)

    fig.tight_layout(pad=0.6, w_pad=0.25)
    _savefig_pair(fig, f'{output_dir}/trajectories_temporal_compact')
    plt.close(fig)

    # 3D Spatial profiles (space vs time surface) - Student vs Analytical
    from mpl_toolkits.mplot3d import Axes3D
    
    x_plot = x_full
    
    for cond_plot_idx, cond_idx in enumerate(representative_indices):
        fig = plt.figure(figsize=(16, 6))
        
        # Analytical 3D surface
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Create mesh
        T_mesh, X_mesh = np.meshgrid(time_points, x_plot)
        Z_analytical = results['analytical'][cond_idx, :, :].T
        
        surf1 = ax1.plot_surface(T_mesh, X_mesh, Z_analytical, cmap='coolwarm', 
                                 alpha=0.85, edgecolor='none', antialiased=True)
        
        ax1.set_xlabel(r'$t$ (time)', fontsize=12, fontweight='bold')
        ax1.set_ylabel(r'$x$ (space)', fontsize=12, fontweight='bold')
        ax1.set_zlabel(r'$u(x,t)$', fontsize=12, fontweight='bold')
        
        if num_to_plot == 1:
            ax1.set_title('Analytical Solution (Base IC)', fontsize=13, fontweight='bold', pad=10)
        else:
            ax1.set_title(f'Analytical Solution (Condition {cond_plot_idx+1})', fontsize=13, fontweight='bold', pad=10)
        
        ax1.set_zlim(0, 1)
        ax1.view_init(elev=25, azim=45)
        cbar1 = fig.colorbar(surf1, ax=ax1, pad=0.1, shrink=0.8)
        cbar1.set_label(r'$u(x,t)$', fontsize=11)
        
        # Student prediction 3D surface
        ax2 = fig.add_subplot(122, projection='3d')
        
        Z_student = results['student'][cond_idx, :, :].T
        
        surf2 = ax2.plot_surface(T_mesh, X_mesh, Z_student, cmap='coolwarm', 
                                 alpha=0.85, edgecolor='none', antialiased=True)
        
        ax2.set_xlabel(r'$t$ (time)', fontsize=12, fontweight='bold')
        ax2.set_ylabel(r'$x$ (space)', fontsize=12, fontweight='bold')
        ax2.set_zlabel(r'$u(x,t)$', fontsize=12, fontweight='bold')
        ax2.set_title('Student Prediction', fontsize=13, fontweight='bold', pad=10)
        
        ax2.set_zlim(0, 1)
        ax2.view_init(elev=25, azim=45)
        cbar2 = fig.colorbar(surf2, ax=ax2, pad=0.1, shrink=0.8)
        cbar2.set_label(r'$u(x,t)$', fontsize=11)
        
        plt.tight_layout()
        
        cond_suffix = f"_cond{cond_plot_idx+1}" if num_to_plot > 1 else ""
        plt.savefig(f'{output_dir}/trajectories_spatial_3d_student{cond_suffix}.pdf', bbox_inches='tight')
        plt.savefig(f'{output_dir}/trajectories_spatial_3d_student{cond_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3D comparison with Teacher (if available)
    if 'teacher' in results:
        for cond_plot_idx, cond_idx in enumerate(representative_indices):
            fig = plt.figure(figsize=(18, 6))
            
            T_mesh, X_mesh = np.meshgrid(time_points, x_plot)
            
            # Teacher 3D surface
            ax1 = fig.add_subplot(131, projection='3d')
            Z_teacher = results['teacher'][cond_idx, :, :].T
            surf1 = ax1.plot_surface(T_mesh, X_mesh, Z_teacher, cmap='coolwarm', 
                                     alpha=0.85, edgecolor='none', antialiased=True)
            ax1.set_xlabel(r'$t$ (time)', fontsize=11, fontweight='bold')
            ax1.set_ylabel(r'$x$ (space)', fontsize=11, fontweight='bold')
            ax1.set_zlabel(r'$u(x,t)$', fontsize=11, fontweight='bold')
            ax1.set_title('Teacher', fontsize=12, fontweight='bold', pad=8)
            ax1.set_zlim(0, 1)
            ax1.view_init(elev=25, azim=45)
            cbar1 = fig.colorbar(surf1, ax=ax1, pad=0.08, shrink=0.8)
            cbar1.set_label(r'$u(x,t)$', fontsize=10)
            
            # Student 3D surface
            ax2 = fig.add_subplot(132, projection='3d')
            Z_student = results['student'][cond_idx, :, :].T
            surf2 = ax2.plot_surface(T_mesh, X_mesh, Z_student, cmap='coolwarm', 
                                     alpha=0.85, edgecolor='none', antialiased=True)
            ax2.set_xlabel(r'$t$ (time)', fontsize=11, fontweight='bold')
            ax2.set_ylabel(r'$x$ (space)', fontsize=11, fontweight='bold')
            ax2.set_zlabel(r'$u(x,t)$', fontsize=11, fontweight='bold')
            ax2.set_title('Student', fontsize=12, fontweight='bold', pad=8)
            ax2.set_zlim(0, 1)
            ax2.view_init(elev=25, azim=45)
            cbar2 = fig.colorbar(surf2, ax=ax2, pad=0.08, shrink=0.8)
            cbar2.set_label(r'$u(x,t)$', fontsize=10)
            
            # Error (Student - Analytical)
            ax3 = fig.add_subplot(133, projection='3d')
            Z_error = results['student'][cond_idx, :, :].T - results['analytical'][cond_idx, :, :].T
            surf3 = ax3.plot_surface(T_mesh, X_mesh, Z_error, cmap='RdBu_r', 
                                     alpha=0.85, edgecolor='none', antialiased=True)
            ax3.set_xlabel(r'$t$ (time)', fontsize=11, fontweight='bold')
            ax3.set_ylabel(r'$x$ (space)', fontsize=11, fontweight='bold')
            ax3.set_zlabel(r'Error', fontsize=11, fontweight='bold')
            ax3.set_title('Error (Student - Analytical)', fontsize=12, fontweight='bold', pad=8)
            ax3.view_init(elev=25, azim=45)
            cbar3 = fig.colorbar(surf3, ax=ax3, pad=0.08, shrink=0.8)
            cbar3.set_label(r'Error', fontsize=10)
            
            plt.tight_layout()
            
            cond_suffix = f"_cond{cond_plot_idx+1}" if num_to_plot > 1 else ""
            plt.savefig(f'{output_dir}/trajectories_spatial_3d_comparison{cond_suffix}.pdf', bbox_inches='tight')
            plt.savefig(f'{output_dir}/trajectories_spatial_3d_comparison{cond_suffix}.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Compact 3D triplets for NC figure assembly.
    for cond_plot_idx, cond_idx in enumerate(representative_indices):
        cond_suffix = f"_cond{cond_plot_idx+1}" if num_to_plot > 1 else ""
        analytical_data = results['analytical'][cond_idx]
        student_data = results['student'][cond_idx]
        student_error = student_data - analytical_data

        _plot_compact_3d_triplet(
            output_dir=output_dir,
            base_name='trajectories_spatial_3d_student_compact',
            cond_suffix=cond_suffix,
            time_points=time_points,
            x_plot=x_plot,
            left_data=analytical_data,
            middle_data=student_data,
            error_data=student_error,
            titles=('Analytical', 'Student', 'Abs. error')
        )

        if 'teacher' in results:
            teacher_data = results['teacher'][cond_idx]
            teacher_student_error = student_data - teacher_data
            _plot_compact_3d_triplet(
                output_dir=output_dir,
                base_name='trajectories_spatial_3d_comparison_compact',
                cond_suffix=cond_suffix,
                time_points=time_points,
                x_plot=x_plot,
                left_data=teacher_data,
                middle_data=student_data,
                error_data=teacher_student_error,
                titles=('Teacher', 'Student', 'Abs. error')
            )

    # Original 2D spatial profiles at different times
    n_time_snapshots = 6
    time_indices = np.linspace(0, len(time_points) - 1, n_time_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # FEWER markers for spatial plot (consistent with test_teacher.py)
    n_markers_spatial = 30
    marker_indices_spatial = np.linspace(0, n_grid - 1, n_markers_spatial, dtype=int)
    
    for plot_idx, t_idx in enumerate(time_indices):
        ax = axes[plot_idx]
        cond_idx = representative_indices[0]  # Use first condition
        
        for method in available_methods:
            data = results[method][cond_idx, t_idx, :]
            label = method.capitalize() if plot_idx == 0 else ""
            
            if method == 'analytical':
                # Ground truth as open circles (consistent with test_teacher.py)
                ax.plot(x_full[marker_indices_spatial], data[marker_indices_spatial],
                       marker='o', markersize=5.5, linestyle='none',
                       markeredgewidth=1.2, markeredgecolor=colors[method],
                       markerfacecolor='white', color=colors[method],
                       zorder=5, label='Ground Truth')
            else:
                # Teacher and student as lines
                ax.plot(x_full, data, color=colors[method], 
                       linewidth=2, linestyle=linestyles[method], label=label)
        
        if plot_idx >= 3:  # Bottom row
            ax.set_xlabel('$x$', fontsize=12, fontweight='bold')
        ax.set_ylabel('$u(x,t)$', fontsize=12, fontweight='bold')
        
        rmse = np.sqrt(mean_squared_error(results['analytical'][cond_idx, t_idx, :], 
                                           results['student'][cond_idx, t_idx, :]))
        ax.set_title(f'$t={time_points[t_idx]:.3f}$\nRMSE={rmse:.2e}', fontsize=11, fontweight='bold')
        
        if plot_idx == 0:
            ax.legend(loc='best', fontsize=9)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=1.5)
    plt.savefig(f'{output_dir}/trajectories_spatial.pdf')
    plt.savefig(f'{output_dir}/trajectories_spatial.png', dpi=300)
    plt.close()

    # Compact spatial snapshots for NC subfigures.
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.0), sharex=True, sharey=True)
    axes = axes.flatten()

    for plot_idx, t_idx in enumerate(time_indices):
        ax = axes[plot_idx]
        cond_idx = representative_indices[0]

        for method in available_methods:
            data = results[method][cond_idx, t_idx, :]

            if method == 'analytical':
                ax.plot(
                    x_full[marker_indices_spatial],
                    data[marker_indices_spatial],
                    marker='o', markersize=4.4, linestyle='none',
                    markeredgewidth=1.1, markeredgecolor=colors[method],
                    markerfacecolor='white', color=colors[method],
                    zorder=5, label='Analytical' if plot_idx == 0 else ""
                )
            else:
                ax.plot(
                    x_full, data, color=colors[method], linewidth=1.8,
                    linestyle=linestyles[method],
                    label=method.capitalize() if plot_idx == 0 else ""
                )

        rmse = np.sqrt(mean_squared_error(results['analytical'][cond_idx, t_idx, :],
                                           results['student'][cond_idx, t_idx, :]))
        ax.set_title(f'$t={time_points[t_idx]:.2g}$  RMSE={rmse:.1e}',
                     fontsize=9.5, fontweight='bold', pad=4)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.05, 1.1)
        if plot_idx >= 3:
            ax.set_xlabel(r'$x$', fontsize=10, fontweight='bold')
        if plot_idx % 3 == 0:
            ax.set_ylabel(r'$u(x,t)$', fontsize=10, fontweight='bold')
        if plot_idx == 0:
            ax.legend(frameon=True, fancybox=False, shadow=False,
                      loc='best', fontsize=7.5, framealpha=0.9)
        _style_compact_2d_axes(ax, labelsize=8.5)

    fig.tight_layout(pad=0.6, w_pad=0.35, h_pad=0.6)
    _savefig_pair(fig, f'{output_dir}/trajectories_spatial_compact')
    plt.close(fig)
    
    print(f"✓ Trajectory plots saved to: {output_dir}")


def plot_error_analysis(results, model_type, output_dir):
    """Error analysis visualization for Fisher-KPP"""
    os.makedirs(output_dir, exist_ok=True)
    
    time_points = results['time_points']
    gt = results['analytical']
    pred = results['student']
    n_grid = results['n_grid']
    
    C, T, N = gt.shape
    eps = 1e-12
    
    diff = pred - gt
    abs_error = np.abs(diff)
    
    # Handle single condition case
    if C == 1:
        print("  Single condition detected - generating simplified error analysis plots...")
        
        rmse_by_time = np.sqrt(np.mean(diff[0]**2, axis=1))
        mae_by_time = np.mean(np.abs(diff[0]), axis=1)
        
        # Figure 1: Error over time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f'{model_type}: Error over time', fontsize=13, fontweight='bold')
        
        ax.loglog(time_points, rmse_by_time + eps, linewidth=2.5, label=r'RMSE')
        ax.loglog(time_points, mae_by_time + eps, linewidth=2.0, linestyle='--', label=r'MAE')
        
        ax.set_xlabel(r'$t$', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'Error', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_over_time.pdf')
        plt.savefig(f'{output_dir}/error_over_time.png', dpi=300)
        plt.close()
        
        # Figure 2: Error heatmap (space vs time)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        log_error = np.log10(abs_error[0].T + eps)
        
        im = ax.imshow(log_error, aspect='auto', cmap='viridis',
                       extent=[np.log10(time_points[0] + eps), np.log10(time_points[-1]), N-0.5, -0.5])
        
        ax.set_xlabel(r'$\log_{10}(t)$', fontsize=12, fontweight='bold')
        ax.set_ylabel('Grid point index', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_type}: $\\log_{{10}}$(|error|) heatmap', fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$\log_{10}$(|error|)', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_heatmap.pdf')
        plt.savefig(f'{output_dir}/error_heatmap.png', dpi=300)
        plt.close()
        
        # Figure 3: Error at different spatial locations
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_full = np.linspace(0, 1, N + 2)[1:-1]
        key_indices = [0, N//4, N//2, 3*N//4, N-1]
        colors_line = plt.cm.viridis(np.linspace(0, 1, len(key_indices)))
        
        for i, idx in enumerate(key_indices):
            ax.loglog(time_points, abs_error[0, :, idx] + eps, 
                     color=colors_line[i], linewidth=2, label=f'$x={x_full[idx]:.2f}$')
        
        ax.set_xlabel(r'$t$', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'|Error|', fontsize=12, fontweight='bold')
        ax.set_title('Error at key spatial locations', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_by_location.pdf')
        plt.savefig(f'{output_dir}/error_by_location.png', dpi=300)
        plt.close()

        # Compact Figure 1: Error over time
        positive_time_min = _positive_time_min(time_points)
        fig, ax = plt.subplots(figsize=(4.8, 3.1))
        ax.loglog(time_points, rmse_by_time + eps, linewidth=2.2, label=r'RMSE')
        ax.loglog(time_points, mae_by_time + eps, linewidth=1.9,
                  linestyle='--', label=r'MAE')
        ax.set_xlim(positive_time_min, time_points[-1])
        ax.set_xlabel(r'$t$', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'Error', fontsize=11, fontweight='bold')
        ax.set_title('Error over time', fontsize=11, fontweight='bold', pad=5)
        ax.legend(frameon=True, fancybox=False, shadow=False,
                  loc='best', fontsize=8, framealpha=0.9)
        _style_compact_2d_axes(ax, labelsize=9)
        fig.tight_layout(pad=0.8)
        _savefig_pair(fig, f'{output_dir}/error_over_time_compact')
        plt.close(fig)

        # Compact Figure 2: Error heatmap in physical space.
        x_full = np.linspace(0, 1, N + 2)[1:-1]
        positive_time_mask = time_points > 0
        heatmap_times = time_points[positive_time_mask]
        log_error_compact = np.log10(abs_error[0, positive_time_mask, :].T + eps)

        fig, ax = plt.subplots(figsize=(5.6, 2.9))
        ax.grid(False)
        im = ax.imshow(
            log_error_compact, aspect='auto', cmap='RdYlBu_r',
            extent=[np.log10(heatmap_times[0]), np.log10(heatmap_times[-1]),
                    x_full[0], x_full[-1]],
            origin='lower', interpolation='bilinear'
        )
        ax.set_xlabel(r'$\log_{10}(t)$', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'$x$', fontsize=11, fontweight='bold')
        ax.set_title(r'$\log_{10}|u-\hat{u}|$', fontsize=11,
                     fontweight='bold', pad=5)
        _style_compact_2d_axes(ax, labelsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label(r'$\log_{10}|u-\hat{u}|$', fontsize=9,
                       fontweight='bold', labelpad=6)
        cbar.ax.tick_params(labelsize=8.5, width=1.0, length=3)
        cbar.outline.set_linewidth(1.0)
        fig.tight_layout(pad=0.8)
        _savefig_pair(fig, f'{output_dir}/error_heatmap_compact')
        plt.close(fig)

        # Compact Figure 3: Error at representative locations.
        fig, ax = plt.subplots(figsize=(4.8, 3.1))
        colors_line = plt.cm.viridis(np.linspace(0, 1, len(key_indices)))
        for i, idx in enumerate(key_indices):
            ax.loglog(
                time_points, abs_error[0, :, idx] + eps,
                color=colors_line[i], linewidth=1.8,
                label=f'$x={x_full[idx]:.2f}$'
            )
        ax.set_xlim(positive_time_min, time_points[-1])
        ax.set_xlabel(r'$t$', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'$|u-\hat{u}|$', fontsize=11, fontweight='bold')
        ax.set_title('Spatial error traces', fontsize=11, fontweight='bold', pad=5)
        ax.legend(frameon=True, fancybox=False, shadow=False,
                  loc='best', fontsize=7.5, framealpha=0.9)
        _style_compact_2d_axes(ax, labelsize=9)
        fig.tight_layout(pad=0.8)
        _savefig_pair(fig, f'{output_dir}/error_by_location_compact')
        plt.close(fig)
        
        print(f"Error analysis plots (single condition) saved to {output_dir}")
        return

    # Multiple conditions case
    rmse_ct = np.sqrt(np.mean(diff**2, axis=2))
    mae_ct = np.mean(np.abs(diff), axis=2)

    qs = [0.1, 0.5, 0.9]
    rmse_q = np.quantile(rmse_ct, qs, axis=0)
    mae_q = np.quantile(mae_ct, qs, axis=0)

    tail_rmse = (rmse_q[2] + eps) / (rmse_q[1] + eps)

    # Figure 1: Quantile band + tail index
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_title(f'{model_type}: error distribution over time', fontsize=13, fontweight='bold')

    rmse_q_safe = np.maximum(rmse_q, eps)
    mae_q_safe = np.maximum(mae_q, eps)

    ax.loglog(time_points, rmse_q_safe[1], linewidth=2.5, label=r'RMSE median')
    ax.fill_between(time_points, rmse_q_safe[0], rmse_q_safe[2], alpha=0.25, label=r'RMSE 10–90%')

    ax.loglog(time_points, mae_q_safe[1], linewidth=2.0, linestyle='--', label=r'MAE median')
    ax.fill_between(time_points, mae_q_safe[0], mae_q_safe[2], alpha=0.15, label=r'MAE 10–90%')

    ax.set_xlabel(r'$t$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    ax2 = ax.twinx()
    ax2.semilogx(time_points, tail_rmse, linewidth=1.8, linestyle=':', color='gray', 
                 label=r'tail = q$_{0.9}$/q$_{0.5}$')
    ax2.set_ylabel(r'q$_{0.9}$/q$_{0.5}$', fontsize=11, fontweight='bold')
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_quantile_tail.pdf')
    plt.savefig(f'{output_dir}/error_quantile_tail.png', dpi=300)
    plt.close()

    # Figure 2: Condition difficulty ranking
    difficulty = np.median(rmse_ct, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    idx_sorted = np.argsort(difficulty)
    ax.plot(np.arange(C), difficulty[idx_sorted] + eps, linewidth=2.5)
    ax.set_yscale('log')
    ax.set_title(r'Condition difficulty ranking', fontsize=13, fontweight='bold')
    ax.set_xlabel(r'Rank (easy → hard)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Median RMSE over time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_condition_difficulty.pdf')
    plt.savefig(f'{output_dir}/error_condition_difficulty.png', dpi=300)
    plt.close()

    # Compact Figure 1: Quantile band + tail index
    positive_time_min = _positive_time_min(time_points)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.loglog(time_points, rmse_q_safe[1], linewidth=2.2,
              label=r'RMSE median')
    ax.fill_between(time_points, rmse_q_safe[0], rmse_q_safe[2],
                    alpha=0.25)
    ax.loglog(time_points, mae_q_safe[1], linewidth=1.9,
              linestyle='--', label=r'MAE median')
    ax.fill_between(time_points, mae_q_safe[0], mae_q_safe[2],
                    alpha=0.15)
    ax.set_xlim(positive_time_min, time_points[-1])
    ax.set_xlabel(r'$t$', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Error', fontsize=11, fontweight='bold')
    ax.set_title('Error distribution', fontsize=11, fontweight='bold', pad=5)
    ax.legend(frameon=True, fancybox=False, shadow=False,
              loc='best', fontsize=7.5, framealpha=0.9)
    _style_compact_2d_axes(ax, labelsize=9)

    ax2 = ax.twinx()
    ax2.semilogx(time_points, tail_rmse, linewidth=1.6,
                 linestyle=':', color='gray')
    ax2.set_ylabel(r'q$_{0.9}$/q$_{0.5}$', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=8.5, width=1.0, length=3,
                    direction='out')
    for spine in ax2.spines.values():
        spine.set_linewidth(1.3)

    fig.tight_layout(pad=0.8)
    _savefig_pair(fig, f'{output_dir}/error_quantile_tail_compact')
    plt.close(fig)

    # Compact Figure 2: Condition difficulty ranking
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    ax.plot(np.arange(C), difficulty[idx_sorted] + eps, linewidth=2.2)
    ax.set_yscale('log')
    ax.set_xlabel(r'Rank (easy to hard)', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Median RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Condition difficulty', fontsize=11, fontweight='bold', pad=5)
    _style_compact_2d_axes(ax, labelsize=9)
    fig.tight_layout(pad=0.8)
    _savefig_pair(fig, f'{output_dir}/error_condition_difficulty_compact')
    plt.close(fig)
    
    print(f"Error analysis plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate student models for Fisher-KPP')
    parser.add_argument('--student_model', type=str, required=True,
                       help='Path to student model')
    parser.add_argument('--teacher_model', type=str, default=None,
                       help='Path to teacher model (optional)')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'])
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
    parser.add_argument('--output_dir', type=str, default='results/student_evaluation')
    
    args = parser.parse_args()
    
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    
    # Load student model
    try:
        model, X_scaler, y_scaler, model_type, n_grid = load_model(
            args.student_model, device, is_student=True
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load teacher model if provided
    teacher_model = teacher_X_scaler = teacher_y_scaler = None
    if args.teacher_model:
        teacher_model, teacher_X_scaler, teacher_y_scaler, _, _ = load_model(
            args.teacher_model, device, is_student=False
        )
    
    # Generate time points (log-spaced to capture fast dynamics)
    t_min = 1e-4
    time_points = np.logspace(np.log10(t_min), np.log10(args.t_end), args.n_time_points)
    time_points = np.concatenate([[0.0], time_points])
    
    # Generate test conditions
    if args.use_base_ic:
        print("\nTesting on base initial condition only...")
        test_conditions = generate_test_conditions(n_grid, num_conditions=1, 
                                                    ic_type=args.ic_type, use_base_ic=True)
    else:
        print(f"\nGenerating {args.num_test_conditions} test conditions (type: {args.ic_type})...")
        test_conditions = generate_test_conditions(n_grid, args.num_test_conditions, 
                                                    ic_type=args.ic_type, use_base_ic=False)
    
    print(f"Test conditions shape: {test_conditions.shape}")
    print(f"Time points: {len(time_points)} (from {time_points[0]:.2e} to {time_points[-1]:.2e})")
    
    output_dir = os.path.join(args.output_dir, f"Student_{model_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EVALUATING STUDENT MODEL ({model_type}) - FISHER-KPP")
    print(f"{'='*60}")
    
    # Evaluate all methods
    results = evaluate_all_methods(
        model, X_scaler, y_scaler, device, test_conditions,
        time_points, n_grid, args.epsilon,
        teacher_model, teacher_X_scaler, teacher_y_scaler
    )
    
    # Compute and print metrics
    metrics = compute_metrics(results)
    print_metrics(metrics, n_grid)
    
    # Generate plots
    print("\nGenerating trajectory plots...")
    plot_trajectories_comparison(results, model_type, output_dir)
    
    print("\nGenerating error analysis plots...")
    plot_error_analysis(results, model_type, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}/")
    print("Compact NC subfigure outputs include:")
    print("  - trajectories_temporal_compact.pdf/.png")
    print("  - trajectories_spatial_compact.pdf/.png")
    print("  - trajectories_spatial_3d_student_compact*.pdf/.png")
    if args.teacher_model:
        print("  - trajectories_spatial_3d_comparison_compact*.pdf/.png")
    print("  - error_*_compact.pdf/.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
