"""
Test Student Model on High-Resolution Teacher-Generated Data.

This script evaluates student models trained via PAKD on the full temporal
dynamics learned by the teacher model, with publication-quality visualizations
consistent with test_teacher.py.

Uses raw data directly (no preprocessing, no scaling) — consistent with
train_teacher_multi.py, teacher_generation.py, HMM_clustering.py, and PAKD.py.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import argparse
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from tqdm import tqdm

# Import models
from models import MLP, ResidualMLP

# Import data loading and evaluation infrastructure from test_teacher
from test_teacher import (
    load_midas_data,
    preprocess_time,
    prepare_raw_data,
    align_baseline_t0,
    aggregate_replicates,
    get_condition_names,
    compute_confidence_interval,
    TeacherModelEvaluator,
    evaluate_model_with_replicates,
    evaluate_by_time_with_replicates,
    evaluate_by_stimuli,
    plot_predictions_vs_ground_truth_publication as teacher_plot_pred_vs_gt,
    plot_protein_performance_publication as teacher_plot_protein_perf,
    plot_timecourse_with_ci_publication as teacher_plot_timecourse,
    plot_error_heatmap_publication as teacher_plot_error_heatmap,
    plot_error_quantiles_over_time as teacher_plot_error_quantiles,
    plot_replicate_analysis as teacher_plot_replicate,
    plot_time_analysis_publication as teacher_plot_time_analysis,
    MIDAS_TREATMENT_PREFIX,
    MIDAS_DATA_AVG_PREFIX,
    MIDAS_DATA_VAL_PREFIX,
    STIMULI_COLORS,
    COLORS as TEACHER_COLORS,
    PROTEIN_DISPLAY_NAMES as TEACHER_PROTEIN_DISPLAY_NAMES,
    KEY_PROTEINS as TEACHER_KEY_PROTEINS,
)

# Configure matplotlib for publication quality (matching test_teacher.py / POLLU style)
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
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

# ============================================================================
# Constants (matching test_teacher.py / teacher_generation.py)
# ============================================================================
KEY_PROTEINS = TEACHER_KEY_PROTEINS

PROTEIN_DISPLAY_NAMES = TEACHER_PROTEIN_DISPLAY_NAMES

COLORS = {
    **TEACHER_COLORS,
    'teacher': '#2ecc71',
    'student': '#e74c3c',
}

# R² quality thresholds — for TEACHER evaluation only
# For student, we use fidelity/retention metrics instead
R2_EXCELLENT = 0.9
R2_GOOD = 0.7

# Student fidelity thresholds (how well student matches teacher)
FIDELITY_EXCELLENT = 0.95  # >95% of teacher's variance captured
FIDELITY_GOOD = 0.85


# ============================================================================
# Utility Functions
# ============================================================================
def get_device(device_str: str = 'auto') -> torch.device:
    """Get the best available device."""
    if device_str == 'auto':
        if torch.backends.mps.is_available():
            name = "Apple Silicon GPU"
            dev = "mps"
        elif torch.cuda.is_available():
            name = "NVIDIA GPU"
            dev = "cuda"
        else:
            name = "CPU"
            dev = "cpu"
    else:
        name = device_str
        dev = device_str
    print(f"Using {name}")
    return torch.device(dev)


def display_name(protein: str) -> str:
    """Get display name for a protein."""
    return PROTEIN_DISPLAY_NAMES.get(protein, protein)


def r2_quality_color(r2: float) -> str:
    """Get color for R² quality tier."""
    if r2 > R2_EXCELLENT:
        return COLORS['good']
    elif r2 > R2_GOOD:
        return COLORS['moderate']
    return COLORS['poor']


def r2_quality_counts(r2_values: list) -> tuple:
    """Count proteins in each R² quality tier. Returns (n_excellent, n_good, n_poor)."""
    n_excellent = sum(1 for r2 in r2_values if r2 > R2_EXCELLENT)
    n_good = sum(1 for r2 in r2_values if R2_GOOD < r2 <= R2_EXCELLENT)
    n_poor = sum(1 for r2 in r2_values if r2 <= R2_GOOD)
    return n_excellent, n_good, n_poor


def _style_ax(ax):
    """Apply consistent POLLU styling to an axis."""
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save_fig(fig, output_dir: str, name: str):
    """Save figure in both PDF and PNG formats, then close."""
    for ext in ('pdf', 'png'):
        fig.savefig(f'{output_dir}/{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved {name} plot")


# ============================================================================
# Model Loading (unified for student & teacher)
# ============================================================================
def _build_model_from_state_dict(state_dict: dict, checkpoint: dict = None,
                                  dropout: float = 0.0) -> tuple:
    """
    Infer architecture and build model from state dict.

    Returns
    -------
    tuple
        (model, input_size, output_size, hidden_dim, num_blocks, model_type_str, arch_str)
    """
    if 'input_proj.weight' in state_dict:
        # ResidualMLP
        input_size = state_dict['input_proj.weight'].shape[1]
        output_size = state_dict['output_proj.weight'].shape[0]
        hidden_dim = state_dict['input_proj.weight'].shape[0]
        num_blocks = sum(1 for k in state_dict if 'blocks.' in k and '.ln.weight' in k)

        if checkpoint:
            input_size = checkpoint.get('input_size', input_size)
            output_size = checkpoint.get('output_size', output_size)

        model = ResidualMLP(
            input_size=input_size, output_size=output_size,
            hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout
        )
        arch_str = f"{num_blocks} blocks, hidden_dim={hidden_dim}"
        return model, input_size, output_size, hidden_dim, num_blocks, 'ResidualMLP', arch_str

    elif 'network.0.weight' in state_dict:
        # MLP
        input_size = state_dict['network.0.weight'].shape[1]
        layer_keys = sorted(
            [k for k in state_dict if 'network.' in k and '.weight' in k],
            key=lambda x: int(x.split('.')[1])
        )
        last_layer_key = layer_keys[-1]
        output_size = state_dict[last_layer_key].shape[0]

        if checkpoint:
            input_size = checkpoint.get('input_size', input_size)
            output_size = checkpoint.get('output_size', output_size)

        hidden_sizes = [state_dict[k].shape[0] for k in layer_keys[:-1]]
        hidden_dim = hidden_sizes[0] if hidden_sizes else 128
        num_blocks = len(hidden_sizes)

        model = MLP(
            input_size=input_size, output_size=output_size,
            hidden_sizes=hidden_sizes, dropout=dropout
        )
        arch_str = f"hidden_sizes={hidden_sizes}"
        return model, input_size, output_size, hidden_dim, num_blocks, 'MLP', arch_str

    raise ValueError("Unknown model architecture in checkpoint")


def load_student_model(model_path: str, device: torch.device) -> tuple:
    """
    Load a student model from checkpoint (raw mode — no scalers).

    Returns
    -------
    tuple
        (model, config_dict)
    """
    print(f"Loading student model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    is_student = any(key in checkpoint for key in
                     ['teacher_model_path', 'training_args', 'projection_state_dict'])
    if not is_student:
        raise ValueError("This script is for student models only. Use test_teacher.py for teacher models.")

    training_args = checkpoint.get('training_args', {})
    state_dict = checkpoint['model_state_dict']
    dropout = training_args.get('student_dropout', 0.0)

    model, input_size, output_size, hidden_dim, num_blocks, model_type, arch_str = \
        _build_model_from_state_dict(state_dict, checkpoint, dropout)

    print(f"  Model type: {model_type}")
    print(f"  Architecture: {arch_str}")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size}")

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    config = {
        'model_type': model_type,
        'input_size': input_size,
        'output_size': output_size,
        'hidden_dim': hidden_dim,
        'num_blocks': num_blocks,
        'dropout': dropout,
        'hidden_layer': checkpoint.get('hidden_layer', training_args.get('hidden_layer', 'last')),
        'teacher_model_path': checkpoint.get('teacher_model_path',
                                             training_args.get('teacher_model', None)),
        'column_info': checkpoint.get('column_info', None),
        'final_r2': checkpoint.get('final_r2', None),
        'r2_by_protein': checkpoint.get('r2_by_protein', None),
    }

    print(f"✓ Student model loaded (raw mode, no scalers)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def load_teacher_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a teacher model from checkpoint (raw mode — no scalers)."""
    print(f"Loading teacher model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    model, input_size, output_size, _, _, model_type, arch_str = \
        _build_model_from_state_dict(state_dict)

    print(f"  Model type: {model_type}")
    print(f"  Architecture: {arch_str}")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size}")

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"✓ Teacher model loaded (raw mode, no scalers)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


# ============================================================================
# Data Loading (consistent with HMM_clustering.py / PAKD.py)
# ============================================================================
def _load_companion_file(path: str) -> list | None:
    """Load a companion text file (one name per line). Returns None if not found."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return None


def _resolve_names(data: np.lib.npyio.NpzFile, key: str,
                   companion_path: str, fallback_count: int,
                   fallback_prefix: str) -> list:
    """Resolve names from npz > companion file > generic fallback."""
    if key in data:
        return list(data[key])
    names = _load_companion_file(companion_path)
    if names is not None:
        return names
    print(f"  ⚠ {key} not found at {companion_path}, using generic names")
    return [f'{fallback_prefix} {i}' for i in range(fallback_count)]


def load_high_res_data(file_path: str) -> tuple:
    """
    Load high-resolution teacher-generated data from .npz file.
    Raw mode — no time log-transform, no scaling.

    Returns
    -------
    tuple
        (X, teacher_predictions, metadata)
    """
    print(f"\nLoading high-resolution data from: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    X = data['X_high_res']
    predictions = data['predictions']

    metadata = {
        'time_points': data['time_points'],
        'treatment_conditions': data['treatment_conditions'],
        'condition_indices': data['condition_indices'],
        'time_indices': data['time_indices'],
        'n_conditions': int(data['n_conditions']),
        'n_time_points': int(data['n_time_points']),
        'n_proteins': int(data['n_proteins']),
    }

    # Resolve companion file paths
    base_name = file_path.replace('.npz', '').rsplit('_high_res', 1)[0]

    metadata['protein_names'] = _resolve_names(
        data, 'protein_names',
        f'{base_name}_protein_names.txt',
        predictions.shape[1], 'Protein'
    )
    metadata['condition_names'] = _resolve_names(
        data, 'condition_names',
        f'{base_name}_condition_names.txt',
        metadata['n_conditions'], 'Condition'
    )

    print(f"  Samples: {len(X):,}")
    print(f"  Conditions: {metadata['n_conditions']}")
    print(f"  Time points: {metadata['n_time_points']}")
    print(f"  Proteins: {metadata['n_proteins']}")
    print(f"  Time range: {data['time_points'].min():.1f} - {data['time_points'].max():.1f} min")
    print(f"  Input range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  Target range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    return X, predictions, metadata


def predict_with_model(model: nn.Module, X: np.ndarray,
                       device: torch.device, batch_size: int = 1024) -> np.ndarray:
    """Generate predictions from model (raw mode — no scaling)."""
    model.eval()
    predictions_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size].astype(np.float32)).to(device)
            predictions_list.append(model(batch).cpu().numpy())
    return np.vstack(predictions_list)


# ============================================================================
# StudentModelEvaluator — wraps student model using TeacherModelEvaluator API
# ============================================================================
class StudentModelEvaluator(TeacherModelEvaluator):
    """
    Evaluator for student models, reusing TeacherModelEvaluator's infrastructure.
    
    Overrides load_model to handle student checkpoint format while inheriting
    the predict() method and compatibility with all test_teacher.py evaluation
    and plotting functions.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        # Don't call super().__init__ — we override load_model entirely
        self.device = device
        self.model = None
        self.X_scaler = None
        self.y_scaler = None
        self.column_info = None
        self.raw_mode = True  # Students always use raw mode
        self._load_student_model(model_path)
    
    def _load_student_model(self, model_path: str):
        """Load student model from checkpoint into TeacherModelEvaluator-compatible format."""
        print(f"Loading student model (as evaluator) from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        training_args = checkpoint.get('training_args', {})
        state_dict = checkpoint['model_state_dict']
        dropout = training_args.get('student_dropout', 0.0)
        
        model, input_size, output_size, hidden_dim, num_blocks, model_type, arch_str = \
            _build_model_from_state_dict(state_dict, checkpoint, dropout)
        
        print(f"  Model type: {model_type}")
        print(f"  Architecture: {arch_str}")
        print(f"  Input size: {input_size}")
        print(f"  Output size: {output_size}")
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        
        # Student models use raw mode — no scalers
        self.X_scaler = None
        self.y_scaler = None
        self.column_info = checkpoint.get('column_info', None)
        
        self.teacher_model_path = checkpoint.get('teacher_model_path',
                                                  training_args.get('teacher_model', None))
        
        print(f"✓ Student model loaded as evaluator (raw mode)")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


# ============================================================================
# Evaluation Functions (high-res dynamics — student-specific)
# ============================================================================
def _safe_r2(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    """Compute R² with variance guard."""
    return r2_score(gt, pred) if np.var(gt) > eps else 0.0


def evaluate_dynamics(teacher_pred: np.ndarray, student_pred: np.ndarray,
                      metadata: dict) -> dict:
    """Evaluate how well student captures temporal dynamics from teacher."""
    condition_indices = metadata['condition_indices']
    time_indices = metadata['time_indices']
    n_conditions = metadata['n_conditions']
    n_time = metadata['n_time_points']
    protein_names = metadata['protein_names']
    eps = 1e-12

    gt_flat = teacher_pred.flatten()
    pred_flat = student_pred.flatten()

    results = {
        'overall': {
            'MAE': mean_absolute_error(gt_flat, pred_flat),
            'RMSE': np.sqrt(mean_squared_error(gt_flat, pred_flat)),
            'R2': r2_score(gt_flat, pred_flat),
            'Relative_Error': np.mean(np.abs(gt_flat - pred_flat) / (np.abs(gt_flat) + eps)),
        },
        'by_time': {},
        'by_condition': {},
        'by_protein': {},
    }

    # By time point
    for t_idx in range(n_time):
        mask = time_indices == t_idx
        if mask.sum() > 0:
            gt_t, pred_t = teacher_pred[mask].flatten(), student_pred[mask].flatten()
            results['by_time'][t_idx] = {
                'MAE': mean_absolute_error(gt_t, pred_t),
                'RMSE': np.sqrt(mean_squared_error(gt_t, pred_t)),
                'R2': _safe_r2(gt_t, pred_t),
                'n_samples': int(mask.sum()),
            }

    # By condition
    for c_idx in range(n_conditions):
        mask = condition_indices == c_idx
        if mask.sum() > 0:
            gt_c, pred_c = teacher_pred[mask].flatten(), student_pred[mask].flatten()
            results['by_condition'][c_idx] = {
                'MAE': mean_absolute_error(gt_c, pred_c),
                'RMSE': np.sqrt(mean_squared_error(gt_c, pred_c)),
                'R2': _safe_r2(gt_c, pred_c),
                'n_samples': int(mask.sum()),
            }

    # By protein
    for i, protein in enumerate(protein_names):
        gt_p, pred_p = teacher_pred[:, i], student_pred[:, i]
        results['by_protein'][protein] = {
            'MAE': mean_absolute_error(gt_p, pred_p),
            'RMSE': np.sqrt(mean_squared_error(gt_p, pred_p)),
            'R2': _safe_r2(gt_p, pred_p),
            'Correlation': np.corrcoef(gt_p, pred_p)[0, 1] if np.std(gt_p) > eps else 0.0,
            'Relative_Error': np.mean(np.abs(gt_p - pred_p) / (np.abs(gt_p) + eps)),
        }

    return results


def print_dynamics_evaluation(results: dict, metadata: dict):
    """Print dynamics evaluation results in formatted output."""
    protein_names = metadata['protein_names']
    condition_names = metadata.get('condition_names', [])

    print(f"\n{'='*70}")
    print("DYNAMICS EVALUATION (Student vs Teacher on High-Res Data)")
    print("(Raw mode — no preprocessing)")
    print(f"{'='*70}")

    overall = results['overall']
    print(f"\nOVERALL METRICS:")
    print(f"  MAE:  {overall['MAE']:.6f}")
    print(f"  RMSE: {overall['RMSE']:.6f}")
    print(f"  R²:   {overall['R2']:.6f}")
    print(f"  Relative Error: {overall['Relative_Error']:.4f}")

    # Time phase analysis
    time_indices = sorted(results['by_time'].keys())
    n_times = len(time_indices)
    thirds = [n_times // 3, 2 * n_times // 3]
    phases = [
        ('Early', time_indices[:thirds[0]], 0, max(0, thirds[0] - 1)),
        ('Mid', time_indices[thirds[0]:thirds[1]], thirds[0], max(thirds[0], thirds[1] - 1)),
        ('Late', time_indices[thirds[1]:], thirds[1], n_times - 1),
    ]
    print(f"\nERROR BY TIME PHASE:")
    for name, idxs, lo, hi in phases:
        mae = np.mean([results['by_time'][t]['MAE'] for t in idxs]) if idxs else 0
        print(f"  {name} phase MAE (t_idx {lo}-{hi}):   {mae:.6f}")

    # Condition analysis
    cond_r2 = [results['by_condition'][c]['R2'] for c in sorted(results['by_condition'])]
    print(f"\nCONDITION-WISE TRAJECTORY R²:")
    for stat, val in [('Mean', np.mean), ('Std', np.std), ('Min', np.min), ('Max', np.max)]:
        print(f"  {stat}: {val(cond_r2):.4f}")

    sorted_conds = sorted(results['by_condition'].items(), key=lambda x: x[1]['R2'], reverse=True)
    for label, items in [('Best', sorted_conds[:5]), ('Worst', sorted_conds[-5:])]:
        print(f"\n  {label} 5 conditions (by trajectory R²):")
        for c_idx, m in items:
            cname = condition_names[c_idx] if c_idx < len(condition_names) else f'Condition {c_idx}'
            print(f"    {cname}: R²={m['R2']:.4f}, MAE={m['MAE']:.6f}")

    # Protein analysis
    prot_sorted = sorted(protein_names, key=lambda p: results['by_protein'][p]['R2'], reverse=True)
    for label, items in [('BEST', prot_sorted[:5]), ('WORST', prot_sorted[-5:])]:
        print(f"\nTOP 5 {label} CAPTURED PROTEINS (by R²):")
        for p in items:
            m = results['by_protein'][p]
            print(f"  {display_name(p)}: R²={m['R2']:.4f}, MAE={m['MAE']:.6f}")

    r2_values = [results['by_protein'][p]['R2'] for p in protein_names]
    n_exc, n_good, n_poor = r2_quality_counts(r2_values)
    n_total = len(r2_values)
    print(f"\nPROTEIN PREDICTION QUALITY:")
    print(f"  Excellent (R² > {R2_EXCELLENT}):  {n_exc}/{n_total} proteins")
    print(f"  Good ({R2_GOOD} < R² ≤ {R2_EXCELLENT}): {n_good}/{n_total} proteins")
    print(f"  Poor (R² ≤ {R2_GOOD}):       {n_poor}/{n_total} proteins")
    print(f"\n{'='*70}")


# ============================================================================
# Raw Data Evaluation — reusing test_teacher.py infrastructure
# ============================================================================
def evaluate_student_vs_raw_data(student_model_path: str, raw_data_path: str,
                                  device: torch.device) -> dict:
    """
    Evaluate student model against raw experimental data using 
    TeacherModelEvaluator infrastructure from test_teacher.py.
    
    Creates a StudentModelEvaluator (which inherits TeacherModelEvaluator),
    loads raw MIDAS data via load_midas_data(), aggregates replicates via
    aggregate_replicates(), and evaluates via evaluate_model_with_replicates().
    
    This guarantees identical data handling as test_teacher.py.
    
    Parameters
    ----------
    student_model_path : str
        Path to student model checkpoint
    raw_data_path : str
        Path to raw MIDAS CSV file
    device : torch.device
        Computation device
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'evaluator': StudentModelEvaluator instance
        - 'results': evaluation results from evaluate_model_with_replicates
        - 'aggregated_data': aggregated replicate data
        - 'column_info': column information
        - 'time_results': per-time-point results
        - 'stimuli_results': per-stimulus results
        - 'X': raw input features
        - 'y': raw target values
        - 'df': raw DataFrame
    """
    print("\n" + "=" * 60)
    print("EVALUATING STUDENT vs RAW EXPERIMENTAL DATA")
    print("(Using test_teacher.py infrastructure)")
    print("=" * 60)
    
    # 1. Create StudentModelEvaluator (inherits TeacherModelEvaluator)
    evaluator = StudentModelEvaluator(student_model_path, device)
    
    # 2. Load raw data using test_teacher.py's load_midas_data
    print("\nLoading raw MIDAS data...")
    X, y, column_info, df = load_midas_data(raw_data_path)
    print(f"  Total samples: {len(X)}")
    print(f"  Input features: {X.shape[1]}")
    print(f"  Output proteins: {y.shape[1]}")
    
    # 3. Prepare data — raw mode (no baseline alignment, no aggregation preprocessing)
    #    Use prepare_raw_data for individual samples (matching student's raw training mode)
    aggregated_data = prepare_raw_data(X, y, column_info, df)
    n_unique = len(aggregated_data)
    print(f"  Individual samples (raw mode): {n_unique}")
    
    # 4. Evaluate using test_teacher.py's evaluate_model_with_replicates
    results = evaluate_model_with_replicates(evaluator, aggregated_data, column_info)
    
    # 5. Stratified evaluations
    time_results = evaluate_by_time_with_replicates(results, aggregated_data)
    stimuli_results = evaluate_by_stimuli(results, aggregated_data)
    
    return {
        'evaluator': evaluator,
        'results': results,
        'aggregated_data': aggregated_data,
        'column_info': column_info,
        'time_results': time_results,
        'stimuli_results': stimuli_results,
        'X': X,
        'y': y,
        'df': df,
    }


def evaluate_teacher_vs_raw_data(teacher_model_path: str, raw_data_path: str,
                                  device: torch.device) -> dict:
    """
    Evaluate teacher model against raw experimental data for comparison.
    Uses TeacherModelEvaluator directly from test_teacher.py.
    
    Parameters
    ----------
    teacher_model_path : str
        Path to teacher model checkpoint
    raw_data_path : str
        Path to raw MIDAS CSV file
    device : torch.device
        Computation device
        
    Returns
    -------
    dict
        Same structure as evaluate_student_vs_raw_data
    """
    print("\n" + "=" * 60)
    print("EVALUATING TEACHER vs RAW EXPERIMENTAL DATA (reference)")
    print("(Using test_teacher.py infrastructure)")
    print("=" * 60)
    
    # 1. Create TeacherModelEvaluator directly
    evaluator = TeacherModelEvaluator(teacher_model_path, device)
    
    # 2. Load raw data
    print("\nLoading raw MIDAS data...")
    X, y, column_info, df = load_midas_data(raw_data_path)
    print(f"  Total samples: {len(X)}")
    
    # 3. Match teacher's raw_mode flag
    if evaluator.raw_mode:
        aggregated_data = prepare_raw_data(X, y, column_info, df)
    else:
        y = align_baseline_t0(X, y, column_info)
        aggregated_data = aggregate_replicates(X, y, column_info, df)
    
    n_unique = len(aggregated_data)
    print(f"  Conditions: {n_unique}")
    
    # 4. Evaluate
    results = evaluate_model_with_replicates(evaluator, aggregated_data, column_info)
    time_results = evaluate_by_time_with_replicates(results, aggregated_data)
    stimuli_results = evaluate_by_stimuli(results, aggregated_data)
    
    return {
        'evaluator': evaluator,
        'results': results,
        'aggregated_data': aggregated_data,
        'column_info': column_info,
        'time_results': time_results,
        'stimuli_results': stimuli_results,
        'X': X,
        'y': y,
        'df': df,
    }


# ============================================================================
# Publication-Quality Plotting — Student-specific (high-res dynamics)
# ============================================================================
def _organize_raw_data_by_condition(raw_eval: dict, protein_names: list,
                                     condition_names: list) -> dict:
    """
    Organize raw experimental data by condition name for trajectory overlay.
    
    Uses aggregated_data from test_teacher.py's aggregate_replicates/prepare_raw_data.
    
    Parameters
    ----------
    raw_eval : dict
        Output from evaluate_student_vs_raw_data or evaluate_teacher_vs_raw_data
    protein_names : list
        Protein names from high-res metadata
    condition_names : list
        Condition names from high-res metadata
        
    Returns
    -------
    dict
        Maps condition_name -> protein -> {times, means, stds, all_points}
    """
    aggregated = raw_eval['aggregated_data']
    raw_protein_names = raw_eval['column_info']['protein_names']
    
    # Build lookup: base_condition (stim|inhib) -> list of entries
    base_lookup = {}
    for key, data in aggregated.items():
        stim = data['stimuli']
        inhib = data['inhibitors']
        base_key = f"{stim}|{inhib}"
        
        if base_key not in base_lookup:
            base_lookup[base_key] = []
        
        base_lookup[base_key].append({
            'time': data['time'],
            'y_mean': data['y_mean'],
            'y_std': data['y_std'],
            'y_all': data['y_all'],
        })
    
    print(f"  Raw data base conditions ({len(base_lookup)}):")
    for bk in sorted(base_lookup.keys())[:10]:
        n_t = len(base_lookup[bk])
        print(f"    {bk} ({n_t} time points)")
    if len(base_lookup) > 10:
        print(f"    ... ({len(base_lookup) - 10} more)")
    
    # Build alias map for flexible matching
    alias_map = {}
    for bk in base_lookup:
        alias_map[bk] = bk
        alias_map[bk.lower()] = bk
        bk_alt = bk.replace('None|', 'No_Stimuli|').replace('|None', '|No_Inhibitor')
        alias_map[bk_alt] = bk
        alias_map[bk_alt.lower()] = bk
        alias_map[bk.replace(' ', '').replace('_', '').lower()] = bk
        alias_map[bk_alt.replace(' ', '').replace('_', '').lower()] = bk
    
    # Map condition_names to raw data
    result = {}
    n_matched = 0
    n_unmatched = 0
    
    for cname in condition_names:
        matched_key = None
        
        # Try multiple matching strategies
        for candidate in [cname, cname.lower(), 
                          cname.replace(' ', '').replace('_', '').lower()]:
            if candidate in alias_map:
                matched_key = alias_map[candidate]
                break
        
        # Try splitting and normalizing
        if matched_key is None:
            parts = cname.split('|')
            if len(parts) >= 2:
                stim_part = parts[0].strip()
                inhib_part = parts[1].strip()
                if not stim_part or stim_part.lower() in ('none', 'no_stimuli', ''):
                    stim_part = 'None'
                if not inhib_part or inhib_part.lower() in ('none', 'no_inhibitor', ''):
                    inhib_part = 'None'
                candidate = f"{stim_part}|{inhib_part}"
                if candidate in alias_map:
                    matched_key = alias_map[candidate]
                elif candidate in base_lookup:
                    matched_key = candidate
        
        if matched_key is None:
            n_unmatched += 1
            continue
        
        n_matched += 1
        result[cname] = {}
        entries = base_lookup[matched_key]
        
        for protein in protein_names:
            if protein not in raw_protein_names:
                continue
            p_idx = raw_protein_names.index(protein)
            
            times, means, stds, all_points = [], [], [], []
            for entry in entries:
                times.append(entry['time'])
                means.append(entry['y_mean'][p_idx])
                stds.append(entry['y_std'][p_idx])
                if entry['y_all'] is not None and len(entry['y_all']) > 0:
                    all_points.append((entry['time'], entry['y_all'][:, p_idx].tolist()))
            
            if times:
                sort_idx = np.argsort(times)
                result[cname][protein] = {
                    'times': np.array([times[k] for k in sort_idx]),
                    'means': np.array([means[k] for k in sort_idx]),
                    'stds': np.array([stds[k] for k in sort_idx]),
                    'all_points': [all_points[k] for k in sort_idx] if all_points else None,
                }
    
    print(f"  Raw data overlay: {n_matched}/{n_matched + n_unmatched} conditions matched")
    if n_unmatched > 0:
        print(f"  ⚠ {n_unmatched} conditions not matched")
    
    return result


def plot_trajectory_comparison_publication(teacher_pred: np.ndarray, student_pred: np.ndarray,
                                           metadata: dict, model_name: str, output_dir: str,
                                           n_conditions_to_plot: int = 6,
                                           raw_eval: dict = None,
                                           results: dict = None):
    """
    Publication-quality trajectory comparison plots (POLLU style).
    
    Generates a 4×4 grid: 4 key proteins × 4 non-trivial conditions.
    If raw_eval is provided, overlays raw experimental data points.
    """
    time_points = metadata['time_points']
    condition_indices = metadata['condition_indices']
    time_indices = metadata['time_indices']
    protein_names = metadata['protein_names']
    condition_names = metadata.get('condition_names', [])

    # Select 4 key proteins
    available_proteins = [p for p in KEY_PROTEINS if p in protein_names]
    if len(available_proteins) < 4:
        available_proteins = protein_names[:4]
    available_proteins = available_proteins[:4]

    # Filter out none|none / empty conditions AND inhibitor-only conditions
    unique_conditions = np.unique(condition_indices)
    valid_conditions = []
    for cond_idx in unique_conditions:
        if cond_idx < len(condition_names):
            cname = condition_names[cond_idx]
            parts = cname.split('|')
            stim_part = parts[0].strip() if len(parts) > 0 else ''
            if not stim_part or stim_part.lower() in ('none', 'no_stimuli'):
                continue
        valid_conditions.append(cond_idx)

    if len(valid_conditions) == 0:
        # Fallback to non-trivial conditions
        for cond_idx in unique_conditions:
            if cond_idx < len(condition_names):
                cname = condition_names[cond_idx]
                parts = cname.split('|')
                stim_part = parts[0].strip() if len(parts) > 0 else ''
                inhib_part = parts[1].strip() if len(parts) > 1 else ''
                if (not stim_part or stim_part.lower() in ('none', 'no_stimuli')) and \
                   (not inhib_part or inhib_part.lower() in ('none', 'no_inhibitor')):
                    continue
            valid_conditions.append(cond_idx)
    
    if len(valid_conditions) == 0:
        valid_conditions = list(unique_conditions)

    step = max(1, len(valid_conditions) // 4)
    conditions_to_plot = valid_conditions[::step][:4]

    n_prot = len(available_proteins)
    n_cond = len(conditions_to_plot)

    fig, axes = plt.subplots(n_prot, n_cond,
                             figsize=(3.8 * n_cond, 2.8 * n_prot))
    if n_prot == 1:
        axes = axes.reshape(1, -1)
    if n_cond == 1:
        axes = axes.reshape(-1, 1)

    # Organize raw data for overlay
    raw_by_condition = None
    if raw_eval is not None:
        raw_by_condition = _organize_raw_data_by_condition(
            raw_eval, protein_names, condition_names
        )

    for i, protein in enumerate(available_proteins):
        protein_idx = protein_names.index(protein)
        for j, cond_idx in enumerate(conditions_to_plot):
            ax = axes[i, j]
            mask = condition_indices == cond_idx
            times = time_points[time_indices[mask]]
            gt = teacher_pred[mask, protein_idx]
            pred = student_pred[mask, protein_idx]

            sort_idx = np.argsort(times)
            times_s, gt_s, pred_s = times[sort_idx], gt[sort_idx], pred[sort_idx]

            # Look up raw data for this condition + protein
            cname_for_raw = condition_names[cond_idx] if cond_idx < len(condition_names) else None
            raw_cond_protein = None
            if raw_by_condition is not None and cname_for_raw and cname_for_raw in raw_by_condition:
                raw_cond = raw_by_condition[cname_for_raw]
                if protein in raw_cond and len(raw_cond[protein]['times']) > 0:
                    raw_cond_protein = raw_cond[protein]

            # Trim trajectories to start from t=1
            t_start = 1.0
            trim_mask = times_s >= t_start
            times_s = times_s[trim_mask]
            gt_s = gt_s[trim_mask]
            pred_s = pred_s[trim_mask]

            # Teacher (solid line, black)
            ax.plot(times_s, gt_s, color=COLORS['ground_truth'], linewidth=2.0,
                    linestyle='-', alpha=0.85, zorder=4,
                    label='Teacher' if i == 0 and j == 0 else '')

            # Student (dashed line, red/orange)
            ax.plot(times_s, pred_s, color=COLORS['prediction'], linewidth=2.2,
                    linestyle='--', alpha=0.9, zorder=5,
                    label='Student' if i == 0 and j == 0 else '')

            # Raw experimental data overlay
            if raw_cond_protein is not None:
                raw_times = np.array(raw_cond_protein['times'])
                raw_means = np.array(raw_cond_protein['means'])
                raw_stds = np.array(raw_cond_protein['stds'])
                raw_all = raw_cond_protein.get('all_points', None)

                # Individual replicates
                if raw_all is not None:
                    first_rep = True
                    for rt, rv_list in raw_all:
                        for rv in rv_list:
                            ax.scatter(rt, rv, marker='.', s=15,
                                      color='#7f8c8d', alpha=0.4, zorder=3,
                                      label='Replicates' if first_rep and i == 0 and j == 0 else '')
                            first_rep = False

                # Mean ± std (diamonds with error bars)
                has_err = raw_stds > 1e-10
                if np.any(has_err):
                    ax.errorbar(raw_times[has_err], raw_means[has_err],
                               yerr=raw_stds[has_err],
                               fmt='D', markersize=6, color='#2980b9',
                               ecolor='#2980b9', elinewidth=1.0, capsize=3,
                               markeredgecolor='black', markeredgewidth=0.8,
                               alpha=0.85, zorder=6,
                               label='Exp. data' if i == 0 and j == 0 else '')
                if np.any(~has_err):
                    ax.scatter(raw_times[~has_err], raw_means[~has_err],
                              marker='D', s=36, color='#2980b9',
                              edgecolors='black', linewidths=0.8,
                              alpha=0.85, zorder=6,
                              label='Exp. data' if i == 0 and j == 0 and not np.any(has_err) else '')

            ax.set_xscale('symlog', linthresh=1)
            if i == n_prot - 1:
                ax.set_xlabel('Time (min)', fontsize=10)
            if j == 0:
                ax.set_ylabel(display_name(protein), fontsize=10, fontweight='bold')
            if i == 0:
                cname_full = condition_names[cond_idx] if cond_idx < len(condition_names) else f'Cond. {cond_idx}'
                parts = cname_full.split('|')
                if len(parts) >= 2:
                    stim_d = parts[0].strip()
                    inhib_d = parts[1].strip()
                    if inhib_d.lower() in ('none', 'no_inhibitor', ''):
                        cname_display = stim_d
                    elif stim_d.lower() in ('none', 'no_stimuli', ''):
                        cname_display = f'{inhib_d} only'
                    else:
                        cname_display = f'{stim_d} + {inhib_d}'
                else:
                    cname_display = cname_full
                    stim_d = cname_full
                
                stim_color = STIMULI_COLORS.get(stim_d, 'black') if len(parts) >= 2 else 'black'
                ax.set_title(cname_display, fontsize=12, fontweight='bold', color=stim_color)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='best', framealpha=0.9)
            _style_ax(ax)
            ax.tick_params(labelsize=8)

    suffix = ' (with Experimental Data)' if raw_eval is not None else ''
    plt.suptitle(f'{model_name}: Student vs Teacher Dynamics{suffix}',
                fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'trajectory_comparison')


def plot_predictions_vs_teacher_publication(teacher_pred: np.ndarray, student_pred: np.ndarray,
                                            metadata: dict, results: dict,
                                            model_name: str, output_dir: str):
    """Publication-quality scatter plots: student vs teacher predictions."""
    protein_names = metadata['protein_names']
    n_proteins = len(protein_names)
    n_cols = 6
    n_rows = (n_proteins + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows))
    axes_flat = axes.flatten()

    for i, protein in enumerate(protein_names):
        ax = axes_flat[i]
        gt, pred = teacher_pred[:, i], student_pred[:, i]

        if len(gt) > 3000:
            idx = np.random.choice(len(gt), 3000, replace=False)
            gt_plot, pred_plot = gt[idx], pred[idx]
        else:
            gt_plot, pred_plot = gt, pred

        ax.scatter(gt_plot, pred_plot, alpha=0.4, s=15, edgecolors='none',
                  c=COLORS['prediction'])

        min_val, max_val = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                '--', color='black', linewidth=1.5, alpha=0.7, zorder=0)

        r2 = results['by_protein'][protein]['R2']
        ax.set_title(f'{display_name(protein)}\n$R^2$={r2:.3f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Teacher', fontsize=9)
        ax.set_ylabel('Student', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect('equal', adjustable='box')
        _style_ax(ax)

    for i in range(n_proteins, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle(f'{model_name}: Student vs Teacher Predictions',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'pred_vs_teacher_publication')


def plot_protein_performance_publication(results: dict, metadata: dict,
                                         model_name: str, output_dir: str):
    """Publication-quality per-protein performance bar charts."""
    protein_names = metadata['protein_names']
    metrics = results['by_protein']

    r2_vals = [metrics[p]['R2'] for p in protein_names]
    rmse_vals = [metrics[p]['RMSE'] for p in protein_names]
    mae_vals = [metrics[p]['MAE'] for p in protein_names]
    sorted_idx = np.argsort(r2_vals)[::-1]
    names = [display_name(protein_names[i]) for i in sorted_idx]

    panels = [
        ('$R^2$', [r2_vals[i] for i in sorted_idx],
         [r2_quality_color(r2_vals[i]) for i in sorted_idx], '$R^2$ by Protein\n(Student vs Teacher)', (0, 1.0)),
        ('RMSE', [rmse_vals[i] for i in sorted_idx], '#3498db', 'RMSE by Protein', None),
        ('MAE', [mae_vals[i] for i in sorted_idx], '#9b59b6', 'MAE by Protein', None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    for ax, (xlabel, values, colors, title, xlim) in zip(axes, panels):
        ax.barh(range(len(protein_names)), values, color=colors,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(protein_names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25, axis='x')
        _style_ax(ax)
        if xlim:
            ax.set_xlim(*xlim)
            ax.axvline(x=R2_EXCELLENT, color='green', linestyle='--', alpha=0.7, label=f'Excellent ({R2_EXCELLENT})')
            ax.axvline(x=R2_GOOD, color='orange', linestyle='--', alpha=0.7, label=f'Good ({R2_GOOD})')
            ax.legend(fontsize=9, loc='lower right')

    plt.suptitle(f'{model_name}: Per-Protein Performance', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'protein_performance_publication')


def plot_error_quantiles_over_time(teacher_pred: np.ndarray, student_pred: np.ndarray,
                                   metadata: dict, model_name: str, output_dir: str):
    """POLLU-style error quantile plot over time with 10–90% bands."""
    time_points = metadata['time_points']
    time_indices = metadata['time_indices']
    n_time = metadata['n_time_points']
    qs = [0.1, 0.5, 0.9]
    eps = 1e-12

    rmse_by_time, mae_by_time, times_used = [], [], []
    for t_idx in range(n_time):
        mask = time_indices == t_idx
        if not mask.any():
            continue
        diff = teacher_pred[mask] - student_pred[mask]
        rmse_by_time.append(np.sqrt(np.mean(diff**2, axis=1)))
        mae_by_time.append(np.mean(np.abs(diff), axis=1))
        times_used.append(time_points[t_idx])

    times_used = np.array(times_used)
    rmse_q = np.array([np.quantile(r, qs) for r in rmse_by_time]).T
    mae_q = np.array([np.quantile(m, qs) for m in mae_by_time]).T

    fig, ax = plt.subplots(figsize=(10, 6))
    for q, color, label_prefix, ls in [
        (rmse_q, COLORS['prediction'], 'RMSE', '-'),
        (mae_q, '#3498db', 'MAE', '--'),
    ]:
        alpha_fill = 0.25 if ls == '-' else 0.15
        ax.semilogy(times_used, q[1] + eps, linewidth=2.5 if ls == '-' else 2.0,
                    linestyle=ls, color=color, label=f'{label_prefix} median')
        ax.fill_between(times_used, q[0] + eps, q[2] + eps,
                       alpha=alpha_fill, color=color, label=f'{label_prefix} 10–90%')

    ax.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Error Distribution Over Time', fontsize=13, fontweight='bold')
    ax.set_xscale('symlog', linthresh=1)
    ax.legend(fontsize=10)
    _style_ax(ax)

    plt.tight_layout()
    _save_fig(fig, output_dir, 'error_quantiles_time')


def plot_error_heatmap_publication(teacher_pred: np.ndarray, student_pred: np.ndarray,
                                   metadata: dict, model_name: str, output_dir: str):
    """Publication-quality error heatmap by condition and time."""
    condition_indices = metadata['condition_indices']
    time_indices = metadata['time_indices']
    n_conditions = metadata['n_conditions']
    n_time = metadata['n_time_points']

    error_matrix = np.full((n_conditions, n_time), np.nan)
    r2_matrix = np.full((n_conditions, n_time), np.nan)

    for c_idx in range(n_conditions):
        for t_idx in range(n_time):
            mask = (condition_indices == c_idx) & (time_indices == t_idx)
            if mask.sum() > 0:
                gt, pred = teacher_pred[mask].flatten(), student_pred[mask].flatten()
                error_matrix[c_idx, t_idx] = np.sqrt(mean_squared_error(gt, pred))
                if np.var(gt) > 1e-10:
                    r2_matrix[c_idx, t_idx] = r2_score(gt, pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    heatmaps = [
        (error_matrix, 'YlOrRd', 'RMSE by Condition × Time', 'RMSE', {}),
        (r2_matrix, 'RdYlGn', '$R^2$ by Condition × Time', '$R^2$', {'vmin': 0, 'vmax': 1}),
    ]
    for ax, (matrix, cmap, title, clabel, kwargs) in zip(axes, heatmaps):
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, **kwargs)
        ax.set_xlabel('Time Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Condition Index', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        _style_ax(ax)
        plt.colorbar(im, ax=ax).set_label(clabel, fontsize=11)

    plt.suptitle(f'{model_name}: Performance Heatmaps', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'error_heatmap_publication')


def plot_condition_r2_distribution(results: dict, metadata: dict,
                                    model_name: str, output_dir: str):
    """Plot distribution of trajectory R² across conditions."""
    cond_r2 = [results['by_condition'][c]['R2'] for c in sorted(results['by_condition'].keys())]
    mean_r2 = np.mean(cond_r2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.hist(cond_r2, bins=20, color=COLORS['prediction'], edgecolor='black',
             linewidth=0.5, alpha=0.7)
    ax1.axvline(x=mean_r2, color='black', linestyle='-', linewidth=2,
                label=f'Mean: {mean_r2:.3f}')
    ax1.axvline(x=R2_EXCELLENT, color='green', linestyle='--', alpha=0.7, label=f'Excellent ({R2_EXCELLENT})')
    ax1.axvline(x=R2_GOOD, color='orange', linestyle='--', alpha=0.7, label=f'Good ({R2_GOOD})')
    ax1.set_xlabel('Trajectory R²', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Trajectory R²', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    _style_ax(ax1)

    ax2 = axes[1]
    sorted_r2 = np.sort(cond_r2)[::-1]
    colors = [r2_quality_color(r2) for r2 in sorted_r2]
    ax2.bar(range(len(sorted_r2)), sorted_r2, color=colors, edgecolor='black', linewidth=0.3)
    ax2.axhline(y=R2_EXCELLENT, color='green', linestyle='--', alpha=0.7, label='Excellent')
    ax2.axhline(y=R2_GOOD, color='orange', linestyle='--', alpha=0.7, label='Good')
    ax2.set_xlabel('Condition (sorted by R²)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Trajectory R²', fontsize=12, fontweight='bold')
    ax2.set_title('Trajectory R² by Condition', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.25, axis='y')
    _style_ax(ax2)

    plt.suptitle(f'{model_name}: Condition-wise Performance', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'condition_r2_distribution')


def plot_time_analysis_publication(results: dict, metadata: dict,
                                   model_name: str, output_dir: str):
    """Plot performance metrics over time (POLLU-style)."""
    time_points = metadata['time_points']
    time_results = results['by_time']
    t_indices = sorted(time_results.keys())

    times = [time_points[t] for t in t_indices]
    r2_vals = [time_results[t]['R2'] for t in t_indices]
    rmse_vals = [time_results[t]['RMSE'] for t in t_indices]
    n_samples = [time_results[t]['n_samples'] for t in t_indices]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    line_configs = [
        (axes[0], r2_vals, 'o-', COLORS['good'], '$R^2$', '$R^2$ by Time Point', True),
        (axes[1], rmse_vals, 's-', COLORS['poor'], 'RMSE', 'RMSE by Time Point', False),
    ]
    for ax, vals, marker, color, ylabel, title, show_thresholds in line_configs:
        ax.plot(times, vals, marker, linewidth=2.5, markersize=8,
                color=color, markeredgecolor='black', markeredgewidth=1)
        ax.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xscale('symlog', linthresh=1)
        _style_ax(ax)
        if show_thresholds:
            ax.axhline(y=R2_EXCELLENT, color='green', linestyle='--', alpha=0.7, label=f'Excellent ({R2_EXCELLENT})')
            ax.axhline(y=R2_GOOD, color='orange', linestyle='--', alpha=0.7, label=f'Good ({R2_GOOD})')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)

    ax3 = axes[2]
    ax3.bar(range(len(times)), n_samples, color='#3498db', edgecolor='black', linewidth=0.5)
    tick_step = max(1, len(times) // 10)
    ax3.set_xticks(range(0, len(times), tick_step))
    ax3.set_xticklabels([f'{times[i]:.0f}' if times[i] >= 1 else f'{times[i]:.2f}'
                         for i in range(0, len(times), tick_step)], rotation=45)
    ax3.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax3.set_title('Samples per Time Point', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.25, axis='y')
    _style_ax(ax3)

    plt.suptitle(f'{model_name}: Performance by Time', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'time_analysis_publication')


# ============================================================================
# Three-Way Comparison Plot (uses test_teacher.py evaluation results)
# ============================================================================
def plot_three_way_comparison(teacher_raw_eval: dict, student_raw_eval: dict,
                               model_name: str, output_dir: str):
    """
    Three-way comparison: Teacher vs Exp, Student vs Exp, Student vs Teacher.
    
    Uses evaluation results from test_teacher.py's evaluate_model_with_replicates
    for both teacher and student against raw experimental data.
    """
    teacher_results = teacher_raw_eval['results']
    student_results = student_raw_eval['results']
    column_info = student_raw_eval['column_info']
    protein_names = column_info['protein_names']
    
    teacher_pred = teacher_results['predictions']
    student_pred = student_results['predictions']
    gt_means = student_results['gt_means']  # Same raw data for both
    
    teacher_metrics = teacher_results['protein_metrics']
    student_metrics = student_results['protein_metrics']
    
    # Select top 6 proteins by teacher R²
    sorted_proteins = sorted(protein_names, 
                            key=lambda p: teacher_metrics.get(p, {}).get('R2', 0),
                            reverse=True)
    selected = sorted_proteins[:min(6, len(protein_names))]
    n_sel = len(selected)
    
    fig, axes = plt.subplots(3, n_sel, figsize=(3.5 * n_sel, 9))
    if n_sel == 1:
        axes = axes.reshape(3, 1)
    
    comparisons = [
        ('Teacher vs Exp.', teacher_pred, gt_means, COLORS['teacher'], teacher_metrics),
        ('Student vs Exp.', student_pred, gt_means, '#2980b9', student_metrics),
        ('Student vs Teacher', student_pred, teacher_pred, COLORS['prediction'], None),
    ]
    
    for row, (comp_name, pred, gt, color, metrics) in enumerate(comparisons):
        for col, protein in enumerate(selected):
            ax = axes[row, col]
            p_idx = protein_names.index(protein)
            
            g = gt[:, p_idx]
            p = pred[:, p_idx]
            
            if len(g) > 2000:
                idx = np.random.choice(len(g), 2000, replace=False)
                g_plot, p_plot = g[idx], p[idx]
            else:
                g_plot, p_plot = g, p
            
            ax.scatter(g_plot, p_plot, alpha=0.4, s=15, edgecolors='none', c=color)
            
            min_val = min(g.min(), p.min())
            max_val = max(g.max(), p.max())
            margin = (max_val - min_val) * 0.05
            ax.plot([min_val - margin, max_val + margin],
                    [min_val - margin, max_val + margin],
                    '--', color='black', linewidth=1.2, alpha=0.7, zorder=0)
            
            r2 = _safe_r2(g, p)
            
            if row == 0:
                ax.set_title(display_name(protein), fontsize=10, fontweight='bold')
            
            ax.text(0.05, 0.95, f'$R^2$={r2:.3f}', transform=ax.transAxes,
                   fontsize=9, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            if col == 0:
                ax.set_ylabel(comp_name, fontsize=10, fontweight='bold')
            
            ax.tick_params(labelsize=7)
            ax.set_aspect('equal', adjustable='box')
            _style_ax(ax)
    
    plt.suptitle(f'{model_name}: Three-Way Comparison\n(Teacher, Student, Experimental)',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'three_way_comparison')


# ============================================================================
# Model Compression Statistics
# ============================================================================
def compute_model_compression_stats(student_model: nn.Module, teacher_model: nn.Module) -> dict:
    """Compute model compression statistics."""
    student_params = sum(p.numel() for p in student_model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    compression_ratio = teacher_params / student_params

    print(f"\n{'='*70}")
    print("MODEL COMPRESSION STATISTICS")
    print(f"{'='*70}")
    print(f"  Teacher parameters: {teacher_params:,}")
    print(f"  Student parameters: {student_params:,}")
    print(f"  Compression ratio:  {compression_ratio:.2f}x")
    print(f"  Parameter reduction: {(1 - 1/compression_ratio)*100:.1f}%")
    print(f"{'='*70}")

    return {
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': compression_ratio,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Test student model on high-resolution teacher-generated data (raw mode)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--student_model', type=str, required=True,
                       help='Path to student model checkpoint')
    parser.add_argument('--teacher_model', type=str, default=None,
                       help='Path to teacher model (for compression stats & comparison)')
    parser.add_argument('--high_res_data', type=str, required=True,
                       help='Path to high-resolution teacher predictions (.npz)')
    parser.add_argument('--raw_data', type=str, default='experimental/MIDAS/MD_MCF7_main.csv',
                       help='Path to raw MIDAS experimental data (.csv) for overlay')
    parser.add_argument('--output_dir', type=str, default='results/student_evaluation',
                       help='Output directory for plots')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    args = parser.parse_args()

    device = get_device(args.device)

    model_name = os.path.splitext(os.path.basename(args.student_model))[0]
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # Load Models
    # ========================================================================
    print("\n" + "=" * 60 + "\nLoading Models\n" + "=" * 60)
    student_model, student_config = load_student_model(args.student_model, device)

    teacher_model = None
    if args.teacher_model is None:
        teacher_path = student_config.get('teacher_model_path')
        if teacher_path and os.path.exists(teacher_path):
            args.teacher_model = teacher_path
            print(f"Using teacher model from student checkpoint: {teacher_path}")

    if args.teacher_model and os.path.exists(args.teacher_model):
        teacher_model = load_teacher_model(args.teacher_model, device)

    # ========================================================================
    # Load High-Res Data & Generate Student Predictions
    # ========================================================================
    X, teacher_pred, metadata = load_high_res_data(args.high_res_data)

    print("\n" + "=" * 60 + "\nGenerating Student Predictions (raw mode)\n" + "=" * 60)
    student_pred = predict_with_model(student_model, X, device)
    print(f"✓ Generated {len(student_pred):,} predictions")
    print(f"  Student prediction range: [{student_pred.min():.4f}, {student_pred.max():.4f}]")
    print(f"  Teacher prediction range: [{teacher_pred.min():.4f}, {teacher_pred.max():.4f}]")

    # ========================================================================
    # Evaluate Against Teacher (High-Res Dynamics)
    # ========================================================================
    print("\n" + "=" * 60 + "\nEvaluating Dynamics (Student vs Teacher)\n" + "=" * 60)
    dynamics_results = evaluate_dynamics(teacher_pred, student_pred, metadata)
    print_dynamics_evaluation(dynamics_results, metadata)

    # ========================================================================
    # Evaluate Against Raw Experimental Data (using test_teacher.py infra)
    # ========================================================================
    student_raw_eval = None
    teacher_raw_eval = None
    
    if args.raw_data and os.path.exists(args.raw_data):
        # Student vs raw experimental data
        student_raw_eval = evaluate_student_vs_raw_data(
            args.student_model, args.raw_data, device
        )
        
        # Teacher vs raw experimental data (for comparison)
        if args.teacher_model and os.path.exists(args.teacher_model):
            teacher_raw_eval = evaluate_teacher_vs_raw_data(
                args.teacher_model, args.raw_data, device
            )
    elif args.raw_data:
        print(f"  ⚠ Raw data file not found: {args.raw_data}")

    # ========================================================================
    # Generate Publication-Quality Plots
    # ========================================================================
    print("\n" + "=" * 60 + "\nGenerating Publication-Quality Plots\n" + "=" * 60)
    
    # High-res dynamics plots (student vs teacher)
    plot_trajectory_comparison_publication(
        teacher_pred, student_pred, metadata, model_name, output_dir,
        raw_eval=student_raw_eval, results=dynamics_results
    )
    plot_predictions_vs_teacher_publication(
        teacher_pred, student_pred, metadata, dynamics_results, model_name, output_dir
    )
    plot_protein_performance_publication(
        dynamics_results, metadata, model_name, output_dir
    )
    plot_error_quantiles_over_time(
        teacher_pred, student_pred, metadata, model_name, output_dir
    )
    plot_error_heatmap_publication(
        teacher_pred, student_pred, metadata, model_name, output_dir
    )
    plot_condition_r2_distribution(
        dynamics_results, metadata, model_name, output_dir
    )
    plot_time_analysis_publication(
        dynamics_results, metadata, model_name, output_dir
    )

    # Raw data plots — reuse test_teacher.py's plotting functions
    if student_raw_eval is not None:
        student_raw_results = student_raw_eval['results']
        student_raw_aggregated = student_raw_eval['aggregated_data']
        student_raw_column_info = student_raw_eval['column_info']
        
        # Create output directory BEFORE saving any plots
        os.makedirs(os.path.join(output_dir, 'raw_data'), exist_ok=True)
        
        # Student predictions vs raw experimental data (scatter)
        teacher_plot_pred_vs_gt(
            student_raw_results, f"{model_name} (Student)",
            os.path.join(output_dir, 'raw_data')
        )
        
        # Student protein performance vs raw data
        teacher_plot_protein_perf(
            student_raw_results, f"{model_name} (Student vs Exp.)",
            os.path.join(output_dir, 'raw_data')
        )
        
        # Student time course vs raw data
        teacher_plot_timecourse(
            student_raw_results, student_raw_aggregated,
            f"{model_name} (Student vs Exp.)",
            os.path.join(output_dir, 'raw_data')
        )
        
        # Student error heatmap vs raw data
        teacher_plot_error_heatmap(
            student_raw_results, student_raw_aggregated,
            f"{model_name} (Student vs Exp.)",
            os.path.join(output_dir, 'raw_data')
        )
        
        # Student replicate analysis
        teacher_plot_replicate(
            student_raw_results, student_raw_aggregated,
            f"{model_name} (Student vs Exp.)",
            os.path.join(output_dir, 'raw_data')
        )
        
        # Student time analysis vs raw
        student_raw_time_results = student_raw_eval['time_results']
        teacher_plot_time_analysis(
            student_raw_time_results, f"{model_name} (Student vs Exp.)",
            os.path.join(output_dir, 'raw_data')
        )
        
        # Three-way comparison (Teacher vs Exp, Student vs Exp, Student vs Teacher)
        if teacher_raw_eval is not None:
            plot_three_way_comparison(
                teacher_raw_eval, student_raw_eval,
                model_name, output_dir
            )

    # ========================================================================
    # Compression Statistics
    # ========================================================================
    compression_stats = None
    if teacher_model is not None:
        compression_stats = compute_model_compression_stats(student_model, teacher_model)

    # ========================================================================
    # Summary
    # ========================================================================
    protein_names = metadata['protein_names']
    r2_values = [dynamics_results['by_protein'][p]['R2'] for p in protein_names]
    cond_r2 = [dynamics_results['by_condition'][c]['R2'] for c in dynamics_results['by_condition']]
    n_exc, n_good, n_poor = r2_quality_counts(r2_values)
    overall = dynamics_results['overall']

    print(f"\n{'='*60}\nEVALUATION SUMMARY (RAW MODE)\n{'='*60}")
    print(f"\nStudent Model: {args.student_model}")
    if args.teacher_model:
        print(f"Teacher Model: {args.teacher_model}")
    print(f"High-Res Data: {args.high_res_data}")
    if args.raw_data:
        print(f"Raw Exp. Data: {args.raw_data}")
    print(f"Mode: RAW (no preprocessing, no scaling)")
    
    print(f"\nDATA SUMMARY:")
    print(f"  Total high-res samples: {len(X):,}")
    print(f"  Conditions: {metadata['n_conditions']}")
    print(f"  Time points: {metadata['n_time_points']}")
    print(f"  Proteins: {metadata['n_proteins']}")
    if student_raw_eval is not None:
        print(f"  Raw experimental samples: {len(student_raw_eval['X']):,}")
    
    print(f"\nPERFORMANCE vs TEACHER (High-Res):")
    for key in ('MAE', 'RMSE', 'R2', 'Relative_Error'):
        fmt = '.6f' if key != 'Relative_Error' else '.4f'
        print(f"  {key}: {overall[key]:{fmt}}")
    
    if student_raw_eval is not None:
        student_raw_overall = student_raw_eval['results']['mae_overall']
        print(f"\nPERFORMANCE vs RAW EXPERIMENTAL DATA (Student):")
        print(f"  MAE:  {student_raw_eval['results']['mae_overall']:.6f}")
        print(f"  RMSE: {student_raw_eval['results']['rmse_overall']:.6f}")
        print(f"  R²:   {student_raw_eval['results']['r2_overall']:.6f}")
        
        if teacher_raw_eval is not None:
            print(f"\nPERFORMANCE vs RAW EXPERIMENTAL DATA (Teacher — reference):")
            print(f"  MAE:  {teacher_raw_eval['results']['mae_overall']:.6f}")
            print(f"  RMSE: {teacher_raw_eval['results']['rmse_overall']:.6f}")
            print(f"  R²:   {teacher_raw_eval['results']['r2_overall']:.6f}")
    
    print(f"\nPROTEIN PREDICTION QUALITY (vs Teacher):")
    print(f"  Excellent (R² > {R2_EXCELLENT}):  {n_exc}/{len(r2_values)} proteins")
    print(f"  Good ({R2_GOOD} < R² ≤ {R2_EXCELLENT}): {n_good}/{len(r2_values)} proteins")
    print(f"  Poor (R² ≤ {R2_GOOD}):       {n_poor}/{len(r2_values)} proteins")
    
    print(f"\nCONDITION-WISE TRAJECTORY QUALITY:")
    print(f"  Mean R²: {np.mean(cond_r2):.4f}")
    print(f"  Std R²:  {np.std(cond_r2):.4f}")
    
    if compression_stats:
        print(f"\nMODEL COMPRESSION:")
        print(f"  Compression ratio: {compression_stats['compression_ratio']:.2f}x")
        print(f"  Parameter reduction: {(1 - 1/compression_stats['compression_ratio'])*100:.1f}%")
    
    print(f"\nPlots saved to: {output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()