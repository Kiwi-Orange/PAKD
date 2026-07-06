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
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/hpn_dream_matplotlib')

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.lines import Line2D
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
    set_publication_style,
    save_publication_figure,
    style_publication_axes,
    format_log_time_axis,
    MIDAS_TREATMENT_PREFIX,
    MIDAS_DATA_AVG_PREFIX,
    MIDAS_DATA_VAL_PREFIX,
    STIMULI_COLORS,
    COLORS as TEACHER_COLORS,
    PROTEIN_DISPLAY_NAMES as TEACHER_PROTEIN_DISPLAY_NAMES,
    KEY_PROTEINS as TEACHER_KEY_PROTEINS,
)

set_publication_style()

# ============================================================================
# Constants (matching test_teacher.py / teacher_generation.py)
# ============================================================================
KEY_PROTEINS = TEACHER_KEY_PROTEINS

PROTEIN_DISPLAY_NAMES = TEACHER_PROTEIN_DISPLAY_NAMES

COLORS = {
    **TEACHER_COLORS,
    'teacher': TEACHER_COLORS['ground_truth'],
    'student': TEACHER_COLORS['prediction'],
    'experiment': TEACHER_COLORS['blue'],
    'replicate': TEACHER_COLORS['gray'],
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


def compact_display_name(protein: str, max_len: int = 18) -> str:
    """Display long protein labels on two compact lines."""
    name = display_name(protein)
    if len(name) <= max_len:
        return name
    if '_' in name:
        parts = name.split('_')
        mid = max(1, len(parts) // 2)
        return '_'.join(parts[:mid]) + '\n' + '_'.join(parts[mid:])
    if ' ' in name:
        parts = name.split()
        mid = max(1, len(parts) // 2)
        return ' '.join(parts[:mid]) + '\n' + ' '.join(parts[mid:])
    return name[:max_len - 1] + '.'


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
    """Apply the shared Nature Communications-style axis formatting."""
    style_publication_axes(ax)


def _save_fig(fig, output_dir: str, name: str):
    """Save figure as PDF, SVG, and high-resolution PNG."""
    save_publication_figure(fig, output_dir, name)
    print(f"✓ Saved {name} plot")


def _condition_parts(condition_name: str) -> tuple:
    """Return normalized stimulus and inhibitor labels from a condition name."""
    parts = str(condition_name).split('|')
    stim = parts[0].strip() if len(parts) > 0 else ''
    inhib = parts[1].strip() if len(parts) > 1 else ''
    return stim, inhib


def _is_none_label(value: str, kind: str) -> bool:
    value = str(value).strip().lower()
    none_labels = {'', 'none'}
    none_labels.add('no_stimuli' if kind == 'stimulus' else 'no_inhibitor')
    return value in none_labels


def _condition_display(condition_name: str) -> tuple:
    """Return a compact display label and primary stimulus for a condition."""
    stim, inhib = _condition_parts(condition_name)
    if _is_none_label(inhib, 'inhibitor'):
        return stim if stim else 'No stimulus', stim
    if _is_none_label(stim, 'stimulus'):
        return f'{inhib} only', stim
    return f'{stim} + {inhib}', stim


def _find_no_inhibitor_condition(condition_names: list, stimulus: str):
    """Find a condition index for stimulus with no inhibitor."""
    stimulus_norm = stimulus.strip().lower()
    for idx, cname in enumerate(condition_names):
        stim, inhib = _condition_parts(cname)
        if stim.strip().lower() == stimulus_norm and _is_none_label(inhib, 'inhibitor'):
            return idx
    return None


def _select_key_proteins_by_r2(protein_names: list, results: dict = None, n: int = 4) -> list:
    """Select key proteins ranked by student-vs-teacher R², with a stable fallback."""
    available = [p for p in KEY_PROTEINS if p in protein_names]
    if len(available) < n:
        available.extend([p for p in protein_names if p not in available])

    if results is not None and 'by_protein' in results:
        available = sorted(
            available,
            key=lambda p: results['by_protein'].get(p, {}).get('R2', -np.inf),
            reverse=True,
        )
    return available[:min(n, len(available))]


def _set_panel_ylim(ax, *arrays, pad_frac: float = 0.12, min_pad: float = 0.025):
    """Set a compact y-limit from finite plotted values."""
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.size:
            vals.extend(arr[np.isfinite(arr)].tolist())
    if not vals:
        return
    ymin, ymax = min(vals), max(vals)
    span = ymax - ymin
    pad = max(span * pad_frac, min_pad)
    ax.set_ylim(ymin - pad, ymax + pad)


def _set_model_centered_trajectory_ylim(ax, teacher_values: np.ndarray,
                                        student_values: np.ndarray,
                                        experimental_means: np.ndarray = None,
                                        pad_frac: float = 0.10):
    """
    Set trajectory y-limits from teacher/student dynamics, with light raw-data influence.

    The main panel should show temporal structure without visually exaggerating small
    teacher/student gaps. Experimental means can expand the window when they are close
    to the model trajectories, but noisy raw points and SD bars do not set the scale.
    """
    teacher_values = np.asarray(teacher_values, dtype=float)
    student_values = np.asarray(student_values, dtype=float)
    model_vals = np.concatenate([
        teacher_values[np.isfinite(teacher_values)],
        student_values[np.isfinite(student_values)],
    ])
    if model_vals.size == 0:
        return

    model_min, model_max = float(np.min(model_vals)), float(np.max(model_vals))
    center = 0.5 * (model_min + model_max)
    model_span = model_max - model_min

    teacher_dynamic = np.ptp(teacher_values[np.isfinite(teacher_values)]) if np.isfinite(teacher_values).any() else 0.0
    student_dynamic = np.ptp(student_values[np.isfinite(student_values)]) if np.isfinite(student_values).any() else 0.0
    model_dynamic = max(float(teacher_dynamic), float(student_dynamic))

    paired = np.isfinite(teacher_values) & np.isfinite(student_values)
    max_gap = float(np.max(np.abs(teacher_values[paired] - student_values[paired]))) if paired.any() else 0.0

    min_span = max(
        1.35 * model_dynamic,
        2.0 * max_gap,
        0.16 * max(abs(center), 1.0),
        0.08,
    )
    base_span = max(model_span, min_span)
    low = center - 0.5 * base_span
    high = center + 0.5 * base_span

    if experimental_means is not None:
        exp_vals = np.asarray(experimental_means, dtype=float)
        exp_vals = exp_vals[np.isfinite(exp_vals)]
        if exp_vals.size:
            near_margin = 0.35 * base_span
            near = exp_vals[(exp_vals >= low - near_margin) & (exp_vals <= high + near_margin)]
            if near.size:
                low = min(low, float(np.min(near)))
                high = max(high, float(np.max(near)))
                max_span = 1.35 * base_span
                if high - low > max_span:
                    low = max(low, center - 0.5 * max_span)
                    high = min(high, center + 0.5 * max_span)

    final_span = max(high - low, base_span)
    if final_span > high - low:
        center = 0.5 * (low + high)
        low = center - 0.5 * final_span
        high = center + 0.5 * final_span

    pad = max(final_span * pad_frac, 0.015)
    lower, upper = low - pad, high + pad
    if np.min(model_vals) >= 0 and lower < 0:
        lower = 0
    ax.set_ylim(lower, upper)


def _sample_points(x: np.ndarray, y: np.ndarray, max_points: int = 2500):
    """Deterministically downsample dense scatter plots for compact vector output."""
    if len(x) <= max_points:
        return x, y
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), max_points, replace=False)
    return x[idx], y[idx]


def _plot_experiment_summary(ax, raw_cond_protein: dict, show_replicates: bool = False):
    """Overlay experimental replicate means with SD error bars."""
    if raw_cond_protein is None:
        return None

    raw_times = np.asarray(raw_cond_protein['times'], dtype=float)
    raw_means = np.asarray(raw_cond_protein['means'], dtype=float)
    raw_stds = np.asarray(raw_cond_protein['stds'], dtype=float)
    keep = raw_times >= 1
    raw_times, raw_means, raw_stds = raw_times[keep], raw_means[keep], raw_stds[keep]

    if show_replicates and raw_cond_protein.get('all_points') is not None:
        for rt, rv_list in raw_cond_protein['all_points']:
            if rt < 1:
                continue
            ax.scatter(
                np.repeat(rt, len(rv_list)), rv_list,
                marker='.', s=5, color=COLORS['replicate'], alpha=0.22,
                linewidths=0, zorder=2,
            )

    handle = None
    if raw_times.size:
        handle = ax.errorbar(
            raw_times, raw_means, yerr=raw_stds,
            fmt='o', markersize=2.7, markerfacecolor='white',
            markeredgecolor=COLORS['experiment'], markeredgewidth=0.8,
            ecolor=COLORS['experiment'], elinewidth=0.7, capsize=1.8,
            linestyle='none', alpha=0.95, zorder=5,
        )
    return handle


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
        rmse_p = np.sqrt(mean_squared_error(gt_p, pred_p))
        range_p = np.ptp(gt_p)
        results['by_protein'][protein] = {
            'MAE': mean_absolute_error(gt_p, pred_p),
            'RMSE': rmse_p,
            'nRMSE': rmse_p / (range_p + eps),
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


def plot_trajectory_comparison_main(teacher_pred: np.ndarray, student_pred: np.ndarray,
                                    metadata: dict, model_name: str, output_dir: str,
                                    raw_eval: dict = None, results: dict = None):
    """Main-text 4x4 trajectory panel for no-inhibitor stimulus dynamics."""
    time_points = metadata['time_points']
    condition_indices = metadata['condition_indices']
    time_indices = metadata['time_indices']
    protein_names = metadata['protein_names']
    condition_names = metadata.get('condition_names', [])

    selected_stimuli = ['EGF', 'Insulin', 'FGF1', 'HGF']
    conditions_to_plot = [
        _find_no_inhibitor_condition(condition_names, stim)
        for stim in selected_stimuli
    ]
    plotted = [(stim, idx) for stim, idx in zip(selected_stimuli, conditions_to_plot) if idx is not None]
    if not plotted:
        print("  ⚠ No no-inhibitor EGF/Insulin/FGF1/HGF conditions found; skipping main trajectory plot")
        return

    selected_proteins = _select_key_proteins_by_r2(protein_names, results, n=4)
    raw_by_condition = (
        _organize_raw_data_by_condition(raw_eval, protein_names, condition_names)
        if raw_eval is not None else None
    )

    n_prot, n_cond = len(selected_proteins), len(plotted)
    fig, axes = plt.subplots(n_prot, n_cond, figsize=(7.2, 5.2), sharex=True, squeeze=False)

    for i, protein in enumerate(selected_proteins):
        p_idx = protein_names.index(protein)
        r2 = results['by_protein'].get(protein, {}).get('R2', np.nan) if results else np.nan
        for j, (stimulus, cond_idx) in enumerate(plotted):
            ax = axes[i, j]
            mask = condition_indices == cond_idx
            times = time_points[time_indices[mask]]
            order = np.argsort(times)
            times_s = np.asarray(times[order], dtype=float)
            teacher_s = teacher_pred[mask, p_idx][order]
            student_s = student_pred[mask, p_idx][order]
            keep = times_s >= 1
            times_s, teacher_s, student_s = times_s[keep], teacher_s[keep], student_s[keep]

            ax.plot(times_s, teacher_s, color=COLORS['teacher'], linewidth=1.0, zorder=3)
            ax.plot(times_s, student_s, color=COLORS['student'], linewidth=1.05,
                    linestyle='--', zorder=4)

            raw_cond_protein = None
            cname = condition_names[cond_idx] if cond_idx < len(condition_names) else None
            if raw_by_condition is not None and cname in raw_by_condition:
                raw_cond_protein = raw_by_condition[cname].get(protein)
                _plot_experiment_summary(ax, raw_cond_protein, show_replicates=False)

            raw_means_for_ylim = None
            if raw_cond_protein is not None:
                raw_times = np.asarray(raw_cond_protein['times'], dtype=float)
                raw_keep = raw_times >= 1
                raw_means_for_ylim = np.asarray(raw_cond_protein['means'], dtype=float)[raw_keep]
            _set_model_centered_trajectory_ylim(
                ax, teacher_s, student_s, experimental_means=raw_means_for_ylim
            )

            if i == 0:
                ax.set_title(stimulus, color=STIMULI_COLORS.get(stimulus, COLORS['ground_truth']),
                             pad=2.5)
            if j == 0:
                ax.set_ylabel(display_name(protein))
            if j == n_cond - 1 and np.isfinite(r2):
                ax.text(0.98, 0.92, f'$R^2$={r2:.2f}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=5.2)
            if i == n_prot - 1:
                ax.set_xlabel('Time (min)')
            format_log_time_axis(ax, ticks=[1, 5, 15, 30, 60, 120, 240])
            _style_ax(ax)

    handles = [
        Line2D([0], [0], color=COLORS['teacher'], lw=1.0, label='Teacher'),
        Line2D([0], [0], color=COLORS['student'], lw=1.0, linestyle='--', label='Student'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor=COLORS['experiment'], markersize=3.0,
               label='Experiment mean ± SD'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, handlelength=1.8, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.35, w_pad=0.35)
    _save_fig(fig, output_dir, 'trajectory_comparison_main')


def plot_trajectory_comparison_publication(teacher_pred: np.ndarray, student_pred: np.ndarray,
                                           metadata: dict, model_name: str, output_dir: str,
                                           n_conditions_to_plot: int = 6,
                                           raw_eval: dict = None,
                                           results: dict = None):
    """Supplement-style trajectory comparison across selected non-trivial conditions."""
    time_points = metadata['time_points']
    condition_indices = metadata['condition_indices']
    time_indices = metadata['time_indices']
    protein_names = metadata['protein_names']
    condition_names = metadata.get('condition_names', [])

    selected_proteins = _select_key_proteins_by_r2(protein_names, results, n=4)
    unique_conditions = np.unique(condition_indices)
    valid_conditions = []
    for cond_idx in unique_conditions:
        if cond_idx < len(condition_names):
            stim, inhib = _condition_parts(condition_names[cond_idx])
            if _is_none_label(stim, 'stimulus') and _is_none_label(inhib, 'inhibitor'):
                continue
            if _is_none_label(stim, 'stimulus'):
                continue
        valid_conditions.append(cond_idx)
    if not valid_conditions:
        valid_conditions = list(unique_conditions)

    n_plot = min(n_conditions_to_plot, len(valid_conditions))
    step = max(1, len(valid_conditions) // n_plot)
    conditions_to_plot = valid_conditions[::step][:n_plot]

    raw_by_condition = (
        _organize_raw_data_by_condition(raw_eval, protein_names, condition_names)
        if raw_eval is not None else None
    )

    n_prot, n_cond = len(selected_proteins), len(conditions_to_plot)
    fig, axes = plt.subplots(n_prot, n_cond, figsize=(7.2, 5.4), sharex=True, squeeze=False)

    for i, protein in enumerate(selected_proteins):
        p_idx = protein_names.index(protein)
        for j, cond_idx in enumerate(conditions_to_plot):
            ax = axes[i, j]
            mask = condition_indices == cond_idx
            times = time_points[time_indices[mask]]
            order = np.argsort(times)
            times_s = np.asarray(times[order], dtype=float)
            teacher_s = teacher_pred[mask, p_idx][order]
            student_s = student_pred[mask, p_idx][order]
            keep = times_s >= 1
            times_s, teacher_s, student_s = times_s[keep], teacher_s[keep], student_s[keep]

            ax.plot(times_s, teacher_s, color=COLORS['teacher'], linewidth=0.9, zorder=3)
            ax.plot(times_s, student_s, color=COLORS['student'], linewidth=0.95,
                    linestyle='--', zorder=4)

            raw_cond_protein = None
            cname = condition_names[cond_idx] if cond_idx < len(condition_names) else None
            if raw_by_condition is not None and cname in raw_by_condition:
                raw_cond_protein = raw_by_condition[cname].get(protein)
                _plot_experiment_summary(ax, raw_cond_protein, show_replicates=True)

            if i == 0:
                cname_full = condition_names[cond_idx] if cond_idx < len(condition_names) else f'Cond. {cond_idx}'
                cname_display, stim = _condition_display(cname_full)
                ax.set_title(cname_display, color=STIMULI_COLORS.get(stim, COLORS['ground_truth']),
                             pad=2.5)
            if j == 0:
                ax.set_ylabel(display_name(protein))
            if i == n_prot - 1:
                ax.set_xlabel('Time (min)')
            format_log_time_axis(ax, ticks=[1, 5, 15, 30, 60, 120, 240])
            _style_ax(ax)

    handles = [
        Line2D([0], [0], color=COLORS['teacher'], lw=1.0, label='Teacher'),
        Line2D([0], [0], color=COLORS['student'], lw=1.0, linestyle='--', label='Student'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
               markeredgecolor=COLORS['experiment'], markersize=3.0,
               label='Experiment mean ± SD'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, handlelength=1.8, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.35, w_pad=0.3)
    _save_fig(fig, output_dir, 'trajectory_comparison')


def plot_predictions_vs_teacher_publication(teacher_pred: np.ndarray, student_pred: np.ndarray,
                                            metadata: dict, results: dict,
                                            model_name: str, output_dir: str):
    """Publication-quality scatter plots: student vs teacher predictions."""
    protein_names = metadata['protein_names']
    n_proteins = len(protein_names)
    n_cols = 6
    n_rows = (n_proteins + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 1.15 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, protein in enumerate(protein_names):
        ax = axes_flat[i]
        gt, pred = teacher_pred[:, i], student_pred[:, i]

        gt_plot, pred_plot = _sample_points(gt, pred, max_points=2500)

        ax.scatter(gt_plot, pred_plot, alpha=0.35, s=3.5, edgecolors='none',
                   c=COLORS['student'])

        min_val, max_val = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        margin = max((max_val - min_val) * 0.05, 1e-3)
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                '--', color=COLORS['gray'], linewidth=0.7, alpha=0.75, zorder=0)
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)

        r2 = results['by_protein'][protein]['R2']
        ax.set_title(f'{compact_display_name(protein)}\n$R^2$={r2:.2f}', pad=2)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel('Teacher')
        if i % n_cols == 0:
            ax.set_ylabel('Student')
        ax.set_aspect('equal', adjustable='box')
        _style_ax(ax)

    for i in range(n_proteins, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout(h_pad=0.25, w_pad=0.25)
    _save_fig(fig, output_dir, 'pred_vs_teacher_publication')


def plot_protein_performance_publication(results: dict, metadata: dict,
                                         model_name: str, output_dir: str):
    """Publication-quality per-protein performance bar charts."""
    protein_names = metadata['protein_names']
    metrics = results['by_protein']

    r2_vals = [metrics[p]['R2'] for p in protein_names]
    rmse_vals = [metrics[p]['RMSE'] for p in protein_names]
    nrmse_vals = [metrics[p].get('nRMSE', metrics[p]['Relative_Error']) for p in protein_names]
    sorted_idx = np.argsort(r2_vals)[::-1]
    names = [display_name(protein_names[i]) for i in sorted_idx]

    panels = [
        ('$R^2$', [r2_vals[i] for i in sorted_idx],
         [r2_quality_color(r2_vals[i]) for i in sorted_idx], '$R^2$', (0, 1.0)),
        ('RMSE', [rmse_vals[i] for i in sorted_idx], COLORS['blue'], 'RMSE', None),
        ('nRMSE', [nrmse_vals[i] for i in sorted_idx], COLORS['moderate'], 'nRMSE', None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.4), sharey=True)

    for ax, (xlabel, values, colors, title, xlim) in zip(axes, panels):
        ax.barh(range(len(protein_names)), values, color=colors, edgecolor='none')
        ax.set_yticks(range(len(protein_names)))
        ax.set_yticklabels(names)
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=2.5)
        ax.invert_yaxis()
        ax.grid(axis='x', color=COLORS['light_gray'], linewidth=0.35)
        _style_ax(ax)
        if xlim:
            ax.set_xlim(*xlim)
            ax.axvline(x=R2_EXCELLENT, color=COLORS['good'], linestyle='--',
                       linewidth=0.7, alpha=0.9)
            ax.axvline(x=R2_GOOD, color=COLORS['moderate'], linestyle='--',
                       linewidth=0.7, alpha=0.9)

    fig.tight_layout(w_pad=0.7)
    _save_fig(fig, output_dir, 'protein_performance_publication')


def plot_error_quantiles_over_time(teacher_pred: np.ndarray, student_pred: np.ndarray,
                                   metadata: dict, model_name: str, output_dir: str):
    """Error quantile plot over time with 10-90% bands."""
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

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for q, color, label_prefix, ls in [
        (rmse_q, COLORS['student'], 'RMSE', '-'),
        (mae_q, COLORS['blue'], 'MAE', '--'),
    ]:
        alpha_fill = 0.18 if ls == '-' else 0.12
        ax.semilogy(times_used, q[1] + eps, linewidth=0.9,
                    linestyle=ls, color=color, label=f'{label_prefix} median')
        ax.fill_between(times_used, q[0] + eps, q[2] + eps,
                       alpha=alpha_fill, color=color, label=f'{label_prefix} 10–90%')

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Error')
    format_log_time_axis(ax, ticks=[1, 5, 15, 30, 60, 120, 240])
    ax.set_yscale('log')
    ax.legend(frameon=False, loc='best', handlelength=1.6)
    _style_ax(ax)

    fig.tight_layout()
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    heatmaps = [
        (error_matrix, 'mako_r', 'RMSE', 'RMSE', {}),
        (r2_matrix, 'viridis', '$R^2$', '$R^2$', {'vmin': 0, 'vmax': 1}),
    ]
    for ax, (matrix, cmap, title, clabel, kwargs) in zip(axes, heatmaps):
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, **kwargs)
        ax.set_xlabel('Time index')
        ax.set_ylabel('Condition index')
        ax.set_title(title, pad=2.5)
        ax.set_xticks(np.linspace(0, n_time - 1, min(5, n_time), dtype=int))
        ax.set_yticks(np.linspace(0, n_conditions - 1, min(6, n_conditions), dtype=int))
        _style_ax(ax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(clabel)
        cbar.ax.tick_params(width=0.6, length=2)

    fig.tight_layout(w_pad=0.7)
    _save_fig(fig, output_dir, 'error_heatmap_publication')


def plot_condition_r2_distribution(results: dict, metadata: dict,
                                    model_name: str, output_dir: str):
    """Plot distribution of trajectory R² across conditions."""
    cond_r2 = [results['by_condition'][c]['R2'] for c in sorted(results['by_condition'].keys())]
    mean_r2 = np.mean(cond_r2)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))

    ax1 = axes[0]
    ax1.hist(cond_r2, bins=16, color=COLORS['student'], edgecolor='white',
             linewidth=0.4, alpha=0.85)
    ax1.axvline(x=mean_r2, color=COLORS['ground_truth'], linestyle='-', linewidth=0.8,
                label=f'Mean: {mean_r2:.3f}')
    ax1.axvline(x=R2_EXCELLENT, color=COLORS['good'], linestyle='--', linewidth=0.7)
    ax1.axvline(x=R2_GOOD, color=COLORS['moderate'], linestyle='--', linewidth=0.7)
    ax1.set_xlabel('Trajectory $R^2$')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution', pad=2.5)
    ax1.legend(frameon=False, loc='upper left', handlelength=1.4)
    _style_ax(ax1)

    ax2 = axes[1]
    sorted_r2 = np.sort(cond_r2)[::-1]
    colors = [r2_quality_color(r2) for r2 in sorted_r2]
    ax2.bar(range(len(sorted_r2)), sorted_r2, color=colors, edgecolor='none')
    ax2.axhline(y=R2_EXCELLENT, color=COLORS['good'], linestyle='--', linewidth=0.7)
    ax2.axhline(y=R2_GOOD, color=COLORS['moderate'], linestyle='--', linewidth=0.7)
    ax2.set_xlabel('Condition rank')
    ax2.set_ylabel('Trajectory $R^2$')
    ax2.set_title('Sorted by condition', pad=2.5)
    ax2.grid(axis='y', color=COLORS['light_gray'], linewidth=0.35)
    _style_ax(ax2)

    fig.tight_layout(w_pad=0.75)
    _save_fig(fig, output_dir, 'condition_r2_distribution')


def plot_time_analysis_publication(results: dict, metadata: dict,
                                   model_name: str, output_dir: str):
    """Plot performance metrics over time."""
    time_points = metadata['time_points']
    time_results = results['by_time']
    t_indices = sorted(time_results.keys())

    times = [time_points[t] for t in t_indices]
    r2_vals = [time_results[t]['R2'] for t in t_indices]
    rmse_vals = [time_results[t]['RMSE'] for t in t_indices]
    n_samples = [time_results[t]['n_samples'] for t in t_indices]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))

    line_configs = [
        (axes[0], r2_vals, 'o-', COLORS['good'], '$R^2$', '$R^2$', True),
        (axes[1], rmse_vals, 's-', COLORS['poor'], 'RMSE', 'RMSE', False),
    ]
    for ax, vals, marker, color, ylabel, title, show_thresholds in line_configs:
        ax.plot(times, vals, marker, linewidth=0.9, markersize=2.4,
                color=color, markeredgecolor='white', markeredgewidth=0.35)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=2.5)
        format_log_time_axis(ax, ticks=[1, 5, 15, 30, 60, 120, 240])
        _style_ax(ax)
        if show_thresholds:
            ax.axhline(y=R2_EXCELLENT, color=COLORS['good'], linestyle='--',
                       linewidth=0.65, alpha=0.85)
            ax.axhline(y=R2_GOOD, color=COLORS['moderate'], linestyle='--',
                       linewidth=0.65, alpha=0.85)
            ax.set_ylim(0, 1.05)

    ax3 = axes[2]
    ax3.bar(range(len(times)), n_samples, color=COLORS['blue'], edgecolor='none')
    tick_step = max(1, len(times) // 10)
    ax3.set_xticks(range(0, len(times), tick_step))
    ax3.set_xticklabels([f'{times[i]:.0f}' if times[i] >= 1 else f'{times[i]:.2f}'
                         for i in range(0, len(times), tick_step)], rotation=45)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Samples')
    ax3.set_title('Samples', pad=2.5)
    ax3.grid(axis='y', color=COLORS['light_gray'], linewidth=0.35)
    _style_ax(ax3)

    fig.tight_layout(w_pad=0.65)
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
    
    fig, axes = plt.subplots(3, n_sel, figsize=(7.2, 4.45), squeeze=False)
    if n_sel == 1:
        axes = axes.reshape(3, 1)
    
    comparisons = [
        ('Teacher vs Exp.', teacher_pred, gt_means, COLORS['teacher'], teacher_metrics),
        ('Student vs Exp.', student_pred, gt_means, COLORS['experiment'], student_metrics),
        ('Student vs Teacher', student_pred, teacher_pred, COLORS['student'], None),
    ]
    
    for row, (comp_name, pred, gt, color, metrics) in enumerate(comparisons):
        for col, protein in enumerate(selected):
            ax = axes[row, col]
            p_idx = protein_names.index(protein)
            
            g = gt[:, p_idx]
            p = pred[:, p_idx]
            
            g_plot, p_plot = _sample_points(g, p, max_points=1800)
            
            ax.scatter(g_plot, p_plot, alpha=0.36, s=4, edgecolors='none', c=color)
            
            min_val = min(g.min(), p.min())
            max_val = max(g.max(), p.max())
            margin = max((max_val - min_val) * 0.05, 1e-3)
            ax.plot([min_val - margin, max_val + margin],
                    [min_val - margin, max_val + margin],
                    '--', color=COLORS['gray'], linewidth=0.65, alpha=0.75, zorder=0)
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            
            r2 = _safe_r2(g, p)
            
            if row == 0:
                ax.set_title(compact_display_name(protein), pad=2)
            
            ax.text(0.05, 0.95, f'$R^2$={r2:.3f}', transform=ax.transAxes,
                    va='top', fontsize=5.2,
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.75))
            
            if col == 0:
                ax.set_ylabel(comp_name)
            
            ax.set_aspect('equal', adjustable='box')
            _style_ax(ax)
    
    fig.tight_layout(h_pad=0.35, w_pad=0.25)
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
    plot_trajectory_comparison_main(
        teacher_pred, student_pred, metadata, model_name, output_dir,
        raw_eval=student_raw_eval, results=dynamics_results
    )
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
