"""
Generate CSV for Online Learning Regulation using a trained Student Model.

Loads a PAKD-trained student model and the high-resolution teacher-generated data,
produces student predictions, and exports a structured CSV with:
  - Treatment conditions (stimuli, inhibitors)
  - Time points
  - Teacher predictions (ground truth for online learning)
  - Student predictions (current model output)
  - Prediction errors (for regulation signals)
  - Confidence/uncertainty estimates

This CSV can be consumed by an online learning loop to identify
which conditions/proteins/time-phases need further training.

Uses raw data directly (no preprocessing, no scaling) — consistent with
train_teacher_multi.py, teacher_generation.py, HMM_clustering.py, PAKD.py,
and test_student.py.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

# ---------------------------------------------------------------------------
# Import from project modules
# ---------------------------------------------------------------------------
from models import MLP, ResidualMLP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIDAS_TREATMENT_PREFIX = "TR:"
MIDAS_DATA_VAL_PREFIX = "DV:"


# ============================================================================
# Utility Functions (consistent with test_student.py)
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


# ============================================================================
# Model building helpers (consistent with test_student.py)
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
        arch_str = f"ResidualMLP({input_size}->{hidden_dim}x{num_blocks}->{output_size})"
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
        arch_str = f"MLP({input_size}->{hidden_sizes}->{output_size})"
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
        print("  ⚠ Checkpoint does not appear to be a student model. Proceeding anyway.")

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
        'column_info': checkpoint.get('column_info', None),
        'teacher_model_path': checkpoint.get('teacher_model_path',
                                             training_args.get('teacher_model', None)),
        'final_r2': checkpoint.get('final_r2', None),
        'r2_by_protein': checkpoint.get('r2_by_protein', None),
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"✓ Student model loaded (raw mode, no scalers)")

    return model, config


# ============================================================================
# Data loading (consistent with test_student.py)
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


# ============================================================================
# Prediction (consistent with test_student.py)
# ============================================================================
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
# CSV Generation
# ============================================================================
def generate_online_learning_csv(
    teacher_pred: np.ndarray,
    student_pred: np.ndarray,
    metadata: dict,
    output_path: str,
    error_threshold: float = 0.05,
):
    """
    Build and save a CSV for online-learning regulation.

    Columns produced
    ----------------
    condition_idx, condition_name,
    time_idx, time_min,
    protein_idx, protein_name,
    teacher_pred, student_pred,
    abs_error, rel_error, squared_error,
    needs_update          (bool flag: abs_error > threshold)
    priority_score        (composite score for scheduling online updates)
    time_phase            (early / mid / late)
    """
    print(f"\nGenerating online-learning regulation CSV …")

    condition_indices = metadata['condition_indices']
    time_indices = metadata['time_indices']
    time_points = metadata['time_points']
    protein_names = metadata['protein_names']
    condition_names = metadata.get('condition_names', [])
    n_proteins = len(protein_names)

    rows = []
    n_samples = len(teacher_pred)

    # Pre-compute time-phase boundaries (thirds)
    unique_t = np.unique(time_indices)
    n_t = len(unique_t)
    t_thirds = [n_t // 3, 2 * n_t // 3]

    for s_idx in range(n_samples):
        c_idx = int(condition_indices[s_idx])
        t_idx = int(time_indices[s_idx])
        t_min = float(time_points[s_idx]) if s_idx < len(time_points) else float(t_idx)
        c_name = condition_names[c_idx] if c_idx < len(condition_names) else f"Cond_{c_idx}"

        # Time phase
        t_rank = int(np.searchsorted(unique_t, t_idx))
        if t_rank < t_thirds[0]:
            phase = 'early'
        elif t_rank < t_thirds[1]:
            phase = 'mid'
        else:
            phase = 'late'

        for p_idx in range(n_proteins):
            t_val = float(teacher_pred[s_idx, p_idx])
            s_val = float(student_pred[s_idx, p_idx])
            abs_err = abs(t_val - s_val)
            rel_err = abs_err / (abs(t_val) + 1e-12)
            sq_err = (t_val - s_val) ** 2
            needs_update = abs_err > error_threshold

            # Priority score: combine relative error with a late-phase boost
            phase_weight = {'early': 1.0, 'mid': 1.2, 'late': 1.5}[phase]
            priority = rel_err * phase_weight

            rows.append({
                'condition_idx': c_idx,
                'condition_name': c_name,
                'time_idx': t_idx,
                'time_min': round(t_min, 4),
                'protein_idx': p_idx,
                'protein_name': protein_names[p_idx],
                'teacher_pred': round(t_val, 6),
                'student_pred': round(s_val, 6),
                'abs_error': round(abs_err, 6),
                'rel_error': round(rel_err, 6),
                'squared_error': round(sq_err, 8),
                'needs_update': int(needs_update),
                'priority_score': round(priority, 6),
                'time_phase': phase,
            })

    df = pd.DataFrame(rows)

    # ---- Summary statistics appended as a separate section -----------------
    summary_rows = []

    # Per-protein summary
    for p_idx, pname in enumerate(protein_names):
        mask = df['protein_idx'] == p_idx
        sub = df.loc[mask]
        summary_rows.append({
            'level': 'protein',
            'name': pname,
            'mean_abs_error': round(sub['abs_error'].mean(), 6),
            'mean_rel_error': round(sub['rel_error'].mean(), 6),
            'max_abs_error': round(sub['abs_error'].max(), 6),
            'pct_needs_update': round(sub['needs_update'].mean() * 100, 2),
            'mean_priority': round(sub['priority_score'].mean(), 6),
        })

    # Per-condition summary
    for c_idx in sorted(df['condition_idx'].unique()):
        mask = df['condition_idx'] == c_idx
        sub = df.loc[mask]
        c_name = sub['condition_name'].iloc[0]
        summary_rows.append({
            'level': 'condition',
            'name': c_name,
            'mean_abs_error': round(sub['abs_error'].mean(), 6),
            'mean_rel_error': round(sub['rel_error'].mean(), 6),
            'max_abs_error': round(sub['abs_error'].max(), 6),
            'pct_needs_update': round(sub['needs_update'].mean() * 100, 2),
            'mean_priority': round(sub['priority_score'].mean(), 6),
        })

    # Per-phase summary
    for phase in ['early', 'mid', 'late']:
        mask = df['time_phase'] == phase
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        summary_rows.append({
            'level': 'time_phase',
            'name': phase,
            'mean_abs_error': round(sub['abs_error'].mean(), 6),
            'mean_rel_error': round(sub['rel_error'].mean(), 6),
            'max_abs_error': round(sub['abs_error'].max(), 6),
            'pct_needs_update': round(sub['needs_update'].mean() * 100, 2),
            'mean_priority': round(sub['priority_score'].mean(), 6),
        })

    df_summary = pd.DataFrame(summary_rows)

    # ---- Save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✓ Main CSV saved: {output_path}  ({len(df):,} rows)")

    summary_path = output_path.replace('.csv', '_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ Summary CSV saved: {summary_path}  ({len(df_summary)} rows)")

    # ---- Console report -----------------------------------------------------
    n_update = int(df['needs_update'].sum())
    pct_update = n_update / len(df) * 100
    print(f"\n{'='*60}")
    print(f"ONLINE LEARNING REGULATION REPORT")
    print(f"{'='*60}")
    print(f"  Total prediction points : {len(df):,}")
    print(f"  Points needing update   : {n_update:,} ({pct_update:.1f}%)")
    print(f"  Error threshold         : {error_threshold}")
    print(f"  Mean absolute error     : {df['abs_error'].mean():.6f}")
    print(f"  Mean relative error     : {df['rel_error'].mean():.4f}")
    print(f"  Mean priority score     : {df['priority_score'].mean():.4f}")
    print(f"\n  Phase breakdown:")
    for phase in ['early', 'mid', 'late']:
        sub = df[df['time_phase'] == phase]
        if len(sub):
            print(f"    {phase:6s}: MAE={sub['abs_error'].mean():.6f}  "
                  f"update={sub['needs_update'].mean()*100:.1f}%  "
                  f"priority={sub['priority_score'].mean():.4f}")

    # Top-10 proteins needing most updates
    print(f"\n  Top-10 proteins by mean priority:")
    top = df_summary[df_summary['level'] == 'protein'].nlargest(10, 'mean_priority')
    for _, row in top.iterrows():
        print(f"    {row['name']:30s}  priority={row['mean_priority']:.4f}  "
              f"update={row['pct_needs_update']:.1f}%")
    print(f"{'='*60}")

    return df, df_summary


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CSV for online learning regulation from a student model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--student-model', type=str, required=True,
                        help='Path to student model checkpoint (.pt)')
    parser.add_argument('--high-res-data', type=str, required=True,
                        help='Path to high-resolution teacher data (.npz)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: auto-generated in model dir)')
    parser.add_argument('--error-threshold', type=float, default=0.05,
                        help='Absolute error threshold for flagging updates')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Compute device')
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = get_device(args.device)

    # Load student model
    print("\n" + "=" * 60 + "\nLoading Student Model\n" + "=" * 60)
    student_model, student_config = load_student_model(args.student_model, device)

    # Load high-res teacher data
    print("\n" + "=" * 60 + "\nLoading High-Resolution Data\n" + "=" * 60)
    X, teacher_pred, metadata = load_high_res_data(args.high_res_data)

    # Generate student predictions
    print("\n" + "=" * 60 + "\nGenerating Student Predictions (raw mode)\n" + "=" * 60)
    student_pred = predict_with_model(student_model, X, device, args.batch_size)
    print(f"✓ Generated {len(student_pred):,} predictions")
    print(f"  Student output shape: {student_pred.shape}")
    print(f"  Student prediction range: [{student_pred.min():.4f}, {student_pred.max():.4f}]")
    print(f"  Teacher prediction range: [{teacher_pred.min():.4f}, {teacher_pred.max():.4f}]")

    # Output path
    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(args.student_model), 'online_learning')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'online_learning_regulation_{ts}.csv')
    else:
        output_path = args.output

    # Generate CSV
    df, df_summary = generate_online_learning_csv(
        teacher_pred=teacher_pred,
        student_pred=student_pred,
        metadata=metadata,
        output_path=output_path,
        error_threshold=args.error_threshold,
    )

    print(f"\n✓ Done. Use the CSV to drive your online learning loop.")


if __name__ == "__main__":
    main()