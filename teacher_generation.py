"""
Generate High-Resolution Predictions from Trained MCF7 Teacher Model.

This script generates dense predictions across treatment conditions and time points
for downstream analysis (e.g., HMM clustering, trajectory analysis).
Uses raw data directly (no preprocessing, no scaling) — consistent with train_teacher_multi.py.
"""

import numpy as np
import pandas as pd
import torch
from models import ResidualMLP, MLP
from tqdm import tqdm
import argparse
import os

# ============================================================================
# MCF7 Dataset Constants
# ============================================================================
MIDAS_TREATMENT_PREFIX = 'TR:'
MIDAS_DATA_AVG_PREFIX = 'DA:'
MIDAS_DATA_VAL_PREFIX = 'DV:'


def load_teacher_model(model_path: str, device: str = 'cpu') -> tuple:
    """
    Load trained teacher model from checkpoint.
    
    Parameters
    ----------
    model_path : str
        Path to model checkpoint
    device : str
        Device to load model on
        
    Returns
    -------
    tuple
        (model, column_info)
    """
    print(f"Loading teacher model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = checkpoint.get('model_type', 'ResidualMLP')
    
    # Get dimensions from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Infer input/output size from weights
        if 'input_proj.weight' in state_dict:
            # ResidualMLP
            input_size = state_dict['input_proj.weight'].shape[1]
            output_size = state_dict['output_proj.weight'].shape[0]
            hidden_dim = state_dict['input_proj.weight'].shape[0]
            num_blocks = sum(1 for k in state_dict.keys() if 'blocks.' in k and '.ln.weight' in k)
            
            model = ResidualMLP(
                input_size=input_size,
                output_size=output_size,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                dropout=0.0
            )
            print(f"  Model type: ResidualMLP")
            print(f"  Architecture: {num_blocks} blocks, hidden_dim={hidden_dim}")
            
        elif 'network.0.weight' in state_dict:
            # MLP
            input_size = state_dict['network.0.weight'].shape[1]
            layer_keys = [k for k in state_dict.keys() if 'network.' in k and '.weight' in k]
            last_layer_key = sorted(layer_keys, key=lambda x: int(x.split('.')[1]))[-1]
            output_size = state_dict[last_layer_key].shape[0]
            
            hidden_sizes = []
            for k in sorted(layer_keys[:-1], key=lambda x: int(x.split('.')[1])):
                hidden_sizes.append(state_dict[k].shape[0])
            
            model = MLP(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=hidden_sizes,
                dropout=0.0
            )
            print(f"  Model type: MLP")
            print(f"  Architecture: hidden_sizes={hidden_sizes}")
        
        print(f"  Input size: {input_size}")
        print(f"  Output size: {output_size}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    column_info = checkpoint.get('column_info', None)
    print(f"  Has column_info: {column_info is not None}")
    
    return model, column_info


def load_midas_data(file_path: str) -> tuple:
    """
    Load and parse MIDAS format data.
    
    Parameters
    ----------
    file_path : str
        Path to MIDAS CSV file
        
    Returns
    -------
    tuple
        (X, y, column_info, df)
    """
    df = pd.read_csv(file_path)
    
    treatment_cols = [c for c in df.columns if c.startswith(MIDAS_TREATMENT_PREFIX)]
    da_cols = [c for c in df.columns if c.startswith(MIDAS_DATA_AVG_PREFIX)]
    dv_cols = [c for c in df.columns if c.startswith(MIDAS_DATA_VAL_PREFIX)]
    
    stimuli_cols = [c for c in treatment_cols if ':Stimuli' in c]
    inhibitor_cols = [c for c in treatment_cols if ':Inhibitors' in c]
    cell_line_cols = [c for c in treatment_cols if ':CellLine' in c]
    
    # Build input matrix: [cell_line, stimuli, inhibitors, time]
    X_parts = []
    if cell_line_cols:
        X_parts.append(df[cell_line_cols].values)
    X_parts.append(df[stimuli_cols].values)
    X_parts.append(df[inhibitor_cols].values)
    if da_cols:
        X_parts.append(df[da_cols].values)
    
    X = np.hstack(X_parts)
    y = df[dv_cols].values
    
    column_info = {
        'treatment_cols': treatment_cols,
        'stimuli_cols': stimuli_cols,
        'inhibitor_cols': inhibitor_cols,
        'cell_line_cols': cell_line_cols,
        'da_cols': da_cols,
        'dv_cols': dv_cols,
        'protein_names': [c.replace(MIDAS_DATA_VAL_PREFIX, '') for c in dv_cols],
        'stimuli_names': [c.split(':')[1] for c in stimuli_cols],
        'inhibitor_names': [c.split(':')[1] for c in inhibitor_cols],
    }
    
    return X, y, column_info, df


def get_unique_treatment_conditions(X: np.ndarray, column_info: dict) -> np.ndarray:
    """
    Extract unique treatment conditions (excluding time).
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    column_info : dict
        Column information
        
    Returns
    -------
    np.ndarray
        Unique treatment condition patterns
    """
    n_cell = len(column_info['cell_line_cols'])
    n_stim = len(column_info['stimuli_cols'])
    n_inhib = len(column_info['inhibitor_cols'])
    
    # Extract treatment pattern (excluding time)
    treatment_pattern = X[:, :n_cell + n_stim + n_inhib]
    
    # Get unique patterns
    unique_patterns = np.unique(treatment_pattern, axis=0)
    
    return unique_patterns


def generate_high_resolution_data(
    model: torch.nn.Module, 
    treatment_conditions: np.ndarray,
    column_info: dict,
    n_time_points: int = 100,
    time_range: tuple = (0, 240),
    device: str = 'cpu',
    observed_times: np.ndarray = None,
) -> dict:
    """
    Generate high-resolution predictions from teacher model for MCF7 data.
    
    Uses raw time values directly (no log transform, no scaling) — consistent
    with train_teacher_multi.py.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained teacher model
    treatment_conditions : np.ndarray
        Unique treatment conditions (n_conditions, n_treatment_features)
    column_info : dict
        Column information from MIDAS data
    n_time_points : int
        Number of high-resolution time points per interval
    time_range : tuple
        (min_time, max_time) in minutes
    device : str
        Device for computation
    observed_times : np.ndarray or None
        Actual measured time points. If provided, dense points are generated
        only between consecutive observed times (no gap extrapolation).
        
    Returns
    -------
    dict
        Dictionary containing high-resolution predictions and metadata
    """
    print(f"\nGenerating high-resolution predictions...")
    print(f"  Treatment conditions: {len(treatment_conditions)}")
    print(f"  Time range: {time_range[0]} to {time_range[1]} min")
    
    if observed_times is not None:
        # ── Observed-anchored: dense points between known time points only ──
        observed_times = np.sort(np.unique(observed_times))
        print(f"  Observed time points: {observed_times}")
        print(f"  Strategy: interpolation between observed times only")
        
        n_intervals = len(observed_times) - 1
        pts_per_interval = max(2, n_time_points // n_intervals)
        
        time_segments = []
        for k in range(n_intervals):
            t_start = observed_times[k]
            t_end = observed_times[k + 1]
            # Linear fill within each observed interval
            segment = np.linspace(t_start, t_end, pts_per_interval, endpoint=False)
            time_segments.append(segment)
        
        # Add the last observed time
        time_segments.append([observed_times[-1]])
        time_points = np.sort(np.unique(np.concatenate(time_segments)))
    else:
        # ── Fallback: log-spaced (includes unseen gaps) ──
        print(f"  Strategy: log-spaced (WARNING: includes unseen time gaps)")
        time_points = np.concatenate([
            [0],
            np.logspace(np.log10(1), np.log10(time_range[1]), n_time_points - 1)
        ])
        time_points = np.sort(np.unique(time_points))
    
    print(f"  Generated time points: {len(time_points)}")
    print(f"  Range: [{time_points[0]:.1f}, ..., {time_points[-1]:.1f}]")
    
    # Create input grid: [treatment_conditions, time]
    # Raw time values — no log transform, no scaling
    X_high_res = []
    condition_indices = []
    time_indices = []
    
    for cond_idx, condition in enumerate(treatment_conditions):
        for t_idx, t in enumerate(time_points):
            # Combine treatment condition with raw time value
            x_sample = np.concatenate([condition, [t]])
            X_high_res.append(x_sample)
            condition_indices.append(cond_idx)
            time_indices.append(t_idx)
    
    X_high_res = np.array(X_high_res, dtype=np.float32)
    condition_indices = np.array(condition_indices)
    time_indices = np.array(time_indices)
    
    print(f"  Total samples: {len(X_high_res)}")
    print(f"  Input shape: {X_high_res.shape}")
    
    # Predict in batches (raw input, no preprocessing)
    print("  Generating predictions...")
    X_tensor = torch.tensor(X_high_res, dtype=torch.float32).to(device)
    
    predictions = []
    batch_size = 512
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size), desc="Predicting"):
            batch = X_tensor[i:i+batch_size]
            pred = model(batch).cpu().numpy()
            predictions.append(pred)
    
    predictions = np.vstack(predictions)
    
    # Ensure non-negative (physical constraint for phosphorylation)
    predictions = np.maximum(predictions, 0.0)
    
    print(f"  Output shape: {predictions.shape}")
    print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Organize results
    results = {
        'predictions': predictions,
        'X_high_res': X_high_res,
        'time_points': time_points,
        'treatment_conditions': treatment_conditions,
        'condition_indices': condition_indices,
        'time_indices': time_indices,
        'column_info': column_info,
        'n_conditions': len(treatment_conditions),
        'n_time_points': len(time_points),
        'n_proteins': predictions.shape[1],
    }
    
    return results


def get_condition_name(condition: np.ndarray, column_info: dict) -> str:
    """
    Get human-readable name for a treatment condition.
    
    Parameters
    ----------
    condition : np.ndarray
        Treatment condition vector
    column_info : dict
        Column information
        
    Returns
    -------
    str
        Human-readable condition name
    """
    n_cell = len(column_info['cell_line_cols'])
    n_stim = len(column_info['stimuli_cols'])
    n_inhib = len(column_info['inhibitor_cols'])
    
    # Extract components
    stimuli = condition[n_cell:n_cell+n_stim]
    inhibitors = condition[n_cell+n_stim:n_cell+n_stim+n_inhib]
    
    # Get active stimuli names
    active_stimuli = []
    for i, val in enumerate(stimuli):
        if val == 1:
            active_stimuli.append(column_info['stimuli_names'][i])
    
    # Get active inhibitor names
    active_inhibitors = []
    for i, val in enumerate(inhibitors):
        if val == 1:
            active_inhibitors.append(column_info['inhibitor_names'][i])
    
    stim_str = '+'.join(active_stimuli) if active_stimuli else 'None'
    inhib_str = '+'.join(active_inhibitors) if active_inhibitors else 'None'
    
    return f"{stim_str}|{inhib_str}"


def save_results(results: dict, output_dir: str, model_name: str):
    """
    Save high-resolution prediction results.
    
    Parameters
    ----------
    results : dict
        Results from generate_high_resolution_data
    output_dir : str
        Output directory
    model_name : str
        Name for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_cond = results['n_conditions']
    n_time = results['n_time_points']
    n_prot = results['n_proteins']
    
    # Save as NPZ file (efficient binary format)
    output_path = os.path.join(output_dir, f'{model_name}_high_res_{n_cond}cond_{n_time}times.npz')
    
    np.savez_compressed(
        output_path,
        predictions=results['predictions'],
        X_high_res=results['X_high_res'],
        time_points=results['time_points'],
        treatment_conditions=results['treatment_conditions'],
        condition_indices=results['condition_indices'],
        time_indices=results['time_indices'],
        n_conditions=n_cond,
        n_time_points=n_time,
        n_proteins=n_prot,
        protein_names=np.array(results['column_info']['protein_names']),
        condition_names=np.array([get_condition_name(c, results['column_info']) 
                                  for c in results['treatment_conditions']]),
    )
    print(f"✓ Saved predictions to: {output_path}")
    
    # Save protein names separately (for easier loading)
    protein_names_path = os.path.join(output_dir, f'{model_name}_protein_names.txt')
    with open(protein_names_path, 'w') as f:
        for name in results['column_info']['protein_names']:
            f.write(f"{name}\n")
    print(f"✓ Saved protein names to: {protein_names_path}")
    
    # Save condition names
    condition_names_path = os.path.join(output_dir, f'{model_name}_condition_names.txt')
    with open(condition_names_path, 'w') as f:
        for cond in results['treatment_conditions']:
            name = get_condition_name(cond, results['column_info'])
            f.write(f"{name}\n")
    print(f"✓ Saved condition names to: {condition_names_path}")
    
    # Save as CSV for easy inspection
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    # Create summary CSV with all data
    summary_data = []
    for i in range(len(results['predictions'])):
        cond_idx = results['condition_indices'][i]
        time_idx = results['time_indices'][i]
        cond = results['treatment_conditions'][cond_idx]
        cond_name = get_condition_name(cond, results['column_info'])
        time_val = results['time_points'][time_idx]
        
        row = {
            'condition': cond_name,
            'condition_idx': cond_idx,
            'time': time_val,
            'time_idx': time_idx,
        }
        
        # Add protein predictions
        for j, prot_name in enumerate(results['column_info']['protein_names']):
            row[prot_name] = results['predictions'][i, j]
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(csv_dir, f'{model_name}_predictions_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✓ Saved summary CSV to: {summary_csv_path}")
    
    return output_path


def plot_teacher_overview(
    model: torch.nn.Module,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    high_res_results: dict,
    column_info: dict,
    output_dir: str,
    device: str = 'cpu'
):
    """
    Publication figure: raw data, training predictions, and high-res extension
    in one plot. Shows model fidelity and interpolation quality.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained teacher model
    X_raw : np.ndarray
        Raw input features from MIDAS (all samples including replicates)
    y_raw : np.ndarray
        Raw target values from MIDAS
    high_res_results : dict
        Results from generate_high_resolution_data
    column_info : dict
        Column information
    output_dir : str
        Output directory
    device : str
        Device for inference
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    mpl.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.dpi": 300, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    
    PROTEIN_DISPLAY_NAMES = {
        'EGFR_pY1068': 'p-EGFR (Y1068)', 'AKT_pS473': 'p-AKT (S473)',
        'MAPK_pT202_Y204': 'p-ERK1/2', 'MEK1_pS217_S221': 'p-MEK1',
        'mTOR_pS2448': 'p-mTOR', 'S6_pS235_S236': 'p-S6 (S235/236)',
        'p70S6K_pT389': 'p-p70S6K', 'STAT3_pY705': 'p-STAT3',
    }
    
    STIMULI_COLORS = {
        'EGF': '#E64B35', 'Insulin': '#4DBBD5', 'FGF1': '#00A087',
        'HGF': '#3C5488', 'IGF1': '#F39B7F', 'Serum': '#8491B4',
        'NRG1': '#91D1C2',
    }
    
    protein_names = column_info['protein_names']
    
    # --- Compute training predictions at raw time points ---
    X_tensor = torch.tensor(X_raw.astype(np.float32)).to(device)
    with torch.no_grad():
        y_pred_train = model(X_tensor).cpu().numpy()
    
    # --- Parse raw sample metadata ---
    n_cell = len(column_info['cell_line_cols'])
    n_stim = len(column_info['stimuli_cols'])
    n_inhib = len(column_info['inhibitor_cols'])
    
    raw_metadata = []
    for idx in range(len(X_raw)):
        stimuli = X_raw[idx, n_cell:n_cell+n_stim]
        inhibitors = X_raw[idx, n_cell+n_stim:n_cell+n_stim+n_inhib]
        time_val = X_raw[idx, -1]  # last column is time
        
        active_stim = [column_info['stimuli_names'][i] 
                       for i, v in enumerate(stimuli) if v == 1]
        active_inhib = [column_info['inhibitor_names'][i] 
                        for i, v in enumerate(inhibitors) if v == 1]
        
        raw_metadata.append({
            'stimuli': '+'.join(active_stim) if active_stim else 'None',
            'inhibitors': '+'.join(active_inhib) if active_inhib else 'None',
            'time': time_val,
        })
    
    # --- Select proteins: top 4 by R² from KEY_PROTEINS ---
    KEY_PROTEINS = ['EGFR_pY1068', 'AKT_pS473', 'MAPK_pT202_Y204', 
                    'mTOR_pS2448', 'S6_pS235_S236', 'p70S6K_pT389',
                    'MEK1_pS217_S221', 'STAT3_pY705']
    
    available_key = [p for p in KEY_PROTEINS if p in protein_names]
    
    # Compute per-protein R²
    from sklearn.metrics import r2_score
    protein_r2 = {}
    for p in available_key:
        pidx = protein_names.index(p)
        gt = y_raw[:, pidx]
        pred = y_pred_train[:, pidx]
        protein_r2[p] = r2_score(gt, pred) if np.var(gt) > 1e-10 else 0.0
    
    available_key.sort(key=lambda p: protein_r2[p], reverse=True)
    selected_proteins = available_key[:4]
    
    # --- Select stimuli (no-inhibitor only) ---
    preferred_stimuli = ['EGF', 'Insulin', 'FGF1', 'HGF']
    no_inhib_stimuli = set()
    for m in raw_metadata:
        if m['inhibitors'] == 'None':
            no_inhib_stimuli.add(m['stimuli'])
    key_stimuli = [s for s in preferred_stimuli if s in no_inhib_stimuli][:4]
    
    n_proteins = len(selected_proteins)
    n_stimuli = len(key_stimuli)
    
    if n_proteins == 0 or n_stimuli == 0:
        print("  ⚠ Not enough data for overview plot")
        return
    
    # --- Plot ---
    fig, axes = plt.subplots(n_proteins, n_stimuli,
                             figsize=(3.0 * n_stimuli, 2.4 * n_proteins),
                             sharex=True)
    if n_proteins == 1:
        axes = axes.reshape(1, -1)
    if n_stimuli == 1:
        axes = axes.reshape(-1, 1)
    
    # Organize high-res data by condition name
    hr_by_condition = {}
    for cond_idx, cond in enumerate(high_res_results['treatment_conditions']):
        cond_name = get_condition_name(cond, column_info)
        mask = high_res_results['condition_indices'] == cond_idx
        times = high_res_results['time_points']
        preds = high_res_results['predictions'][mask]
        hr_by_condition[cond_name] = {'times': times, 'predictions': preds}
    
    for i, protein in enumerate(selected_proteins):
        pidx = protein_names.index(protein)
        r2 = protein_r2[protein]
        
        for j, stim in enumerate(key_stimuli):
            ax = axes[i, j]
            color = STIMULI_COLORS.get(stim, '#666666')
            
            # 1) Raw data points (open circles) — all replicates
            raw_times = []
            raw_vals = []
            for idx, m in enumerate(raw_metadata):
                if m['stimuli'] == stim and m['inhibitors'] == 'None':
                    raw_times.append(m['time'])
                    raw_vals.append(y_raw[idx, pidx])
            
            if raw_times:
                ax.scatter(raw_times, raw_vals, 
                          marker='o', s=25, linewidths=1.0,
                          edgecolors='#333333', facecolors='white',
                          zorder=10, label='Raw data' if i == 0 and j == 0 else '')
            
            # 2) Training predictions at raw time points (filled diamonds)
            train_times = []
            train_vals = []
            for idx, m in enumerate(raw_metadata):
                if m['stimuli'] == stim and m['inhibitors'] == 'None':
                    train_times.append(m['time'])
                    train_vals.append(y_pred_train[idx, pidx])
            
            if train_times:
                # Average predictions at each unique time (for cleaner display)
                unique_t = np.unique(train_times)
                avg_pred = []
                for t in unique_t:
                    mask_t = np.array(train_times) == t
                    avg_pred.append(np.mean(np.array(train_vals)[mask_t]))
                
                ax.scatter(unique_t, avg_pred, 
                          marker='D', s=30, linewidths=0.8,
                          edgecolors=color, facecolors=color, alpha=0.8,
                          zorder=15, label='Train pred' if i == 0 and j == 0 else '')
            
            # 3) High-res extension (smooth line)
            hr_key = f"{stim}|None"
            if hr_key in hr_by_condition:
                hr_data = hr_by_condition[hr_key]
                hr_times = hr_data['times']
                hr_preds = hr_data['predictions'][:, pidx]
                
                ax.plot(hr_times, hr_preds, 
                       color=color, linewidth=1.8, alpha=0.9,
                       zorder=5, label='Dense pred' if i == 0 and j == 0 else '')
            
            # Formatting
            ax.set_xscale('symlog', linthresh=1)
            
            if i == 0:
                ax.set_title(stim, fontsize=11, fontweight='bold', color=color)
            if j == 0:
                display_name = PROTEIN_DISPLAY_NAMES.get(protein, protein)
                ax.set_ylabel(f'{display_name}\n($R^2$={r2:.2f})', 
                            fontsize=9, fontweight='bold')
            if i == n_proteins - 1:
                ax.set_xlabel('Time (min)', fontsize=10)
            
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.4)
            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Shared legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3,
                  fontsize=9, framealpha=0.9,
                  bbox_to_anchor=(0.5, 1.03))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/teacher_overview.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/teacher_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved teacher overview plot to {output_dir}/teacher_overview.pdf")


def main():
    parser = argparse.ArgumentParser(
        description='Generate high-resolution predictions from trained MCF7 teacher model (raw mode)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained teacher model')
    parser.add_argument('--data_path', type=str, 
                       default='experimental/MIDAS/MD_MCF7_main.csv',
                       help='Path to MIDAS data file (to get treatment conditions)')
    parser.add_argument('--n_time_points', type=int, default=100,
                       help='Number of high-resolution time points')
    parser.add_argument('--time_max', type=float, default=240,
                       help='Maximum time in minutes')
    parser.add_argument('--output_dir', type=str, default='data/teacher_predictions',
                       help='Output directory')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate overview plot')
    args = parser.parse_args()
    
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load model
    model, model_column_info = load_teacher_model(args.model, device)
    
    # Load MIDAS data to get treatment conditions
    print(f"\nLoading MIDAS data from: {args.data_path}")
    X, y, column_info, df = load_midas_data(args.data_path)
    
    print(f"  Samples: {len(X)}")
    print(f"  Input features: {X.shape[1]}")
    print(f"  Output proteins: {y.shape[1]}")
    print(f"  Protein names: {column_info['protein_names'][:5]}...")
    
    # Get unique treatment conditions
    treatment_conditions = get_unique_treatment_conditions(X, column_info)
    print(f"\n  Unique treatment conditions: {len(treatment_conditions)}")
    
    # Print some example conditions
    print("\n  Example conditions:")
    for i, cond in enumerate(treatment_conditions[:5]):
        name = get_condition_name(cond, column_info)
        print(f"    {i+1}. {name}")
    
    # Extract observed time points from data
    observed_times = np.sort(np.unique(X[:, -1]))
    print(f"\n  Observed time points: {observed_times}")
    
    # Generate high-resolution predictions
    results = generate_high_resolution_data(
        model=model,
        treatment_conditions=treatment_conditions,
        column_info=column_info,
        n_time_points=args.n_time_points,
        time_range=(0, args.time_max),
        device=device,
        observed_times=observed_times,
    )
    
    # Save results
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    output_path = save_results(results, args.output_dir, model_name)
    
    # Generate overview plot
    if args.plot:
        print("\nGenerating overview plot...")
        plot_teacher_overview(
            model=model,
            X_raw=X,
            y_raw=y,
            high_res_results=results,
            column_info=column_info,
            output_dir=args.output_dir,
            device=device
        )
    
    # Print summary
    print("\n" + "=" * 70)
    print("HIGH-RESOLUTION PREDICTION COMPLETE!")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Treatment conditions: {results['n_conditions']}")
    print(f"Time points: {results['n_time_points']}")
    print(f"Proteins: {results['n_proteins']}")
    print(f"Total predictions: {len(results['predictions'])}")
    print(f"\nOutput saved to: {args.output_dir}/")
    
    print("\n" + "=" * 70)
    print("READY FOR DOWNSTREAM ANALYSIS!")
    print("=" * 70)
    print(f"\nTo perform trajectory analysis or HMM clustering:")
    print(f"  python trajectory_analysis.py --data {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()