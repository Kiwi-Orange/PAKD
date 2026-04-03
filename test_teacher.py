"""
Test Teacher Model on MCF7 Signaling Data.

This script evaluates trained MLP or ResidualMLP models on MCF7
phosphoprotein prediction tasks, with proper handling of biological
replicates and publication-quality visualizations.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import os
import argparse
from tqdm import tqdm
from models import MLP, ResidualMLP
import matplotlib as mpl
import seaborn as sns

# Configure matplotlib for publication quality (matching POLLU style)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Roman']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'stix'

# ============================================================================
# MCF7 Dataset Constants
# ============================================================================
MIDAS_TREATMENT_PREFIX = 'TR:'
MIDAS_DATA_AVG_PREFIX = 'DA:'
MIDAS_DATA_VAL_PREFIX = 'DV:'

# Key proteins for focused analysis
KEY_PROTEINS = ['EGFR_pY1068', 'AKT_pS473', 'MAPK_pT202_Y204', 
                'mTOR_pS2448', 'S6_pS235_S236', 'p70S6K_pT389',
                'MEK1_pS217_S221', 'STAT3_pY705']

# Protein display names for cleaner figures
PROTEIN_DISPLAY_NAMES = {
    'EGFR_pY1068': 'p-EGFR (Y1068)',
    'EGFR_pY1173': 'p-EGFR (Y1173)',
    'AKT_pS473': 'p-AKT (S473)',
    'AKT_pT308': 'p-AKT (T308)',
    'MAPK_pT202_Y204': 'p-ERK1/2',
    'MEK1_pS217_S221': 'p-MEK1',
    'mTOR_pS2448': 'p-mTOR',
    'S6_pS235_S236': 'p-S6 (S235/236)',
    'S6_pS240_S244': 'p-S6 (S240/244)',
    'p70S6K_pT389': 'p-p70S6K',
    '4EBP1_pS65': 'p-4EBP1',
    'STAT3_pY705': 'p-STAT3',
}

# Color scheme matching POLLU style
COLORS = {
    'ground_truth': '#000000',
    'prediction': '#E84A27',
    'ci_band': '#E84A27',
    'good': '#2ecc71',
    'moderate': '#f39c12', 
    'poor': '#e74c3c',
}

# Stimuli colors
STIMULI_COLORS = {
    'EGF': '#E64B35',
    'Insulin': '#4DBBD5',
    'FGF1': '#00A087',
    'HGF': '#3C5488',
    'IGF1': '#F39B7F',
    'Serum': '#8491B4',
    'NRG1': '#91D1C2',
}


# ============================================================================
# Data Loading
# ============================================================================
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
        (X, y, column_info, df) where X is inputs, y is outputs
    """
    df = pd.read_csv(file_path)
    
    # Parse column types
    treatment_cols = [c for c in df.columns if c.startswith(MIDAS_TREATMENT_PREFIX)]
    da_cols = [c for c in df.columns if c.startswith(MIDAS_DATA_AVG_PREFIX)]
    dv_cols = [c for c in df.columns if c.startswith(MIDAS_DATA_VAL_PREFIX)]
    
    # Extract stimuli and inhibitor columns
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


def preprocess_time(X: np.ndarray, time_col_idx: int = -1) -> np.ndarray:
    """Apply log transformation to time column."""
    X_processed = X.copy()
    time = X_processed[:, time_col_idx]
    X_processed[:, time_col_idx] = np.log10(time + 1.0)  # log10(time + 1)
    return X_processed


def align_baseline_t0(X: np.ndarray, y: np.ndarray, column_info: dict) -> np.ndarray:
    """
    Align t=0 baseline by replacing targets with the global median per inhibitor condition.
    
    In real biological experiments, cells should be in the same state before stimulus
    introduction (t=0). However, batch effects cause t=0 readings to vary across
    different stimuli conditions under the same inhibitor. This creates a "one-to-many"
    mapping problem for neural networks (same input features map to different targets),
    hindering convergence.
    
    Solution: For each unique inhibitor condition, compute the median of all t=0
    target values and replace individual t=0 targets with that median.
    
    Parameters
    ----------
    X : np.ndarray
        Input features (before aggregation), with time in the last column
    y : np.ndarray
        Target phosphoprotein measurements, shape (N, num_proteins)
    column_info : dict
        Column information from load_midas_data
        
    Returns
    -------
    np.ndarray
        y with t=0 samples replaced by per-inhibitor-condition median
    """
    y_aligned = y.copy()
    time_col = X[:, -1]
    
    # Identify t=0 samples (time == 0)
    t0_mask = (time_col == 0)
    n_t0 = np.sum(t0_mask)
    
    if n_t0 == 0:
        print("  ⚠ No t=0 samples found, skipping baseline alignment")
        return y_aligned
    
    # Extract inhibitor columns from X
    n_cell = len(column_info['cell_line_cols'])
    n_stim = len(column_info['stimuli_cols'])
    n_inhib = len(column_info['inhibitor_cols'])
    
    # Inhibitor columns start after cell_line and stimuli columns
    inhib_start = n_cell + n_stim
    inhib_end = inhib_start + n_inhib
    inhibitor_pattern = X[:, inhib_start:inhib_end]
    
    # Find unique inhibitor conditions
    unique_inhib, inhib_labels = np.unique(inhibitor_pattern, axis=0, return_inverse=True)
    
    n_replaced = 0
    for inhib_idx in range(len(unique_inhib)):
        # Find t=0 samples with this inhibitor condition
        condition_mask = (inhib_labels == inhib_idx) & t0_mask
        n_samples = np.sum(condition_mask)
        
        if n_samples == 0:
            continue
        
        # Compute median across all t=0 samples for this inhibitor condition
        median_target = np.median(y[condition_mask], axis=0)
        
        # Replace all t=0 targets under this inhibitor condition with the median
        y_aligned[condition_mask] = median_target
        n_replaced += n_samples
    
    print(f"  ✓ Baseline alignment: replaced {n_replaced} t=0 samples "
          f"across {len(unique_inhib)} inhibitor conditions with per-condition median")
    
    # Report the reduction in target variance at t=0
    if n_t0 > 1:
        var_before = np.mean(np.var(y[t0_mask], axis=0))
        var_after = np.mean(np.var(y_aligned[t0_mask], axis=0))
        print(f"    Mean target variance at t=0: {var_before:.6f} → {var_after:.6f} "
              f"({(1 - var_after / (var_before + 1e-12)) * 100:.1f}% reduction)")
    
    return y_aligned


def create_condition_labels(X: np.ndarray, column_info: dict) -> np.ndarray:
    """Create condition labels based on treatment combinations."""
    n_cell = len(column_info['cell_line_cols'])
    n_stim = len(column_info['stimuli_cols'])
    n_inhib = len(column_info['inhibitor_cols'])
    
    treatment_pattern = X[:, :n_cell + n_stim + n_inhib]
    unique_patterns, labels = np.unique(treatment_pattern, axis=0, return_inverse=True)
    
    return labels


def get_condition_names(df: pd.DataFrame, column_info: dict) -> list:
    """Get human-readable condition names for each sample."""
    condition_names = []
    
    for idx, row in df.iterrows():
        # Get active stimuli
        active_stimuli = []
        for col in column_info['stimuli_cols']:
            if row[col] == 1:
                stimuli_name = col.split(':')[1]
                active_stimuli.append(stimuli_name)
        
        # Get active inhibitors
        active_inhibitors = []
        for col in column_info['inhibitor_cols']:
            if row[col] == 1:
                inhibitor_name = col.split(':')[1]
                active_inhibitors.append(inhibitor_name)
        
        stim_str = '+'.join(active_stimuli) if active_stimuli else 'None'
        inhib_str = '+'.join(active_inhibitors) if active_inhibitors else 'None'
        time_point = row['DA:ALL'] if 'DA:ALL' in df.columns else 0
        
        condition_names.append(f"{stim_str}|{inhib_str}|t={time_point}")
    
    return condition_names


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for data.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    confidence : float
        Confidence level (default 0.95)
        
    Returns
    -------
    tuple
        (mean, ci_low, ci_high, std, n)
    """
    n = len(data)
    if n < 2:
        mean = np.mean(data) if n > 0 else np.nan
        return mean, mean, mean, 0.0, n
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - h, mean + h, std, n


# ============================================================================
# Replicate-Aware Data Aggregation
# ============================================================================
def aggregate_replicates(X: np.ndarray, y: np.ndarray, column_info: dict, 
                         df: pd.DataFrame) -> dict:
    """
    Aggregate biological replicates for each unique condition.
    
    When the same input condition (stimuli + inhibitor + time) appears multiple
    times in the data, we compute statistical summaries (mean, CI, std).
    
    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Output values (n_samples, n_proteins)
    column_info : dict
        Column information
    df : pd.DataFrame
        Original dataframe
        
    Returns
    -------
    dict
        Aggregated data with structure:
        {condition_key: {
            'X': input_vector,
            'y_mean': mean output,
            'y_std': std output,
            'y_ci_low': CI lower bound,
            'y_ci_high': CI upper bound,
            'y_all': all replicate values,
            'n_replicates': number of replicates,
            'time': time point,
            'stimuli': stimuli name,
            'inhibitors': inhibitor name,
        }}
    """
    n_cell = len(column_info['cell_line_cols'])
    n_stim = len(column_info['stimuli_cols'])
    n_inhib = len(column_info['inhibitor_cols'])
    
    # Create condition keys (treatment pattern as tuple)
    condition_data = {}
    
    for idx in range(len(X)):
        # Create key from treatment pattern (excluding time for grouping)
        treatment_pattern = tuple(X[idx, :n_cell + n_stim + n_inhib])
        time_point = df.iloc[idx]['DA:ALL'] if 'DA:ALL' in df.columns else 0
        
        # Create unique key for condition + time
        condition_key = treatment_pattern + (time_point,)
        
        if condition_key not in condition_data:
            # Get stimuli and inhibitor names
            active_stimuli = []
            for col in column_info['stimuli_cols']:
                if df.iloc[idx][col] == 1:
                    active_stimuli.append(col.split(':')[1])
            
            active_inhibitors = []
            for col in column_info['inhibitor_cols']:
                if df.iloc[idx][col] == 1:
                    active_inhibitors.append(col.split(':')[1])
            
            condition_data[condition_key] = {
                'X': X[idx],
                'y_all': [],
                'time': time_point,
                'stimuli': '+'.join(active_stimuli) if active_stimuli else 'None',
                'inhibitors': '+'.join(active_inhibitors) if active_inhibitors else 'None',
            }
        
        condition_data[condition_key]['y_all'].append(y[idx])
    
    # Compute statistics for each condition
    for key, data in condition_data.items():
        y_all = np.array(data['y_all'])
        n_replicates = len(y_all)
        
        data['n_replicates'] = n_replicates
        data['y_mean'] = np.median(y_all, axis=0)  # Use median for robustness against outliers
        
        if n_replicates > 1:
            data['y_std'] = np.std(y_all, axis=0, ddof=1)
            # Compute CI for each protein
            ci_lows = []
            ci_highs = []
            for p in range(y_all.shape[1]):
                _, ci_low, ci_high, _, _ = compute_confidence_interval(y_all[:, p])
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
            data['y_ci_low'] = np.array(ci_lows)
            data['y_ci_high'] = np.array(ci_highs)
        else:
            data['y_std'] = np.zeros_like(data['y_mean'])
            data['y_ci_low'] = data['y_mean']
            data['y_ci_high'] = data['y_mean']
        
        data['y_all'] = y_all
    
    return condition_data


# ============================================================================
# Raw-mode Data Preparation (no aggregation)
# ============================================================================
def prepare_raw_data(X: np.ndarray, y: np.ndarray, column_info: dict, 
                     df: pd.DataFrame) -> dict:
    """
    Prepare data in raw mode without any aggregation.
    
    Each individual sample becomes its own "condition" with n_replicates=1.
    This matches the raw training mode where no preprocessing is applied.
    
    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features)
    y : np.ndarray
        Output values (n_samples, n_proteins)
    column_info : dict
        Column information
    df : pd.DataFrame
        Original dataframe
        
    Returns
    -------
    dict
        Data formatted like aggregate_replicates output but with individual samples
    """
    n_cell = len(column_info['cell_line_cols'])
    n_stim = len(column_info['stimuli_cols'])
    n_inhib = len(column_info['inhibitor_cols'])
    
    condition_data = {}
    
    for idx in range(len(X)):
        # Each sample gets a unique key
        condition_key = (idx,)
        
        # Get stimuli and inhibitor names
        active_stimuli = []
        for col in column_info['stimuli_cols']:
            if df.iloc[idx][col] == 1:
                active_stimuli.append(col.split(':')[1])
        
        active_inhibitors = []
        for col in column_info['inhibitor_cols']:
            if df.iloc[idx][col] == 1:
                active_inhibitors.append(col.split(':')[1])
        
        time_point = df.iloc[idx]['DA:ALL'] if 'DA:ALL' in df.columns else 0
        
        condition_data[condition_key] = {
            'X': X[idx],
            'y_all': np.array([y[idx]]),
            'y_mean': y[idx],
            'y_std': np.zeros_like(y[idx]),
            'y_ci_low': y[idx],
            'y_ci_high': y[idx],
            'n_replicates': 1,
            'time': time_point,
            'stimuli': '+'.join(active_stimuli) if active_stimuli else 'None',
            'inhibitors': '+'.join(active_inhibitors) if active_inhibitors else 'None',
        }
    
    return condition_data


# ============================================================================
# Model Evaluator
# ============================================================================
class TeacherModelEvaluator:
    """Evaluator for trained MCF7 signaling models."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.X_scaler = None
        self.y_scaler = None
        self.column_info = None
        self.raw_mode = False
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model_type = checkpoint.get('model_type', 'ResidualMLP')
        self.raw_mode = checkpoint.get('raw_mode', False)
        
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
                
                self.model = ResidualMLP(
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
                # Find output layer
                layer_keys = [k for k in state_dict.keys() if 'network.' in k and '.weight' in k]
                last_layer_key = sorted(layer_keys, key=lambda x: int(x.split('.')[1]))[-1]
                output_size = state_dict[last_layer_key].shape[0]
                
                # Infer hidden sizes
                hidden_sizes = []
                for k in sorted(layer_keys[:-1], key=lambda x: int(x.split('.')[1])):
                    hidden_sizes.append(state_dict[k].shape[0])
                
                self.model = MLP(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_sizes=hidden_sizes,
                    dropout=0.0
                )
                print(f"  Model type: MLP")
                print(f"  Architecture: hidden_sizes={hidden_sizes}")
            
            print(f"  Input size: {input_size}")
            print(f"  Output size: {output_size}")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scalers
        self.X_scaler = checkpoint.get('X_scaler', None)
        self.y_scaler = checkpoint.get('y_scaler', None)
        self.column_info = checkpoint.get('column_info', None)
        
        if self.raw_mode:
            print(f"  ✓ Raw mode: no scaling applied")
        elif self.y_scaler is not None:
            print(f"  ✓ Using normalized targets (y_scaler loaded)")
        else:
            print(f"  ⚠️  No y_scaler found (old model format)")
        
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Raw mode: {self.raw_mode}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict phosphoprotein levels.
        
        Parameters
        ----------
        X : np.ndarray
            Input features (preprocessed if not raw mode), shape (n_samples, input_size)
            
        Returns
        -------
        np.ndarray
            Predicted phosphoprotein levels, shape (n_samples, output_size)
        """
        if self.raw_mode:
            # Raw mode: no scaling at all
            X_norm = X
        elif self.X_scaler is not None:
            X_norm = self.X_scaler.transform(X)
        else:
            X_norm = X
        
        # Predict
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        if self.raw_mode:
            # Raw mode: no inverse scaling
            pass
        elif self.y_scaler is not None:
            predictions = self.y_scaler.inverse_transform(predictions)
        
        return predictions


# ============================================================================
# Evaluation Functions with Replicate Handling
# ============================================================================
def evaluate_model_with_replicates(evaluator: TeacherModelEvaluator, 
                                    aggregated_data: dict,
                                    column_info: dict) -> dict:
    """
    Evaluate model using replicate-aggregated data.
    
    For conditions with multiple replicates, we compare model predictions
    to the median ground truth and compute errors relative to biological variability.
    
    Parameters
    ----------
    evaluator : TeacherModelEvaluator
        Trained model evaluator
    aggregated_data : dict
        Replicate-aggregated data from aggregate_replicates()
    column_info : dict
        Column information
        
    Returns
    -------
    dict
        Evaluation results with replicate-aware metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL PERFORMANCE")
    if evaluator.raw_mode:
        print("(RAW MODE — no preprocessing)")
    else:
        print("(REPLICATE-AWARE)")
    print("=" * 60)
    
    protein_names = column_info['protein_names']
    n_proteins = len(protein_names)
    n_conditions = len(aggregated_data)
    
    # Prepare data for batch prediction
    X_batch = np.array([data['X'] for data in aggregated_data.values()])
    
    if evaluator.raw_mode:
        # Raw mode: feed raw X directly (no time log-transform)
        X_processed = X_batch
    else:
        X_processed = preprocess_time(X_batch, time_col_idx=-1)
    
    # Predict
    predictions = evaluator.predict(X_processed)
    
    # Collect ground truth medians and predictions
    gt_means = np.array([data['y_mean'] for data in aggregated_data.values()])
    gt_stds = np.array([data['y_std'] for data in aggregated_data.values()])
    n_replicates = np.array([data['n_replicates'] for data in aggregated_data.values()])
    
    # Overall metrics (vs median ground truth)
    results = {
        'predictions': predictions,
        'gt_means': gt_means,
        'gt_stds': gt_stds,
        'n_replicates': n_replicates,
        'mae_overall': mean_absolute_error(gt_means.flatten(), predictions.flatten()),
        'rmse_overall': np.sqrt(mean_squared_error(gt_means.flatten(), predictions.flatten())),
        'r2_overall': r2_score(gt_means.flatten(), predictions.flatten()),
    }
    
    # Compute relative error (normalized by signal magnitude)
    gt_flat = gt_means.flatten()
    pred_flat = predictions.flatten()
    eps = 1e-12
    results['relative_error_overall'] = np.mean(np.abs(gt_flat - pred_flat) / (np.abs(gt_flat) + eps))
    
    # Per-protein metrics
    protein_metrics = {}
    for i, protein in enumerate(protein_names):
        gt = gt_means[:, i]
        pred = predictions[:, i]
        gt_std = gt_stds[:, i]
        
        # Basic metrics
        mae = mean_absolute_error(gt, pred)
        rmse = np.sqrt(mean_squared_error(gt, pred))
        r2 = r2_score(gt, pred) if np.var(gt) > eps else 0.0
        corr = np.corrcoef(gt, pred)[0, 1] if np.std(gt) > eps else 0.0
        
        # Biological variability-normalized error
        # Error relative to intrinsic biological variability
        mean_bio_std = np.mean(gt_std[gt_std > eps]) if np.any(gt_std > eps) else 1.0
        normalized_rmse = rmse / mean_bio_std if mean_bio_std > eps else np.nan
        
        protein_metrics[protein] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Correlation': corr,
            'Mean_Bio_Std': mean_bio_std,
            'Normalized_RMSE': normalized_rmse,  # RMSE / biological variability
            'Relative_Error': np.mean(np.abs(gt - pred) / (np.abs(gt) + eps)),
        }
    
    results['protein_metrics'] = protein_metrics
    results['aggregated_data'] = aggregated_data
    results['column_info'] = column_info
    
    # Print summary
    mode_str = "raw values" if evaluator.raw_mode else "median of replicates"
    print(f"\nOVERALL METRICS (vs {mode_str}):")
    print(f"  Number of unique conditions: {n_conditions}")
    print(f"  Total replicates: {np.sum(n_replicates)} (mean {np.mean(n_replicates):.1f} per condition)")
    print(f"  MAE:  {results['mae_overall']:.4f}")
    print(f"  RMSE: {results['rmse_overall']:.4f}")
    print(f"  R²:   {results['r2_overall']:.4f}")
    print(f"  Relative Error: {results['relative_error_overall']:.4f}")
    
    # Best and worst proteins
    r2_sorted = sorted(protein_metrics.items(), key=lambda x: x[1]['R2'], reverse=True)
    
    print(f"\nTOP 5 BEST PREDICTED PROTEINS (by R²):")
    for protein, metrics in r2_sorted[:5]:
        display_name = PROTEIN_DISPLAY_NAMES.get(protein, protein)
        print(f"  {display_name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, "
              f"Norm.RMSE={metrics['Normalized_RMSE']:.2f}×bio.var")
    
    print(f"\nTOP 5 WORST PREDICTED PROTEINS (by R²):")
    for protein, metrics in r2_sorted[-5:]:
        display_name = PROTEIN_DISPLAY_NAMES.get(protein, protein)
        print(f"  {display_name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")
    
    return results


def evaluate_by_time_with_replicates(results: dict, aggregated_data: dict) -> dict:
    """
    Evaluate model performance stratified by time point.
    
    Parameters
    ----------
    results : dict
        Evaluation results from evaluate_model_with_replicates
    aggregated_data : dict
        Aggregated replicate data
        
    Returns
    -------
    dict
        Per-time-point evaluation results
    """
    print("\n" + "=" * 60)
    print("EVALUATION BY TIME POINT")
    print("=" * 60)
    
    predictions = results['predictions']
    gt_means = results['gt_means']
    
    # Get time points for each condition
    times = np.array([data['time'] for data in aggregated_data.values()])
    unique_times = np.sort(np.unique(times))
    
    time_results = {}
    for t in unique_times:
        mask = times == t
        gt_t = gt_means[mask].flatten()
        pred_t = predictions[mask].flatten()
        
        time_results[t] = {
            'n_conditions': mask.sum(),
            'MAE': mean_absolute_error(gt_t, pred_t),
            'RMSE': np.sqrt(mean_squared_error(gt_t, pred_t)),
            'R2': r2_score(gt_t, pred_t) if np.var(gt_t) > 1e-10 else 0.0,
        }
    
    print(f"\nMetrics by time point:")
    for t, metrics in sorted(time_results.items()):
        print(f"  t={t:3.0f} min: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, n={metrics['n_conditions']}")
    
    return time_results


def evaluate_by_stimuli(results: dict, aggregated_data: dict) -> dict:
    """
    Evaluate model performance stratified by stimulus.
    
    Parameters
    ----------
    results : dict
        Evaluation results
    aggregated_data : dict
        Aggregated replicate data
        
    Returns
    -------
    dict
        Per-stimulus evaluation results
    """
    print("\n" + "=" * 60)
    print("EVALUATION BY STIMULUS")
    print("=" * 60)
    
    predictions = results['predictions']
    gt_means = results['gt_means']
    
    # Get stimuli for each condition
    stimuli_list = [data['stimuli'] for data in aggregated_data.values()]
    unique_stimuli = list(set(stimuli_list))
    
    stimuli_results = {}
    for stim in unique_stimuli:
        mask = np.array([s == stim for s in stimuli_list])
        if not mask.any():
            continue
            
        gt_s = gt_means[mask].flatten()
        pred_s = predictions[mask].flatten()
        
        stimuli_results[stim] = {
            'n_conditions': mask.sum(),
            'MAE': mean_absolute_error(gt_s, pred_s),
            'RMSE': np.sqrt(mean_squared_error(gt_s, pred_s)),
            'R2': r2_score(gt_s, pred_s) if np.var(gt_s) > 1e-10 else 0.0,
        }
    
    # Sort by R²
    sorted_stimuli = sorted(stimuli_results.items(), key=lambda x: x[1]['R2'], reverse=True)
    
    print(f"\nMetrics by stimulus:")
    for stim, metrics in sorted_stimuli:
        print(f"  {stim:20s}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, n={metrics['n_conditions']}")
    
    return stimuli_results


# ============================================================================
# Publication-Quality Plotting (POLLU-style)
# ============================================================================
def plot_predictions_vs_ground_truth_publication(results: dict, model_name: str, output_dir: str):
    """
    Publication-quality scatter plots: predictions vs ground truth (median of replicates).
    Styled to match POLLU visualization.
    """
    predictions = results['predictions']
    gt_means = results['gt_means']
    gt_stds = results['gt_stds']
    column_info = results['column_info']
    protein_names = column_info['protein_names']
    protein_metrics = results['protein_metrics']
    
    n_proteins = len(protein_names)
    n_cols = 6
    n_rows = (n_proteins + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows))
    axes = axes.flatten()
    
    for i, protein in enumerate(protein_names):
        ax = axes[i]
        gt = gt_means[:, i]
        pred = predictions[:, i]
        gt_std = gt_stds[:, i]
        
        # Scatter plot with error bars showing biological variability
        has_variability = gt_std > 1e-10
        
        # Points without variability
        if np.any(~has_variability):
            ax.scatter(gt[~has_variability], pred[~has_variability], 
                      alpha=0.6, s=25, edgecolors='none', 
                      c=COLORS['prediction'], label='Single rep.')
        
        # Points with variability (error bars)
        if np.any(has_variability):
            ax.errorbar(gt[has_variability], pred[has_variability], 
                       xerr=gt_std[has_variability],
                       fmt='o', markersize=5, alpha=0.7,
                       color=COLORS['prediction'], ecolor='gray',
                       elinewidth=0.8, capsize=2, label='Multi rep.')
        
        # Perfect prediction line
        min_val = min(gt.min(), pred.min())
        max_val = max(gt.max(), pred.max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin], 
                [min_val - margin, max_val + margin], 
                '--', color='black', linewidth=1.5, alpha=0.7, zorder=0)
        
        # Metrics
        r2 = protein_metrics[protein]['R2']
        rmse = protein_metrics[protein]['RMSE']
        
        display_name = PROTEIN_DISPLAY_NAMES.get(protein, protein)
        ax.set_title(f'{display_name}\n$R^2$={r2:.3f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Ground Truth', fontsize=9)
        ax.set_ylabel('Prediction', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide empty subplots
    for i in range(n_proteins, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{model_name}: Predictions vs Ground Truth\n(error bars = biological variability)', 
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pred_vs_gt_publication.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/pred_vs_gt_publication.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions vs ground truth plot")


def plot_protein_performance_publication(results: dict, model_name: str, output_dir: str):
    """
    Publication-quality per-protein performance metrics.
    Shows R², RMSE, and normalized RMSE (relative to biological variability).
    """
    column_info = results['column_info']
    protein_names = column_info['protein_names']
    protein_metrics = results['protein_metrics']
    
    # Extract metrics
    r2_values = [protein_metrics[p]['R2'] for p in protein_names]
    rmse_values = [protein_metrics[p]['RMSE'] for p in protein_names]
    norm_rmse_values = [protein_metrics[p]['Normalized_RMSE'] for p in protein_names]
    
    # Sort by R²
    sorted_idx = np.argsort(r2_values)[::-1]
    
    # Create display names
    display_names = [PROTEIN_DISPLAY_NAMES.get(protein_names[i], protein_names[i]) 
                     for i in sorted_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    # R² bar chart
    ax1 = axes[0]
    colors = [COLORS['good'] if r2_values[i] > 0.5 
              else COLORS['moderate'] if r2_values[i] > 0.2 
              else COLORS['poor'] for i in sorted_idx]
    ax1.barh(range(len(protein_names)), [r2_values[i] for i in sorted_idx], 
             color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(protein_names)))
    ax1.set_yticklabels(display_names, fontsize=8)
    ax1.set_xlabel('$R^2$', fontsize=12, fontweight='bold')
    ax1.set_title('$R^2$ by Protein', fontsize=13, fontweight='bold')
    ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='$R^2$=0.5')
    ax1.axvline(x=0.2, color='gray', linestyle=':', alpha=0.7, label='$R^2$=0.2')
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_xlim(-0.3, 1.0)
    ax1.invert_yaxis()
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25, axis='x')
    
    # RMSE bar chart
    ax2 = axes[1]
    ax2.barh(range(len(protein_names)), [rmse_values[i] for i in sorted_idx], 
             color='#3498db', edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(protein_names)))
    ax2.set_yticklabels(display_names, fontsize=8)
    ax2.set_xlabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('RMSE by Protein', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.25, axis='x')
    
    # Normalized RMSE (relative to biological variability)
    ax3 = axes[2]
    norm_rmse_sorted = [norm_rmse_values[i] for i in sorted_idx]
    # Handle NaN values
    norm_rmse_plot = [v if not np.isnan(v) else 0 for v in norm_rmse_sorted]
    colors_norm = [COLORS['good'] if v < 1 else COLORS['moderate'] if v < 2 else COLORS['poor'] 
                   for v in norm_rmse_plot]
    ax3.barh(range(len(protein_names)), norm_rmse_plot, 
             color=colors_norm, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(range(len(protein_names)))
    ax3.set_yticklabels(display_names, fontsize=8)
    ax3.set_xlabel('RMSE / Bio. Variability', fontsize=12, fontweight='bold')
    ax3.set_title('Normalized RMSE\n(vs biological variability)', fontsize=13, fontweight='bold')
    ax3.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='= Bio. var.')
    ax3.invert_yaxis()
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25, axis='x')
    
    plt.suptitle(f'{model_name}: Per-Protein Performance', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/protein_performance_publication.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/protein_performance_publication.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved protein performance plot")


def plot_timecourse_with_ci_publication(results: dict, aggregated_data: dict,
                                        model_name: str, output_dir: str):
    """
    Compact time course plots for publication.
    Shows ground truth vs model predictions for top proteins × stimuli.
    """
    column_info = results['column_info']
    protein_names = column_info['protein_names']
    protein_metrics = results['protein_metrics']
    
    # Organize data by stimuli and time (no-inhibitor conditions only)
    stimuli_time_data = {}
    for i, (key, data) in enumerate(aggregated_data.items()):
        stim = data['stimuli']
        inhib = data['inhibitors']
        
        if inhib != 'None':
            continue
        
        if stim not in stimuli_time_data:
            stimuli_time_data[stim] = {'times': [], 'gt_means': [], 'predictions': []}
        
        stimuli_time_data[stim]['times'].append(data['time'])
        stimuli_time_data[stim]['gt_means'].append(data['y_mean'])
        stimuli_time_data[stim]['predictions'].append(results['predictions'][i])
    
    # Pick top 4 stimuli by number of time points (most informative)
    preferred_stimuli = ['EGF', 'Insulin', 'FGF1', 'HGF', 'IGF1', 'Serum', 'NRG1']
    key_stimuli = [s for s in preferred_stimuli if s in stimuli_time_data][:4]
    if len(key_stimuli) < 4:
        remaining = [s for s in stimuli_time_data if s not in key_stimuli]
        remaining.sort(key=lambda s: len(stimuli_time_data[s]['times']), reverse=True)
        key_stimuli.extend(remaining[:4 - len(key_stimuli)])
    
    # Pick top 4 proteins by R² (most representative)
    available_key = [p for p in KEY_PROTEINS if p in protein_names]
    if len(available_key) >= 4:
        # Sort KEY_PROTEINS by R² and pick top 4
        available_key.sort(key=lambda p: protein_metrics[p]['R2'], reverse=True)
        selected_proteins = available_key[:4]
    else:
        # Fallback: top 4 by R² from all proteins
        all_sorted = sorted(protein_names, key=lambda p: protein_metrics[p]['R2'], reverse=True)
        selected_proteins = all_sorted[:4]
    
    n_proteins = len(selected_proteins)
    n_stimuli = len(key_stimuli)
    
    if n_stimuli == 0 or n_proteins == 0:
        print("  ⚠ Not enough data for time course plot")
        return
    
    fig, axes = plt.subplots(n_proteins, n_stimuli,
                             figsize=(2.8 * n_stimuli, 2.2 * n_proteins),
                             sharex=True)
    if n_proteins == 1:
        axes = axes.reshape(1, -1)
    if n_stimuli == 1:
        axes = axes.reshape(-1, 1)
    
    for i, protein in enumerate(selected_proteins):
        protein_idx = protein_names.index(protein)
        r2 = protein_metrics[protein]['R2']
        
        for j, stim in enumerate(key_stimuli):
            ax = axes[i, j]
            
            data = stimuli_time_data.get(stim, None)
            if data is None or len(data['times']) == 0:
                ax.set_visible(False)
                continue
            
            times = np.array(data['times'])
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            
            gt_means = np.array([data['gt_means'][k][protein_idx] for k in sort_idx])
            preds = np.array([data['predictions'][k][protein_idx] for k in sort_idx])
            
            color_stim = STIMULI_COLORS.get(stim, '#666666')
            
            # Ground truth (open circles)
            ax.plot(times, gt_means,
                   marker='o', markersize=5, linestyle='none',
                   markeredgewidth=1.2, markeredgecolor=COLORS['ground_truth'],
                   markerfacecolor='white', zorder=5,
                   label='GT' if i == 0 and j == 0 else '')
            
            # Prediction (line)
            ax.plot(times, preds, color=COLORS['prediction'],
                   linewidth=1.8, label='Pred' if i == 0 and j == 0 else '')
            
            ax.set_xscale('symlog', linthresh=1)
            
            if i == 0:
                ax.set_title(stim, fontsize=10, fontweight='bold', color=color_stim)
            if j == 0:
                display_name = PROTEIN_DISPLAY_NAMES.get(protein, protein)
                ax.set_ylabel(f'{display_name}\n($R^2$={r2:.2f})', fontsize=8, fontweight='bold')
            if i == n_proteins - 1:
                ax.set_xlabel('Time (min)', fontsize=9)
            
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.4)
            ax.tick_params(labelsize=7)
            
            # Remove top/right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Single shared legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2,
                  fontsize=8, framealpha=0.9,
                  bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'{output_dir}/timecourse_with_ci.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/timecourse_with_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved time course plot (4×4 compact)")


def plot_error_heatmap_publication(results: dict, aggregated_data: dict,
                                   model_name: str, output_dir: str):
    """
    Publication-quality error heatmap by stimuli and protein.
    """
    predictions = results['predictions']
    gt_means = results['gt_means']
    column_info = results['column_info']
    protein_names = column_info['protein_names']
    
    # Get stimuli for each condition
    stimuli_list = [data['stimuli'] for data in aggregated_data.values()]
    unique_stimuli = sorted(set(stimuli_list))
    
    # Calculate RMSE for each stimuli-protein combination
    error_matrix = np.zeros((len(unique_stimuli), len(protein_names)))
    r2_matrix = np.zeros((len(unique_stimuli), len(protein_names)))
    
    for i, stim in enumerate(unique_stimuli):
        mask = np.array([s == stim for s in stimuli_list])
        for j, protein in enumerate(protein_names):
            gt = gt_means[mask, j]
            pred = predictions[mask, j]
            if len(gt) > 0:
                error_matrix[i, j] = np.sqrt(mean_squared_error(gt, pred))
                r2_matrix[i, j] = r2_score(gt, pred) if np.var(gt) > 1e-10 else 0.0
    
    # Display names
    display_names = [PROTEIN_DISPLAY_NAMES.get(p, p) for p in protein_names]
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # RMSE heatmap
    ax1 = axes[0]
    im1 = sns.heatmap(error_matrix, annot=False, cmap='YlOrRd', 
                xticklabels=display_names, yticklabels=unique_stimuli,
                ax=ax1, cbar_kws={'label': 'RMSE'})
    ax1.set_xlabel('Protein', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Stimulus', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE by Stimulus × Protein', fontsize=13, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), fontsize=9)
    
    # R² heatmap
    ax2 = axes[1]
    im2 = sns.heatmap(r2_matrix, annot=False, cmap='RdYlGn', center=0.5,
                      vmin=0, vmax=1,
                      xticklabels=display_names, yticklabels=unique_stimuli,
                      ax=ax2, cbar_kws={'label': '$R^2$'})
    ax2.set_xlabel('Protein', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Stimulus', fontsize=12, fontweight='bold')
    ax2.set_title('$R^2$ by Stimulus × Protein', fontsize=13, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), fontsize=9)
    
    plt.suptitle(f'{model_name}: Performance Heatmaps', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_heatmap_publication.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/error_heatmap_publication.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved error heatmap")


def plot_error_quantiles_over_time(results: dict, aggregated_data: dict,
                                   time_results: dict, model_name: str, output_dir: str):
    """
    POLLU-style error quantile plot over time.
    Shows median error with 10-90% quantile bands.
    """
    predictions = results['predictions']
    gt_means = results['gt_means']
    column_info = results['column_info']
    
    # Get times for each condition
    times = np.array([data['time'] for data in aggregated_data.values()])
    unique_times = np.sort(np.unique(times))
    
    # Compute error quantiles at each time point
    qs = [0.1, 0.5, 0.9]
    rmse_by_time = []
    mae_by_time = []
    
    for t in unique_times:
        mask = times == t
        if not mask.any():
            continue
        
        gt_t = gt_means[mask]
        pred_t = predictions[mask]
        
        # Compute per-condition errors
        condition_rmse = np.sqrt(np.mean((gt_t - pred_t)**2, axis=1))
        condition_mae = np.mean(np.abs(gt_t - pred_t), axis=1)
        
        rmse_by_time.append(condition_rmse)
        mae_by_time.append(condition_mae)
    
    # Compute quantiles
    rmse_q = np.array([np.quantile(rmse, qs) for rmse in rmse_by_time]).T  # (3, T)
    mae_q = np.array([np.quantile(mae, qs) for mae in mae_by_time]).T
    
    eps = 1e-12
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # RMSE with quantile band
    ax.semilogy(unique_times, rmse_q[1] + eps, linewidth=2.5, 
               color=COLORS['prediction'], label=r'RMSE median')
    ax.fill_between(unique_times, rmse_q[0] + eps, rmse_q[2] + eps, 
                   alpha=0.25, color=COLORS['prediction'], label=r'RMSE 10–90%')
    
    # MAE with quantile band  
    ax.semilogy(unique_times, mae_q[1] + eps, linewidth=2.0, linestyle='--',
               color='#3498db', label=r'MAE median')
    ax.fill_between(unique_times, mae_q[0] + eps, mae_q[2] + eps, 
                   alpha=0.15, color='#3498db', label=r'MAE 10–90%')
    
    ax.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Error Distribution Over Time', fontsize=13, fontweight='bold')
    ax.set_xscale('symlog', linthresh=1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_quantiles_time.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/error_quantiles_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved error quantile plot")


def plot_replicate_analysis(results: dict, aggregated_data: dict,
                            model_name: str, output_dir: str):
    """
    Analyze how model error relates to number of replicates and biological variability.
    """
    predictions = results['predictions']
    gt_means = results['gt_means']
    gt_stds = results['gt_stds']
    n_replicates = results['n_replicates']
    
    # Compute per-condition errors
    condition_rmse = np.sqrt(np.mean((predictions - gt_means)**2, axis=1))
    condition_mae = np.mean(np.abs(predictions - gt_means), axis=1)
    mean_bio_std = np.mean(gt_stds, axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Error vs number of replicates
    ax1 = axes[0]
    unique_n_reps = np.unique(n_replicates)
    for n_rep in unique_n_reps:
        mask = n_replicates == n_rep
        ax1.scatter(np.full(mask.sum(), n_rep) + np.random.normal(0, 0.1, mask.sum()),
                   condition_rmse[mask], alpha=0.6, s=30, label=f'n={n_rep}')
    
    # Box plot overlay
    bp_data = [condition_rmse[n_replicates == n] for n in unique_n_reps]
    bp = ax1.boxplot(bp_data, positions=unique_n_reps, widths=0.3)
    
    ax1.set_xlabel('Number of Replicates', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('Error vs. Replicate Count', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.25)
    
    # Error vs biological variability
    ax2 = axes[1]
    valid_mask = mean_bio_std > 1e-10
    if np.any(valid_mask):
        ax2.scatter(mean_bio_std[valid_mask], condition_rmse[valid_mask], 
                   alpha=0.6, s=30, color=COLORS['prediction'])
        
        # Add diagonal reference line (error = bio variability)
        max_val = max(mean_bio_std[valid_mask].max(), condition_rmse[valid_mask].max())
        ax2.plot([0, max_val], [0, max_val], '--', color='black', alpha=0.7, 
                label='Error = Bio. var.')
        ax2.legend(fontsize=10)
    
    ax2.set_xlabel('Mean Biological Variability (std)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Model RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('Error vs. Biological Variability', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.25)
    
    # Distribution of normalized error
    ax3 = axes[2]
    valid_mask = mean_bio_std > 1e-10
    if np.any(valid_mask):
        normalized_error = condition_rmse[valid_mask] / mean_bio_std[valid_mask]
        ax3.hist(normalized_error, bins=30, alpha=0.7, color=COLORS['prediction'],
                edgecolor='black', linewidth=0.5)
        ax3.axvline(x=1.0, color='black', linestyle='--', linewidth=2, 
                   label='Error = Bio. var.')
        ax3.axvline(x=np.median(normalized_error), color='red', linestyle='-', 
                   linewidth=2, label=f'Median = {np.median(normalized_error):.2f}')
        ax3.legend(fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No replicate variability\n(raw mode: all n=1)', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    
    ax3.set_xlabel('RMSE / Biological Variability', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Normalized Error', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.25)
    
    plt.suptitle(f'{model_name}: Replicate & Variability Analysis', 
                fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/replicate_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/replicate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved replicate analysis plot")


def plot_time_analysis_publication(time_results: dict, model_name: str, output_dir: str):
    """Plot performance metrics over time (POLLU-style)."""
    times = sorted(time_results.keys())
    r2_values = [time_results[t]['R2'] for t in times]
    rmse_values = [time_results[t]['RMSE'] for t in times]
    n_conditions = [time_results[t]['n_conditions'] for t in times]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # R² over time
    ax1 = axes[0]
    ax1.plot(times, r2_values, 'o-', linewidth=2.5, markersize=10, 
            color=COLORS['good'], markeredgecolor='black', markeredgewidth=1)
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='$R^2$=0.5')
    ax1.axhline(y=0.2, color='gray', linestyle=':', alpha=0.7, label='$R^2$=0.2')
    ax1.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('$R^2$', fontsize=12, fontweight='bold')
    ax1.set_title('$R^2$ by Time Point', fontsize=13, fontweight='bold')
    ax1.set_xscale('symlog', linthresh=1)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(-0.1, 1.05)
    
    # RMSE over time
    ax2 = axes[1]
    ax2.plot(times, rmse_values, 's-', linewidth=2.5, markersize=10, 
            color=COLORS['poor'], markeredgecolor='black', markeredgewidth=1)
    ax2.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('RMSE by Time Point', fontsize=13, fontweight='bold')
    ax2.set_xscale('symlog', linthresh=1)
    ax2.grid(True, alpha=0.25)
    
    # Sample count
    ax3 = axes[2]
    ax3.bar(range(len(times)), n_conditions, color='#3498db', 
           edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(len(times)))
    ax3.set_xticklabels([f'{t:.0f}' if t >= 1 else f'{t}' for t in times], rotation=45)
    ax3.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Conditions', fontsize=12, fontweight='bold')
    ax3.set_title('Conditions per Time Point', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.25, axis='y')
    
    plt.suptitle(f'{model_name}: Performance by Time', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_analysis_publication.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/time_analysis_publication.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved time analysis plot")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Test teacher model on MCF7 signaling data (replicate-aware)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to saved model checkpoint')
    parser.add_argument('--data_path', type=str, 
                       default='experimental/MIDAS/MD_MCF7_main.csv',
                       help='Path to MIDAS data file')
    parser.add_argument('--output_dir', type=str, default='results/teacher_evaluation',
                       help='Output directory for plots')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
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
    
    # Create output directory
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    evaluator = TeacherModelEvaluator(args.model_path, device)
    
    # Detect raw mode from checkpoint
    raw_mode = evaluator.raw_mode
    
    # Load data
    print("\n" + "=" * 60)
    print("Loading MCF7 Data")
    print("=" * 60)
    
    X, y, column_info, df = load_midas_data(args.data_path)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Input features: {X.shape[1]}")
    print(f"  Output proteins: {y.shape[1]}")
    
    if raw_mode:
        # ====================================================================
        # Raw mode: no preprocessing, no aggregation
        # ====================================================================
        print("\n" + "=" * 60)
        print("RAW MODE: Skipping all preprocessing")
        print("=" * 60)
        print("  ✗ No baseline alignment")
        print("  ✗ No replicate aggregation")
        print("  ✗ No time log-transform")
        print("  ✗ No input scaling")
        print("  ✗ No output scaling")
        
        # Each sample is its own condition (no aggregation)
        aggregated_data = prepare_raw_data(X, y, column_info, df)
        
        n_unique = len(aggregated_data)
        print(f"  Individual samples (no aggregation): {n_unique}")
        
    else:
        # ====================================================================
        # Standard mode: align baseline, aggregate replicates
        # ====================================================================
        # Align t=0 baseline (must match training preprocessing)
        print("\n" + "=" * 60)
        print("Aligning t=0 Baseline (per inhibitor condition)")
        print("=" * 60)
        
        y = align_baseline_t0(X, y, column_info)
        
        # Aggregate replicates
        print("\n" + "=" * 60)
        print("Aggregating Biological Replicates")
        print("=" * 60)
        
        aggregated_data = aggregate_replicates(X, y, column_info, df)
        
        n_unique = len(aggregated_data)
        n_reps = [data['n_replicates'] for data in aggregated_data.values()]
        print(f"  Unique conditions: {n_unique}")
        print(f"  Replicates per condition: min={min(n_reps)}, max={max(n_reps)}, mean={np.mean(n_reps):.1f}")
        print(f"  Conditions with >1 replicate: {sum(1 for n in n_reps if n > 1)} ({sum(1 for n in n_reps if n > 1)/n_unique*100:.1f}%)")
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    results = evaluate_model_with_replicates(evaluator, aggregated_data, column_info)
    time_results = evaluate_by_time_with_replicates(results, aggregated_data)
    stimuli_results = evaluate_by_stimuli(results, aggregated_data)
    
    # ========================================================================
    # Plotting (Publication Quality, POLLU-style)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Generating Publication-Quality Plots")
    print("=" * 60)
    
    plot_predictions_vs_ground_truth_publication(results, model_name, output_dir)
    plot_protein_performance_publication(results, model_name, output_dir)
    plot_timecourse_with_ci_publication(results, aggregated_data, model_name, output_dir)
    plot_error_heatmap_publication(results, aggregated_data, model_name, output_dir)
    plot_error_quantiles_over_time(results, aggregated_data, time_results, model_name, output_dir)
    plot_replicate_analysis(results, aggregated_data, model_name, output_dir)
    plot_time_analysis_publication(time_results, model_name, output_dir)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    if raw_mode:
        print("EVALUATION SUMMARY (RAW MODE)")
    else:
        print("EVALUATION SUMMARY (REPLICATE-AWARE)")
    print("=" * 60)
    print(f"\nModel: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Mode: {'RAW (no preprocessing)' if raw_mode else 'Standard (baseline-aligned, aggregated)'}")
    
    n_reps = [data['n_replicates'] for data in aggregated_data.values()]
    print(f"\nDATA SUMMARY:")
    print(f"  Total samples: {len(X)}")
    print(f"  Unique conditions: {n_unique}")
    print(f"  Mean replicates per condition: {np.mean(n_reps):.1f}")
    
    mode_str = "raw values" if raw_mode else "median of replicates"
    print(f"\nOVERALL PERFORMANCE (vs {mode_str}):")
    print(f"  MAE:  {results['mae_overall']:.4f}")
    print(f"  RMSE: {results['rmse_overall']:.4f}")
    print(f"  R²:   {results['r2_overall']:.4f}")
    print(f"  Relative Error: {results['relative_error_overall']:.4f}")
    
    # Count proteins with good/bad R²
    r2_values = [results['protein_metrics'][p]['R2'] for p in column_info['protein_names']]
    n_good = sum(1 for r2 in r2_values if r2 > 0.5)
    n_moderate = sum(1 for r2 in r2_values if 0.2 < r2 <= 0.5)
    n_poor = sum(1 for r2 in r2_values if r2 <= 0.2)
    
    print(f"\nPROTEIN PREDICTION QUALITY:")
    print(f"  Good (R² > 0.5):     {n_good}/{len(r2_values)} proteins")
    print(f"  Moderate (0.2-0.5):  {n_moderate}/{len(r2_values)} proteins")
    print(f"  Poor (R² < 0.2):     {n_poor}/{len(r2_values)} proteins")
    
    # Biological variability comparison (only meaningful when replicates exist)
    gt_stds = results['gt_stds']
    valid_std_mask = np.mean(gt_stds, axis=1) > 1e-10
    if valid_std_mask.any():
        mean_bio_var = np.mean(gt_stds[valid_std_mask])
        print(f"\nBIOLOGICAL VARIABILITY COMPARISON:")
        print(f"  Mean biological std: {mean_bio_var:.4f}")
        print(f"  Model RMSE / Bio. var: {results['rmse_overall']/mean_bio_var:.2f}×")
    elif raw_mode:
        print(f"\nBIOLOGICAL VARIABILITY COMPARISON:")
        print(f"  ⚠ Not available in raw mode (no replicate aggregation)")
    
    print(f"\nPlots saved to: {output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()