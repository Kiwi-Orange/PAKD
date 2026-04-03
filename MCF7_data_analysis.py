import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings
from itertools import product

warnings.filterwarnings('ignore')

# Publication-ready matplotlib settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "mathtext.fontset": "cm",  # Use Computer Modern for math
})

# Define key proteins and stimuli for focused analysis
KEY_PROTEINS = ['EGFR_pY1068', 'AKT_pS473', 'MAPK_pT202_Y204', 
                'mTOR_pS2448', 'S6_pS235_S236', 'p70S6K_pT389']
KEY_STIMULI = ['EGF', 'Insulin', 'FGF1', 'HGF', 'IGF1']

# Protein display names for cleaner figures
PROTEIN_DISPLAY_NAMES = {
    'EGFR_pY1068': 'p-EGFR\n(Y1068)',
    'EGFR_pY1173': 'p-EGFR\n(Y1173)',
    'AKT_pS473': 'p-AKT\n(S473)',
    'AKT_pT308': 'p-AKT\n(T308)',
    'MAPK_pT202_Y204': 'p-ERK1/2',
    'MEK1_pS217_S221': 'p-MEK1',
    'mTOR_pS2448': 'p-mTOR',
    'S6_pS235_S236': 'p-S6\n(S235/236)',
    'S6_pS240_S244': 'p-S6\n(S240/244)',
    'p70S6K_pT389': 'p-p70S6K',
    '4EBP1_pS65': 'p-4EBP1',
    'STAT3_pY705': 'p-STAT3',
}


@dataclass
class MIDASConfig:
    """Configuration for MIDAS data analysis."""
    data_path: str = 'experimental/MIDAS/MD_MCF7_main.csv'
    output_dir: str = 'plots/MCF7_analysis'
    
    # Column prefixes in MIDAS format
    treatment_prefix: str = 'TR:'
    data_average_prefix: str = 'DA:'
    data_value_prefix: str = 'DV:'


def load_midas_data(config: MIDASConfig) -> pd.DataFrame:
    """
    Load MIDAS format data.
    
    Parameters
    ----------
    config : MIDASConfig
        Configuration object
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    df = pd.read_csv(config.data_path)
    print(f"Loaded data shape: {df.shape}")
    return df


def parse_midas_columns(df: pd.DataFrame, config: MIDASConfig) -> Dict:
    """
    Parse MIDAS column names into treatments, time, and measurements.
    
    Parameters
    ----------
    df : pd.DataFrame
        MIDAS data
    config : MIDASConfig
        Configuration object
        
    Returns
    -------
    dict
        Dictionary with parsed column information
    """
    columns = df.columns.tolist()
    
    # Separate column types
    treatment_cols = [c for c in columns if c.startswith(config.treatment_prefix)]
    da_cols = [c for c in columns if c.startswith(config.data_average_prefix)]
    dv_cols = [c for c in columns if c.startswith(config.data_value_prefix)]
    
    # Parse treatment columns
    stimuli_cols = [c for c in treatment_cols if ':Stimuli' in c]
    inhibitor_cols = [c for c in treatment_cols if ':Inhibitors' in c]
    cell_line_cols = [c for c in treatment_cols if ':CellLine' in c]
    
    # Extract protein names from DV columns
    proteins = [c.replace(config.data_value_prefix, '') for c in dv_cols]
    
    # Extract stimuli names
    stimuli = [c.split(':')[1] for c in stimuli_cols]
    
    # Extract inhibitor names
    inhibitors = [c.split(':')[1] for c in inhibitor_cols]
    
    parsed = {
        'treatment_cols': treatment_cols,
        'stimuli_cols': stimuli_cols,
        'inhibitor_cols': inhibitor_cols,
        'cell_line_cols': cell_line_cols,
        'da_cols': da_cols,
        'dv_cols': dv_cols,
        'proteins': proteins,
        'stimuli': stimuli,
        'inhibitors': inhibitors
    }
    
    print("\n=== MIDAS Data Structure ===")
    print(f"Treatment columns: {len(treatment_cols)}")
    print(f"  - Stimuli: {stimuli}")
    print(f"  - Inhibitors: {inhibitors}")
    print(f"Data value columns (proteins): {len(dv_cols)}")
    print(f"Proteins measured: {proteins[:5]}... (showing first 5)")
    
    return parsed


def get_time_points(df: pd.DataFrame, config: MIDASConfig) -> np.ndarray:
    """
    Extract time points from DA:ALL column.
    
    Parameters
    ----------
    df : pd.DataFrame
        MIDAS data
    config : MIDASConfig
        Configuration object
        
    Returns
    -------
    np.ndarray
        Unique time points
    """
    if 'DA:ALL' in df.columns:
        time_points = df['DA:ALL'].unique()
        time_points = np.sort(time_points)
        print(f"\nTime points: {time_points}")
        return time_points
    return np.array([])


def get_treatment_conditions(df: pd.DataFrame, parsed: Dict) -> pd.DataFrame:
    """
    Create a summary of treatment conditions.
    
    Parameters
    ----------
    df : pd.DataFrame
        MIDAS data
    parsed : dict
        Parsed column information
        
    Returns
    -------
    pd.DataFrame
        Treatment condition summary
    """
    conditions = []
    
    for idx, row in df.iterrows():
        # Get active stimuli
        active_stimuli = []
        for col in parsed['stimuli_cols']:
            if row[col] == 1:
                stimuli_name = col.split(':')[1]
                active_stimuli.append(stimuli_name)
        
        # Get active inhibitors
        active_inhibitors = []
        for col in parsed['inhibitor_cols']:
            if row[col] == 1:
                inhibitor_name = col.split(':')[1]
                active_inhibitors.append(inhibitor_name)
        
        time_point = row['DA:ALL'] if 'DA:ALL' in df.columns else 0
        
        conditions.append({
            'index': idx,
            'stimuli': '+'.join(active_stimuli) if active_stimuli else 'None',
            'inhibitors': '+'.join(active_inhibitors) if active_inhibitors else 'None',
            'time': time_point
        })
    
    return pd.DataFrame(conditions)


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
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
    Tuple[float, float]
        Lower and upper bounds of CI
    """
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)
    
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return (mean - h, mean + h)


def plot_main_figure_timecourse(df: pd.DataFrame, parsed: Dict, conditions: pd.DataFrame, config: MIDASConfig):
    """
    Create publication-quality time-course figure for key proteins and stimuli.
    
    This is designed as a main figure candidate showing:
    - 4-6 key signaling proteins
    - 3-5 key stimuli (without inhibitor background)
    - Mean ± 95% CI
    
    Parameters
    ----------
    df : pd.DataFrame
        MIDAS data
    parsed : dict
        Parsed column information
    conditions : pd.DataFrame
        Treatment conditions
    config : MIDASConfig
        Configuration object
    """
    # Select key proteins that exist in the dataset
    available_key_proteins = [p for p in KEY_PROTEINS if f'DV:{p}' in df.columns]
    
    if len(available_key_proteins) < 4:
        # Fallback to first 6 proteins
        available_key_proteins = parsed['proteins'][:6]
    
    # Select key stimuli (without inhibitor background)
    available_key_stimuli = [s for s in KEY_STIMULI if s in conditions['stimuli'].values]
    
    if len(available_key_stimuli) < 3:
        # Fallback to top stimuli by sample count
        stim_counts = conditions[conditions['inhibitors'] == 'None']['stimuli'].value_counts()
        available_key_stimuli = [s for s in stim_counts.index if s not in ['None', 'PBS']][:5]
    
    print(f"  Key proteins: {available_key_proteins}")
    print(f"  Key stimuli: {available_key_stimuli}")
    
    # Filter to no-inhibitor conditions only
    no_inhib_mask = conditions['inhibitors'] == 'None'
    
    # Get time points
    time_points = np.sort(conditions['time'].unique())
    
    # Define color palette for stimuli
    stimuli_colors = {
        'EGF': '#E64B35',      # Red
        'Insulin': '#4DBBD5',   # Cyan
        'FGF1': '#00A087',      # Teal
        'HGF': '#3C5488',       # Blue
        'IGF1': '#F39B7F',      # Orange
        'Serum': '#8491B4',     # Gray-blue
        'NRG1': '#91D1C2',      # Light teal
    }
    
    # Create figure
    n_proteins = len(available_key_proteins)
    n_cols = 3
    n_rows = (n_proteins + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows))
    axes = axes.flatten() if n_proteins > 1 else [axes]
    
    for idx, protein in enumerate(available_key_proteins):
        ax = axes[idx]
        col_name = f'DV:{protein}'
        
        for stim in available_key_stimuli:
            # Get data for this stimulus without inhibitor
            mask = (conditions['stimuli'] == stim) & no_inhib_mask
            
            if not mask.any():
                continue
            
            # Calculate mean and CI for each time point
            means = []
            ci_lowers = []
            ci_uppers = []
            valid_times = []
            
            for t in time_points:
                time_mask = mask & (conditions['time'] == t)
                if time_mask.any():
                    data = df.loc[time_mask, col_name].values
                    if len(data) > 0:
                        mean_val = np.mean(data)
                        ci_low, ci_high = compute_confidence_interval(data)
                        
                        means.append(mean_val)
                        ci_lowers.append(ci_low)
                        ci_uppers.append(ci_high)
                        valid_times.append(t)
            
            if len(valid_times) > 0:
                valid_times = np.array(valid_times)
                means = np.array(means)
                ci_lowers = np.array(ci_lowers)
                ci_uppers = np.array(ci_uppers)
                
                color = stimuli_colors.get(stim, '#666666')
                
                # Plot line with CI shading
                ax.fill_between(valid_times, ci_lowers, ci_uppers, 
                               alpha=0.2, color=color, linewidth=0)
                ax.plot(valid_times, means, 'o-', color=color, 
                       label=stim, markersize=6, linewidth=2)
        
        # Format subplot
        display_name = PROTEIN_DISPLAY_NAMES.get(protein, protein)
        ax.set_title(display_name.replace('\n', ' '), fontweight='bold', fontsize=12)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Phosphorylation (a.u.)')
        ax.set_xscale('symlog', linthresh=1)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    # Hide empty subplots
    for idx in range(len(available_key_proteins), len(axes)):
        axes[idx].set_visible(False)
    
    # Add figure title
    fig.suptitle('Temporal Dynamics of Key Signaling Proteins\n(Mean ± 95% CI, no inhibitor)', 
                 fontweight='bold', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{config.output_dir}/main_figure_timecourse.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.output_dir}/main_figure_timecourse.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved main figure time-course plot")
    
    return available_key_proteins, available_key_stimuli


def plot_pca_trajectory_with_aggregation(df: pd.DataFrame, parsed: Dict, conditions: pd.DataFrame, 
                                         key_proteins: List[str], key_stimuli: List[str], 
                                         config: MIDASConfig):
    """
    Create PCA trajectory plot with condition × time aggregated points.
    
    This complements the time-course analysis by showing how cell states
    evolve in a global state space.
    
    Parameters
    ----------
    df : pd.DataFrame
        MIDAS data
    parsed : dict
        Parsed column information
    conditions : pd.DataFrame
        Treatment conditions
    key_proteins : List[str]
        Key proteins from time-course analysis
    key_stimuli : List[str]
        Key stimuli from time-course analysis
    config : MIDASConfig
        Configuration object
    """
    # Use all proteins for PCA but focus visualization on key stimuli
    proteins_data = df[parsed['dv_cols']].values
    
    # Standardize data
    scaler = StandardScaler()
    proteins_scaled = scaler.fit_transform(proteins_data)
    
    # Perform PCA
    pca = PCA(n_components=min(10, proteins_scaled.shape[1]))
    pca_result = pca.fit_transform(proteins_scaled)
    variance_ratio = pca.explained_variance_ratio_
    
    # Filter to no-inhibitor conditions
    no_inhib_mask = conditions['inhibitors'] == 'None'
    
    # Get time points
    time_points = np.sort(conditions['time'].unique())
    
    # Define colors (same as time-course)
    stimuli_colors = {
        'EGF': '#E64B35',
        'Insulin': '#4DBBD5',
        'FGF1': '#00A087',
        'HGF': '#3C5488',
        'IGF1': '#F39B7F',
        'Serum': '#8491B4',
        'NRG1': '#91D1C2',
    }
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Panel A: PCA trajectories with aggregated points =====
    ax1 = axes[0]
    
    # Aggregate data by condition × time
    aggregated_data = {}
    
    for stim in key_stimuli:
        mask = (conditions['stimuli'] == stim) & no_inhib_mask
        
        if not mask.any():
            continue
        
        aggregated_data[stim] = {'times': [], 'pc1_mean': [], 'pc2_mean': [], 
                                  'pc1_std': [], 'pc2_std': [], 'n': []}
        
        for t in time_points:
            time_mask = mask & (conditions['time'] == t)
            if time_mask.any():
                indices = conditions.loc[time_mask].index.tolist()
                pc1_vals = pca_result[indices, 0]
                pc2_vals = pca_result[indices, 1]
                
                aggregated_data[stim]['times'].append(t)
                aggregated_data[stim]['pc1_mean'].append(np.mean(pc1_vals))
                aggregated_data[stim]['pc2_mean'].append(np.mean(pc2_vals))
                aggregated_data[stim]['pc1_std'].append(np.std(pc1_vals))
                aggregated_data[stim]['pc2_std'].append(np.std(pc2_vals))
                aggregated_data[stim]['n'].append(len(indices))
    
    # Plot trajectories
    for stim, data in aggregated_data.items():
        if len(data['times']) < 2:
            continue
        
        times = np.array(data['times'])
        pc1 = np.array(data['pc1_mean'])
        pc2 = np.array(data['pc2_mean'])
        pc1_std = np.array(data['pc1_std'])
        pc2_std = np.array(data['pc2_std'])
        
        color = stimuli_colors.get(stim, '#666666')
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        pc1 = pc1[sort_idx]
        pc2 = pc2[sort_idx]
        pc1_std = pc1_std[sort_idx]
        pc2_std = pc2_std[sort_idx]
        
        # Plot trajectory line
        ax1.plot(pc1, pc2, '-', color=color, linewidth=2.5, alpha=0.8, label=stim)
        
        # Plot error ellipses (simplified as crosses)
        for i in range(len(times)):
            # Horizontal error bar
            ax1.plot([pc1[i] - pc1_std[i], pc1[i] + pc1_std[i]], 
                    [pc2[i], pc2[i]], '-', color=color, alpha=0.4, linewidth=1)
            # Vertical error bar
            ax1.plot([pc1[i], pc1[i]], 
                    [pc2[i] - pc2_std[i], pc2[i] + pc2_std[i]], '-', color=color, alpha=0.4, linewidth=1)
        
        # Plot aggregated points with size proportional to time
        sizes = 50 + (times - times.min()) / (times.max() - times.min() + 1e-6) * 150
        ax1.scatter(pc1, pc2, c=[color]*len(times), s=sizes, alpha=0.9, 
                   edgecolors='white', linewidth=1.5, zorder=5)
        
        # Mark start (t=0) with a star
        ax1.scatter(pc1[0], pc2[0], marker='*', s=300, c=color,
                   edgecolors='black', linewidth=1.5, zorder=10)
        
        # Add time annotations
        for i, t in enumerate(times):
            ax1.annotate(f'{int(t)}', (pc1[i], pc2[i]), 
                        textcoords="offset points", xytext=(8, 5), 
                        fontsize=8, color=color, fontweight='bold')
        
        # Add arrow to show direction
        if len(pc1) > 1:
            dx = pc1[-1] - pc1[-2]
            dy = pc2[-1] - pc2[-2]
            ax1.annotate('', xy=(pc1[-1], pc2[-1]), 
                        xytext=(pc1[-1] - dx*0.3, pc2[-1] - dy*0.3),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    ax1.set_xlabel(f'PC1 ({variance_ratio[0]*100:.1f}% variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({variance_ratio[1]*100:.1f}% variance)', fontsize=12)
    ax1.set_title('A. State Space Trajectories\n(Aggregated by Condition × Time)', 
                  fontweight='bold', fontsize=12)
    ax1.legend(loc='best', framealpha=0.9, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add annotation explaining the plot
    ax1.text(0.02, 0.98, r'$\bigstar$ = $t_0$' + '\nSize $\\propto$ time\nBars = ±1 SD', 
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== Panel B: PC loadings for key proteins =====
    ax2 = axes[1]
    
    # Get loadings
    loadings = pca.components_[:2, :]  # First 2 PCs
    protein_names = parsed['proteins']
    
    # Create loading dataframe
    loading_df = pd.DataFrame({
        'Protein': protein_names,
        'PC1': loadings[0, :],
        'PC2': loadings[1, :]
    })
    
    # Highlight key proteins
    loading_df['is_key'] = loading_df['Protein'].isin(key_proteins)
    
    # Plot all proteins as arrows
    for _, row in loading_df.iterrows():
        color = '#E64B35' if row['is_key'] else '#CCCCCC'
        alpha = 1.0 if row['is_key'] else 0.3
        linewidth = 2 if row['is_key'] else 0.5
        
        ax2.arrow(0, 0, row['PC1']*3, row['PC2']*3, 
                 head_width=0.05, head_length=0.02, 
                 fc=color, ec=color, alpha=alpha, linewidth=linewidth)
        
        if row['is_key']:
            display_name = PROTEIN_DISPLAY_NAMES.get(row['Protein'], row['Protein'])
            ax2.annotate(display_name.replace('\n', ' '), 
                        (row['PC1']*3.2, row['PC2']*3.2),
                        fontsize=9, fontweight='bold', color='#E64B35',
                        ha='center', va='center')
    
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax2.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel('PC1 Loading', fontsize=12)
    ax2.set_ylabel('PC2 Loading', fontsize=12)
    ax2.set_title('B. PCA Loadings\n(Key proteins highlighted)', fontweight='bold', fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{config.output_dir}/pca_trajectory_aggregated.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.output_dir}/pca_trajectory_aggregated.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA trajectory plot with aggregation")
    
    return pca, pca_result


def plot_inhibitor_fold_change_stratified(df: pd.DataFrame, parsed: Dict, conditions: pd.DataFrame, 
                                          key_proteins: List[str], config: MIDASConfig):
    """
    Plot inhibitor effects as fold change stratified by stimulus background.
    
    This avoids diluting effects by global averaging and shows stimulus-specific
    inhibitor efficacy.
    
    Parameters
    ----------
    df : pd.DataFrame
        MIDAS data
    parsed : dict
        Parsed column information
    conditions : pd.DataFrame
        Treatment conditions
    key_proteins : List[str]
        Key proteins to analyze
    config : MIDASConfig
        Configuration object
    """
    # Get available key proteins
    available_proteins = [p for p in key_proteins if f'DV:{p}' in df.columns]
    
    if not available_proteins:
        available_proteins = parsed['proteins'][:6]
    
    # Get inhibitors (exclude 'None')
    unique_inhibitors = [i for i in conditions['inhibitors'].unique() if i != 'None']
    
    if not unique_inhibitors:
        print("No inhibitor conditions found. Skipping inhibitor analysis.")
        return
    
    # Get key stimuli
    key_stimuli_available = [s for s in KEY_STIMULI if s in conditions['stimuli'].values]
    if not key_stimuli_available:
        key_stimuli_available = [s for s in conditions['stimuli'].unique() if s not in ['None', 'PBS']][:5]
    
    # Calculate fold change for each inhibitor
    fold_change_results = []
    
    for inhib in unique_inhibitors:
        for stim in key_stimuli_available:
            # Baseline: stimulus without inhibitor
            baseline_mask = (conditions['stimuli'] == stim) & (conditions['inhibitors'] == 'None')
            # With inhibitor
            inhib_mask = (conditions['stimuli'] == stim) & (conditions['inhibitors'] == inhib)
            
            if not baseline_mask.any() or not inhib_mask.any():
                continue
            
            for protein in available_proteins:
                col_name = f'DV:{protein}'
                
                baseline_vals = df.loc[baseline_mask, col_name].values
                inhib_vals = df.loc[inhib_mask, col_name].values
                
                baseline_mean = np.mean(baseline_vals)
                inhib_mean = np.mean(inhib_vals)
                
                if baseline_mean > 0.01:  # Avoid division by very small numbers
                    fold_change = inhib_mean / baseline_mean
                    
                    # Perform t-test
                    if len(baseline_vals) > 1 and len(inhib_vals) > 1:
                        _, pval = stats.ttest_ind(baseline_vals, inhib_vals)
                    else:
                        pval = np.nan
                    
                    fold_change_results.append({
                        'Inhibitor': inhib,
                        'Stimulus': stim,
                        'Protein': protein,
                        'Fold_Change': fold_change,
                        'Log2_FC': np.log2(fold_change) if fold_change > 0 else np.nan,
                        'P_value': pval,
                        'Significant': pval < 0.05 if not np.isnan(pval) else False,
                        'Baseline_Mean': baseline_mean,
                        'Inhibitor_Mean': inhib_mean
                    })
    
    if not fold_change_results:
        print("No fold change results computed. Skipping inhibitor analysis.")
        return
    
    fc_df = pd.DataFrame(fold_change_results)
    
    # Create multi-panel figure
    n_inhibitors = len(unique_inhibitors)
    fig, axes = plt.subplots(1, n_inhibitors, figsize=(7*n_inhibitors, 6))
    
    if n_inhibitors == 1:
        axes = [axes]
    
    for ax, inhib in zip(axes, unique_inhibitors):
        # Filter to this inhibitor
        inhib_data = fc_df[fc_df['Inhibitor'] == inhib]
        
        if inhib_data.empty:
            ax.set_visible(False)
            continue
        
        # Create pivot table
        pivot = inhib_data.pivot(index='Stimulus', columns='Protein', values='Log2_FC')
        
        # Create significance annotation
        sig_pivot = inhib_data.pivot(index='Stimulus', columns='Protein', values='Significant')
        
        # Reorder columns to match key_proteins order
        col_order = [p for p in available_proteins if p in pivot.columns]
        pivot = pivot[col_order]
        sig_pivot = sig_pivot[col_order]
        
        # Rename columns for display
        display_cols = [PROTEIN_DISPLAY_NAMES.get(p, p).replace('\n', ' ') for p in col_order]
        pivot.columns = display_cols
        
        # Plot heatmap
        mask = pivot.isna()
        
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-1.5, vmax=1.5, ax=ax, mask=mask,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': r'$\log_2$(Fold Change)', 'shrink': 0.8},
                    annot_kws={'fontsize': 9})
        
        # Add significance markers
        for i, stim in enumerate(pivot.index):
            for j, prot in enumerate(pivot.columns):
                orig_prot = col_order[j]
                sig_row = inhib_data[(inhib_data['Stimulus'] == stim) & 
                                     (inhib_data['Protein'] == orig_prot)]
                if not sig_row.empty and sig_row['Significant'].values[0]:
                    ax.text(j + 0.5, i + 0.15, '*', ha='center', va='center', 
                           fontsize=14, fontweight='bold', color='black')
        
        ax.set_title(f'{inhib}\n(vs. no inhibitor baseline)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Protein')
        ax.set_ylabel('Stimulus Background')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
    
    # Add annotation
    fig.text(0.5, -0.02, 
             r'* $p < 0.05$ (t-test). Blue = inhibition ($\log_2$FC < 0), Red = activation ($\log_2$FC > 0)',
             ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Inhibitor Effects Stratified by Stimulus Background', 
                 fontweight='bold', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{config.output_dir}/inhibitor_fold_change_stratified.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.output_dir}/inhibitor_fold_change_stratified.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stratified inhibitor fold change analysis")
    
    # Save detailed results to CSV
    fc_df.to_csv(f'{config.output_dir}/inhibitor_fold_change_data.csv', index=False)
    print(f"Saved fold change data to CSV")


def plot_time_lagged_correlation_exploratory(df: pd.DataFrame, parsed: Dict, conditions: pd.DataFrame, 
                                             key_proteins: List[str], config: MIDASConfig):
    """
    Plot time-lagged correlation analysis as EXPLORATORY results.
    
    This analysis provides clues about potential signal flow direction but
    is NOT causal evidence. Limitations are clearly stated.
    
    Parameters
    ----------
    df : pd.DataFrame
        MIDAS data
    parsed : dict
        Parsed column information
    conditions : pd.DataFrame
        Treatment conditions
    key_proteins : List[str]
        Key proteins to analyze
    config : MIDASConfig
        Configuration object
    """
    # Define pathway order (upstream to downstream)
    pathway_order = ['EGFR_pY1068', 'EGFR_pY1173', 'c-Met_pY1235', 'HER2_pY1248',  # Receptors
                     'MEK1_pS217_S221', 'MAPK_pT202_Y204',  # MAPK pathway
                     'AKT_pS473', 'AKT_pT308',  # PI3K/AKT
                     'mTOR_pS2448', 'p70S6K_pT389', 'S6_pS235_S236', 'S6_pS240_S244', '4EBP1_pS65']  # mTOR
    
    # Filter to available proteins
    available_proteins = [p for p in pathway_order if f'DV:{p}' in df.columns]
    
    if len(available_proteins) < 4:
        available_proteins = parsed['proteins'][:10]
    
    # Get time points
    time_points = np.sort(conditions['time'].unique())
    
    if len(time_points) < 2:
        print("Not enough time points for time-lagged analysis. Skipping.")
        return
    
    # Focus on no-inhibitor conditions
    no_inhib_mask = conditions['inhibitors'] == 'None'
    conditions_filtered = conditions[no_inhib_mask].copy()
    
    # Calculate lagged correlations
    # For each consecutive time pair, correlate protein A at t with protein B at t+1
    
    lagged_corr_data = []
    same_time_corr_data = []
    
    unique_stimuli = conditions_filtered['stimuli'].unique()
    
    for stim in unique_stimuli:
        stim_mask = conditions_filtered['stimuli'] == stim
        stim_times = conditions_filtered.loc[stim_mask, 'time'].values
        stim_indices = conditions_filtered.loc[stim_mask].index.tolist()
        
        # Sort by time
        sort_order = np.argsort(stim_times)
        sorted_times = stim_times[sort_order]
        sorted_indices = [stim_indices[i] for i in sort_order]
        
        # For consecutive time pairs
        for i in range(len(sorted_times) - 1):
            t1_idx = sorted_indices[i]
            t2_idx = sorted_indices[i + 1]
            
            for p1 in available_proteins:
                for p2 in available_proteins:
                    col1 = f'DV:{p1}'
                    col2 = f'DV:{p2}'
                    
                    # Lagged: p1 at t, p2 at t+1
                    lagged_corr_data.append({
                        'protein_t': p1,
                        'protein_t1': p2,
                        'value_t': df.loc[t1_idx, col1],
                        'value_t1': df.loc[t2_idx, col2],
                        'stimulus': stim
                    })
                    
                    # Same time: p1 at t, p2 at t
                    same_time_corr_data.append({
                        'protein1': p1,
                        'protein2': p2,
                        'value1': df.loc[t1_idx, col1],
                        'value2': df.loc[t1_idx, col2],
                        'stimulus': stim
                    })
    
    if not lagged_corr_data:
        print("No lagged correlation data computed. Skipping.")
        return
    
    lagged_df = pd.DataFrame(lagged_corr_data)
    same_df = pd.DataFrame(same_time_corr_data)
    
    # Calculate correlation matrices
    lagged_corr_matrix = pd.DataFrame(index=available_proteins, columns=available_proteins, dtype=float)
    same_corr_matrix = pd.DataFrame(index=available_proteins, columns=available_proteins, dtype=float)
    
    for p1 in available_proteins:
        for p2 in available_proteins:
            # Lagged correlation
            mask = (lagged_df['protein_t'] == p1) & (lagged_df['protein_t1'] == p2)
            subset = lagged_df.loc[mask]
            if len(subset) > 3:
                corr, _ = stats.pearsonr(subset['value_t'], subset['value_t1'])
                lagged_corr_matrix.loc[p1, p2] = corr
            else:
                lagged_corr_matrix.loc[p1, p2] = np.nan
            
            # Same-time correlation
            mask = (same_df['protein1'] == p1) & (same_df['protein2'] == p2)
            subset = same_df.loc[mask]
            if len(subset) > 3:
                corr, _ = stats.pearsonr(subset['value1'], subset['value2'])
                same_corr_matrix.loc[p1, p2] = corr
            else:
                same_corr_matrix.loc[p1, p2] = np.nan
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Get display names
    display_names = [PROTEIN_DISPLAY_NAMES.get(p, p).replace('\n', ' ') for p in available_proteins]
    
    # ===== Panel A: Same-time correlation =====
    ax1 = axes[0, 0]
    same_corr_plot = same_corr_matrix.copy()
    same_corr_plot.index = display_names
    same_corr_plot.columns = display_names
    
    sns.heatmap(same_corr_plot.astype(float), annot=False, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation', 'shrink': 0.6})
    ax1.set_title('A. Same-Time Correlation\n(Protein $i$ at $t$ vs Protein $j$ at $t$)', 
                  fontweight='bold', fontsize=11)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), fontsize=8)
    
    # ===== Panel B: Time-lagged correlation =====
    ax2 = axes[0, 1]
    lagged_corr_plot = lagged_corr_matrix.copy()
    lagged_corr_plot.index = display_names
    lagged_corr_plot.columns = display_names
    
    sns.heatmap(lagged_corr_plot.astype(float), annot=False, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax2, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation', 'shrink': 0.6})
    ax2.set_title('B. Time-Lagged Correlation\n(Protein $i$ at $t$ vs Protein $j$ at $t+\\Delta t$)', 
                  fontweight='bold', fontsize=11)
    ax2.set_xlabel('Protein $j$ (at $t+\\Delta t$)')
    ax2.set_ylabel('Protein $i$ (at $t$)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), fontsize=8)
    
    # ===== Panel C: Asymmetry (lagged - lagged.T) =====
    ax3 = axes[1, 0]
    
    lagged_vals = lagged_corr_matrix.values.astype(float)
    asymmetry = lagged_vals - lagged_vals.T
    asymmetry_df = pd.DataFrame(asymmetry, index=display_names, columns=display_names)
    
    sns.heatmap(asymmetry_df, annot=False, cmap='PiYG', center=0,
                vmin=-0.5, vmax=0.5, ax=ax3, square=True, linewidths=0.5,
                cbar_kws={'label': 'Asymmetry', 'shrink': 0.6})
    ax3.set_title('C. Directional Asymmetry\n($r_{i \\rightarrow j} - r_{j \\rightarrow i}$)', 
                  fontweight='bold', fontsize=11)
    ax3.set_xlabel('Protein $j$')
    ax3.set_ylabel('Protein $i$')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax3.get_yticklabels(), fontsize=8)
    
    # Add interpretation
    ax3.text(0.5, -0.15, 
             'Green: $i$ predicts $j$ more than vice versa\nMagenta: $j$ predicts $i$ more',
             transform=ax3.transAxes, fontsize=9, ha='center', style='italic')
    
    # ===== Panel D: Summary and caveats =====
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create text summary
    summary_text = """
    ══════════════════════════════════════════════════════════
    EXPLORATORY ANALYSIS: TIME-LAGGED CORRELATIONS
    ══════════════════════════════════════════════════════════
    
    ▸ PURPOSE:
      Identify potential temporal relationships between proteins
      that may suggest signal flow direction.
    
    ▸ INTERPRETATION:
      • Higher asymmetry (Panel C) between proteins i and j 
        suggests i may temporally precede j in signaling.
      • Upstream proteins (receptors) activating before 
        downstream proteins (S6, 4EBP1) is expected.
    
    ▸ IMPORTANT CAVEATS:
    
      ⚠️  This is CORRELATIONAL, not CAUSAL evidence.
      
      ⚠️  High correlation does not imply direct interaction.
      
      ⚠️  Time resolution may be insufficient to capture
          true signaling dynamics (seconds to minutes).
      
      ⚠️  Confounding factors (feedback loops, parallel 
          pathways) can obscure true relationships.
      
      ⚠️  Sample sizes per time point are limited.
    
    ▸ RECOMMENDED FOLLOW-UP:
      • Validate with perturbation experiments
      • Use causal inference methods (e.g., Granger causality)
      • Higher temporal resolution measurements
    
    ══════════════════════════════════════════════════════════
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', 
                      edgecolor='orange', linewidth=2, alpha=0.9))
    
    plt.suptitle('Time-Lagged Correlation Analysis (Exploratory)', 
                 fontweight='bold', fontsize=14, y=1.01)
    
    plt.tight_layout()
    plt.savefig(f'{config.output_dir}/time_lagged_correlation_exploratory.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.output_dir}/time_lagged_correlation_exploratory.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved exploratory time-lagged correlation analysis")


def plot_protein_distributions(df: pd.DataFrame, parsed: Dict, config: MIDASConfig):
    """Plot distribution of protein measurements."""
    proteins_data = df[parsed['dv_cols']]
    
    n_proteins = len(parsed['proteins'])
    n_cols = 6
    n_rows = (n_proteins + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows))
    axes = axes.flatten()
    
    for i, (col, protein) in enumerate(zip(parsed['dv_cols'], parsed['proteins'])):
        ax = axes[i]
        data = df[col].dropna()
        
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
        display_name = PROTEIN_DISPLAY_NAMES.get(protein, protein).replace('\n', ' ')
        ax.set_title(display_name, fontsize=9)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=7)
        
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1)
    
    for i in range(n_proteins, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{config.output_dir}/protein_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.output_dir}/protein_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved protein distributions plot")


def plot_correlation_heatmap(df: pd.DataFrame, parsed: Dict, config: MIDASConfig):
    """Plot correlation heatmap of protein measurements."""
    proteins_data = df[parsed['dv_cols']]
    corr_matrix = proteins_data.corr()
    
    # Use display names
    display_names = [PROTEIN_DISPLAY_NAMES.get(p, p).replace('\n', ' ') for p in parsed['proteins']]
    corr_matrix.columns = display_names
    corr_matrix.index = display_names
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Correlation', 'shrink': 0.8})
    
    ax.set_title('Protein-Protein Correlation Matrix', fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{config.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.output_dir}/correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap")


def plot_clustering_analysis(df: pd.DataFrame, parsed: Dict, conditions: pd.DataFrame, config: MIDASConfig):
    """Perform hierarchical clustering analysis."""
    proteins_data = df[parsed['dv_cols']].values
    
    scaler = StandardScaler()
    proteins_scaled = scaler.fit_transform(proteins_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Sample clustering
    ax1 = axes[0]
    linkage_samples = linkage(proteins_scaled, method='ward')
    
    sample_labels = [f"{conditions.loc[i, 'stimuli'][:8]}_{int(conditions.loc[i, 'time'])}" 
                     for i in range(len(conditions))]
    
    dendrogram(linkage_samples, ax=ax1, labels=sample_labels, 
               leaf_rotation=90, leaf_font_size=5)
    ax1.set_title('Sample Clustering (Ward)', fontweight='bold')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Distance')
    
    # Protein clustering
    ax2 = axes[1]
    linkage_proteins = linkage(proteins_scaled.T, method='ward')
    
    display_names = [PROTEIN_DISPLAY_NAMES.get(p, p).replace('\n', ' ') for p in parsed['proteins']]
    
    dendrogram(linkage_proteins, ax=ax2, labels=display_names, 
               leaf_rotation=90, leaf_font_size=8)
    ax2.set_title('Protein Clustering (Ward)', fontweight='bold')
    ax2.set_xlabel('Protein')
    ax2.set_ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig(f'{config.output_dir}/clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.output_dir}/clustering_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved clustering analysis plot")


def plot_stimuli_response_heatmap(df: pd.DataFrame, parsed: Dict, conditions: pd.DataFrame, config: MIDASConfig):
    """Plot heatmap of protein responses to different stimuli."""
    stimuli_list = conditions['stimuli'].unique()
    
    response_data = []
    for stim in stimuli_list:
        mask = conditions['stimuli'] == stim
        means = df.loc[mask, parsed['dv_cols']].mean()
        response_data.append(means.values)
    
    display_names = [PROTEIN_DISPLAY_NAMES.get(p, p).replace('\n', ' ') for p in parsed['proteins']]
    
    response_df = pd.DataFrame(response_data, index=stimuli_list, columns=display_names)
    response_standardized = (response_df - response_df.mean()) / response_df.std()
    
    g = sns.clustermap(response_standardized, 
                       cmap='RdBu_r', center=0, figsize=(18, 10),
                       dendrogram_ratio=(0.1, 0.2),
                       cbar_pos=(0.02, 0.8, 0.02, 0.15),
                       xticklabels=True, yticklabels=True)
    
    g.ax_heatmap.set_xlabel('Protein', fontsize=12)
    g.ax_heatmap.set_ylabel('Stimuli', fontsize=12)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=9)
    
    g.fig.suptitle('Stimuli-Specific Protein Response Patterns (Z-scored)', 
                   fontweight='bold', fontsize=14, y=1.02)
    
    plt.savefig(f'{config.output_dir}/stimuli_response_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.output_dir}/stimuli_response_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stimuli response heatmap")


def generate_summary_statistics(df: pd.DataFrame, parsed: Dict, conditions: pd.DataFrame, config: MIDASConfig):
    """Generate summary statistics report."""
    report = []
    report.append("=" * 60)
    report.append("MCF7 Cell Line Data Analysis Summary")
    report.append("=" * 60)
    
    report.append(f"\n### Dataset Overview ###")
    report.append(f"Total samples: {len(df)}")
    report.append(f"Number of proteins measured: {len(parsed['proteins'])}")
    report.append(f"Time points: {sorted(conditions['time'].unique())}")
    report.append(f"Unique stimuli conditions: {len(conditions['stimuli'].unique())}")
    report.append(f"Unique inhibitor conditions: {len(conditions['inhibitors'].unique())}")
    
    report.append(f"\n### Key Proteins for Main Figure ###")
    for protein in KEY_PROTEINS:
        status = "✓ available" if f'DV:{protein}' in df.columns else "✗ not found"
        report.append(f"  - {protein}: {status}")
    
    report.append(f"\n### Key Stimuli for Main Figure ###")
    for stim in KEY_STIMULI:
        status = "✓ available" if stim in conditions['stimuli'].values else "✗ not found"
        report.append(f"  - {stim}: {status}")
    
    report.append(f"\n### All Proteins Measured ###")
    for i, protein in enumerate(parsed['proteins'], 1):
        report.append(f"  {i:2d}. {protein}")
    
    report.append(f"\n### All Stimuli Tested ###")
    for stim in parsed['stimuli']:
        report.append(f"  - {stim}")
    
    report.append(f"\n### All Inhibitors Tested ###")
    for inhib in parsed['inhibitors']:
        report.append(f"  - {inhib}")
    
    report.append(f"\n### Treatment Condition Summary ###")
    condition_counts = conditions.groupby(['stimuli', 'inhibitors']).size()
    report.append(f"\nSamples per condition:")
    for (stim, inhib), count in condition_counts.items():
        report.append(f"  {stim} + {inhib}: {count} samples")
    
    report_text = '\n'.join(report)
    with open(f'{config.output_dir}/analysis_summary.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved summary report to {config.output_dir}/analysis_summary.txt")


def create_integrated_main_figure(df: pd.DataFrame, parsed: Dict, conditions: pd.DataFrame, 
                                   key_proteins: List[str], key_stimuli: List[str],
                                   pca, pca_result: np.ndarray, config: MIDASConfig):
    """
    Create an integrated main figure combining time-course and PCA trajectory.
    
    This forms a closed-loop narrative of "state space + specific pathway readout."
    
    Parameters
    ----------
    df : pd.DataFrame
        MIDAS data
    parsed : dict
        Parsed column information
    conditions : pd.DataFrame
        Treatment conditions
    key_proteins : List[str]
        Key proteins
    key_stimuli : List[str]
        Key stimuli
    pca : PCA object
        Fitted PCA
    pca_result : np.ndarray
        PCA-transformed data
    config : MIDASConfig
        Configuration object
    """
    # Filter to no-inhibitor conditions
    no_inhib_mask = conditions['inhibitors'] == 'None'
    time_points = np.sort(conditions['time'].unique())
    variance_ratio = pca.explained_variance_ratio_
    
    # Colors
    stimuli_colors = {
        'EGF': '#E64B35', 'Insulin': '#4DBBD5', 'FGF1': '#00A087',
        'HGF': '#3C5488', 'IGF1': '#F39B7F', 'Serum': '#8491B4',
    }
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    
    # GridSpec: 2 rows, 4 columns
    # Top row: PCA trajectory (spanning 2 cols) + 2 protein time courses
    # Bottom row: 4 protein time courses
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    # Panel A: PCA trajectory (top-left, spanning 2 columns)
    ax_pca = fig.add_subplot(gs[0, 0:2])
    
    # Aggregate and plot PCA trajectories
    for stim in key_stimuli:
        mask = (conditions['stimuli'] == stim) & no_inhib_mask
        if not mask.any():
            continue
        
        pc1_means, pc2_means, times_sorted = [], [], []
        
        for t in time_points:
            time_mask = mask & (conditions['time'] == t)
            if time_mask.any():
                indices = conditions.loc[time_mask].index.tolist()
                pc1_means.append(np.mean(pca_result[indices, 0]))
                pc2_means.append(np.mean(pca_result[indices, 1]))
                times_sorted.append(t)
        
        if len(times_sorted) < 2:
            continue
        
        times_sorted = np.array(times_sorted)
        pc1_means = np.array(pc1_means)
        pc2_means = np.array(pc2_means)
        
        sort_idx = np.argsort(times_sorted)
        times_sorted = times_sorted[sort_idx]
        pc1_means = pc1_means[sort_idx]
        pc2_means = pc2_means[sort_idx]
        
        color = stimuli_colors.get(stim, '#666666')
        
        ax_pca.plot(pc1_means, pc2_means, '-', color=color, linewidth=2.5, alpha=0.8, label=stim)
        
        sizes = 60 + (times_sorted - times_sorted.min()) / (times_sorted.max() - times_sorted.min() + 1e-6) * 140
        ax_pca.scatter(pc1_means, pc2_means, c=[color]*len(times_sorted), s=sizes, 
                      alpha=0.9, edgecolors='white', linewidth=1.5, zorder=5)
        ax_pca.scatter(pc1_means[0], pc2_means[0], marker='*', s=250, c=color,
                      edgecolors='black', linewidth=1.5, zorder=10)
        
        for i, t in enumerate(times_sorted):
            ax_pca.annotate(f'{int(t)}', (pc1_means[i], pc2_means[i]), 
                           textcoords="offset points", xytext=(6, 4), 
                           fontsize=8, color=color, fontweight='bold')
    
    ax_pca.set_xlabel(f'PC1 ({variance_ratio[0]*100:.1f}%)', fontsize=11)
    ax_pca.set_ylabel(f'PC2 ({variance_ratio[1]*100:.1f}%)', fontsize=11)
    ax_pca.set_title('A. Global State Space Trajectories', fontweight='bold', fontsize=12)
    ax_pca.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_pca.grid(True, alpha=0.3, linestyle='--')
    ax_pca.text(0.02, 0.98, r'$\bigstar$=$t_0$, size$\propto$time', 
               transform=ax_pca.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panels B-G: Time courses for 6 key proteins
    panel_letters = ['B', 'C', 'D', 'E', 'F', 'G']
    protein_axes = [
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]
    
    for idx, (ax, protein, letter) in enumerate(zip(protein_axes, key_proteins[:6], panel_letters)):
        col_name = f'DV:{protein}'
        
        if col_name not in df.columns:
            ax.set_visible(False)
            continue
        
        for stim in key_stimuli:
            mask = (conditions['stimuli'] == stim) & no_inhib_mask
            if not mask.any():
                continue
            
            means, ci_lowers, ci_uppers, valid_times = [], [], [], []
            
            for t in time_points:
                time_mask = mask & (conditions['time'] == t)
                if time_mask.any():
                    data = df.loc[time_mask, col_name].values
                    if len(data) > 0:
                        mean_val = np.mean(data)
                        ci_low, ci_high = compute_confidence_interval(data)
                        means.append(mean_val)
                        ci_lowers.append(ci_low)
                        ci_uppers.append(ci_high)
                        valid_times.append(t)
            
            if len(valid_times) > 0:
                color = stimuli_colors.get(stim, '#666666')
                valid_times = np.array(valid_times)
                means = np.array(means)
                ci_lowers = np.array(ci_lowers)
                ci_uppers = np.array(ci_uppers)
                
                ax.fill_between(valid_times, ci_lowers, ci_uppers, 
                               alpha=0.2, color=color, linewidth=0)
                ax.plot(valid_times, means, 'o-', color=color, 
                       markersize=5, linewidth=1.8, label=stim if idx == 0 else '')
        
        display_name = PROTEIN_DISPLAY_NAMES.get(protein, protein).replace('\n', ' ')
        ax.set_title(f'{letter}. {display_name}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Time (min)', fontsize=10)
        ax.set_ylabel('Level (a.u.)', fontsize=10)
        ax.set_xscale('symlog', linthresh=1)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Signaling Dynamics in MCF7 Cells: State Space and Pathway Readouts\n(Mean ± 95% CI, without inhibitor)', 
                 fontweight='bold', fontsize=14, y=1.01)
    
    plt.savefig(f'{config.output_dir}/integrated_main_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{config.output_dir}/integrated_main_figure.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved integrated main figure")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze MCF7 MIDAS experimental data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, default='experimental/MIDAS/MD_MCF7_main.csv',
                       help='Path to MIDAS data file')
    parser.add_argument('--output_dir', type=str, default='plots/MCF7_analysis',
                       help='Output directory for plots')
    parser.add_argument('--analysis', type=str, nargs='+', 
                       default=['all'],
                       choices=['all', 'main', 'distribution', 'correlation', 'pca', 
                               'inhibitor', 'clustering', 'heatmap', 'summary', 'lagged'],
                       help='Analyses to perform')
    args = parser.parse_args()
    
    config = MIDASConfig(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("MCF7 Cell Line Data Analysis")
    print("=" * 60)
    
    # Load data
    df = load_midas_data(config)
    parsed = parse_midas_columns(df, config)
    time_points = get_time_points(df, config)
    conditions = get_treatment_conditions(df, parsed)
    
    # Run analyses
    analyses_to_run = args.analysis
    if 'all' in analyses_to_run:
        analyses_to_run = ['main', 'distribution', 'correlation', 'pca', 
                          'inhibitor', 'clustering', 'heatmap', 'summary', 'lagged']
    
    print(f"\n### Running Analyses: {analyses_to_run} ###\n")
    
    # Track key proteins and stimuli from main analysis
    key_proteins = KEY_PROTEINS
    key_stimuli = KEY_STIMULI
    pca_obj = None
    pca_result = None
    
    if 'main' in analyses_to_run:
        print("="*50)
        print("1. MAIN FIGURE: Time-course analysis")
        print("="*50)
        key_proteins, key_stimuli = plot_main_figure_timecourse(df, parsed, conditions, config)
        
        print("\n" + "="*50)
        print("2. PCA trajectories with aggregation")
        print("="*50)
        pca_obj, pca_result = plot_pca_trajectory_with_aggregation(df, parsed, conditions, 
                                                                     key_proteins, key_stimuli, config)
        
        print("\n" + "="*50)
        print("3. Creating integrated main figure")
        print("="*50)
        create_integrated_main_figure(df, parsed, conditions, key_proteins, key_stimuli,
                                       pca_obj, pca_result, config)
    
    if 'inhibitor' in analyses_to_run:
        print("\n" + "="*50)
        print("4. Inhibitor effects: Stratified fold change analysis")
        print("="*50)
        plot_inhibitor_fold_change_stratified(df, parsed, conditions, key_proteins, config)
    
    if 'lagged' in analyses_to_run:
        print("\n" + "="*50)
        print("5. EXPLORATORY: Time-lagged correlation analysis")
        print("="*50)
        plot_time_lagged_correlation_exploratory(df, parsed, conditions, key_proteins, config)
    
    # Other analyses
    if 'distribution' in analyses_to_run:
        print("\nGenerating protein distributions...")
        plot_protein_distributions(df, parsed, config)
    
    if 'correlation' in analyses_to_run:
        print("Generating correlation heatmap...")
        plot_correlation_heatmap(df, parsed, config)
    
    if 'pca' in analyses_to_run and pca_obj is None:
        print("Performing PCA analysis...")
        pca_obj, pca_result = plot_pca_trajectory_with_aggregation(df, parsed, conditions, 
                                                                    key_proteins, key_stimuli, config)
    
    if 'clustering' in analyses_to_run:
        print("Performing clustering analysis...")
        plot_clustering_analysis(df, parsed, conditions, config)
    
    if 'heatmap' in analyses_to_run:
        print("Generating stimuli response heatmap...")
        plot_stimuli_response_heatmap(df, parsed, conditions, config)
    
    if 'summary' in analyses_to_run:
        print("\nGenerating summary statistics...")
        generate_summary_statistics(df, parsed, conditions, config)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to: {config.output_dir}")
    print("=" * 60)
    print("\n📊 KEY OUTPUTS:")
    print("  • integrated_main_figure.pdf - Combined state space + pathway readout")
    print("  • main_figure_timecourse.pdf - Time-course with 95% CI")
    print("  • pca_trajectory_aggregated.pdf - PCA with condition×time aggregation")
    print("  • inhibitor_fold_change_stratified.pdf - Stimulus-specific inhibitor effects")
    print("  • time_lagged_correlation_exploratory.pdf - Exploratory temporal analysis")


if __name__ == '__main__':
    main()