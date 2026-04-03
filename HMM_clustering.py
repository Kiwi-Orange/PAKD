"""
HMM-based Clustering for MCF7 Signaling Data.

This module performs Hidden Markov Model clustering on MCF7 phosphoprotein
time series to identify distinct signaling phases (e.g., early/late response).
Uses high-resolution predictions from teacher_generation.py (raw mode).
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
import argparse

# ============================================================================
# MCF7 Dataset Constants (consistent with train_teacher_multi.py)
# ============================================================================
MIDAS_TREATMENT_PREFIX = 'TR:'
MIDAS_DATA_AVG_PREFIX = 'DA:'
MIDAS_DATA_VAL_PREFIX = 'DV:'

# Protein display names (consistent with test_teacher.py / teacher_generation.py)
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

# Phase colors
PHASE_COLORS = {0: '#E74C3C', 1: '#3498DB', -1: '#95a5a6'}
PHASE_NAMES = {0: 'Early Phase', 1: 'Late Phase', -1: 'Invalid'}


# ============================================================================
# Data Loading (consistent with teacher_generation.py output)
# ============================================================================
def load_high_res_data(data_path: str) -> dict:
    """
    Load high-resolution prediction data from teacher_generation.py.
    
    Parameters
    ----------
    data_path : str
        Path to .npz file produced by teacher_generation.py
        
    Returns
    -------
    dict
        Loaded data with keys: predictions, time_points, condition_indices,
        time_indices, treatment_conditions, X_high_res, column_info, etc.
    """
    print(f"Loading high-resolution data from: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    
    # Load protein names (saved alongside .npz by teacher_generation.py)
    base_name = data_path.replace('.npz', '').rsplit('_high_res', 1)[0]
    
    protein_names_path = f'{base_name}_protein_names.txt'
    if os.path.exists(protein_names_path):
        with open(protein_names_path, 'r') as f:
            protein_names = [line.strip() for line in f.readlines()]
    else:
        protein_names = [f'Protein_{i}' for i in range(data['predictions'].shape[1])]
        print(f"  ⚠ Protein names file not found, using generic names")
    
    # Load condition names
    condition_names_path = f'{base_name}_condition_names.txt'
    if os.path.exists(condition_names_path):
        with open(condition_names_path, 'r') as f:
            condition_names = [line.strip() for line in f.readlines()]
    else:
        condition_names = [f'Condition_{i}' for i in range(int(data['n_conditions']))]
        print(f"  ⚠ Condition names file not found, using generic names")
    
    result = {
        'predictions': data['predictions'],
        'X_high_res': data['X_high_res'],
        'time_points': data['time_points'],
        'treatment_conditions': data['treatment_conditions'],
        'condition_indices': data['condition_indices'],
        'time_indices': data['time_indices'],
        'n_conditions': int(data['n_conditions']),
        'n_time_points': int(data['n_time_points']),
        'n_proteins': int(data['n_proteins']),
        'column_info': {
            'protein_names': protein_names,
            'condition_names': condition_names,
        }
    }
    
    print(f"  Predictions shape: {result['predictions'].shape}")
    print(f"  Conditions: {result['n_conditions']}")
    print(f"  Time points: {result['n_time_points']}")
    print(f"  Proteins: {result['n_proteins']}")
    print(f"  Time range: [{result['time_points'].min():.1f}, {result['time_points'].max():.1f}] min")
    
    return result


# ============================================================================
# HMM Clustering
# ============================================================================
class MCF7SignalingHMMClustering:
    """
    HMM-based clustering for MCF7 signaling data.
    
    Identifies temporal phases in phosphoprotein dynamics
    (e.g., early response vs sustained signaling).
    """
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        """
        Initialize HMM clustering.
        
        Parameters
        ----------
        n_components : int
            Number of hidden states (e.g., 2 for early/late phases)
        random_state : int
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.phase_labels = None
        self.scaler = StandardScaler()
        
    def select_key_proteins(self, phospho_data: np.ndarray, protein_names: list,
                           n_key_proteins: int = 10) -> list:
        """
        Automatically select key proteins based on variance and redundancy.
        
        Selects proteins with highest dynamic range while penalizing
        highly correlated (redundant) proteins.
        
        Parameters
        ----------
        phospho_data : np.ndarray
            Phosphoprotein measurements, shape (n_samples, n_proteins)
        protein_names : list
            Names of proteins
        n_key_proteins : int
            Number of key proteins to select
            
        Returns
        -------
        list
            Indices of selected key proteins
        """
        print(f"\nAutomatically selecting {n_key_proteins} key proteins...")
        
        n_proteins = phospho_data.shape[1]
        n_key_proteins = min(n_key_proteins, n_proteins)
        
        # Variance (captures dynamic range)
        variances = np.var(phospho_data, axis=0)
        
        # Correlation matrix (for redundancy penalty)
        corr_matrix = np.corrcoef(phospho_data.T)
        np.fill_diagonal(corr_matrix, 0)
        
        # Score: high variance, low redundancy
        protein_scores = np.zeros(n_proteins)
        for i in range(n_proteins):
            max_correlation = np.max(np.abs(corr_matrix[i, :]))
            protein_scores[i] = variances[i] * (1 - max_correlation * 0.5)
        
        # Select top-k
        key_protein_idx = np.argsort(protein_scores)[-n_key_proteins:][::-1]
        key_protein_idx = sorted(key_protein_idx.tolist())
        
        # Print selection summary
        print(f"  Selected key proteins (sorted by importance):")
        for rank, idx in enumerate(key_protein_idx[:5]):
            display = PROTEIN_DISPLAY_NAMES.get(protein_names[idx], protein_names[idx])
            print(f"    Rank {rank+1}: {display} (var={variances[idx]:.4f}, score={protein_scores[idx]:.4f})")
        
        if len(key_protein_idx) > 5:
            print(f"    ... and {len(key_protein_idx) - 5} more proteins")
        
        # Coverage analysis
        total_variance = np.sum(variances)
        selected_variance = np.sum(variances[key_protein_idx])
        coverage = selected_variance / total_variance * 100
        print(f"  Variance coverage: {coverage:.1f}% ({n_key_proteins}/{n_proteins} proteins)")
        
        return key_protein_idx
    
    def prepare_features(self, data: dict, n_key_proteins: int = 10) -> tuple:
        """
        Prepare features for HMM clustering from high-resolution predictions.
        
        For each condition trajectory, computes:
        - log(time) feature
        - Current phosphorylation levels (key proteins)
        - Rates of change (key proteins)
        - Overall rate magnitude
        
        Parameters
        ----------
        data : dict
            High-resolution prediction data from teacher_generation.py
        n_key_proteins : int
            Number of key proteins to automatically select
            
        Returns
        -------
        tuple
            (features, metadata)
        """
        predictions = data['predictions']
        time_points = data['time_points']
        condition_indices = data['condition_indices']
        time_indices = data['time_indices']
        protein_names = data['column_info']['protein_names']
        
        print(f"\nPreparing features for HMM clustering...")
        print(f"  Total samples: {len(predictions)}")
        print(f"  Proteins: {len(protein_names)}")
        print(f"  Conditions: {data['n_conditions']}")
        print(f"  Time points: {data['n_time_points']}")
        
        # Automatically select key proteins
        key_protein_idx = self.select_key_proteins(
            predictions, protein_names, n_key_proteins=n_key_proteins
        )
        
        # Extract key protein data
        key_phospho = predictions[:, key_protein_idx]
        
        # Compute features for each time step in each condition trajectory
        features_list = []
        metadata_list = []
        
        n_conditions = data['n_conditions']
        
        for cond_idx in range(n_conditions):
            # Get data for this condition (sorted by time)
            cond_mask = condition_indices == cond_idx
            cond_phospho = key_phospho[cond_mask]
            cond_orig_indices = np.where(cond_mask)[0]
            cond_times = np.array([time_points[time_indices[i]] for i in cond_orig_indices])
            
            # Sort by time
            sort_idx = np.argsort(cond_times)
            cond_phospho = cond_phospho[sort_idx]
            cond_times = cond_times[sort_idx]
            cond_orig_indices = cond_orig_indices[sort_idx]
            
            # Compute rates (skip first time point — no previous point for rate)
            for t_idx in range(1, len(cond_times)):
                dt = cond_times[t_idx] - cond_times[t_idx - 1]
                if dt > 0:
                    rate = (cond_phospho[t_idx] - cond_phospho[t_idx - 1]) / dt
                else:
                    rate = np.zeros(len(key_protein_idx))
                
                # Log time (avoid log(0))
                log_time = np.log10(cond_times[t_idx] + 1.0)
                
                # Rate magnitude
                rate_magnitude = np.sqrt(np.sum(rate**2))
                
                # Feature vector: [log_time, phospho_levels, rates, rate_magnitude]
                feature = np.concatenate([
                    [log_time],
                    cond_phospho[t_idx],
                    rate,
                    [rate_magnitude]
                ])
                
                features_list.append(feature)
                metadata_list.append({
                    'condition_idx': cond_idx,
                    'time_idx': t_idx,
                    'time': cond_times[t_idx],
                    'original_idx': cond_orig_indices[t_idx]
                })
        
        features = np.array(features_list)
        
        metadata = {
            'sample_info': metadata_list,
            'key_protein_idx': key_protein_idx,
            'key_protein_names': [protein_names[i] for i in key_protein_idx],
            'n_features': features.shape[1],
            'feature_description': f'log_time + {len(key_protein_idx)} phospho + {len(key_protein_idx)} rates + rate_mag'
        }
        
        print(f"\n  Feature matrix: {features.shape}")
        print(f"  Features: {metadata['feature_description']}")
        
        return features, metadata
    
    def detect_sequence_boundaries(self, data: dict) -> list:
        """
        Detect boundaries between different treatment condition trajectories.
        
        Each condition has n_time_points - 1 feature vectors (since we compute
        rates which require a previous time point).
        
        Parameters
        ----------
        data : dict
            High-resolution prediction data
            
        Returns
        -------
        list
            List of sequence lengths for each condition
        """
        n_conditions = data['n_conditions']
        n_time_points = data['n_time_points']
        
        # Each condition trajectory produces n_time_points - 1 feature vectors
        sequence_lengths = [n_time_points - 1] * n_conditions
        
        print(f"  Sequences: {n_conditions} conditions × {n_time_points - 1} time steps = {sum(sequence_lengths)} total")
        
        return sequence_lengths

    def fit_hmm(self, data: dict, n_key_proteins: int = 10, use_gmm_init: bool = True):
        """
        Fit HMM model to identify temporal phases in MCF7 signaling data.
        
        Parameters
        ----------
        data : dict
            High-resolution prediction data from teacher_generation.py
        n_key_proteins : int
            Number of key proteins to automatically select
        use_gmm_init : bool
            Whether to use GMM for HMM parameter initialization
            
        Returns
        -------
        tuple
            (phase_labels, features_scaled, metadata, sequence_lengths)
        """
        print("\n" + "=" * 60)
        print("Fitting HMM for Phase Identification")
        print("=" * 60)
        
        # Prepare features
        features, metadata = self.prepare_features(data, n_key_proteins=n_key_proteins)
        
        # Detect sequence boundaries
        sequence_lengths = self.detect_sequence_boundaries(data)
        
        # Remove NaN/Inf values
        valid_mask = np.all(np.isfinite(features), axis=1)
        features_clean = features[valid_mask]
        
        if len(features_clean) == 0:
            raise ValueError("No valid features after cleaning")
        
        print(f"\n  Valid samples: {len(features_clean)}/{len(features)}")
        
        # Adjust sequence lengths for removed samples
        if len(features_clean) != len(features):
            final_sequence_lengths = []
            start_idx = 0
            for seq_len in sequence_lengths:
                end_idx = start_idx + seq_len
                if end_idx <= len(valid_mask):
                    valid_count = int(np.sum(valid_mask[start_idx:end_idx]))
                    if valid_count > 0:
                        final_sequence_lengths.append(valid_count)
                start_idx = end_idx
            
            if sum(final_sequence_lengths) != len(features_clean):
                print(f"  ⚠ Length mismatch after cleaning, using single sequence")
                final_sequence_lengths = [len(features_clean)]
        else:
            final_sequence_lengths = sequence_lengths
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features_clean)
        
        try:
            if use_gmm_init:
                print("\n  Initializing with GMM...")
                gmm = GaussianMixture(
                    n_components=self.n_components,
                    random_state=self.random_state,
                    covariance_type='full'
                )
                gmm.fit(features_scaled)
                
                self.model = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type="diag",
                    random_state=self.random_state,
                    n_iter=100,
                    tol=1e-4
                )
                
                # Initialize from GMM
                self.model.means_ = gmm.means_
                self.model.covars_ = np.array([np.diag(cov) for cov in gmm.covariances_])
                self._set_sticky_transitions()
                
            else:
                self.model = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type="spherical",
                    random_state=self.random_state,
                    n_iter=100
                )
                self._set_sticky_transitions()
            
            # Fit
            print("  Fitting HMM with sequence boundaries...")
            self.model.fit(features_scaled, lengths=final_sequence_lengths)
            
            # Print transition matrix
            print("\n  Learned transition matrix:")
            for i in range(self.n_components):
                row = " ".join([f"{self.model.transmat_[i,j]:.3f}" for j in range(self.n_components)])
                print(f"    State {i}: [{row}]")
            
            # Predict phases
            phase_labels_clean = self.model.predict(features_scaled, lengths=final_sequence_lengths)
            
            # Map back to full feature array
            full_phase_labels = np.full(len(features), -1)
            full_phase_labels[valid_mask] = phase_labels_clean
            
            # Calculate phase characteristics and relabel
            phase_characteristics = self._calculate_phase_characteristics(
                features_scaled, phase_labels_clean, metadata
            )
            phase_mapping = self._robust_phase_labeling(phase_characteristics)
            
            self.phase_labels = np.array([
                phase_mapping.get(p, -1) for p in full_phase_labels
            ])
            
            # Print summary
            print("\n  ✓ Phase identification successful:")
            for label in range(self.n_components):
                mask = self.phase_labels == label
                if np.any(mask):
                    print(f"    {PHASE_NAMES[label]}: {np.sum(mask)} points")
            
            return self.phase_labels, features_scaled, metadata, final_sequence_lengths
            
        except Exception as e:
            print(f"\n  ⚠ HMM fitting failed: {str(e)}")
            print("  Using fallback time-based clustering...")
            
            times = np.array([m['time'] for m in metadata['sample_info']])
            median_time = np.median(times)
            self.phase_labels = np.where(times <= median_time, 0, 1)
            
            print(f"  Fallback: {np.sum(self.phase_labels == 0)} early, "
                  f"{np.sum(self.phase_labels == 1)} late")
            
            return self.phase_labels, features_scaled, metadata, None

    def _set_sticky_transitions(self, self_loop_prob: float = 0.9):
        """Set sticky transition matrix to encourage longer dwell times."""
        transition_matrix = np.full(
            (self.n_components, self.n_components), 
            (1 - self_loop_prob) / (self.n_components - 1)
        )
        np.fill_diagonal(transition_matrix, self_loop_prob)
        self.model.transmat_ = transition_matrix
    
    def _calculate_dwell_times(self, phase_labels: np.ndarray, phase_id: int) -> list:
        """Calculate consecutive dwell times for a specific phase."""
        dwell_times = []
        current_dwell = 0
        
        for label in phase_labels:
            if label == phase_id:
                current_dwell += 1
            else:
                if current_dwell > 0:
                    dwell_times.append(current_dwell)
                    current_dwell = 0
        if current_dwell > 0:
            dwell_times.append(current_dwell)
        
        return dwell_times
    
    def _calculate_phase_characteristics(self, features_scaled: np.ndarray, 
                                         phase_labels: np.ndarray, 
                                         metadata: dict) -> dict:
        """Calculate characteristics for each phase (avg time, rate, dwell)."""
        print("\n  Calculating phase characteristics...")
        
        sample_info = metadata['sample_info']
        phase_characteristics = {}
        
        for phase in range(self.n_components):
            mask = phase_labels == phase
            if not np.any(mask):
                phase_characteristics[phase] = {
                    'avg_time': np.inf, 'avg_rate_magnitude': 0.0, 'count': 0
                }
                continue
            
            phase_data = features_scaled[mask]
            phase_times = np.array([sample_info[i]['time'] for i, m in enumerate(mask) if m])
            
            avg_time = np.mean(phase_times)
            avg_rate_magnitude = np.mean(phase_data[:, -1])  # last feature = rate_magnitude
            
            dwell_times = self._calculate_dwell_times(phase_labels, phase)
            avg_dwell = np.mean(dwell_times) if dwell_times else 0.0
            
            phase_characteristics[phase] = {
                'avg_time': avg_time,
                'avg_rate_magnitude': avg_rate_magnitude,
                'avg_dwell_time': avg_dwell,
                'count': int(np.sum(mask))
            }
            
            print(f"    Phase {phase}: avg_time={avg_time:.1f} min, "
                  f"avg_rate={avg_rate_magnitude:.4f}, n={np.sum(mask)}")
        
        return phase_characteristics

    def _robust_phase_labeling(self, phase_characteristics: dict) -> dict:
        """
        Relabel phases so that 0=Early (lower avg time) and 1=Late.
        
        Uses average time as primary criterion.
        """
        print("\n  Performing robust phase labeling...")
        
        if len(phase_characteristics) != 2:
            sorted_phases = sorted(
                phase_characteristics.keys(), 
                key=lambda p: phase_characteristics[p]['avg_time']
            )
            return {old: new for new, old in enumerate(sorted_phases)}
        
        phase_ids = list(phase_characteristics.keys())
        p0, p1 = phase_ids
        char0 = phase_characteristics[p0]
        char1 = phase_characteristics[p1]
        
        time_criterion = char0['avg_time'] < char1['avg_time']
        rate_criterion = char0['avg_rate_magnitude'] > char1['avg_rate_magnitude']
        
        print(f"    Phase {p0}: avg_time={char0['avg_time']:.1f}, avg_rate={char0['avg_rate_magnitude']:.4f}")
        print(f"    Phase {p1}: avg_time={char1['avg_time']:.1f}, avg_rate={char1['avg_rate_magnitude']:.4f}")
        print(f"    Time criterion: Phase {p0} earlier = {time_criterion}")
        print(f"    Rate criterion: Phase {p0} faster = {rate_criterion}")
        
        # Primary criterion: time
        if time_criterion:
            phase_mapping = {p0: 0, p1: 1}
            print(f"    → Phase {p0} = EARLY, Phase {p1} = LATE")
        else:
            phase_mapping = {p1: 0, p0: 1}
            print(f"    → Phase {p1} = EARLY, Phase {p0} = LATE")
        
        return phase_mapping

    def extract_phase_data(self, data: dict, phase_labels: np.ndarray, 
                          target_phase: int = 1) -> dict:
        """
        Extract data for a specific phase (for knowledge distillation).
        
        Parameters
        ----------
        data : dict
            Original high-resolution prediction data
        phase_labels : np.ndarray
            Phase labels for each sample
        target_phase : int
            Phase to extract (0=early, 1=late)
            
        Returns
        -------
        dict
            Phase mask and summary
        """
        phase_name = PHASE_NAMES.get(target_phase, f"Phase {target_phase}")
        print(f"\n  Extracting {phase_name} data...")
        
        phase_mask = phase_labels == target_phase
        
        print(f"    Total labeled samples: {len(phase_labels)}")
        print(f"    {phase_name} samples: {np.sum(phase_mask)}")
        
        return {
            'phase_mask': phase_mask,
            'phase_name': phase_name,
            'n_samples': int(np.sum(phase_mask))
        }
    
    def visualize_clustering(self, data: dict, phase_labels: np.ndarray, 
                            metadata: dict, save_path: str = 'plots/hmm'):
        """
        Visualize HMM clustering results.
        
        Generates:
        1. Key proteins over time colored by phase
        2. Phase distribution over time + summary bar chart
        
        Parameters
        ----------
        data : dict
            High-resolution prediction data
        phase_labels : np.ndarray
            Phase labels
        metadata : dict
            Metadata from prepare_features
        save_path : str
            Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)
        
        mpl.rcParams.update({
            "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
            "xtick.labelsize": 8, "ytick.labelsize": 8,
            "figure.dpi": 300, "savefig.dpi": 300,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        })
        
        predictions = data['predictions']
        protein_names = data['column_info']['protein_names']
        key_protein_idx = metadata['key_protein_idx']
        key_protein_names = metadata['key_protein_names']
        sample_info = metadata['sample_info']
        
        # ===== Plot 1: Key proteins over time colored by phase =====
        n_key = len(key_protein_idx)
        n_cols = min(5, n_key)
        n_rows = (n_key + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows))
        fig.suptitle('MCF7 Signaling Phases: Key Proteins', fontsize=13, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()
        
        for i, (prot_idx, prot_name) in enumerate(zip(key_protein_idx, key_protein_names)):
            ax = axes[i]
            display_name = PROTEIN_DISPLAY_NAMES.get(prot_name, prot_name)
            
            for phase in [0, 1]:
                phase_mask = phase_labels == phase
                if not np.any(phase_mask):
                    continue
                
                phase_times = []
                phase_values = []
                
                for j, is_phase in enumerate(phase_mask):
                    if is_phase and j < len(sample_info):
                        orig_idx = sample_info[j]['original_idx']
                        if orig_idx < len(predictions):
                            phase_times.append(sample_info[j]['time'])
                            phase_values.append(predictions[orig_idx, prot_idx])
                
                if phase_times:
                    ax.scatter(phase_times, phase_values, 
                              c=PHASE_COLORS[phase], alpha=0.5, s=15,
                              label=PHASE_NAMES[phase] if i == 0 else "",
                              edgecolors='none', rasterized=True)
            
            ax.set_xlabel('Time (min)', fontsize=9)
            ax.set_ylabel('Phosphorylation', fontsize=9)
            ax.set_title(display_name, fontsize=10, fontweight='bold')
            ax.set_xscale('symlog', linthresh=1)
            ax.grid(True, alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if i == 0:
                ax.legend(loc='best', fontsize=8, framealpha=0.9)
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/mcf7_hmm_phases.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/mcf7_hmm_phases.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ===== Plot 2: Phase distribution =====
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        
        # Phase distribution over time
        ax1 = axes[0]
        times_by_phase = {0: [], 1: []}
        for j, label in enumerate(phase_labels):
            if label in [0, 1] and j < len(sample_info):
                times_by_phase[label].append(sample_info[j]['time'])
        
        for phase in [0, 1]:
            if times_by_phase[phase]:
                ax1.hist(times_by_phase[phase], bins=30, alpha=0.6, 
                        color=PHASE_COLORS[phase], label=PHASE_NAMES[phase],
                        edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax1.set_title('Phase Distribution Over Time', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.set_xscale('symlog', linthresh=1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Phase count bar chart
        ax2 = axes[1]
        phase_counts = [int(np.sum(phase_labels == 0)), int(np.sum(phase_labels == 1))]
        bars = ax2.bar([PHASE_NAMES[0], PHASE_NAMES[1]], phase_counts, 
                      color=[PHASE_COLORS[0], PHASE_COLORS[1]],
                      edgecolor='black', linewidth=1)
        
        for bar, count in zip(bars, phase_counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax2.set_title('Phase Summary', fontsize=12, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/mcf7_phase_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/mcf7_phase_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Visualizations saved to {save_path}/")

    def save_data_with_gammas(self, data: dict, features_scaled: np.ndarray, 
                              phase_labels: np.ndarray, metadata: dict,
                              sequence_lengths: list, save_path: str):
        """
        Save data with posterior probabilities (gammas) for PAKD.
        
        Parameters
        ----------
        data : dict
            Original high-resolution prediction data
        features_scaled : np.ndarray
            Scaled features used for HMM
        phase_labels : np.ndarray
            Phase labels
        metadata : dict
            Feature metadata
        sequence_lengths : list
            Sequence lengths for HMM
        save_path : str
            Output path (.npz)
        """
        print("\n  Computing posteriors for PAKD...")
        
        if self.model is None or sequence_lengths is None:
            print("    ⚠ HMM model not available, skipping gamma computation")
            return None
        
        # Compute posteriors per sequence
        posteriors_list = []
        start_idx = 0
        
        for seq_len in sequence_lengths:
            end_idx = start_idx + seq_len
            if end_idx <= len(features_scaled):
                seq_features = features_scaled[start_idx:end_idx]
                _, posteriors = self.model.score_samples(seq_features, lengths=[seq_len])
                posteriors_list.append(posteriors)
            start_idx = end_idx
        
        if not posteriors_list:
            print("    ⚠ No posteriors computed")
            return None
        
        all_posteriors = np.vstack(posteriors_list)
        print(f"    Posteriors shape: {all_posteriors.shape}")
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        results = {
            'predictions': data['predictions'],
            'time_points': data['time_points'],
            'condition_indices': data['condition_indices'],
            'time_indices': data['time_indices'],
            'treatment_conditions': data['treatment_conditions'],
            'phase_labels': phase_labels,
            'posteriors': all_posteriors,
            'key_protein_idx': np.array(metadata['key_protein_idx']),
            'key_protein_names': np.array(metadata['key_protein_names']),
            'n_components': self.n_components,
        }
        
        np.savez_compressed(save_path, **results)
        print(f"  ✓ Saved data with gammas to: {save_path}")
        
        # Save transition matrix
        if self.model is not None:
            trans_path = save_path.replace('.npz', '_transition_matrix.npy')
            np.save(trans_path, self.model.transmat_)
            print(f"  ✓ Saved transition matrix to: {trans_path}")
        
        return results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='HMM Clustering for MCF7 Signaling Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to high-resolution prediction file (.npz) from teacher_generation.py')
    parser.add_argument('--n_components', type=int, default=2,
                       help='Number of HMM components (phases)')
    parser.add_argument('--n_key_proteins', type=int, default=30,
                       help='Number of key proteins to automatically select')
    parser.add_argument('--output_dir', type=str, default='data/hmm_clusters',
                       help='Output directory for clustered data')
    parser.add_argument('--plot_dir', type=str, default='plots/hmm',
                       help='Directory for plots')
    args = parser.parse_args()
    
    # Load high-resolution prediction data
    data = load_high_res_data(args.data_file)
    
    # Initialize HMM clustering
    hmm_clusterer = MCF7SignalingHMMClustering(n_components=args.n_components)
    
    # Fit HMM
    phase_labels, features_scaled, metadata, sequence_lengths = hmm_clusterer.fit_hmm(
        data, n_key_proteins=args.n_key_proteins
    )
    
    # Visualize results
    hmm_clusterer.visualize_clustering(data, phase_labels, metadata, args.plot_dir)
    
    # Save data with gammas for PAKD
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.data_file).replace('.npz', '')
    output_path = os.path.join(args.output_dir, f'{base_name}_with_phases.npz')
    
    if features_scaled is not None and sequence_lengths is not None:
        hmm_clusterer.save_data_with_gammas(
            data, features_scaled, phase_labels, metadata,
            sequence_lengths, output_path
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("HMM CLUSTERING COMPLETE!")
    print("=" * 60)
    print(f"\n  Input: {args.data_file}")
    print(f"  Phases: {args.n_components}")
    print(f"  Early phase: {np.sum(phase_labels == 0)} samples")
    print(f"  Late phase: {np.sum(phase_labels == 1)} samples")
    print(f"\n  Clustered data: {output_path}")
    print(f"  Visualizations: {args.plot_dir}/")
    print(f"\n  ✓ Ready for PAKD knowledge distillation!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()