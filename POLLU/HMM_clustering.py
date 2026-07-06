import numpy as np
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler

class POLLUReactionHMMClustering:
    """
    HMM-based clustering for POLLU reaction data (20 species)
    to identify different temporal phases (fast/slow)
    """
    
    def __init__(self, n_components=2, random_state=42):
        """
        Initialize HMM clustering
        
        Parameters:
        ----------
        n_components : int
            Number of hidden states (e.g., 2 for fast/slow)
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.phase_labels = None
        self.scaler = StandardScaler()
        
    def select_key_species(self, concentrations, n_key_species=10):
        """
        Automatically select key species based on variance and information content
        
        Parameters:
        ----------
        concentrations : np.ndarray
            All species concentrations (n_samples, n_species)
        n_key_species : int
            Number of key species to select
            
        Returns:
        -------
        list
            Indices of selected key species
        """
        print(f"\nAutomatically selecting {n_key_species} key species...")
        
        n_species = concentrations.shape[1]
        
        # 1. Calculate variance in log-space (captures dynamics range)
        log_conc = np.log10(concentrations + 1e-10)
        variances = np.var(log_conc, axis=0)
        
        # 2. Calculate correlation matrix to avoid redundant species
        corr_matrix = np.corrcoef(log_conc.T)
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation
        
        # 3. Score each species
        species_scores = np.zeros(n_species)
        
        for i in range(n_species):
            # High variance is good (captures more dynamics)
            variance_score = variances[i]
            
            # Low correlation with others is good (unique information)
            max_correlation = np.max(np.abs(corr_matrix[i, :]))
            redundancy_penalty = max_correlation
            
            # Combined score (higher is better)
            species_scores[i] = variance_score * (1 - redundancy_penalty * 0.5)
        
        # 4. Select top-k species
        key_species_idx = np.argsort(species_scores)[-n_key_species:][::-1]
        key_species_idx = sorted(key_species_idx.tolist())
        
        # Print selection summary
        print(f"Selected key species (sorted by importance):")
        for rank, idx in enumerate(key_species_idx):
            print(f"  Rank {rank+1}: y{idx+1} (index {idx})")
            print(f"    Variance: {variances[idx]:.3f}")
            print(f"    Max correlation: {np.max(np.abs(corr_matrix[idx, :])):.3f}")
            print(f"    Score: {species_scores[idx]:.3f}")
        
        # 5. Analyze coverage
        total_variance = np.sum(variances)
        selected_variance = np.sum(variances[key_species_idx])
        coverage = selected_variance / total_variance * 100
        
        print(f"\nVariance coverage: {coverage:.1f}% ({n_key_species}/{n_species} species)")
        
        return key_species_idx
    
    def prepare_features(self, pollu_data, n_key_species=10):
        """
        Prepare features for HMM clustering from POLLU data
        
        Parameters:
        ----------
        pollu_data : np.ndarray
            POLLU dataset with columns [time, IC1, IC2, ..., IC20, y1, y2, ..., y20]
            Shape: (n_samples, 41) - time + 20 ICs + 20 species concentrations
        n_key_species : int
            Number of key species to automatically select
            
        Returns:
        -------
        tuple
            (features, metadata) - features for HMM and metadata for reconstruction
        """
        # Extract time and concentrations
        time = pollu_data[:, 0]
        initial_conditions = pollu_data[:, 1:21]  # 20 initial conditions
        concentrations = pollu_data[:, 21:41]  # 20 species concentrations
        
        # Automatically select key species
        key_species_idx = self.select_key_species(concentrations, n_key_species=n_key_species)
        
        # Compute rates of change
        dt = np.diff(time)
        dt = np.where(dt == 0, 1e-12, dt)  # Avoid division by zero
        
        # Rates for all 20 species
        concentration_rates = np.diff(concentrations, axis=0) / dt[:, np.newaxis]
        
        # Normalize concentrations to [0,1] range per species
        conc_min = concentrations.min(axis=0, keepdims=True)
        conc_max = concentrations.max(axis=0, keepdims=True)
        conc_range = conc_max - conc_min
        conc_range = np.where(conc_range == 0, 1.0, conc_range)
        concentrations_norm = (concentrations[1:] - conc_min) / conc_range
        
        # Normalize rates by characteristic time scale
        initial_rate_scale = np.maximum(
            np.abs(concentration_rates[:100]).mean() if len(concentration_rates) > 100 else np.abs(concentration_rates).mean(),
            1e-12
        )
        concentration_rates_norm = concentration_rates / (initial_rate_scale + 1e-12)
        
        # Dimensionless time features
        log_time = np.log10(time[1:] + 1e-12)
        log_time_norm = (log_time - np.min(log_time)) / (np.max(log_time) - np.min(log_time) + 1e-12)
        
        # Rate magnitude (dimensionless measure of dynamics speed)
        rate_magnitude = np.sqrt(np.sum(concentration_rates_norm**2, axis=1))
        
        # Use automatically selected key species
        features = np.column_stack([
            log_time_norm,                              # Normalized time [0,1]
            concentrations_norm[:, key_species_idx],    # Key species concentrations
            concentration_rates_norm[:, key_species_idx], # Key species rates
            rate_magnitude                              # Overall dynamics speed
        ])
        
        # Store metadata for reconstruction
        metadata = {
            'time': time[1:],
            'original_indices': np.arange(1, len(pollu_data)),
            'key_species_idx': key_species_idx,  # Store for later use
            'normalization_factors': {
                'initial_rate_scale': initial_rate_scale,
                'log_time_min': np.min(log_time),
                'log_time_range': np.max(log_time) - np.min(log_time),
                'conc_min': conc_min,
                'conc_range': conc_range
            }
        }
        
        print(f"\nNormalized features to dimensionless form:")
        print(f"  Feature matrix shape: {features.shape}")
        print(f"  Features: log_time + {len(key_species_idx)} key species conc + {len(key_species_idx)} rates + rate_mag")
        print(f"  Rate scale factor: {initial_rate_scale:.3e}")
        
        return features, metadata
    
    def detect_sequence_boundaries(self, pollu_data):
        """
        Detect boundaries between different initial conditions in concatenated POLLU data
        
        Parameters:
        ----------
        pollu_data : np.ndarray
            POLLU dataset with columns [time, IC1, IC2, ..., IC20, y1, y2, ..., y20]
            
        Returns:
        -------
        list
            List of sequence lengths for each initial condition
        """
        print("Detecting sequence boundaries...")
        
        # Extract time and initial conditions
        time = pollu_data[:, 0]
        initial_conditions = pollu_data[:, 1:21]  # 20 initial conditions
        
        sequence_lengths = []
        current_length = 1
        
        for i in range(1, len(pollu_data)):
            # Check for time rollback (new sequence starts)
            time_rollback = time[i] < time[i-1]
            
            # Check for initial condition change (with tolerance for numerical precision)
            ic_change = np.any(np.abs(initial_conditions[i] - initial_conditions[i-1]) > 1e-10)
            
            if time_rollback or ic_change:
                # Found boundary - end current sequence
                sequence_lengths.append(current_length)
                current_length = 1
            else:
                current_length += 1
        
        # Don't forget the last sequence
        sequence_lengths.append(current_length)
        
        print(f"Detected {len(sequence_lengths)} sequences:")
        print(f"  Sequence lengths: {sequence_lengths[:10]}..." if len(sequence_lengths) > 10 else f"  Sequence lengths: {sequence_lengths}")
        print(f"  Total points: {sum(sequence_lengths)}")
        print(f"  Expected total: {len(pollu_data)}")
        
        # Sanity check
        if sum(sequence_lengths) != len(pollu_data):
            raise ValueError(f"Sequence length mismatch: {sum(sequence_lengths)} != {len(pollu_data)}")
        
        return sequence_lengths

    def fit_hmm(self, pollu_data, n_key_species=10, use_gmm_init=True):
        """
        Fit HMM model to identify temporal phases in POLLU data
        
        Parameters:
        ----------
        pollu_data : np.ndarray
            POLLU dataset [time, IC1-IC20, y1-y20]
        n_key_species : int
            Number of key species to automatically select
        use_gmm_init : bool
            Whether to use GMM for initialization
            
        Returns:
        -------
        tuple
            (phase_labels, full_reconstructed, ...) - phase labels and reconstructed data
        """
        print("Fitting HMM for phase identification in POLLU data...")
        
        # Detect sequence boundaries first
        original_sequence_lengths = self.detect_sequence_boundaries(pollu_data)
        
        # Prepare features with automatic key species selection
        features, metadata = self.prepare_features(pollu_data, n_key_species=n_key_species)
        
        # Remove any NaN or infinite values
        valid_mask = np.all(np.isfinite(features), axis=1)
        features_clean = features[valid_mask]
        
        if len(features_clean) == 0:
            raise ValueError("No valid features after cleaning")
        
        print(f"Using {len(features_clean)} valid data points out of {len(features)}")
        
        # Adjust sequence lengths for the reduced feature array
        adjusted_sequence_lengths = []
        for orig_length in original_sequence_lengths:
            adjusted_length = max(1, orig_length - 1)
            adjusted_sequence_lengths.append(adjusted_length)
        
        # Further adjust for valid_mask
        final_sequence_lengths = []
        start_idx = 0
        
        for adj_length in adjusted_sequence_lengths:
            end_idx = start_idx + adj_length
            if end_idx <= len(valid_mask):
                valid_count = np.sum(valid_mask[start_idx:end_idx])
                if valid_count > 0:
                    final_sequence_lengths.append(valid_count)
            start_idx = end_idx
        
        print(f"Final sequence lengths for HMM: {len(final_sequence_lengths)} sequences")
        
        # Verify the lengths match
        if sum(final_sequence_lengths) != len(features_clean):
            print(f"Warning: Length mismatch {sum(final_sequence_lengths)} != {len(features_clean)}")
            final_sequence_lengths = [len(features_clean)]
            print("Falling back to single sequence")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features_clean)
        
        try:
            if use_gmm_init:
                # Initialize with GMM
                print("Initializing with GMM...")
                gmm = GaussianMixture(
                    n_components=self.n_components,
                    random_state=self.random_state,
                    covariance_type='full'
                )
                gmm.fit(features_scaled)
                
                # Create HMM with GMM initialization
                self.model = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type="diag",
                    random_state=self.random_state,
                    n_iter=100,
                    tol=1e-4
                )
                
                # Set initial parameters from GMM
                self.model.means_ = gmm.means_
                self.model.covars_ = np.array([np.diag(cov) for cov in gmm.covariances_])
                
                # Set sticky transition matrix
                self.set_sticky_transitions()
                
            else:
                # Direct HMM initialization
                self.model = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type="spherical",
                    random_state=self.random_state,
                    n_iter=100
                )
                
                self.set_sticky_transitions()
        
            # Fit the model with sequence lengths
            print("Fitting HMM with proper sequence boundaries...")
            self.model.fit(features_scaled, lengths=final_sequence_lengths)
            
            # Print transition matrix after fitting
            print("Learned transition matrix:")
            for i in range(self.n_components):
                row_str = " ".join([f"{self.model.transmat_[i,j]:.3f}" for j in range(self.n_components)])
                print(f"  State {i}: [{row_str}]")
            
            # Predict phases with sequence lengths
            phase_labels_clean = self.model.predict(features_scaled, lengths=final_sequence_lengths)
            
            # Map back to full feature array
            full_phase_labels = np.full(len(features), -1)  # -1 for invalid
            full_phase_labels[valid_mask] = phase_labels_clean
            
            # Calculate phase characteristics
            phase_characteristics = self.calculate_phase_characteristics(
                features_scaled, phase_labels_clean, metadata, final_sequence_lengths
            )
            
            # Robust phase labeling
            phase_mapping = self.robust_phase_labeling(phase_characteristics)
            
            # Apply robust relabeling
            self.phase_labels = np.array([
                phase_mapping[p] if p in phase_mapping else -1 
                for p in full_phase_labels
            ])
            
            # Print final phase statistics
            print("Robust phase identification successful:")
            for new_label in [0, 1]:  # Fast, Slow
                mask = self.phase_labels == new_label
                if np.any(mask):
                    count = np.sum(mask)
                    avg_time = np.mean(metadata['time'][mask])
                    dwell_times = self.calculate_dwell_times(self.phase_labels, new_label)
                    avg_dwell = np.mean(dwell_times) if len(dwell_times) > 0 else 0
                    phase_name = "FAST" if new_label == 0 else "SLOW"
                    print(f"  {phase_name} phase: {count} points, avg_time = {avg_time:.3e}, avg_dwell = {avg_dwell:.1f}")
        
            # Reconstruct full data with phase labels
            full_reconstructed = self.reconstruct_full_data(pollu_data, self.phase_labels, metadata)
            
            return self.phase_labels, full_reconstructed, features_scaled, phase_labels_clean, metadata, final_sequence_lengths
            
        except Exception as e:
            print(f"HMM fitting failed: {str(e)}")
            print("Using fallback time-based clustering...")
            
            # Fallback: simple time-based clustering
            median_time = np.median(metadata['time'])
            self.phase_labels = np.where(metadata['time'] <= median_time, 0, 1)
            
            print(f"Fallback: {np.sum(self.phase_labels == 0)} fast, {np.sum(self.phase_labels == 1)} slow points")
            
            full_reconstructed = self.reconstruct_full_data(pollu_data, self.phase_labels, metadata)
            return self.phase_labels, full_reconstructed, None, None, metadata, None

    def reconstruct_full_data(self, pollu_data, phase_labels, metadata):
        """
        Reconstruct full data with phase labels
        
        Parameters:
        ----------
        pollu_data : np.ndarray
            Original POLLU dataset
        phase_labels : np.ndarray
            Phase labels from HMM
        metadata : dict
            Metadata including original indices
            
        Returns:
        -------
        np.ndarray
            Full reconstructed dataset with phase labels
        """
        # Get the relevant subset of original data
        original_indices = metadata['original_indices']
        subset_data = pollu_data[original_indices]
        
        # Add phase labels as additional column
        full_reconstructed = np.column_stack([subset_data, phase_labels])
        
        return full_reconstructed

    def set_sticky_transitions(self, self_loop_prob=0.9):
        """Set sticky transition matrix to encourage longer dwell times"""
        print(f"Setting sticky transitions with self-loop probability = {self_loop_prob}")
        
        transition_matrix = np.full((self.n_components, self.n_components), 
                                   (1 - self_loop_prob) / (self.n_components - 1))
        np.fill_diagonal(transition_matrix, self_loop_prob)
        
        self.model.transmat_ = transition_matrix
        
        print("Initial transition matrix (sticky):")
        for i in range(self.n_components):
            row_str = " ".join([f"{transition_matrix[i,j]:.3f}" for j in range(self.n_components)])
            print(f"  State {i}: [{row_str}]")
    
    def calculate_dwell_times(self, phase_labels, phase_id):
        """Calculate dwell times for a specific phase"""
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
    
    def calculate_phase_characteristics(self, features_scaled, phase_labels_clean, metadata, final_sequence_lengths):
        """Calculate comprehensive characteristics for each phase"""
        print("Calculating comprehensive phase characteristics...")
        
        phase_characteristics = {}
        
        for phase in range(self.n_components):
            mask = phase_labels_clean == phase
            if not np.any(mask):
                phase_characteristics[phase] = {
                    'avg_time': np.inf,
                    'avg_velocity_norm': 0.0,
                    'avg_dwell_time': 0.0,
                    'phase_fraction': 0.0,
                    'count': 0
                }
                continue
            
            phase_data = features_scaled[mask]
            phase_times = metadata['time'][mask]
            
            # Average time
            avg_time = np.mean(phase_times)
            
            # Average velocity norm (rate magnitude is last feature)
            velocity_norms = phase_data[:, -1]
            avg_velocity_norm = np.mean(velocity_norms)
            
            # Average dwell time
            dwell_times = self.calculate_dwell_times_with_sequences(
                phase_labels_clean, phase, final_sequence_lengths
            )
            avg_dwell_time = np.mean(dwell_times) if len(dwell_times) > 0 else 0.0
            
            # Phase fraction
            phase_fraction = np.sum(mask) / len(phase_labels_clean)
            
            phase_characteristics[phase] = {
                'avg_time': avg_time,
                'avg_velocity_norm': avg_velocity_norm,
                'avg_dwell_time': avg_dwell_time,
                'phase_fraction': phase_fraction,
                'count': np.sum(mask)
            }
            
            print(f"  Phase {phase}:")
            print(f"    Average time: {avg_time:.3e}")
            print(f"    Average velocity norm: {avg_velocity_norm:.3f}")
            print(f"    Average dwell time: {avg_dwell_time:.1f}")
            print(f"    Phase fraction: {phase_fraction:.3f}")
        
        return phase_characteristics

    def calculate_dwell_times_with_sequences(self, phase_labels, phase_id, sequence_lengths):
        """Calculate dwell times respecting sequence boundaries"""
        all_dwell_times = []
        start_idx = 0
        
        for seq_len in sequence_lengths:
            end_idx = start_idx + seq_len
            if end_idx <= len(phase_labels):
                seq_labels = phase_labels[start_idx:end_idx]
                seq_dwell_times = self.calculate_dwell_times(seq_labels, phase_id)
                all_dwell_times.extend(seq_dwell_times)
            start_idx = end_idx
        
        return all_dwell_times

    def robust_phase_labeling(self, phase_characteristics):
        """Robust phase labeling using multiple criteria"""
        print("Performing robust phase labeling...")
        
        if len(phase_characteristics) != 2:
            print(f"Warning: Expected 2 phases, got {len(phase_characteristics)}")
            sorted_phases = sorted(phase_characteristics.keys(), 
                                 key=lambda p: phase_characteristics[p]['avg_time'])
            return {old: new for new, old in enumerate(sorted_phases)}
        
        phase_ids = list(phase_characteristics.keys())
        p0, p1 = phase_ids
        
        char0 = phase_characteristics[p0]
        char1 = phase_characteristics[p1]
        
        # Criteria
        velocity_criterion = char0['avg_velocity_norm'] < char1['avg_velocity_norm']
        dwell_criterion = char0['avg_dwell_time'] > char1['avg_dwell_time']
        time_criterion = char0['avg_time'] > char1['avg_time']
        
        print(f"Phase labeling criteria:")
        print(f"  Phase {p0} vs Phase {p1}:")
        print(f"    Velocity: {char0['avg_velocity_norm']:.3f} vs {char1['avg_velocity_norm']:.3f} → {p0} slower: {velocity_criterion}")
        print(f"    Dwell time: {char0['avg_dwell_time']:.1f} vs {char1['avg_dwell_time']:.1f} → {p0} slower: {dwell_criterion}")
        print(f"    Avg time: {char0['avg_time']:.3e} vs {char1['avg_time']:.3e} → {p0} slower: {time_criterion}")
        
        # Majority vote
        votes_p0_slow = sum([velocity_criterion, dwell_criterion, time_criterion])
        votes_p1_slow = 3 - votes_p0_slow
        
        print(f"  Votes: Phase {p0} slow = {votes_p0_slow}, Phase {p1} slow = {votes_p1_slow}")
        
        if votes_p0_slow > votes_p1_slow:
            phase_mapping = {p0: 1, p1: 0}
            slow_phase, fast_phase = p0, p1
        elif votes_p1_slow > votes_p0_slow:
            phase_mapping = {p1: 1, p0: 0}
            slow_phase, fast_phase = p1, p0
        else:
            # Tie-breaker: use velocity criterion
            if velocity_criterion:
                phase_mapping = {p0: 1, p1: 0}
                slow_phase, fast_phase = p0, p1
            else:
                phase_mapping = {p1: 1, p0: 0}
                slow_phase, fast_phase = p1, p0
            print(f"  Tie-breaker: using velocity criterion")
        
        print(f"  Final decision: Phase {slow_phase} → SLOW (1), Phase {fast_phase} → FAST (0)")
        
        return phase_mapping

    def extract_slow_phase(self, full_reconstructed_data, slow_phase_id=None):
        """Extract slow-phase data for knowledge distillation"""
        if slow_phase_id is None:
            valid_phases = full_reconstructed_data[:, -1]
            valid_phases = valid_phases[valid_phases != -1]
            if len(valid_phases) == 0:
                raise ValueError("No valid phases found")
            slow_phase_id = int(np.max(valid_phases))
        
        slow_mask = full_reconstructed_data[:, -1] == slow_phase_id
        slow_data = full_reconstructed_data[slow_mask, :-1]
        
        print(f"Extracted slow phase data:")
        print(f"  Phase ID: {slow_phase_id}")
        print(f"  Original points: {len(full_reconstructed_data)}")
        print(f"  Slow phase points: {len(slow_data)}")
        if len(slow_data) > 0:
            print(f"  Reduction factor: {len(full_reconstructed_data) / len(slow_data):.2f}x")
            time_range = [np.min(slow_data[:, 0]), np.max(slow_data[:, 0])]
            print(f"  Time range: [{time_range[0]:.3e}, {time_range[1]:.3e}]")
        
        return slow_data
    
    def visualize_clustering(self, full_reconstructed_data, metadata=None, save_path='plots/hmm'):
        """Visualize HMM clustering results for POLLU data"""
        os.makedirs(save_path, exist_ok=True)
        
        # Extract data
        data_no_phases = full_reconstructed_data[:, :-1]
        phase_labels = full_reconstructed_data[:, -1].astype(int)
        
        time = data_no_phases[:, 0]
        concentrations = data_no_phases[:, 21:41]  # 20 species
        
        # Get key species indices from metadata if available
        if metadata is not None and 'key_species_idx' in metadata:
            key_species_idx = metadata['key_species_idx']
        else:
            # Use all 20 species
            key_species_idx = list(range(20))
        
        species_names = [f'y{i+1}' for i in key_species_idx]
        
        # Define colors - dynamically create for all phases
        unique_phases = sorted(np.unique(phase_labels))
        color_palette = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        phase_colors = {p: color_palette[i % len(color_palette)] for i, p in enumerate(unique_phases)}
        phase_names = {0: 'Fast Phase', 1: 'Slow Phase', 2: 'Phase 3', -1: 'Invalid'}
        
        # Plot all species (4 rows x 5 columns = 20 species)
        n_species = len(key_species_idx)
        n_cols = 5
        n_rows = (n_species + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        fig.suptitle('Phase-Aware POLLU Reaction Dynamics (All Species)', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, (ax, species_idx, species_name) in enumerate(zip(axes, key_species_idx, species_names)):
            unique_phases_to_plot = sorted([p for p in np.unique(phase_labels) if p != -1])
            
            for phase in unique_phases_to_plot:
                mask = phase_labels == phase
                if np.any(mask):
                    plot_mask = mask
                    if np.sum(mask) > 10000:
                        indices = np.where(mask)[0]
                        subsample_indices = np.random.choice(indices, 10000, replace=False)
                        plot_mask = np.zeros_like(mask, dtype=bool)
                        plot_mask[subsample_indices] = True
                    
                    phase_label_str = phase_names.get(phase, f'Phase {phase}')
                    ax.scatter(time[plot_mask], concentrations[plot_mask, species_idx], 
                              c=phase_colors[phase], alpha=0.5, s=10, 
                              label=phase_label_str if i == 0 else "",
                              edgecolors='none', rasterized=True)
            
            ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Concentration (M)', fontsize=10, fontweight='bold')
            ax.set_title(species_name, fontsize=11, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='best', frameon=True)
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/pollu_hmm_phases.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/pollu_hmm_phases.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {save_path}/")

    def save_data_with_gammas(self, full_reconstructed_data, features_scaled, phase_labels_clean, 
                              metadata, final_sequence_lengths, save_path):
        """Save data with posterior probabilities (gammas) for PAKD"""
        print("Computing posteriors for PAKD...")
        
        # Compute posteriors
        posteriors_list = []
        start_idx = 0
        
        for seq_len in final_sequence_lengths:
            end_idx = start_idx + seq_len
            seq_features = features_scaled[start_idx:end_idx]
            
            _, posteriors = self.model.score_samples(seq_features, lengths=[seq_len])
            posteriors_list.append(posteriors)
            
            start_idx = end_idx
        
        all_posteriors = np.vstack(posteriors_list)
        
        print(f"Posteriors computed:")
        print(f"  Shape: {all_posteriors.shape}")
        
        # Get base data (without phase label)
        data_without_phase_label = full_reconstructed_data[:, :-1]
        
        # Combine with gammas
        data_with_gammas = np.column_stack([data_without_phase_label, all_posteriors])
        
        print(f"Data with gammas created:")
        print(f"  Shape: {data_with_gammas.shape}")
        print(f"  Format: [time, IC1-IC20, y1-y20, gamma_0, ..., gamma_{self.n_components-1}]")
        
        # Save data
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, data_with_gammas)
        print(f"Saved data with gammas to: {save_path}")
        
        # Save transition matrix
        transition_matrix_path = save_path.replace('.npy', '_transition_matrix.npy')
        np.save(transition_matrix_path, self.model.transmat_)
        print(f"Saved transition matrix to: {transition_matrix_path}")
        
        return data_with_gammas

def main():
    """
    Run HMM clustering on TabPFN-generated POLLU data
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='HMM Clustering for POLLU Data')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Path to POLLU data file')
    parser.add_argument('--n_components', type=int, default=2,
                       help='Number of HMM components (phases)')
    parser.add_argument('--n_key_species', type=int, default=10,
                       help='Number of key species to automatically select')
    args = parser.parse_args()
    
    # Find TabPFN-generated data
    if args.data_file is None:
        data_files = [f for f in os.listdir('data/teacher') 
                     if f.startswith('tabpfn_high_res') and f.endswith('.npy')]
        if not data_files:
            raise FileNotFoundError("No TabPFN-generated POLLU dataset found. Run TabPFN_teacher.py with --generate_data first.")
        data_path = os.path.join('data/teacher', sorted(data_files)[-1])
        print(f"Found TabPFN-generated dataset: {data_path}")
    else:
        data_path = args.data_file
    
    # Load data
    pollu_data = np.load(data_path)
    print(f"Loaded POLLU dataset: {pollu_data.shape}")
    print(f"  Expected format: [time, IC1-IC20, y1-y20]")
    
    # Initialize HMM clustering
    hmm_clusterer = POLLUReactionHMMClustering(n_components=args.n_components)
    
    # Fit HMM and get results (with automatic key species selection)
    phase_labels, full_reconstructed, features_scaled, phase_labels_clean, metadata, final_sequence_lengths = hmm_clusterer.fit_hmm(
        pollu_data, n_key_species=args.n_key_species
    )
    
    # Save data with gammas for PAKD
    os.makedirs('data/teacher', exist_ok=True)
    base_name = os.path.basename(data_path).replace('.npy', '')
    output_path = f'data/teacher/{base_name}_with_gammas.npy'
    
    if features_scaled is not None:
        data_with_gammas = hmm_clusterer.save_data_with_gammas(
            full_reconstructed, features_scaled, phase_labels_clean, 
            metadata, final_sequence_lengths, output_path
        )
    
    # Visualize clustering (using auto-selected key species)
    hmm_clusterer.visualize_clustering(full_reconstructed, metadata=None)
    
    # Extract slow-phase data
    slow_data = hmm_clusterer.extract_slow_phase(full_reconstructed)
    
    # Save slow-phase data
    os.makedirs('data/teacher_slow', exist_ok=True)
    slow_path = f'data/teacher_slow/{base_name}_slow_phase.npy'
    np.save(slow_path, slow_data)
    
    print("\n" + "="*60)
    print("HMM CLUSTERING COMPLETE!")
    print("="*60)
    print(f"✓ Data with gammas: {output_path}")
    print(f"✓ Transition matrix: {output_path.replace('.npy', '_transition_matrix.npy')}")
    print(f"✓ Slow-phase data: {slow_path}")
    print("\nReady for PAKD knowledge distillation!")

if __name__ == "__main__":
    main()