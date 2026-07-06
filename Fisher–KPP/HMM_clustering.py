import numpy as np
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler

# Publication-ready matplotlib settings (consistent with Fisher_KPP_simulation.py)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
})


class FisherKPPHMMClustering:
    """
    HMM-based clustering for Fisher-KPP reaction-diffusion data (n grid points)
    to identify different temporal phases (fast/slow dynamics)
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
        
    def select_key_grid_points(self, solutions, n_key_points=10):
        """
        Automatically select key grid points based on variance and information content
        
        Parameters:
        ----------
        solutions : np.ndarray
            All grid point solutions (n_samples, n_grid)
        n_key_points : int
            Number of key grid points to select
            
        Returns:
        -------
        list
            Indices of selected key grid points
        """
        print(f"\nAutomatically selecting {n_key_points} key grid points...")
        
        n_grid = solutions.shape[1]
        
        # 1. Calculate variance (captures dynamics range)
        variances = np.var(solutions, axis=0)
        
        # 2. Calculate correlation matrix to avoid redundant points
        corr_matrix = np.corrcoef(solutions.T)
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation
        
        # 3. Score each grid point
        grid_scores = np.zeros(n_grid)
        
        for i in range(n_grid):
            # High variance is good (captures more dynamics)
            variance_score = variances[i]
            
            # Low correlation with others is good (unique information)
            max_correlation = np.max(np.abs(corr_matrix[i, :]))
            redundancy_penalty = max_correlation
            
            # Combined score (higher is better)
            grid_scores[i] = variance_score * (1 - redundancy_penalty * 0.5)
        
        # 4. Select top-k grid points
        key_grid_idx = np.argsort(grid_scores)[-n_key_points:][::-1]
        key_grid_idx = sorted(key_grid_idx.tolist())
        
        # Print selection summary
        print(f"Selected key grid points (sorted by index):")
        for rank, idx in enumerate(key_grid_idx):
            x_val = (idx + 1) / (n_grid + 1)  # Approximate x position
            print(f"  Index {idx} (x ≈ {x_val:.3f})")
            print(f"    Variance: {variances[idx]:.6f}")
            print(f"    Max correlation: {np.max(np.abs(corr_matrix[idx, :])):.3f}")
            print(f"    Score: {grid_scores[idx]:.6f}")
        
        # 5. Analyze coverage
        total_variance = np.sum(variances)
        selected_variance = np.sum(variances[key_grid_idx])
        coverage = selected_variance / total_variance * 100
        
        print(f"\nVariance coverage: {coverage:.1f}% ({n_key_points}/{n_grid} grid points)")
        
        return key_grid_idx
    
    def prepare_features(self, fisher_kpp_data, n_key_points=10):
        """
        Prepare features for HMM clustering from Fisher-KPP data
        
        Parameters:
        ----------
        fisher_kpp_data : np.ndarray
            Fisher-KPP dataset with columns [time, u0_1, ..., u0_n, u_1, ..., u_n]
            Shape: (n_samples, 1 + 2*n_grid)
        n_key_points : int
            Number of key grid points to automatically select
            
        Returns:
        -------
        tuple
            (features, metadata) - features for HMM and metadata for reconstruction
        """
        # Infer n_grid from data shape
        n_cols = fisher_kpp_data.shape[1]
        n_grid = (n_cols - 1) // 2
        
        # Extract time and solutions
        time = fisher_kpp_data[:, 0]
        initial_conditions = fisher_kpp_data[:, 1:n_grid+1]  # n_grid initial conditions
        solutions = fisher_kpp_data[:, n_grid+1:2*n_grid+1]  # n_grid solution values
        
        # Automatically select key grid points
        key_grid_idx = self.select_key_grid_points(solutions, n_key_points=n_key_points)
        
        # Compute rates of change
        dt = np.diff(time)
        dt = np.where(dt == 0, 1e-12, dt)  # Avoid division by zero
        
        # Rates for all grid points
        solution_rates = np.diff(solutions, axis=0) / dt[:, np.newaxis]
        
        # Normalize solutions to [0,1] range per grid point
        sol_min = solutions.min(axis=0, keepdims=True)
        sol_max = solutions.max(axis=0, keepdims=True)
        sol_range = sol_max - sol_min
        sol_range = np.where(sol_range == 0, 1.0, sol_range)
        solutions_norm = (solutions[1:] - sol_min) / sol_range
        
        # Normalize rates by characteristic time scale
        initial_rate_scale = np.maximum(
            np.abs(solution_rates[:100]).mean() if len(solution_rates) > 100 else np.abs(solution_rates).mean(),
            1e-12
        )
        solution_rates_norm = solution_rates / (initial_rate_scale + 1e-12)
        
        # Dimensionless time features
        log_time = np.log(time[1:] + 1.0)
        log_time_norm = (log_time - log_time.min()) / (log_time.max() - log_time.min() + 1e-12)
        
        # Rate magnitude (dimensionless measure of dynamics speed)
        rate_magnitude = np.sqrt(np.sum(solution_rates_norm**2, axis=1))
        
        # Use automatically selected key grid points
        features = np.column_stack([
            log_time_norm,                              # Normalized time [0,1]
            solutions_norm[:, key_grid_idx],            # Key grid point solutions
            solution_rates_norm[:, key_grid_idx],       # Key grid point rates
            rate_magnitude                              # Overall dynamics speed
        ])
        
        # Store metadata for reconstruction
        metadata = {
            'time': time[1:],
            'original_indices': np.arange(1, len(fisher_kpp_data)),
            'key_grid_idx': key_grid_idx,  # Store for later use
            'n_grid': n_grid,
            'normalization_factors': {
                'initial_rate_scale': initial_rate_scale,
                'log_time_min': np.min(log_time),
                'log_time_range': np.max(log_time) - np.min(log_time),
                'sol_min': sol_min,
                'sol_range': sol_range
            }
        }
        
        print(f"\nNormalized features to dimensionless form:")
        print(f"  Feature matrix shape: {features.shape}")
        print(f"  Features: log_time + {len(key_grid_idx)} key grid solutions + {len(key_grid_idx)} rates + rate_mag")
        print(f"  Rate scale factor: {initial_rate_scale:.3e}")
        
        return features, metadata
    
    def detect_sequence_boundaries(self, fisher_kpp_data):
        """
        Detect boundaries between different initial conditions in concatenated Fisher-KPP data
        
        Parameters:
        ----------
        fisher_kpp_data : np.ndarray
            Fisher-KPP dataset with columns [time, u0_1, ..., u0_n, u_1, ..., u_n]
            
        Returns:
        -------
        list
            List of sequence lengths for each initial condition
        """
        print("Detecting sequence boundaries...")
        
        # Infer n_grid from data shape
        n_cols = fisher_kpp_data.shape[1]
        n_grid = (n_cols - 1) // 2
        
        # Extract time and initial conditions
        time = fisher_kpp_data[:, 0]
        initial_conditions = fisher_kpp_data[:, 1:n_grid+1]  # n_grid initial conditions
        
        sequence_lengths = []
        current_length = 1
        
        for i in range(1, len(fisher_kpp_data)):
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
        print(f"  Expected total: {len(fisher_kpp_data)}")
        
        # Sanity check
        if sum(sequence_lengths) != len(fisher_kpp_data):
            raise ValueError(f"Sequence length mismatch: {sum(sequence_lengths)} != {len(fisher_kpp_data)}")
        
        return sequence_lengths

    def fit_hmm(self, fisher_kpp_data, n_key_points=10, use_gmm_init=True):
        """
        Fit HMM model to identify temporal phases in Fisher-KPP data
        
        Parameters:
        ----------
        fisher_kpp_data : np.ndarray
            Fisher-KPP dataset [time, u0_1, ..., u0_n, u_1, ..., u_n]
        n_key_points : int
            Number of key grid points to automatically select
        use_gmm_init : bool
            Whether to use GMM for initialization
            
        Returns
        -------
        tuple
            (phase_labels, full_reconstructed, ...) - phase labels and reconstructed data
        """
        print("Fitting HMM for phase identification in Fisher-KPP data...")
        
        # Detect sequence boundaries first
        original_sequence_lengths = self.detect_sequence_boundaries(fisher_kpp_data)
        
        # Prepare features with automatic key grid point selection
        features, metadata = self.prepare_features(fisher_kpp_data, n_key_points=n_key_points)
        
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
            full_reconstructed = self.reconstruct_full_data(fisher_kpp_data, self.phase_labels, metadata)
            
            return self.phase_labels, full_reconstructed, features_scaled, phase_labels_clean, metadata, final_sequence_lengths
            
        except Exception as e:
            print(f"HMM fitting failed: {str(e)}")
            print("Using fallback time-based clustering...")
            
            # Fallback: simple time-based clustering
            median_time = np.median(metadata['time'])
            self.phase_labels = np.where(metadata['time'] <= median_time, 0, 1)
            
            print(f"Fallback: {np.sum(self.phase_labels == 0)} fast, {np.sum(self.phase_labels == 1)} slow points")
            
            full_reconstructed = self.reconstruct_full_data(fisher_kpp_data, self.phase_labels, metadata)
            return self.phase_labels, full_reconstructed, None, None, metadata, None

    def reconstruct_full_data(self, fisher_kpp_data, phase_labels, metadata):
        """
        Reconstruct full data with phase labels
        
        Parameters:
        ----------
        fisher_kpp_data : np.ndarray
            Original Fisher-KPP dataset
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
        subset_data = fisher_kpp_data[original_indices]
        
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
    
    def plot_spacetime_separation(self, full_reconstructed_data, features_scaled, 
                                  final_sequence_lengths, metadata, fisher_kpp_data,
                                  save_path='plots/hmm', max_plot_points=1000):
        """
        Optimized time-scale separation visualization with memory-efficient subsampling.
        
        Parameters:
        ----------
        max_plot_points : int
            Maximum number of time points to plot (subsample if larger)
        """
        os.makedirs(save_path, exist_ok=True)
        
        if self.model is None or features_scaled is None:
            print("Warning: HMM model or features not available for spacetime separation plot")
            return
        
        # Extract data
        n_grid = metadata['n_grid']
        x_grid = np.linspace(0, 1, n_grid + 2)[1:-1]  # Interior points
        
        # Get first sequence for plotting
        seq_len = min(final_sequence_lengths[0], len(metadata['time']))
        
        # MEMORY OPTIMIZATION: Subsample if too many points
        if seq_len > max_plot_points:
            print(f"  Subsampling from {seq_len} to {max_plot_points} time points for plotting...")
            # Use log-spaced indices to capture early dynamics better
            subsample_idx = np.unique(np.logspace(0, np.log10(seq_len-1), max_plot_points, dtype=int))
        else:
            subsample_idx = np.arange(seq_len)
        
        # Get subsampled data
        t_plot = metadata['time'][subsample_idx]
        
        # FIXED: Get solution data using correct indexing (subsample_idx + 1 for fisher_kpp_data which starts at row 0)
        solutions = fisher_kpp_data[subsample_idx + 1, n_grid+1:2*n_grid+1]
        
        # Compute HMM posterior probabilities (subsampled)
        seq_features = features_scaled[subsample_idx]
        _, posteriors = self.model.score_samples(seq_features, lengths=[len(subsample_idx)])
        
        # Determine which column is "fast phase"
        avg_t_c0 = np.mean(t_plot[posteriors[:, 0] > 0.5]) if np.any(posteriors[:, 0] > 0.5) else np.inf
        avg_t_c1 = np.mean(t_plot[posteriors[:, 1] > 0.5]) if np.any(posteriors[:, 1] > 0.5) else np.inf
        fast_col = 0 if avg_t_c0 < avg_t_c1 else 1
        prob_fast = posteriors[:, fast_col]
        
        print(f"\nSpacetime separation plots:")
        print(f"  Fast phase column: {fast_col}")
        print(f"  Time range: [{t_plot[0]:.3e}, {t_plot[-1]:.3e}]")
        print(f"  Plotting {len(t_plot)} time points")
        
        # Find transition time(s)
        transition_indices = np.where(np.diff(prob_fast > 0.5))[0]
        transition_times = [t_plot[idx] for idx in transition_indices]
        
        if len(transition_times) > 0:
            print(f"  Transition time(s): {[f'{t:.3f}' for t in transition_times]}")
        
        # Create meshgrid
        X, T = np.meshgrid(x_grid, t_plot)
        
        # ================================================================
        # Figure 1: 2D Spacetime with Phase Separation
        # ================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        levels_sol = np.linspace(0, 1, 100)
        cf = ax.contourf(X, T, solutions, levels=levels_sol, cmap='RdYlBu_r', extend='both')
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(r'$u(x,t)$', fontsize=12, fontweight='bold')
        
        # Add contour lines
        cs = ax.contour(X, T, solutions, levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
                        colors='white', linewidths=1, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        
        # Enhanced transition line visualization
        if len(transition_times) > 0:
            for i, (idx, t_star) in enumerate(zip(transition_indices, transition_times)):
                ax.axhline(y=t_star, color='lime', linestyle='-', linewidth=4.0, 
                          alpha=0.9, label='Phase Transition' if i == 0 else '')
                ax.axhline(y=t_star, color='black', linestyle='--', linewidth=2.0, alpha=0.7)
                
                if i == 0:
                    ax.fill_between([0, 1], 0, t_star, color='red', alpha=0.12, 
                                   label=f'Fast Phase ($t < {t_star:.2f}$)', zorder=1)
                    t_max = transition_times[-1] if len(transition_times) > 1 else t_star
                    ax.fill_between([0, 1], t_max, t_plot[-1], color='blue', alpha=0.12,
                                   label=f'Slow Phase ($t > {t_max:.2f}$)', zorder=1)
                
                y_offset = (t_plot[-1] - t_plot[0]) * 0.15
                ax.annotate(f'$t^* = {t_star:.2f}$', 
                           xy=(0.5, t_star), xytext=(0.75, t_star + y_offset),
                           fontsize=12, fontweight='bold', color='black',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                    edgecolor='black', linewidth=2, alpha=0.95),
                           arrowprops=dict(arrowstyle='->', color='black', lw=2.5,
                                         connectionstyle='arc3,rad=0.3'))
        
        ax.annotate('', xy=(0.5, 0.9), xytext=(0.5, 0.1),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='darkblue'))
        ax.text(0.53, 0.5, 'time', transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='darkblue', va='center')
        
        ax.set_xlabel(r'$x$', fontweight='bold', fontsize=12)
        ax.set_ylabel(r'$t$', fontweight='bold', fontsize=12)
        ax.set_title(r'Fisher-KPP: HMM Time-Scale Separation', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, t_plot[-1])
        ax.grid(True, alpha=0.15)
        ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', 
                 framealpha=0.95, edgecolor='black')
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_linewidth(1.5)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/separation_2d.pdf', bbox_inches='tight', dpi=600)
        plt.savefig(f'{save_path}/separation_2d.png', bbox_inches='tight', dpi=600)
        plt.close()
        print(f"  ✓ Saved: {save_path}/separation_2d.pdf/png")
        
        # ================================================================
        # Figure 2: 3D Surface with Phase Separation
        # ================================================================
        from mpl_toolkits.mplot3d import Axes3D
        
        # Further subsample for 3D (more memory intensive)
        max_3d_points = min(500, len(t_plot))
        if len(t_plot) > max_3d_points:
            subsample_3d_idx = np.linspace(0, len(t_plot)-1, max_3d_points, dtype=int)
            X_3d, T_3d = np.meshgrid(x_grid, t_plot[subsample_3d_idx])
            solutions_3d = solutions[subsample_3d_idx, :]
            print(f"  3D plot: using {max_3d_points} time points")
        else:
            X_3d, T_3d = X, T
            solutions_3d = solutions
        
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Main surface
        surf = ax.plot_surface(X_3d, T_3d, solutions_3d, cmap='RdYlBu_r', 
                              linewidth=0, antialiased=True, 
                              alpha=0.9, shade=True, vmin=0, vmax=1)
        
        # Add contour projection
        ax.contourf(X_3d, T_3d, solutions_3d, levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
                   zdir='z', offset=-0.05, cmap='RdYlBu_r', alpha=0.4)
        
        # Draw separation plane(s)
        if len(transition_times) > 0:
            for t_star in transition_times:
                t_plane = np.full_like(X_3d, t_star)
                z_plane = np.zeros_like(X_3d)
                ax.plot_surface(X_3d, t_plane, z_plane, color='lime', 
                              alpha=0.3, linewidth=0, shade=False)
                ax.plot([0, 1], [t_star, t_star], [-0.05, -0.05], 
                       color='lime', linewidth=4, alpha=0.9)
                ax.plot([0, 1], [t_star, t_star], [0, 0], 
                       color='black', linewidth=2, linestyle='--', alpha=0.7)
        
        cbar = plt.colorbar(surf, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label(r'$u(x,t)$', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        
        ax.set_xlabel(r'$x$ (space)', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel(r'$t$ (time)', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_zlabel(r'$u(x,t)$', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_title(f'Fisher-KPP 3D: Phase Separation at $t^* = {transition_times[0]:.2f}$' 
                    if len(transition_times) > 0 else 'Fisher-KPP 3D Spacetime',
                    fontweight='bold', fontsize=14, pad=20)
        
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_zlim(-0.05, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/separation_3d.pdf', bbox_inches='tight', dpi=600)
        plt.savefig(f'{save_path}/separation_3d.png', bbox_inches='tight', dpi=600)
        plt.close()
        print(f"  ✓ Saved: {save_path}/separation_3d.pdf/png")
        
        # ================================================================
        # Figure 3: Phase Probability Evolution
        # ================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(t_plot, 0, prob_fast, alpha=0.35, color='#E74C3C', 
                       label='Fast Phase Probability', edgecolor='darkred', linewidth=1.5)
        ax.fill_between(t_plot, prob_fast, 1, alpha=0.35, color='#3498DB', 
                       label='Slow Phase Probability', edgecolor='darkblue', linewidth=1.5)
        
        ax.plot(t_plot, prob_fast, color='black', linewidth=2.5, 
               label=r'$P(\mathrm{Fast}|t)$', zorder=10)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
                  label='Decision Threshold')
        
        for idx in transition_indices:
            t_trans = t_plot[idx]
            ax.axvline(x=t_trans, color='lime', linestyle='-', linewidth=4, alpha=0.9, zorder=5)
            ax.axvline(x=t_trans, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=6)
            
            ax.annotate(f'$t^* = {t_trans:.2f}$\n(Transition)', 
                       xy=(t_trans, 0.5), xytext=(t_trans * 1.3, 0.75),
                       fontsize=12, fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                edgecolor='black', linewidth=2, alpha=0.95),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2.5,
                                     connectionstyle='arc3,rad=0.2'))
            
            ax.plot(t_trans, 0.5, 'o', color='lime', markersize=12, 
                   markeredgecolor='black', markeredgewidth=2, zorder=15)
        
        ax.set_xlabel(r'$t$ (time)', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$P(\mathrm{Fast}|t)$', fontsize=12, fontweight='bold')
        ax.set_title('HMM Phase Probability Evolution', fontsize=13, fontweight='bold')
        ax.set_xlim(t_plot[0], t_plot[-1])
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=11, frameon=True, facecolor='white', 
                 framealpha=0.95, edgecolor='black')
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_linewidth(1.5)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        
        if len(transition_times) > 0:
            t_mid_fast = transition_times[0] / 2
            t_mid_slow = (transition_times[-1] + t_plot[-1]) / 2
            ax.text(t_mid_fast, 0.85, 'FAST\nPHASE', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='darkred',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='darkred', linewidth=2, alpha=0.8))
            ax.text(t_mid_slow, 0.15, 'SLOW\nPHASE', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='darkblue',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='darkblue', linewidth=2, alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/phase_probability.pdf', bbox_inches='tight', dpi=600)
        plt.savefig(f'{save_path}/phase_probability.png', bbox_inches='tight', dpi=600)
        plt.close()
        print(f"  ✓ Saved: {save_path}/phase_probability.pdf/png")
        
        # Statistics
        print(f"\n  Phase separation statistics:")
        if len(transition_times) > 0:
            fast_duration = transition_times[0]
            slow_duration = t_plot[-1] - transition_times[-1]
            total_duration = t_plot[-1]
            
            print(f"    Fast phase duration: {fast_duration:.3f} ({fast_duration/total_duration*100:.1f}%)")
            print(f"    Slow phase duration: {slow_duration:.3f} ({slow_duration/total_duration*100:.1f}%)")
            print(f"    Speed-up potential: {total_duration/slow_duration:.2f}x (if fast phase skipped)")
        
        print(f"\n  All spacetime separation plots saved to {save_path}/")

    def visualize_clustering(self, full_reconstructed_data, metadata=None, save_path='plots/hmm'):
        """Visualize HMM clustering results for Fisher-KPP data"""
        os.makedirs(save_path, exist_ok=True)
        
        # Extract data
        data_no_phases = full_reconstructed_data[:, :-1]
        phase_labels = full_reconstructed_data[:, -1].astype(int)
        
        # Infer n_grid from data shape
        n_cols = data_no_phases.shape[1]
        n_grid = (n_cols - 1) // 2
        
        time = data_no_phases[:, 0]
        solutions = data_no_phases[:, n_grid+1:2*n_grid+1]  # n_grid solution values
        
        # Get key grid indices from metadata if available
        if metadata is not None and 'key_grid_idx' in metadata:
            key_grid_idx = metadata['key_grid_idx']
        else:
            # Use evenly spaced grid points (10 points)
            n_plot = min(10, n_grid)
            key_grid_idx = np.linspace(0, n_grid-1, n_plot, dtype=int).tolist()
        
        # Create x positions for labels
        x_positions = np.linspace(0, 1, n_grid + 2)[1:-1]
        grid_labels = [f'$x={x_positions[i]:.2f}$' for i in key_grid_idx]
        
        # Define colors - dynamically create for all phases
        unique_phases = sorted(np.unique(phase_labels))
        color_palette = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        phase_colors = {p: color_palette[i % len(color_palette)] for i, p in enumerate(unique_phases)}
        phase_names = {0: 'Fast Phase', 1: 'Slow Phase', 2: 'Phase 3', -1: 'Invalid'}
        
        # Plot key grid points
        n_plot_points = len(key_grid_idx)
        n_cols_fig = 5
        n_rows = (n_plot_points + n_cols_fig - 1) // n_cols_fig
        
        fig, axes = plt.subplots(n_rows, n_cols_fig, figsize=(20, 4*n_rows))
        fig.suptitle('Phase-Aware Fisher-KPP Dynamics (Key Grid Points)', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, (ax, grid_idx, label) in enumerate(zip(axes, key_grid_idx, grid_labels)):
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
                    ax.scatter(time[plot_mask], solutions[plot_mask, grid_idx], 
                              c=phase_colors[phase], alpha=0.5, s=10, 
                              label=phase_label_str if i == 0 else "",
                              edgecolors='none', rasterized=True)
            
            ax.set_xlabel(r'$t$', fontsize=10, fontweight='bold')
            ax.set_ylabel('$u(x,t)$', fontsize=10, fontweight='bold')
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.1)
            
            if i == 0:
                ax.legend(loc='best', frameon=True)
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/fisher_kpp_hmm_phases.pdf', dpi=600, bbox_inches='tight')
        plt.savefig(f'{save_path}/fisher_kpp_hmm_phases.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        # Create space-time phase diagram
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Subsample for visualization
        if len(time) > 50000:
            sample_idx = np.random.choice(len(time), 50000, replace=False)
            sample_idx = np.sort(sample_idx)
        else:
            sample_idx = np.arange(len(time))
        
        # Use average solution for coloring
        avg_solution = np.mean(solutions[sample_idx], axis=1)
        
        scatter = ax.scatter(time[sample_idx], avg_solution, 
                           c=phase_labels[sample_idx], cmap='coolwarm',
                           alpha=0.5, s=5, rasterized=True)
        
        ax.set_xlabel(r'$t$', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average $u(x,t)$', fontsize=12, fontweight='bold')
        ax.set_title('Fisher-KPP Phase Identification', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Phase', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/fisher_kpp_phase_diagram.pdf', dpi=600, bbox_inches='tight')
        plt.savefig(f'{save_path}/fisher_kpp_phase_diagram.png', dpi=600, bbox_inches='tight')
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
        
        n_grid = metadata['n_grid']
        print(f"Data with gammas created:")
        print(f"  Shape: {data_with_gammas.shape}")
        print(f"  Format: [time, u0_1-u0_{n_grid}, u_1-u_{n_grid}, gamma_0, ..., gamma_{self.n_components-1}]")
        
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
    Run HMM clustering on Fisher-KPP high-resolution data
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='HMM Clustering for Fisher-KPP Data')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Path to Fisher-KPP data file')
    parser.add_argument('--n_components', type=int, default=2,
                       help='Number of HMM components (phases)')
    parser.add_argument('--n_key_points', type=int, default=50,
                       help='Number of key grid points to automatically select')
    args = parser.parse_args()
    
    # Find high-resolution data
    if args.data_file is None:
        data_files = [f for f in os.listdir('data/fisher_kpp') 
                     if f.startswith('teacher_high_res_fisher_kpp') and f.endswith('.npy')]
        if not data_files:
            # Fall back to regular teacher data
            data_files = [f for f in os.listdir('data/fisher_kpp') 
                         if f.startswith('teacher_fisher_kpp') and f.endswith('.npy')]
        if not data_files:
            raise FileNotFoundError("No Fisher-KPP dataset found. Run Fisher_KPP_simulation.py or teacher_generation.py first.")
        data_path = os.path.join('data/fisher_kpp', sorted(data_files)[-1])
        print(f"Found Fisher-KPP dataset: {data_path}")
    else:
        data_path = args.data_file
    
    # Load data
    fisher_kpp_data = np.load(data_path)
    print(f"Loaded Fisher-KPP dataset: {fisher_kpp_data.shape}")
    
    # Infer n_grid from data shape
    n_cols = fisher_kpp_data.shape[1]
    n_grid = (n_cols - 1) // 2
    print(f"  Inferred n_grid: {n_grid}")
    print(f"  Expected format: [time, u0_1-u0_{n_grid}, u_1-u_{n_grid}]")
    
    # Initialize HMM clustering
    hmm_clusterer = FisherKPPHMMClustering(n_components=args.n_components)
    
    # Fit HMM and get results (with automatic key grid point selection)
    phase_labels, full_reconstructed, features_scaled, phase_labels_clean, metadata, final_sequence_lengths = hmm_clusterer.fit_hmm(
        fisher_kpp_data, n_key_points=args.n_key_points
    )
    
    # Save data with gammas for PAKD
    os.makedirs('data/fisher_kpp', exist_ok=True)
    base_name = os.path.basename(data_path).replace('.npy', '')
    output_path = f'data/fisher_kpp/{base_name}_with_gammas.npy'
    
    if features_scaled is not None:
        data_with_gammas = hmm_clusterer.save_data_with_gammas(
            full_reconstructed, features_scaled, phase_labels_clean, 
            metadata, final_sequence_lengths, output_path
        )
    
    # Visualize clustering
    hmm_clusterer.visualize_clustering(full_reconstructed, metadata=metadata)
    
    # Plot spacetime separation (simplified version matching Fisher_KPP_simulation.py style)
    if features_scaled is not None and final_sequence_lengths is not None:
        hmm_clusterer.plot_spacetime_separation(
            full_reconstructed, features_scaled, 
            final_sequence_lengths, metadata, fisher_kpp_data
        )
    
    # Extract slow-phase data
    slow_data = hmm_clusterer.extract_slow_phase(full_reconstructed)
    
    # Save slow-phase data
    os.makedirs('data/fisher_kpp_slow', exist_ok=True)
    slow_path = f'data/fisher_kpp_slow/{base_name}_slow_phase.npy'
    np.save(slow_path, slow_data)
    
    print("\n" + "="*60)
    print("HMM CLUSTERING COMPLETE!")
    print("="*60)
    print(f"✓ Data with gammas: {output_path}")
    print(f"✓ Transition matrix: {output_path.replace('.npy', '_transition_matrix.npy')}")
    print(f"✓ Slow-phase data: {slow_path}")
    print(f"✓ Visualizations: plots/hmm/")
    print(f"✓ Spacetime separation: plots/hmm/solution_with_separation.pdf")
    print(f"✓ Phase probability: plots/hmm/phase_probability_vs_time.pdf")
    print("\nReady for PAKD knowledge distillation!")


if __name__ == "__main__":
    main()