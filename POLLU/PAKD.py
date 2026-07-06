import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
from tqdm import tqdm
import multiprocessing
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Import models
from models import MLP, ResidualMLP
from HMM_clustering import POLLUReactionHMMClustering

# ---------------------------
# PAKDLoss
# ---------------------------
class PAKDLoss(nn.Module):
    """Phase-Aware Knowledge Distillation Loss with Transition Smoothness"""
    def __init__(self, teacher_hidden_dim, student_hidden_dim, 
                 use_phase_weights=True, output_weight=0.6, hidden_weight=0.4,
                 smoothness_weight=0.0, transition_sensitivity=1.0):
        super().__init__()
        self.use_phase_weights = use_phase_weights
        self.mse_loss = nn.MSELoss(reduction='none')
        self.projection = nn.Linear(student_hidden_dim, teacher_hidden_dim)
        
        # Normalize weights to sum to 1
        total = output_weight + hidden_weight
        self.output_weight = output_weight / total
        self.hidden_weight = hidden_weight / total
        
        # Smoothness loss for transitions
        self.smoothness_weight = smoothness_weight
        if smoothness_weight > 0:
            self.smoothness_loss = PhaseTransitionSmoothnessLoss(transition_sensitivity)
        else:
            self.smoothness_loss = None
        
        print(f"PAKDLoss initialized: output={self.output_weight:.3f}, hidden={self.hidden_weight:.3f}")
        if smoothness_weight > 0:
            print(f"  + Smoothness loss: weight={smoothness_weight}, sensitivity={transition_sensitivity}")

    def forward(self, student_outputs, teacher_outputs, student_hidden, teacher_hidden, 
                phase_weights=None, gammas=None):
        """Forward pass with phase-aware weighting and smoothness loss"""
        # Per-sample losses
        output_loss_per_sample = self.mse_loss(student_outputs, teacher_outputs).mean(dim=1)
        hidden_loss_per_sample = self.mse_loss(self.projection(student_hidden), teacher_hidden).mean(dim=1)
        
        # Apply phase weights if provided
        if self.use_phase_weights and phase_weights is not None:
            phase_weights = phase_weights.squeeze() if phase_weights.dim() > 1 else phase_weights
            output_loss = (output_loss_per_sample * phase_weights).mean()
            hidden_loss = (hidden_loss_per_sample * phase_weights).mean()
        else:
            output_loss = output_loss_per_sample.mean()
            hidden_loss = hidden_loss_per_sample.mean()
        
        total_loss = self.output_weight * output_loss + self.hidden_weight * hidden_loss
        
        # Add smoothness loss at transitions
        smoothness = 0.0
        transition_score = 0.0
        if self.smoothness_loss is not None and gammas is not None:
            smoothness, transition_score = self.smoothness_loss(student_outputs, gammas)
            total_loss = total_loss + self.smoothness_weight * smoothness
        
        return total_loss, output_loss, hidden_loss, smoothness, transition_score

# ---------------------------
# Smoothness Loss for Phase Transitions
# ---------------------------
class PhaseTransitionSmoothnessLoss(nn.Module):
    """Smoothness loss for phase transition regions"""
    def __init__(self, transition_sensitivity=1.0):
        super().__init__()
        self.transition_sensitivity = transition_sensitivity
        print(f"PhaseTransitionSmoothnessLoss: sensitivity={transition_sensitivity}")
    
    def forward(self, student_outputs, gammas, time_indices=None):
        """
        Compute smoothness loss by penalizing output differences near transitions
        
        Args:
            student_outputs: Student predictions (batch_size, 20)
            gammas: Phase posteriors (batch_size, n_phases)
            time_indices: Optional indices to identify consecutive time points
        
        Returns:
            smoothness_loss: Loss penalizing abrupt changes at transitions
            transition_score: Average transition uncertainty in batch
        """
        # Identify transition regions: high entropy in gamma distribution
        gamma_entropy = -torch.sum(gammas * torch.log(gammas + 1e-10), dim=1)  # (batch_size,)
        max_entropy = np.log(gammas.shape[1])  # log(n_phases)
        transition_score = gamma_entropy / max_entropy  # (batch_size,) in [0, 1]
        
        # Compute output differences between consecutive samples
        if time_indices is not None:
            # Group by trajectory and compute differences within each trajectory
            output_diff = torch.zeros_like(student_outputs[:-1])
            for i in range(len(student_outputs) - 1):
                if time_indices[i+1] == time_indices[i] + 1:  # consecutive in same trajectory
                    output_diff[i] = student_outputs[i+1] - student_outputs[i]
        else:
            # Simple consecutive differences (assumes batch is time-ordered)
            output_diff = student_outputs[1:] - student_outputs[:-1]  # (batch_size-1, 20)
        
        # Compute magnitude of change (L2 norm across species)
        change_magnitude = torch.norm(output_diff, dim=1)  # (batch_size-1,)
        
        # Weight by transition score at midpoint between consecutive samples
        transition_weights = (transition_score[:-1] + transition_score[1:]) / 2.0
        
        # Smoothness loss: penalize large changes at high-uncertainty regions
        weighted_smoothness = transition_weights * change_magnitude
        
        smoothness_loss = self.transition_sensitivity * weighted_smoothness.mean()
        
        return smoothness_loss, transition_score.mean().item()

# ---------------------------
# Dataset
# ---------------------------
class PAKDDataset(Dataset):
    """Dataset with phase-aware weights from HMM"""
    def __init__(self, X_processed, teacher_outputs, teacher_hidden, gammas, phase_timescales, weight_power=1.0):
        self.X_processed = torch.tensor(X_processed, dtype=torch.float32)
        self.teacher_outputs = torch.tensor(teacher_outputs, dtype=torch.float32)
        self.teacher_hidden = torch.tensor(teacher_hidden, dtype=torch.float32)
        self.gammas_raw = torch.tensor(gammas, dtype=torch.float32)  # Store raw gammas
        
        # Pre-compute and cache phase weights
        gammas_normalized = gammas / (gammas.sum(axis=1, keepdims=True) + 1e-10)
        if np.any(~np.isfinite(gammas_normalized)):
            warnings.warn("Invalid gammas detected, using uniform distribution")
            gammas_normalized = np.ones_like(gammas_normalized) / gammas_normalized.shape[1]
        
        phase_weights = gammas_normalized @ phase_timescales
        phase_weights = phase_weights / (phase_weights.mean() + 1e-8)
        
        if weight_power != 1.0:
            phase_weights = np.power(phase_weights, weight_power)
            phase_weights = phase_weights / (phase_weights.mean() + 1e-8)
        
        self.phase_weights = torch.tensor(phase_weights, dtype=torch.float32)
        
        print(f"PAKDDataset: {len(X_processed):,} samples, weight_power={weight_power}")
        print(f"  Weight range: [{phase_weights.min():.3e}, {phase_weights.max():.3e}], "
              f"90%ile={np.percentile(phase_weights, 90):.3e}")
    
    def __len__(self):
        return len(self.X_processed)
    
    def __getitem__(self, idx):
        return (self.X_processed[idx], self.teacher_outputs[idx], 
                self.teacher_hidden[idx], self.phase_weights[idx], 
                self.gammas_raw[idx])  # Return raw gammas

# ---------------------------
# Utility Functions
# ---------------------------
def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using NVIDIA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def get_model_hidden_representation(model, inputs, layer='last'):
    """
    Get hidden representation from model
    
    Args:
        model: Student model
        inputs: Input tensor
        layer: 'first' or 'last' hidden layer
    
    Returns:
        Hidden representation tensor
    """
    if layer == 'first':
        if hasattr(model, 'get_first_hidden'):
            return model.get_first_hidden(inputs)
        else:
            raise AttributeError(f"Model {type(model).__name__} doesn't support first hidden extraction")
    
    elif layer == 'last':
        if hasattr(model, 'get_hidden_representation'):
            return model.get_hidden_representation(inputs)
        else:
            raise AttributeError(f"Model {type(model).__name__} has no hidden representation method")
    
    else:
        raise ValueError(f"Unknown layer type: {layer}. Use 'first' or 'last'")

def create_student_model(model_type, hidden_dim, num_blocks=1, dropout=0.0):
    """Create student model (POLLU: 21 inputs, 20 outputs)"""
    if model_type == 'MLP':
        return MLP(input_size=21, output_size=20, hidden_sizes=[hidden_dim]*num_blocks, dropout=dropout)
    elif model_type == 'ResidualMLP':
        return ResidualMLP(input_size=21, output_size=20, hidden_dim=hidden_dim, 
                          num_blocks=num_blocks, dropout=dropout)
    else:
        raise ValueError(f"Unknown student_type: {model_type}. Available: ['MLP', 'ResidualMLP']")

def load_teacher_model(teacher_model_path, device):
    """
    Load pre-trained teacher model from train_teacher_multi.py
    
    Returns:
        teacher_model, X_scaler, y_scaler
    """
    print(f"\nLoading teacher model: {teacher_model_path}")
    
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(f"Teacher model not found: {teacher_model_path}")
    
    checkpoint = torch.load(teacher_model_path, map_location=device)
    
    # Get model configuration
    model_type = checkpoint.get('model_type', 'ResidualMLP')
    
    # Reconstruct model
    if model_type == 'ResidualMLP':
        teacher_model = ResidualMLP(input_size=21, output_size=20, hidden_dim=128, 
                                   num_blocks=3, dropout=0.0)
    else:
        teacher_model = MLP(input_size=21, output_size=20, hidden_sizes=[128]*3, dropout=0.0)
    
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    X_scaler = checkpoint.get('X_scaler')
    y_scaler = checkpoint.get('y_scaler')
    
    print(f"✓ Loaded {model_type} teacher model")
    print(f"  State dict keys: {len(checkpoint['model_state_dict'])} parameters")
    
    return teacher_model, X_scaler, y_scaler

def extract_teacher_hidden_representations(teacher_model, X_norm, device, batch_size=256, hidden_layer='last'):
    """
    Extract hidden representations from teacher model
    
    Args:
        teacher_model: Pre-trained teacher model
        X_norm: Normalized input features
        device: torch device
        batch_size: Batch size for processing
        hidden_layer: 'first' or 'last' hidden layer
    
    Returns:
        hidden_representations: Teacher hidden layer outputs (not true embeddings)
    """
    print(f"\nExtracting teacher hidden representations ({hidden_layer} layer)...")
    
    teacher_model.eval()
    hidden_reps_list = []
    
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    dataloader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            hidden = get_model_hidden_representation(teacher_model, batch, layer=hidden_layer)
            hidden_reps_list.append(hidden.cpu().numpy())
    
    hidden_representations = np.vstack(hidden_reps_list)
    print(f"✓ Extracted hidden representations: {hidden_representations.shape}")
    
    return hidden_representations

def estimate_phase_timescales(transition_matrix, dt=1e-8, slow_phase_emphasis=1.0):
    """Estimate phase timescales from HMM transition matrix"""
    print(f"\nEstimating timescales (dt={dt:.3e}):")
    print(f"Transition matrix:\n{transition_matrix}")
    
    K = transition_matrix.shape[0]
    timescales = np.zeros(K)
    
    for k in range(K):
        A_kk = np.clip(transition_matrix[k, k], 1e-6, 0.9999)
        timescales[k] = -dt / np.log(A_kk)
        print(f"  Phase {k}: A_kk={A_kk:.6f}, tau={timescales[k]:.3e}")
    
    # Normalize by max
    timescales = timescales / np.max(timescales)
    
    # Emphasize slow phase
    if slow_phase_emphasis != 1.0:
        slow_idx = np.argmax(timescales)
        timescales[slow_idx] *= slow_phase_emphasis
        print(f"  Emphasized slow phase {slow_idx} by {slow_phase_emphasis}x")
    
    print(f"  Normalized timescales: {timescales}")
    return timescales

def create_time_features(time_array):
    """
    Create log10 time feature - matches train_teacher_multi.py
    
    Args:
        time_array: Raw time values
    
    Returns:
        log10(time + 1e-12) with shape (n_samples, 1)
    """
    t = time_array
    t_log = np.log10(t + 1e-12)
    return t_log.reshape(-1, 1)

def load_data_with_gammas(data_path, run_hmm_if_missing=True, n_key_species=10):
    """
    Load data with HMM posteriors (gammas) from HMM_clustering.py
    If gammas are not present, optionally run HMM clustering on the fly.
    
    Format: [time, IC1-IC20, y1-y20, gamma_0, gamma_1, ...]
    
    Returns:
        X: Features [time_log10, IC1-IC20] (21 features)
        y: Targets [y1-y20]
        gammas: HMM posteriors
        transition_matrix: HMM transition matrix
    """
    print(f"\nLoading data: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = np.load(data_path)
    
    # Check if data contains gammas (should have > 41 columns)
    if data.shape[1] <= 41:
        print("⚠️  Data does not contain gammas (HMM posteriors)")
        
        if not run_hmm_if_missing:
            raise ValueError("Gammas not found and run_hmm_if_missing=False. "
                           "Run HMM_clustering.py first or set run_hmm_if_missing=True")
        
        print("\n" + "="*60)
        print("RUNNING HMM CLUSTERING ON THE FLY")
        print("="*60)
        
        # Run HMM clustering
        hmm_clusterer = POLLUReactionHMMClustering(n_components=2)
        
        phase_labels, full_reconstructed, features_scaled, phase_labels_clean, metadata, final_sequence_lengths = hmm_clusterer.fit_hmm(
            data, n_key_species=n_key_species
        )
        
        if features_scaled is None:
            raise ValueError("HMM fitting failed. Cannot proceed.")
        
        # Compute posteriors
        print("Computing posteriors...")
        posteriors_list = []
        start_idx = 0
        
        for seq_len in final_sequence_lengths:
            end_idx = start_idx + seq_len
            seq_features = features_scaled[start_idx:end_idx]
            
            _, posteriors = hmm_clusterer.model.score_samples(seq_features, lengths=[seq_len])
            posteriors_list.append(posteriors)
            
            start_idx = end_idx
        
        all_posteriors = np.vstack(posteriors_list)
        
        # Extract components - align dimensions (HMM reduces by 1)
        n_hmm_points = all_posteriors.shape[0]
        time_raw = data[1:n_hmm_points+1, 0]
        ICs = data[1:n_hmm_points+1, 1:21]
        y = data[1:n_hmm_points+1, 21:41]
        
        # Apply time feature engineering (log10)
        time_log10 = create_time_features(time_raw)
        X = np.concatenate([time_log10, ICs], axis=1)  # [log10(t), IC1-IC20]
        
        gammas = all_posteriors
        transition_matrix = hmm_clusterer.model.transmat_
        
        print(f"✓ Generated gammas on the fly: {gammas.shape}")
        print(f"✓ Generated transition matrix: {transition_matrix.shape}")
        print(f"✓ Applied log10 time features: X={X.shape} [1 time + 20 ICs]")
        
        # Optionally save for future use
        save_dir = os.path.dirname(data_path)
        base_name = os.path.basename(data_path).replace('.npy', '')
        output_path = os.path.join(save_dir, f'{base_name}_with_gammas.npy')
        
        data_with_gammas = np.column_stack([X, y, gammas])
        np.save(output_path, data_with_gammas)
        print(f"✓ Saved data with gammas to: {output_path}")
        
        transition_matrix_path = output_path.replace('.npy', '_transition_matrix.npy')
        np.save(transition_matrix_path, transition_matrix)
        print(f"✓ Saved transition matrix to: {transition_matrix_path}")
        
    else:
        # Data already contains gammas
        time_raw = data[:, 0]
        ICs = data[:, 1:21]
        y = data[:, 21:41]
        gammas = data[:, 41:]
        
        # Apply time feature engineering (log10)
        time_log10 = create_time_features(time_raw)
        X = np.concatenate([time_log10, ICs], axis=1)  # [log10(t), IC1-IC20]
        
        print(f"✓ Loaded data: X={X.shape} [1 time + 20 ICs], y={y.shape}, gammas={gammas.shape}")
        print(f"✓ Applied log10 time transformation")
        
        # Load transition matrix
        transition_matrix_path = data_path.replace('.npy', '_transition_matrix.npy')
        if os.path.exists(transition_matrix_path):
            transition_matrix = np.load(transition_matrix_path)
            print(f"✓ Loaded transition matrix: {transition_matrix.shape}")
        else:
            print(f"⚠️  Transition matrix not found: {transition_matrix_path}")
            transition_matrix = None
    
    return X, y, gammas, transition_matrix

def train_epoch(student_model, dataloader, optimizer, loss_fn, device, hidden_layer='first'):
    """Train for one epoch - optimized"""
    student_model.train()
    total_loss = 0.0
    losses_dict = {'output': 0.0, 'hidden': 0.0, 'smoothness': 0.0}
    transition_scores = []
    n_batches = 0
    
    for batch_data in dataloader:
        X_norm, y_teacher, teacher_hidden, phase_weights, gammas = batch_data  # Unpack gammas
        
        # Single device transfer
        X_norm = X_norm.to(device)
        y_teacher = y_teacher.to(device)
        teacher_hidden = teacher_hidden.to(device)
        phase_weights = phase_weights.to(device)
        gammas = gammas.to(device)
        
        # Forward pass
        student_outputs = student_model(X_norm)
        student_hidden = get_model_hidden_representation(student_model, X_norm, layer=hidden_layer)
        
        # Backward pass
        optimizer.zero_grad()
        total, output_loss, hidden_loss, smoothness, transition_score = loss_fn(
            student_outputs, y_teacher, student_hidden, teacher_hidden, phase_weights, gammas
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += total.item()
        losses_dict['output'] += output_loss.item()
        losses_dict['hidden'] += hidden_loss.item()
        losses_dict['smoothness'] += smoothness if isinstance(smoothness, float) else smoothness.item()
        transition_scores.append(transition_score)
        n_batches += 1
    
    avg_transition_score = np.mean(transition_scores) if transition_scores else 0.0
    
    return (total_loss/n_batches, losses_dict['output']/n_batches, 
            losses_dict['hidden']/n_batches, losses_dict['smoothness']/n_batches,
            avg_transition_score)

def evaluate_model(student_model, dataloader, device):
    """Evaluate model - vectorized operations"""
    student_model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_norm, y_teacher, _, _, _ in dataloader:  # Unpack 5 values (added gammas)
            preds = student_model(X_norm.to(device)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_teacher.numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Vectorized R² computation
    residuals = all_targets - all_preds
    ss_res = np.sum(residuals**2, axis=0)
    ss_tot = np.sum((all_targets - all_targets.mean(axis=0))**2, axis=0)
    r2_species = 1 - ss_res / (ss_tot + 1e-10)
    r2 = r2_species.mean()
    
    # Vectorized MAE/RMSE
    abs_error = np.abs(residuals)
    mae = abs_error.mean()
    rmse = np.sqrt((residuals**2).mean())
    
    return r2, r2_species, mae, rmse

def save_figures(all_losses, student_name, r2_species, final_mae, final_rmse):
    """Save publication-quality training loss curves"""
    os.makedirs('results/pakd', exist_ok=True)
    
    import matplotlib as mpl
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
    })
    
    # Check if smoothness loss exists and is non-zero
    has_smoothness = 'smoothness' in all_losses and max(all_losses['smoothness']) > 0
    
    if has_smoothness:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(len(all_losses['total']))
    
    # Main losses
    ax1.semilogy(epochs, all_losses['total'], color='#1f77b4', linewidth=2.5, 
                label='Total Loss', alpha=0.9, marker='o', markersize=3, markevery=max(1, len(epochs)//20))
    ax1.semilogy(epochs, all_losses['output'], color='#ff7f0e', linewidth=2.5, 
                label='Output Loss', alpha=0.9, marker='s', markersize=3, markevery=max(1, len(epochs)//20))
    ax1.semilogy(epochs, all_losses['hidden'], color='#2ca02c', linewidth=2.5, 
                label='Hidden Loss', alpha=0.9, marker='^', markersize=3, markevery=max(1, len(epochs)//20))
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Training Progression for {student_name}\n', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, fancybox=False, shadow=False, 
             framealpha=0.9, loc='best')
    ax1.grid(True, alpha=0.3, linewidth=0.8, which='both')
    
    # Smoothness loss (if enabled)
    if has_smoothness:
        ax2.semilogy(epochs, all_losses['smoothness'], color='#d62728', linewidth=2.5,
                    label='Smoothness Loss', alpha=0.9, marker='D', markersize=3, markevery=max(1, len(epochs)//20))
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Smoothness Loss (log scale)', fontsize=12, fontweight='bold')
        ax2.set_title('Phase Transition Smoothness', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11, frameon=True, loc='best')
        ax2.grid(True, alpha=0.3, linewidth=0.8, which='both')
    
    plt.tight_layout()
    plt.savefig(f'results/pakd/{student_name}_training_losses.pdf')
    plt.savefig(f'results/pakd/{student_name}_training_losses.png', dpi=600)
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"✓ Training loss curves saved")
    print(f"{'='*60}\n")

def visualize_hidden_alignment(student_model, dataloader, loss_fn, device, student_name, 
                               output_dir='results/pakd', hidden_layer='first'):
    """Visualize hidden alignment between student and teacher using PCA and t-SNE"""
    import matplotlib as mpl
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print(f"\nVisualizing hidden representation alignment (layer={hidden_layer})...")
    
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
    })
    
    MAX_SAMPLES = 5000
    student_model.eval()
    
    student_hiddens_list = []
    teacher_hiddens_list = []
    times_list = []
    sample_count = 0
    
    with torch.no_grad():
        for X_norm, _, teacher_hidden, _, _ in dataloader:  # FIXED: unpack 5 values (added gammas)
            X_norm = X_norm.to(device)
            student_hidden = get_model_hidden_representation(student_model, X_norm, layer=hidden_layer)
            student_hidden_projected = loss_fn.projection(student_hidden)
            
            batch_size = len(X_norm)
            if sample_count + batch_size > MAX_SAMPLES:
                remaining = MAX_SAMPLES - sample_count
                idx = np.random.choice(batch_size, remaining, replace=False)
                student_hiddens_list.append(student_hidden_projected[idx].cpu().numpy())
                teacher_hiddens_list.append(teacher_hidden[idx].numpy())
                times_list.append(X_norm[idx, 0].cpu().numpy())
                break
            else:
                student_hiddens_list.append(student_hidden_projected.cpu().numpy())
                teacher_hiddens_list.append(teacher_hidden.numpy())
                times_list.append(X_norm[:, 0].cpu().numpy())
                sample_count += batch_size
    
    student_hiddens = np.vstack(student_hiddens_list)
    teacher_hiddens = np.vstack(teacher_hiddens_list)
    times = np.concatenate(times_list)
    times_linear = np.power(10, times) - 1e-12
    
    print(f"  Collected {len(student_hiddens):,} representations")
    
    # ============================================================
    # PCA Visualization
    # ============================================================
    print(f"  Computing PCA...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    pca = PCA(n_components=2)
    teacher_pca = pca.fit_transform(teacher_hiddens)
    student_pca = pca.transform(student_hiddens)
    
    # Teacher PCA
    ax = axes[0]
    scatter = ax.scatter(teacher_pca[:, 0], teacher_pca[:, 1], 
                        c=np.log10(times_linear + 1e-12), 
                        cmap='viridis', alpha=0.6, s=10, edgecolors='none')
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title('Teacher Hidden (PCA)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\log_{10}$(Time)', fontweight='bold')
    
    # Student PCA
    ax = axes[1]
    scatter = ax.scatter(student_pca[:, 0], student_pca[:, 1], 
                        c=np.log10(times_linear + 1e-12), 
                        cmap='viridis', alpha=0.6, s=10, edgecolors='none')
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title(f'Student {hidden_layer.capitalize()} Hidden (PCA)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\log_{10}$(Time)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{student_name}_hidden_alignment_pca.pdf')
    plt.savefig(f'{output_dir}/{student_name}_hidden_alignment_pca.png', dpi=600)
    plt.close()
    
    print(f"  ✓ Saved PCA visualization")
    
    # ============================================================
    # t-SNE Visualization
    # ============================================================
    print(f"  Computing t-SNE (this may take a few minutes)...")
    
    # Use fewer samples for t-SNE if dataset is large
    MAX_TSNE_SAMPLES = 3000
    if len(student_hiddens) > MAX_TSNE_SAMPLES:
        indices = np.random.choice(len(student_hiddens), MAX_TSNE_SAMPLES, replace=False)
        teacher_hiddens_tsne = teacher_hiddens[indices]
        student_hiddens_tsne = student_hiddens[indices]
        times_linear_tsne = times_linear[indices]
    else:
        teacher_hiddens_tsne = teacher_hiddens
        student_hiddens_tsne = student_hiddens
        times_linear_tsne = times_linear
    
    # Fit t-SNE on teacher, transform student
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    
    # Concatenate for joint embedding
    combined_hiddens = np.vstack([teacher_hiddens_tsne, student_hiddens_tsne])
    combined_tsne = tsne.fit_transform(combined_hiddens)
    
    # Split back
    n_samples = len(teacher_hiddens_tsne)
    teacher_tsne = combined_tsne[:n_samples]
    student_tsne = combined_tsne[n_samples:]
    
    # Plot t-SNE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Teacher t-SNE
    ax = axes[0]
    scatter = ax.scatter(teacher_tsne[:, 0], teacher_tsne[:, 1], 
                        c=np.log10(times_linear_tsne + 1e-12), 
                        cmap='viridis', alpha=0.6, s=15, edgecolors='none')
    ax.set_xlabel('t-SNE 1', fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontweight='bold')
    ax.set_title('Teacher Hidden (t-SNE)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\log_{10}$(Time)', fontweight='bold')
    
    # Student t-SNE
    ax = axes[1]
    scatter = ax.scatter(student_tsne[:, 0], student_tsne[:, 1], 
                        c=np.log10(times_linear_tsne + 1e-12), 
                        cmap='viridis', alpha=0.6, s=15, edgecolors='none')
    ax.set_xlabel('t-SNE 1', fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontweight='bold')
    ax.set_title(f'Student {hidden_layer.capitalize()} Hidden (t-SNE)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\log_{10}$(Time)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{student_name}_hidden_alignment_tsne.pdf')
    plt.savefig(f'{output_dir}/{student_name}_hidden_alignment_tsne.png', dpi=600)
    plt.close()
    
    print(f"  ✓ Saved t-SNE visualization")
    
    # ============================================================
    # Combined Overlay Plot (Teacher vs Student)
    # ============================================================
    print(f"  Creating overlay comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PCA Overlay
    ax = axes[0]
    ax.scatter(teacher_pca[:, 0], teacher_pca[:, 1], 
              c='#1f77b4', alpha=0.3, s=10, label='Teacher', edgecolors='none')
    ax.scatter(student_pca[:, 0], student_pca[:, 1], 
              c='#ff7f0e', alpha=0.3, s=10, label='Student', edgecolors='none')
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_title('PCA: Teacher vs Student Overlay', fontweight='bold')
    ax.legend(fontsize=10, frameon=True, loc='best')
    ax.grid(True, alpha=0.3)
    
    # t-SNE Overlay
    ax = axes[1]
    ax.scatter(teacher_tsne[:, 0], teacher_tsne[:, 1], 
              c='#1f77b4', alpha=0.3, s=15, label='Teacher', edgecolors='none')
    ax.scatter(student_tsne[:, 0], student_tsne[:, 1], 
              c='#ff7f0e', alpha=0.3, s=15, label='Student', edgecolors='none')
    ax.set_xlabel('t-SNE 1', fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontweight='bold')
    ax.set_title('t-SNE: Teacher vs Student Overlay', fontweight='bold')
    ax.legend(fontsize=10, frameon=True, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{student_name}_hidden_overlay.pdf')
    plt.savefig(f'{output_dir}/{student_name}_hidden_overlay.png', dpi=600)
    plt.close()
    
    print(f"  ✓ Saved overlay comparison")
    
    # ============================================================
    # Quantitative Alignment Metrics
    # ============================================================
    print(f"\n  Computing alignment metrics...")
    
    # Cosine similarity (sample-wise)
    from scipy.spatial.distance import cosine
    cos_similarities = [1 - cosine(t, s) for t, s in zip(teacher_hiddens[:1000], student_hiddens[:1000])]
    mean_cos_sim = np.mean(cos_similarities)
    
    # MSE between representations
    mse_repr = np.mean((teacher_hiddens - student_hiddens)**2)
    
    print(f"  Mean Cosine Similarity: {mean_cos_sim:.4f}")
    print(f"  MSE (representations): {mse_repr:.6e}")
    
    return {
        'mean_cosine_similarity': mean_cos_sim,
        'mse_representations': mse_repr
    }

# ---------------------------
# Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Phase-Aware Knowledge Distillation')
    
    # Data
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data file (with or without gammas)')
    parser.add_argument('--teacher_model', type=str, required=True,
                       help='Path to teacher model (from train_teacher_multi.py)')
    
    # Student
    parser.add_argument('--student_type', type=str, default='ResidualMLP', 
                       choices=['MLP', 'ResidualMLP'])
    parser.add_argument('--student_hidden_dim', type=int, default=128)
    parser.add_argument('--student_num_blocks', type=int, default=1)
    parser.add_argument('--student_dropout', type=float, default=0.0)
    
    # Training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_layer', type=str, default='last', 
                       choices=['first', 'last'],
                       help='Which hidden layer to use for distillation')
    
    # PAKD
    parser.add_argument('--use_phase_weights', action='store_true', default=True,
                       help='Use phase-aware weighting')
    parser.add_argument('--output_weight', type=float, default=0.7,
                       help='Weight for output loss')
    parser.add_argument('--hidden_weight', type=float, default=0.3,
                       help='Weight for hidden loss')
    parser.add_argument('--weight_power', type=float, default=7.0,
                       help='Power to raise phase weights')
    parser.add_argument('--slow_phase_emphasis', type=float, default=7.0,
                       help='Additional emphasis on slow phase timescale')
    parser.add_argument('--dt', type=str, default='auto',
                       help='Time step for timescale estimation')
    
    # NEW: Smoothness loss
    parser.add_argument('--smoothness_weight', type=float, default=0.01,
                       help='Weight for transition smoothness loss (0=disabled)')
    parser.add_argument('--transition_sensitivity', type=float, default=1.0,
                       help='Sensitivity of smoothness penalty at transitions')
    
    # HMM (for on-the-fly clustering if needed)
    parser.add_argument('--run_hmm_if_missing', action='store_true', default=True,
                       help='Run HMM clustering if gammas are not in data file')
    parser.add_argument('--n_key_species', type=int, default=10,
                       help='Number of key species for HMM (if running on the fly)')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs('models/students', exist_ok=True)
    os.makedirs('results/pakd', exist_ok=True)
    device = get_device()
    
    # Load data with gammas (run HMM if needed)
    X, y, gammas, transition_matrix = load_data_with_gammas(
        args.data, 
        run_hmm_if_missing=args.run_hmm_if_missing,
        n_key_species=args.n_key_species
    )
    
    if transition_matrix is None:
        raise ValueError("Transition matrix not found. Enable --run_hmm_if_missing or run HMM_clustering.py first.")
    
    # Load teacher model
    teacher_model, X_scaler, y_scaler = load_teacher_model(args.teacher_model, device)
    
    # Normalize inputs with teacher's scaler
    X_norm = X_scaler.transform(X)
    print(f"✓ Normalized inputs using teacher's scaler")
    
    # Normalize targets with teacher's scaler
    y_norm = y_scaler.transform(y)
    print(f"✓ Normalized targets using teacher's scaler")
    
    # Extract teacher hidden representations
    teacher_hiddens = extract_teacher_hidden_representations(
        teacher_model, X_norm, device, hidden_layer=args.hidden_layer
    )
    
    # Estimate timescales
    dt = float(args.dt) if args.dt != 'auto' else np.median(np.diff(np.unique(X[:, 0])))
    phase_timescales = estimate_phase_timescales(transition_matrix, dt, args.slow_phase_emphasis)
    
    # Create student model
    student_model = create_student_model(
        args.student_type, 
        args.student_hidden_dim,
        args.student_num_blocks,
        args.student_dropout
    ).to(device)
    
    # Get hidden dimensions
    teacher_hidden_dim = teacher_hiddens.shape[1]
    student_hidden_dim = get_model_hidden_representation(
        student_model, torch.randn(1, 21).to(device), layer=args.hidden_layer
    ).shape[1]
    
    print(f"\nHidden dimensions ({args.hidden_layer} layer):")
    print(f"  Teacher: {teacher_hidden_dim}")
    print(f"  Student: {student_hidden_dim}")
    
    # Create dataset
    dataset = PAKDDataset(X_norm, y_norm, teacher_hiddens, gammas, phase_timescales, args.weight_power)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=min(multiprocessing.cpu_count() - 1, 4), 
        pin_memory=True
    )
    
    # Setup training
    loss_fn = PAKDLoss(
        teacher_hidden_dim, 
        student_hidden_dim,
        args.use_phase_weights, 
        args.output_weight, 
        args.hidden_weight,
        args.smoothness_weight,  # NEW
        args.transition_sensitivity  # NEW
    ).to(device)
    
    optimizer = optim.Adam(
        list(student_model.parameters()) + list(loss_fn.parameters()), 
        lr=args.lr, 
        weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training: {args.student_type}, {len(dataset):,} samples")
    print(f"Using {args.hidden_layer} hidden layer for distillation")
    print(f"{'='*60}")
    
    all_losses = {'total': [], 'output': [], 'hidden': [], 'smoothness': []}  # NEW
    
    for epoch in tqdm(range(args.epochs), desc="Training"):
        losses = train_epoch(student_model, dataloader, optimizer, loss_fn, device, 
                           hidden_layer=args.hidden_layer)
        scheduler.step(losses[0])
        
        all_losses['total'].append(losses[0])
        all_losses['output'].append(losses[1])
        all_losses['hidden'].append(losses[2])
        all_losses['smoothness'].append(losses[3])  # NEW
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: total={losses[0]:.6f}, out={losses[1]:.6f}, "
                  f"hidden={losses[2]:.6f}, smooth={losses[3]:.6f}, trans_score={losses[4]:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.3e}")
    
    # Evaluate
    r2, r2_species, mae, rmse = evaluate_model(student_model, dataloader, device)
    print(f"\n{'='*60}\nFinal: R²={r2:.6f}, MAE={mae:.6e}, RMSE={rmse:.6e}\n{'='*60}\n")
    
    # Student name
    data_basename = os.path.basename(args.data).replace('.npy', '').replace('_with_gammas', '')
    method = "PAKD" if args.use_phase_weights else "KD"
    
    student_name = f"student_{method}_{args.student_type}_from_{data_basename}"
    if args.student_type == 'ResidualMLP':
        student_name += f"_blocks{args.student_num_blocks}"
    if args.weight_power != 1.0:
        student_name += f"_wp{args.weight_power}"
    student_name += f"_{args.hidden_layer}hidden"
    
    # Save figures
    save_figures(all_losses, student_name, r2_species, mae, rmse)
    
    # Visualize alignment
    visualize_hidden_alignment(student_model, dataloader, loss_fn, device, 
                              student_name, 'results/pakd', 
                              hidden_layer=args.hidden_layer)
    
    # Save model
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'model_type': args.student_type,
        'projection_state_dict': loss_fn.projection.state_dict(),
        'teacher_model_path': args.teacher_model,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'training_args': vars(args),
        'hidden_layer': args.hidden_layer,
        'final_r2': r2,
        'r2_by_species': r2_species.tolist(),
        'final_mae': mae,
        'final_rmse': rmse,
        'training_losses': all_losses,
        'input_size': 21,
        'use_time_features': True,
    }, f'models/students/{student_name}.pt')
    
    print(f"\n{'='*60}")
    print(f"✓ Model saved: models/students/{student_name}.pt")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()