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
from HMM_clustering import MMReactionHMMClustering

# ---------------------------
# PAKDLoss
# ---------------------------
class PAKDLoss(nn.Module):
    """Phase-Aware Knowledge Distillation Loss with Transition Smoothness"""
    def __init__(self, teacher_hidden_dim, student_hidden_dim, 
                 use_phase_weights=True, output_weight=0.6, hidden_weight=0.4,
                 smoothness_weight=0.0, transition_sensitivity=1.0):  # NEW
        super().__init__()
        self.use_phase_weights = use_phase_weights
        self.mse_loss = nn.MSELoss(reduction='none')
        self.projection = nn.Linear(student_hidden_dim, teacher_hidden_dim)
        
        # Normalize weights to sum to 1
        total = output_weight + hidden_weight
        self.output_weight = output_weight / total
        self.hidden_weight = hidden_weight / total
        
        # NEW: Smoothness loss for transitions
        self.smoothness_weight = smoothness_weight
        if smoothness_weight > 0:
            self.smoothness_loss = PhaseTransitionSmoothnessLoss(transition_sensitivity)
        else:
            self.smoothness_loss = None
        
        print(f"PAKDLoss initialized: output={self.output_weight:.3f}, hidden={self.hidden_weight:.3f}")
        if smoothness_weight > 0:
            print(f"  + Smoothness loss: weight={smoothness_weight}, sensitivity={transition_sensitivity}")

    def forward(self, student_outputs, teacher_outputs, student_hidden, teacher_hidden, 
                phase_weights=None, gammas=None):  # NEW: add gammas
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
        
        # NEW: Add smoothness loss at transitions
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
            student_outputs: Student predictions (batch_size, 4)
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
        # Assumes batch contains time-consecutive samples (or use time_indices)
        if time_indices is not None:
            # Group by trajectory and compute differences within each trajectory
            # This requires batch organization by trajectory
            output_diff = torch.zeros_like(student_outputs[:-1])
            for i in range(len(student_outputs) - 1):
                if time_indices[i+1] == time_indices[i] + 1:  # consecutive in same trajectory
                    output_diff[i] = student_outputs[i+1] - student_outputs[i]
        else:
            # Simple consecutive differences (assumes batch is time-ordered)
            output_diff = student_outputs[1:] - student_outputs[:-1]  # (batch_size-1, 4)
        
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
        self.gammas_raw = torch.tensor(gammas, dtype=torch.float32)  # NEW: store raw gammas
        
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
                self.gammas_raw[idx])  # NEW: return raw gammas

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
    """Create student model (MM: 5 inputs, 4 outputs)"""
    if model_type == 'MLP':
        return MLP(input_size=5, output_size=4, hidden_sizes=[hidden_dim]*num_blocks, dropout=dropout)
    elif model_type == 'ResidualMLP':
        return ResidualMLP(input_size=5, output_size=4, hidden_dim=hidden_dim, 
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
    
    checkpoint = torch.load(teacher_model_path, map_location=device, weights_only=False)
    
    # Get model configuration
    model_type = checkpoint.get('model_type', 'ResidualMLP')
    hidden_dim = checkpoint.get('hidden_dim', 128)
    num_layers = checkpoint.get('num_layers', 3)
    dropout = checkpoint.get('dropout', 0.0)
    
    # Reconstruct model
    if model_type == 'ResidualMLP':
        teacher_model = ResidualMLP(input_size=5, output_size=4, hidden_dim=hidden_dim, 
                                   num_blocks=num_layers, dropout=dropout)
    else:
        teacher_model = MLP(input_size=5, output_size=4, hidden_sizes=[hidden_dim]*num_layers, dropout=dropout)
    
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    X_scaler = checkpoint.get('X_scaler')
    y_scaler = checkpoint.get('y_scaler')
    
    print(f"✓ Loaded {model_type} teacher model")
    print(f"  Hidden dim: {hidden_dim}, Layers: {num_layers}")
    
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
        hidden_representations: Teacher hidden layer outputs
    """
    print(f"\nExtracting teacher hidden representations ({hidden_layer} layer)...")
    
    teacher_model.eval()
    hidden_reps_list = []
    
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    dataloader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
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

def load_data_with_gammas(data_path, run_hmm_if_missing=True):
    """
    Load data with CORRECTED HMM posteriors (gammas) from HMM_clustering.py
    
    Returns:
        X: Features [time_log10, E0, S0, ES0, P0] (5 features)
        y: Targets [E, S, ES, P]
        gammas: CORRECTED HMM posteriors
        transition_matrix: HMM transition matrix
        raw_time: Original time values (for dt calculation)  # NEW
    """
    print(f"\nLoading data: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = np.load(data_path)
    
    # Check if data contains gammas (should have > 9 columns)
    if data.shape[1] <= 9:
        print("⚠️  Data does not contain gammas (HMM posteriors)")
        
        if not run_hmm_if_missing:
            raise ValueError("Gammas not found and run_hmm_if_missing=False. "
                           "Run HMM_clustering.py first or set run_hmm_if_missing=True")
        
        print("\n" + "="*60)
        print("RUNNING HMM CLUSTERING ON THE FLY WITH GAMMA CORRECTION")
        print("="*60)
        
        # Run HMM clustering (this now stores phase_labels_original and phase_labels_corrected)
        hmm_clusterer = MMReactionHMMClustering(n_components=2)
        phase_labels, full_reconstructed = hmm_clusterer.fit_hmm(data)
        
        # Prepare features for gamma computation
        features, conservation_data = hmm_clusterer.prepare_features(data)
        valid_mask = np.all(np.isfinite(features), axis=1)
        features_scaled = hmm_clusterer.scaler.transform(features[valid_mask])
        
        # Get sequence lengths
        seq_lengths = hmm_clusterer.detect_sequence_boundaries(data)
        adj_lengths = [max(1, l - 1) for l in seq_lengths]
        
        final_lengths = []
        start = 0
        for adj_len in adj_lengths:
            end = start + adj_len
            if end <= len(valid_mask):
                valid_count = np.sum(valid_mask[start:end])
                if valid_count > 0:
                    final_lengths.append(valid_count)
            start = end
        
        if sum(final_lengths) != len(features_scaled):
            final_lengths = [len(features_scaled)]
        
        # Compute raw posteriors from HMM
        print("\nComputing raw posteriors from HMM...")
        posteriors_list = []
        start = 0
        for seq_len in final_lengths:
            end = start + seq_len
            _, posteriors = hmm_clusterer.model.score_samples(features_scaled[start:end], lengths=[seq_len])
            posteriors_list.append(posteriors)
            start = end
        
        all_posteriors = np.vstack(posteriors_list)
        
        # ============ APPLY GAMMA CORRECTION ============
        print("\n🔧 Applying gamma correction based on phase label corrections...")
        
        # Check if correction info is available
        if hasattr(hmm_clusterer, 'phase_labels_original') and hasattr(hmm_clusterer, 'phase_labels_corrected'):
            corrected_posteriors = hmm_clusterer.adjust_gammas_from_corrected_labels(
                all_posteriors,
                hmm_clusterer.phase_labels_original,
                hmm_clusterer.phase_labels_corrected
            )
            print("✓ Gammas corrected to match phase label corrections")
        else:
            corrected_posteriors = all_posteriors
            print("⚠️  No correction info available, using raw posteriors")
        # ================================================
        
        # Map to full feature array
        gammas = np.zeros((len(features), hmm_clusterer.n_components))
        gammas[valid_mask] = corrected_posteriors
        gammas[~valid_mask] = 1.0 / hmm_clusterer.n_components
        
        original_indices = conservation_data['original_indices']
        subset_data = data[original_indices]
        
        time_raw = subset_data[:, 0]
        ICs = subset_data[:, 1:5]
        y = subset_data[:, 5:9]
        
        # Apply time feature engineering (log10)
        time_log10 = create_time_features(time_raw)
        X = np.concatenate([time_log10, ICs], axis=1)  # [log10(t), E0, S0, ES0, P0]
        
        transition_matrix = hmm_clusterer.model.transmat_
        
        print(f"✓ Generated CORRECTED gammas on the fly: {gammas.shape}")
        print(f"✓ Applied log10 time features: X={X.shape} [1 time + 4 ICs]")
        
        # Save for future use
        save_dir = os.path.dirname(data_path)
        base_name = os.path.basename(data_path).replace('.npy', '')
        output_path = os.path.join(save_dir, f'{base_name}_with_gammas.npy')
        
        data_with_gammas = np.column_stack([X, y, gammas])
        np.save(output_path, data_with_gammas)
        print(f"✓ Saved data with CORRECTED gammas to: {output_path}")
        
        transition_matrix_path = output_path.replace('.npy', '_transition_matrix.npy')
        np.save(transition_matrix_path, transition_matrix)
        print(f"✓ Saved transition matrix to: {transition_matrix_path}")
        
    else:
        # Data already contains gammas (should already be corrected if saved via HMM_clustering.py)
        time_raw = data[:, 0]
        ICs = data[:, 1:5]
        y = data[:, 5:9]
        gammas = data[:, 9:]
        
        # Apply time feature engineering (log10)
        time_log10 = create_time_features(time_raw)
        X = np.concatenate([time_log10, ICs], axis=1)  # [log10(t), E0, S0, ES0, P0]
        
        print(f"✓ Loaded data: X={X.shape} [1 time + 4 ICs], y={y.shape}, gammas={gammas.shape}")
        print(f"✓ Applied log10 time transformation")
        print(f"ℹ️  Assuming gammas are already corrected (saved via HMM_clustering.py)")
        
        # Load transition matrix
        transition_matrix_path = data_path.replace('.npy', '_transition_matrix.npy')
        if os.path.exists(transition_matrix_path):
            transition_matrix = np.load(transition_matrix_path)
            print(f"✓ Loaded transition matrix: {transition_matrix.shape}")
        else:
            print(f"⚠️  Transition matrix not found: {transition_matrix_path}")
            transition_matrix = None
    
    return X, y, gammas, transition_matrix, time_raw  # RETURN raw_time

def train_epoch(student_model, dataloader, optimizer, loss_fn, device, hidden_layer='last'):
    """Train for one epoch - optimized"""
    student_model.train()
    total_loss = 0.0
    losses_dict = {'output': 0.0, 'hidden': 0.0, 'smoothness': 0.0}  # NEW
    transition_scores = []
    n_batches = 0
    
    for batch_data in dataloader:
        X_norm, y_teacher, teacher_hidden, phase_weights, gammas = batch_data  # NEW: unpack gammas
        
        # Single device transfer
        X_norm = X_norm.to(device)
        y_teacher = y_teacher.to(device)
        teacher_hidden = teacher_hidden.to(device)
        phase_weights = phase_weights.to(device)
        gammas = gammas.to(device)  # NEW
        
        # Forward pass
        student_outputs = student_model(X_norm)
        student_hidden = get_model_hidden_representation(student_model, X_norm, layer=hidden_layer)
        
        # Backward pass
        optimizer.zero_grad()
        total, output_loss, hidden_loss, smoothness, transition_score = loss_fn(
            student_outputs, y_teacher, student_hidden, teacher_hidden, phase_weights, gammas  # NEW
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
        for X_norm, y_teacher, _, _, _ in dataloader:  # FIXED: unpack 5 values (added one more _)
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
    
    # Vectorized RMSE
    rmse = np.sqrt((residuals**2).mean())
    
    return r2, r2_species, rmse

def save_figures(all_losses, student_name, r2_species, final_rmse):
    """Save publication-quality training loss curves"""
    os.makedirs('results/pakd', exist_ok=True)
    
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
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
        "pdf.fonttype": 42,
        "axes.linewidth": 1.2,
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
    ax1.set_title(f'Training Progression', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, loc='best')
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
    
    print(f"✓ Training loss curves saved")

# ---------------------------
# Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Phase-Aware Knowledge Distillation with Gamma Correction')
    
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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
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
    
    # HMM
    parser.add_argument('--run_hmm_if_missing', action='store_true', default=True,
                       help='Run HMM clustering if gammas are not in data file')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs('models/students', exist_ok=True)
    os.makedirs('results/pakd', exist_ok=True)
    device = get_device()
    
    # Load data with CORRECTED gammas (run HMM with correction if needed)
    X, y, gammas, transition_matrix, raw_time = load_data_with_gammas(  # UPDATED
        args.data, 
        run_hmm_if_missing=args.run_hmm_if_missing
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
    
    # Estimate timescales - FIX: use raw_time, not logged X[:, 0]
    if args.dt == 'auto':
        # Calculate from raw time differences
        dt = np.median(np.diff(np.unique(raw_time)))
    else:
        dt = float(args.dt)
    
    print(f"\nUsing dt = {dt:.3e} for timescale estimation")
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
        student_model, torch.randn(1, 5).to(device), layer=args.hidden_layer
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
    print(f"Using CORRECTED gammas (phase-aware weights)")
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
            r2, _, rmse = evaluate_model(student_model, dataloader, device)
            print(f"\nEpoch {epoch+1}: total={losses[0]:.6f}, out={losses[1]:.6f}, "
                  f"hidden={losses[2]:.6f}, smooth={losses[3]:.6f}, trans_score={losses[4]:.4f}, "
                  f"R²={r2:.6f}, RMSE={rmse:.6e}, LR={optimizer.param_groups[0]['lr']:.3e}")
    
    # Evaluate
    r2, r2_species, rmse = evaluate_model(student_model, dataloader, device)
    print(f"\n{'='*60}\nFinal: R²={r2:.6f}, RMSE={rmse:.6e}\n{'='*60}\n")
    
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
    save_figures(all_losses, student_name, r2_species, rmse)
    
    # Save model
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'model_type': args.student_type,
        'hidden_dim': args.student_hidden_dim,
        'num_blocks': args.student_num_blocks,
        'dropout': args.student_dropout,
        'projection_state_dict': loss_fn.projection.state_dict(),
        'teacher_model_path': args.teacher_model,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'training_args': vars(args),
        'hidden_layer': args.hidden_layer,
        'final_r2': r2,
        'r2_by_species': r2_species.tolist(),
        'final_rmse': rmse,
        'training_losses': all_losses,
        'input_size': 5,
        'output_size': 4,
        'use_time_features': True,
        'gamma_correction_applied': True,  # NEW: flag to indicate corrected gammas were used
    }, f'models/students/{student_name}.pt')
    
    print(f"\n{'='*60}")
    print(f"✓ Model saved: models/students/{student_name}.pt")
    print(f"✓ Training used CORRECTED gammas (phase label corrections applied)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()