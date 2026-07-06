import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from models import MLP, ResidualMLP, GradientBoostedNN

class InternalKDLoss(nn.Module):
    """Internal Knowledge Distillation Loss - Last block teaches middle block"""
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha    # Weight for supervision loss (final output vs targets)
        self.beta = beta      # Weight for internal distillation loss (middle block learns from last block)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, final_outputs, middle_block_output, last_block_output, targets):
        """
        Args:
            final_outputs: Final model predictions
            middle_block_output: Output from middle block (student)
            last_block_output: Output from last block (teacher)
            targets: Ground truth targets
        """
        # Standard supervision loss (final output vs targets)
        supervision_loss = self.mse_loss(final_outputs, targets)
        
        # Internal distillation loss (middle block learns from last block)
        # Temperature scaling for soft targets
        soft_teacher = last_block_output / self.temperature
        soft_student = middle_block_output / self.temperature
        
        internal_kd_loss = self.mse_loss(soft_student, soft_teacher.detach()) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * supervision_loss + self.beta * internal_kd_loss
        
        return total_loss, internal_kd_loss, supervision_loss

class AlternatingTrainingLoss(nn.Module):
    """Wrapper for alternating between KD and supervision training"""
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.internal_kd_loss = InternalKDLoss(temperature, alpha, beta)
        self.supervision_loss = nn.MSELoss()
        
    def forward(self, final_outputs, middle_block_output, last_block_output, targets, mode='supervision'):
        if mode == 'internal_kd':
            return self.internal_kd_loss(final_outputs, middle_block_output, last_block_output, targets)
        else:  # supervision mode
            supervision_loss = self.supervision_loss(final_outputs, targets)
            return supervision_loss, torch.tensor(0.0, device=targets.device), supervision_loss

class InternalKDModel(nn.Module):
    """Wrapper to extract middle and last block features for internal distillation"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.model_type = base_model.__class__.__name__
        
        # Store features from middle and last blocks
        self.middle_block_output = None
        self.last_block_output = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture middle and last block outputs"""
        if self.model_type == 'ResidualMLP':
            # For ResidualMLP, hook into middle and last residual blocks
            total_blocks = len(self.base_model.blocks)
            if total_blocks >= 3:  # Need at least 3 blocks for meaningful middle
                # Middle block (student) - use block at 1/2 position
                middle_idx = total_blocks // 2
                middle_hook = self.base_model.blocks[middle_idx].register_forward_hook(
                    lambda module, input, output: self._hook_middle_block(output)
                )
                self.hooks.append(middle_hook)
                
                # Last block (teacher)
                last_hook = self.base_model.blocks[-1].register_forward_hook(
                    lambda module, input, output: self._hook_last_block(output)
                )
                self.hooks.append(last_hook)
                
                print(f"Registered hooks for ResidualMLP: middle block ({middle_idx}) and last block ({total_blocks-1})")
            else:
                print("Warning: ResidualMLP has less than 3 blocks, cannot perform meaningful internal KD")
                    
        elif self.model_type == 'MLP':
            # For MLP, hook into middle and last hidden layers
            total_layers = len(self.base_model.layers)
            if total_layers >= 4:  # At least input, middle, penultimate, output
                # Middle hidden layer (student) - layers at 1/2 position
                middle_idx = total_layers // 2
                middle_hook = self.base_model.layers[middle_idx].register_forward_hook(
                    lambda module, input, output: self._hook_middle_block(output)
                )
                self.hooks.append(middle_hook)
                
                # Last hidden layer (teacher) - layers[-2] since layers[-1] is output
                last_hook = self.base_model.layers[-2].register_forward_hook(
                    lambda module, input, output: self._hook_last_block(output)
                )
                self.hooks.append(last_hook)
                
                print(f"Registered hooks for MLP: middle hidden layer ({middle_idx}) and last hidden layer ({total_layers-2})")
            else:
                print("Warning: MLP has less than 4 layers, cannot perform meaningful internal KD")
        
        else:
            print(f"Warning: Model type {self.model_type} not supported for internal KD")
    
    def _hook_middle_block(self, output):
        """Hook function to store middle block output"""
        self.middle_block_output = output.clone()
    
    def _hook_last_block(self, output):
        """Hook function to store last block output"""
        self.last_block_output = output.clone()
    
    def forward(self, x):
        # Clear previous outputs
        self.middle_block_output = None
        self.last_block_output = None
        
        # Forward pass through base model (hooks will capture middle and last block outputs)
        final_output = self.base_model(x)
        
        return final_output, self.middle_block_output, self.last_block_output
    
    def remove_hooks(self):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class MultiConditionReactionDataset(Dataset):
    def __init__(self, X, y, condition_labels=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        if condition_labels is None:
            initial_conditions = X[:, 1:5]
            unique_conditions, labels = np.unique(initial_conditions, axis=0, return_inverse=True)
            self.condition_labels = labels
            self.num_conditions = len(unique_conditions)
        else:
            self.condition_labels = condition_labels
            self.num_conditions = len(np.unique(condition_labels))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.condition_labels[idx]

def load_teacher_model(teacher_path, device):
    """Load teacher model from checkpoint"""
    print(f"Loading teacher model from: {teacher_path}")
    checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', 'MLP')
    state_dict = checkpoint['model_state_dict']

    # Create model based on type
    if model_type == 'ResidualMLP':
        base_model = ResidualMLP(input_size=5, output_size=4, hidden_dim=512, num_blocks=32)
    elif model_type == 'GBNN':
        base_model = GradientBoostedNN(input_size=5, output_size=4, n_estimators=100)
    else:  # MLP
        base_model = MLP(input_size=5, output_size=4, hidden_sizes=[512] * 32)

    base_model.load_state_dict(state_dict)
    
    # Wrap in InternalKDModel for feature extraction
    teacher_model = InternalKDModel(base_model)
    teacher_model = teacher_model.to(device)
    
    return (teacher_model, checkpoint.get('X_scaler', None), 
            checkpoint.get('y_scaler', None), checkpoint.get('log_transform_y', False))

def train_epoch_alternating(model, dataloader, kd_optimizer, sup_optimizer, loss_fn, device, mode='supervision'):
    """Training epoch with alternating between KD and supervision"""
    model.train()
    
    # Choose optimizer based on mode
    if mode == 'internal_kd':
        optimizer = kd_optimizer
    else:
        optimizer = sup_optimizer
    
    running_total_loss = 0.0
    running_internal_kd_loss = 0.0
    running_supervision_loss = 0.0
    running_mae = 0.0
    
    for inputs, targets, conditions in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with middle and last block feature extraction
        final_outputs, middle_block_output, last_block_output = model(inputs)
        
        # Check if we have both middle and last block outputs for KD mode
        if mode == 'internal_kd' and (middle_block_output is None or last_block_output is None):
            print("Warning: Could not extract middle/last block outputs, switching to supervision mode")
            mode = 'supervision'
        
        # Compute loss based on mode
        total_loss, internal_kd_loss, supervision_loss = loss_fn(
            final_outputs, middle_block_output, last_block_output, targets, mode=mode
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_size = inputs.size(0)
        running_total_loss += total_loss.item() * batch_size
        running_internal_kd_loss += internal_kd_loss.item() * batch_size
        running_supervision_loss += supervision_loss.item() * batch_size
        running_mae += torch.abs(final_outputs - targets).mean().item() * batch_size
    
    dataset_size = len(dataloader.dataset)
    return (running_total_loss / dataset_size, 
            running_internal_kd_loss / dataset_size,
            running_supervision_loss / dataset_size,
            running_mae / dataset_size)

def evaluate_model_internal_kd(model, dataloader, device):
    """Evaluate model with internal KD"""
    model.eval()
    
    all_targets, all_outputs = [], []
    
    with torch.no_grad():
        for inputs, targets, conditions in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            final_outputs, _, _ = model(inputs)  # We don't need block outputs for evaluation
            
            all_targets.append(targets.cpu())
            all_outputs.append(final_outputs.cpu())
    
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    # Calculate metrics
    mae = torch.mean(torch.abs(all_targets - all_outputs)).item()
    rmse = torch.sqrt(torch.mean((all_targets - all_outputs)**2)).item()
    
    return {'mae': mae, 'rmse': rmse}

def plot_alternating_training(train_losses_kd, train_losses_sup, internal_kd_losses, 
                             supervision_losses, train_maes_kd, train_maes_sup, model_name):
    """Plot alternating training progress"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Total losses comparison
    epochs_kd = range(0, len(train_losses_kd) * 2, 2)  # Even epochs
    epochs_sup = range(1, len(train_losses_sup) * 2, 2)  # Odd epochs
    
    axes[0, 0].plot(epochs_kd, train_losses_kd, 'r-', label='Internal KD Epochs', marker='o', markersize=3)
    axes[0, 0].plot(epochs_sup, train_losses_sup, 'b-', label='Supervision Epochs', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss by Mode')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss components
    axes[0, 1].plot(internal_kd_losses, 'r-', label='Internal KD Loss\n(Middle ← Last Block)')
    axes[0, 1].plot(supervision_losses, 'g-', label='Supervision Loss')
    axes[0, 1].set_xlabel('Epoch (KD mode only)')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components (KD Epochs)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # MAE comparison
    axes[0, 2].plot(epochs_kd, train_maes_kd, 'r-', label='Internal KD MAE', marker='o', markersize=3)
    axes[0, 2].plot(epochs_sup, train_maes_sup, 'b-', label='Supervision MAE', marker='s', markersize=3)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].set_title('MAE by Training Mode')
    axes[0, 2].set_yscale('log')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Combined loss trend
    all_epochs = list(range(max(len(train_losses_kd), len(train_losses_sup)) * 2))
    combined_losses = []
    for i in all_epochs:
        if i % 2 == 0 and i // 2 < len(train_losses_kd):  # KD epoch
            combined_losses.append(train_losses_kd[i // 2])
        elif i % 2 == 1 and i // 2 < len(train_losses_sup):  # Supervision epoch
            combined_losses.append(train_losses_sup[i // 2])
    
    axes[1, 0].plot(combined_losses, 'k-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Combined Training Progress')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Learning rate effect visualization
    if len(train_losses_kd) > 0 and len(train_losses_sup) > 0:
        kd_improvement = [(train_losses_kd[0] - loss) / train_losses_kd[0] * 100 for loss in train_losses_kd]
        sup_improvement = [(train_losses_sup[0] - loss) / train_losses_sup[0] * 100 for loss in train_losses_sup]
        
        axes[1, 1].plot(kd_improvement, 'r-', label='KD Mode Improvement', linewidth=2)
        axes[1, 1].plot(sup_improvement, 'b-', label='Supervision Mode Improvement', linewidth=2)
        axes[1, 1].set_xlabel('Epoch (within mode)')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('Training Effectiveness by Mode')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Training mode indicator
    axes[1, 2].text(0.1, 0.8, f"Training Pattern:", transform=axes[1, 2].transAxes, fontsize=12, weight='bold')
    axes[1, 2].text(0.1, 0.7, f"Even epochs: Internal KD", transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].text(0.1, 0.6, f"Odd epochs: Supervision", transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].text(0.1, 0.5, f"Strategy: Middle ← Last Block", transform=axes[1, 2].transAxes, fontsize=10, weight='bold')
    axes[1, 2].text(0.1, 0.3, f"Total KD epochs: {len(train_losses_kd)}", transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].text(0.1, 0.2, f"Total SUP epochs: {len(train_losses_sup)}", transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].set_title('Training Configuration')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_alternating_training.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Alternating Internal KD and Supervision Training - Middle Block as Student')
    parser.add_argument('--teacher_path', type=str, required=True, help='Path to saved teacher model')
    parser.add_argument('--data_file', type=str, default=None, help='Training data file')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs (total)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--kd_lr', type=float, default=1e-7, help='Learning rate for KD training (lower)')
    parser.add_argument('--sup_lr', type=float, default=1e-7, help='Learning rate for supervision training')
    parser.add_argument('--temperature', type=float, default=1.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.2, help='Weight for supervision loss in KD mode')
    parser.add_argument('--beta', type=float, default=0.8, help='Weight for internal KD loss')
    parser.add_argument('--kd_frequency', type=int, default=1, help='Do KD every N epochs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = parser.parse_args()
    
    # Setup
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    
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
    
    # Load teacher model (which becomes the model to improve)
    model, X_scaler, y_scaler, log_transform_y = load_teacher_model(args.teacher_path, device)
    model_type = model.base_model.__class__.__name__
    
    print(f"Loaded model: {model_type}")
    print(f"Log transform y: {log_transform_y}")
    print(f"Number of hooks registered: {len(model.hooks)}")
    
    # Load training data
    if args.data_file is None:
        data_files = [f for f in os.listdir('data/teacher') if f.startswith('teacher_combined_multiple') and f.endswith('.npy')]
        if not data_files:
            raise FileNotFoundError("No multiple condition dataset found.")
        data_path = os.path.join('data/teacher', sorted(data_files)[-1])
    else:
        data_path = args.data_file
    
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    
    X = data[:, 0:5]
    y = data[:, 5:9]
    
    # Apply same preprocessing as original model
    if X_scaler is not None:
        # Check if we need to log-transform time
        if hasattr(X_scaler, 'mean_') and X_scaler.mean_[0] < 0:  # Likely log-transformed
            X_copy = X.copy()
            X_copy[:, 0] = np.log10(X_copy[:, 0] + 1e-20)
            X_norm = X_scaler.transform(X_copy)
            print("Applied log-transform and scaling to time feature in X")
        else:
            X_norm = X_scaler.transform(X)
    else:
        X_norm = X
    
    if log_transform_y:
        y_processed = np.log10(y + 1e-20)
    else:
        y_processed = y
    
    # Create dataset
    initial_conditions = X[:, 1:5]
    unique_conditions, condition_labels = np.unique(initial_conditions, axis=0, return_inverse=True)
    
    dataset = MultiConditionReactionDataset(X_norm, y_processed, condition_labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    print(f"Training on {len(dataset)} samples from {len(unique_conditions)} conditions")
    
    # Get baseline performance
    print("\nEvaluating baseline performance...")
    baseline_results = evaluate_model_internal_kd(model, dataloader, device)
    print(f"Baseline MAE: {baseline_results['mae']:.6e}")
    print(f"Baseline RMSE: {baseline_results['rmse']:.6e}")
    
    # Training setup with different optimizers
    model_name = f"AlternatingKD_Middle_{model_type}_T{args.temperature}_KDlr{args.kd_lr}_SUPlr{args.sup_lr}"
    
    # Create separate optimizers with different learning rates
    kd_optimizer = optim.Adam(model.parameters(), lr=args.kd_lr, weight_decay=1e-6)
    sup_optimizer = optim.Adam(model.parameters(), lr=args.sup_lr, weight_decay=1e-6)
    
    # Schedulers for both optimizers
    kd_scheduler = ReduceLROnPlateau(kd_optimizer, 'min', factor=0.5, patience=5, verbose=True)
    sup_scheduler = ReduceLROnPlateau(sup_optimizer, 'min', factor=0.5, patience=5, verbose=True)
    
    loss_fn = AlternatingTrainingLoss(temperature=args.temperature, alpha=args.alpha, beta=args.beta)
    
    # Training loop with alternating modes
    print(f"\nStarting alternating training...")
    print(f"Strategy: Alternate between Internal KD and Supervision")
    print(f"KD LR: {args.kd_lr}, Supervision LR: {args.sup_lr}")
    print(f"Temperature: {args.temperature}, Alpha: {args.alpha}, Beta: {args.beta}")
    print(f"KD frequency: every {args.kd_frequency} epochs")
    
    train_losses_kd, train_losses_sup = [], []
    internal_kd_losses, supervision_losses = [], []
    train_maes_kd, train_maes_sup = [], []
    best_mae = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(args.epochs), desc="Alternating Training"):
        # Determine training mode based on epoch
        if epoch % args.kd_frequency == 0:
            mode = 'internal_kd'
            optimizer_to_use = kd_optimizer
            scheduler_to_use = kd_scheduler
        else:
            mode = 'supervision'
            optimizer_to_use = sup_optimizer
            scheduler_to_use = sup_scheduler
        
        # Train one epoch
        total_loss, internal_kd_loss, supervision_loss, train_mae = train_epoch_alternating(
            model, dataloader, kd_optimizer, sup_optimizer, loss_fn, device, mode=mode
        )
        
        # Store results based on mode
        if mode == 'internal_kd':
            train_losses_kd.append(total_loss)
            train_maes_kd.append(train_mae)
            internal_kd_losses.append(internal_kd_loss)
            supervision_losses.append(supervision_loss)
        else:
            train_losses_sup.append(total_loss)
            train_maes_sup.append(train_mae)
        
        # Update appropriate scheduler
        scheduler_to_use.step(train_mae)
        
        # Save best model based on MAE
        if train_mae < best_mae:
            best_mae = train_mae
            patience_counter = 0
            torch.save({
                'model_state_dict': model.base_model.state_dict(),
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'log_transform_y': log_transform_y,
                'model_type': model_type,
                'epoch': epoch,
                'mae': best_mae,
                'baseline_results': baseline_results,
                'training_params': {
                    'temperature': args.temperature,
                    'alpha': args.alpha,
                    'beta': args.beta,
                    'kd_lr': args.kd_lr,
                    'sup_lr': args.sup_lr,
                    'kd_frequency': args.kd_frequency,
                    'strategy': 'alternating_kd_supervision'
                }
            }, f'models/{model_name}_best.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= 40:  # Early stopping
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 1 == 0:
            current_results = evaluate_model_internal_kd(model, dataloader, device)
            improvement = (baseline_results['mae'] - current_results['mae']) / baseline_results['mae'] * 100
            print(f"Epoch {epoch+1} ({mode}): Loss={total_loss:.4e}, "
                  f"MAE={current_results['mae']:.4e} ({improvement:+.2f}%), "
                  f"LR_KD={kd_optimizer.param_groups[0]['lr']:.2e}, LR_SUP={sup_optimizer.param_groups[0]['lr']:.2e}")
    
    # Load best model for final evaluation
    if os.path.exists(f'models/{model_name}_best.pt'):
        checkpoint = torch.load(f'models/{model_name}_best.pt')
        model.base_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model for final evaluation")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_results = evaluate_model_internal_kd(model, dataloader, device)
    improvement_mae = (baseline_results['mae'] - final_results['mae']) / baseline_results['mae'] * 100
    improvement_rmse = (baseline_results['rmse'] - final_results['rmse']) / baseline_results['rmse'] * 100
    
    print(f"\nAlternating Training Results:")
    print(f"Baseline MAE: {baseline_results['mae']:.6e}")
    print(f"Final MAE: {final_results['mae']:.6e}")
    print(f"MAE Improvement: {improvement_mae:+.2f}%")
    print(f"RMSE Improvement: {improvement_rmse:+.2f}%")
    print(f"KD epochs: {len(train_losses_kd)}")
    print(f"Supervision epochs: {len(train_losses_sup)}")
    
    # Plot results
    plot_alternating_training(train_losses_kd, train_losses_sup, internal_kd_losses, 
                             supervision_losses, train_maes_kd, train_maes_sup, model_name)
    
    # Clean up hooks
    model.remove_hooks()
    
    print(f"\nAlternating Training completed!")
    print(f"Strategy: Internal KD - Middle Block (student) ← Last Block (teacher)")
    print(f"Learning Rates: KD={args.kd_lr} ↔ Supervision={args.sup_lr}")
    print(f"Best model saved: models/{model_name}_best.pt")
    print(f"Training plots saved: results/{model_name}_alternating_training.png")
    
    if improvement_mae > 0:
        print(f"🎉 Middle-block alternating training was successful! Achieved {improvement_mae:.2f}% improvement in MAE")
    else:
        print("⚠️ Alternating training did not improve performance. Consider adjusting hyperparameters.")

if __name__ == "__main__":
    main()