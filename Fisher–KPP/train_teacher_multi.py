import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from models import MLP, ResidualMLP

# ============================================================================
# Dataset
# ============================================================================
class MultiConditionReactionDataset(Dataset):
    """Dataset for multi-condition Fisher-KPP trajectories"""
    def __init__(self, X, y, condition_labels=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Infer condition labels if not provided
        if condition_labels is None:
            initial_conditions = X[:, 1:]  # u0_1 to u0_n (skip time)
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


# ============================================================================
# Feature Engineering
# ============================================================================
def create_time_features(time_array):
    """Create log10 time feature for better temporal representation"""
    t = time_array
    
    # Single time transformation - log scale (good for exponential dynamics)
    t1 = np.log10(t + 1.0)
    
    return t1.reshape(-1, 1)  # Shape: (n_samples, 1)


# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Standard training epoch"""
    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    
    for inputs, targets, _ in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        # Compute RMSE
        total_rmse += torch.sqrt(torch.mean((outputs - targets)**2)).item() * inputs.size(0)
    
    dataset_size = len(dataloader.dataset)
    return total_loss / dataset_size, total_rmse / dataset_size


# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_model(model, dataloader, device):
    """Comprehensive model evaluation"""
    model.eval()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets, conditions in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())
    
    # Overall metrics
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    # Compute RMSE by grid point
    rmse_by_grid = torch.sqrt(torch.mean((all_targets - all_outputs)**2, dim=0))
    
    return {
        'overall_rmse': torch.mean(rmse_by_grid).item(),
        'rmse_by_grid': rmse_by_grid.tolist(),
    }


# ============================================================================
# Plotting Functions
# ============================================================================
def plot_training_history(train_losses, train_rmses, error_results, model_name):
    """Publication-quality training history plots"""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    
    # Training Loss
    plt.figure(figsize=(6, 5))
    plt.plot(train_losses, color=colors[0], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training_loss.pdf', bbox_inches='tight')
    plt.savefig(f'results/{model_name}_training_loss.png', bbox_inches='tight', dpi=600)
    plt.close()
    
    # Training RMSE
    plt.figure(figsize=(6, 5))
    plt.plot(train_rmses, color=colors[1], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.yscale('log')
    plt.title('Training RMSE')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training_rmse.pdf', bbox_inches='tight')
    plt.savefig(f'results/{model_name}_training_rmse.png', bbox_inches='tight', dpi=600)
    plt.close()
    
    print(f"✓ Plots saved to results/{model_name}_*.pdf/png")


# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train teacher model on Fisher-KPP dataset')
    
    # Model
    parser.add_argument('--model', type=str, default='ResidualMLP', 
                       choices=['MLP', 'ResidualMLP'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stop_patience', type=int, default=500)
    
    # Data
    parser.add_argument('--file', type=str, default=None, help='Data file path')
    parser.add_argument('--n_grid', type=int, default=100, help='Number of interior grid points')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load data
    print("\nLoading Fisher-KPP dataset...")
    if args.file is None:
        data_files = [f for f in os.listdir('data/fisher_kpp') 
                     if f.startswith('teacher_fisher_kpp') and f.endswith('.npy')]
        if not data_files:
            raise FileNotFoundError("No Fisher-KPP dataset found. Run Fisher_KPP_simulation.py first.")
        data_path = os.path.join('data/fisher_kpp', sorted(data_files)[-1])
    else:
        data_path = args.file
    
    print(f"  File: {data_path}")
    data = np.load(data_path)
    print(f"  Shape: {data.shape}")
    
    # Extract features and targets
    # Data format: [time, u0_1, ..., u0_n, u_1, ..., u_n]
    n_grid = args.n_grid
    X = data[:, 0:n_grid+1]        # time + n_grid initial conditions
    y = data[:, n_grid+1:2*n_grid+1]  # n_grid solution values
    
    unique_conditions = np.unique(X[:, 1:n_grid+1], axis=0)
    print(f"  Grid points: {n_grid}")
    print(f"  Conditions: {len(unique_conditions)}")
    print(f"  Samples: {len(X)}")
    
    # Preprocessing
    print("\nPreprocessing:")
    X_copy = X.copy()

    # Create 1 time feature from single time column
    time_features = create_time_features(X_copy[:, 0])  # Shape: (n_samples, 1)
    print(f"  ✓ Created 1 time feature: log10(t)")

    # Combine time features with initial conditions
    X_augmented = np.column_stack([
        time_features,           # 1 time feature
        X_copy[:, 1:n_grid+1]    # n_grid initial conditions (unchanged)
    ])  # Total: n_grid + 1 features

    print(f"  Input shape: {X.shape} → {X_augmented.shape}")

    # Normalize all features
    X_scaler = StandardScaler()
    X_norm = X_scaler.fit_transform(X_augmented)
    print(f"  ✓ Normalized all {n_grid + 1} features")
    
    # Check for negative values in targets
    print(f"\nTarget statistics:")
    print(f"  Shape: {y.shape}")
    print(f"  Range: [{y.min():.4e}, {y.max():.4e}]")
    print(f"  Mean: {y.mean():.4e}")
    
    # Count negative/zero values
    n_negative = np.sum(y < 0)
    n_zero = np.sum(y == 0)
    n_small = np.sum((y > 0) & (y < 1e-15))
    
    if n_negative > 0:
        print(f"  ⚠️  WARNING: {n_negative} negative values detected!")
        print(f"     Most negative: {y.min():.4e}")
    if n_zero > 0:
        print(f"  ⚠️  WARNING: {n_zero} exact zeros detected!")
    if n_small > 0:
        print(f"  ℹ️  {n_small} very small positive values (< 1e-15)")
    
    # Handle negatives (clip to zero - concentrations should be non-negative)
    if n_negative > 0:
        print(f"\n  Processing negative values...")
        print(f"     Original range: [{y.min():.4e}, {y.max():.4e}]")
        y_processed = np.maximum(y, 0.0)
        n_clipped = np.sum(y < 0)
        print(f"     ✓ Clipped {n_clipped} negative values to 0.0")
        print(f"     New range: [{y_processed.min():.4e}, {y_processed.max():.4e}]")
    else:
        y_processed = y
        print(f"  ✓ Original scale targets (no negatives)")
    
    # Normalize targets (y)
    y_scaler = StandardScaler()
    y_norm = y_scaler.fit_transform(y_processed)
    print(f"  ✓ Normalized targets ({n_grid} grid points)")
    print(f"    Normalized range: [{y_norm.min():.4f}, {y_norm.max():.4f}]")
    print(f"    Normalized mean: {y_norm.mean():.4f} (≈0)")
    print(f"    Normalized std: {y_norm.std():.4f} (≈1)")
    
    # Create dataset
    initial_conditions = X[:, 1:n_grid+1]
    _, condition_labels = np.unique(initial_conditions, axis=0, return_inverse=True)
    
    dataset = MultiConditionReactionDataset(X_norm, y_norm, condition_labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                           pin_memory=True, num_workers=4)
    
    # Create model
    input_size = n_grid + 1  # 1 time feature + n_grid initial conditions
    output_size = n_grid     # n_grid solution values
    
    print(f"\nModel: {args.model}")
    if args.model == 'ResidualMLP':
        model = ResidualMLP(
            input_size=input_size,
            output_size=output_size, 
            hidden_dim=args.hidden_dim, 
            num_blocks=args.num_layers,
            dropout=0.0
        )
        print(f"  Architecture: ResidualMLP")
        print(f"  Blocks: {args.num_layers}, hidden_dim: {args.hidden_dim}")
    else:  # MLP
        model = MLP(
            input_size=input_size,
            output_size=output_size, 
            hidden_sizes=[args.hidden_dim] * args.num_layers,
            dropout=0.0
        )
        print(f"  Architecture: MLP")
        print(f"  Layers: {args.num_layers}, hidden_dim: {args.hidden_dim}")
    
    print(f"  Input size: {input_size} (1 time + {n_grid} ICs)")
    print(f"  Output size: {output_size} ({n_grid} grid points)")
    
    model = model.to(device)
    
    # Loss function - always MSE (targets are normalized)
    criterion = nn.MSELoss()
    print(f"  Loss: MSE (on normalized targets)")
    
    # Model name
    model_name = f"{args.model}_fisher_kpp_{len(unique_conditions)}cond_n{n_grid}"
    
    # Training
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50)
    
    train_losses = []
    train_rmses = []
    best_loss = float('inf')
    patience_counter = 0
    
    # Standard training loop
    for epoch in tqdm(range(args.epochs), desc="Training"):
        loss, rmse = train_epoch(model, dataloader, optimizer, criterion, device)
        train_losses.append(loss)
        train_rmses.append(rmse)
        
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'train_losses': train_losses,
                'train_rmses': train_rmses,
                'model_type': args.model,
                'epoch': epoch,
                'n_grid': n_grid,
            }, f'models/{model_name}_best.pt')
        else:
            patience_counter += 1
        
        scheduler.step(loss)
        
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={loss:.4e}, RMSE={rmse:.4e}, "
                  f"lr={optimizer.param_groups[0]['lr']:.3e}")
    
    # Load best and evaluate
    if os.path.exists(f'models/{model_name}_best.pt'):
        checkpoint = torch.load(f'models/{model_name}_best.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("\n✓ Loaded best model")
    
    print("\nEvaluating...")
    error_results = evaluate_model(model, dataloader, device)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Overall RMSE: {error_results['overall_rmse']:.6e}")
    print(f"  RMSE by grid point (first 5): {[f'{x:.4e}' for x in error_results['rmse_by_grid'][:5]]}...")
    print(f"  RMSE by grid point (last 5): {[f'{x:.4e}' for x in error_results['rmse_by_grid'][-5:]]}...")
    
    # Save plots
    plot_training_history(train_losses, train_rmses, error_results, model_name)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'error_results': error_results,
        'model_type': args.model,
        'train_losses': train_losses,
        'train_rmses': train_rmses,
        'input_size': input_size,
        'output_size': output_size,
        'n_grid': n_grid,
        'use_time_features': True,
        'time_feature_types': ['log10'],
    }, f'models/{model_name}_final.pt')
    
    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"  Model: models/{model_name}_final.pt")
    print(f"  Plots: results/{model_name}_*.pdf")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()