"""
Train Teacher Model on MCF7 Signaling Data.

This script trains MLP or ResidualMLP models to predict phosphoprotein
responses from treatment conditions in the HPN-DREAM MCF7 dataset.
Uses raw data directly (no preprocessing, no aggregation, no scaling).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from models import MLP, ResidualMLP

# ============================================================================
# MCF7 Dataset Constants
# ============================================================================
MIDAS_TREATMENT_PREFIX = 'TR:'
MIDAS_DATA_AVG_PREFIX = 'DA:'
MIDAS_DATA_VAL_PREFIX = 'DV:'


# ============================================================================
# Dataset
# ============================================================================
class MCF7Dataset(Dataset):
    """Dataset for MCF7 signaling data in MIDAS format."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, conditions: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.conditions = conditions if conditions is not None else np.zeros(len(X))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.conditions[idx]


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
        (X, y, column_info) where X is inputs, y is outputs
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
    
    return X, y, column_info


# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Standard training epoch."""
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
        total_rmse += torch.sqrt(torch.mean((outputs - targets)**2)).item() * inputs.size(0)
    
    dataset_size = len(dataloader.dataset)
    return total_loss / dataset_size, total_rmse / dataset_size


# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_model(model, dataloader, device):
    """Comprehensive model evaluation."""
    model.eval()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())
    
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    # RMSE by protein
    rmse_by_protein = torch.sqrt(torch.mean((all_targets - all_outputs)**2, dim=0))
    
    # R² by protein
    ss_res = torch.sum((all_targets - all_outputs)**2, dim=0)
    ss_tot = torch.sum((all_targets - all_targets.mean(dim=0))**2, dim=0)
    r2_by_protein = 1 - ss_res / (ss_tot + 1e-8)
    
    return {
        'overall_rmse': torch.mean(rmse_by_protein).item(),
        'overall_r2': torch.mean(r2_by_protein).item(),
        'rmse_by_protein': rmse_by_protein.tolist(),
        'r2_by_protein': r2_by_protein.tolist(),
    }


# ============================================================================
# Plotting Functions
# ============================================================================
def plot_training_history(train_losses, train_rmses, model_name, output_dir='results'):
    """Publication-quality training history plots."""
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
    
    plt.figure(figsize=(6, 5))
    plt.plot(train_losses, color=colors[0], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_training_loss.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/{model_name}_training_loss.png', bbox_inches='tight', dpi=600)
    plt.close()
    
    plt.figure(figsize=(6, 5))
    plt.plot(train_rmses, color=colors[1], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.yscale('log')
    plt.title('Training RMSE')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_training_rmse.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/{model_name}_training_rmse.png', bbox_inches='tight', dpi=600)
    plt.close()
    
    print(f"✓ Plots saved to {output_dir}/{model_name}_*.pdf/png")


def plot_protein_errors(error_results, protein_names, model_name, output_dir='results'):
    """Plot per-protein RMSE and R²."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    rmse_values = error_results['rmse_by_protein']
    sorted_idx = np.argsort(rmse_values)[::-1]
    
    ax1 = axes[0]
    ax1.barh(range(len(rmse_values)), [rmse_values[i] for i in sorted_idx], color='#3498db')
    ax1.set_yticks(range(len(rmse_values)))
    ax1.set_yticklabels([protein_names[i] for i in sorted_idx], fontsize=8)
    ax1.set_xlabel('RMSE')
    ax1.set_title('RMSE by Protein')
    ax1.invert_yaxis()
    
    ax2 = axes[1]
    r2_values = error_results['r2_by_protein']
    colors = ['#2ecc71' if r2 > 0.5 else '#e74c3c' for r2 in [r2_values[i] for i in sorted_idx]]
    ax2.barh(range(len(r2_values)), [r2_values[i] for i in sorted_idx], color=colors)
    ax2.set_yticks(range(len(r2_values)))
    ax2.set_yticklabels([protein_names[i] for i in sorted_idx], fontsize=8)
    ax2.set_xlabel('R²')
    ax2.set_title('R² by Protein')
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_protein_errors.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/{model_name}_protein_errors.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Protein error plot saved")


# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train teacher model on MCF7 signaling data (raw mode)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model
    parser.add_argument('--model', type=str, default='ResidualMLP', 
                       choices=['MLP', 'ResidualMLP'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Training
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stop_patience', type=int, default=300)
    
    # Data
    parser.add_argument('--data_path', type=str, 
                       default='experimental/MIDAS/MD_MCF7_main.csv',
                       help='Path to MIDAS data file')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--model_dir', type=str, default='models')
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seeds
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
    
    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("Loading MCF7 MIDAS Data")
    print("=" * 70)
    
    X, y, column_info = load_midas_data(args.data_path)
    
    print(f"  Data path: {args.data_path}")
    print(f"  Samples: {len(X)}")
    print(f"  Input features: {X.shape[1]}")
    print(f"    - Cell line: {len(column_info['cell_line_cols'])}")
    print(f"    - Stimuli: {len(column_info['stimuli_cols'])} ({column_info['stimuli_names']})")
    print(f"    - Inhibitors: {len(column_info['inhibitor_cols'])} ({column_info['inhibitor_names']})")
    print(f"    - Time: {len(column_info['da_cols'])}")
    print(f"  Output proteins: {y.shape[1]}")
    
    X_final = X.astype(np.float32)
    y_final = y.astype(np.float32)
    
    print(f"\n  Input statistics:")
    print(f"    Range: [{X_final.min():.4f}, {X_final.max():.4f}]")
    print(f"    Mean: {X_final.mean():.4f}")
    print(f"  Target statistics:")
    print(f"    Range: [{y_final.min():.4f}, {y_final.max():.4f}]")
    print(f"    Mean: {y_final.mean():.4f}")
    print(f"    Std:  {y_final.std():.4f}")
    
    # ========================================================================
    # Create DataLoader
    # ========================================================================
    dataset = MCF7Dataset(X_final, y_final)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    # ========================================================================
    # Create Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("Model Configuration")
    print("=" * 70)
    
    actual_input_size = X_final.shape[1]
    actual_output_size = y_final.shape[1]
    
    if args.model == 'ResidualMLP':
        model = ResidualMLP(
            input_size=actual_input_size,
            output_size=actual_output_size,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_layers,
            dropout=args.dropout
        )
    else:
        model = MLP(
            input_size=actual_input_size,
            output_size=actual_output_size,
            hidden_sizes=[args.hidden_dim] * args.num_layers,
            dropout=args.dropout
        )
    
    model = model.to(device)
    
    print(f"  Model: {args.model}")
    print(f"  Input size: {actual_input_size}")
    print(f"  Output size: {actual_output_size}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers/Blocks: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model_name = f"{args.model}_MCF7_{len(X_final)}samples_raw"

    # ========================================================================
    # Training
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50)
    
    train_losses = []
    train_rmses = []
    best_loss = float('inf')
    patience_counter = 0
    
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
                'column_info': column_info,
                'train_losses': train_losses,
                'train_rmses': train_rmses,
                'model_type': args.model,
                'epoch': epoch,
                'args': vars(args),
            }, f'{args.model_dir}/{model_name}_best.pt')
        else:
            patience_counter += 1
        
        scheduler.step(loss)
        
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss={loss:.4e}, RMSE={rmse:.4e}, "
                  f"lr={optimizer.param_groups[0]['lr']:.3e}")
    
    # ========================================================================
    # Load Best Model and Evaluate
    # ========================================================================
    best_path = f'{args.model_dir}/{model_name}_best.pt'
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("\n✓ Loaded best model")
    
    print("\nEvaluating...")
    error_results = evaluate_model(model, dataloader, device)
    
    # ========================================================================
    # Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Overall RMSE: {error_results['overall_rmse']:.6f}")
    print(f"  Overall R²: {error_results['overall_r2']:.4f}")
    
    r2_sorted = sorted(zip(column_info['protein_names'], error_results['r2_by_protein']),
                       key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 best predicted proteins (by R²):")
    for name, r2 in r2_sorted[:5]:
        print(f"    {name}: R²={r2:.4f}")
    
    print(f"\n  Top 5 worst predicted proteins (by R²):")
    for name, r2 in r2_sorted[-5:]:
        print(f"    {name}: R²={r2:.4f}")
    
    # ========================================================================
    # Save Plots and Final Model
    # ========================================================================
    plot_training_history(train_losses, train_rmses, model_name, args.output_dir)
    plot_protein_errors(error_results, column_info['protein_names'], model_name, args.output_dir)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'column_info': column_info,
        'error_results': error_results,
        'model_type': args.model,
        'train_losses': train_losses,
        'train_rmses': train_rmses,
        'args': vars(args),
    }, f'{args.model_dir}/{model_name}_final.pt')
    
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print(f"  Model: {args.model_dir}/{model_name}_final.pt")
    print(f"  Plots: {args.output_dir}/{model_name}_*.pdf")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()