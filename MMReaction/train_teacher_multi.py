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

# Unified dataset class that handles training data
class MultiConditionReactionDataset(Dataset):
    def __init__(self, X, y, condition_labels=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Infer condition labels if not provided
        if condition_labels is None:
            initial_conditions = X[:, 1:5]  # E0, S0, ES0, P0
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

def train_epoch_multi(model, dataloader, optimizer, device, loss_weights):
    """Training function with weighted MSE loss"""
    model.train()
    running_loss = running_rmse = 0.0
    condition_losses = {}
    
    # Convert loss_weights to tensor on the correct device
    if not isinstance(loss_weights, torch.Tensor):
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32, device=device)
    else:
        loss_weights = loss_weights.to(device)

    for inputs, targets, conditions in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Weighted MSE loss
        squared_errors = (outputs - targets) ** 2
        weighted_squared_errors = squared_errors * loss_weights
        loss = torch.mean(weighted_squared_errors)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_rmse += torch.sqrt(torch.mean((outputs - targets)**2)).item() * inputs.size(0)
        
        for i, cond in enumerate(conditions.numpy()):
            if cond not in condition_losses:
                condition_losses[cond] = []
            condition_losses[cond].append(loss.item())
    
    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_rmse = running_rmse / dataset_size
    
    return epoch_loss, epoch_rmse, condition_losses

def evaluate_model_errors_multi(model, dataloader, device):
    """Evaluate model and return comprehensive error metrics"""
    model.eval()
    all_targets, all_outputs = [], []
    condition_errors = {}
    condition_targets, condition_outputs = {}, {}
    
    with torch.no_grad():
        for inputs, targets, conditions in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            batch_targets = targets.cpu()
            batch_outputs = outputs.cpu()
            
            all_targets.append(batch_targets)
            all_outputs.append(batch_outputs)
            
            # Store by condition
            for i, cond in enumerate(conditions.numpy()):
                if cond not in condition_targets:
                    condition_targets[cond] = []
                    condition_outputs[cond] = []
                condition_targets[cond].append(batch_targets[i])
                condition_outputs[cond].append(batch_outputs[i])
    
    # Calculate overall metrics
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    mae_by_species = torch.mean(torch.abs(all_targets - all_outputs), dim=0)
    rmse_by_species = torch.sqrt(torch.mean((all_targets - all_outputs)**2, dim=0))
    relative_error_by_species = torch.mean(torch.abs(all_targets - all_outputs) / (torch.abs(all_targets) + 1e-12), dim=0)
    
    # Calculate per-condition metrics
    for cond in condition_targets:
        cond_targets = torch.stack(condition_targets[cond])
        cond_outputs = torch.stack(condition_outputs[cond])
        
        cond_mae = torch.mean(torch.abs(cond_targets - cond_outputs), dim=0)
        cond_rmse = torch.sqrt(torch.mean((cond_targets - cond_outputs)**2, dim=0))
        cond_relative_error = torch.mean(torch.abs(cond_targets - cond_outputs) / (torch.abs(cond_targets) + 1e-12), dim=0)
        
        condition_errors[cond] = {
            'mae': torch.mean(cond_mae).item(),
            'rmse': torch.mean(cond_rmse).item(),
            'relative_error': torch.mean(cond_relative_error).item(),
            'mae_by_species': cond_mae.tolist(),
            'rmse_by_species': cond_rmse.tolist()
        }
    
    return {
        'overall_mae': torch.mean(mae_by_species).item(),
        'overall_rmse': torch.mean(rmse_by_species).item(),
        'overall_relative_error': torch.mean(relative_error_by_species).item(),
        'mae_by_species': mae_by_species.tolist(),
        'rmse_by_species': rmse_by_species.tolist(),
        'condition_errors': condition_errors
    }

def plot_training_history(train_losses, train_rmses, error_results, model_name):
    """Save three separate, polished training history plots for publication"""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']  # colorblind-friendly

    # Training Loss
    plt.figure(figsize=(6, 5))
    plt.plot(train_losses, color=colors[0])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss')
    plt.tight_layout(pad=2.0)
    plt.savefig(f'results/{model_name}_training_loss.pdf', bbox_inches='tight')
    plt.close()

    # Training RMSE
    plt.figure(figsize=(6, 5))
    plt.plot(train_rmses, color=colors[1])
    plt.xlabel('Epoch')
    plt.ylabel('Root Mean Squared Error')
    plt.yscale('log')
    plt.title('Training RMSE')
    plt.tight_layout(pad=2.0)
    plt.savefig(f'results/{model_name}_training_rmse.pdf', bbox_inches='tight')
    plt.close()

    # Final RMSE by Species
    species_names = ['E', 'S', 'ES', 'P']
    plt.figure(figsize=(6, 5))
    plt.bar(species_names, error_results['rmse_by_species'], color=colors, alpha=0.8)
    plt.ylabel('Root Mean Squared Error')
    plt.title('Final RMSE by Species')
    plt.tight_layout(pad=2.0)
    plt.savefig(f'results/{model_name}_final_rmse_by_species.pdf', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train model on multi-condition MM reaction kinetics dataset')
    parser.add_argument('--model', type=str, default='ResidualMLP', choices=['MLP', 'ResidualMLP'])
    parser.add_argument('--epochs', type=int, default=3000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--file', type=str, default=None, help='Data file to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers/blocks')
    parser.add_argument('--early_stop_patience', type=int, default=500, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--p_weight', type=float, default=1e4, help='Loss weight for species P')
    args = parser.parse_args()

    # Setup
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load dataset
    print("Loading multi-condition dataset...")
    if args.file is None:
        data_files = [f for f in os.listdir('data/teacher') if f.startswith('teacher_combined_multiple') and f.endswith('.npy')]
        if not data_files:
            raise FileNotFoundError("No multiple condition dataset found. Please run MAE_simulation.py first.")
        data_path = os.path.join('data/teacher', sorted(data_files)[-1])
    else:
        data_path = args.file
    
    print(f"Using dataset: {data_path}")
    data = np.load(data_path)
    print(f"Dataset shape: {data.shape}")

    # Extract features and targets
    X = data[:, 0:5]  # time and initial conditions (E0, S0, ES0, P0)
    y = data[:, 5:9]  # trajectories (E, S, ES, P)
    
    unique_conditions = np.unique(X[:, 1:5], axis=0)
    print(f"Dataset contains {len(unique_conditions)} unique initial conditions")

    # Preprocessing - normalize X
    X_scaler = StandardScaler()
    
    print("Log-transforming time, then normalizing X...")
    X_copy = X.copy()
    X_copy[:, 0] = np.log10(X_copy[:, 0] + 1e-12)  # Log transform time
    X_norm = X_scaler.fit_transform(X_copy)
    
    # Normalize y globally (as a whole)
    print("Normalizing y globally...")
    y_scaler = StandardScaler()
    y_normalized = y_scaler.fit_transform(y)
    
    print(f"  y normalization - mean: {y_scaler.mean_}, std: {y_scaler.scale_}")
    
    # Get condition labels for tracking
    initial_conditions = X[:, 1:5]
    unique_conditions_arr, condition_labels = np.unique(initial_conditions, axis=0, return_inverse=True)
    
    print(f"Data preprocessing completed. Total samples: {len(X_norm)}")

    # Create dataset and dataloader
    dataset = MultiConditionReactionDataset(X_norm, y_normalized, condition_labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    print(f"Training on {len(dataset)} samples from {len(unique_conditions_arr)} conditions")

    # Model setup
    if args.model == 'ResidualMLP':
        model = ResidualMLP(input_size=5, output_size=4, 
                           hidden_dim=args.hidden_dim, 
                           num_blocks=args.num_layers,
                           dropout=args.dropout)
    else:  # MLP
        model = MLP(input_size=5, output_size=4, 
                   hidden_sizes=[args.hidden_dim] * args.num_layers,
                   dropout=args.dropout)
    
    model = model.to(device)
    print(f"Using model: {model.__class__.__name__}")
    print(f"Model parameters: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, dropout={args.dropout}")

    # Training setup
    model_name = f"{args.model}_{args.model}_multi_{len(unique_conditions_arr)}cond_normalized"

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, verbose=True)

    # Training loop
    print(f"Training {args.model} on {len(unique_conditions_arr)} conditions...")
    train_losses, train_rmses = [], []
    best_train_loss = float('inf')
    patience_counter = 0

    # Loss weights: [E, S, ES, P]
    loss_weights = [1.0, 1.0, 1.0, args.p_weight]
    print(f"Loss weights (E, S, ES, P): {loss_weights}")

    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss, train_rmse, condition_losses = train_epoch_multi(
            model, dataloader, optimizer, device, loss_weights
        )
        
        train_losses.append(train_loss)
        train_rmses.append(train_rmse)
        
        # Early stopping
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'train_losses': train_losses,
                'train_rmses': train_rmses,
                'model_type': args.model,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'epoch': epoch,
            }, f'models/{model_name}_best.pt')
        else:
            patience_counter += 1
        
        scheduler.step(train_loss)
        
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Progress updates
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}: Loss={train_loss:.6e}, RMSE={train_rmse:.6e}, "
                  f"lr={scheduler.optimizer.param_groups[0]['lr']:.6e}")

    # Load best model and evaluate
    if os.path.exists(f'models/{model_name}_best.pt'):
        checkpoint = torch.load(f'models/{model_name}_best.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("\nLoaded best model for evaluation")

    # Final evaluation
    print("\nEvaluating model...")
    error_results = evaluate_model_errors_multi(model, dataloader, device)
    
    print(f"\nFinal Metrics:")
    print(f"  Overall MAE: {error_results['overall_mae']:.6e}")
    print(f"  Overall RMSE: {error_results['overall_rmse']:.6e}")
    print(f"  Overall Relative Error: {error_results['overall_relative_error']:.6f}")
    print(f"\nRMSE by Species:")
    species_names = ['E', 'S', 'ES', 'P']
    for i, species in enumerate(species_names):
        print(f"  {species}: {error_results['rmse_by_species'][i]:.6e}")

    # Plot results
    plot_training_history(train_losses, train_rmses, error_results, model_name)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'error_results': error_results,
        'model_type': args.model,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'train_losses': train_losses,
        'train_rmses': train_rmses,
    }, f'models/{model_name}_final.pt')

    print(f"\nTraining completed!")
    print(f"Best model saved: models/{model_name}_best.pt")
    print(f"Final model saved: models/{model_name}_final.pt")
    print(f"Plots saved: results/{model_name}_*.pdf")

if __name__ == "__main__":
    main()