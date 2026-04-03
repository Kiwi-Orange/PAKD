import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import glob
# Import models from models.py
from models import MLP, TransformerModel, ResidualMLP, RNNModel

# Define time-series dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size=20, stride=1):
        """
        Dataset for time-series data with sliding window approach
        
        Parameters:
        -----------
        X : numpy array
            Features [time, E0, S0, ES0, P0]
        y : numpy array
            Target trajectories [E(t), S(t), ES(t), P(t)]
        window_size : int
            Number of time steps in each sequence
        stride : int
            Step size between consecutive sequences
        """
        self.X = X
        self.y = y
        self.window_size = window_size
        self.stride = stride
        
        # Sort data by time
        sort_indices = np.argsort(X[:, 0])
        self.X = self.X[sort_indices]
        self.y = self.y[sort_indices]
        
        # Find boundaries between different initial conditions
        # Assume X[:, 1:5] contains initial conditions [E0, S0, ES0, P0]
        self.condition_changes = np.where(np.any(np.diff(self.X[:, 1:5], axis=0) != 0, axis=1))[0] + 1
        self.condition_indices = np.split(np.arange(len(self.X)), self.condition_changes)
        
        # Create sequences for each set of initial conditions
        self.sequences = []
        for indices in self.condition_indices:
            if len(indices) >= window_size:
                # Create sequences with sliding window
                for i in range(0, len(indices) - window_size + 1, stride):
                    self.sequences.append((indices[i:i+window_size]))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        indices = self.sequences[idx]
        
        # Get sequence data
        X_seq = self.X[indices]
        y_seq = self.y[indices]
        
        # Return as tensors
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

# Regular dataset for non-sequence models
class ReactionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Training function for sequence models
def train_epoch_sequence(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    
    for inputs, targets in dataloader:
        # inputs: [batch_size, seq_len, features]
        # targets: [batch_size, seq_len, output_features]
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Check if we're using a model that returns sequence or just final prediction
        if outputs.shape == targets.shape:
            # Full sequence prediction
            loss = criterion(outputs, targets)
            mae = torch.abs(outputs - targets).mean()
        else:
            # Final state prediction - use last timestep from targets
            loss = criterion(outputs, targets[:, -1, :])
            mae = torch.abs(outputs - targets[:, -1, :]).mean()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_mae += mae.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    
    return epoch_loss, epoch_mae

# Training function for non-sequence models
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate MSE loss
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_mae += torch.abs(outputs - targets).mean().item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    
    return epoch_loss, epoch_mae

def evaluate_model_r2(model, dataloader, device, is_sequence_model=False):
    """
    Evaluate model on the dataset and return only the R² score
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        is_sequence_model: Whether the model produces sequential outputs
    
    Returns:
        r2: R-squared score (average across all species)
        r2_by_species: R-squared score for each individual species
    """
    model.eval()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Handle different output formats based on model type
            if is_sequence_model:
                if outputs.shape == targets.shape:
                    # Full sequence prediction
                    all_targets.append(targets.cpu())
                    all_outputs.append(outputs.cpu())
                else:
                    # Final state prediction
                    all_targets.append(targets[:, -1, :].cpu())
                    all_outputs.append(outputs.cpu())
            else:
                # Non-sequence model
                all_targets.append(targets.cpu())
                all_outputs.append(outputs.cpu())
    
    # Concatenate results
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    # Calculate R^2 score for each species
    ss_tot = torch.sum((all_targets - all_targets.mean(dim=0))**2, dim=0)
    ss_res = torch.sum((all_targets - all_outputs)**2, dim=0)
    r2_by_species = 1 - ss_res / ss_tot
    
    # Calculate mean R^2 across all species
    r2 = torch.mean(r2_by_species).item()
    
    return r2, r2_by_species.tolist()

def generate_prediction_data(args, E0, S0, X_scaler, model, device, window_size=None, reference_times=None):
    """
    Generate prediction data for both sequence and non-sequence models
    
    Parameters:
    -----------
    reference_times : array, optional
        If provided, use these time points instead of generating new ones
    
    Returns:
        pred_times: array of time points
        final_predictions: array of predictions
    """
    if reference_times is not None:
        # Use provided reference time points
        t_long = reference_times
        print(f"Using {len(t_long)} reference time points for prediction")
    else:
        # Create time points with proper resolution for different phases
        # Fast phase: 0 to 1e-4 seconds (most dynamics happen here)
        t_fast = np.linspace(0, 1e-4, 1000)  # Log-spaced for fast phase (better resolution at early times)

        # Slow phase: 1e-4 to 10 seconds (steady-state behavior)
        t_slow = np.linspace(1e-4, 10, 1000)  # Linear spacing for slow phase

        # Combine and ensure no duplicates
        t_long = np.concatenate([t_fast, t_slow[1:]])  # Skip first point of t_slow to avoid duplicate
        t_long = np.unique(t_long)  # Remove any potential duplicates and sort

    # Generate input features
    X_pred = np.zeros((len(t_long), 5))
    if not args.normalize_only:
        # Apply log transform to match preprocessing
        X_pred[:, 0] = np.log10(t_long + 1e-20)
    else:
        X_pred[:, 0] = t_long
    
    X_pred[:, 1] = E0  # E0
    X_pred[:, 2] = S0  # S0
    X_pred[:, 3] = 0   # ES0
    X_pred[:, 4] = 0   # P0

    # Normalize input features
    X_pred_norm = X_scaler.transform(X_pred)

    model.eval()
    with torch.no_grad():
        if window_size is not None:  # Sequence model
            # Generate sequences with sliding window
            sequences = []
            for i in range(0, len(t_long) - window_size + 1, 1):
                sequences.append(X_pred_norm[i:i+window_size])
            
            # Make predictions for each sequence
            predictions = []
            for seq in tqdm(sequences, desc="Generating predictions", ncols=100):
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                pred = model(seq_tensor)
                predictions.append(pred.cpu().numpy())
            
            predictions = np.array(predictions).squeeze()
            
            # Handle different prediction types
            if args.predict_sequence:
                # Take the first prediction for each sequence
                final_predictions = predictions.reshape(-1, window_size, 4)[:, 0, :]
            else:
                # Each prediction is a single timestep
                final_predictions = predictions
            
            pred_times = t_long[:len(final_predictions)]
        else:  # Non-sequence model
            X_pred_tensor = torch.tensor(X_pred_norm, dtype=torch.float32).to(device)
            final_predictions = model(X_pred_tensor).cpu().numpy()
            pred_times = t_long
    
    return pred_times, final_predictions

def load_ground_truth_data(data_path, args):
    """
    Load and preprocess ground truth data for comparison
    
    Returns:
        true_times: original time points
        true_trajectories: true concentration trajectories
    """
    print("Loading original dataset for ground truth comparison...")
    original_data = np.load(data_path)
    true_X = original_data[:, 0:5].copy()
    true_y = original_data[:, 5:9]

    # Store original times before transformation
    original_times = true_X[:, 0].copy()

    # Apply same preprocessing for consistent sorting
    if not args.normalize_only:
        true_X[:, 0] = np.log10(true_X[:, 0] + 1e-20)

    # Sort by processed time but use original times for plotting
    true_sort_idx = np.argsort(true_X[:, 0])
    true_times = original_times[true_sort_idx]
    true_trajectories = true_y[true_sort_idx]
    
    return true_times, true_trajectories

def create_trajectory_plots(pred_times, final_predictions, true_times, true_trajectories, 
                          model_name, E0, S0, plot_type="full"):
    """
    Create trajectory comparison plots
    
    Args:
        pred_times: prediction time points
        final_predictions: model predictions
        true_times: ground truth time points
        true_trajectories: ground truth trajectories
        model_name: name for saving plots
        E0, S0: initial conditions
        plot_type: "full", "logscale", or "short_term"
    """
    plt.figure(figsize=(15, 10))
    species_names = ['Enzyme (E)', 'Substrate (S)', 'Enzyme-Substrate Complex (ES)', 'Product (P)']
    
    for j in range(4):
        plt.subplot(2, 2, j+1)
        
        if plot_type == "short_term":
            # Filter data for short-term plot
            time_limit = 1e-4  # 100 µs
            true_mask = true_times <= time_limit
            pred_mask = pred_times <= time_limit
            
            plt.plot(true_times[true_mask], true_trajectories[true_mask, j], 
                    'b-', marker='o', markersize=3, 
                    label='True', linewidth=1.5, alpha=0.7)
            plt.plot(pred_times[pred_mask], final_predictions[pred_mask, j], 
                    'r--', marker='x', markersize=3,
                    label='Predicted', linewidth=1.5)
            plt.title(f'{species_names[j]} Short-term Dynamics (≤ {time_limit:.0e}s)')
        else:
            # Full trajectory plots
            if plot_type == "logscale":
                # Filter out zero and negative time values for log scale
                true_mask = true_times > 0
                pred_mask = pred_times > 0
                
                if np.any(true_mask):
                    plt.semilogx(true_times[true_mask], true_trajectories[true_mask, j], 'b-', 
                            label='True', linewidth=2, alpha=0.7)
                if np.any(pred_mask):
                    plt.semilogx(pred_times[pred_mask], final_predictions[pred_mask, j], 'r--', 
                            label='Predicted', linewidth=1.5)
                plt.xlabel('Time (log scale)')
                plt.title(f'{species_names[j]} Trajectory (Log Time)')
            else:
                plt.plot(true_times, true_trajectories[:, j], 'b-', 
                        label='True', linewidth=2, alpha=0.7)
                plt.plot(pred_times, final_predictions[:, j], 'r--', 
                        label='Predicted', linewidth=1.5)
                plt.xlabel('Time')
                plt.title(f'{species_names[j]} Trajectory')
        
        plt.ylabel('Concentration')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Add initial conditions as text
    text = f"Initial: E0={E0:.2e}, S0={S0:.2e}"
    plt.figtext(0.5, 0.01, text, ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plots
    suffix = f"_{plot_type}" if plot_type != "full" else "_full_trajectory"
    plt.savefig(f'results/{model_name}{suffix}.pdf')
    plt.savefig(f'results/{model_name}{suffix}.png', dpi=300)
    plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train model on reaction kinetics dataset')
    parser.add_argument('--model', type=str, default='MLP', 
                        choices=['MLP', 'Transformer', 'RNN', 'LSTM', 'GRU', 'ResidualMLP'],
                        help='Model to use: MLP, Transformer, RNN, LSTM, GRU, or ResidualMLP')
    parser.add_argument('--epochs', type=int, default=5000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--file', type=str, default=None, 
                        help='Data file to use (defaults to most recent single condition file)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension size for networks')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers in RNN/LSTM/GRU or Transformer')
    parser.add_argument('--window_size', type=int, default=2,
                        help='Sequence length for time-series models')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window in time-series data')
    parser.add_argument('--normalize_only', action='store_true', default=False,
                        help='Only normalize input data without log-transforming time')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Use bidirectional RNN/LSTM/GRU')
    parser.add_argument('--predict_sequence', action='store_true', default=False,
                        help='Predict full sequence instead of just final state')
    parser.add_argument('--positive_output', action='store_true', default=True,
                        help='Ensure positive outputs for concentrations')
    args = parser.parse_args()

    # Create directories for model and results
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check for GPU - use MPS for Apple Silicon GPU if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using Apple Silicon GPU: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU: {device}")

    # Load the dataset
    print("Loading dataset...")
    
    # Find the dataset file
    if args.file is None:
        data_files = [f for f in os.listdir('data/teacher') if f.startswith('teacher_combined_single') and f.endswith('.npy')]
        if not data_files:
            raise FileNotFoundError("No single condition dataset found. Please run MAE_simulation.py first with --single flag.")
        data_file = sorted(data_files)[-1]  # Take the most recent file
        data_path = os.path.join('data/teacher', data_file)
    else:
        data_path = args.file
        
    print(f"Using dataset: {data_path}")
    data = np.load(data_path)
    print(f"Dataset shape: {data.shape}")
    
    # Extract file name for model naming
    file_base = os.path.basename(data_path).replace('.npy', '')

    # Extract features and targets
    # Format: time, E0, S0, ES0, P0, E(t), S(t), ES(t), P(t)
    X = data[:, 0:5]  # time and initial conditions [t, E0, S0, ES0, P0]
    y = data[:, 5:9]  # trajectories [E(t), S(t), ES(t), P(t)]
    
    # Extract initial conditions for reference (first row is sufficient as all have same initial values)
    E0 = data[0, 1]
    S0 = data[0, 2]
    
    # Print initial conditions used in this dataset
    print(f"Initial conditions: E0={E0:.2e}, S0={S0:.2e}")

    # Normalize input data
    X_scaler = StandardScaler()
    if args.normalize_only:
        print("Normalizing input data...")
        X_normalized = X_scaler.fit_transform(X)
    else:
        print("Log-transforming time values first, then normalizing...")
        epsilon = 1e-20
        # First apply log transform to time values
        X[:, 0] = np.log10(X[:, 0] + epsilon)  # Log-transform time to avoid numerical issues
        # Then normalize the data (including the log-transformed time)
        X_normalized = X_scaler.fit_transform(X)

    # No need to normalize outputs for this physical problem
    # But keep scalers for reference
    y_scaler = StandardScaler()
    y_scaler.fit(y)  # Just fit but don't transform

    # Determine if we're using a sequence model
    sequence_models = ['Transformer', 'RNN', 'LSTM', 'GRU']
    is_sequence_model = args.model in sequence_models
    
    # Create appropriate dataset based on model type
    if is_sequence_model:
        print(f"Creating time-series dataset with window size {args.window_size} and stride {args.stride}")
        full_dataset = TimeSeriesDataset(X_normalized, y, window_size=args.window_size, stride=args.stride)
        print(f"Created {len(full_dataset)} sequences")
    else:
        full_dataset = ReactionDataset(X_normalized, y)
    
    # Create dataloader
    dataloader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True
    )

    print(f"Training on dataset with {len(full_dataset)} samples")

    # Choose model based on argument
    if args.model == 'Transformer':
        model = TransformerModel(
            input_size=5, 
            output_size=4, 
            d_model=args.hidden_dim, 
            nhead=4,
            num_layers=args.num_layers,
            dim_feedforward=args.hidden_dim * 4,
            dropout=0.0,
            max_seq_length=args.window_size,
            positive_output=args.positive_output
        )
    elif args.model in ['RNN', 'LSTM', 'GRU']:
        model = RNNModel(
            input_size=5,
            output_size=4 if not args.predict_sequence else 4 * args.window_size,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            rnn_type=args.model.lower(),
            bidirectional=args.bidirectional,
            dropout=0.0,
            positive_output=args.positive_output
        )
    elif args.model == 'ResidualMLP':
        model = ResidualMLP(
            input_size=5, 
            output_size=4, 
            hidden_dim=args.hidden_dim, 
            num_blocks=args.num_layers,
            positive_output=args.positive_output
        )
    else:  # Default to MLP
        model = MLP(
            input_size=5, 
            output_size=4, 
            hidden_sizes=[args.hidden_dim] * args.num_layers,
            positive_output=args.positive_output
        )
    
    model = model.to(device)
    print(f"Using model: {model.__class__.__name__}")

    # Create model name with initial conditions and normalization info
    norm_tag = "norm" if args.normalize_only else "log_norm"
    seq_tag = f"_seq{args.window_size}" if is_sequence_model else ""
    bidir_tag = "_bidir" if args.bidirectional and args.model in ['RNN', 'LSTM', 'GRU'] else ""
    fullseq_tag = "_fullseq" if args.predict_sequence else ""
    
    model_name = f"{model.__class__.__name__}_{args.model}_E{E0:.1e}_S{S0:.1e}_{norm_tag}{seq_tag}{bidir_tag}{fullseq_tag}"

    # Optimizer without weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss function
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100)

    # Train the model
    print(f"Training the {model.__class__.__name__} ({args.model})...")
    
    # Choose appropriate training function based on model type
    train_func = train_epoch_sequence if is_sequence_model else train_epoch

    # Track training metrics
    train_losses, train_maes = [], []

    for epoch in tqdm(range(args.epochs), desc="Training Epochs", unit="epoch", leave=True):
        train_loss, train_mae = train_func(model, dataloader, criterion, optimizer, device)
        
        # Update learning rate based on training loss
        scheduler.step(train_loss)
        
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, "
                f"Train Loss: {train_loss:.4e}, Train MAE: {train_mae:.4e}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping based on very low loss
        if train_loss < 1e-6:
            print(f"Training loss below threshold at epoch {epoch+1}. Early stopping.")
            break

    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'initial_conditions': {'E0': E0, 'S0': S0},
        'train_losses': train_losses,
        'is_sequence_model': is_sequence_model,
        'window_size': args.window_size if is_sequence_model else None,
        'predict_sequence': args.predict_sequence if is_sequence_model else False,
        'model_type': args.model,
        'time_log_transformed': True,
    }, f'models/{model_name}.pt')

    # Evaluate on the entire dataset for visualization
    print("Evaluating model and generating trajectory visualizations...")
    r2, r2_by_species = evaluate_model_r2(
        model, dataloader, device, is_sequence_model=is_sequence_model
    )
    print(f"Model R²: {r2:.6f}")
    print(f"R² by species: E: {r2_by_species[0]:.6f}, S: {r2_by_species[1]:.6f}, " 
          f"ES: {r2_by_species[2]:.6f}, P: {r2_by_species[3]:.6f}")

    # Plot training history
    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.legend()

    # Plot training MAE
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, 'r-', label='Training MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training MAE History')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training_history.pdf')
    plt.savefig(f'results/{model_name}_training_history.png', dpi=300)
    plt.close()
    
    # Load ground truth data
    true_times, true_trajectories = load_ground_truth_data(data_path, args)

    # Generate predictions for visualization
    print("Generating predictions for visualization...")
    window_size = args.window_size if is_sequence_model else None
    pred_times, final_predictions = generate_prediction_data(
        args, E0, S0, X_scaler, model, device, window_size, reference_times=true_times
    )
    
    # Create all trajectory plots
    print("Creating trajectory plots...")
    create_trajectory_plots(pred_times, final_predictions, true_times, true_trajectories, 
                          model_name, E0, S0, "full")
    create_trajectory_plots(pred_times, final_predictions, true_times, true_trajectories, 
                          model_name, E0, S0, "logscale")
    create_trajectory_plots(pred_times, final_predictions, true_times, true_trajectories, 
                          model_name, E0, S0, "short_term")
    
    print("All visualizations completed successfully!")

if __name__ == "__main__":
    main()
