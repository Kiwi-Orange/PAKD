import numpy as np
import torch
from models import ResidualMLP, MLP
from tqdm import tqdm
import argparse
import os
from scipy.sparse import diags

def create_time_features(time_array):
    """Create log10 time feature (must match training)"""
    t = time_array
    t1 = np.log10(t + 1.0)
    return t1.reshape(-1, 1)


def get_fisher_kpp_initial_condition(x, ic_type='step', params=None):
    """
    Generate initial condition for Fisher-KPP equation.
    
    Parameters
    ----------
    x : np.ndarray
        Spatial grid points
    ic_type : str
        Type: 'step', 'gaussian', 'sine', 'double_gaussian'
    params : dict, optional
        Parameters for the initial condition
    """
    if params is None:
        params = {}
    
    if ic_type == 'step':
        x_step = params.get('x_step', 0.3)
        transition_width = params.get('transition_width', 0.05)
        u0 = 0.5 * (1 - np.tanh((x - x_step) / transition_width))
    elif ic_type == 'gaussian':
        center = params.get('center', 0.5)
        width = params.get('width', 0.1)
        amplitude = params.get('amplitude', 0.8)
        u0 = amplitude * np.exp(-((x - center) / width) ** 2)
    elif ic_type == 'sine':
        n_modes = params.get('n_modes', 1)
        u0 = 0.5 * (1 + 0.5 * np.sin(n_modes * np.pi * x))
    elif ic_type == 'double_gaussian':
        u0 = (0.6 * np.exp(-((x - 0.3) / 0.08) ** 2) + 
              0.4 * np.exp(-((x - 0.7) / 0.08) ** 2))
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")
    
    return np.clip(u0, 0.0, 1.0)


def load_teacher_model(model_path, device='cpu'):
    """Load trained teacher model for Fisher-KPP"""
    print(f"Loading teacher model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = checkpoint.get('model_type', 'ResidualMLP')
    n_grid = checkpoint.get('n_grid', 100)
    input_size = checkpoint.get('input_size', n_grid + 1)
    output_size = checkpoint.get('output_size', n_grid)
    
    # Create model
    if model_type == 'ResidualMLP':
        # Infer architecture from state dict
        hidden_dim = 128
        num_blocks = 3
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'input_proj.weight' in state_dict:
                hidden_dim = state_dict['input_proj.weight'].shape[0]
            num_blocks = sum(1 for k in state_dict.keys() if 'blocks.' in k and '.ln.weight' in k)
        
        model = ResidualMLP(
            input_size=input_size,
            output_size=output_size,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=0.0
        )
        print(f"  Architecture: ResidualMLP ({num_blocks} blocks, hidden_dim={hidden_dim})")
    else:
        # MLP
        hidden_dim = 128
        num_layers = 3
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            num_layers = sum(1 for k in state_dict.keys() if 'network.' in k and '.weight' in k) - 1
            if 'network.0.weight' in state_dict:
                hidden_dim = state_dict['network.0.weight'].shape[0]
        
        model = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=[hidden_dim] * num_layers,
            dropout=0.0
        )
        print(f"  Architecture: MLP ({num_layers} layers, hidden_dim={hidden_dim})")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    X_scaler = checkpoint['X_scaler']
    y_scaler = checkpoint.get('y_scaler', None)
    
    print(f"  Model type: {model_type}")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size}")
    print(f"  Grid points: {n_grid}")
    print(f"  Has y_scaler: {y_scaler is not None}")
    
    return model, X_scaler, y_scaler, n_grid


def generate_high_resolution_data(model, X_scaler, y_scaler, initial_conditions, 
                                  n_grid, n_time_points=5000, t_end=10.0,
                                  device='cpu'):
    """
    Generate high-resolution predictions from teacher model for Fisher-KPP.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained teacher model
    X_scaler : StandardScaler
        Scaler for input features
    y_scaler : StandardScaler or None
        Scaler for targets (if model was trained with normalized targets)
    initial_conditions : np.ndarray
        Initial conditions array (n_conditions, n_grid)
    n_grid : int
        Number of interior grid points
    n_time_points : int
        Number of high-resolution time points
    t_end : float
        End time for simulation
    device : str
        Device for computation
        
    Returns
    -------
    np.ndarray
        High-resolution data [time, u0_1, ..., u0_n, u_1, ..., u_n]
    """
    print(f"\nGenerating high-resolution data...")
    print(f"  Initial conditions: {len(initial_conditions)}")
    print(f"  Grid points: {n_grid}")
    print(f"  Time points: {n_time_points}")
    print(f"  Time range: [0, {t_end}]")
    
    # Generate log-spaced time points (same range as training)
    # Start from small time to capture fast dynamics
    t_min = 1e-4
    time_points = np.logspace(np.log10(t_min), np.log10(t_end), n_time_points - 1)
    time_points = np.concatenate([[0.0], time_points])
    
    print(f"  Time sampling: LOG-SPACED")
    print(f"    First 5 times: {time_points[:5]}")
    print(f"    Last 5 times: {time_points[-5:]}")
    
    # Create input grid
    X_high_res = []
    for ic in initial_conditions:
        for t in time_points:
            x_sample = np.concatenate([[t], ic])
            X_high_res.append(x_sample)
    
    X_high_res = np.array(X_high_res)
    print(f"  Input shape: {X_high_res.shape}")
    
    # Preprocess: create time features
    X_copy = X_high_res.copy()
    time_features = create_time_features(X_copy[:, 0])
    X_augmented = np.column_stack([time_features, X_copy[:, 1:n_grid+1]])
    
    # Normalize
    X_norm = X_scaler.transform(X_augmented)
    
    # Predict
    print("  Generating predictions...")
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    
    predictions = []
    batch_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size), desc="Predicting"):
            batch = X_tensor[i:i+batch_size]
            pred = model(batch).cpu().numpy()
            predictions.append(pred)
    
    predictions = np.vstack(predictions)
    
    # Inverse transform if y_scaler exists
    if y_scaler is not None:
        print("  Inverse transforming predictions...")
        predictions = y_scaler.inverse_transform(predictions)
    
    # Ensure physical constraints (u in [0, 1] for Fisher-KPP)
    predictions = np.clip(predictions, 0.0, 1.0)
    
    # Combine: [time, u0_1, ..., u0_n, u_1, ..., u_n]
    high_res_data = np.concatenate([X_high_res, predictions], axis=1)
    
    print(f"  Output shape: {high_res_data.shape}")
    print(f"  Solution range: [{predictions.min():.4e}, {predictions.max():.4e}]")
    
    return high_res_data, time_points


def main():
    parser = argparse.ArgumentParser(description='Generate high-resolution data from teacher model for Fisher-KPP')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained teacher model')
    parser.add_argument('--n_time_points', type=int, default=5000,
                       help='Number of high-resolution time points')
    parser.add_argument('--t_end', type=float, default=10.0,
                       help='End time for simulation')
    parser.add_argument('--use_training_ics', action='store_true',
                       help='Use initial conditions from training data')
    parser.add_argument('--training_data', type=str, default=None,
                       help='Path to training data (to get ICs)')
    parser.add_argument('--ic_type', type=str, default='step',
                       choices=['step', 'gaussian', 'sine', 'double_gaussian'],
                       help='Initial condition type (if not using training ICs)')
    parser.add_argument('--num_ics', type=int, default=1,
                       help='Number of initial conditions to generate')
    parser.add_argument('--output_dir', type=str, default='data/fisher_kpp',
                       help='Output directory')
    args = parser.parse_args()
    
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
    
    # Load model
    model, X_scaler, y_scaler, n_grid = load_teacher_model(args.model, device)
    
    # Get initial conditions
    if args.use_training_ics and args.training_data:
        print(f"\nLoading training data from: {args.training_data}")
        train_data = np.load(args.training_data)
        train_X = train_data[:, 0:n_grid+1]
        initial_conditions = np.unique(train_X[:, 1:n_grid+1], axis=0)
        print(f"  Found {len(initial_conditions)} unique initial conditions")
    else:
        # Generate initial conditions
        print(f"\nGenerating {args.num_ics} initial condition(s) (type: {args.ic_type})...")
        x = np.linspace(0, 1, n_grid + 2)[1:-1]  # Interior points
        
        np.random.seed(42)
        initial_conditions = []
        
        if args.num_ics == 1:
            # Single IC: use a representative one
            u0 = get_fisher_kpp_initial_condition(x, args.ic_type)
            initial_conditions.append(u0)
        else:
            # Multiple ICs: cycle through types for diversity
            ic_types = ['step', 'gaussian', 'sine', 'double_gaussian']
            for i in range(args.num_ics):
                ic_t = ic_types[i % len(ic_types)]
                
                if ic_t == 'step':
                    params = {
                        'x_step': np.random.uniform(0.2, 0.5),
                        'transition_width': np.random.uniform(0.03, 0.08)
                    }
                elif ic_t == 'gaussian':
                    params = {
                        'center': np.random.uniform(0.3, 0.7),
                        'width': np.random.uniform(0.05, 0.15),
                        'amplitude': np.random.uniform(0.5, 1.0)
                    }
                elif ic_t == 'sine':
                    params = {'n_modes': np.random.randint(1, 4)}
                else:
                    params = {}
                
                u0 = get_fisher_kpp_initial_condition(x, ic_t, params)
                initial_conditions.append(u0)
        
        initial_conditions = np.array(initial_conditions)
        print(f"  Initial conditions shape: {initial_conditions.shape}")
    
    # Generate high-resolution data
    high_res_data, time_points = generate_high_resolution_data(
        model, X_scaler, y_scaler, initial_conditions,
        n_grid=n_grid,
        n_time_points=args.n_time_points,
        t_end=args.t_end,
        device=device
    )
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    n_conditions = len(initial_conditions)
    output_filename = f'teacher_high_res_fisher_kpp_{n_conditions}cond_{args.n_time_points}times_n{n_grid}.npy'
    output_path = os.path.join(args.output_dir, output_filename)
    
    np.save(output_path, high_res_data)
    print(f"\n✓ High-resolution data saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'n_time_points': args.n_time_points,
        'n_conditions': n_conditions,
        'n_grid': n_grid,
        'time_range': (0.0, args.t_end),
        'time_points': time_points,
        'initial_conditions': initial_conditions,
        'model_path': args.model,
        'has_y_scaler': y_scaler is not None,
        'ic_type': args.ic_type if not args.use_training_ics else 'from_training',
    }
    metadata_path = output_path.replace('.npy', '_metadata.npz')
    np.savez(metadata_path, **metadata)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"FISHER-KPP HIGH-RESOLUTION DATA GENERATED")
    print(f"{'='*70}")
    print(f"\nData summary:")
    print(f"  Grid points (n_grid): {n_grid}")
    print(f"  Initial conditions: {n_conditions}")
    print(f"  Time points: {args.n_time_points}")
    print(f"  Time range: [0, {args.t_end}]")
    print(f"  Total samples: {len(high_res_data)}")
    print(f"  Data shape: {high_res_data.shape}")
    print(f"    Columns: time (1) + u0 ({n_grid}) + u ({n_grid}) = {1 + 2*n_grid}")
    
    print(f"\n{'='*70}")
    print(f"READY FOR HMM CLUSTERING!")
    print(f"{'='*70}")
    print(f"\nTo run HMM clustering, execute:")
    print(f"  python HMM_clustering.py --data_file {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()