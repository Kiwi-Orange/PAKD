import numpy as np
import torch
from models import ResidualMLP, MLP
from tqdm import tqdm
import argparse
import os

def create_time_features(time_array):
    """Create log10 time feature (must match training)"""
    t = time_array
    t1 = np.log10(t + 1e-12)
    return t1.reshape(-1, 1)

def load_teacher_model(model_path, device='cpu'):
    """Load trained teacher model"""
    print(f"Loading teacher model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = checkpoint.get('model_type', 'ResidualMLP')
    input_size = checkpoint.get('input_size', 21)
    
    # Create model
    if model_type == 'ResidualMLP':
        model = ResidualMLP(input_size=input_size, output_size=20)
    else:
        model = MLP(input_size=input_size, output_size=20)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    X_scaler = checkpoint['X_scaler']
    y_scaler = checkpoint.get('y_scaler', None)
    
    print(f"  Model type: {model_type}")
    print(f"  Has y_scaler: {y_scaler is not None}")
    
    return model, X_scaler, y_scaler

def generate_high_resolution_data(model, X_scaler, y_scaler, initial_conditions, 
                                  n_time_points=5000, device='cpu'):
    """
    Generate high-resolution predictions from teacher model
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained teacher model
    X_scaler : StandardScaler
        Scaler for input features
    y_scaler : StandardScaler or None
        Scaler for targets (if model was trained with normalized targets)
    initial_conditions : np.ndarray
        Initial conditions array (n_conditions, 20)
    n_time_points : int
        Number of high-resolution time points
    device : str
        Device for computation
        
    Returns
    -------
    np.ndarray
        High-resolution data [time, IC1-IC20, y1-y20]
    """
    print(f"\nGenerating high-resolution data...")
    print(f"  Initial conditions: {len(initial_conditions)}")
    print(f"  Time points: {n_time_points}")
    
    # Generate log-spaced time points (same range as training)
    time_points = np.logspace(-12, 4, n_time_points)
    
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
    X_augmented = np.column_stack([time_features, X_copy[:, 1:21]])
    
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
    
    # Ensure non-negative (physical constraint)
    predictions = np.maximum(predictions, 0.0)
    
    # Combine: [time, IC1-IC20, y1-y20]
    high_res_data = np.concatenate([X_high_res, predictions], axis=1)
    
    print(f"  Output shape: {high_res_data.shape}")
    print(f"  Concentration range: [{predictions.min():.4e}, {predictions.max():.4e}]")
    
    return high_res_data

def main():
    parser = argparse.ArgumentParser(description='Generate high-resolution data from teacher model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained teacher model')
    parser.add_argument('--n_time_points', type=int, default=5000,
                       help='Number of high-resolution time points')
    parser.add_argument('--use_training_ics', action='store_true',
                       help='Use initial conditions from training data')
    parser.add_argument('--training_data', type=str, default=None,
                       help='Path to training data (to get ICs)')
    parser.add_argument('--output_dir', type=str, default='data/teacher',
                       help='Output directory')
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, X_scaler, y_scaler = load_teacher_model(args.model, device)
    
    # Get initial conditions
    if args.use_training_ics and args.training_data:
        print(f"\nLoading training data from: {args.training_data}")
        train_data = np.load(args.training_data)
        train_X = train_data[:, 0:21]
        initial_conditions = np.unique(train_X[:, 1:21], axis=0)
        print(f"  Found {len(initial_conditions)} unique initial conditions")
    else:
        # Use base initial condition from MAE_simulation
        from MAE_simulation import get_pollu_initial_conditions
        base_ic = get_pollu_initial_conditions()
        initial_conditions = base_ic.reshape(1, -1)
        print(f"\nUsing base initial condition from MAE_simulation")
    
    # Generate high-resolution data
    high_res_data = generate_high_resolution_data(
        model, X_scaler, y_scaler, initial_conditions,
        n_time_points=args.n_time_points,
        device=device
    )
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    n_conditions = len(initial_conditions)
    output_filename = f'teacher_high_res_{n_conditions}cond_{args.n_time_points}times.npy'
    output_path = os.path.join(args.output_dir, output_filename)
    
    np.save(output_path, high_res_data)
    print(f"\n✓ High-resolution data saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'n_time_points': args.n_time_points,
        'n_conditions': n_conditions,
        'time_range': (1e-12, 1e4),
        'initial_conditions': initial_conditions,
        'model_path': args.model,
        'has_y_scaler': y_scaler is not None
    }
    metadata_path = output_path.replace('.npy', '_metadata.npz')
    np.savez(metadata_path, **metadata)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    print(f"\n{'='*70}")
    print(f"READY FOR HMM CLUSTERING!")
    print(f"{'='*70}")
    print(f"\nTo run HMM clustering, execute:")
    print(f"  python HMM_clustering.py --data_file {output_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()