import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Import models
from models import MLP

def load_student_model(model_path, device):
    """Load a saved student model"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    log_transform_y = checkpoint.get('log_transform_y', False)
    state_dict = checkpoint['model_state_dict']
    
    # Extract hidden sizes
    network_layers = [key for key in state_dict.keys() 
                     if key.startswith('network.') and key.endswith('.weight')]
    network_layers.sort(key=lambda x: int(x.split('.')[1]))
    
    hidden_sizes = []
    for layer_key in network_layers[:-1]:
        layer_weight = state_dict[layer_key]
        hidden_sizes.append(layer_weight.shape[0])
    
    # Create and load model
    student_model = MLP(input_size=5, output_size=4, hidden_sizes=hidden_sizes, positive_output=True)
    student_model.load_state_dict(state_dict)
    student_model = student_model.to(device)
    student_model.eval()
    
    X_scaler = checkpoint['X_scaler']
    
    return student_model, X_scaler, log_transform_y

def sample_inputs(domain, n_samples=5000):
    """
    Step 1: Sample inputs from domain
    X = sample_inputs(domain)  # (N, d)
    """
    print(f"📊 Step 1: Sampling {n_samples} inputs from domain...")
    
    # Define parameter ranges for enzyme kinetics
    t_range = domain.get('t', [0.1, 10])
    E0_range = domain.get('E0', [500, 1000])
    S0_range = domain.get('S0', [500, 1000])
    
    # Generate diverse samples
    X = np.zeros((n_samples, 5))
    
    # Time (log-uniform distribution for better coverage)
    X[:, 0] = np.exp(np.random.uniform(np.log(t_range[0]), np.log(t_range[1]), n_samples))
    
    # Initial concentrations (uniform distribution)
    X[:, 1] = np.random.uniform(E0_range[0], E0_range[1], n_samples)  # E0
    X[:, 2] = np.random.uniform(S0_range[0], S0_range[1], n_samples)  # S0
    X[:, 3] = 0  # ES0 = 0
    X[:, 4] = 0  # P0 = 0
    
    print(f"✅ Generated input matrix X: {X.shape}")
    return X

def evaluate_neural_network(student_model, X_scaler, X, device, log_transform_y):
    """
    Step 2: Evaluate neural network
    Y = f_NN(X)  # (N, m)
    """
    print(f"🧠 Step 2: Evaluating neural network...")
    
    # Prepare input for neural network
    X_input = X.copy()
    X_input[:, 0] = np.log10(X_input[:, 0] + 1e-20)  # Log transform time
    
    # Normalize inputs
    X_norm = X_scaler.transform(X_input)
    
    # Predict using neural network
    with torch.no_grad():
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
        Y = student_model(X_tensor).cpu().numpy()
        
        if log_transform_y:
            Y = np.power(10, Y) - 1e-20
            Y = np.clip(Y, a_min=0.0, a_max=None)
    
    print(f"✅ Generated output matrix Y: {Y.shape}")
    return Y

def nondimensionalize(X, Y):
    """
    Step 3: Nondimensionalize/scale (optional)
    X_tilde, Y_tilde, params = nondim(X, Y)
    """
    print(f"📐 Step 3: Nondimensionalizing variables...")
    
    # Extract characteristic scales
    t_char = np.mean(X[:, 0])
    E_char = np.mean(X[:, 1])
    S_char = np.mean(X[:, 2])
    
    # Concentration scales from outputs
    conc_chars = np.mean(Y, axis=0)
    conc_chars = np.maximum(conc_chars, 1e-6)  # Avoid division by zero
    
    # Nondimensionalize inputs
    X_tilde = X.copy()
    X_tilde[:, 0] = X[:, 0] / t_char  # Dimensionless time
    X_tilde[:, 1] = X[:, 1] / E_char  # Dimensionless E0
    X_tilde[:, 2] = X[:, 2] / S_char  # Dimensionless S0
    
    # Nondimensionalize outputs
    Y_tilde = Y / conc_chars[np.newaxis, :]
    
    # Store scaling parameters
    params = {
        't_char': t_char,
        'E_char': E_char, 
        'S_char': S_char,
        'conc_chars': conc_chars
    }
    
    print(f"✅ Scaling parameters: t_char={t_char:.2f}, E_char={E_char:.2f}, S_char={S_char:.2f}")
    return X_tilde, Y_tilde, params

def build_library(X_tilde, params, use_mm_terms=True):
    """
    Step 4: Build candidate library
    Theta, names = build_library(X_tilde, params)  # (N, k)
    """
    print(f"🏗️ Step 4: Building symbolic library...")
    
    t = X_tilde[:, 0]
    E0 = X_tilde[:, 1] 
    S0 = X_tilde[:, 2]
    
    # For this example, we'll build a simple polynomial library
    # In practice, you'd include domain-specific terms
    features = []
    names = []
    
    # Polynomial terms up to degree 2
    features.extend([
        np.ones(len(t)),  # constant
        t, E0, S0,        # linear
        t**2, E0**2, S0**2,  # quadratic
        t*E0, t*S0, E0*S0,   # cross terms
    ])
    names.extend([
        '1', 't', 'E0', 'S0', 
        't^2', 'E0^2', 'S0^2',
        't*E0', 't*S0', 'E0*S0'
    ])
    
    # Add some enzyme kinetics specific terms (optional)
    if use_mm_terms:
        print("  Including Michaelis-Menten type terms...")
        # Michaelis-Menten type terms with different K values
        K_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        for K in K_values:
            features.extend([
                E0*S0/(K + S0),
                S0/(K + S0),
                1/(K + S0)
            ])
            names.extend([
                f'E0*S0/({K}+S0)',
                f'S0/({K}+S0)', 
                f'1/({K}+S0)'
            ])
    else:
        print("  Skipping Michaelis-Menten type terms...")
    
    # Exponential terms
    features.extend([
        np.exp(-t), np.exp(-2*t), np.exp(-0.5*t),
        t*np.exp(-t), S0*np.exp(-t)
    ])
    names.extend([
        'exp(-t)', 'exp(-2*t)', 'exp(-0.5*t)',
        't*exp(-t)', 'S0*exp(-t)'
    ])
    
    Theta = np.column_stack(features)
    
    # Remove invalid features (NaN, inf)
    valid_mask = np.all(np.isfinite(Theta), axis=0)
    Theta_clean = Theta[:, valid_mask]
    names_clean = [name for i, name in enumerate(names) if valid_mask[i]]
    
    print(f"✅ Built library with {len(names_clean)} candidate terms")
    return Theta_clean, names_clean

def sparse_regression(Theta, Y_tilde):
    """
    Step 5: Sparse regression
    """
    print(f"🎯 Step 5: Performing sparse regression...")
    
    # Scale features for better regularization (KEEP IN SCALE SPACE)
    scaler = StandardScaler(with_mean=False)
    Theta_scaled = scaler.fit_transform(Theta)
    
    models = []
    coefficients_scaled = []  # Keep coefficients in scale space
    
    # Fit model for each output dimension
    for i in range(Y_tilde.shape[1]):
        print(f"  Fitting model for output {i+1}/{Y_tilde.shape[1]}...")
        
        # ElasticNet with cross-validation
        model = ElasticNetCV(
            l1_ratio=[0.7, 0.8, 0.9], 
            fit_intercept=False,
            cv=5,
            max_iter=5000,
            random_state=42
        )
        
        model.fit(Theta_scaled, Y_tilde[:, i])
        
        # Keep coefficients in scale space (don't transform back)
        coefficients_scaled.append(model.coef_)
        models.append(model)
    
    coefficients_scaled = np.array(coefficients_scaled).T  # (n_features, n_outputs)
    
    print(f"✅ Sparse regression completed")
    return models, coefficients_scaled, scaler, Theta_scaled, Y_tilde

def prune_coefficients(coefficients_scaled, threshold=1e-6):
    """
    Prune small coefficients (in scale space)
    c = prune(model.coef_)
    """
    print(f"✂️ Pruning coefficients with threshold {threshold}...")
    
    coefficients_pruned = coefficients_scaled.copy()
    
    for i in range(coefficients_scaled.shape[1]):
        # Find maximum coefficient for this output
        max_coef = np.max(np.abs(coefficients_scaled[:, i]))
        
        # Adaptive threshold
        adaptive_threshold = max(threshold, max_coef * 0.01)
        
        # Prune small coefficients
        mask = np.abs(coefficients_scaled[:, i]) < adaptive_threshold
        coefficients_pruned[mask, i] = 0
        
        n_nonzero = np.sum(coefficients_pruned[:, i] != 0)
        print(f"  Output {i+1}: {n_nonzero} non-zero terms")
    
    return coefficients_pruned

def validate_model(Theta_scaled, Y_tilde, coefficients_scaled):
    """
    Step 6: Validation (using scale space)
    R2_val = r2_score(Y_tilde, Theta_scaled @ c_scaled)
    """
    print(f"✅ Step 6: Validating discovered models...")
    
    # Make predictions (in scale space)
    Y_pred = Theta_scaled @ coefficients_scaled
    
    # Calculate R² for each output
    r2_scores = []
    for i in range(Y_tilde.shape[1]):
        r2 = r2_score(Y_tilde[:, i], Y_pred[:, i])
        r2_scores.append(r2)
        print(f"  Output {i+1} R²: {r2:.4f}")
    
    return r2_scores, Y_pred

def report_formula(coefficients_scaled, names, params, scaler):
    """
    Step 7: Report discovered formulas (transform back for interpretation)
    report_formula(c_scaled, names)
    """
    print(f"\n📜 Step 7: Discovered Symbolic Formulas")
    print("="*60)
    
    species_names = ['E', 'S', 'ES', 'P']
    formulas = {}
    
    # Transform coefficients back to original scale for interpretability
    coefficients_original = coefficients_scaled / scaler.scale_[:, np.newaxis]
    
    for i, species in enumerate(species_names):
        terms = []
        
        for j, (coef, name) in enumerate(zip(coefficients_original[:, i], names)):
            if abs(coef) > 1e-10:  # Only include significant terms
                if len(terms) == 0:
                    terms.append(f"{coef:.3e}*{name}")
                else:
                    sign = "+" if coef > 0 else "-"
                    terms.append(f" {sign} {abs(coef):.3e}*{name}")
        
        formula = "".join(terms) if terms else "0"
        formulas[species] = formula
        
        print(f"{species}(t) = {formula}")
    
    # Add dimensional analysis info
    print(f"\nScaling parameters:")
    for key, value in params.items():
        if key == 'conc_chars':
            print(f"  {key} = {value}")
            for i, species in enumerate(species_names):
                print(f"    {species}_char = {value[i]:.3e}")
        elif isinstance(value, (int, float)):
            print(f"  {key} = {value:.3e}")
        else:
            print(f"  {key} = {value}")
    
    return formulas

def plot_results(Y_tilde, Y_pred, r2_scores):
    """Visualize results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    species_names = ['E', 'S', 'ES', 'P']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (species, color) in enumerate(zip(species_names, colors)):
        ax = axes[i]
        
        # Plot actual vs predicted
        ax.scatter(Y_tilde[:, i], Y_pred[:, i], 
                  alpha=0.6, color=color, s=20, label='Predictions')
        
        # Perfect prediction line
        y_range = [min(Y_tilde[:, i].min(), Y_pred[:, i].min()),
                   max(Y_tilde[:, i].max(), Y_pred[:, i].max())]
        ax.plot(y_range, y_range, 'k--', alpha=0.8, label='Perfect fit')
        
        ax.set_xlabel(f'True {species}')
        ax.set_ylabel(f'Predicted {species}')
        ax.set_title(f'{species} (R² = {r2_scores[i]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Symbolic Regression Results', fontsize=16, weight='bold')
    plt.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Structured Symbolic Regression')
    parser.add_argument('--student_model', type=str, required=True,
                       help='Path to saved student model (.pt file)')
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Number of samples to generate')
    parser.add_argument('--threshold', type=float, default=1e-6,
                       help='Threshold for coefficient pruning')
    parser.add_argument('--use_mm_terms', action='store_true',
                       help='Include Michaelis-Menten type terms in the library')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs('results/symbolic_regression', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print("🚀 Starting Structured Symbolic Regression")
    print("="*50)
    
    # Load model
    try:
        student_model, X_scaler, log_transform_y = load_student_model(args.student_model, device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Define domain
    domain = {
        't': [0.1, 10],
        'E0': [500, 1000], 
        'S0': [500, 1000]
    }
    
    # Step 1: Sample inputs
    X = sample_inputs(domain, args.n_samples)
    
    # Step 2: Evaluate neural network
    Y = evaluate_neural_network(student_model, X_scaler, X, device, log_transform_y)
    
    # Step 3: Nondimensionalize
    X_tilde, Y_tilde, params = nondimensionalize(X, Y)
    
    # Step 4: Build library
    Theta, names = build_library(X_tilde, params, use_mm_terms=args.use_mm_terms)
    
    # Step 5: Sparse regression (coefficients stay in scale space)
    models, coefficients_scaled, scaler, Theta_scaled, Y_tilde = sparse_regression(Theta, Y_tilde)
    
    # Prune coefficients (in scale space)
    coefficients_pruned = prune_coefficients(coefficients_scaled, args.threshold)
    
    # Step 6: Validate (using scale space)
    r2_scores, Y_pred = validate_model(Theta_scaled, Y_tilde, coefficients_pruned)
    
    # Step 7: Report formulas (transform back for interpretation)
    formulas = report_formula(coefficients_pruned, names, params, scaler)
    
    # Plot results
    print(f"\n📊 Creating visualization...")
    fig = plot_results(Y_tilde, Y_pred, r2_scores)
    
    # Save results
    model_name = os.path.basename(args.student_model).replace('.pt', '')
    mm_suffix = "_with_mm" if args.use_mm_terms else "_no_mm"
    base_path = f'results/symbolic_regression/{model_name}_structured{mm_suffix}'
    
    # Save plot
    fig.savefig(f'{base_path}_results.png', dpi=300, bbox_inches='tight')
    
    # Save results
    results_df = pd.DataFrame([
        {'Species': species, 'Formula': formula, 'R2': r2}
        for (species, formula), r2 in zip(formulas.items(), r2_scores)
    ])
    results_df.to_csv(f'{base_path}_formulas.csv', index=False)
    
    # Save detailed summary
    with open(f'{base_path}_summary.txt', 'w') as f:
        f.write("STRUCTURED SYMBOLIC REGRESSION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {args.student_model}\n")
        f.write(f"Samples: {args.n_samples}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Michaelis-Menten terms: {args.use_mm_terms}\n\n")
        
        f.write("Discovered Formulas:\n")
        for species, formula in formulas.items():
            f.write(f"{species}(t) = {formula}\n")
        
        f.write(f"\nR² Scores:\n")
        for species, r2 in zip(['E', 'S', 'ES', 'P'], r2_scores):
            f.write(f"{species}: {r2:.4f}\n")
        
        f.write(f"\nScaling Parameters:\n")
        for key, value in params.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.3e}\n")
    
    plt.close()
    
    print(f"\n🎉 Structured symbolic regression complete!")
    print(f"📁 Results saved to: {base_path}_*")
    
    # Print summary
    avg_r2 = np.mean(r2_scores)
    print(f"\n📊 Summary:")
    print(f"  Average R²: {avg_r2:.4f}")
    print(f"  Best species: {['E', 'S', 'ES', 'P'][np.argmax(r2_scores)]} (R² = {max(r2_scores):.4f})")
    print(f"  Michaelis-Menten terms used: {args.use_mm_terms}")

if __name__ == "__main__":
    main()
