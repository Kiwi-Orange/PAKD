import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def michaelis_menten(S, Vmax, Km):
    """Michaelis-Menten rate law"""
    return Vmax * S / (Km + S)

def full_system_odes(y, t, S, k1, k_minus1, k2, E_tot):
    """Full ODE system for E + S <-> ES -> E + P"""
    ES, P = y
    E = E_tot - ES
    
    dES_dt = k1 * E * S - (k_minus1 + k2) * ES
    dP_dt = k2 * ES
    
    return [dES_dt, dP_dt]

def get_steady_state_rate(S, k1, k_minus1, k2, E_tot):
    """Get steady-state rate for given substrate concentration"""
    # Solve to steady state
    t = np.linspace(0, 50, 1000)  # Long enough to reach steady state
    y0 = [0, 0]  # Initial conditions: ES=0, P=0
    
    sol = odeint(full_system_odes, y0, t, args=(S, k1, k_minus1, k2, E_tot))
    
    # Calculate rate at steady state (slope of P vs t at the end)
    ES_ss = sol[-1, 0]
    rate_ss = k2 * ES_ss
    
    return rate_ss

# Parameters
k1 = 100.0      # Forward binding rate
k_minus1 = 10.0  # Reverse binding rate
k2 = 1.0      # Catalysis rate
E_tot = 1e6-10000   # Total enzyme concentration
Vmax = k2 * E_tot

# Calculate Km for both approximations
Km_PEA = k_minus1 / k1  # PEA: Km = 1/Keq
Km_QSSA = (k_minus1 + k2) / k1  # QSSA: Km = (k-1 + k2)/k1

print(f"PEA Km: {Km_PEA:.3f}")
print(f"QSSA Km: {Km_QSSA:.3f}")
print(f"Ratio (QSSA/PEA): {Km_QSSA/Km_PEA:.3f}")

# Substrate concentration range
S = np.linspace(1e6-10000, 1e6, 50)  # Reduced points for faster computation

# Calculate rates for both approximations
rate_PEA = michaelis_menten(S, Vmax, Km_PEA)
rate_QSSA = michaelis_menten(S, Vmax, Km_QSSA)

# Calculate rates for full system (this will take a moment)
print("Computing full system rates...")
rate_full = np.array([get_steady_state_rate(s, k1, k_minus1, k2, E_tot) for s in S])

# Create visualization
plt.figure(figsize=(12, 8))

# Plot 1: Rate vs Substrate concentration
plt.subplot(2, 2, 1)
plt.plot(S, rate_full, 'k-', linewidth=3, label='Full System (True)', alpha=0.8)
plt.plot(S, rate_PEA, 'b--', linewidth=2, label=f'PEA ($K_M$ = {Km_PEA:.3f})')
plt.plot(S, rate_QSSA, 'r:', linewidth=2, label=f'QSSA ($K_M$ = {Km_QSSA:.3f})')
plt.xlabel('Substrate Concentration [S]')
plt.ylabel('Reaction Rate')
plt.title('PEA vs QSSA vs Full System')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Relative difference from full system
plt.subplot(2, 2, 2)
rel_diff_PEA = (rate_PEA - rate_full) / rate_full * 100
rel_diff_QSSA = (rate_QSSA - rate_full) / rate_full * 100

plt.plot(S, rel_diff_PEA, 'b--', linewidth=2, label='PEA vs Full')
plt.plot(S, rel_diff_QSSA, 'r:', linewidth=2, label='QSSA vs Full')
plt.xlabel('Substrate Concentration [S]')
plt.ylabel('Relative Difference (%)')
plt.title('Approximation Errors vs Full System')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='-', alpha=0.3)

# Plot 3: PEA vs QSSA comparison
plt.subplot(2, 2, 3)
relative_diff_approx = (rate_PEA - rate_QSSA) / rate_QSSA * 100
plt.plot(S, relative_diff_approx, 'g-', linewidth=2)
plt.xlabel('Substrate Concentration [S]')
plt.ylabel('Relative Difference (%)')
plt.title('PEA vs QSSA: (PEA - QSSA)/QSSA × 100%')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='-', alpha=0.3)

# Plot 4: Log-scale comparison
plt.subplot(2, 2, 4)
plt.semilogx(S, rate_full, 'k-', linewidth=3, label='Full System', alpha=0.8)
plt.semilogx(S, rate_PEA, 'b--', linewidth=2, label='PEA')
plt.semilogx(S, rate_QSSA, 'r:', linewidth=2, label='QSSA')
plt.xlabel('Substrate Concentration [S] (log scale)')
plt.ylabel('Reaction Rate')
plt.title('Log Scale Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PEA_vs_QSSA_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print key differences
print(f"\nKey Insights:")
print(f"- QSSA is closer to the full system than PEA")
print(f"- Maximum PEA error: {np.max(np.abs(rel_diff_PEA)):.1f}%")
print(f"- Maximum QSSA error: {np.max(np.abs(rel_diff_QSSA)):.1f}%")
print(f"- QSSA accounts for enzyme consumption during catalysis")
print(f"- PEA overestimates rates, especially at intermediate [S]")