#!/usr/bin/env python3
"""Debug script to analyze why QSSA stiffness appears larger than analytical."""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parameters
k1, km1, k2 = 100.0, 10.0, 1.0
PARAMS = {"k1": k1, "km1": km1, "k2": k2}
TIME_POINTS = np.logspace(-8, 2, 500)

def mm_reaction(_t, y):
    E, S, ES, _P = y
    return [
        -k1 * E * S + km1 * ES + k2 * ES,
        -k1 * E * S + km1 * ES,
        k1 * E * S - km1 * ES - k2 * ES,
        k2 * ES,
    ]

def analytical_solution(E0, S0):
    sol = solve_ivp(mm_reaction, (TIME_POINTS[0], TIME_POINTS[-1]), [E0, S0, 0.0, 0.0],
                    method="BDF", t_eval=TIME_POINTS, rtol=1e-8, atol=1e-10)
    return sol.y.T

def qssa_solution(E0, S0):
    km = (km1 + k2) / k1
    et = E0
    S = np.zeros_like(TIME_POINTS)
    P = np.zeros_like(TIME_POINTS)
    E = np.zeros_like(TIME_POINTS)
    ES = np.zeros_like(TIME_POINTS)
    S[0] = S0
    for i in range(len(TIME_POINTS) - 1):
        dt = TIME_POINTS[i + 1] - TIME_POINTS[i]
        ES[i] = et * S[i] / (km + S[i])
        E[i] = et - ES[i]
        dsdt = -k2 * et * S[i] / (km + S[i])
        S[i + 1] = max(0.0, S[i] + dsdt * dt)
        P[i + 1] = P[i] - dsdt * dt
    ES[-1] = et * S[-1] / (km + S[-1])
    E[-1] = et - ES[-1]
    return np.column_stack([E, S, ES, P])

def stiffness_metrics(trajectory):
    ratios = []
    all_eigs = []
    for E, S, _ES, _P in trajectory:
        J = np.array([
            [-k1 * S, -k1 * E, km1 + k2, 0],
            [-k1 * S, -k1 * E, km1, 0],
            [k1 * S, k1 * E, -(km1 + k2), 0],
            [0, 0, k2, 0],
        ])
        eig = np.abs(np.linalg.eigvals(J))
        eig = eig[eig > 1e-10]
        ratios.append(np.max(eig) / (np.min(eig) + 1e-10) if len(eig) else 1.0)
        all_eigs.append(eig)
    return np.asarray(ratios), all_eigs

# Test with one condition
E0, S0 = 800.0, 900.0

traj_ana = analytical_solution(E0, S0)
traj_qssa = qssa_solution(E0, S0)

ratios_ana, eigs_ana = stiffness_metrics(traj_ana)
ratios_qssa, eigs_qssa = stiffness_metrics(traj_qssa)

print("=== Stiffness comparison ===")
print(f"Analytical mean stiffness: {np.nanmean(ratios_ana):.3e}")
print(f"QSSA mean stiffness:       {np.nanmean(ratios_qssa):.3e}")
print(f"QSSA / Analytical ratio:   {np.nanmean(ratios_qssa)/np.nanmean(ratios_ana):.3f}")

print("\n=== Early time eigenvalues (t=1e-8 s) ===")
print(f"Analytical: {eigs_ana[0]}")
print(f"QSSA:       {eigs_qssa[0]}")
print(f"Analytical min non-zero eig: {np.min(eigs_ana[0]):.3e}, max: {np.max(eigs_ana[0]):.3e}")
print(f"QSSA min non-zero eig:       {np.min(eigs_qssa[0]):.3e}, max: {np.max(eigs_qssa[0]):.3e}")

print("\n=== Late time eigenvalues (t=1e2 s) ===")
print(f"Analytical: {eigs_ana[-1]}")
print(f"QSSA:       {eigs_qssa[-1]}")

# Check the rank of Jacobian for QSSA trajectory
print("\n=== Jacobian rank analysis ===")
for label, traj in [("analytical", traj_ana), ("qssa", traj_qssa)]:
    E, S, ES, P = traj[0]
    J = np.array([
        [-k1 * S, -k1 * E, km1 + k2, 0],
        [-k1 * S, -k1 * E, km1, 0],
        [k1 * S, k1 * E, -(km1 + k2), 0],
        [0, 0, k2, 0],
    ])
    rank = np.linalg.matrix_rank(J, tol=1e-10)
    print(f"{label} Jacobian rank at t=1e-8: {rank}")
    
# Check if QSSA trajectory satisfies conservation
print("\n=== Conservation check ===")
print(f"Analytical: E+ES at t=0 = {traj_ana[0,0] + traj_ana[0,2]:.3f}, at t=end = {traj_ana[-1,0] + traj_ana[-1,2]:.3f}")
print(f"QSSA:       E+ES at t=0 = {traj_qssa[0,0] + traj_qssa[0,2]:.3f}, at t=end = {traj_qssa[-1,0] + traj_qssa[-1,2]:.3f}")
print(f"E0 = {E0}")

# Check S+ES+P conservation (should equal S0)
print(f"\nAnalytical: S+ES+P at t=0 = {traj_ana[0,1] + traj_ana[0,2] + traj_ana[0,3]:.3f}, at t=end = {traj_ana[-1,1] + traj_ana[-1,2] + traj_ana[-1,3]:.3f}")
print(f"QSSA:       S+ES+P at t=0 = {traj_qssa[0,1] + traj_qssa[0,2] + traj_qssa[0,3]:.3f}, at t=end = {traj_qssa[-1,1] + traj_qssa[-1,2] + traj_qssa[-1,3]:.3f}")
print(f"S0 = {S0}")

# Look at where QSSA stiffness is highest
idx_max_qssa = np.argmax(ratios_qssa)
print(f"\n=== QSSA max stiffness at time index {idx_max_qssa}, t={TIME_POINTS[idx_max_qssa]:.3e} ===")
print(f"QSSA eigenvalues: {eigs_qssa[idx_max_qssa]}")
print(f"Analytical eigenvalues at same time: {eigs_ana[idx_max_qssa]}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Stiffness vs time
ax = axes[0, 0]
ax.semilogy(TIME_POINTS, ratios_ana, label='Analytical', lw=2)
ax.semilogy(TIME_POINTS, ratios_qssa, label='QSSA', lw=2, ls='--')
ax.set_xscale('log')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Stiffness ratio')
ax.set_title('Stiffness vs time')
ax.legend()
ax.grid(True, alpha=0.3)

# Min eigenvalue vs time
ax = axes[0, 1]
min_eig_ana = [np.min(e) for e in eigs_ana]
min_eig_qssa = [np.min(e) for e in eigs_qssa]
ax.semilogy(TIME_POINTS, min_eig_ana, label='Analytical', lw=2)
ax.semilogy(TIME_POINTS, min_eig_qssa, label='QSSA', lw=2, ls='--')
ax.set_xscale('log')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Min non-zero eigenvalue')
ax.set_title('Min eigenvalue vs time')
ax.legend()
ax.grid(True, alpha=0.3)

# Max eigenvalue vs time
ax = axes[1, 0]
max_eig_ana = [np.max(e) for e in eigs_ana]
max_eig_qssa = [np.max(e) for e in eigs_qssa]
ax.semilogy(TIME_POINTS, max_eig_ana, label='Analytical', lw=2)
ax.semilogy(TIME_POINTS, max_eig_qssa, label='QSSA', lw=2, ls='--')
ax.set_xscale('log')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Max eigenvalue')
ax.set_title('Max eigenvalue vs time')
ax.legend()
ax.grid(True, alpha=0.3)

# ES comparison
ax = axes[1, 1]
ax.semilogy(TIME_POINTS, traj_ana[:, 2], label='Analytical ES', lw=2)
ax.semilogy(TIME_POINTS, traj_qssa[:, 2], label='QSSA ES', lw=2, ls='--')
ax.set_xscale('log')
ax.set_xlabel('Time (s)')
ax.set_ylabel('ES concentration')
ax.set_title('ES trajectory comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "/Users/ransheng/PythonProjects/PAKD/MMReaction/stiffness_debug.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
