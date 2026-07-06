import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def cle_mm_simulation_adaptive(k1, km1, k2, x0, t_switch,
                              dt_fast, dt_slow, T, seed=None):
    """
    CLE with two phases:
      [0, t_switch] with dt=dt_fast
      [t_switch, T]  with dt=dt_slow
    """
    if seed is not None:
        np.random.seed(seed)

    def integrate_segment(t0, X0, dt, t_end):
        n = int(np.ceil((t_end - t0) / dt))
        ts = t0 + np.arange(n+1)*dt
        ts[-1] = t_end
        X = np.zeros((n+1,4))
        X[0] = X0
        for i in range(n):
            E, S, ES, P = X[i]
            # drift
            dE = (-k1*E*S + km1*ES + k2*ES)*dt
            dS = (-k1*E*S + km1*ES)*dt
            dES= ( k1*E*S - (km1+k2)*ES)*dt
            dP = ( k2*ES)*dt
            # diffusion scales
            s1 = np.sqrt(max(k1*E*S,0))*np.sqrt(dt)
            s2 = np.sqrt(max(km1*ES,0))*np.sqrt(dt)
            s3 = np.sqrt(max(k2*ES,0))*np.sqrt(dt)
            dW1, dW2, dW3 = np.random.randn(3)
            X[i+1,0] = max(E  + dE + (-s1*dW1 + s2*dW2 + s3*dW3), 0)
            X[i+1,1] = max(S  + dS + (-s1*dW1 + s2*dW2),        0)
            X[i+1,2] = max(ES + dES+ ( s1*dW1 - s2*dW2 - s3*dW3),0)
            X[i+1,3] = max(P  + dP + ( s3*dW3),                  0)
        return ts, X

    t1, X1 = integrate_segment(0.0, x0,         dt_fast, t_switch)
    t2, X2 = integrate_segment(t_switch, X1[-1], dt_slow, T)
    # stitch
    t = np.concatenate([t1, t2[1:]])
    X = np.vstack([X1, X2[1:]])
    return t, X

def simulate_cle_ensemble(k1, km1, k2, x0,
                          t_switch, dt_fast, dt_slow,
                          T, num_runs, seed=0):
    """
    Run an ensemble of adaptive‐CLE runs, then resample onto a common grid.
    """
    M = 1000
    t_ref = np.linspace(0, T, M)
    ens = np.zeros((num_runs, M, 4))
    rng = np.random.RandomState(seed)

    for i in tqdm(range(num_runs), desc="CLE ensemble"):
        ts, Xs = cle_mm_simulation_adaptive(
            k1, km1, k2, x0,
            t_switch, dt_fast, dt_slow, T,
            seed=int(rng.randint(1e9))
        )
        # piecewise‐constant resample
        idx = 0
        for j, tt in enumerate(t_ref):
            while idx < len(ts)-1 and ts[idx+1] <= tt:
                idx += 1
            ens[i,j] = Xs[idx]

    return t_ref, ens

def plot_mean_trajectory(t, ens, fname):
    mean = ens.mean(axis=0)
    labels = ['[E]','[S]','[ES]','[P]']
    plt.figure(figsize=(8,5))
    for j,lbl in enumerate(labels):
        plt.plot(t, mean[:,j], label=lbl)
    plt.xlabel('Time')
    plt.ylabel('Mean count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, format='pdf')
    plt.close()

def plot_final_distributions(ens, out_dir, prefix):
    final = ens[:,-1,:]
    species = ['E','S','ES','P']
    os.makedirs(out_dir, exist_ok=True)
    for j,lbl in enumerate(species):
        plt.figure(figsize=(6,4))
        plt.hist(final[:,j], bins=50, alpha=0.7)
        plt.xlabel(f'{lbl} at T_final')
        plt.ylabel('Frequency')
        plt.title(f'{prefix} CLE final dist of {lbl}')
        fn = os.path.join(out_dir, f'{prefix}_cle_final_{lbl}.pdf')
        plt.tight_layout()
        plt.savefig(fn, format='pdf')
        plt.close()

def save_data(t, ens, prefix):
    mean = ens.mean(axis=0)
    final = ens[:,-1,:]
    os.makedirs('data', exist_ok=True)
    # mean trajectory
    np.save(f'data/{prefix}_cle_mean.npy', np.column_stack((t, mean)))
    pd.DataFrame(
        np.column_stack((t, mean)),
        columns=['time','E','S','ES','P']
    ).to_csv(f'data/{prefix}_cle_mean.csv', index=False)
    # final‐state ensemble
    pd.DataFrame(final, columns=['E','S','ES','P']).to_csv(
        f'data/{prefix}_cle_final.csv', index=False
    )

if __name__=='__main__':
    k1, km1, k2 = 100., 10., 1.0
    x0 = [1e6, 1e6, 0., 0.]
    num_runs = 200

    # --- Short term (all fast) ---
    t_ref, ens = simulate_cle_ensemble(
        k1, km1, k2, x0,
        t_switch=1e-6,    # full window is fast
        dt_fast=1e-9, 
        dt_slow=1e-9,
        T=1e-6,
        num_runs=num_runs, seed=42
    )
    os.makedirs('plots/cle_short', exist_ok=True)
    plot_mean_trajectory(t_ref, ens,  'plots/cle_short/short_cle_mean.pdf')
    plot_final_distributions(ens,    'plots/cle_short/final_dists', 'short')
    save_data(t_ref, ens, 'cle_short')

    # --- Long term (two phases) ---
    t_ref, ens = simulate_cle_ensemble(
        k1, km1, k2, x0,
        t_switch=1e-6,
        dt_fast=1e-9,
        dt_slow=1e-6,
        T=10.0,
        num_runs=num_runs, seed=42
    )
    os.makedirs('plots/cle_long', exist_ok=True)
    plot_mean_trajectory(t_ref, ens,  'plots/cle_long/long_cle_mean.pdf')
    plot_final_distributions(ens,    'plots/cle_long/final_dists', 'long')
    save_data(t_ref, ens, 'cle_long')

    print("Done — CLE short & long ensembles complete.")