# PAKD: Phase-Aware Knowledge Distillation for Stiff Dynamical Systems

PAKD is a data-driven framework for reducing stiff dynamical systems. It combines neural network surrogates, unsupervised phase inference via Hidden Markov Models, and phase-weighted knowledge distillation to produce compact reduced models that preserve macroscopic slow dynamics while capturing fast transients.

Validated on four benchmark systems: Fisher–KPP reaction-diffusion PDE, Michaelis–Menten enzyme kinetics, POLLU atmospheric chemistry, and HPN-DREAM breast-cancer signaling.

## Project Structure

```
PAKD/
├── figures/                 # Publication overview figures
├── Fisher–KPP/              # Fisher-KPP reaction-diffusion PDE
├── MMReaction/              # Michaelis-Menten enzyme kinetics (4 species)
├── POLLU/                   # Atmospheric chemistry network (20 species, 25 reactions)
├── HPN-DREAM/               # MCF7 breast-cancer signaling (41 proteins, 36 conditions)
└── README.md
```

## Pipeline

```
Simulation/Data → Teacher Surrogate → HMM Phase Inference → PAKD Distillation → Evaluation
```

## Dependencies

```bash
pip install torch numpy scipy scikit-learn hmmlearn matplotlib pandas networkx torchode PyMuPDF Pillow
```

## Usage

Each subdirectory is self-contained. General workflow:

```bash
# 1. Generate training data
python <benchmark>/MAE_simulation.py   # or Fisher_KPP_simulation.py

# 2. Train teacher surrogate
python <benchmark>/train_teacher_multi.py

# 3. Teacher trajectories & HMM clustering
python <benchmark>/teacher_generation.py
python <benchmark>/HMM_clustering.py

# 4. PAKD distillation
python <benchmark>/PAKD.py

# 5. Evaluation & visualization
python <benchmark>/test_teacher.py
python <benchmark>/test_student.py
python <benchmark>/make_nature_figure.py
```

### Additional Analyses

- **Fisher-KPP**: `analyze_learned_dynamics.py` — extract effective PDE coefficients.
- **MMReaction**: `PEAvsQSSA.py` — PAKD vs. classical QSSA reduction.
- **HPN-DREAM**: `darts_hill_discovery.py` — differentiable signaling network discovery.

## Publication Figures

```bash
python scripts/optimize_academic_figures.py
```

Outputs to `figures/optimized/`.

## Authors

Sheng Ran, Liu Hong, Wuyue Yang — BIMSA, Beijing
