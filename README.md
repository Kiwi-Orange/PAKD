# PAKD: Phase-Aware Knowledge Distillation for Stiff Dynamical Systems

**PAKD** is a data-driven framework for reducing stiff dynamical systems without manual intervention. It combines deep neural network surrogates, unsupervised phase inference via Hidden Markov Models, and phase-weighted knowledge distillation to produce compact reduced models that preserve macroscopic slow dynamics while faithfully capturing fast transients.

The framework is validated on **four benchmark systems** of increasing complexity: a Fisher–KPP reaction-diffusion PDE, Michaelis–Menten enzyme kinetics, the POLLU atmospheric chemistry network, and the HPN-DREAM MCF7 breast-cancer signaling dataset.

## Key Contributions

1. **Unsupervised phase discovery**: An HMM automatically identifies fast/slow dynamical phases from trajectory data, eliminating the need for manual timescale separation.

2. **Phase-aware distillation**: The student model concentrates its capacity on slow dynamics that govern long-term behavior, with phase-dependent loss weights derived from HMM posterior probabilities.

3. **Slow-manifold theory connection**: Mathematical analysis shows that the student's learned behavior corresponds to a slow-manifold projection — it freezes during fast transients and activates on the slow manifold, recovering the true governing equations (verified on Fisher–KPP).

4. **Advantage over classical reduction**: PAKD avoids the systematic errors and spurious oscillations of Quasi-Steady-State Approximation (QSSA) while maintaining comparable compactness (demonstrated on MMReaction and POLLU).

5. **Differentiable network discovery** (HPN-DREAM): Extended to infer interpretable signaling pathway topologies via DARTS-style gated Hill-type ODEs from sparse, noisy experimental data.

## Project Structure

```
PAKD/
├── figures/              # Publication overview figures (Fig1, Fig2)
│   └── optimized/        # Optimized PDF/PNG/SVG outputs
├── scripts/              # Utility scripts
│   └── optimize_academic_figures.py   # Fig1 & Fig2 generation
│
├── Fisher–KPP/           # Fisher-KPP reaction-diffusion PDE (stiff ODE)
├── MMReaction/           # Michaelis-Menten enzyme kinetics (4 species)
├── POLLU/                # Atmospheric chemistry network (20 species, 25 reactions)
├── HPN-DREAM/            # MCF7 breast-cancer signaling (41 proteins, 36 conditions)
│
├── Fig1.pdf / Fig1.png   # Pipeline architecture overview
├── Fig2.pdf              # MMReaction results montage
├── HMM.pdf               # HMM phase inference illustration
└── PAKD.key              # Keynote source for figures
```

## Pipeline Overview

Each benchmark follows the same three-stage pipeline:

```
Simulation / Experimental Data
        │
        ▼
  Teacher Surrogate (MLP / ResidualMLP)
  Learns the full stiff dynamics from trajectory data
        │
        ▼
  HMM Phase Inference
  Identifies fast vs. slow dynamical phases unsupervised
        │
        ▼
  PAKD Distillation
  Student trained with phase-weighted loss → compact reduced model
        │
        ▼
  Evaluation & Analysis
  Rollout comparison, error analysis, model reduction metrics
```

## Dependencies

The project requires Python 3.8+ with the following packages:

| Package | Purpose |
|---------|---------|
| `torch` | Neural network models (MLP, ResidualMLP) |
| `numpy` | Numerical computation |
| `scipy` | ODE integration (`solve_ivp`) |
| `scikit-learn` | Data preprocessing, metrics |
| `hmmlearn` | Hidden Markov Model clustering |
| `matplotlib` | Visualization and figure generation |
| `pandas` | Data handling |
| `networkx` | Network graph visualization (HPN-DREAM) |
| `torchode` | Differentiable ODE integration (HPN-DREAM) |
| `PyMuPDF` (fitz) | PDF rendering (Fig2 generation) |
| `Pillow` | Image processing |

Install with pip:

```bash
pip install torch numpy scipy scikit-learn hmmlearn matplotlib pandas networkx torchode PyMuPDF Pillow
```

## Usage

Each subdirectory is self-contained and follows the same workflow. Below is the general pattern; see individual directories for system-specific configurations.

### 1. Generate Training Data

```bash
# Fisher-KPP: PDE simulation
python Fisher-KPP/Fisher_KPP_simulation.py

# MMReaction: ODE simulation
python MMReaction/MAE_simulation.py

# POLLU: Stiff ODE simulation
python POLLU/MAE_simulation.py

# HPN-DREAM: Load and preprocess experimental data
python HPN-DREAM/MCF7_data_analysis.py
```

### 2. Train Teacher Surrogate

```bash
python <benchmark>/train_teacher_multi.py
```

### 3. Generate Teacher Trajectories & HMM Clustering

```bash
python <benchmark>/teacher_generation.py
python <benchmark>/HMM_clustering.py
```

### 4. PAKD Distillation

```bash
python <benchmark>/PAKD.py
```

### 5. Evaluate & Visualize

```bash
python <benchmark>/test_teacher.py
python <benchmark>/test_student.py
python <benchmark>/make_nature_figure.py
python <benchmark>/make_supplementary_figures.py
```

### Additional Analyses

- **Fisher-KPP**: `analyze_learned_dynamics.py` extracts effective PDE coefficients from the student, revealing the time-gated two-regime structure.
- **MMReaction**: `PEAvsQSSA.py` compares PAKD against classical QSSA reduction.
- **HPN-DREAM**: `darts_hill_discovery.py` performs differentiable signaling network discovery.

## Benchmark Systems

### Fisher–KPP

The Fisher–KPP equation $u_t = \varepsilon u_{xx} + u(1-u)$ with $\varepsilon = 0.01$, discretized via Method of Lines into a stiff ODE. The student autonomously discovers a time-gated two-regime dynamics: it freezes during the fast transient ($t < t^*$) and recovers the true PDE coefficients ($D \approx 0.01$, $r \approx 1.0$) to within 2% during the slow regime.

### MMReaction

The classic Michaelis–Menten enzyme kinetics: $E + S \rightleftharpoons ES \rightarrow E + P$ (4 species). Under stiff parameter regimes (fast binding, slow catalysis), PAKD is compared against the classical QSSA reduction. PAKD avoids QSSA's systematic errors while maintaining compactness. Also tested under stochastic conditions (CLE and CME).

### POLLU

A stiff atmospheric chemistry network with 20 species and 25 reactions. Tests PAKD's scalability to multi-species stiff networks where manual QSSA identification becomes unreliable.

### HPN-DREAM

Real experimental data from the HPN-DREAM challenge: 41 phosphoproteins measured under 36 perturbation conditions (8 stimuli × 3 inhibitors + controls) with sparse time sampling. Beyond surrogate modeling, this benchmark demonstrates **differentiable network discovery** — recovering a sparse, interpretable consensus signaling network through DARTS-style gated Hill-type ODEs.

## Generating Publication Figures

```bash
# Generate Fig1 (pipeline schematic) and Fig2 (MMReaction montage)
python scripts/optimize_academic_figures.py
```

Outputs are written to `figures/optimized/` in PDF, PNG, and SVG formats.

## Authors

- Sheng Ran
- Liu Hong
- Wuyue Yang

BIMSA, Beijing

## License

This project is provided for research and academic purposes. Please contact the authors for usage terms.
