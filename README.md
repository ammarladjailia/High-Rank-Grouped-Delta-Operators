# High-Rank Grouped Delta Operators (HRG-Delta)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23110502-blue.svg)](https://doi.org/10.5281/zenodo.23110502)

Official PyTorch implementation and empirical replication package for the research paper:  
**"High-Rank Grouped Delta Operators Enable Dynamic Geometric Transformations in Residual Networks"** (*Scientific Reports*, Springer Nature).

---

## Table of Contents
1. [Overview & Theoretical Background](#overview--theoretical-background)
2. [Authors & Affiliations](#authors--affiliations)
3. [Repository Structure](#repository-structure)
4. [System Requirements & Prerequisites](#system-requirements--prerequisites)
5. [Installation Guide](#installation-guide)
6. [Quick Start Demo (< 2 seconds)](#quick-start-demo--2-seconds)
7. [Unit Testing](#unit-testing)
8. [Reproduction Guide](#reproduction-guide)
   - [Synthetic Dynamical Systems (Table 4 & Figure 1)](#synthetic-dynamical-systems-table-4--figure-1)
   - [CIFAR-10 Baselines (Table 5 & Figure 2)](#cifar-10-baselines-table-5--figure-2)
   - [Gating Domain Analysis (Table 6 & Figure 3)](#gating-domain-analysis-table-6--figure-3)
   - [Component Disentanglement (Table 7)](#component-disentanglement-table-7)
   - [Subspace Rank Sweep (Table 8 & Figure 4)](#subspace-rank-sweep-table-8--figure-4)
   - [Group Partitioning at Fixed Total Rank (Table 9 & Figure 5)](#group-partitioning-at-fixed-total-rank-table-9--figure-5)
   - [Feature Representations & Linear Probes (Table 10)](#feature-representations--linear-probes-table-10)
   - [Channel Widths & Value Maps (Tables 11 & 12)](#channel-widths--value-maps-tables-11--12)
   - [Complexity, FLOPs & Inference Latency (Table 13)](#complexity-flops--inference-latency-table-13)
   - [Unified Figure Reproduction (Figures 1–10)](#unified-figure-reproduction-figures-110)
   - [Statistical Significance Testing](#statistical-significance-testing)
9. [Instant Replication via Pre-Computed Metrics](#instant-replication-via-pre-computed-metrics)
10. [Citation](#citation)
11. [License](#license)

---

## Overview & Theoretical Background

Standard residual connections compute translations of the form $x_{l+1} = x_l + F(x_l)$, which act along rank-1 unconstrained directions. **High-Rank Grouped Delta (HRG-Delta)** operators generalize residual learning by partitioning feature channels into $G$ disjoint groups and projecting the residual update onto dynamically generated $r$-dimensional orthonormal subspaces:

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \sum_{g=1}^G \mathbf{Q}^{(g)}(\mathbf{x}_l) \, \mathbf{B}^{(g)}(\mathbf{x}_l) \, {\mathbf{Q}^{(g)}(\mathbf{x}_l)}^	op \Big( \mathcal{V}^{(g)}(\mathbf{x}_l) - \mathbf{x}_l \Big)$$

### Key Mathematical Properties
1. **Dynamic Orthonormal Basis $\mathbf{Q}^{(g)}(\mathbf{x}) \in \mathbb{R}^{d_g \times r}$**: Generated on the fly from input channel descriptors via hypernetworks and reduced QR decomposition, guaranteeing ${\mathbf{Q}^{(g)}}^\top \mathbf{Q}^{(g)} = \mathbf{I}_r$.
2. **Bounded Directional Gating $\mathbf{B}^{(g)}(\mathbf{x}) = \operatorname{diag}(\beta_1, \dots, \beta_r)$**: Gates are bounded within controlled intervals (e.g. $[0, 2]$) via scaled sigmoidal activations.
3. **Active Subspace vs. Orthogonal Complement**:
   - On the active subspace $\mathcal{S} = \operatorname{span}(\mathbf{Q})$, the fixed-basis surrogate operator $A = I - \mathbf{Q}\mathbf{B}\mathbf{Q}^\top$ has eigenvalues $\lambda_i = 1 - \beta_i$.
   - Strict Schur stability ($|\lambda_i| < 1$) holds if and only if $\beta_i \in (0, 2)$.
   - When $\beta_i \in (1, 2)$ (the **over-relaxed / hyper-projective regime**), $\lambda_i \in (-1, 0)$, producing controlled discrete directional sign inversions.
   - On the orthogonal complement $\mathcal{S}^\perp$, eigenvalues are identically $1$, preserving feature components outside the active subspace.
4. **Post-Block Architectural Placement**: Positioned directly following the residual block addition and its ReLU activation ($\mathbf{x}_{l+1} = T_{\text{HRG}}(\operatorname{ReLU}(F(\mathbf{x}_l) + \mathbf{x}_l))$), applying subspace corrections to non-negative activated representations without intervening second ReLUs.

---

## Authors & Affiliations

- **Ammar Ladjailia**$^1$ (Software, Methodology, Validation) — `a.ladjailia@univ-soukahras.dz`
- **Aida Bouzid**$^1$ (Formal Analysis, Investigation) — `a.bouzid@univ-soukahras.dz`
- **Hafed Zarzour**$^1$* (Conceptualization, Supervision, Project Administration) — `hafed.zarzour@univ-soukahras.dz`
- **Yonis Gulzar**$^2$* (Conceptualization, Funding Acquisition, Supervision) — `ygulzar@kfu.edu.sa`
- **Mohannad Alkanan**$^2$ (Software, Validation, Visualization) — `malkanaan@kfu.edu.sa`
- **Musrrat Ali**$^3$ (Formal Analysis, Validation) — `mkasim@kfu.edu.sa`

$^1$ Laboratory of Computer Science and Mathematics, University of Souk Ahras, Souk Ahras 41000, Algeria  
$^2$ Department of Management Information Systems, College of Business Administration, King Faisal University, Al-Ahsa 31982, Saudi Arabia  
$^3$ Department of Basic Sciences, General Administration of Preparatory Year, King Faisal University, Al-Ahsa 31982, Saudi Arabia  
*\*Corresponding authors*

---

## Repository Structure

```
├── configs/               # 22 YAML configurations for all models, seeds, and sweeps
│   ├── cifar10_vanilla.yaml
│   ├── cifar10_senet.yaml
│   ├── cifar10_canet.yaml
│   ├── cifar10_residual_adapter_parameter_matched.yaml
│   ├── cifar10_full_dynamic.yaml          # Canonical HRG-Delta (r=4, G=4, beta in [0,2])
│   ├── cifar10_beta_01.yaml               # HRG-Delta (beta in [0,1])
│   ├── cifar10_beta_03.yaml               # HRG-Delta (beta in [0,3])
│   ├── cifar10_rank_*.yaml                # Rank sweep configs (r=1, 2, 4, 8, 16)
│   └── cifar10_group_*.yaml               # Group sweep configs (G=1, 2, 4, 8)
├── models/                # PyTorch neural network modules
│   ├── hrg_delta.py       # Core HRGDeltaLayer implementation
│   ├── resnet.py          # ResNet-8 backbone and post-block integration
│   ├── baselines.py       # Parameter-matched adapter baseline
│   ├── se_net.py          # Squeeze-and-Excitation attention baseline
│   └── coordinate_attention.py # Coordinate Attention baseline
├── metrics/               # Evaluation and diagnostic tools
│   ├── spectral.py        # Active fixed-basis spectral proxy computation
│   ├── gating.py          # Gate distribution tracking and GateCollector
│   ├── representation.py  # Fisher ratio, Sw/Sb scatter, linear probe
│   └── complexity.py      # FLOPs, MACs, parameter counting, and latency
├── experiments/           # Benchmark and simulation experiments
│   ├── toy_dynamics.py    # Synthetic linear dynamical system simulation
│   └── geometric_advantage_benchmark.py # Representation margin benchmarking
├── scripts/               # Figure generation and statistical analysis scripts
│   ├── regenerate_all_figures_revision2.py # Master figure generation (Figs 1–10)
│   ├── statistical_tests_canonical.py      # Statistical tests with Holm-Bonferroni
│   ├── measure_baseline_complexity.py      # Latency & FLOPs profiler (Table 13)
│   ├── compute_linear_probe.py             # Linear probe and Fisher ratio (Table 10)
│   ├── compare_groups_fixed_rank.py        # Group sweep analysis (Table 9)
│   ├── make_fig4_fig5_canonical.py         # Rank & group figures (Figs 4 & 5)
│   └── plot_beta_distributions.py          # Gate histograms (Fig 8)
├── tests/                 # Unit test suite (pytest)
│   ├── test_hrg.py        # Forward/backward, QR orthogonality, gate bounds
│   ├── test_toy_dynamics.py # Eigenvalue recovery and stability boundaries
│   ├── test_parameter_matched_adapter.py # Baseline dimension matching
│   └── test_config.py     # Configuration loader validation
├── replication_data/      # Canonical CSV results across all 5 seeds
│   ├── results_canonical.csv
│   ├── representation_diagnostics_statistics.csv
│   ├── statistical_tests_cifar10_accuracy.csv
│   └── complexity_results.csv
├── demo.py                # Minimal Working Example (< 2s execution)
├── train.py               # Main training pipeline
├── evaluate.py            # Model evaluation on test split
├── evaluate_checkpoint.py # Comprehensive diagnostic checkpoint evaluator
├── requirements.txt       # Dependencies with version bounds
├── environment.yml        # Conda environment definition
├── pyproject.toml         # Packaging configuration
├── setup.py               # Setuptools installation script
├── LICENSE                # MIT License
└── README.md              # This documentation
```

---

## System Requirements & Prerequisites

- **Operating System:** Linux (Ubuntu 20.04/22.04 LTS tested), macOS, or Windows 10/11 with WSL2.
- **Python:** Version 3.9, 3.10, 3.11, or 3.12.
- **Hardware:**
  - *Standard replication:* Any CUDA-capable NVIDIA GPU with >= 6 GB VRAM (tested on NVIDIA RTX 4500 Ada 24GB).
  - *CPU fallback:* The code executes in CPU mode without modification for demos and small-scale testing.
- **Software Dependencies:** PyTorch >= 2.0.0, torchvision >= 0.15.0, NumPy, SciPy, Pandas, PyYAML, Matplotlib, scikit-learn, tqdm, pytest.

---

## Installation Guide

### Option A: Standard Virtual Environment (Recommended, ~2 minutes)
```bash
# Clone the repository
git clone https://github.com/ammarladjailia/High-Rank-Grouped-Delta-Operators.git
cd High-Rank-Grouped-Delta-Operators

# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Install repository in editable mode
pip install -e .
```

### Option B: Conda / Mamba Environment (~3 minutes)
```bash
conda env create -f environment.yml
conda activate hrg-delta
```

---

## Quick Start Demo (< 2 seconds)

To verify the installation and test the HRG-Delta layer mechanics immediately, execute:
```bash
python demo.py
```

**Expected Output:**
```
Testing HRG-Delta layer initialization and forward pass...
Input shape:  [2, 64, 8, 8]
Output shape: [2, 64, 8, 8]
Gate mean:    0.5339
Orthogonality error: 3.93e-07
Backward pass successful: gradients verified.
HRG-Delta verification complete.
```
This minimal working example validates:
1. Dynamic generation of orthonormal bases $\mathbf{Q}$ via QR decomposition.
2. Orthogonality precision ($\|\mathbf{Q}^\top \mathbf{Q} - \mathbf{I}\|_F < 10^{-6}$).
3. Bounded directional gating ($eta_i \in (0, 2)$).
4. Gradient propagation through basis generator, gating network, and value map.

---

## Unit Testing

Run the automated test suite using `pytest`:
```bash
python -m pytest tests/
```
All 8 test suites should pass in under 5 seconds:
- `test_config.py`: YAML schema validation.
- `test_hrg.py`: Output shapes, orthogonality, gate clamping, and stability.
- `test_parameter_matched_adapter.py`: Parameter equivalence with baseline models.
- `test_toy_dynamics.py`: Active-subspace eigenvalue matching and stability limits.

---

## Reproduction Guide

All experimental results reported in the paper can be trained from scratch or verified via canonical replication data.

### Synthetic Dynamical Systems (Table 4 & Figure 1)
To simulate the discrete linear dynamical systems $\mathbf{x}_{k+1} = (I - \mathbf{Q}\mathbf{B}\mathbf{Q}^\top)\mathbf{x}_k$ across gating regimes $eta \in \{0.5, 1.0, 1.5, 2.0\}$:
```bash
python experiments/toy_dynamics.py --output-dir outputs/toy
```
Outputs:
- Active-coordinate trajectories ($c_k = \mathbf{Q}^\top \mathbf{x}_k$) demonstrating contraction, deadbeat nullification, and sign-alternating dynamics.
- Full-space decomposition confirming strict preservation of the invariant orthogonal component $\|\mathbf{x}_{\perp}\|$.

### CIFAR-10 Baselines (Table 5 & Figure 2)
To train the canonical models across the 5 evaluation seeds (`42, 123, 456, 789, 2026`):

1. **Vanilla ResNet-8** ($92.47 \pm 0.17\%$):
   ```bash
   python train.py --config configs/cifar10_vanilla.yaml --seed 42
   ```
2. **Parameter-Matched ConvNet Adapter** ($93.49 \pm 0.15\%$):
   ```bash
   python train.py --config configs/cifar10_residual_adapter_parameter_matched.yaml --seed 42
   ```
3. **Squeeze-and-Excitation ResNet** ($92.90 \pm 0.19\%$):
   ```bash
   python train.py --config configs/cifar10_senet.yaml --seed 42
   ```
4. **Coordinate Attention ResNet** ($93.40 \pm 0.15\%$):
   ```bash
   python train.py --config configs/cifar10_canet.yaml --seed 42
   ```
5. **HRG-Delta (Ours, $\beta \in [0,2]$)** ($93.65 \pm 0.11\%$):
   ```bash
   python train.py --config configs/cifar10_full_dynamic.yaml --seed 42
   ```

### Gating Domain Analysis (Table 6 & Figure 3)
Evaluate the impact of gating bounds on accuracy and spectral proxy:
```bash
# Contractive regime [0, 1]
python train.py --config configs/cifar10_beta_01.yaml --seed 42

# Over-relaxed regime [0, 2] (Canonical)
python train.py --config configs/cifar10_full_dynamic.yaml --seed 42

# Extended regime [0, 3]
python train.py --config configs/cifar10_beta_03.yaml --seed 42
```

### Component Disentanglement (Table 7)
Isolate the contribution of dynamic orthonormal bases versus dynamic gating:
```bash
# Static Q, Dynamic Beta
python train.py --config configs/cifar10_dynamic_beta.yaml --seed 42

# Dynamic Q, Static Beta
python train.py --config configs/cifar10_dynamic_q.yaml --seed 42

# Static Q, Static Beta
python train.py --config configs/cifar10_static.yaml --seed 42
```

### Subspace Rank Sweep (Table 8 & Figure 4)
Evaluate subspace capacity with $r \in \{1, 2, 4, 8, 16\}$:
```bash
for r in 1 2 4 8 16; do
    python train.py --config configs/cifar10_rank_${r}.yaml --seed 42
done
```

### Group Partitioning at Fixed Total Rank (Table 9 & Figure 5)
Analyze the effect of group count $G \in \{1, 2, 4, 8\}$ at fixed total subspace rank $Gr = 16$:
```bash
python scripts/compare_groups_fixed_rank.py
```

### Feature Representations & Linear Probes (Table 10)
Compute linear classification probe accuracy and Fisher discriminant ratios ($S_b / S_w$):
```bash
python scripts/compute_linear_probe.py
```

### Complexity, FLOPs & Inference Latency (Table 13)
Profile parameter counts, MACs, peak GPU memory, and inference latency across batch sizes 1 and 128:
```bash
python scripts/measure_baseline_complexity.py
```

### Unified Figure Reproduction (Figures 1–10)
Generate all high-resolution figures from the manuscript directly into `./figures/`:
```bash
python scripts/regenerate_all_figures_revision2.py
```

### Statistical Significance Testing
Compute paired $t$-tests, Wilcoxon signed-rank tests, Holm-Bonferroni adjusted $p$-values, and Cohen's $d$ effect sizes:
```bash
python scripts/statistical_tests_canonical.py
```

---

## Instant Replication via Pre-Computed Metrics

To enable instant verification without requiring hundreds of GPU hours to re-train all models across 5 random seeds, the `replication_data/` directory provides complete canonical evaluation records:

- **`results_canonical.csv`**: Full training runs across all models, seeds, accuracies, and parameter counts.
- **`representation_diagnostics_statistics.csv`**: Intra-class scatter ($S_w$), inter-class scatter ($S_b$), Fisher ratios, and linear probe scores.
- **`statistical_tests_cifar10_accuracy.csv`**: Complete statistical test battery (raw and Holm-corrected $p$-values, effect sizes).
- **`complexity_results.csv`**: Measured inference latencies, MACs, parameter counts, and training durations.

---

## Citation

If you find this code or our methodology useful in your research, please cite:

```bibtex
@article{ladjailia2026highrank,
  title={High-Rank Grouped Delta Operators Enable Dynamic Geometric Transformations in Residual Networks},
  author={Ladjailia, Ammar and Bouzid, Aida and Zarzour, Hafed and Gulzar, Yonis and Alkanan, Mohannad and Ali, Musrrat},
  journal={Scientific Reports},
  publisher={Springer Nature},
  year={2026},
  doi={10.5281/zenodo.23110502},
  url={https://github.com/ammarladjailia/High-Rank-Grouped-Delta-Operators}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
