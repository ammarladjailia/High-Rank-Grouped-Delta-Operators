# High-Rank Grouped Delta Operators (HRG-Delta)

This repository contains the official PyTorch implementation for the paper:
**"High-Rank Grouped Delta Operators Enable Dynamic Geometric Transformations in Residual Networks"**

## Abstract
We introduce High-Rank Grouped Delta (HRG-Delta) operators, a parametric family of geometric residual mappings that partition feature channels into disjoint groups and apply learned low-rank subspace corrections with dynamically generated orthonormal bases and bounded directional gates. For the fixed-basis, fixed-gating analytical surrogate $A = I_d - \mathbf{Q}\mathbf{B}\mathbf{Q}^\top$, we derive the explicit active-subspace eigenvalues $\lambda_i = 1 - \beta_i$. The restriction to the active subspace is strictly Schur stable on the active subspace $\mathcal{S}$ if and only if $\beta_i \in (0,2)$, while the orthogonal complement is invariant with unit eigenvalues in the low-rank setting ($r < d$), ensuring full-space non-expansiveness without asymptotic stability to the origin. Gates in the over-relaxed (hyper-projective) regime $\beta_i \in (1,2)$ yield negative active eigenvalues ($\lambda_i \in (-1,0)$) and controlled finite-step sign-alternating discrete dynamics, whereas the continuous-time Euler limit yields monotonic exponential decay along active coordinates. For the implemented state-dependent layer, derivatives of the basis generator, gating network, and value map contribute additional terms to the full Jacobian, meaning that the fixed-basis eigenvalue law serves as a local spectral proxy rather than an exact spectrum. Controlled experiments on synthetic dynamical systems validate the fixed-basis eigenvalue formula, the signed active-coordinate estimator, and the active/orthogonal decomposition. On CIFAR-10 over five seeds, comparing gate domains $\beta \in [0,1]$ and $\beta \in [0,2]$ yields comparable test accuracy ($93.58 \pm 0.07\%$ vs.\ $93.65 \pm 0.11\%$), while the latter produces a substantially higher active fixed-basis spectral proxy ($0.9103 \pm 0.0115$ vs.\ $0.5703 \pm 0.0449$), corroborating distinct active-subspace dynamics without establishing a universal feature-sharpening or discriminability advantage.

## Keywords
residual networks, subspace projections, spectral analysis, over-relaxed projections, neural ordinary differential equations, Lyapunov stability, grouped convolutions, directional sign inversion

## Authors
- **Ammar Ladjailia** - Laboratory of Computer Science and Mathematics, Department of Computer Science, University of Souk Ahras, Souk Ahras, 41000, Algeria
- **Aida Bouzid** - Laboratory of Computer Science and Mathematics, Department of Computer Science, University of Souk Ahras, Souk Ahras, 41000, Algeria
- **Hafed Zarzour** - Laboratory of Computer Science and Mathematics, Department of Computer Science, University of Souk Ahras, Souk Ahras, 41000, Algeria
- **Yonis Gulzar** - Department of Management Information Systems, College of Business Administration, King Faisal University, Al-Ahsa, 31982, Saudi Arabia
- **Mohannad Alkanan** - Department of Management Information Systems, College of Business Administration, King Faisal University, Al-Ahsa, 31982, Saudi Arabia
- **Musrrat Ali** - Department of Basic Sciences, General Administration of Preparatory Year, King Faisal University, Al-Ahsa, 31982, Saudi Arabia

## Requirements
```bash
pip install -r requirements.txt
```

## Repository Structure
- `models/`: Contains the implementation of the `HRGDeltaLayer`, baseline architectures (Vanilla ResNet, SENet, CANet, Parameter-Matched), and basis generators.
- `configs/`: YAML configuration files to reproduce the exact 5-seed experiments from the paper.
- `scripts/`: Evaluation and ablation scripts (e.g., linear probe, complexity analysis, gate plotting).
- `metrics/`: Custom metric tracking for spectral norm, latency, and representation margins (Fisher Ratio).
- `train.py`: Main training loop.

## Reproducing Paper Results (CIFAR-10)

To train the HRG-Delta model ($r=4, G=4$) to achieve the reported 93.65% accuracy:
```bash
python train.py --config configs/cifar10_full_dynamic.yaml --seed 123
```

To train the Parameter-Matched Baseline (93.49% accuracy):
```bash
python train.py --config configs/cifar10_residual_adapter_parameter_matched.yaml --seed 123
```

## Evaluating Representation Margins (Fisher Ratio)
After training, to evaluate the Linear Probe accuracy and the latent intra/inter-class margins ($S_w, S_b$):
```bash
python scripts/compute_linear_probe.py
```

## Citation
If you use this code in your research, please cite our paper:
*(Citation details will be updated upon publication)*
# High-Rank-Grouped-Delta-Operators
# High-Rank-Grouped-Delta-Operators
