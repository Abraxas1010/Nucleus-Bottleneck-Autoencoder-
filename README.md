<img src="assets/Apoth3osis.webp" alt="Apoth3osis Logo" width="140"/>

<sub><strong>Our tech stack is ontological:</strong><br>
<strong>Hardware — Physics</strong><br>
<strong>Software — Mathematics</strong><br><br>
<strong>Our engineering workflow is simple:</strong> discover, build, grow, learn & teach</sub>

---

<sub>
<strong>Notice of Proprietary Information</strong><br>
This document outlines foundational concepts and methodologies developed during internal research and development at Apoth3osis. To protect our intellectual property and adhere to client confidentiality agreements, the code, architectural details, and performance metrics presented herein may be simplified, redacted, or presented for illustrative purposes only. This paper is intended to share our conceptual approach and does not represent the full complexity, scope, or performance of our production-level systems. The complete implementation and its derivatives remain proprietary.
</sub>

---

# Nucleus-Bottleneck Autoencoder (NBA)

[![Lean 4](https://img.shields.io/badge/Lean-4-blue.svg)](https://leanprover.github.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: Dual](https://img.shields.io/badge/License-AGPL--3.0%20%2F%20Commercial-green.svg)](LICENSE.md)

## Credo

> *"The genome doesn't specify the organism; it offers a set of pointers to regions in the space of all possible forms, relying on the laws of physics and computation to do the heavy lifting."*
> — **Michael Levin**

Our company operates as a lens for cognitive pointers: identifying established theoretical work and translating it into computationally parsable structures. By turning ideas into formal axioms, and axioms into verifiable code, we create the "Lego blocks" required to build complex systems with confidence.

### Acknowledgment

We humbly thank the collective intelligence of humanity for providing the technology and culture we cherish. We do our best to properly reference the authors of the works utilized herein, though we may occasionally fall short. Our formalization acts as a reciprocal validation—confirming the structural integrity of their original insights while securing the foundation upon which we build. In truth, all creative work is derivative; we stand on the shoulders of those who came before, and our contributions are simply the next link in an unbroken chain of human ingenuity.

---

**A proof-carrying neural architecture that embeds formally verified algebraic operators as architectural constraints for learning dynamical systems.**

---

## The Core Idea

Traditional neural network bottlenecks (e.g., standard autoencoders) have no algebraic guarantees. The **Nucleus-Bottleneck Autoencoder** replaces the bottleneck with a **nucleus operator** — a construct from lattice theory with three machine-verified properties:

| Property | Mathematical Statement | Intuition |
|----------|------------------------|-----------|
| **Inflationary** | `v ≤ R(v)` | The operator only "lifts" values |
| **Idempotent** | `R(R(v)) = R(v)` | Applying twice equals applying once |
| **Meet-preserving** | `R(v ⊓ w) = R(v) ⊓ R(w)` | Distributes over lattice meets |

These properties are **proven in Lean 4** — not assumed, not tested, but mathematically verified.

---

## Why This Matters

### For Machine Learning

- **Architectural guarantees**: The bottleneck's behavior is constrained by proof, not just regularization
- **Interpretable latent space**: Nucleus fixed points have algebraic meaning
- **Stable dynamics**: Koopman evolution respects the lattice structure

### For Formal Verification

- **Proof-carrying code**: The neural architecture comes with machine-checked certificates
- **Verifiable by construction**: Properties hold for *all* inputs, not just test cases
- **Bridges theory and practice**: Same operator defined in Lean and implemented in PyTorch

---

## Architecture

```
Input x_t ──► Encoder E ──► z_raw ──► Nucleus R ──► z ──► Decoder D ──► x̂_t
                                       │
                                       ▼
                              Koopman Generator G
                                       │
                                       ▼
                           z' = exp(G·dt)·z ──► R ──► Decoder D ──► x̂_{t+1}
```

The nucleus `R` enforces that latent representations lie in a sublattice with verified algebraic properties.

---

## Verified Components (Lean 4)

All proofs build without `sorry` or axioms beyond Mathlib's foundations.

| File | What's Proven |
|------|---------------|
| `NucleusReLU.lean` | ReLU is a nucleus on `Fin n → ℝ` |
| `NucleusThreshold.lean` | Threshold `max(x, a)` is a nucleus |
| `HeytingOps.lean` | Bounded interval forms Heyting algebra |
| `ParametricHeyting.lean` | Learnable bounds preserve Heyting structure |

### Key Theorem

```lean
def reluNucleus (n : Nat) : Nucleus (Fin n → ℝ) where
  toFun v i := relu (v i)
  map_inf' v w := by funext i; simp [relu, max_min_distrib_right]
  idempotent' v := by intro i; apply le_of_eq; simp [relu]
  le_apply' v := by intro i; exact le_max_left (v i) 0
```

This isn't a claim — it's a proof that the Lean type checker has verified.

---

## Quick Start

### Verify Lean Proofs

```bash
cd verified_koopman_release/lean
lake build
```

### Install Python Package

```bash
cd verified_koopman_release
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Train a Model

```bash
# Nucleus-bottleneck model on toy 2D system
python scripts/train_nba.py --system toy2d --model nba --epochs 50

# End-to-end learnable Heyting model
python scripts/train_nba.py --system vdp --model e2e --epochs 300
```

### Analyze Heyting Structure

```bash
python scripts/analyze_heyting.py --system vdp --checkpoint outputs/vdp_e2e_0/best.pt
```

---

## Supported Systems

| System | Description | Config |
|--------|-------------|--------|
| `toy2d` | Simple 2D linear rotation | `configs/toy2d.yaml` |
| `vdp` | Van der Pol oscillator | `configs/vdp.yaml` |
| `duffing` | Duffing oscillator (chaotic) | `configs/duffing.yaml` |
| `lorenz` | Lorenz attractor | `configs/lorenz.yaml` |

---

## Repository Structure

```
Nucleus-Bottleneck-Autoencoder-/
├── README.md                           # This file
├── LICENSE.md                          # Dual license (AGPL-3.0 / Commercial)
├── verified_koopman_release/
│   ├── lean/                           # Lean 4 verification project
│   │   ├── VerifiedKoopman/
│   │   │   ├── NucleusReLU.lean        # ★ ReLU nucleus proof
│   │   │   ├── NucleusThreshold.lean   # ★ Threshold nucleus proof
│   │   │   ├── HeytingOps.lean         # ★ Heyting algebra ops
│   │   │   └── ParametricHeyting.lean  # ★ Learnable bounds
│   │   └── lakefile.lean
│   ├── src/verified_koopman/           # Python implementation
│   │   ├── models/
│   │   │   ├── nucleus_bottleneck.py   # NBA architecture
│   │   │   ├── koopman_ae.py           # Base Koopman AE
│   │   │   └── learnable_heyting.py    # Learnable Heyting ops
│   │   ├── losses/                     # Training losses
│   │   ├── verification/               # dReal + Lean gates
│   │   └── analysis/                   # Lyapunov, Heyting analysis
│   ├── scripts/                        # Training & analysis scripts
│   ├── configs/                        # System configurations
│   ├── tests/                          # Unit tests
│   └── docs/                           # Documentation
```

---

## Model Variants

| Model | Description | Use Case |
|-------|-------------|----------|
| `nba` | Fixed nucleus (ReLU or threshold) | Baseline, fast training |
| `e2e` | Learnable threshold + Heyting bounds | Research, curriculum learning |

---

## Experiments

### Capability Experiment
Tests whether NBA learns accurate Koopman representations.

```bash
python scripts/run_experiments.py --experiment capability --epochs 100
```

### Lyapunov Experiment
Analyzes stability via learned Lyapunov functions.

```bash
python scripts/run_experiments.py --experiment lyapunov --epochs 100
```

### Heyting Experiment
Measures how well learned representations respect Heyting algebra structure.

```bash
python scripts/run_experiments.py --experiment heyting --epochs 100
```

---

## Citation

```bibtex
@software{nucleus_bottleneck_ae_2026,
  title = {Nucleus-Bottleneck Autoencoder: Proof-Carrying Neural Architecture for Dynamical Systems},
  author = {Apoth3osis},
  year = {2026},
  url = {https://github.com/Abraxas1010/Nucleus-Bottleneck-Autoencoder-},
  note = {Lean 4 verified nucleus operators as neural network bottlenecks}
}
```

---

## License

This project is provided under the Apoth3osis License Stack v1.
See `LICENSE.md` and the files under `licenses/`.
