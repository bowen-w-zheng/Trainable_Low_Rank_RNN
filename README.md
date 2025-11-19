# Trainable Low-Rank RNN

Implementation of a trainable low-rank recurrent neural network for the contextual switch task, based on [Mastrogiuseppe & Ostojic (2018)](https://www.cell.com/neuron/fulltext/S0896-6273(18)30543-9).

## Overview

This project implements a continuous-time low-rank RNN where:
- The random bulk connectivity `gC` is fixed (sampled randomly)
- The low-rank factors `M` and `N` are trained
- Only O(NR) parameters are learned, not O(N²)

### Network Dynamics

```
τ dx/dt = -x + J φ(x) + B u(t)
J = g C + (1/N) M N^T
y = (1/N) w^T φ(x) + b
```

### Contextual Switch Task

The network learns to report different stimulus features based on context:
- Context 1: Report sign of stimulus 1
- Context 2: Report sign of stimulus 2

## Installation

```bash
# Install dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.10
- JAX, JAXlib
- diffrax (ODE solving)
- optax (optimization)
- numpy, matplotlib

## Usage

### Training

```bash
# Train with default configuration
./scripts/run_train_context_switch.sh

# Train with custom config
./scripts/run_train_context_switch.sh configs/contextual_switch_large.yaml results/large

# Quick sanity check
./scripts/run_sanity_check.sh
```

### Evaluation and Analysis

```bash
# Analyze trained model
./scripts/run_eval_context_switch.sh results/contextual_switch_TIMESTAMP
```

### Running Tests

```bash
./scripts/run_tests.sh

# Or with pytest directly
pytest tests/ -v
```

## Project Structure

```
project_root/
├── README.md
├── pyproject.toml
├── src/
│   ├── config.py              # Configuration dataclasses
│   ├── models/
│   │   ├── lowrank_rnn.py     # Low-rank RNN model
│   │   └── integrators.py     # ODE integration (diffrax)
│   ├── data/
│   │   └── contextual_switch_dataset.py
│   ├── training/
│   │   ├── losses.py          # Loss functions
│   │   ├── metrics.py         # Accuracy metrics
│   │   └── train_context_switch.py
│   └── analysis/
│       ├── plot_trajectories.py
│       └── inspect_lowrank.py
├── configs/
│   ├── contextual_switch_default.yaml
│   ├── contextual_switch_sanity.yaml
│   └── contextual_switch_large.yaml
├── tests/
│   ├── test_config.py
│   ├── test_integrators.py
│   ├── test_lowrank_rnn.py
│   ├── test_dataset.py
│   ├── test_losses_and_metrics.py
│   └── test_training_step.py
└── scripts/
    ├── run_train_context_switch.sh
    ├── run_eval_context_switch.sh
    ├── run_tests.sh
    └── run_sanity_check.sh
```

## Configuration

Key configuration parameters:

### RNN Config
- `N`: Number of recurrent units (default: 500)
- `R`: Rank of low-rank connectivity (default: 2)
- `g`: Gain for random bulk (default: 0.8)
- `tau`: Time constant (default: 1.0)

### Task Config
- `stim_mean_abs`: Stimulus magnitude (default: 1.2)
- `gamma_on`: Active context signal (default: 0.08)
- `gamma_off`: Suppressed context signal (default: -0.14)
- `T_burn`: Burn-in period (default: 15.0)
- `T_stim`: Stimulus period (default: 85.0)

### Training Config
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-3)
- `n_train_trials`: Number of training trials (default: 10000)

## Technical Details

### ODE Integration

Uses diffrax for differentiable ODE solving:
- Default solver: Tsit5 (adaptive)
- Alternative: Euler (fixed-step, faster for training)

### Backpropagation

Gradients computed through the ODE solve via diffrax's automatic differentiation.

### Parameter Count

For N=500, R=2:
- Trainable: M (1000) + N (1000) + B (2000) + w (500) + b (1) = 4501 parameters
- Fixed: C (250,000) - not trained

## References

Mastrogiuseppe, F., & Ostojic, S. (2018). Linking connectivity, dynamics, and computations in low-rank recurrent neural networks. Neuron, 99(3), 609-623.

## License

MIT
