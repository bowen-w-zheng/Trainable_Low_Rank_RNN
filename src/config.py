"""Configuration dataclasses for the low-rank RNN experiment."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class RNNConfig:
    """Configuration for the low-rank RNN model."""
    N: int = 500  # Number of recurrent units
    R: int = 2  # Rank of low-rank connectivity
    g: float = 0.8  # Gain for random bulk connectivity
    tau: float = 1.0  # Time constant
    phi: str = "tanh"  # Nonlinearity name
    d_in: int = 4  # Input dimension (s1, s2, ctx1, ctx2)
    use_bias: bool = False  # Whether to use bias in readout

    # Initialization scales
    M_init_std: float = 1.0  # Std for M initialization
    N_init_std: float = 1.0  # Std for N initialization
    B_init_std: float = 1.0  # Std for B initialization
    w_init_std: float = 1.0  # Std for w initialization
    J_init_std: float = 1.0  # Std for full J initialization (full_rank mode)


@dataclass
class IntegratorConfig:
    """Configuration for the ODE integrator."""
    dt: float = 0.1  # Time step for solver
    T: float = 100.0  # Trial duration in time units
    solver_name: str = "tsit5"  # ODE solver name
    save_every: int = 1  # Store states every k steps
    rtol: float = 1e-3  # Relative tolerance
    atol: float = 1e-6  # Absolute tolerance


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    log_every: int = 10  # Log every N iterations
    eval_every: int = 100  # Evaluate every N iterations
    n_train_trials: int = 10000
    n_val_trials: int = 1000

    # Optimizer settings
    optimizer: str = "adam"  # adam or adamw
    grad_clip: float = 1.0  # Gradient clipping norm

    # Training mode: "low_rank" or "full_rank"
    # - low_rank: Train only M, N (low-rank factors), C is fixed
    # - full_rank: Train the entire J connectivity matrix directly
    training_mode: str = "low_rank"

    # What to train (used in low_rank mode)
    train_M: bool = True
    train_N: bool = True
    train_B: bool = True
    train_w: bool = True

    # Checkpointing
    save_every: int = 1000  # Save checkpoint every N iterations
    checkpoint_dir: str = "checkpoints"


@dataclass
class TaskConfig:
    """Configuration for the contextual switch task."""
    # Stimulus parameters
    stim_std: float = 0.0  # Noise std on stimulus (per trial)
    stim_mean_abs: float = 1.2  # Coherence magnitude (Sii in paper)

    # Context parameters (following paper)
    gamma_on: float = 0.08  # Active context strength
    gamma_off: float = -0.14  # Suppressed context strength

    # Trial timing
    T_burn: float = 15.0  # Burn-in time at start of trial
    T_stim: float = 85.0  # Stimulus presentation time
    T_avg: float = 10.0  # Time window at end for averaging output

    # Label format
    label_type: str = "pm1"  # "pm1" for ±1, "binary" for {0, 1}

    # Loss type
    loss_type: str = "mse"  # "mse" or "bce"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    rnn: RNNConfig = field(default_factory=RNNConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    # Experiment metadata
    name: str = "contextual_switch"
    description: str = "Contextual switch task with low-rank RNN"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            rnn=RNNConfig(**data.get('rnn', {})),
            integrator=IntegratorConfig(**data.get('integrator', {})),
            training=TrainingConfig(**data.get('training', {})),
            task=TaskConfig(**data.get('task', {})),
            name=data.get('name', 'contextual_switch'),
            description=data.get('description', ''),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        data = {
            'name': self.name,
            'description': self.description,
            'rnn': {
                'N': self.rnn.N,
                'R': self.rnn.R,
                'g': self.rnn.g,
                'tau': self.rnn.tau,
                'phi': self.rnn.phi,
                'd_in': self.rnn.d_in,
                'use_bias': self.rnn.use_bias,
                'M_init_std': self.rnn.M_init_std,
                'N_init_std': self.rnn.N_init_std,
                'B_init_std': self.rnn.B_init_std,
                'w_init_std': self.rnn.w_init_std,
                'J_init_std': self.rnn.J_init_std,
            },
            'integrator': {
                'dt': self.integrator.dt,
                'T': self.integrator.T,
                'solver_name': self.integrator.solver_name,
                'save_every': self.integrator.save_every,
                'rtol': self.integrator.rtol,
                'atol': self.integrator.atol,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'n_epochs': self.training.n_epochs,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'seed': self.training.seed,
                'log_every': self.training.log_every,
                'eval_every': self.training.eval_every,
                'n_train_trials': self.training.n_train_trials,
                'n_val_trials': self.training.n_val_trials,
                'optimizer': self.training.optimizer,
                'grad_clip': self.training.grad_clip,
                'training_mode': self.training.training_mode,
                'train_M': self.training.train_M,
                'train_N': self.training.train_N,
                'train_B': self.training.train_B,
                'train_w': self.training.train_w,
                'save_every': self.training.save_every,
                'checkpoint_dir': self.training.checkpoint_dir,
            },
            'task': {
                'stim_std': self.task.stim_std,
                'stim_mean_abs': self.task.stim_mean_abs,
                'gamma_on': self.task.gamma_on,
                'gamma_off': self.task.gamma_off,
                'T_burn': self.task.T_burn,
                'T_stim': self.task.T_stim,
                'T_avg': self.task.T_avg,
                'label_type': self.task.label_type,
                'loss_type': self.task.loss_type,
            },
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
