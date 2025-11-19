"""Low-rank recurrent neural network model."""

from typing import Dict, Tuple, Any, NamedTuple, Optional
import jax
import jax.numpy as jnp
from functools import partial

from src.config import RNNConfig, IntegratorConfig, TrainingConfig
from src.models.integrators import integrate_rnn_dynamics, make_input_interpolator


class RNNParams(NamedTuple):
    """Parameters for the low-rank RNN.

    In low_rank mode: J = g*C + (1/N)*M @ N_lr^T (C fixed, M/N_lr trainable)
    In full_rank mode: J is directly trainable (N, N) matrix
    """
    C: jnp.ndarray  # Fixed random bulk connectivity (N, N) - used in low_rank mode
    M: jnp.ndarray  # Trainable low-rank factor (N, R) - used in low_rank mode
    N_lr: jnp.ndarray  # Trainable low-rank factor (N, R) - used in low_rank mode
    B: jnp.ndarray  # Input projection (N, d_in)
    w: jnp.ndarray  # Readout weights (N,)
    b: jnp.ndarray  # Readout bias (scalar)
    J: Optional[jnp.ndarray] = None  # Full connectivity matrix (N, N) - used in full_rank mode


class LowRankRNN:
    """
    Low-rank recurrent neural network.

    Dynamics:
        tau * dx/dt = -x + J @ phi(x) + B @ u(t)
        J = g * C + (1/N) * M @ N^T

    where:
        - C is fixed random Gaussian bulk connectivity
        - M, N are trainable low-rank factors (rank R)
        - B maps inputs to recurrent population
        - Output: y = (1/N) * w^T @ phi(x) + b
    """

    def __init__(self, cfg: RNNConfig, key: jax.random.PRNGKey, training_mode: str = "low_rank"):
        """
        Initialize the low-rank RNN.

        Args:
            cfg: RNN configuration
            key: JAX random key for initialization
            training_mode: "low_rank" or "full_rank"
        """
        self.cfg = cfg
        self.N = cfg.N
        self.R = cfg.R
        self.g = cfg.g
        self.tau = cfg.tau
        self.d_in = cfg.d_in
        self.training_mode = training_mode

        # Initialize parameters
        self.params = self._init_params(key)

    def _init_params(self, key: jax.random.PRNGKey) -> RNNParams:
        """Initialize network parameters."""
        keys = jax.random.split(key, 7)

        # Fixed random bulk connectivity: C ~ N(0, 1/sqrt(N))
        # Following paper convention for proper spectral scaling
        C = jax.random.normal(keys[0], (self.N, self.N)) / jnp.sqrt(self.N)

        # Trainable low-rank factors
        # Initialize with small values scaled by 1/sqrt(N)
        M = jax.random.normal(keys[1], (self.N, self.R)) * self.cfg.M_init_std / jnp.sqrt(self.N)
        N_lr = jax.random.normal(keys[2], (self.N, self.R)) * self.cfg.N_init_std / jnp.sqrt(self.N)

        # Input projection
        B = jax.random.normal(keys[3], (self.N, self.d_in)) * self.cfg.B_init_std / jnp.sqrt(self.d_in)

        # Readout weights
        w = jax.random.normal(keys[4], (self.N,)) * self.cfg.w_init_std / jnp.sqrt(self.N)

        # Readout bias
        b = jnp.zeros(())

        # Full connectivity matrix for full_rank mode
        # Initialize as J ~ N(0, g * J_init_std / sqrt(N))
        if self.training_mode == "full_rank":
            J = jax.random.normal(keys[6], (self.N, self.N)) * self.g * self.cfg.J_init_std / jnp.sqrt(self.N)
        else:
            J = None

        return RNNParams(C=C, M=M, N_lr=N_lr, B=B, w=w, b=b, J=J)

    @staticmethod
    def phi(x: jnp.ndarray) -> jnp.ndarray:
        """Nonlinearity (tanh)."""
        return jnp.tanh(x)

    @staticmethod
    def compute_J(params: RNNParams, g: float, N: int, training_mode: str = "low_rank") -> jnp.ndarray:
        """
        Compute effective connectivity matrix.

        In low_rank mode: J = g * C + (1/N) * M @ N^T
        In full_rank mode: J is directly returned from params
        """
        if training_mode == "full_rank" and params.J is not None:
            return params.J
        else:
            return g * params.C + (1.0 / N) * params.M @ params.N_lr.T

    def rhs(self, t: float, x: jnp.ndarray, args: Tuple) -> jnp.ndarray:
        """
        Compute right-hand side of the ODE.

        dx/dt = (-x + J @ phi(x) + B @ u(t)) / tau

        Args:
            t: Current time
            x: Current state (N,)
            args: Tuple of (params, u_of_t) where u_of_t is input function

        Returns:
            dxdt: Time derivative of state (N,)
        """
        params, u_of_t = args

        # Get input at time t
        u_t = u_of_t(t)

        # Compute J matrix
        J = self.compute_J(params, self.g, self.N, self.training_mode)

        # Compute activation
        r = self.phi(x)

        # Compute derivative
        dxdt = (-x + J @ r + params.B @ u_t) / self.tau

        return dxdt

    def readout(self, x: jnp.ndarray, params: RNNParams) -> jnp.ndarray:
        """
        Compute readout from state.

        y = (1/N) * w^T @ phi(x) + b

        Args:
            x: State vector (N,) or trajectory (T, N)
            params: Network parameters

        Returns:
            y: Readout (scalar or (T,))
        """
        r = self.phi(x)
        if x.ndim == 1:
            return (1.0 / self.N) * jnp.dot(params.w, r) + params.b
        else:
            # Batch over time
            return (1.0 / self.N) * jnp.einsum('n,tn->t', params.w, r) + params.b

    def simulate_trial(
        self,
        params: RNNParams,
        trial_input: Dict[str, jnp.ndarray],
        integ_cfg: IntegratorConfig,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Simulate a single trial.

        Args:
            params: Network parameters
            trial_input: Dict with 'times_u' (T_u,) and 'u_seq' (T_u, d_in)
            integ_cfg: Integrator configuration

        Returns:
            times: Time points (n_steps,)
            xs: State trajectory (n_steps, N)
            ys: Readout trajectory (n_steps,)
        """
        # Create input interpolator
        u_of_t = make_input_interpolator(
            trial_input['times_u'],
            trial_input['u_seq']
        )

        # Initial state (zeros)
        x0 = jnp.zeros(self.N)

        # Integrate dynamics
        times, xs = integrate_rnn_dynamics(
            self.rhs,
            x0,
            args=(params, u_of_t),
            integ_cfg=integ_cfg,
        )

        # Compute readout
        ys = self.readout(xs, params)

        return times, xs, ys

    def simulate_trial_fast(
        self,
        params: RNNParams,
        u_seq: jnp.ndarray,
        dt: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Fast trial simulation using Euler integration (for training).

        This avoids the overhead of diffrax for simple cases.

        Args:
            params: Network parameters
            u_seq: Input sequence (T, d_in)
            dt: Time step

        Returns:
            xs: State trajectory (T, N)
            ys: Readout trajectory (T,)
        """
        T = u_seq.shape[0]

        # Compute J matrix once
        J = self.compute_J(params, self.g, self.N, self.training_mode)

        def step_fn(x, u_t):
            """One Euler step."""
            r = self.phi(x)
            dxdt = (-x + J @ r + params.B @ u_t) / self.tau
            x_next = x + dt * dxdt
            y = (1.0 / self.N) * jnp.dot(params.w, self.phi(x_next)) + params.b
            return x_next, (x_next, y)

        # Initial state
        x0 = jnp.zeros(self.N)

        # Run simulation
        _, (xs, ys) = jax.lax.scan(step_fn, x0, u_seq)

        return xs, ys

    def get_trainable_params(self, params: RNNParams, training_cfg: TrainingConfig) -> Dict[str, jnp.ndarray]:
        """
        Get dictionary of trainable parameters.

        Args:
            params: Full parameter set
            training_cfg: Training configuration specifying what to train

        Returns:
            Dictionary of trainable parameters
        """
        trainable = {}

        if training_cfg.training_mode == "full_rank":
            # In full_rank mode, train J directly
            if params.J is not None:
                trainable['J'] = params.J
        else:
            # In low_rank mode, train M and N_lr
            if training_cfg.train_M:
                trainable['M'] = params.M
            if training_cfg.train_N:
                trainable['N_lr'] = params.N_lr

        # B and w are trainable in both modes
        if training_cfg.train_B:
            trainable['B'] = params.B
        if training_cfg.train_w:
            trainable['w'] = params.w
            trainable['b'] = params.b
        return trainable

    def update_params(
        self,
        params: RNNParams,
        trainable_updates: Dict[str, jnp.ndarray],
        training_cfg: TrainingConfig
    ) -> RNNParams:
        """
        Update parameters with new trainable values.

        Args:
            params: Current full parameter set
            trainable_updates: Updated trainable parameters
            training_cfg: Training configuration

        Returns:
            Updated parameter set
        """
        if training_cfg.training_mode == "full_rank":
            # In full_rank mode, update J directly
            J = trainable_updates.get('J', params.J)
            M = params.M
            N_lr = params.N_lr
        else:
            # In low_rank mode, update M and N_lr
            M = trainable_updates.get('M', params.M) if training_cfg.train_M else params.M
            N_lr = trainable_updates.get('N_lr', params.N_lr) if training_cfg.train_N else params.N_lr
            J = params.J

        B = trainable_updates.get('B', params.B) if training_cfg.train_B else params.B
        w = trainable_updates.get('w', params.w) if training_cfg.train_w else params.w
        b = trainable_updates.get('b', params.b) if training_cfg.train_w else params.b

        return RNNParams(C=params.C, M=M, N_lr=N_lr, B=B, w=w, b=b, J=J)


def create_rnn_and_params(
    cfg: RNNConfig,
    key: jax.random.PRNGKey,
    training_mode: str = "low_rank"
) -> Tuple[LowRankRNN, RNNParams]:
    """
    Create RNN model and initial parameters.

    Args:
        cfg: RNN configuration
        key: Random key
        training_mode: "low_rank" or "full_rank"

    Returns:
        model: LowRankRNN instance
        params: Initial parameters
    """
    model = LowRankRNN(cfg, key, training_mode)
    return model, model.params


def count_parameters(params: RNNParams, trainable_only: bool = True, training_mode: str = "low_rank") -> int:
    """
    Count number of parameters.

    Args:
        params: Network parameters
        trainable_only: If True, exclude fixed parameters
        training_mode: "low_rank" or "full_rank"

    Returns:
        Total number of parameters
    """
    total = 0

    if training_mode == "full_rank":
        # In full_rank mode, J is trainable
        if params.J is not None:
            total += params.J.size
        if not trainable_only:
            total += params.C.size
            total += params.M.size
            total += params.N_lr.size
    else:
        # In low_rank mode, M and N_lr are trainable, C is fixed
        if not trainable_only:
            total += params.C.size
        total += params.M.size
        total += params.N_lr.size

    # B, w, b are always counted
    total += params.B.size
    total += params.w.size
    total += params.b.size

    return total
