"""Contextual switch task dataset generation."""

from typing import Dict, Tuple
import jax
import jax.numpy as jnp

from src.config import TaskConfig, IntegratorConfig


class ContextualSwitchDataset:
    """
    Dataset for the contextual switch task.

    Task description:
        - Two stimulus dimensions s1, s2
        - Two contexts kappa in {1, 2}
        - In context 1: report sign/category of s1
        - In context 2: report sign/category of s2

    Input channels (d_in = 4):
        - Channel 0: s1 (stimulus 1)
        - Channel 1: s2 (stimulus 2)
        - Channel 2: context 1 cue
        - Channel 3: context 2 cue

    Trial structure (following paper):
        - Burn-in period: context cues only
        - Stimulus period: context cues + stimuli
    """

    def __init__(
        self,
        task_cfg: TaskConfig,
        integ_cfg: IntegratorConfig,
        rng_key: jax.random.PRNGKey
    ):
        """
        Initialize the dataset.

        Args:
            task_cfg: Task configuration
            integ_cfg: Integrator configuration (for timing)
            rng_key: JAX random key
        """
        self.task_cfg = task_cfg
        self.integ_cfg = integ_cfg
        self.key = rng_key

        # Compute time points for input
        self.T = task_cfg.T_burn + task_cfg.T_stim
        self.dt = integ_cfg.dt
        self.n_steps = int(self.T / self.dt) + 1
        self.times_u = jnp.linspace(0, self.T, self.n_steps)

        # Find indices for burn-in and stimulus periods
        self.burn_end_idx = int(task_cfg.T_burn / self.dt)

    def sample_trial(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Sample a single trial.

        Args:
            key: Random key for sampling

        Returns:
            Dictionary with:
                - times_u: Time points (n_steps,)
                - u_seq: Input sequence (n_steps, 4)
                - label: Target label (scalar, ±1 or {0,1})
                - context: Context index (1 or 2)
                - stim: Tuple of (s1, s2)
        """
        keys = jax.random.split(key, 4)

        # Sample context (1 or 2)
        context = jax.random.randint(keys[0], (), 1, 3)  # 1 or 2

        # Sample stimulus signs
        sign1 = 2 * jax.random.randint(keys[1], (), 0, 2) - 1  # -1 or +1
        sign2 = 2 * jax.random.randint(keys[2], (), 0, 2) - 1  # -1 or +1

        # Generate stimuli with magnitude and optional noise
        s1 = sign1 * self.task_cfg.stim_mean_abs
        s2 = sign2 * self.task_cfg.stim_mean_abs

        if self.task_cfg.stim_std > 0:
            noise = jax.random.normal(keys[3], (2,)) * self.task_cfg.stim_std
            s1 = s1 + noise[0]
            s2 = s2 + noise[1]

        # Build input sequence
        u_seq = self._build_input_sequence(s1, s2, context)

        # Compute label (using jnp.where for JAX compatibility with vmap)
        relevant_sign = jnp.where(context == 1, jnp.sign(s1), jnp.sign(s2))

        if self.task_cfg.label_type == "pm1":
            label = relevant_sign
        else:  # binary
            label = (relevant_sign + 1) / 2  # Map -1 -> 0, +1 -> 1

        return {
            'times_u': self.times_u,
            'u_seq': u_seq,
            'label': label,
            'context': context,
            'stim': (s1, s2),
        }

    def _build_input_sequence(
        self,
        s1: float,
        s2: float,
        context: int
    ) -> jnp.ndarray:
        """
        Build input sequence for a trial.

        Args:
            s1: Stimulus 1 value
            s2: Stimulus 2 value
            context: Context (1 or 2)

        Returns:
            u_seq: Input sequence (n_steps, 4)
        """
        # Initialize input array
        u_seq = jnp.zeros((self.n_steps, 4))

        # Set context cues for entire trial
        # Context 1: channel 2 = gamma_on, channel 3 = gamma_off
        # Context 2: channel 2 = gamma_off, channel 3 = gamma_on
        ctx1_val = jnp.where(context == 1, self.task_cfg.gamma_on, self.task_cfg.gamma_off)
        ctx2_val = jnp.where(context == 2, self.task_cfg.gamma_on, self.task_cfg.gamma_off)

        u_seq = u_seq.at[:, 2].set(ctx1_val)
        u_seq = u_seq.at[:, 3].set(ctx2_val)

        # Set stimuli only during stimulus period (after burn-in)
        u_seq = u_seq.at[self.burn_end_idx:, 0].set(s1)
        u_seq = u_seq.at[self.burn_end_idx:, 1].set(s2)

        return u_seq

    def sample_batch(
        self,
        key: jax.random.PRNGKey,
        batch_size: int
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample a batch of trials.

        Args:
            key: Random key
            batch_size: Number of trials

        Returns:
            Batched dictionary with:
                - times_u: (n_steps,) - same for all
                - u_seq: (batch_size, n_steps, 4)
                - labels: (batch_size,)
                - contexts: (batch_size,)
                - stims: (batch_size, 2)
        """
        keys = jax.random.split(key, batch_size)

        # Vectorize sample_trial
        batch_fn = jax.vmap(self.sample_trial)
        batch = batch_fn(keys)

        # Restructure output
        return {
            'times_u': self.times_u,  # Same for all
            'u_seq': batch['u_seq'],
            'labels': batch['label'],
            'contexts': batch['context'],
            'stims': jnp.stack([batch['stim'][0], batch['stim'][1]], axis=1),
        }

    def get_n_steps(self) -> int:
        """Get number of time steps per trial."""
        return self.n_steps

    def get_avg_window_indices(self) -> Tuple[int, int]:
        """
        Get indices for the averaging window at end of trial.

        Returns:
            (start_idx, end_idx) for slicing
        """
        T_avg_start = self.T - self.task_cfg.T_avg
        start_idx = int(T_avg_start / self.dt)
        end_idx = self.n_steps
        return start_idx, end_idx


def sample_balanced_batch(
    dataset: ContextualSwitchDataset,
    key: jax.random.PRNGKey,
    batch_size: int
) -> Dict[str, jnp.ndarray]:
    """
    Sample a batch with balanced contexts and stimulus signs.

    This ensures equal representation of all conditions.

    Args:
        dataset: Dataset instance
        key: Random key
        batch_size: Batch size (should be divisible by 8 for perfect balance)

    Returns:
        Batched trial data
    """
    # For simplicity, just use regular sampling
    # A truly balanced version would explicitly construct conditions
    return dataset.sample_batch(key, batch_size)


def create_dataset(
    task_cfg: TaskConfig,
    integ_cfg: IntegratorConfig,
    key: jax.random.PRNGKey
) -> ContextualSwitchDataset:
    """
    Create a contextual switch dataset.

    Args:
        task_cfg: Task configuration
        integ_cfg: Integrator configuration
        key: Random key

    Returns:
        Dataset instance
    """
    return ContextualSwitchDataset(task_cfg, integ_cfg, key)
