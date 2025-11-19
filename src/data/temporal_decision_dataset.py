"""Temporal decision task dataset generation (Interpolating Go-No-Go)."""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TemporalDecisionTaskConfig:
    """Configuration for the temporal decision task."""
    # Time parameters
    dt: float = 0.01  # Time step (seconds)
    T_trial: float = 1.0  # Trial duration (seconds)

    # Stimulus timing
    t_stim_on: float = 0.2  # Stimulus onset (seconds)
    t_stim_off: float = 0.7  # Stimulus offset (seconds)

    # Response timing
    t_response_on: float = 0.8  # Response window start (seconds)
    t_response_off: float = 1.0  # Response window end (seconds)

    # Stimulus distribution parameters
    mu1: float = 0.0  # Mean of stimulus 1
    sigma1: float = 1.0  # Std of stimulus 1
    mu2: float = 0.0  # Mean of stimulus 2
    sigma2: float = 1.0  # Std of stimulus 2

    # Input noise
    input_noise_std: float = 0.0  # Temporal noise on input signals

    # Decision threshold
    theta: float = 0.0  # Threshold for Go/No-Go

    # Label format (legacy for compatibility)
    label_type: str = "binary"  # "binary" for {0, 1}, "pm1" for ±1

    # Loss type
    loss_type: str = "mse"  # "mse" for regression, "bce" for classification

    # Context sampling for train/test split
    # If None, sample uniformly from [0, 1]
    # Otherwise, sample from these discrete values
    train_contexts: tuple = None  # e.g., (0.0, 0.25, 0.5, 0.75, 1.0)
    test_contexts: tuple = None   # e.g., (0.125, 0.375, 0.625, 0.875)


class TemporalDecisionDataset:
    """
    Dataset for the temporal decision task (Evidence Integration Regression).

    Task description:
        - Two time-varying input features u1(t), u2(t)
        - Context cue c determines which feature is relevant
        - Network outputs g_bar: the integrated evidence over stimulus window
        - For c=0: only u1 matters, g_bar = mean(u1(t))
        - For c=1: only u2 matters, g_bar = mean(u2(t))
        - For intermediate c: linear mixture, g_bar = mean((1-c)*u1(t) + c*u2(t))

    Input channels (d_in = 3):
        - Channel 0: u1 (stimulus feature 1)
        - Channel 1: u2 (stimulus feature 2)
        - Channel 2: c (context cue)

    Trial structure:
        - t in [0, t_stim_on): No stimulus, context present
        - t in [t_stim_on, t_stim_off]: Stimulus window (integrate evidence)
        - t in [t_response_on, t_response_off]: Response window (output g_bar)
    """

    def __init__(
        self,
        task_cfg: TemporalDecisionTaskConfig,
        rng_key: jax.random.PRNGKey
    ):
        """
        Initialize the dataset.

        Args:
            task_cfg: Task configuration
            rng_key: JAX random key
        """
        self.task_cfg = task_cfg
        self.key = rng_key

        # Compute time parameters
        self.dt = task_cfg.dt
        self.T = task_cfg.T_trial
        self.n_steps = int(self.T / self.dt)
        self.times = jnp.linspace(0, self.T - self.dt, self.n_steps)

        # Compute indices for different periods
        self.stim_on_idx = int(task_cfg.t_stim_on / self.dt)
        self.stim_off_idx = int(task_cfg.t_stim_off / self.dt)
        self.response_on_idx = int(task_cfg.t_response_on / self.dt)
        self.response_off_idx = int(task_cfg.t_response_off / self.dt)

        # Number of steps in stimulus window
        self.n_stim_steps = self.stim_off_idx - self.stim_on_idx

    def sample_trial(self, key: jax.random.PRNGKey, use_test_contexts: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Sample a single trial.

        Args:
            key: Random key for sampling
            use_test_contexts: If True, sample from test_contexts instead of train_contexts

        Returns:
            Dictionary with:
                - times: Time points (n_steps,)
                - u_seq: Input sequence (n_steps, 3) - [u1, u2, c]
                - label: Trial label (scalar, Go=1/No-Go=0)
                - y_time: Time-resolved target (n_steps,)
                - context: Context value c
                - a1: Stimulus 1 amplitude
                - a2: Stimulus 2 amplitude
                - g_bar: Average evidence
        """
        keys = jax.random.split(key, 4)

        # Sample context
        if use_test_contexts and self.task_cfg.test_contexts is not None:
            # Sample from test contexts (held-out)
            contexts = jnp.array(self.task_cfg.test_contexts)
            idx = jax.random.randint(keys[0], (), 0, len(contexts))
            context = contexts[idx]
        elif not use_test_contexts and self.task_cfg.train_contexts is not None:
            # Sample from train contexts
            contexts = jnp.array(self.task_cfg.train_contexts)
            idx = jax.random.randint(keys[0], (), 0, len(contexts))
            context = contexts[idx]
        else:
            # Sample uniformly from [0, 1]
            context = jax.random.uniform(keys[0], ())

        # Sample stimulus amplitudes
        a1 = jax.random.normal(keys[1], ()) * self.task_cfg.sigma1 + self.task_cfg.mu1
        a2 = jax.random.normal(keys[2], ()) * self.task_cfg.sigma2 + self.task_cfg.mu2

        # Build input sequence (with noise)
        u_seq = self._build_input_sequence(a1, a2, context, keys[3])

        # Compute evidence g(t) = (1-c)*u1(t) + c*u2(t)
        # Note: use clean signal for computing label (before noise)
        u1_clean = jnp.zeros(self.n_steps).at[self.stim_on_idx:self.stim_off_idx].set(a1)
        u2_clean = jnp.zeros(self.n_steps).at[self.stim_on_idx:self.stim_off_idx].set(a2)
        g = (1 - 2*context) * u1_clean + (2*context - 1) * u2_clean

        # Compute average evidence over stimulus window
        g_stim = g[self.stim_on_idx:self.stim_off_idx]
        g_bar = jnp.mean(g_stim)

        # Compute trial label (legacy, for compatibility)
        label = (g_bar > self.task_cfg.theta).astype(jnp.float32)

        if self.task_cfg.label_type == "pm1":
            label = 2 * label - 1  # Map 0 -> -1, 1 -> 1

        # Build time-resolved target (now uses g_bar instead of binary label)
        y_time = self._build_target_sequence(g_bar)

        return {
            'times': self.times,
            'u_seq': u_seq,
            'label': label,
            'y_time': y_time,
            'context': context,
            'a1': a1,
            'a2': a2,
            'g_bar': g_bar,
        }

    def sample_trial_fixed_context(
        self,
        key: jax.random.PRNGKey,
        context: float,
        a1: Optional[float] = None,
        a2: Optional[float] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample a trial with fixed context and optionally fixed amplitudes.

        Args:
            key: Random key for sampling
            context: Fixed context value
            a1: Optional fixed amplitude for stimulus 1
            a2: Optional fixed amplitude for stimulus 2

        Returns:
            Trial dictionary
        """
        keys = jax.random.split(key, 3)

        # Use provided amplitudes or sample
        if a1 is None:
            a1 = jax.random.normal(keys[0], ()) * self.task_cfg.sigma1 + self.task_cfg.mu1
        if a2 is None:
            a2 = jax.random.normal(keys[1], ()) * self.task_cfg.sigma2 + self.task_cfg.mu2

        # Build input sequence (with noise)
        u_seq = self._build_input_sequence(a1, a2, context, keys[2])

        # Compute evidence using clean signal (before noise)
        u1_clean = jnp.zeros(self.n_steps).at[self.stim_on_idx:self.stim_off_idx].set(a1)
        u2_clean = jnp.zeros(self.n_steps).at[self.stim_on_idx:self.stim_off_idx].set(a2)
        g = (1 - context) * u1_clean + context * u2_clean

        # Compute average evidence over stimulus window
        g_stim = g[self.stim_on_idx:self.stim_off_idx]
        g_bar = jnp.mean(g_stim)

        # Compute trial label (legacy, for compatibility)
        label = (g_bar > self.task_cfg.theta).astype(jnp.float32)

        if self.task_cfg.label_type == "pm1":
            label = 2 * label - 1

        # Build time-resolved target (now uses g_bar instead of binary label)
        y_time = self._build_target_sequence(g_bar)

        return {
            'times': self.times,
            'u_seq': u_seq,
            'label': label,
            'y_time': y_time,
            'context': context,
            'a1': a1,
            'a2': a2,
            'g_bar': g_bar,
        }

    def _build_input_sequence(
        self,
        a1: float,
        a2: float,
        context: float,
        noise_key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Build input sequence for a trial.

        Args:
            a1: Stimulus 1 amplitude
            a2: Stimulus 2 amplitude
            context: Context value
            noise_key: Optional key for generating input noise

        Returns:
            u_seq: Input sequence (n_steps, 3)
        """
        # Initialize input array
        u_seq = jnp.zeros((self.n_steps, 3))

        # Set context for entire trial
        u_seq = u_seq.at[:, 2].set(context)

        # Set stimuli only during stimulus window
        u_seq = u_seq.at[self.stim_on_idx:self.stim_off_idx, 0].set(a1)
        u_seq = u_seq.at[self.stim_on_idx:self.stim_off_idx, 1].set(a2)

        # Add temporal noise to stimulus channels
        if noise_key is not None and self.task_cfg.input_noise_std > 0:
            noise = jax.random.normal(noise_key, (self.n_steps, 2)) * self.task_cfg.input_noise_std
            # Only add noise during stimulus window
            noise_mask = jnp.zeros((self.n_steps, 2))
            noise_mask = noise_mask.at[self.stim_on_idx:self.stim_off_idx, :].set(1.0)
            u_seq = u_seq.at[:, :2].add(noise * noise_mask)

        return u_seq

    def _build_target_sequence(self, g_bar: float) -> jnp.ndarray:
        """
        Build time-resolved target sequence.

        Target is 0 before/during stimulus, and g_bar during response window.

        Args:
            g_bar: Integrated evidence (target value for regression)

        Returns:
            y_time: Target sequence (n_steps,)
        """
        y_time = jnp.zeros(self.n_steps)

        # Set target to g_bar in response window
        y_time = y_time.at[self.response_on_idx:self.response_off_idx].set(g_bar)

        return y_time

    def sample_batch(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        use_test_contexts: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample a batch of trials.

        Args:
            key: Random key
            batch_size: Number of trials
            use_test_contexts: If True, sample from test_contexts (held-out)

        Returns:
            Batched dictionary with:
                - times: (n_steps,) - same for all
                - u_seq: (batch_size, n_steps, 3)
                - labels: (batch_size,)
                - y_time: (batch_size, n_steps)
                - contexts: (batch_size,)
                - a1s: (batch_size,)
                - a2s: (batch_size,)
                - g_bars: (batch_size,)
        """
        keys = jax.random.split(key, batch_size)

        # Vectorize sample_trial with use_test_contexts
        from functools import partial
        sample_fn = partial(self.sample_trial, use_test_contexts=use_test_contexts)
        batch_fn = jax.vmap(sample_fn)
        batch = batch_fn(keys)

        return {
            'times': self.times,
            'u_seq': batch['u_seq'],
            'labels': batch['label'],
            'y_time': batch['y_time'],
            'contexts': batch['context'],
            'a1s': batch['a1'],
            'a2s': batch['a2'],
            'g_bars': batch['g_bar'],
        }

    def get_n_steps(self) -> int:
        """Get number of time steps per trial."""
        return self.n_steps

    def get_avg_window_indices(self) -> Tuple[int, int]:
        """
        Get indices for the response/averaging window.

        Returns:
            (start_idx, end_idx) for slicing
        """
        return self.response_on_idx, self.response_off_idx

    def get_stim_window_indices(self) -> Tuple[int, int]:
        """
        Get indices for the stimulus window.

        Returns:
            (start_idx, end_idx) for slicing
        """
        return self.stim_on_idx, self.stim_off_idx


def plot_single_trial(
    trial: Dict[str, jnp.ndarray],
    task_cfg: TemporalDecisionTaskConfig,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a single trial visualization.

    Args:
        trial: Trial dictionary from sample_trial
        task_cfg: Task configuration
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Convert JAX arrays to numpy
    times = np.array(trial['times'])
    u_seq = np.array(trial['u_seq'])
    y_time = np.array(trial['y_time'])
    context = float(trial['context'])
    label = float(trial['label'])
    a1 = float(trial['a1'])
    a2 = float(trial['a2'])
    g_bar = float(trial['g_bar'])

    # Extract signals
    u1 = u_seq[:, 0]
    u2 = u_seq[:, 1]
    c = u_seq[:, 2]

    # Compute evidence
    g = (1 - context) * u1 + context * u2

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Subplot 1: Stimulus features
    ax1 = axes[0]
    ax1.plot(times, u1, 'b-', label=f'u1 (a1={a1:.2f})', linewidth=1.5)
    ax1.plot(times, u2, 'r-', label=f'u2 (a2={a2:.2f})', linewidth=1.5)
    ax1.axvline(task_cfg.t_stim_on, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(task_cfg.t_stim_off, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Stimulus')
    ax1.legend(loc='upper right')
    ax1.set_title('Stimulus Features')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Context and evidence
    ax2 = axes[1]
    ax2.plot(times, c, 'g-', label=f'Context c={context:.2f}', linewidth=1.5)
    ax2.plot(times, g, 'm-', label=f'Evidence g(t)', linewidth=1.5)
    ax2.axhline(task_cfg.theta, color='k', linestyle='--', alpha=0.5,
                label=f'Threshold θ={task_cfg.theta:.2f}')
    ax2.axhline(g_bar, color='m', linestyle=':', alpha=0.7,
                label=f'g_bar={g_bar:.2f}')
    ax2.axvline(task_cfg.t_stim_on, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(task_cfg.t_stim_off, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Value')
    ax2.legend(loc='upper right')
    ax2.set_title('Context and Evidence')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Target output
    ax3 = axes[2]
    ax3.plot(times, y_time, 'k-', linewidth=2, label='Target')
    ax3.axvline(task_cfg.t_response_on, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(task_cfg.t_response_off, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(g_bar, color='m', linestyle=':', alpha=0.7, linewidth=2,
                label=f'g_bar={g_bar:.2f}')
    ax3.axvspan(task_cfg.t_stim_on, task_cfg.t_stim_off,
                alpha=0.1, color='blue', label='Stimulus window')
    ax3.axvspan(task_cfg.t_response_on, task_cfg.t_response_off,
                alpha=0.1, color='green', label='Response window')
    ax3.set_ylabel('Target (g_bar)')
    ax3.set_xlabel('Time (s)')

    ax3.legend(loc='upper right')
    ax3.set_title('Target Output (Integrated Evidence)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_interpolation_comparison(
    dataset: TemporalDecisionDataset,
    key: jax.random.PRNGKey,
    a1: float = 0.5,
    a2: float = -0.3,
    figsize: Tuple[float, float] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of three trials with c=0, 0.5, 1.0 using same stimuli.

    This visualizes how changing context changes the decision.

    Args:
        dataset: TemporalDecisionDataset instance
        key: Random key
        a1: Fixed stimulus 1 amplitude
        a2: Fixed stimulus 2 amplitude
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    task_cfg = dataset.task_cfg
    contexts = [0.0, 0.5, 1.0]

    # Generate trials
    keys = jax.random.split(key, 3)
    trials = []
    for i, c in enumerate(contexts):
        trial = dataset.sample_trial_fixed_context(keys[i], c, a1, a2)
        trials.append(trial)

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True)

    for col, (c, trial) in enumerate(zip(contexts, trials)):
        # Convert to numpy
        times = np.array(trial['times'])
        u_seq = np.array(trial['u_seq'])
        y_time = np.array(trial['y_time'])
        context = float(trial['context'])
        label = float(trial['label'])
        g_bar = float(trial['g_bar'])

        # Extract signals
        u1 = u_seq[:, 0]
        u2 = u_seq[:, 1]
        g = (1 - context) * u1 + context * u2

        # Row 1: Stimulus features
        ax1 = axes[0, col]
        ax1.plot(times, u1, 'b-', label='u1', linewidth=1.5)
        ax1.plot(times, u2, 'r-', label='u2', linewidth=1.5)
        ax1.axvline(task_cfg.t_stim_on, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(task_cfg.t_stim_off, color='gray', linestyle='--', alpha=0.5)
        if col == 0:
            ax1.set_ylabel('Stimulus')
        ax1.set_title(f'c = {c:.1f}')
        ax1.grid(True, alpha=0.3)
        if col == 2:
            ax1.legend(loc='upper right', fontsize=8)

        # Row 2: Evidence
        ax2 = axes[1, col]
        ax2.plot(times, g, 'm-', label='g(t)', linewidth=1.5)
        ax2.axhline(task_cfg.theta, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(g_bar, color='m', linestyle=':', alpha=0.7)
        ax2.axvline(task_cfg.t_stim_on, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(task_cfg.t_stim_off, color='gray', linestyle='--', alpha=0.5)
        if col == 0:
            ax2.set_ylabel('Evidence')
        ax2.text(0.05, 0.9, f'g_bar={g_bar:.2f}', transform=ax2.transAxes,
                fontsize=9, va='top')
        ax2.grid(True, alpha=0.3)

        # Row 3: Target
        ax3 = axes[2, col]
        ax3.plot(times, y_time, 'k-', linewidth=2, label='Target')
        ax3.axhline(g_bar, color='m', linestyle=':', alpha=0.7, linewidth=2)
        ax3.axvspan(task_cfg.t_stim_on, task_cfg.t_stim_off,
                    alpha=0.1, color='blue')
        ax3.axvspan(task_cfg.t_response_on, task_cfg.t_response_off,
                    alpha=0.1, color='green')
        if col == 0:
            ax3.set_ylabel('Target (g_bar)')
        ax3.set_xlabel('Time (s)')

        # Add g_bar value annotation
        ax3.text(0.5, 0.9, f'Target: {g_bar:.2f}', transform=ax3.transAxes,
                fontsize=10, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle(f'Evidence Integration Regression Task (a1={a1:.2f}, a2={a2:.2f})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_temporal_decision_dataset(
    task_cfg: TemporalDecisionTaskConfig,
    key: jax.random.PRNGKey
) -> TemporalDecisionDataset:
    """
    Create a temporal decision dataset.

    Args:
        task_cfg: Task configuration
        key: Random key

    Returns:
        Dataset instance
    """
    return TemporalDecisionDataset(task_cfg, key)


# Example usage and demonstration
if __name__ == "__main__":
    # Create configuration
    task_cfg = TemporalDecisionTaskConfig(
        dt=0.01,
        T_trial=1.0,
        t_stim_on=0.2,
        t_stim_off=0.7,
        t_response_on=0.8,
        t_response_off=1.0,
        mu1=0.0,
        sigma1=1.0,
        mu2=0.0,
        sigma2=1.0,
        theta=0.0,
        label_type="binary",
        loss_type="bce"
    )

    # Create dataset
    key = jax.random.PRNGKey(42)
    dataset = create_temporal_decision_dataset(task_cfg, key)

    print("Temporal Decision Task Dataset")
    print("=" * 50)
    print(f"Time step: {task_cfg.dt} s")
    print(f"Trial duration: {task_cfg.T_trial} s")
    print(f"Number of time steps: {dataset.get_n_steps()}")
    print(f"Stimulus window: [{task_cfg.t_stim_on}, {task_cfg.t_stim_off}] s")
    print(f"Response window: [{task_cfg.t_response_on}, {task_cfg.t_response_off}] s")
    print(f"Decision threshold: {task_cfg.theta}")
    print()

    # Sample and display a single trial
    key, subkey = jax.random.split(key)
    trial = dataset.sample_trial(subkey)

    print("Example trial:")
    print(f"  Context c: {float(trial['context']):.3f}")
    print(f"  Stimulus 1 amplitude (a1): {float(trial['a1']):.3f}")
    print(f"  Stimulus 2 amplitude (a2): {float(trial['a2']):.3f}")
    print(f"  Average evidence (g_bar): {float(trial['g_bar']):.3f}")
    print(f"  Decision: {'Go' if float(trial['label']) > 0.5 else 'No-Go'}")
    print(f"  Input shape: {trial['u_seq'].shape}")
    print(f"  Target shape: {trial['y_time'].shape}")
    print()

    # Sample a batch
    key, subkey = jax.random.split(key)
    batch = dataset.sample_batch(subkey, batch_size=32)

    print("Batch statistics:")
    print(f"  Batch size: 32")
    print(f"  u_seq shape: {batch['u_seq'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  Go trials: {int(jnp.sum(batch['labels']))} / 32")
    print(f"  Mean context: {float(jnp.mean(batch['contexts'])):.3f}")
    print()

    # Plot single trial
    import os
    figs_dir = "figs/temporal_decision"
    os.makedirs(figs_dir, exist_ok=True)

    key, subkey = jax.random.split(key)
    trial = dataset.sample_trial(subkey)
    fig1 = plot_single_trial(trial, task_cfg, save_path=f"{figs_dir}/single_trial_example.png")
    print(f"Saved single trial plot to: {figs_dir}/single_trial_example.png")

    # Plot interpolation comparison
    key, subkey = jax.random.split(key)
    fig2 = plot_interpolation_comparison(
        dataset, subkey,
        a1=0.5, a2=-0.3,
        save_path=f"{figs_dir}/interpolation_comparison.png"
    )
    print(f"Saved interpolation comparison to: {figs_dir}/interpolation_comparison.png")

    plt.show()
