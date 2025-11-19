"""Plot trajectory visualizations for trained RNN."""

import os
from typing import Dict, List, Optional
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.models.lowrank_rnn import LowRankRNN, RNNParams
from src.data.contextual_switch_dataset import ContextualSwitchDataset
from src.config import ExperimentConfig


def plot_trial_trajectories(
    model: LowRankRNN,
    params: RNNParams,
    dataset: ContextualSwitchDataset,
    cfg: ExperimentConfig,
    n_trials: int = 4,
    save_path: Optional[str] = None,
    key: Optional[jax.random.PRNGKey] = None,
):
    """
    Plot output trajectories for sample trials.

    Args:
        model: Trained LowRankRNN model
        params: Trained parameters
        dataset: Dataset instance
        cfg: Experiment configuration
        n_trials: Number of trials per condition
        save_path: Path to save figure
        key: Random key for sampling
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    dt = cfg.integrator.dt
    T_burn = cfg.task.T_burn

    # Sample trials for each condition
    conditions = [
        (1, +1, 'Context 1, s1 > 0'),
        (1, -1, 'Context 1, s1 < 0'),
        (2, +1, 'Context 2, s2 > 0'),
        (2, -1, 'Context 2, s2 < 0'),
    ]

    for idx, (context, sign, title) in enumerate(conditions):
        ax = axes[idx // 2, idx % 2]

        for trial_idx in range(n_trials):
            key, trial_key = jax.random.split(key)

            # Sample trial with specific condition
            trial = sample_conditioned_trial(dataset, trial_key, context, sign)

            # Simulate
            _, ys = model.simulate_trial_fast(params, trial['u_seq'], dt)

            # Plot
            times = np.arange(len(ys)) * dt
            label = f'Target: {trial["label"]:.0f}' if trial_idx == 0 else None
            ax.plot(times, ys, alpha=0.7, label=label)

        # Add burn-in marker
        ax.axvline(T_burn, color='gray', linestyle='--', alpha=0.5, label='Stimulus onset')

        # Add target lines
        ax.axhline(1, color='green', linestyle=':', alpha=0.5)
        ax.axhline(-1, color='red', linestyle=':', alpha=0.5)

        ax.set_xlabel('Time')
        ax.set_ylabel('Output y(t)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim([-2, 2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()

    plt.close()


def sample_conditioned_trial(
    dataset: ContextualSwitchDataset,
    key: jax.random.PRNGKey,
    target_context: int,
    target_sign: int,
    max_attempts: int = 1000
) -> Dict:
    """
    Sample a trial with specific context and relevant stimulus sign.

    Args:
        dataset: Dataset instance
        key: Random key
        target_context: Desired context (1 or 2)
        target_sign: Desired sign of relevant stimulus
        max_attempts: Maximum sampling attempts

    Returns:
        Trial dict
    """
    for _ in range(max_attempts):
        key, trial_key = jax.random.split(key)
        trial = dataset.sample_trial(trial_key)

        # Check if conditions match
        if trial['context'] == target_context:
            if target_context == 1 and jnp.sign(trial['stim'][0]) == target_sign:
                return trial
            elif target_context == 2 and jnp.sign(trial['stim'][1]) == target_sign:
                return trial

    # If we couldn't find a match, return the last trial
    return trial


def plot_state_projections(
    model: LowRankRNN,
    params: RNNParams,
    dataset: ContextualSwitchDataset,
    cfg: ExperimentConfig,
    n_trials: int = 10,
    save_path: Optional[str] = None,
    key: Optional[jax.random.PRNGKey] = None,
):
    """
    Plot state trajectories projected onto low-rank vectors.

    Args:
        model: Trained model
        params: Trained parameters
        dataset: Dataset
        cfg: Config
        n_trials: Trials per condition
        save_path: Save path
        key: Random key
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    dt = cfg.integrator.dt
    R = cfg.rnn.R

    # Get projection vectors (columns of M)
    M = params.M
    N_lr = params.N_lr

    fig, axes = plt.subplots(1, R, figsize=(6 * R, 5))
    if R == 1:
        axes = [axes]

    colors = {'ctx1_pos': 'blue', 'ctx1_neg': 'lightblue',
              'ctx2_pos': 'red', 'ctx2_neg': 'pink'}

    for r in range(R):
        ax = axes[r]
        proj_vec = M[:, r]

        # Sample and plot trials
        for context in [1, 2]:
            for sign in [+1, -1]:
                if context == 1:
                    color = colors[f'ctx1_{"pos" if sign > 0 else "neg"}']
                else:
                    color = colors[f'ctx2_{"pos" if sign > 0 else "neg"}']

                for _ in range(n_trials):
                    key, trial_key = jax.random.split(key)
                    trial = sample_conditioned_trial(dataset, trial_key, context, sign)

                    xs, _ = model.simulate_trial_fast(params, trial['u_seq'], dt)

                    # Project onto M column
                    proj = jnp.dot(xs, proj_vec)
                    times = np.arange(len(proj)) * dt

                    ax.plot(times, proj, color=color, alpha=0.3)

        ax.axvline(cfg.task.T_burn, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Projection onto M[:, {r}]')
        ax.set_title(f'Low-rank component {r+1}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved projection plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(
    logs: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """
    Plot training loss and accuracy curves.

    Args:
        logs: Training logs dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax = axes[0]
    if 'train_loss' in logs and logs['train_loss']:
        ax.plot(logs.get('iteration', range(len(logs['train_loss']))),
                logs['train_loss'], label='Train')
    if 'val_loss' in logs and logs['val_loss']:
        eval_iters = logs.get('iteration', [])
        if len(eval_iters) > len(logs['val_loss']):
            # Eval is less frequent
            step = len(eval_iters) // len(logs['val_loss'])
            eval_iters = eval_iters[::step][:len(logs['val_loss'])]
        ax.plot(eval_iters, logs['val_loss'], label='Val')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.set_yscale('log')

    # Accuracy plot
    ax = axes[1]
    if 'train_accuracy' in logs and logs['train_accuracy']:
        ax.plot(logs.get('iteration', range(len(logs['train_accuracy']))),
                logs['train_accuracy'], label='Train')
    if 'val_accuracy' in logs and logs['val_accuracy']:
        eval_iters = logs.get('iteration', [])
        if len(eval_iters) > len(logs['val_accuracy']):
            step = len(eval_iters) // len(logs['val_accuracy'])
            eval_iters = eval_iters[::step][:len(logs['val_accuracy'])]
        ax.plot(eval_iters, logs['val_accuracy'], label='Val')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Example usage
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    # Load config and params
    cfg = ExperimentConfig.from_yaml(os.path.join(args.results_dir, 'config.yaml'))

    with open(os.path.join(args.results_dir, 'params.pkl'), 'rb') as f:
        params_dict = pickle.load(f)

    params = RNNParams(
        C=jnp.array(params_dict['C']),
        M=jnp.array(params_dict['M']),
        N_lr=jnp.array(params_dict['N_lr']),
        B=jnp.array(params_dict['B']),
        w=jnp.array(params_dict['w']),
        b=jnp.array(params_dict['b']),
    )

    # Create model and dataset
    model = LowRankRNN(cfg.rnn, jax.random.PRNGKey(0))
    model.params = params  # Override with loaded params
    dataset = ContextualSwitchDataset(cfg.task, cfg.integrator, jax.random.PRNGKey(0))

    # Plot
    output_dir = args.output_dir or args.results_dir
    plot_trial_trajectories(model, params, dataset, cfg,
                            save_path=os.path.join(output_dir, 'trajectories.png'))
    plot_state_projections(model, params, dataset, cfg,
                           save_path=os.path.join(output_dir, 'projections.png'))

    # Load and plot training curves
    with open(os.path.join(args.results_dir, 'logs.json'), 'r') as f:
        import json
        logs = json.load(f)
    plot_training_curves(logs, save_path=os.path.join(output_dir, 'training_curves.png'))
