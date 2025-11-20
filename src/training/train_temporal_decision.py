"""Training script for the temporal decision task (Evidence Integration Regression)."""

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import yaml
import matplotlib.pyplot as plt
import numpy as np

from src.config import RNNConfig, IntegratorConfig, TrainingConfig
from src.models.lowrank_rnn import LowRankRNN, RNNParams, create_rnn_and_params, count_parameters
from src.data.temporal_decision_dataset import (
    TemporalDecisionDataset,
    TemporalDecisionTaskConfig,
    create_temporal_decision_dataset,
    plot_single_trial,
    plot_interpolation_comparison
)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    rnn_cfg = RNNConfig(**data.get('rnn', {}))
    training_cfg = TrainingConfig(**data.get('training', {}))

    # Load task config
    task_data = data.get('task', {})
    task_cfg = TemporalDecisionTaskConfig(**task_data)

    return rnn_cfg, task_cfg, training_cfg, data.get('name', 'temporal_decision')


def create_optimizer(training_cfg: TrainingConfig, n_iterations: int) -> optax.GradientTransformation:
    """Create optimizer with learning rate schedule."""
    schedule = optax.cosine_decay_schedule(
        init_value=training_cfg.learning_rate,
        decay_steps=n_iterations,
        alpha=0.01
    )

    if training_cfg.optimizer == "adam":
        opt = optax.adam(schedule)
    elif training_cfg.optimizer == "adamw":
        opt = optax.adamw(schedule, weight_decay=training_cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {training_cfg.optimizer}")

    if training_cfg.grad_clip > 0:
        opt = optax.chain(
            optax.clip_by_global_norm(training_cfg.grad_clip),
            opt
        )

    return opt


def make_train_step(model, dataset, rnn_cfg, task_cfg, training_cfg, n_iterations: int):
    """Create the training step function."""
    resp_start, resp_end = dataset.get_avg_window_indices()
    dt = dataset.dt
    loss_type = task_cfg.loss_type
    label_type = task_cfg.label_type

    optimizer = create_optimizer(training_cfg, n_iterations)

    def loss_fn(trainable_params, fixed_params, batch):
        params = RNNParams(
            C=fixed_params['C'],
            M=trainable_params.get('M', fixed_params.get('M', None)),
            N_lr=trainable_params.get('N_lr', fixed_params.get('N_lr', None)),
            B=trainable_params.get('B', fixed_params.get('B', None)),
            w=trainable_params.get('w', fixed_params.get('w', None)),
            b=trainable_params.get('b', fixed_params.get('b', jnp.zeros(()))),
            J=trainable_params.get('J', fixed_params.get('J', None)),
        )

        def single_trial(u_seq, target):
            _, ys = model.simulate_trial_fast(params, u_seq, dt)

            # Get outputs during response window
            y_resp = ys[resp_start:resp_end]

            # MSE loss: compute error at each time point in response window, then average
            loss = jnp.mean((y_resp - target) ** 2)

            # Return mean prediction for evaluation
            y_hat = jnp.mean(y_resp)

            return y_hat, loss

        y_hats, losses = jax.vmap(single_trial)(batch['u_seq'], batch['g_bars'])
        loss = jnp.mean(losses)

        return loss, y_hats

    def train_step(trainable_params, fixed_params, opt_state, batch):
        (loss, y_hats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            trainable_params, fixed_params, batch
        )

        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)

        # Compute MSE and correlation for regression
        mse = jnp.mean((y_hats - batch['g_bars']) ** 2)
        correlation = jnp.corrcoef(y_hats, batch['g_bars'])[0, 1]

        return trainable_params, opt_state, {'loss': loss, 'mse': mse, 'correlation': correlation}

    return jax.jit(train_step), optimizer


def make_eval_step(model, dataset, rnn_cfg, task_cfg):
    """Create evaluation step function."""
    resp_start, resp_end = dataset.get_avg_window_indices()
    dt = dataset.dt
    loss_type = task_cfg.loss_type
    label_type = task_cfg.label_type

    def eval_step(trainable_params, fixed_params, batch):
        params = RNNParams(
            C=fixed_params['C'],
            M=trainable_params.get('M', fixed_params.get('M', None)),
            N_lr=trainable_params.get('N_lr', fixed_params.get('N_lr', None)),
            B=trainable_params.get('B', fixed_params.get('B', None)),
            w=trainable_params.get('w', fixed_params.get('w', None)),
            b=trainable_params.get('b', fixed_params.get('b', jnp.zeros(()))),
            J=trainable_params.get('J', fixed_params.get('J', None)),
        )

        def single_trial(u_seq, target, context):
            _, ys = model.simulate_trial_fast(params, u_seq, dt)

            # Get outputs during response window
            y_resp = ys[resp_start:resp_end]

            # MSE loss: compute error at each time point in response window, then average
            loss = jnp.mean((y_resp - target) ** 2)

            # Return mean prediction for evaluation
            y_hat = jnp.mean(y_resp)

            return y_hat, loss

        y_hats, losses = jax.vmap(single_trial)(batch['u_seq'], batch['g_bars'], batch['contexts'])
        loss = jnp.mean(losses)

        # Compute MSE and correlation
        mse = jnp.mean((y_hats - batch['g_bars']) ** 2)
        correlation = jnp.corrcoef(y_hats, batch['g_bars'])[0, 1]

        # Compute MSE for different context ranges
        low_c_mask = batch['contexts'] < 0.5
        high_c_mask = batch['contexts'] >= 0.5

        low_c_mse = jnp.where(
            jnp.sum(low_c_mask) > 0,
            jnp.sum(((y_hats - batch['g_bars']) ** 2) * low_c_mask) / jnp.sum(low_c_mask),
            0.0
        )
        high_c_mse = jnp.where(
            jnp.sum(high_c_mask) > 0,
            jnp.sum(((y_hats - batch['g_bars']) ** 2) * high_c_mask) / jnp.sum(high_c_mask),
            0.0
        )

        return {
            'loss': loss,
            'mse': mse,
            'correlation': correlation,
            'low_c_mse': low_c_mse,
            'high_c_mse': high_c_mse,
            'predictions': y_hats,
            'targets': batch['g_bars'],
        }

    return jax.jit(eval_step)


def make_trial_output_fn(model, dataset, task_cfg):
    """Create function to get full trial outputs and hidden states for plotting."""
    dt = dataset.dt

    def get_trial_outputs(trainable_params, fixed_params, u_seq):
        params = RNNParams(
            C=fixed_params['C'],
            M=trainable_params.get('M', fixed_params.get('M', None)),
            N_lr=trainable_params.get('N_lr', fixed_params.get('N_lr', None)),
            B=trainable_params.get('B', fixed_params.get('B', None)),
            w=trainable_params.get('w', fixed_params.get('w', None)),
            b=trainable_params.get('b', fixed_params.get('b', jnp.zeros(()))),
            J=trainable_params.get('J', fixed_params.get('J', None)),
        )
        xs, ys = model.simulate_trial_fast(params, u_seq, dt)

        # For regression, return raw outputs (no sigmoid/tanh)
        return xs, ys

    return jax.jit(get_trial_outputs)


def plot_network_performance(
    model, trainable_params, fixed_params, dataset, task_cfg, epoch,
    output_dir, key, n_trials=6, n_neurons=20
):
    """
    Plot network performance on sample trials.

    Shows network output vs ground truth target for n_trials.
    If test_contexts is defined, shows trials from both train and test contexts.
    Also shows sampled neuron activities.
    """
    get_trial_outputs = make_trial_output_fn(model, dataset, task_cfg)

    # Determine how to split trials between train and test contexts
    has_test_contexts = (task_cfg.test_contexts is not None or
                          task_cfg.test_context_ranges is not None)
    if has_test_contexts:
        n_train_trials = (n_trials + 1) // 2  # First half from train
        n_test_trials = n_trials - n_train_trials  # Second half from test
    else:
        n_train_trials = n_trials
        n_test_trials = 0

    keys = jax.random.split(key, n_trials + 1)
    neuron_key = keys[0]
    trial_keys = keys[1:]

    fig, axes = plt.subplots(n_trials, 3, figsize=(15, 3 * n_trials))
    if n_trials == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_trials):
        # Sample from train or test contexts
        if i < n_train_trials:
            trial = dataset.sample_trial(trial_keys[i], use_test_contexts=False)
            context_type = "TRAIN"
        else:
            trial = dataset.sample_trial(trial_keys[i], use_test_contexts=True)
            context_type = "HELD-OUT"

        # Get network output and hidden states
        xs, y_pred = get_trial_outputs(trainable_params, fixed_params, trial['u_seq'])

        # Convert to numpy
        times = np.array(trial['times'])
        u_seq = np.array(trial['u_seq'])
        y_time = np.array(trial['y_time'])
        y_pred_np = np.array(y_pred)
        xs_np = np.array(xs)  # (n_steps, N)
        context = float(trial['context'])
        g_bar = float(trial['g_bar'])
        a1 = float(trial['a1'])
        a2 = float(trial['a2'])

        # Sample neurons to plot (same for all trials)
        N = xs_np.shape[1]
        neuron_indices = jax.random.choice(neuron_key, N, shape=(min(n_neurons, N),), replace=False)
        neuron_indices = np.array(neuron_indices)

        # Plot 1: Inputs
        ax1 = axes[i, 0]
        ax1.plot(times, u_seq[:, 0], 'b-', label='u1', linewidth=1.5)
        ax1.plot(times, u_seq[:, 1], 'r-', label='u2', linewidth=1.5)
        ax1.axvline(task_cfg.t_stim_on, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(task_cfg.t_stim_off, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Input')

        # Title with context type
        if has_test_contexts:
            ax1.set_title(f'[{context_type}] c={context:.3f}, a1={a1:.2f}, a2={a2:.2f}')
        else:
            ax1.set_title(f'Trial {i+1}: c={context:.2f}, a1={a1:.2f}, a2={a2:.2f}')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Network output vs target with neuron activities
        ax2 = axes[i, 1]

        # Plot sampled neuron activities (tanh of hidden states)
        neuron_activities = np.tanh(xs_np[:, neuron_indices])
        for j in range(len(neuron_indices)):
            ax2.plot(times, neuron_activities[:, j], '-', alpha=0.4, linewidth=0.8)

        # Plot target and readout with thicker lines
        ax2.plot(times, y_time, 'k-', label='Target', linewidth=3)
        ax2.plot(times, y_pred_np, 'b-', label='Readout', linewidth=2.5)
        ax2.axhline(g_bar, color='m', linestyle=':', alpha=0.7, linewidth=2)
        ax2.axvline(task_cfg.t_response_on, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(task_cfg.t_response_off, color='gray', linestyle='--', alpha=0.5)
        ax2.axvspan(task_cfg.t_response_on, task_cfg.t_response_off, alpha=0.1, color='green')
        ax2.set_ylabel('Activity')
        ax2.set_ylim(-1.2, 1.2)
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'{n_neurons} neurons + readout')

        # Compute prediction
        resp_start, resp_end = dataset.get_avg_window_indices()
        pred_val = float(np.mean(y_pred_np[resp_start:resp_end]))
        error = pred_val - g_bar

        # Plot 3: Response window detail
        ax3 = axes[i, 2]
        resp_times = times[resp_start:resp_end]
        ax3.plot(resp_times, y_time[resp_start:resp_end], 'k-', label='Target', linewidth=3)
        ax3.plot(resp_times, y_pred_np[resp_start:resp_end], 'b-', label='Readout', linewidth=2.5)
        ax3.axhline(g_bar, color='m', linestyle=':', alpha=0.7, linewidth=2, label=f'g_bar={g_bar:.2f}')
        ax3.set_ylabel('Output (g_bar)')
        # set ylim to -1.5, 1.5
        ax3.set_ylim(-1.2, 1.2)
        ax3.set_xlabel('Time (s)')

        ax3.set_title(f'Target: {g_bar:.2f} | Pred: {pred_val:.2f} | Error: {error:.2f}')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Create title with context info
    title = f'Network Performance - Epoch {epoch}'
    if has_test_contexts:
        # Check if using ranges or discrete values
        if task_cfg.train_context_ranges is not None and task_cfg.test_context_ranges is not None:
            # Format ranges as [min, max]
            train_ranges_str = ', '.join([f'[{r[0]:.1f}, {r[1]:.1f}]' for r in task_cfg.train_context_ranges])
            test_ranges_str = ', '.join([f'[{r[0]:.1f}, {r[1]:.1f}]' for r in task_cfg.test_context_ranges])
            title += f'\nTrain ranges: {train_ranges_str} | Held-out: {test_ranges_str}'
        else:
            # Discrete values (legacy)
            train_ctx_str = ', '.join([f'{c:.2f}' for c in task_cfg.train_contexts])
            test_ctx_str = ', '.join([f'{c:.2f}' for c in task_cfg.test_contexts])
            title += f'\nTrain contexts: [{train_ctx_str}] | Held-out: [{test_ctx_str}]'

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f'performance_epoch_{epoch:03d}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def train(
    rnn_cfg: RNNConfig,
    task_cfg: TemporalDecisionTaskConfig,
    training_cfg: TrainingConfig,
    output_dir: str,
    verbose: bool = True
):
    """Main training function with epoch-based training."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    figs_dir = os.path.join(output_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    # Set random seed
    key = jax.random.PRNGKey(training_cfg.seed)

    # Create model
    training_mode = training_cfg.training_mode
    key, model_key = jax.random.split(key)
    model, params = create_rnn_and_params(rnn_cfg, model_key, training_mode)

    if verbose:
        n_params = count_parameters(params, trainable_only=True, training_mode=training_mode)
        print(f"Model created with {n_params} trainable parameters")
        print(f"  N={rnn_cfg.N}, R={rnn_cfg.R}, g={rnn_cfg.g}")
        print(f"  Training mode: {training_mode}")

    # Create dataset
    key, data_key = jax.random.split(key)
    dataset = create_temporal_decision_dataset(task_cfg, data_key)

    if verbose:
        print(f"Dataset created with {dataset.n_steps} time steps per trial")
        print(f"  dt={task_cfg.dt}s, T={task_cfg.T_trial}s")
        print(f"  Stimulus window: [{task_cfg.t_stim_on}, {task_cfg.t_stim_off}]s")
        print(f"  Response window: [{task_cfg.t_response_on}, {task_cfg.t_response_off}]s")

    # Split params
    trainable_params = {}
    fixed_params = {'C': params.C}

    if training_mode == "full_rank":
        if params.J is not None:
            trainable_params['J'] = params.J
        fixed_params['M'] = params.M
        fixed_params['N_lr'] = params.N_lr
    else:
        if training_cfg.train_M:
            trainable_params['M'] = params.M
        else:
            fixed_params['M'] = params.M

        if training_cfg.train_N:
            trainable_params['N_lr'] = params.N_lr
        else:
            fixed_params['N_lr'] = params.N_lr

    if training_cfg.train_B:
        trainable_params['B'] = params.B
    else:
        fixed_params['B'] = params.B

    if training_cfg.train_w:
        trainable_params['w'] = params.w
        trainable_params['b'] = params.b
    else:
        fixed_params['w'] = params.w
        fixed_params['b'] = params.b

    # Compute iterations
    iters_per_epoch = training_cfg.n_train_trials // training_cfg.batch_size
    n_epochs = training_cfg.n_epochs
    n_iterations = n_epochs * iters_per_epoch

    # Create functions
    train_step, optimizer = make_train_step(
        model, dataset, rnn_cfg, task_cfg, training_cfg, n_iterations
    )
    eval_step = make_eval_step(model, dataset, rnn_cfg, task_cfg)

    # Initialize optimizer
    opt_state = optimizer.init(trainable_params)

    # Logs
    logs = {
        'train_loss': [],
        'train_mse': [],
        'train_correlation': [],
        'val_loss': [],
        'val_mse': [],
        'val_correlation': [],
        'val_low_c_mse': [],
        'val_high_c_mse': [],
        'test_mse': [],  # Held-out contexts
        'test_correlation': [],
        'epoch': [],
    }

    # Check if we have held-out contexts (either discrete values or ranges)
    has_test_contexts = (task_cfg.test_contexts is not None or
                          task_cfg.test_context_ranges is not None)

    if verbose:
        print(f"\nStarting training for {n_epochs} epochs")
        print(f"  {iters_per_epoch} iterations per epoch ({training_cfg.n_train_trials} trials)")
        print(f"  batch_size={training_cfg.batch_size}, lr={training_cfg.learning_rate}")
        print(f"  Plots every 10 epochs saved to: {figs_dir}")
        print()

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_corr = 0.0

        for _ in range(iters_per_epoch):
            key, batch_key = jax.random.split(key)
            batch = dataset.sample_batch(batch_key, training_cfg.batch_size)

            trainable_params, opt_state, metrics = train_step(
                trainable_params, fixed_params, opt_state, batch
            )

            epoch_loss += float(metrics['loss'])
            epoch_mse += float(metrics['mse'])
            epoch_corr += float(metrics['correlation'])

        avg_loss = epoch_loss / iters_per_epoch
        avg_mse = epoch_mse / iters_per_epoch
        avg_corr = epoch_corr / iters_per_epoch

        logs['train_loss'].append(avg_loss)
        logs['train_mse'].append(avg_mse)
        logs['train_correlation'].append(avg_corr)
        logs['epoch'].append(epoch + 1)

        # Evaluation on train contexts
        key, eval_key = jax.random.split(key)
        val_batch = dataset.sample_batch(eval_key, training_cfg.n_val_trials, use_test_contexts=False)
        val_metrics = eval_step(trainable_params, fixed_params, val_batch)

        logs['val_loss'].append(float(val_metrics['loss']))
        logs['val_mse'].append(float(val_metrics['mse']))
        logs['val_correlation'].append(float(val_metrics['correlation']))
        logs['val_low_c_mse'].append(float(val_metrics['low_c_mse']))
        logs['val_high_c_mse'].append(float(val_metrics['high_c_mse']))

        # Evaluation on held-out test contexts
        if has_test_contexts:
            key, test_key = jax.random.split(key)
            test_batch = dataset.sample_batch(test_key, training_cfg.n_val_trials, use_test_contexts=True)
            test_metrics = eval_step(trainable_params, fixed_params, test_batch)
            test_mse = float(test_metrics['mse'])
            test_corr = float(test_metrics['correlation'])
            logs['test_mse'].append(test_mse)
            logs['test_correlation'].append(test_corr)
        else:
            test_mse = None
            test_corr = None
            logs['test_mse'].append(float(val_metrics['mse']))
            logs['test_correlation'].append(float(val_metrics['correlation']))

        if verbose:
            base_msg = (f"Epoch {epoch+1:3d}/{n_epochs}: "
                       f"train_loss={avg_loss:.4f}, train_r={avg_corr:.3f} | "
                       f"val_mse={val_metrics['mse']:.4f}, val_r={val_metrics['correlation']:.3f}")
            if has_test_contexts and test_mse is not None:
                print(f"{base_msg} | test_mse={test_mse:.4f}, test_r={test_corr:.3f} (held-out)")
            else:
                print(f"{base_msg} (c<0.5: mse={val_metrics['low_c_mse']:.4f}, c>=0.5: mse={val_metrics['high_c_mse']:.4f})")

        # Plot every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            key, plot_key = jax.random.split(key)
            plot_path = plot_network_performance(
                model, trainable_params, fixed_params, dataset, task_cfg,
                epoch + 1, figs_dir, plot_key, n_trials=5
            )
            if verbose:
                print(f"  -> Saved performance plot: {plot_path}")

            # Create scatter plots for train and test predictions
            # Train scatter
            plot_scatter_correlation(
                np.array(val_metrics['predictions']),
                np.array(val_metrics['targets']),
                f'Train Correlation - Epoch {epoch + 1}',
                figs_dir,
                f'scatter_train_epoch_{epoch+1:03d}.png'
            )

            # Test scatter (if available)
            if has_test_contexts:
                plot_scatter_correlation(
                    np.array(test_metrics['predictions']),
                    np.array(test_metrics['targets']),
                    f'Test (Held-out) Correlation - Epoch {epoch + 1}',
                    figs_dir,
                    f'scatter_test_epoch_{epoch+1:03d}.png'
                )

    # Final params
    final_params = RNNParams(
        C=fixed_params['C'],
        M=trainable_params.get('M', fixed_params.get('M')),
        N_lr=trainable_params.get('N_lr', fixed_params.get('N_lr')),
        B=trainable_params.get('B', fixed_params.get('B')),
        w=trainable_params.get('w', fixed_params.get('w')),
        b=trainable_params.get('b', fixed_params.get('b', jnp.zeros(()))),
        J=trainable_params.get('J', fixed_params.get('J', None)),
    )

    return final_params, logs, model, dataset


def save_results(params: RNNParams, logs: dict, output_dir: str):
    """Save training results."""
    params_dict = {
        'C': params.C.tolist(),
        'M': params.M.tolist(),
        'N_lr': params.N_lr.tolist(),
        'B': params.B.tolist(),
        'w': params.w.tolist(),
        'b': float(params.b),
    }
    if params.J is not None:
        params_dict['J'] = params.J.tolist()

    with open(os.path.join(output_dir, 'params.pkl'), 'wb') as f:
        pickle.dump(params_dict, f)

    with open(os.path.join(output_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f, indent=2)


def plot_training_curves(logs: dict, output_dir: str):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = logs['epoch']

    # MSE
    axes[0].plot(epochs, logs['train_mse'], 'b-', label='Train')
    axes[0].plot(epochs, logs['val_mse'], 'r-', label='Val')
    if 'test_mse' in logs and len(logs['test_mse']) > 0:
        axes[0].plot(epochs, logs['test_mse'], 'g-', label='Test (held-out)', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Mean Squared Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Correlation
    axes[1].plot(epochs, logs['train_correlation'], 'b-', label='Train')
    axes[1].plot(epochs, logs['val_correlation'], 'r-', label='Val')
    if 'test_correlation' in logs and len(logs['test_correlation']) > 0:
        axes[1].plot(epochs, logs['test_correlation'], 'g-', label='Test (held-out)', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Prediction-Target Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[1].set_ylim(0, 1.05)

    # MSE by context range
    axes[2].plot(epochs, logs['val_low_c_mse'], 'g-', label='Val (c<0.5)')
    axes[2].plot(epochs, logs['val_high_c_mse'], 'm-', label='Val (c>=0.5)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MSE')
    axes[2].set_title('MSE by Context Range')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str,
    output_dir: str,
    filename: str
):
    """
    Plot scatter plot of predictions vs targets to visualize correlation.

    Args:
        predictions: Network predictions (n_trials,)
        targets: Ground truth targets (n_trials,)
        title: Plot title
        output_dir: Directory to save plot
        filename: Filename for plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Scatter plot
    ax.scatter(targets, predictions, alpha=0.5, s=20, edgecolors='none')

    # Diagonal line (perfect prediction)
    lim_min = min(targets.min(), predictions.min())
    lim_max = max(targets.max(), predictions.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect')

    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    correlation = np.corrcoef(predictions, targets)[0, 1]

    # Add text with metrics
    ax.text(0.05, 0.95, f'MSE: {mse:.4f}\nCorr: {correlation:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    ax.set_xlabel('Target g_bar', fontsize=12)
    ax.set_ylabel('Predicted g_bar', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train low-rank RNN on temporal decision task')
    parser.add_argument('--config', type=str, default='configs/temporal_decision_default.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        rnn_cfg, task_cfg, training_cfg, name = load_config(args.config)
        if not args.quiet:
            print(f"Loaded config from: {args.config}")
    else:
        print(f"Config not found: {args.config}, using defaults")
        rnn_cfg = RNNConfig(N=100, R=2, g=0.8, tau=0.1, d_in=3)
        task_cfg = TemporalDecisionTaskConfig()
        training_cfg = TrainingConfig(n_epochs=50, batch_size=64, n_train_trials=10000, n_val_trials=1000)
        name = "temporal_decision"

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'{name}_{timestamp}')

    # Train
    params, logs, model, dataset = train(
        rnn_cfg, task_cfg, training_cfg, output_dir, verbose=not args.quiet
    )

    # Save results
    save_results(params, logs, output_dir)

    # Plot training curves
    plot_training_curves(logs, output_dir)

    # Generate example plots
    key = jax.random.PRNGKey(0)
    figs_dir = os.path.join(output_dir, 'figs')

    trial = dataset.sample_trial(key)
    plot_single_trial(trial, task_cfg, save_path=os.path.join(figs_dir, 'example_trial.png'))

    key, subkey = jax.random.split(key)
    plot_interpolation_comparison(
        dataset, subkey, a1=0.8, a2=-0.5,
        save_path=os.path.join(figs_dir, 'interpolation_comparison.png')
    )

    if not args.quiet:
        print(f"\nResults saved to {output_dir}")
        print(f"Final validation MSE: {logs['val_mse'][-1]:.4f}")
        print(f"Final validation correlation: {logs['val_correlation'][-1]:.3f}")
        print(f"Training curves: {output_dir}/training_curves.png")
        print(f"Performance plots: {output_dir}/figs/")


if __name__ == '__main__':
    main()
