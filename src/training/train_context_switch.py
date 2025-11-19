"""Training script for the contextual switch task."""

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from src.config import ExperimentConfig
from src.models.lowrank_rnn import LowRankRNN, RNNParams, create_rnn_and_params, count_parameters
from src.data.contextual_switch_dataset import ContextualSwitchDataset, create_dataset
from src.training.losses import compute_trial_output, compute_trial_loss, l2_regularization
from src.training.metrics import compute_accuracy, compute_context_accuracy


def create_optimizer(cfg, n_iterations: int) -> optax.GradientTransformation:
    """Create optimizer with learning rate schedule."""
    # Cosine decay schedule
    schedule = optax.cosine_decay_schedule(
        init_value=cfg.learning_rate,
        decay_steps=n_iterations,
        alpha=0.01  # Final LR = 1% of initial
    )

    if cfg.optimizer == "adam":
        opt = optax.adam(schedule)
    elif cfg.optimizer == "adamw":
        opt = optax.adamw(schedule, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # Add gradient clipping
    if cfg.grad_clip > 0:
        opt = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            opt
        )

    return opt


def make_train_step(model, dataset, cfg, n_iterations: int):
    """
    Create the training step function.

    Returns a JIT-compiled function that performs one optimization step.
    """
    avg_start_idx, avg_end_idx = dataset.get_avg_window_indices()
    dt = cfg.integrator.dt
    loss_type = cfg.task.loss_type
    label_type = cfg.task.label_type

    # Create optimizer with schedule
    optimizer = create_optimizer(cfg.training, n_iterations)

    def loss_fn(trainable_params, fixed_params, batch):
        """Compute loss for a batch."""
        # Reconstruct full params
        params = RNNParams(
            C=fixed_params['C'],
            M=trainable_params.get('M', fixed_params.get('M', None)),
            N_lr=trainable_params.get('N_lr', fixed_params.get('N_lr', None)),
            B=trainable_params.get('B', fixed_params.get('B', None)),
            w=trainable_params.get('w', fixed_params.get('w', None)),
            b=trainable_params.get('b', fixed_params.get('b', jnp.zeros(()))),
        )

        def single_trial(u_seq, label):
            _, ys = model.simulate_trial_fast(params, u_seq, dt)
            y_hat = compute_trial_output(ys, avg_start_idx, avg_end_idx)
            return y_hat, compute_trial_loss(y_hat, label, loss_type)

        y_hats, losses = jax.vmap(single_trial)(batch['u_seq'], batch['labels'])
        loss = jnp.mean(losses)

        # Add regularization
        reg = l2_regularization(trainable_params, cfg.training.weight_decay)
        loss = loss + reg

        return loss, y_hats

    def train_step(trainable_params, fixed_params, opt_state, batch):
        """Single training step."""
        (loss, y_hats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            trainable_params, fixed_params, batch
        )

        # Compute updates
        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)

        # Compute accuracy
        acc = compute_accuracy(y_hats, batch['labels'], label_type)

        metrics = {
            'loss': loss,
            'accuracy': acc,
        }

        return trainable_params, opt_state, metrics

    return jax.jit(train_step), optimizer


def make_eval_step(model, dataset, cfg):
    """Create evaluation step function."""
    avg_start_idx, avg_end_idx = dataset.get_avg_window_indices()
    dt = cfg.integrator.dt
    loss_type = cfg.task.loss_type
    label_type = cfg.task.label_type

    def eval_step(trainable_params, fixed_params, batch):
        """Evaluate on a batch."""
        params = RNNParams(
            C=fixed_params['C'],
            M=trainable_params.get('M', fixed_params.get('M', None)),
            N_lr=trainable_params.get('N_lr', fixed_params.get('N_lr', None)),
            B=trainable_params.get('B', fixed_params.get('B', None)),
            w=trainable_params.get('w', fixed_params.get('w', None)),
            b=trainable_params.get('b', fixed_params.get('b', jnp.zeros(()))),
        )

        def single_trial(u_seq, label):
            _, ys = model.simulate_trial_fast(params, u_seq, dt)
            y_hat = compute_trial_output(ys, avg_start_idx, avg_end_idx)
            return y_hat, compute_trial_loss(y_hat, label, loss_type)

        y_hats, losses = jax.vmap(single_trial)(batch['u_seq'], batch['labels'])
        loss = jnp.mean(losses)

        # Context-specific accuracy
        ctx_metrics = compute_context_accuracy(y_hats, batch['labels'], batch['contexts'], label_type)

        return {
            'loss': loss,
            'accuracy': ctx_metrics['total_acc'],
            'ctx1_accuracy': ctx_metrics['ctx1_acc'],
            'ctx2_accuracy': ctx_metrics['ctx2_acc'],
        }

    return jax.jit(eval_step)


def train(cfg: ExperimentConfig, verbose: bool = True):
    """
    Main training function.

    Args:
        cfg: Experiment configuration
        verbose: Whether to print progress

    Returns:
        Trained parameters and training logs
    """
    # Set random seed
    key = jax.random.PRNGKey(cfg.training.seed)

    # Create model
    key, model_key = jax.random.split(key)
    model, params = create_rnn_and_params(cfg.rnn, model_key)

    if verbose:
        n_params = count_parameters(params, trainable_only=True)
        print(f"Model created with {n_params} trainable parameters")
        print(f"  N={cfg.rnn.N}, R={cfg.rnn.R}, g={cfg.rnn.g}")

    # Create dataset
    key, data_key = jax.random.split(key)
    dataset = create_dataset(cfg.task, cfg.integrator, data_key)

    if verbose:
        print(f"Dataset created with {dataset.n_steps} time steps per trial")

    # Split params into trainable and fixed
    trainable_params = {}
    fixed_params = {'C': params.C}

    if cfg.training.train_M:
        trainable_params['M'] = params.M
    else:
        fixed_params['M'] = params.M

    if cfg.training.train_N:
        trainable_params['N_lr'] = params.N_lr
    else:
        fixed_params['N_lr'] = params.N_lr

    if cfg.training.train_B:
        trainable_params['B'] = params.B
    else:
        fixed_params['B'] = params.B

    if cfg.training.train_w:
        trainable_params['w'] = params.w
        trainable_params['b'] = params.b
    else:
        fixed_params['w'] = params.w
        fixed_params['b'] = params.b

    # Compute number of iterations
    n_iterations = cfg.training.n_train_trials // cfg.training.batch_size

    # Create training and evaluation functions
    train_step, optimizer = make_train_step(model, dataset, cfg, n_iterations)
    eval_step = make_eval_step(model, dataset, cfg)

    # Initialize optimizer state
    opt_state = optimizer.init(trainable_params)

    # Training logs
    logs = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_ctx1_accuracy': [],
        'val_ctx2_accuracy': [],
        'iteration': [],
    }

    if verbose:
        print(f"Starting training for {n_iterations} iterations")
        print(f"  batch_size={cfg.training.batch_size}, lr={cfg.training.learning_rate}")
        print(f"  Using cosine LR schedule: {cfg.training.learning_rate} -> {cfg.training.learning_rate * 0.01}")

    # Running averages for stable logging
    running_loss = 0.0
    running_acc = 0.0
    running_count = 0

    # Training loop
    for i in range(n_iterations):
        # Sample batch
        key, batch_key = jax.random.split(key)
        batch = dataset.sample_batch(batch_key, cfg.training.batch_size)

        # Training step
        trainable_params, opt_state, metrics = train_step(
            trainable_params, fixed_params, opt_state, batch
        )

        # Update running averages
        running_loss += float(metrics['loss'])
        running_acc += float(metrics['accuracy'])
        running_count += 1

        # Logging
        if (i + 1) % cfg.training.log_every == 0:
            avg_loss = running_loss / running_count
            avg_acc = running_acc / running_count

            logs['train_loss'].append(avg_loss)
            logs['train_accuracy'].append(avg_acc)
            logs['iteration'].append(i + 1)

            if verbose:
                print(f"Iter {i+1}/{n_iterations}: "
                      f"loss={avg_loss:.4f}, acc={avg_acc:.3f} "
                      f"(last={metrics['loss']:.4f}, {metrics['accuracy']:.3f})")

            # Reset running averages
            running_loss = 0.0
            running_acc = 0.0
            running_count = 0

        # Evaluation
        if (i + 1) % cfg.training.eval_every == 0:
            key, eval_key = jax.random.split(key)
            val_batch = dataset.sample_batch(eval_key, cfg.training.n_val_trials)
            val_metrics = eval_step(trainable_params, fixed_params, val_batch)

            logs['val_loss'].append(float(val_metrics['loss']))
            logs['val_accuracy'].append(float(val_metrics['accuracy']))
            logs['val_ctx1_accuracy'].append(float(val_metrics['ctx1_accuracy']))
            logs['val_ctx2_accuracy'].append(float(val_metrics['ctx2_accuracy']))

            if verbose:
                print(f"  Val: loss={val_metrics['loss']:.4f}, "
                      f"acc={val_metrics['accuracy']:.3f} "
                      f"(ctx1={val_metrics['ctx1_accuracy']:.3f}, "
                      f"ctx2={val_metrics['ctx2_accuracy']:.3f})")

        # Checkpointing
        if cfg.training.save_every > 0 and (i + 1) % cfg.training.save_every == 0:
            save_checkpoint(trainable_params, fixed_params, cfg, i + 1)

    # Reconstruct final params
    final_params = RNNParams(
        C=fixed_params['C'],
        M=trainable_params.get('M', fixed_params.get('M')),
        N_lr=trainable_params.get('N_lr', fixed_params.get('N_lr')),
        B=trainable_params.get('B', fixed_params.get('B')),
        w=trainable_params.get('w', fixed_params.get('w')),
        b=trainable_params.get('b', fixed_params.get('b', jnp.zeros(()))),
    )

    return final_params, logs, model


def save_checkpoint(trainable_params, fixed_params, cfg, iteration):
    """Save training checkpoint."""
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    checkpoint = {
        'trainable_params': {k: v.tolist() for k, v in trainable_params.items()},
        'fixed_params': {k: v.tolist() for k, v in fixed_params.items()},
        'iteration': iteration,
    }

    path = os.path.join(cfg.training.checkpoint_dir, f'checkpoint_{iteration}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def save_results(params: RNNParams, logs: dict, cfg: ExperimentConfig, output_dir: str):
    """Save training results."""
    os.makedirs(output_dir, exist_ok=True)

    # Save parameters
    params_dict = {
        'C': params.C.tolist(),
        'M': params.M.tolist(),
        'N_lr': params.N_lr.tolist(),
        'B': params.B.tolist(),
        'w': params.w.tolist(),
        'b': float(params.b),
    }
    with open(os.path.join(output_dir, 'params.pkl'), 'wb') as f:
        pickle.dump(params_dict, f)

    # Save logs
    with open(os.path.join(output_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f, indent=2)

    # Save config
    cfg.to_yaml(os.path.join(output_dir, 'config.yaml'))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train low-rank RNN on contextual switch task')
    parser.add_argument('--config', type=str, default='configs/contextual_switch_default.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        print(f"Config not found: {args.config}, using defaults")
        cfg = ExperimentConfig()

    # Train
    params, logs, model = train(cfg, verbose=not args.quiet)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'{cfg.name}_{timestamp}')
    save_results(params, logs, cfg, output_dir)

    if not args.quiet:
        print(f"\nResults saved to {output_dir}")
        print(f"Final validation accuracy: {logs['val_accuracy'][-1]:.3f}")


if __name__ == '__main__':
    main()
