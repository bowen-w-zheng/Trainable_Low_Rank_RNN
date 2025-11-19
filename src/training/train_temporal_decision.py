"""Training script for the temporal decision task (Interpolating Go-No-Go)."""

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from src.config import RNNConfig, IntegratorConfig, TrainingConfig
from src.models.lowrank_rnn import LowRankRNN, RNNParams, create_rnn_and_params, count_parameters
from src.data.temporal_decision_dataset import (
    TemporalDecisionDataset,
    TemporalDecisionTaskConfig,
    create_temporal_decision_dataset,
    plot_single_trial,
    plot_interpolation_comparison
)


def create_optimizer(training_cfg: TrainingConfig, n_iterations: int) -> optax.GradientTransformation:
    """Create optimizer with learning rate schedule."""
    # Cosine decay schedule
    schedule = optax.cosine_decay_schedule(
        init_value=training_cfg.learning_rate,
        decay_steps=n_iterations,
        alpha=0.01  # Final LR = 1% of initial
    )

    if training_cfg.optimizer == "adam":
        opt = optax.adam(schedule)
    elif training_cfg.optimizer == "adamw":
        opt = optax.adamw(schedule, weight_decay=training_cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {training_cfg.optimizer}")

    # Add gradient clipping
    if training_cfg.grad_clip > 0:
        opt = optax.chain(
            optax.clip_by_global_norm(training_cfg.grad_clip),
            opt
        )

    return opt


def make_train_step(model, dataset, rnn_cfg, task_cfg, training_cfg, n_iterations: int):
    """
    Create the training step function.

    Returns a JIT-compiled function that performs one optimization step.
    """
    resp_start, resp_end = dataset.get_avg_window_indices()
    dt = dataset.dt
    loss_type = task_cfg.loss_type
    label_type = task_cfg.label_type

    # Create optimizer with schedule
    optimizer = create_optimizer(training_cfg, n_iterations)

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
            J=trainable_params.get('J', fixed_params.get('J', None)),
        )

        def single_trial(u_seq, label):
            _, ys = model.simulate_trial_fast(params, u_seq, dt)
            # Average output in response window
            y_hat = jnp.mean(ys[resp_start:resp_end])

            # Compute loss based on type
            if loss_type == "bce":
                # Binary cross-entropy
                prob = jax.nn.sigmoid(y_hat)
                loss = -label * jnp.log(prob + 1e-7) - (1 - label) * jnp.log(1 - prob + 1e-7)
            else:  # mse
                if label_type == "pm1":
                    target = label
                    pred = jnp.tanh(y_hat)
                else:
                    target = label
                    pred = jax.nn.sigmoid(y_hat)
                loss = (pred - target) ** 2

            return y_hat, loss

        y_hats, losses = jax.vmap(single_trial)(batch['u_seq'], batch['labels'])
        loss = jnp.mean(losses)

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
        if label_type == "pm1":
            preds = jnp.sign(y_hats)
            acc = jnp.mean(preds == batch['labels'])
        else:
            preds = (jax.nn.sigmoid(y_hats) > 0.5).astype(jnp.float32)
            acc = jnp.mean(preds == batch['labels'])

        metrics = {
            'loss': loss,
            'accuracy': acc,
        }

        return trainable_params, opt_state, metrics

    return jax.jit(train_step), optimizer


def make_eval_step(model, dataset, rnn_cfg, task_cfg):
    """Create evaluation step function."""
    resp_start, resp_end = dataset.get_avg_window_indices()
    dt = dataset.dt
    loss_type = task_cfg.loss_type
    label_type = task_cfg.label_type

    def eval_step(trainable_params, fixed_params, batch):
        """Evaluate on a batch."""
        params = RNNParams(
            C=fixed_params['C'],
            M=trainable_params.get('M', fixed_params.get('M', None)),
            N_lr=trainable_params.get('N_lr', fixed_params.get('N_lr', None)),
            B=trainable_params.get('B', fixed_params.get('B', None)),
            w=trainable_params.get('w', fixed_params.get('w', None)),
            b=trainable_params.get('b', fixed_params.get('b', jnp.zeros(()))),
            J=trainable_params.get('J', fixed_params.get('J', None)),
        )

        def single_trial(u_seq, label, context):
            _, ys = model.simulate_trial_fast(params, u_seq, dt)
            y_hat = jnp.mean(ys[resp_start:resp_end])

            # Compute loss
            if loss_type == "bce":
                prob = jax.nn.sigmoid(y_hat)
                loss = -label * jnp.log(prob + 1e-7) - (1 - label) * jnp.log(1 - prob + 1e-7)
            else:
                if label_type == "pm1":
                    target = label
                    pred = jnp.tanh(y_hat)
                else:
                    target = label
                    pred = jax.nn.sigmoid(y_hat)
                loss = (pred - target) ** 2

            return y_hat, loss

        y_hats, losses = jax.vmap(single_trial)(batch['u_seq'], batch['labels'], batch['contexts'])
        loss = jnp.mean(losses)

        # Compute accuracy
        if label_type == "pm1":
            preds = jnp.sign(y_hats)
            acc = jnp.mean(preds == batch['labels'])
        else:
            preds = (jax.nn.sigmoid(y_hats) > 0.5).astype(jnp.float32)
            acc = jnp.mean(preds == batch['labels'])

        # Context-specific accuracy (low c vs high c)
        low_c_mask = batch['contexts'] < 0.5
        high_c_mask = batch['contexts'] >= 0.5

        low_c_acc = jnp.where(
            jnp.sum(low_c_mask) > 0,
            jnp.sum((preds == batch['labels']) * low_c_mask) / jnp.sum(low_c_mask),
            0.0
        )
        high_c_acc = jnp.where(
            jnp.sum(high_c_mask) > 0,
            jnp.sum((preds == batch['labels']) * high_c_mask) / jnp.sum(high_c_mask),
            0.0
        )

        return {
            'loss': loss,
            'accuracy': acc,
            'low_c_accuracy': low_c_acc,
            'high_c_accuracy': high_c_acc,
        }

    return jax.jit(eval_step)


def train(
    rnn_cfg: RNNConfig,
    task_cfg: TemporalDecisionTaskConfig,
    training_cfg: TrainingConfig,
    verbose: bool = True
):
    """
    Main training function with epoch-based training.

    Args:
        rnn_cfg: RNN configuration
        task_cfg: Task configuration
        training_cfg: Training configuration
        verbose: Whether to print progress

    Returns:
        Trained parameters and training logs
    """
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

    # Split params into trainable and fixed
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

    # Compute iterations per epoch and total iterations
    iters_per_epoch = training_cfg.n_train_trials // training_cfg.batch_size
    n_epochs = training_cfg.n_epochs
    n_iterations = n_epochs * iters_per_epoch

    # Create training and evaluation functions
    train_step, optimizer = make_train_step(
        model, dataset, rnn_cfg, task_cfg, training_cfg, n_iterations
    )
    eval_step = make_eval_step(model, dataset, rnn_cfg, task_cfg)

    # Initialize optimizer state
    opt_state = optimizer.init(trainable_params)

    # Training logs
    logs = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_low_c_accuracy': [],
        'val_high_c_accuracy': [],
        'epoch': [],
    }

    if verbose:
        print(f"\nStarting training for {n_epochs} epochs")
        print(f"  {iters_per_epoch} iterations per epoch ({training_cfg.n_train_trials} trials)")
        print(f"  batch_size={training_cfg.batch_size}, lr={training_cfg.learning_rate}")
        print(f"  Using cosine LR schedule: {training_cfg.learning_rate} -> {training_cfg.learning_rate * 0.01}")
        print()

    # Training loop - epoch based
    for epoch in range(n_epochs):
        # Track epoch metrics
        epoch_loss = 0.0
        epoch_acc = 0.0

        # Iterate over all training data
        for _ in range(iters_per_epoch):
            # Sample batch
            key, batch_key = jax.random.split(key)
            batch = dataset.sample_batch(batch_key, training_cfg.batch_size)

            # Training step
            trainable_params, opt_state, metrics = train_step(
                trainable_params, fixed_params, opt_state, batch
            )

            epoch_loss += float(metrics['loss'])
            epoch_acc += float(metrics['accuracy'])

        # Compute epoch averages
        avg_loss = epoch_loss / iters_per_epoch
        avg_acc = epoch_acc / iters_per_epoch

        logs['train_loss'].append(avg_loss)
        logs['train_accuracy'].append(avg_acc)
        logs['epoch'].append(epoch + 1)

        # Evaluation at end of epoch
        key, eval_key = jax.random.split(key)
        val_batch = dataset.sample_batch(eval_key, training_cfg.n_val_trials)
        val_metrics = eval_step(trainable_params, fixed_params, val_batch)

        logs['val_loss'].append(float(val_metrics['loss']))
        logs['val_accuracy'].append(float(val_metrics['accuracy']))
        logs['val_low_c_accuracy'].append(float(val_metrics['low_c_accuracy']))
        logs['val_high_c_accuracy'].append(float(val_metrics['high_c_accuracy']))

        if verbose:
            print(f"Epoch {epoch+1:3d}/{n_epochs}: "
                  f"train_loss={avg_loss:.4f}, train_acc={avg_acc:.1%} | "
                  f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.1%} "
                  f"(c<0.5: {val_metrics['low_c_accuracy']:.1%}, c>=0.5: {val_metrics['high_c_accuracy']:.1%})")

    # Reconstruct final params
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
    if params.J is not None:
        params_dict['J'] = params.J.tolist()

    with open(os.path.join(output_dir, 'params.pkl'), 'wb') as f:
        pickle.dump(params_dict, f)

    # Save logs
    with open(os.path.join(output_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train low-rank RNN on temporal decision task')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    # Model parameters
    parser.add_argument('--N', type=int, default=100, help='Number of recurrent units')
    parser.add_argument('--R', type=int, default=2, help='Rank of low-rank connectivity')
    parser.add_argument('--g', type=float, default=0.8, help='Gain for random bulk')
    parser.add_argument('--tau', type=float, default=0.1, help='Time constant')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n-train', type=int, default=10000, help='Training trials per epoch')
    parser.add_argument('--n-val', type=int, default=1000, help='Validation trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create configurations
    rnn_cfg = RNNConfig(
        N=args.N,
        R=args.R,
        g=args.g,
        tau=args.tau,
        d_in=3,  # u1, u2, c
    )

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

    training_cfg = TrainingConfig(
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        n_train_trials=args.n_train,
        n_val_trials=args.n_val,
        seed=args.seed,
        optimizer="adam",
        grad_clip=1.0,
        training_mode="low_rank",
        train_M=True,
        train_N=True,
        train_B=True,
        train_w=True,
    )

    # Train
    params, logs, model, dataset = train(rnn_cfg, task_cfg, training_cfg, verbose=not args.quiet)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, f'temporal_decision_{timestamp}')
    save_results(params, logs, output_dir)

    if not args.quiet:
        print(f"\nResults saved to {output_dir}")
        print(f"Final validation accuracy: {logs['val_accuracy'][-1]:.1%}")

        # Generate example plots
        key = jax.random.PRNGKey(0)
        trial = dataset.sample_trial(key)
        plot_single_trial(trial, task_cfg, save_path=os.path.join(output_dir, 'example_trial.png'))

        key, subkey = jax.random.split(key)
        plot_interpolation_comparison(
            dataset, subkey, a1=0.8, a2=-0.5,
            save_path=os.path.join(output_dir, 'interpolation_comparison.png')
        )
        print(f"Example plots saved to {output_dir}")


if __name__ == '__main__':
    main()
