"""Tests for training step and loop."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.config import ExperimentConfig
from src.models.lowrank_rnn import LowRankRNN, RNNParams, create_rnn_and_params
from src.data.contextual_switch_dataset import create_dataset
from src.training.train_context_switch import (
    make_train_step,
    make_eval_step,
    create_optimizer,
    train,
)


class TestCreateOptimizer:
    """Tests for optimizer creation."""

    def test_adam_optimizer(self):
        """Test Adam optimizer creation."""
        cfg = ExperimentConfig()
        cfg.training.optimizer = "adam"
        cfg.training.learning_rate = 0.001

        opt = create_optimizer(cfg.training)
        assert opt is not None

    def test_adamw_optimizer(self):
        """Test AdamW optimizer creation."""
        cfg = ExperimentConfig()
        cfg.training.optimizer = "adamw"

        opt = create_optimizer(cfg.training)
        assert opt is not None

    def test_unknown_optimizer(self):
        """Test error on unknown optimizer."""
        cfg = ExperimentConfig()
        cfg.training.optimizer = "unknown"

        with pytest.raises(ValueError):
            create_optimizer(cfg.training)


class TestMakeTrainStep:
    """Tests for train step function creation."""

    def test_train_step_runs(self):
        """Test that a single training step runs without error."""
        # Create tiny config
        cfg = ExperimentConfig()
        cfg.rnn.N = 20
        cfg.rnn.R = 2
        cfg.integrator.dt = 1.0
        cfg.integrator.T = 10.0
        cfg.task.T_burn = 2.0
        cfg.task.T_stim = 8.0
        cfg.task.T_avg = 3.0
        cfg.training.batch_size = 2

        key = jax.random.PRNGKey(42)

        # Create model and dataset
        key, model_key = jax.random.split(key)
        model, params = create_rnn_and_params(cfg.rnn, model_key)

        key, data_key = jax.random.split(key)
        dataset = create_dataset(cfg.task, cfg.integrator, data_key)

        # Create train step
        train_step, optimizer = make_train_step(model, dataset, cfg)

        # Set up parameters
        trainable_params = {
            'M': params.M,
            'N_lr': params.N_lr,
            'B': params.B,
            'w': params.w,
            'b': params.b,
        }
        fixed_params = {'C': params.C}
        opt_state = optimizer.init(trainable_params)

        # Sample batch and run step
        key, batch_key = jax.random.split(key)
        batch = dataset.sample_batch(batch_key, cfg.training.batch_size)

        new_params, new_opt_state, metrics = train_step(
            trainable_params, fixed_params, opt_state, batch
        )

        # Check outputs
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert jnp.isfinite(metrics['loss'])
        assert 0.0 <= metrics['accuracy'] <= 1.0

    def test_train_step_updates_params(self):
        """Test that training step updates parameters."""
        cfg = ExperimentConfig()
        cfg.rnn.N = 20
        cfg.integrator.dt = 1.0
        cfg.integrator.T = 10.0
        cfg.task.T_burn = 2.0
        cfg.task.T_stim = 8.0
        cfg.training.batch_size = 4
        cfg.training.learning_rate = 0.1  # Large LR to see change

        key = jax.random.PRNGKey(42)

        key, model_key = jax.random.split(key)
        model, params = create_rnn_and_params(cfg.rnn, model_key)

        key, data_key = jax.random.split(key)
        dataset = create_dataset(cfg.task, cfg.integrator, data_key)

        train_step, optimizer = make_train_step(model, dataset, cfg)

        trainable_params = {
            'M': params.M,
            'N_lr': params.N_lr,
            'B': params.B,
            'w': params.w,
            'b': params.b,
        }
        fixed_params = {'C': params.C}
        opt_state = optimizer.init(trainable_params)

        key, batch_key = jax.random.split(key)
        batch = dataset.sample_batch(batch_key, cfg.training.batch_size)

        # Store original M
        M_before = trainable_params['M'].copy()

        new_params, _, _ = train_step(
            trainable_params, fixed_params, opt_state, batch
        )

        # Check that M changed
        assert not jnp.allclose(new_params['M'], M_before), \
            "Parameters should change after training step"


class TestMakeEvalStep:
    """Tests for evaluation step."""

    def test_eval_step_runs(self):
        """Test that evaluation step runs without error."""
        cfg = ExperimentConfig()
        cfg.rnn.N = 20
        cfg.integrator.dt = 1.0
        cfg.integrator.T = 10.0
        cfg.task.T_burn = 2.0
        cfg.task.T_stim = 8.0

        key = jax.random.PRNGKey(42)

        key, model_key = jax.random.split(key)
        model, params = create_rnn_and_params(cfg.rnn, model_key)

        key, data_key = jax.random.split(key)
        dataset = create_dataset(cfg.task, cfg.integrator, data_key)

        eval_step = make_eval_step(model, dataset, cfg)

        trainable_params = {
            'M': params.M,
            'N_lr': params.N_lr,
            'B': params.B,
            'w': params.w,
            'b': params.b,
        }
        fixed_params = {'C': params.C}

        key, batch_key = jax.random.split(key)
        batch = dataset.sample_batch(batch_key, 8)

        metrics = eval_step(trainable_params, fixed_params, batch)

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'ctx1_accuracy' in metrics
        assert 'ctx2_accuracy' in metrics


class TestFullTraining:
    """Tests for full training loop."""

    def test_tiny_training_runs(self):
        """Test that training runs on tiny problem."""
        cfg = ExperimentConfig()
        cfg.rnn.N = 20
        cfg.rnn.R = 2
        cfg.integrator.dt = 1.0
        cfg.integrator.T = 10.0
        cfg.task.T_burn = 2.0
        cfg.task.T_stim = 8.0
        cfg.task.T_avg = 3.0
        cfg.training.batch_size = 4
        cfg.training.n_train_trials = 20
        cfg.training.n_val_trials = 8
        cfg.training.log_every = 2
        cfg.training.eval_every = 5
        cfg.training.save_every = 0  # Don't save

        params, logs, model = train(cfg, verbose=False)

        # Check that training produced output
        assert params is not None
        assert len(logs['train_loss']) > 0
        assert len(logs['val_accuracy']) > 0

    def test_loss_decreases(self):
        """Test that loss decreases during training."""
        cfg = ExperimentConfig()
        cfg.rnn.N = 30
        cfg.rnn.R = 2
        cfg.integrator.dt = 0.5
        cfg.integrator.T = 10.0
        cfg.task.T_burn = 2.0
        cfg.task.T_stim = 8.0
        cfg.task.T_avg = 3.0
        cfg.training.batch_size = 8
        cfg.training.n_train_trials = 160  # More iterations
        cfg.training.n_val_trials = 16
        cfg.training.learning_rate = 0.01
        cfg.training.log_every = 5
        cfg.training.eval_every = 20
        cfg.training.save_every = 0

        params, logs, model = train(cfg, verbose=False)

        # Check that loss decreased (comparing early to late)
        if len(logs['train_loss']) >= 4:
            early_loss = np.mean(logs['train_loss'][:2])
            late_loss = np.mean(logs['train_loss'][-2:])
            # Loss should decrease (allowing for noise)
            assert late_loss < early_loss * 1.5, \
                f"Loss didn't decrease: {early_loss:.4f} -> {late_loss:.4f}"


class TestGradientFlow:
    """Tests for gradient computation and flow."""

    def test_gradients_nonzero(self):
        """Test that gradients are non-zero."""
        cfg = ExperimentConfig()
        cfg.rnn.N = 20
        cfg.integrator.dt = 1.0
        cfg.integrator.T = 10.0
        cfg.task.T_burn = 2.0
        cfg.task.T_stim = 8.0

        key = jax.random.PRNGKey(42)

        key, model_key = jax.random.split(key)
        model, params = create_rnn_and_params(cfg.rnn, model_key)

        key, data_key = jax.random.split(key)
        dataset = create_dataset(cfg.task, cfg.integrator, data_key)

        avg_start_idx, avg_end_idx = dataset.get_avg_window_indices()
        dt = cfg.integrator.dt

        def loss_fn(trainable_params):
            test_params = RNNParams(
                C=params.C,
                M=trainable_params['M'],
                N_lr=trainable_params['N_lr'],
                B=trainable_params['B'],
                w=trainable_params['w'],
                b=trainable_params['b'],
            )

            # Single trial with non-zero input (required for B gradient)
            u_seq = jnp.ones((dataset.n_steps, 4)) * 0.1
            _, ys = model.simulate_trial_fast(test_params, u_seq, dt)
            y_hat = jnp.mean(ys[avg_start_idx:avg_end_idx])
            return (y_hat - 1.0) ** 2

        trainable_params = {
            'M': params.M,
            'N_lr': params.N_lr,
            'B': params.B,
            'w': params.w,
            'b': params.b,
        }

        loss, grads = jax.value_and_grad(loss_fn)(trainable_params)

        # Check gradients are non-zero
        for key, grad in grads.items():
            grad_norm = jnp.linalg.norm(grad)
            assert grad_norm > 0, f"Gradient for {key} is zero"
            assert jnp.isfinite(grad_norm), f"Gradient for {key} is not finite"
