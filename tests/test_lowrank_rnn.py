"""Tests for low-rank RNN model."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.config import RNNConfig, IntegratorConfig
from src.models.lowrank_rnn import (
    LowRankRNN,
    RNNParams,
    create_rnn_and_params,
    count_parameters,
)


class TestLowRankRNNInit:
    """Tests for model initialization."""

    def test_param_shapes(self):
        """Test that parameter shapes are correct."""
        N, R, d_in = 100, 2, 4
        cfg = RNNConfig(N=N, R=R, d_in=d_in)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        assert params.C.shape == (N, N)
        assert params.M.shape == (N, R)
        assert params.N_lr.shape == (N, R)
        assert params.B.shape == (N, d_in)
        assert params.w.shape == (N,)
        assert params.b.shape == ()

    def test_trainable_param_count(self):
        """Test that trainable parameter count is O(NR)."""
        N, R, d_in = 1000, 2, 4
        cfg = RNNConfig(N=N, R=R, d_in=d_in)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        n_trainable = count_parameters(params, trainable_only=True)

        # Expected: M(N*R) + N_lr(N*R) + B(N*d_in) + w(N) + b(1)
        expected = N * R + N * R + N * d_in + N + 1
        assert n_trainable == expected

        # Should be O(NR), not O(N^2)
        n_total = count_parameters(params, trainable_only=False)
        assert n_total > n_trainable  # C is not trainable
        assert n_total == n_trainable + N * N

    def test_c_scaling(self):
        """Test that C has correct variance (1/sqrt(N))."""
        N = 1000
        cfg = RNNConfig(N=N)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        # Variance should be approximately 1/N
        variance = jnp.var(params.C)
        expected_variance = 1.0 / N
        assert jnp.abs(variance - expected_variance) < 0.1 * expected_variance


class TestLowRankRNNRHS:
    """Tests for RNN dynamics computation."""

    def test_rhs_runs(self):
        """Test that RHS computes without errors."""
        N, R = 50, 2
        cfg = RNNConfig(N=N, R=R)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        # Random state and input
        x = jax.random.normal(jax.random.PRNGKey(0), (N,))
        u_t = jnp.ones(4) * 0.1

        def u_of_t(t):
            return u_t

        dxdt = model.rhs(0.0, x, (params, u_of_t))

        assert dxdt.shape == (N,)
        assert not jnp.any(jnp.isnan(dxdt))

    def test_rhs_shapes(self):
        """Test RHS output shape."""
        N = 100
        cfg = RNNConfig(N=N)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        x = jnp.zeros(N)
        u_of_t = lambda t: jnp.zeros(4)

        dxdt = model.rhs(0.0, x, (params, u_of_t))
        assert dxdt.shape == (N,)

    def test_compute_J(self):
        """Test J matrix computation."""
        N, R = 50, 2
        g = 0.8
        cfg = RNNConfig(N=N, R=R, g=g)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        J = LowRankRNN.compute_J(params, g, N)

        assert J.shape == (N, N)

        # J should be g*C + (1/N)*M@N^T
        expected_J = g * params.C + (1.0 / N) * params.M @ params.N_lr.T
        assert jnp.allclose(J, expected_J)


class TestLowRankRNNReadout:
    """Tests for readout computation."""

    def test_readout_single(self):
        """Test readout for single state."""
        N = 50
        cfg = RNNConfig(N=N)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        x = jax.random.normal(jax.random.PRNGKey(0), (N,))
        y = model.readout(x, params)

        assert y.shape == ()
        assert not jnp.isnan(y)

    def test_readout_trajectory(self):
        """Test readout for state trajectory."""
        N, T = 50, 100
        cfg = RNNConfig(N=N)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        xs = jax.random.normal(jax.random.PRNGKey(0), (T, N))
        ys = model.readout(xs, params)

        assert ys.shape == (T,)
        assert not jnp.any(jnp.isnan(ys))


class TestLowRankRNNSimulation:
    """Tests for trial simulation."""

    def test_simulate_trial_shapes(self):
        """Test simulation output shapes."""
        N, R = 50, 2
        cfg = RNNConfig(N=N, R=R)
        integ_cfg = IntegratorConfig(dt=0.5, T=10.0)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        # Create dummy trial input
        n_steps = int(10.0 / 0.5) + 1
        trial_input = {
            'times_u': jnp.linspace(0, 10.0, n_steps),
            'u_seq': jnp.zeros((n_steps, 4)),
        }

        times, xs, ys = model.simulate_trial(params, trial_input, integ_cfg)

        assert times.shape[0] == xs.shape[0]
        assert xs.shape[1] == N
        assert ys.shape[0] == times.shape[0]

    def test_simulate_trial_fast_shapes(self):
        """Test fast simulation output shapes."""
        N = 50
        T_steps = 100
        cfg = RNNConfig(N=N)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        u_seq = jnp.zeros((T_steps, 4))
        dt = 0.1

        xs, ys = model.simulate_trial_fast(params, u_seq, dt)

        assert xs.shape == (T_steps, N)
        assert ys.shape == (T_steps,)

    def test_simulate_trial_fast_no_nan(self):
        """Test that fast simulation produces no NaNs."""
        N = 100
        T_steps = 200
        cfg = RNNConfig(N=N, g=0.8)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        # Random input
        u_seq = jax.random.normal(jax.random.PRNGKey(1), (T_steps, 4)) * 0.1
        dt = 0.1

        xs, ys = model.simulate_trial_fast(params, u_seq, dt)

        assert not jnp.any(jnp.isnan(xs))
        assert not jnp.any(jnp.isnan(ys))


class TestLowRankRNNGradients:
    """Tests for gradient computation."""

    def test_gradients_exist(self):
        """Test that gradients can be computed."""
        N = 30
        T_steps = 50
        cfg = RNNConfig(N=N)
        key = jax.random.PRNGKey(42)

        model, params = create_rnn_and_params(cfg, key)

        u_seq = jnp.zeros((T_steps, 4))
        dt = 0.1

        def loss_fn(trainable):
            test_params = RNNParams(
                C=params.C,
                M=trainable['M'],
                N_lr=trainable['N_lr'],
                B=params.B,
                w=params.w,
                b=params.b,
            )
            _, ys = model.simulate_trial_fast(test_params, u_seq, dt)
            return jnp.mean(ys ** 2)

        trainable = {'M': params.M, 'N_lr': params.N_lr}
        loss, grads = jax.value_and_grad(loss_fn)(trainable)

        assert not jnp.isnan(loss)
        assert 'M' in grads
        assert 'N_lr' in grads
        assert grads['M'].shape == params.M.shape
        assert not jnp.any(jnp.isnan(grads['M']))
        assert not jnp.any(jnp.isnan(grads['N_lr']))
