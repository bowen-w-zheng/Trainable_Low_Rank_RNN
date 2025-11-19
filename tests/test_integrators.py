"""Tests for ODE integrators."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.config import IntegratorConfig
from src.models.integrators import (
    integrate_rnn_dynamics,
    integrate_rnn_dynamics_fixed_step,
    get_solver,
    make_input_interpolator,
)


class TestGetSolver:
    """Tests for solver selection."""

    def test_get_tsit5(self):
        """Test getting Tsit5 solver."""
        solver = get_solver("tsit5")
        assert solver is not None

    def test_get_dopri5(self):
        """Test getting Dopri5 solver."""
        solver = get_solver("dopri5")
        assert solver is not None

    def test_get_euler(self):
        """Test getting Euler solver."""
        solver = get_solver("euler")
        assert solver is not None

    def test_unknown_solver(self):
        """Test error on unknown solver."""
        with pytest.raises(ValueError):
            get_solver("unknown_solver")


class TestIntegrateRNNDynamics:
    """Tests for ODE integration."""

    def test_linear_ode(self):
        """
        Test integration of simple linear ODE: dx/dt = -x.

        Analytic solution: x(t) = x0 * exp(-t)
        """
        def f(t, x, args):
            return -x

        x0 = jnp.array([1.0])
        integ_cfg = IntegratorConfig(dt=0.01, T=1.0, solver_name="tsit5")

        times, xs = integrate_rnn_dynamics(f, x0, None, integ_cfg)

        # Check final value
        expected = jnp.exp(-1.0)
        actual = xs[-1, 0]
        assert jnp.abs(actual - expected) < 1e-3, f"Expected {expected}, got {actual}"

        # Check shapes
        assert times.shape[0] == xs.shape[0]
        assert xs.shape[1] == 1

    def test_2d_linear_ode(self):
        """Test integration of 2D linear system."""
        # dx/dt = Ax with A = [[-1, 0], [0, -2]]
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])

        def f(t, x, args):
            return A @ x

        x0 = jnp.array([1.0, 1.0])
        integ_cfg = IntegratorConfig(dt=0.01, T=1.0, solver_name="tsit5")

        times, xs = integrate_rnn_dynamics(f, x0, None, integ_cfg)

        # Check final values
        expected = jnp.array([jnp.exp(-1.0), jnp.exp(-2.0)])
        actual = xs[-1]
        assert jnp.allclose(actual, expected, atol=1e-3)

    def test_with_input(self):
        """Test integration with external input."""
        # dx/dt = -x + u(t) with constant u = 1
        def f(t, x, args):
            u_of_t = args
            u = u_of_t(t)
            return -x + u

        x0 = jnp.array([0.0])
        integ_cfg = IntegratorConfig(dt=0.01, T=2.0, solver_name="tsit5")

        # Constant input
        def u_of_t(t):
            return jnp.array([1.0])

        times, xs = integrate_rnn_dynamics(f, x0, u_of_t, integ_cfg)

        # With constant input u=1, solution approaches 1
        # x(t) = 1 - exp(-t)
        expected = 1.0 - jnp.exp(-2.0)
        actual = xs[-1, 0]
        assert jnp.abs(actual - expected) < 1e-2


class TestFixedStepIntegration:
    """Tests for fixed-step integration."""

    def test_euler_integration(self):
        """Test Euler method integration."""
        def f(t, x, args):
            return -x

        x0 = jnp.array([1.0])
        integ_cfg = IntegratorConfig(dt=0.001, T=1.0)  # Small dt for accuracy

        times, xs = integrate_rnn_dynamics_fixed_step(f, x0, None, integ_cfg)

        # Check final value (Euler has lower accuracy)
        expected = jnp.exp(-1.0)
        actual = xs[-1, 0]
        assert jnp.abs(actual - expected) < 0.01

    def test_shapes(self):
        """Test output shapes."""
        def f(t, x, args):
            return -x

        N = 10
        x0 = jnp.ones(N)
        integ_cfg = IntegratorConfig(dt=0.1, T=1.0)

        times, xs = integrate_rnn_dynamics_fixed_step(f, x0, None, integ_cfg)

        assert times.shape[0] == 11  # 0, 0.1, ..., 1.0
        assert xs.shape == (11, N)


class TestInputInterpolator:
    """Tests for input interpolation."""

    def test_constant_input(self):
        """Test interpolation of constant input."""
        times_u = jnp.array([0.0, 1.0, 2.0])
        u_seq = jnp.array([[1.0], [1.0], [1.0]])

        u_of_t = make_input_interpolator(times_u, u_seq)

        assert jnp.allclose(u_of_t(0.0), jnp.array([1.0]))
        assert jnp.allclose(u_of_t(0.5), jnp.array([1.0]))
        assert jnp.allclose(u_of_t(1.5), jnp.array([1.0]))

    def test_step_input(self):
        """Test interpolation of step input."""
        times_u = jnp.array([0.0, 1.0, 2.0])
        u_seq = jnp.array([[0.0], [1.0], [2.0]])

        u_of_t = make_input_interpolator(times_u, u_seq)

        # Should use piecewise constant (nearest neighbor)
        assert jnp.allclose(u_of_t(0.5), jnp.array([0.0]))
        assert jnp.allclose(u_of_t(1.5), jnp.array([1.0]))

    def test_multidimensional(self):
        """Test interpolation with multiple input channels."""
        times_u = jnp.array([0.0, 1.0])
        u_seq = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        u_of_t = make_input_interpolator(times_u, u_seq)

        u = u_of_t(0.5)
        assert u.shape == (4,)
        assert jnp.allclose(u, jnp.array([1.0, 2.0, 3.0, 4.0]))
