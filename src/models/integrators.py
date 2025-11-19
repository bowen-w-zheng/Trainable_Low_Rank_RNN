"""ODE integrator for RNN dynamics using diffrax."""

from typing import Callable, Any, Tuple
import jax
import jax.numpy as jnp
import diffrax

from src.config import IntegratorConfig


def get_solver(solver_name: str) -> diffrax.AbstractSolver:
    """Get diffrax solver by name."""
    solvers = {
        "tsit5": diffrax.Tsit5(),
        "dopri5": diffrax.Dopri5(),
        "euler": diffrax.Euler(),
        "heun": diffrax.Heun(),
    }
    if solver_name.lower() not in solvers:
        raise ValueError(f"Unknown solver: {solver_name}. Available: {list(solvers.keys())}")
    return solvers[solver_name.lower()]


def integrate_rnn_dynamics(
    f: Callable,  # f(t, x, args) -> dx/dt
    x0: jnp.ndarray,  # Initial state, shape (N,)
    args: Any,  # Arguments passed to f (params, input function, etc.)
    integ_cfg: IntegratorConfig,
    t0: float = 0.0,
    t1: float = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Integrate RNN dynamics using diffrax.

    Args:
        f: RHS function with signature f(t, x, args) -> dx/dt
        x0: Initial state vector
        args: Additional arguments for f (typically params and input)
        integ_cfg: Integrator configuration
        t0: Start time
        t1: End time (defaults to integ_cfg.T)

    Returns:
        times: Array of time points, shape (n_steps,)
        xs: Array of states, shape (n_steps, N)
    """
    if t1 is None:
        t1 = integ_cfg.T

    # Get solver
    solver = get_solver(integ_cfg.solver_name)

    # Create ODE term
    term = diffrax.ODETerm(f)

    # Set up step size controller
    stepsize_controller = diffrax.PIDController(
        rtol=integ_cfg.rtol,
        atol=integ_cfg.atol,
    )

    # Compute save times
    n_steps = int((t1 - t0) / integ_cfg.dt) + 1
    save_times = jnp.linspace(t0, t1, n_steps)

    # Set up saveat to save at specific times
    saveat = diffrax.SaveAt(ts=save_times)

    # Solve ODE
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=integ_cfg.dt,
        y0=x0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e6),
    )

    return solution.ts, solution.ys


def integrate_rnn_dynamics_fixed_step(
    f: Callable,  # f(t, x, args) -> dx/dt
    x0: jnp.ndarray,  # Initial state, shape (N,)
    args: Any,  # Arguments passed to f
    integ_cfg: IntegratorConfig,
    t0: float = 0.0,
    t1: float = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Integrate RNN dynamics using fixed step size (faster for training).

    This uses Euler or Heun method with fixed step size, which can be
    more efficient for training when we don't need adaptive stepping.

    Args:
        f: RHS function with signature f(t, x, args) -> dx/dt
        x0: Initial state vector
        args: Additional arguments for f
        integ_cfg: Integrator configuration
        t0: Start time
        t1: End time (defaults to integ_cfg.T)

    Returns:
        times: Array of time points, shape (n_steps,)
        xs: Array of states, shape (n_steps, N)
    """
    if t1 is None:
        t1 = integ_cfg.T

    # Compute number of steps
    n_steps = int((t1 - t0) / integ_cfg.dt) + 1
    times = jnp.linspace(t0, t1, n_steps)
    dt = integ_cfg.dt

    def scan_fn(x, t):
        """One Euler step."""
        dx = f(t, x, args)
        x_next = x + dt * dx
        return x_next, x

    # Run integration using scan
    _, xs = jax.lax.scan(scan_fn, x0, times[:-1])

    # Prepend initial state
    xs = jnp.concatenate([x0[None, :], xs], axis=0)

    return times, xs


def make_input_interpolator(times_u: jnp.ndarray, u_seq: jnp.ndarray) -> Callable:
    """
    Create an interpolator for input signals.

    Uses piecewise constant (nearest neighbor) interpolation.

    Args:
        times_u: Time points for input, shape (T_u,)
        u_seq: Input values, shape (T_u, d_in)

    Returns:
        Function that takes time t and returns input u(t)
    """
    def u_of_t(t: float) -> jnp.ndarray:
        # Find nearest time index (piecewise constant)
        idx = jnp.searchsorted(times_u, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(times_u) - 1)
        return u_seq[idx]

    return u_of_t
