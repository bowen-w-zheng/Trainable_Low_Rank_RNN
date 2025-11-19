"""Loss functions for training the low-rank RNN."""

import jax
import jax.numpy as jnp


def mse_loss(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared error loss.

    Args:
        preds: Predictions, shape (batch,) or (batch, ...)
        targets: Targets, shape (batch,) or (batch, ...)

    Returns:
        MSE loss (scalar)
    """
    return jnp.mean((preds - targets) ** 2)


def binary_cross_entropy_with_logits(
    logits: jnp.ndarray,
    targets: jnp.ndarray
) -> jnp.ndarray:
    """
    Binary cross entropy with logits.

    Args:
        logits: Logits (before sigmoid), shape (batch,)
        targets: Binary targets {0, 1}, shape (batch,)

    Returns:
        BCE loss (scalar)
    """
    # Stable computation
    # BCE = -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
    # = max(x, 0) - x * y + log(1 + exp(-|x|))
    return jnp.mean(
        jnp.maximum(logits, 0) - logits * targets + jnp.log(1 + jnp.exp(-jnp.abs(logits)))
    )


def compute_trial_output(
    ys: jnp.ndarray,
    avg_start_idx: int,
    avg_end_idx: int
) -> jnp.ndarray:
    """
    Compute trial output by averaging over time window.

    Args:
        ys: Readout trajectory, shape (T,) or (batch, T)
        avg_start_idx: Start index of averaging window
        avg_end_idx: End index of averaging window

    Returns:
        Averaged output (scalar or (batch,))
    """
    if ys.ndim == 1:
        return jnp.mean(ys[avg_start_idx:avg_end_idx])
    else:
        return jnp.mean(ys[:, avg_start_idx:avg_end_idx], axis=1)


def compute_trial_loss(
    y_hat: jnp.ndarray,
    label: jnp.ndarray,
    loss_type: str = "mse"
) -> jnp.ndarray:
    """
    Compute loss for a trial.

    Args:
        y_hat: Predicted output (scalar or batch)
        label: Target label (scalar or batch)
        loss_type: "mse" or "bce"

    Returns:
        Loss value (scalar)
    """
    if loss_type == "mse":
        return mse_loss(y_hat, label)
    elif loss_type == "bce":
        return binary_cross_entropy_with_logits(y_hat, label)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_batch_loss(
    model,
    params,
    batch: dict,
    avg_start_idx: int,
    avg_end_idx: int,
    loss_type: str = "mse",
    dt: float = 0.1,
) -> jnp.ndarray:
    """
    Compute loss for a batch of trials.

    This function is designed to be vmapped over the batch.

    Args:
        model: LowRankRNN model
        params: Network parameters
        batch: Batch data with 'u_seq' (batch, T, d_in) and 'labels' (batch,)
        avg_start_idx: Start index for output averaging
        avg_end_idx: End index for output averaging
        loss_type: Loss type
        dt: Time step

    Returns:
        Batch loss (scalar)
    """
    def single_trial_loss(u_seq, label):
        """Compute loss for a single trial."""
        # Simulate trial
        _, ys = model.simulate_trial_fast(params, u_seq, dt)

        # Get output
        y_hat = compute_trial_output(ys, avg_start_idx, avg_end_idx)

        # Compute loss
        return compute_trial_loss(y_hat, label, loss_type)

    # Vectorize over batch
    losses = jax.vmap(single_trial_loss)(batch['u_seq'], batch['labels'])

    return jnp.mean(losses)


def l2_regularization(params, weight_decay: float = 0.0) -> jnp.ndarray:
    """
    Compute L2 regularization on trainable parameters.

    Args:
        params: Dictionary of trainable parameters
        weight_decay: Regularization strength

    Returns:
        Regularization term (scalar)
    """
    if weight_decay == 0.0:
        return 0.0

    reg = 0.0
    for key, val in params.items():
        reg = reg + jnp.sum(val ** 2)

    return weight_decay * reg
