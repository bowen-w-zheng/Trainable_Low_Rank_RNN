"""Metrics for evaluating the low-rank RNN."""

import jax.numpy as jnp


def accuracy_pm1(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Compute accuracy for ±1 targets.

    Args:
        preds: Predictions, shape (batch,)
        targets: Targets in {-1, +1}, shape (batch,)

    Returns:
        Accuracy (scalar between 0 and 1)
    """
    pred_signs = jnp.sign(preds)
    correct = (pred_signs == targets)
    return jnp.mean(correct)


def accuracy_from_logits(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Compute accuracy from logits for binary targets.

    Args:
        logits: Logits, shape (batch,)
        targets: Binary targets {0, 1}, shape (batch,)

    Returns:
        Accuracy (scalar between 0 and 1)
    """
    preds = (logits > 0).astype(jnp.float32)
    correct = (preds == targets)
    return jnp.mean(correct)


def compute_accuracy(
    y_hat: jnp.ndarray,
    labels: jnp.ndarray,
    label_type: str = "pm1"
) -> jnp.ndarray:
    """
    Compute accuracy based on label type.

    Args:
        y_hat: Predictions, shape (batch,)
        labels: Target labels, shape (batch,)
        label_type: "pm1" or "binary"

    Returns:
        Accuracy (scalar)
    """
    if label_type == "pm1":
        return accuracy_pm1(y_hat, labels)
    else:
        return accuracy_from_logits(y_hat, labels)


def compute_batch_predictions(
    model,
    params,
    batch: dict,
    avg_start_idx: int,
    avg_end_idx: int,
    dt: float = 0.1,
):
    """
    Compute predictions for a batch.

    Args:
        model: LowRankRNN model
        params: Network parameters
        batch: Batch data
        avg_start_idx: Start index for averaging
        avg_end_idx: End index for averaging
        dt: Time step

    Returns:
        y_hats: Predictions (batch,)
    """
    import jax

    def single_trial_pred(u_seq):
        _, ys = model.simulate_trial_fast(params, u_seq, dt)
        return jnp.mean(ys[avg_start_idx:avg_end_idx])

    y_hats = jax.vmap(single_trial_pred)(batch['u_seq'])
    return y_hats


def compute_context_accuracy(
    y_hats: jnp.ndarray,
    labels: jnp.ndarray,
    contexts: jnp.ndarray,
    label_type: str = "pm1"
):
    """
    Compute accuracy separately for each context.

    Args:
        y_hats: Predictions (batch,)
        labels: Labels (batch,)
        contexts: Context indices (batch,), values 1 or 2
        label_type: Label type

    Returns:
        dict with 'ctx1_acc' and 'ctx2_acc'
    """
    mask1 = (contexts == 1)
    mask2 = (contexts == 2)

    if label_type == "pm1":
        preds = jnp.sign(y_hats)
        correct = (preds == labels)
    else:
        preds = (y_hats > 0).astype(jnp.float32)
        correct = (preds == labels)

    # Compute accuracy for each context
    ctx1_correct = jnp.sum(correct * mask1)
    ctx1_total = jnp.sum(mask1)
    ctx1_acc = jnp.where(ctx1_total > 0, ctx1_correct / ctx1_total, 0.0)

    ctx2_correct = jnp.sum(correct * mask2)
    ctx2_total = jnp.sum(mask2)
    ctx2_acc = jnp.where(ctx2_total > 0, ctx2_correct / ctx2_total, 0.0)

    return {
        'ctx1_acc': ctx1_acc,
        'ctx2_acc': ctx2_acc,
        'total_acc': jnp.mean(correct),
    }
