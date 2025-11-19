"""Tests for loss functions and metrics."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.training.losses import (
    mse_loss,
    binary_cross_entropy_with_logits,
    compute_trial_output,
    compute_trial_loss,
    l2_regularization,
)
from src.training.metrics import (
    accuracy_pm1,
    accuracy_from_logits,
    compute_accuracy,
    compute_context_accuracy,
)


class TestMSELoss:
    """Tests for MSE loss function."""

    def test_zero_loss(self):
        """Test MSE is zero for identical arrays."""
        preds = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        loss = mse_loss(preds, targets)
        assert jnp.allclose(loss, 0.0)

    def test_known_value(self):
        """Test MSE with known values."""
        preds = jnp.array([1.0, 2.0])
        targets = jnp.array([2.0, 4.0])
        # MSE = ((1-2)^2 + (2-4)^2) / 2 = (1 + 4) / 2 = 2.5
        loss = mse_loss(preds, targets)
        assert jnp.allclose(loss, 2.5)

    def test_symmetric(self):
        """Test that MSE is symmetric."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        assert jnp.allclose(mse_loss(a, b), mse_loss(b, a))


class TestBCELoss:
    """Tests for binary cross entropy loss."""

    def test_correct_predictions(self):
        """Test BCE for confident correct predictions."""
        # Large positive logit for target 1
        logits = jnp.array([10.0])
        targets = jnp.array([1.0])
        loss = binary_cross_entropy_with_logits(logits, targets)
        assert loss < 0.01

        # Large negative logit for target 0
        logits = jnp.array([-10.0])
        targets = jnp.array([0.0])
        loss = binary_cross_entropy_with_logits(logits, targets)
        assert loss < 0.01

    def test_wrong_predictions(self):
        """Test BCE for confident wrong predictions."""
        logits = jnp.array([10.0])
        targets = jnp.array([0.0])
        loss = binary_cross_entropy_with_logits(logits, targets)
        assert loss > 1.0

    def test_uncertain_predictions(self):
        """Test BCE for uncertain predictions (logit=0)."""
        logits = jnp.array([0.0])
        targets = jnp.array([1.0])
        # BCE = log(2) ≈ 0.693
        loss = binary_cross_entropy_with_logits(logits, targets)
        assert jnp.allclose(loss, jnp.log(2.0), atol=1e-5)


class TestComputeTrialOutput:
    """Tests for trial output computation."""

    def test_single_trajectory(self):
        """Test output averaging for single trajectory."""
        ys = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Average over last 2 elements
        output = compute_trial_output(ys, 3, 5)
        expected = (4.0 + 5.0) / 2
        assert jnp.allclose(output, expected)

    def test_batch_trajectory(self):
        """Test output averaging for batch."""
        ys = jnp.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])
        output = compute_trial_output(ys, 3, 5)
        expected = jnp.array([4.5, 45.0])
        assert jnp.allclose(output, expected)


class TestComputeTrialLoss:
    """Tests for trial loss computation."""

    def test_mse_loss_type(self):
        """Test MSE loss computation."""
        y_hat = jnp.array([1.0, -1.0])
        label = jnp.array([1.0, 1.0])
        loss = compute_trial_loss(y_hat, label, "mse")
        # MSE = ((1-1)^2 + (-1-1)^2) / 2 = 2.0
        assert jnp.allclose(loss, 2.0)

    def test_bce_loss_type(self):
        """Test BCE loss computation."""
        y_hat = jnp.array([10.0])
        label = jnp.array([1.0])
        loss = compute_trial_loss(y_hat, label, "bce")
        assert loss < 0.01


class TestL2Regularization:
    """Tests for L2 regularization."""

    def test_zero_weight_decay(self):
        """Test that zero weight decay gives zero regularization."""
        params = {'M': jnp.ones((10, 2)), 'N': jnp.ones((10, 2))}
        reg = l2_regularization(params, 0.0)
        assert reg == 0.0

    def test_nonzero_regularization(self):
        """Test L2 regularization computation."""
        params = {'w': jnp.array([1.0, 2.0, 3.0])}
        # Sum of squares = 1 + 4 + 9 = 14
        reg = l2_regularization(params, 0.1)
        assert jnp.allclose(reg, 0.1 * 14.0)


class TestAccuracyPM1:
    """Tests for ±1 accuracy metric."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        preds = jnp.array([1.0, -1.0, 1.0, -1.0])
        targets = jnp.array([1.0, -1.0, 1.0, -1.0])
        acc = accuracy_pm1(preds, targets)
        assert jnp.allclose(acc, 1.0)

    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        preds = jnp.array([1.0, 1.0, 1.0, 1.0])
        targets = jnp.array([-1.0, -1.0, -1.0, -1.0])
        acc = accuracy_pm1(preds, targets)
        assert jnp.allclose(acc, 0.0)

    def test_half_accuracy(self):
        """Test 50% accuracy."""
        preds = jnp.array([1.0, 1.0, -1.0, -1.0])
        targets = jnp.array([1.0, -1.0, -1.0, 1.0])
        acc = accuracy_pm1(preds, targets)
        assert jnp.allclose(acc, 0.5)

    def test_sign_thresholding(self):
        """Test that any positive value counts as +1."""
        preds = jnp.array([0.1, -0.1, 100.0, -100.0])
        targets = jnp.array([1.0, -1.0, 1.0, -1.0])
        acc = accuracy_pm1(preds, targets)
        assert jnp.allclose(acc, 1.0)


class TestAccuracyFromLogits:
    """Tests for binary accuracy from logits."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        logits = jnp.array([10.0, -10.0, 5.0, -5.0])
        targets = jnp.array([1.0, 0.0, 1.0, 0.0])
        acc = accuracy_from_logits(logits, targets)
        assert jnp.allclose(acc, 1.0)

    def test_threshold_at_zero(self):
        """Test that threshold is at 0."""
        logits = jnp.array([0.001, -0.001])
        targets = jnp.array([1.0, 0.0])
        acc = accuracy_from_logits(logits, targets)
        assert jnp.allclose(acc, 1.0)


class TestComputeAccuracy:
    """Tests for unified accuracy computation."""

    def test_pm1_mode(self):
        """Test accuracy in pm1 mode."""
        y_hat = jnp.array([1.0, -1.0])
        labels = jnp.array([1.0, -1.0])
        acc = compute_accuracy(y_hat, labels, "pm1")
        assert jnp.allclose(acc, 1.0)

    def test_binary_mode(self):
        """Test accuracy in binary mode."""
        y_hat = jnp.array([1.0, -1.0])
        labels = jnp.array([1.0, 0.0])
        acc = compute_accuracy(y_hat, labels, "binary")
        assert jnp.allclose(acc, 1.0)


class TestContextAccuracy:
    """Tests for context-specific accuracy."""

    def test_separate_context_accuracy(self):
        """Test that accuracy is computed separately per context."""
        y_hats = jnp.array([1.0, 1.0, -1.0, -1.0])
        labels = jnp.array([1.0, -1.0, -1.0, 1.0])  # 50% each context
        contexts = jnp.array([1, 1, 2, 2])

        result = compute_context_accuracy(y_hats, labels, contexts, "pm1")

        assert jnp.allclose(result['ctx1_acc'], 0.5)
        assert jnp.allclose(result['ctx2_acc'], 0.5)
        assert jnp.allclose(result['total_acc'], 0.5)

    def test_different_context_accuracy(self):
        """Test when contexts have different accuracy."""
        y_hats = jnp.array([1.0, 1.0, -1.0, -1.0])
        labels = jnp.array([1.0, 1.0, 1.0, 1.0])  # ctx1 correct, ctx2 wrong
        contexts = jnp.array([1, 1, 2, 2])

        result = compute_context_accuracy(y_hats, labels, contexts, "pm1")

        assert jnp.allclose(result['ctx1_acc'], 1.0)
        assert jnp.allclose(result['ctx2_acc'], 0.0)
