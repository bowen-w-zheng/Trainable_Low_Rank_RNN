"""Tests for contextual switch dataset."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.config import TaskConfig, IntegratorConfig
from src.data.contextual_switch_dataset import (
    ContextualSwitchDataset,
    create_dataset,
)


class TestDatasetCreation:
    """Tests for dataset instantiation."""

    def test_create_dataset(self):
        """Test dataset creation."""
        task_cfg = TaskConfig()
        integ_cfg = IntegratorConfig(dt=0.1, T=100.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)

        assert dataset is not None
        assert dataset.n_steps > 0

    def test_time_computation(self):
        """Test time step computation."""
        task_cfg = TaskConfig(T_burn=15.0, T_stim=85.0)
        integ_cfg = IntegratorConfig(dt=0.1)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)

        expected_T = 15.0 + 85.0
        expected_n_steps = int(expected_T / 0.1) + 1
        assert dataset.n_steps == expected_n_steps


class TestSampleTrial:
    """Tests for trial sampling."""

    def test_trial_shapes(self):
        """Test shapes of sampled trial."""
        task_cfg = TaskConfig(T_burn=5.0, T_stim=15.0)
        integ_cfg = IntegratorConfig(dt=1.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)
        trial = dataset.sample_trial(key)

        n_steps = dataset.n_steps
        assert trial['times_u'].shape == (n_steps,)
        assert trial['u_seq'].shape == (n_steps, 4)
        assert trial['label'].shape == ()
        assert trial['context'].shape == ()

    def test_context_values(self):
        """Test that context is 1 or 2."""
        task_cfg = TaskConfig()
        integ_cfg = IntegratorConfig(dt=1.0, T=20.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)

        contexts = []
        for i in range(100):
            key, subkey = jax.random.split(key)
            trial = dataset.sample_trial(subkey)
            contexts.append(int(trial['context']))

        assert set(contexts) == {1, 2}

    def test_label_pm1(self):
        """Test that labels are ±1 for pm1 mode."""
        task_cfg = TaskConfig(label_type="pm1")
        integ_cfg = IntegratorConfig(dt=1.0, T=20.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)

        labels = []
        for i in range(100):
            key, subkey = jax.random.split(key)
            trial = dataset.sample_trial(subkey)
            labels.append(float(trial['label']))

        assert set(labels) == {-1.0, 1.0}

    def test_label_binary(self):
        """Test that labels are 0/1 for binary mode."""
        task_cfg = TaskConfig(label_type="binary")
        integ_cfg = IntegratorConfig(dt=1.0, T=20.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)

        labels = []
        for i in range(100):
            key, subkey = jax.random.split(key)
            trial = dataset.sample_trial(subkey)
            labels.append(float(trial['label']))

        assert set(labels) == {0.0, 1.0}


class TestContextLabelConsistency:
    """Tests for context-label relationship."""

    def test_context1_label_from_s1(self):
        """Test that context 1 label matches sign of s1."""
        task_cfg = TaskConfig(label_type="pm1", stim_std=0.0)
        integ_cfg = IntegratorConfig(dt=1.0, T=20.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)

        for i in range(50):
            key, subkey = jax.random.split(key)
            trial = dataset.sample_trial(subkey)

            if trial['context'] == 1:
                s1 = trial['stim'][0]
                expected_label = jnp.sign(s1)
                assert trial['label'] == expected_label, \
                    f"Context 1: label={trial['label']}, sign(s1)={expected_label}"

    def test_context2_label_from_s2(self):
        """Test that context 2 label matches sign of s2."""
        task_cfg = TaskConfig(label_type="pm1", stim_std=0.0)
        integ_cfg = IntegratorConfig(dt=1.0, T=20.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)

        for i in range(50):
            key, subkey = jax.random.split(key)
            trial = dataset.sample_trial(subkey)

            if trial['context'] == 2:
                s2 = trial['stim'][1]
                expected_label = jnp.sign(s2)
                assert trial['label'] == expected_label, \
                    f"Context 2: label={trial['label']}, sign(s2)={expected_label}"


class TestInputStructure:
    """Tests for input signal structure."""

    def test_stimulus_timing(self):
        """Test that stimuli appear only after burn-in."""
        task_cfg = TaskConfig(T_burn=5.0, T_stim=15.0, stim_mean_abs=1.0)
        integ_cfg = IntegratorConfig(dt=1.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)
        trial = dataset.sample_trial(key)

        # During burn-in, stimulus channels should be 0
        burn_end = dataset.burn_end_idx
        assert jnp.allclose(trial['u_seq'][:burn_end, 0], 0.0)
        assert jnp.allclose(trial['u_seq'][:burn_end, 1], 0.0)

        # After burn-in, stimulus should be non-zero
        assert not jnp.allclose(trial['u_seq'][burn_end:, 0], 0.0)

    def test_context_signal_values(self):
        """Test context signal values."""
        gamma_on, gamma_off = 0.1, -0.2
        task_cfg = TaskConfig(gamma_on=gamma_on, gamma_off=gamma_off)
        integ_cfg = IntegratorConfig(dt=1.0, T=20.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)

        for i in range(20):
            key, subkey = jax.random.split(key)
            trial = dataset.sample_trial(subkey)

            if trial['context'] == 1:
                # Channel 2 should be gamma_on, channel 3 should be gamma_off
                assert jnp.allclose(trial['u_seq'][:, 2], gamma_on)
                assert jnp.allclose(trial['u_seq'][:, 3], gamma_off)
            else:
                # Channel 2 should be gamma_off, channel 3 should be gamma_on
                assert jnp.allclose(trial['u_seq'][:, 2], gamma_off)
                assert jnp.allclose(trial['u_seq'][:, 3], gamma_on)


class TestBatchSampling:
    """Tests for batch sampling."""

    def test_batch_shapes(self):
        """Test shapes of batch output."""
        task_cfg = TaskConfig(T_burn=3.0, T_stim=7.0)
        integ_cfg = IntegratorConfig(dt=1.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)
        batch_size = 16
        batch = dataset.sample_batch(key, batch_size)

        n_steps = dataset.n_steps
        assert batch['u_seq'].shape == (batch_size, n_steps, 4)
        assert batch['labels'].shape == (batch_size,)
        assert batch['contexts'].shape == (batch_size,)
        assert batch['stims'].shape == (batch_size, 2)

    def test_batch_diversity(self):
        """Test that batch contains diverse samples."""
        task_cfg = TaskConfig()
        integ_cfg = IntegratorConfig(dt=1.0, T=20.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)
        batch = dataset.sample_batch(key, 100)

        # Should have both contexts
        contexts = np.array(batch['contexts'])
        assert 1 in contexts
        assert 2 in contexts

        # Should have both label signs
        labels = np.array(batch['labels'])
        assert -1 in labels or 0 in labels
        assert 1 in labels


class TestAveragingWindow:
    """Tests for averaging window computation."""

    def test_avg_window_indices(self):
        """Test averaging window index computation."""
        task_cfg = TaskConfig(T_burn=10.0, T_stim=90.0, T_avg=10.0)
        integ_cfg = IntegratorConfig(dt=1.0)
        key = jax.random.PRNGKey(42)

        dataset = create_dataset(task_cfg, integ_cfg, key)

        start_idx, end_idx = dataset.get_avg_window_indices()

        # T = 100, T_avg = 10, so start should be at t=90
        expected_start = int(90.0 / 1.0)
        expected_end = dataset.n_steps

        assert start_idx == expected_start
        assert end_idx == expected_end
