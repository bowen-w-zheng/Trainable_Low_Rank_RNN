"""Tests for configuration module."""

import pytest
import tempfile
import os

from src.config import (
    RNNConfig,
    IntegratorConfig,
    TrainingConfig,
    TaskConfig,
    ExperimentConfig,
)


class TestRNNConfig:
    """Tests for RNNConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        cfg = RNNConfig()
        assert cfg.N == 500
        assert cfg.R == 2
        assert cfg.g == 0.8
        assert cfg.tau == 1.0
        assert cfg.phi == "tanh"
        assert cfg.d_in == 4
        assert not cfg.use_bias

    def test_custom_values(self):
        """Test creating config with custom values."""
        cfg = RNNConfig(N=1000, R=3, g=1.0)
        assert cfg.N == 1000
        assert cfg.R == 3
        assert cfg.g == 1.0


class TestIntegratorConfig:
    """Tests for IntegratorConfig dataclass."""

    def test_default_values(self):
        """Test default integrator settings."""
        cfg = IntegratorConfig()
        assert cfg.dt == 0.1
        assert cfg.T == 100.0
        assert cfg.solver_name == "tsit5"

    def test_custom_values(self):
        """Test custom integrator settings."""
        cfg = IntegratorConfig(dt=0.01, T=50.0, solver_name="dopri5")
        assert cfg.dt == 0.01
        assert cfg.T == 50.0
        assert cfg.solver_name == "dopri5"


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default training settings."""
        cfg = TrainingConfig()
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 1e-3
        assert cfg.seed == 42
        assert cfg.train_M is True
        assert cfg.train_N is True
        assert cfg.train_B is True
        assert cfg.train_w is True
        assert cfg.training_mode == "low_rank"

    def test_full_rank_mode(self):
        """Test setting full_rank training mode."""
        cfg = TrainingConfig(training_mode="full_rank")
        assert cfg.training_mode == "full_rank"


class TestTaskConfig:
    """Tests for TaskConfig dataclass."""

    def test_default_values(self):
        """Test default task settings."""
        cfg = TaskConfig()
        assert cfg.stim_mean_abs == 1.2
        assert cfg.gamma_on == 0.08
        assert cfg.gamma_off == -0.14
        assert cfg.T_burn == 15.0
        assert cfg.label_type == "pm1"
        assert cfg.loss_type == "mse"


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_instantiation(self):
        """Test creating default experiment config."""
        cfg = ExperimentConfig()
        assert cfg.rnn.N == 500
        assert cfg.integrator.dt == 0.1
        assert cfg.training.batch_size == 32
        assert cfg.task.stim_mean_abs == 1.2

    def test_yaml_roundtrip(self):
        """Test saving and loading from YAML."""
        cfg = ExperimentConfig()
        cfg.rnn.N = 123
        cfg.training.learning_rate = 0.005

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_config.yaml")
            cfg.to_yaml(path)

            loaded_cfg = ExperimentConfig.from_yaml(path)

            assert loaded_cfg.rnn.N == 123
            assert loaded_cfg.training.learning_rate == 0.005

    def test_load_from_configs_dir(self):
        """Test loading from actual config files."""
        sanity_path = "configs/contextual_switch_sanity.yaml"
        if os.path.exists(sanity_path):
            cfg = ExperimentConfig.from_yaml(sanity_path)
            assert cfg.rnn.N == 50
            assert cfg.integrator.T == 20.0
