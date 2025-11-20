#!/usr/bin/env python3
"""Visualize temporal decision task dataset."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.temporal_decision_dataset import (
    TemporalDecisionDataset,
    TemporalDecisionTaskConfig,
    create_temporal_decision_dataset,
)


def load_task_config(config_path: str):
    """Load task configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    task_data = data.get('task', {})
    task_cfg = TemporalDecisionTaskConfig(**task_data)

    return task_cfg


def visualize_example_trials(dataset, key, n_trials=6, save_path=None):
    """
    Visualize a few example trials with different contexts.

    Args:
        dataset: TemporalDecisionDataset instance
        key: Random key
        n_trials: Number of trials to show
        save_path: Path to save figure
    """
    keys = jax.random.split(key, n_trials)

    fig, axes = plt.subplots(n_trials, 3, figsize=(15, 3 * n_trials))
    if n_trials == 1:
        axes = axes.reshape(1, -1)

    # Sample trials with different contexts
    contexts_to_sample = np.linspace(0, 1, n_trials)

    for i in range(n_trials):
        # Sample trial with specific context
        trial = dataset.sample_trial_fixed_context(keys[i], contexts_to_sample[i])

        times = np.array(trial['times'])
        u_seq = np.array(trial['u_seq'])
        y_time = np.array(trial['y_time'])
        context = float(trial['context'])
        g_bar = float(trial['g_bar'])
        a1 = float(trial['a1'])
        a2 = float(trial['a2'])

        u1 = u_seq[:, 0]
        u2 = u_seq[:, 1]
        g = (1 - context) * u1 + context * u2

        # Plot 1: Inputs
        ax1 = axes[i, 0]
        ax1.plot(times, u1, 'b-', label='u1', linewidth=1.5)
        ax1.plot(times, u2, 'r-', label='u2', linewidth=1.5)
        ax1.axvline(dataset.task_cfg.t_stim_on, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(dataset.task_cfg.t_stim_off, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Stimulus')
        ax1.set_title(f'Trial {i+1}: c={context:.2f}, a1={a1:.2f}, a2={a2:.2f}')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Evidence
        ax2 = axes[i, 1]
        ax2.plot(times, g, 'm-', label='g(t)', linewidth=1.5)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(dataset.task_cfg.t_stim_on, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(dataset.task_cfg.t_stim_off, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Evidence')
        ax2.set_title(f'Evidence g(t) = (1-c)*u1 + c*u2')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Target
        ax3 = axes[i, 2]
        ax3.plot(times, y_time, 'k-', linewidth=2, label='Target')
        ax3.axhline(g_bar, color='m', linestyle=':', alpha=0.7, linewidth=2)
        ax3.axvspan(dataset.task_cfg.t_stim_on, dataset.task_cfg.t_stim_off,
                    alpha=0.1, color='blue', label='Stimulus')
        ax3.axvspan(dataset.task_cfg.t_response_on, dataset.task_cfg.t_response_off,
                    alpha=0.1, color='green', label='Response')
        ax3.set_ylabel('Target g_bar')
        ax3.set_xlabel('Time (s)')
        ax3.text(0.95, 0.5, f'g_bar={g_bar:.3f}', transform=ax3.transAxes,
                fontsize=10, ha='right', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)

    nonlin = dataset.task_cfg.target_nonlinearity
    plt.suptitle(f'Example Trials (nonlinearity: {nonlin})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved example trials to: {save_path}")

    return fig


def visualize_function_mapping(dataset, key, n_contexts=5, n_samples=50, save_path=None):
    """
    Visualize how the function maps inputs to outputs for different contexts.

    Fix a1, vary a2, and show output for different context values.

    Args:
        dataset: TemporalDecisionDataset instance
        key: Random key
        n_contexts: Number of context values to show
        n_samples: Number of samples along a2 axis
        save_path: Path to save figure
    """
    contexts = np.linspace(0, 1, n_contexts)
    a1_fixed = 0.5  # Fix a1
    a2_values = np.linspace(-1, 1, n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute g_bar for each (a2, context) pair
    g_bars = np.zeros((n_contexts, n_samples))

    keys = jax.random.split(key, n_contexts * n_samples)
    key_idx = 0

    for i, c in enumerate(contexts):
        for j, a2 in enumerate(a2_values):
            trial = dataset.sample_trial_fixed_context(
                keys[key_idx], context=c, a1=a1_fixed, a2=a2
            )
            g_bars[i, j] = float(trial['g_bar'])
            key_idx += 1

    # Plot 1: Function mapping
    ax1 = axes[0]
    for i, c in enumerate(contexts):
        ax1.plot(a2_values, g_bars[i], label=f'c={c:.2f}', linewidth=2)

    ax1.set_xlabel('a2 (stimulus 2)', fontsize=12)
    ax1.set_ylabel('g_bar (output)', fontsize=12)
    ax1.set_title(f'Function Mapping: g_bar vs a2 (a1={a1_fixed} fixed)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='--', alpha=0.3)

    # Plot 2: Heatmap
    ax2 = axes[1]
    im = ax2.imshow(g_bars, aspect='auto', origin='lower',
                     extent=[a2_values[0], a2_values[-1], contexts[0], contexts[-1]],
                     cmap='RdBu_r')
    ax2.set_xlabel('a2 (stimulus 2)', fontsize=12)
    ax2.set_ylabel('Context c', fontsize=12)
    ax2.set_title(f'Output Heatmap (a1={a1_fixed} fixed)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('g_bar', fontsize=10)

    nonlin = dataset.task_cfg.target_nonlinearity
    fig.suptitle(f'Function Mapping (nonlinearity: {nonlin})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved function mapping to: {save_path}")

    return fig


def visualize_output_distribution(dataset, key, n_samples=5000, save_path=None):
    """
    Visualize the distribution of outputs by sampling many trials.

    Args:
        dataset: TemporalDecisionDataset instance
        key: Random key
        n_samples: Number of trials to sample
        save_path: Path to save figure
    """
    # Sample many trials
    keys = jax.random.split(key, n_samples)

    g_bars = []
    contexts = []
    a1s = []
    a2s = []

    print(f"Sampling {n_samples} trials...")
    for i, k in enumerate(keys):
        if (i + 1) % 1000 == 0:
            print(f"  Sampled {i+1}/{n_samples}")
        trial = dataset.sample_trial(k)
        g_bars.append(float(trial['g_bar']))
        contexts.append(float(trial['context']))
        a1s.append(float(trial['a1']))
        a2s.append(float(trial['a2']))

    g_bars = np.array(g_bars)
    contexts = np.array(contexts)
    a1s = np.array(a1s)
    a2s = np.array(a2s)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Histogram of g_bar
    ax1 = axes[0, 0]
    ax1.hist(g_bars, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('g_bar (output)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Outputs', fontsize=12, fontweight='bold')
    ax1.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
    ax1.axvline(np.mean(g_bars), color='g', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(g_bars):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Output vs Context
    ax2 = axes[0, 1]
    ax2.scatter(contexts, g_bars, alpha=0.3, s=10)
    ax2.set_xlabel('Context c', fontsize=12)
    ax2.set_ylabel('g_bar (output)', fontsize=12)
    ax2.set_title('Output vs Context', fontsize=12, fontweight='bold')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Output vs a1
    ax3 = axes[1, 0]
    ax3.scatter(a1s, g_bars, alpha=0.3, s=10, c=contexts, cmap='viridis')
    ax3.set_xlabel('a1 (stimulus 1)', fontsize=12)
    ax3.set_ylabel('g_bar (output)', fontsize=12)
    ax3.set_title('Output vs Stimulus 1 (colored by context)', fontsize=12, fontweight='bold')
    ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax3.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Output vs a2
    ax4 = axes[1, 1]
    sc = ax4.scatter(a2s, g_bars, alpha=0.3, s=10, c=contexts, cmap='viridis')
    ax4.set_xlabel('a2 (stimulus 2)', fontsize=12)
    ax4.set_ylabel('g_bar (output)', fontsize=12)
    ax4.set_title('Output vs Stimulus 2 (colored by context)', fontsize=12, fontweight='bold')
    ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label('Context c', fontsize=10)

    # Statistics
    stats_text = (
        f'Statistics (n={n_samples}):\n'
        f'  Mean: {np.mean(g_bars):.4f}\n'
        f'  Std: {np.std(g_bars):.4f}\n'
        f'  Min: {np.min(g_bars):.4f}\n'
        f'  Max: {np.max(g_bars):.4f}'
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    nonlin = dataset.task_cfg.target_nonlinearity
    fig.suptitle(f'Output Distribution (nonlinearity: {nonlin})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved output distribution to: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize temporal decision task dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/temporal_decision_hard_test.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figs/dataset_visualization',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=6,
        help='Number of example trials to show'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5000,
        help='Number of samples for distribution histogram'
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    print(f"Loading config from: {args.config}")
    task_cfg = load_task_config(args.config)

    # Create dataset
    key = jax.random.PRNGKey(args.seed)
    dataset = create_temporal_decision_dataset(task_cfg, key)

    print(f"\nDataset configuration:")
    print(f"  Time step: {task_cfg.dt}s")
    print(f"  Trial duration: {task_cfg.T_trial}s")
    print(f"  Stimulus window: [{task_cfg.t_stim_on}, {task_cfg.t_stim_off}]s")
    print(f"  Response window: [{task_cfg.t_response_on}, {task_cfg.t_response_off}]s")
    print(f"  Input noise std: {task_cfg.input_noise_std}")
    print(f"  Target nonlinearity: {task_cfg.target_nonlinearity}")
    if task_cfg.train_contexts is not None:
        print(f"  Train contexts (discrete): {task_cfg.train_contexts}")
        print(f"  Test contexts (discrete): {task_cfg.test_contexts}")
    elif task_cfg.train_context_ranges is not None:
        print(f"  Train context ranges: {task_cfg.train_context_ranges}")
        print(f"  Test context ranges: {task_cfg.test_context_ranges}")
    else:
        print(f"  Contexts: Uniform [0, 1]")
    print()

    # Generate visualizations
    print("=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # 1. Example trials
    print("\n1. Generating example trials...")
    key, subkey = jax.random.split(key)
    fig1 = visualize_example_trials(
        dataset, subkey,
        n_trials=args.n_trials,
        save_path=os.path.join(args.output_dir, 'example_trials.png')
    )

    # 2. Function mapping
    print("\n2. Generating function mapping...")
    key, subkey = jax.random.split(key)
    fig2 = visualize_function_mapping(
        dataset, subkey,
        save_path=os.path.join(args.output_dir, 'function_mapping.png')
    )

    # 3. Output distribution
    print("\n3. Generating output distribution...")
    key, subkey = jax.random.split(key)
    fig3 = visualize_output_distribution(
        dataset, subkey,
        n_samples=args.n_samples,
        save_path=os.path.join(args.output_dir, 'output_distribution.png')
    )

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {args.output_dir}")
    print("=" * 60)
    print("\nFiles generated:")
    print(f"  - {args.output_dir}/example_trials.png")
    print(f"  - {args.output_dir}/function_mapping.png")
    print(f"  - {args.output_dir}/output_distribution.png")

    # Show plots if in interactive mode
    # plt.show()


if __name__ == '__main__':
    main()
