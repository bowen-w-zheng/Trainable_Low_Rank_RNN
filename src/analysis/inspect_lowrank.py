"""Inspect low-rank structure of trained network."""

import os
from typing import Optional
import jax.numpy as jnp
import numpy as np

from src.models.lowrank_rnn import RNNParams


def compute_overlaps(params: RNNParams) -> dict:
    """
    Compute overlaps between network vectors.

    Computes normalized dot products between columns of M, N,
    input columns of B, and readout w.

    Args:
        params: Network parameters

    Returns:
        Dictionary of overlap values
    """
    N = params.M.shape[0]
    R = params.M.shape[1]
    d_in = params.B.shape[1]

    overlaps = {}

    # Normalize vectors
    def normalize(v):
        norm = jnp.linalg.norm(v)
        return v / (norm + 1e-8)

    # M columns self-overlap
    for i in range(R):
        for j in range(i + 1, R):
            m_i = normalize(params.M[:, i])
            m_j = normalize(params.M[:, j])
            overlaps[f'M{i}_M{j}'] = float(jnp.dot(m_i, m_j))

    # N columns self-overlap
    for i in range(R):
        for j in range(i + 1, R):
            n_i = normalize(params.N_lr[:, i])
            n_j = normalize(params.N_lr[:, j])
            overlaps[f'N{i}_N{j}'] = float(jnp.dot(n_i, n_j))

    # M-N overlaps
    for i in range(R):
        for j in range(R):
            m_i = normalize(params.M[:, i])
            n_j = normalize(params.N_lr[:, j])
            overlaps[f'M{i}_N{j}'] = float(jnp.dot(m_i, n_j))

    # M-w overlaps
    w_norm = normalize(params.w)
    for i in range(R):
        m_i = normalize(params.M[:, i])
        overlaps[f'M{i}_w'] = float(jnp.dot(m_i, w_norm))

    # N-w overlaps
    for i in range(R):
        n_i = normalize(params.N_lr[:, i])
        overlaps[f'N{i}_w'] = float(jnp.dot(n_i, w_norm))

    # M-B overlaps (input channels)
    input_names = ['s1', 's2', 'ctx1', 'ctx2']
    for i in range(R):
        m_i = normalize(params.M[:, i])
        for k in range(min(d_in, 4)):
            b_k = normalize(params.B[:, k])
            overlaps[f'M{i}_{input_names[k]}'] = float(jnp.dot(m_i, b_k))

    # N-B overlaps
    for i in range(R):
        n_i = normalize(params.N_lr[:, i])
        for k in range(min(d_in, 4)):
            b_k = normalize(params.B[:, k])
            overlaps[f'N{i}_{input_names[k]}'] = float(jnp.dot(n_i, b_k))

    # w-B overlaps
    for k in range(min(d_in, 4)):
        b_k = normalize(params.B[:, k])
        overlaps[f'w_{input_names[k]}'] = float(jnp.dot(w_norm, b_k))

    return overlaps


def compute_effective_rank(params: RNNParams) -> dict:
    """
    Compute effective rank and spectral properties of connectivity.

    Args:
        params: Network parameters

    Returns:
        Dictionary with spectral analysis
    """
    N = params.M.shape[0]
    g = 0.8  # Assume default, or pass as argument

    # Low-rank part only
    J_lr = (1.0 / N) * params.M @ params.N_lr.T

    # Compute SVD of low-rank part
    u, s, vh = jnp.linalg.svd(J_lr, full_matrices=False)

    # Effective rank (participation ratio)
    s_norm = s / (jnp.sum(s) + 1e-8)
    effective_rank = 1.0 / jnp.sum(s_norm ** 2)

    # Top singular values
    top_sv = s[:min(10, len(s))]

    return {
        'effective_rank': float(effective_rank),
        'top_singular_values': [float(v) for v in top_sv],
        'singular_value_sum': float(jnp.sum(s)),
        'condition_number': float(s[0] / (s[-1] + 1e-8)),
    }


def inspect_lowrank_structure(
    params: RNNParams,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Inspect and report on the low-rank structure of the network.

    Args:
        params: Network parameters
        output_path: Path to save report
        verbose: Whether to print to stdout

    Returns:
        Dictionary with all computed metrics
    """
    results = {}

    # Basic info
    N = params.M.shape[0]
    R = params.M.shape[1]
    d_in = params.B.shape[1]

    results['network_size'] = N
    results['rank'] = R
    results['input_dim'] = d_in

    # Parameter norms
    results['norms'] = {
        'M': float(jnp.linalg.norm(params.M)),
        'N_lr': float(jnp.linalg.norm(params.N_lr)),
        'B': float(jnp.linalg.norm(params.B)),
        'w': float(jnp.linalg.norm(params.w)),
        'b': float(jnp.abs(params.b)),
    }

    # Overlaps
    results['overlaps'] = compute_overlaps(params)

    # Spectral properties
    results['spectral'] = compute_effective_rank(params)

    # Column norms
    results['column_norms'] = {
        'M': [float(jnp.linalg.norm(params.M[:, i])) for i in range(R)],
        'N_lr': [float(jnp.linalg.norm(params.N_lr[:, i])) for i in range(R)],
        'B': [float(jnp.linalg.norm(params.B[:, k])) for k in range(d_in)],
    }

    # Output
    if verbose:
        print("\n" + "=" * 60)
        print("LOW-RANK NETWORK STRUCTURE ANALYSIS")
        print("=" * 60)

        print(f"\nNetwork: N={N}, R={R}, d_in={d_in}")

        print("\nParameter Norms:")
        for name, norm in results['norms'].items():
            print(f"  {name}: {norm:.4f}")

        print("\nKey Overlaps:")
        key_overlaps = ['M0_N0', 'M0_w', 'N0_w', 'M0_s1', 'M0_ctx1', 'N0_s1', 'N0_ctx1']
        if R > 1:
            key_overlaps += ['M1_N1', 'M1_w', 'N1_w', 'M1_s2', 'M1_ctx2', 'N1_s2', 'N1_ctx2']

        for key in key_overlaps:
            if key in results['overlaps']:
                print(f"  {key}: {results['overlaps'][key]:.4f}")

        print("\nSpectral Properties:")
        print(f"  Effective rank: {results['spectral']['effective_rank']:.2f}")
        print(f"  Top singular values: {results['spectral']['top_singular_values'][:5]}")

        print("\nColumn Norms (M):", [f"{n:.3f}" for n in results['column_norms']['M']])
        print("Column Norms (N):", [f"{n:.3f}" for n in results['column_norms']['N_lr']])

        print("=" * 60)

    # Save to file
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved analysis to {output_path}")

    return results


if __name__ == '__main__':
    import argparse
    import pickle
    import jax

    parser = argparse.ArgumentParser()
    parser.add_argument('--params_file', type=str, required=True,
                        help='Path to params.pkl')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for JSON report')
    args = parser.parse_args()

    # Load params
    with open(args.params_file, 'rb') as f:
        params_dict = pickle.load(f)

    params = RNNParams(
        C=jnp.array(params_dict['C']),
        M=jnp.array(params_dict['M']),
        N_lr=jnp.array(params_dict['N_lr']),
        B=jnp.array(params_dict['B']),
        w=jnp.array(params_dict['w']),
        b=jnp.array(params_dict['b']),
    )

    # Run analysis
    inspect_lowrank_structure(params, args.output)
