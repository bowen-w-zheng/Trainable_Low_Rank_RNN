#!/bin/bash
# Quick sanity check training with small config

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "Running sanity check with small configuration..."
echo "----------------------------------------"

# Run training with sanity config
python -m src.training.train_context_switch \
    --config "configs/contextual_switch_sanity.yaml" \
    --output "results/sanity_check"

echo "----------------------------------------"
echo "Sanity check complete!"
