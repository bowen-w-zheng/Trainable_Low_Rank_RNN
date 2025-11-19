#!/bin/bash
# Train low-rank RNN on contextual switch task

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Default config
CONFIG="${1:-configs/contextual_switch_default.yaml}"
OUTPUT="${2:-results}"

echo "Training low-rank RNN on contextual switch task"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"
echo "----------------------------------------"

# Run training
python -m src.training.train_context_switch \
    --config "$CONFIG" \
    --output "$OUTPUT"

echo "----------------------------------------"
echo "Training complete!"
