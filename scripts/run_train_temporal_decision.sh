#!/bin/bash
# Train the temporal decision task (Interpolating Go-No-Go)

# Default values
CONFIG="${1:-configs/temporal_decision_default.yaml}"
OUTPUT="${2:-results}"

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))"

echo "Training temporal decision task"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"
echo ""

python -m src.training.train_temporal_decision \
    --config "$CONFIG" \
    --output "$OUTPUT"
