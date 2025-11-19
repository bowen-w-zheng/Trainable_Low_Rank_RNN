#!/bin/bash
# Train the temporal decision task (Interpolating Go-No-Go)
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    owner=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')
    if [ "$owner" = "$USER" ]; then
        echo "Force killing PID $pid (owner $owner)"
        kill -9 "$pid"
    fi
done

# Default values
CONFIG="${1:-configs/temporal_decision_hard.yaml}"
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
