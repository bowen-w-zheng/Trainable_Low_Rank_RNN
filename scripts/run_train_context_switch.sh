#!/bin/bash
# Train low-rank RNN on contextual switch task

set -e
# Kill all current hanging processes to free up GPU memory
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    owner=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')
    if [ "$owner" = "$USER" ]; then
        echo "Force killing PID $pid (owner $owner)"
        kill -9 "$pid"
    fi
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Default config
CONFIG="${1:-configs/contextual_switch_full_rank.yaml}"
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
