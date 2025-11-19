#!/bin/bash
# Evaluate and analyze trained low-rank RNN

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Results directory (required argument)
if [ -z "$1" ]; then
    echo "Usage: $0 <results_dir>"
    echo "Example: $0 results/contextual_switch_20240101_120000"
    exit 1
fi

RESULTS_DIR="$1"
OUTPUT_DIR="${2:-$RESULTS_DIR}"

echo "Evaluating low-rank RNN"
echo "Results: $RESULTS_DIR"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------"

# Run trajectory plotting
echo "Plotting trajectories..."
python -m src.analysis.plot_trajectories \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR"

# Run structure inspection
echo "Inspecting low-rank structure..."
python -m src.analysis.inspect_lowrank \
    --params_file "$RESULTS_DIR/params.pkl" \
    --output "$OUTPUT_DIR/structure_analysis.json"

echo "----------------------------------------"
echo "Analysis complete!"
echo "Plots saved to $OUTPUT_DIR"
