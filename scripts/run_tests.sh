#!/bin/bash
# Run all unit tests

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "Running unit tests..."
echo "----------------------------------------"

# Run pytest
pytest tests/ -v --tb=short "$@"

echo "----------------------------------------"
echo "Tests complete!"
