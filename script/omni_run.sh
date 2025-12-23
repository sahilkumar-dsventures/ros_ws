#!/bin/bash

# Save current working directory
CWD=$(pwd)

restore() {
    cd "$CWD" || {
        echo "Failed to restore original working directory."
        exit 1
    }
}

trap restore EXIT

# Validate argument
if [ -z "$1" ]; then
    echo "⚠️  No script path provided. Usage: ./run_in_kit.sh <script_path>"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "❌ Script not found: $1"
    exit 1
fi

script_path=$(realpath "$1")

# Navigate to Isaac Sim's Kit directory
cd /media/sarthak/a/isaac_sim/ || {
    echo "❌ Failed to change directory to Isaac Sim Kit."
    exit 1
}

# Setup Python environment
if ! source setup_python_env.sh; then
    echo "❌ Failed to set up Python environment."
    exit 1
fi

echo "Running script: $script_path 🚀"

# Optional GPU lib path fix
GPU_LIB_PATH="/media/sarthak/a/isaac_sim/extscache/omni.gpu_foundation-0.0.0+69cbf6ad.lx64.r.cp311/bin/deps"
export LD_LIBRARY_PATH="$GPU_LIB_PATH:$LD_LIBRARY_PATH"

./python.sh "$script_path"