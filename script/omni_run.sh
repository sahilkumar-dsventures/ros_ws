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
    echo "⚠️  No script path provided. Usage: ./omni_run.sh <script_path>"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "❌ Script not found: $1"
    exit 1
fi

script_path=$(realpath "$1")

# Navigate to Isaac Sim's directory
cd /media/sarthak/a/isaac_sim/ || {
    echo "❌ Failed to change directory to Isaac Sim's directory."
    exit 1
}

# Setup Python environment
if ! source setup_python_env.sh; then
    echo "❌ Failed to set up Python environment."
    exit 1
fi

# ============================================
# ROS 2 Bridge Configuration for Isaac Sim
# ============================================
# These environment variables are REQUIRED for Isaac Sim's 
# internal ROS 2 bridge to work properly.

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# Use the path relative to Isaac Sim directory as shown in the error message
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/exts/isaacsim.ros2.bridge/humble/lib"

echo "============================================"
echo "🤖 Isaac Sim ROS 2 Bridge Configuration"
echo "============================================"
echo "ROS_DISTRO=$ROS_DISTRO"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "LD_LIBRARY_PATH includes: $(pwd)/exts/isaacsim.ros2.bridge/humble/lib"
echo "============================================"

echo ""
echo "Running script: $script_path 🚀"
echo ""

./python.sh "$script_path"