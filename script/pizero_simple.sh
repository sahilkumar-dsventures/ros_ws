#!/bin/bash

source /opt/ros/humble/setup.bash
source install/setup.sh

# PREPEND the venv path so it takes priority
export PYTHONPATH=$(pwd)/venv/lib/python3.10/site-packages:$PYTHONPATH

ros2 run pizero_simple pizero