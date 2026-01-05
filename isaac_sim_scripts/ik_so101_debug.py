"""
Simplified Isaac Sim OmniGraph Script for SO101 Arm - DEBUG VERSION

This minimal script helps debug OmniGraph Python Script node issues.
It logs every step to help identify where problems occur.

Setup in OmniGraph:
1. Create a Python Script node
2. Set this file as the script path
3. Add inputs: linear (vector3d), angular (vector3d)
4. Add outputs: joint_positions (double[5])
"""

import numpy as np
import omni.graph.core as og
import omni.timeline

# Global state
_robot = None
_initialized = False
_frame_count = 0

# Configuration - CHANGE THIS TO YOUR ROBOT PATH
_ROBOT_PRIM_PATH = "/so101_new_calib/base"


def setup(db: og.Database):
    """Called once when node is created."""
    db.log_info("=== SO101 IK Script: setup() called ===")
    return True


def cleanup(db: og.Database):
    """Called when node is destroyed."""
    global _robot, _initialized, _frame_count
    db.log_info("=== SO101 IK Script: cleanup() called ===")
    _robot = None
    _initialized = False
    _frame_count = 0


def compute(db: og.Database):
    """Main compute function - called every frame."""
    global _robot, _initialized, _frame_count
    
    _frame_count += 1
    
    # Check if simulation is playing
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        return True
    
    # Log every 60 frames (about once per second at 60fps)
    should_log = (_frame_count % 60 == 0)
    
    # Try to initialize robot if not done
    if not _initialized:
        if should_log:
            db.log_info(f"Frame {_frame_count}: Attempting to initialize robot at {_ROBOT_PRIM_PATH}")
        
        try:
            # Use new Isaac Sim 5.1.0 API
            from isaacsim.core.prims import SingleArticulation
            _robot = SingleArticulation(_ROBOT_PRIM_PATH)
            _robot.initialize()
            
            # Test if physics is ready
            joints = _robot.get_joint_positions()
            if joints is not None and len(joints) > 0:
                _initialized = True
                db.log_info(f"SUCCESS: Robot initialized! Found {len(joints)} joints: {joints}")
            else:
                _robot = None
                if should_log:
                    db.log_info("Waiting for physics... joint positions not available yet")
        except Exception as e:
            _robot = None
            if should_log:
                db.log_warning(f"Init attempt failed: {e}")
        
        return True
    
    # Get inputs
    try:
        linear = db.inputs.linear
        angular = db.inputs.angular
        
        if linear is None or angular is None:
            if should_log:
                db.log_info("No ROS2 input yet (linear/angular are None)")
            return True
        
        # Extract values
        delta_pos = np.array([linear[0], linear[1], linear[2]])
        delta_rpy = np.array([angular[0], angular[1], angular[2]])
        
        # Skip if no movement
        if np.allclose(delta_pos, 0) and np.allclose(delta_rpy, 0):
            return True
        
        if should_log:
            db.log_info(f"Received: pos={delta_pos}, rpy={delta_rpy}")
        
        # Get current joint positions
        current_joints = np.array(_robot.get_joint_positions()[:5])
        
        # Simple mapping (for testing)
        scale = 0.5
        joint_delta = np.array([
            delta_pos[1] * scale,   # Rotation
            -delta_pos[2] * scale,  # Pitch
            delta_pos[0] * scale,   # Elbow
            delta_rpy[1] * scale,   # Wrist pitch
            delta_rpy[0] * scale    # Wrist roll
        ])
        
        target_joints = current_joints + joint_delta
        
        # Output
        db.outputs.joint_positions = target_joints.tolist()
        
        if should_log:
            db.log_info(f"Output joints: {target_joints}")
        
    except Exception as e:
        db.log_error(f"Compute error: {e}")
        return False
    
    return True
