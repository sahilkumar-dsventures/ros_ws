"""
Isaac Sim OmniGraph IK Script for Franka Panda.

This script runs as an OmniGraph Python node to:
1. Receive EE delta commands from ROS 2 (/franka/ee_delta_command)
2. Get current EE pose from robot
3. Compute target pose = current + delta
4. Solve IK using Lula solver
5. Output joint positions for ArticulationController

Setup in OmniGraph:
1. Create a Python Script node
2. Set this file as the script path
3. Add inputs: robot_path (string), linear_x/y/z (float), angular_x/y/z (float)
4. Add output: joint_positions (float[])
5. Connect to ROS 2 Twist subscriber for EE delta
"""

import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni.graph.core as og

# Global cache
_solver = None
_robot = None
_ee_frame = "panda_hand"  # Franka's end-effector frame name


def setup(db: og.Database):
    """Initialize Franka robot and Lula IK solver."""
    global _solver, _robot
    
    robot_prim_path = db.inputs.robot_path
    
    # Initialize robot articulation
    if _robot is None:
        _robot = Articulation(robot_prim_path)
        _robot.initialize()
        db.log_info(f"Initialized Franka robot at: {robot_prim_path}")
    
    # Initialize Lula IK Solver with built-in Franka config
    if _solver is None:
        _solver = LulaKinematicsSolver(
            robot_description_path="",  # Uses default Franka description
            urdf_path="",               # Uses built-in URDF
            robot_name="Franka"         # Built-in configuration
        )
        db.log_info("Initialized Lula IK solver for Franka")


def cleanup(db: og.Database):
    """Clean up on node removal."""
    global _solver, _robot
    _solver = None
    _robot = None


def compute(db: og.Database):
    """
    Compute IK from EE delta to joint positions.
    
    Inputs (from ROS 2 Twist via OmniGraph):
        - linear_x, linear_y, linear_z: EE position delta (meters)
        - angular_x, angular_y, angular_z: EE rotation delta (roll, pitch, yaw radians)
        - gripper: Gripper command (-1 to +1)
    
    Outputs:
        - joint_positions: 7-element array for Franka arm joints
        - gripper_width: Mapped gripper position
    """
    global _solver, _robot
    
    if _solver is None or _robot is None:
        db.log_warning("Solver or robot not initialized")
        return False
    
    # 1. Get current EE pose from solver
    try:
        ee_pose = _solver.compute_end_effector_pose()
        current_pos = np.array(ee_pose[:3])
        current_quat = ee_pose[3:]  # Quaternion (w, x, y, z)
    except Exception as e:
        db.log_error(f"Failed to get EE pose: {e}")
        return False
    
    # 2. Get delta from inputs
    delta_pos = np.array([
        db.inputs.linear_x,
        db.inputs.linear_y,
        db.inputs.linear_z
    ])
    
    delta_rpy = np.array([
        db.inputs.angular_x,   # roll
        db.inputs.angular_y,   # pitch
        db.inputs.angular_z    # yaw
    ])
    
    # Skip if no movement
    if np.allclose(delta_pos, 0) and np.allclose(delta_rpy, 0):
        return True
    
    # 3. Compute target pose
    target_pos = current_pos + delta_pos
    
    # Convert current quaternion to euler, add delta, convert back
    current_rpy = quat_to_euler_angles(current_quat)
    target_rpy = current_rpy + delta_rpy
    target_quat = euler_angles_to_quat(target_rpy)  # Returns (w, x, y, z)
    
    # 4. Solve IK
    action, success = _solver.compute_inverse_kinematics(
        target_position=target_pos,
        target_orientation=target_quat
    )
    
    if success:
        # Output joint positions (7 DOF for Franka arm)
        db.outputs.joint_positions = action.joint_positions.tolist()
        
        # Handle gripper separately
        gripper_cmd = db.inputs.gripper if hasattr(db.inputs, 'gripper') else 0.0
        # Map [-1, +1] to Franka gripper width [0, 0.04] meters
        gripper_width = 0.02 + (gripper_cmd * 0.02)  # Center at 0.02, range 0-0.04
        db.outputs.gripper_width = max(0.0, min(0.04, gripper_width))
        
        return True
    else:
        db.log_warning("IK solution failed - target may be unreachable")
        return False
