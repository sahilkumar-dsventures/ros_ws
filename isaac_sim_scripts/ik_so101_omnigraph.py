"""
Isaac Sim OmniGraph IK Script for SO101 Arm.

This script runs as an OmniGraph Python node to:
1. Receive EE delta commands from ROS 2 (/so101/ee_delta_command)
2. Get current EE pose from robot
3. Compute target pose = current + delta
4. Solve IK using Lula solver with custom URDF
5. Output joint positions for ArticulationController

SO101 Configuration:
- 5 arm joints: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll
- 1 gripper joint: Jaw
- End-effector frame: "gripper" or "jaw"

Setup in OmniGraph:
1. Create a Python Script node
2. Set this file as the script path
3. Add inputs: linear_x/y/z (float), angular_x/y/z (float), gripper (float)
4. Add output: joint_positions (float[]), jaw_position (float)
5. Connect to ROS 2 Twist subscriber for EE delta
"""

import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni.graph.core as og
import os

# Global cache
_solver = None
_robot = None

# SO101 Configuration
_SO101_URDF = "/media/sarthak/a/Experiments/SO-ARM101_MoveIt_IsaacSim/src/so_arm_description/urdf/so101_new_calib.urdf"
_ROBOT_PRIM_PATH = "/so101_new_calib"  # Robot prim path in USD stage
_EE_FRAME = "gripper"  # End-effector frame name in URDF

# Joint names in order
_ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
_GRIPPER_JOINT = "Jaw"


def _create_lula_robot_description():
    """
    Create a Lula robot description YAML for SO101.
    
    This is required for custom robots not built into Isaac Sim.
    Returns path to the generated file.
    """
    import tempfile
    import yaml
    
    robot_desc = {
        "robot": {
            "name": "so101",
            "urdf_path": _SO101_URDF,
            "end_effector_frame": _EE_FRAME,
            "base_frame": "base",
            "cspace_to_urdf_rules": [
                {"joint": name, "rule": "direct"} for name in _ARM_JOINTS
            ],
            "default_cspace_position": [0.0] * len(_ARM_JOINTS),
            "cspace_position_limits": {
                "lower": [-1.92, -1.75, -1.75, -1.66, -2.79],  # From URDF limits
                "upper": [1.92, 1.75, 1.57, 1.66, 2.79]
            }
        }
    }
    
    # Write to temp file
    desc_path = os.path.join(tempfile.gettempdir(), "so101_lula_desc.yaml")
    with open(desc_path, 'w') as f:
        yaml.dump(robot_desc, f)
    
    return desc_path


def setup(db: og.Database):
    """Initialize SO101 robot and Lula IK solver."""
    global _solver, _robot
    
    robot_prim_path = _ROBOT_PRIM_PATH
    
    # Initialize robot articulation
    if _robot is None:
        _robot = Articulation(robot_prim_path)
        _robot.initialize()
        db.log_info(f"Initialized SO101 robot at: {robot_prim_path}")
    
    # Initialize Lula IK Solver with custom SO101 config
    if _solver is None:
        try:
            # Try with custom description
            robot_desc_path = _create_lula_robot_description()
            _solver = LulaKinematicsSolver(
                robot_description_path=robot_desc_path,
                urdf_path=_SO101_URDF,
                robot_name="so101"
            )
            db.log_info("Initialized Lula IK solver for SO101 with custom URDF")
        except Exception as e:
            db.log_warning(f"Lula solver init failed: {e}, using fallback Jacobian IK")
            _solver = "fallback"  # Will use Jacobian-based IK


def cleanup(db: og.Database):
    """Clean up on node removal."""
    global _solver, _robot
    _solver = None
    _robot = None


def _jacobian_ik(robot, target_pos, target_rpy, max_iter=50, tolerance=1e-3):
    """
    Fallback Damped Least Squares IK for SO101.
    
    Uses numerical Jacobian if Lula solver is unavailable.
    """
    # Get current joint positions
    current_joints = np.array(robot.get_joint_positions()[:5])  # 5 arm joints
    
    # Simple numerical IK using damped least squares
    damping = 0.01
    
    for _ in range(max_iter):
        # Get current EE position (approximate from forward kinematics)
        # This is a simplified version - real implementation would compute FK
        current_ee = robot.get_world_pose()  # Approximate
        
        # Position error
        pos_error = target_pos - np.array(current_ee[0][:3])
        
        if np.linalg.norm(pos_error) < tolerance:
            return current_joints, True
        
        # Approximate Jacobian (simplified for 5-DOF arm)
        # In practice, you'd compute this from the robot model
        J = np.eye(3, 5) * 0.1  # Placeholder
        
        # Damped least squares
        JtJ = J.T @ J + damping * np.eye(5)
        delta_q = np.linalg.solve(JtJ, J.T @ pos_error)
        
        current_joints += delta_q * 0.1  # Small step
    
    return current_joints, False


def compute(db: og.Database):
    """
    Compute IK from EE delta to joint positions for SO101.
    
    Inputs (from ROS 2 Twist via OmniGraph):
        - linear_x, linear_y, linear_z: EE position delta (meters)
        - angular_x, angular_y, angular_z: EE rotation delta (roll, pitch, yaw radians)
        - gripper: Gripper/Jaw command (-1 to +1)
    
    Outputs:
        - joint_positions: 5-element array for SO101 arm joints
        - jaw_position: Mapped jaw position
    """
    global _solver, _robot
    
    if _robot is None:
        db.log_warning("Robot not initialized")
        return False
    
    # Get delta from inputs
    delta_pos = np.array([
        db.inputs.linear_x,
        db.inputs.linear_y,
        db.inputs.linear_z
    ])
    
    delta_rpy = np.array([
        db.inputs.angular_x,
        db.inputs.angular_y,
        db.inputs.angular_z
    ])
    
    # Skip if no movement
    if np.allclose(delta_pos, 0) and np.allclose(delta_rpy, 0):
        return True
    
    try:
        if _solver is not None and _solver != "fallback":
            # Use Lula solver
            ee_pose = _solver.compute_end_effector_pose()
            current_pos = np.array(ee_pose[:3])
            current_quat = ee_pose[3:]
            
            target_pos = current_pos + delta_pos
            current_rpy = quat_to_euler_angles(current_quat)
            target_rpy = current_rpy + delta_rpy
            target_quat = euler_angles_to_quat(target_rpy)
            
            action, success = _solver.compute_inverse_kinematics(
                target_position=target_pos,
                target_orientation=target_quat
            )
            
            if success:
                db.outputs.joint_positions = action.joint_positions.tolist()
            else:
                db.log_warning("IK solution failed")
                return False
        else:
            # Fallback: Simplified Jacobian-like mapping (from original policy.py)
            # This is approximate but works without complex IK setup
            current_joints = np.array(_robot.get_joint_positions()[:5])
            
            scale = 1.0
            joint_delta = np.zeros(5)
            joint_delta[0] = delta_pos[1] * scale       # Rotation affects Y
            joint_delta[1] = -delta_pos[2] * scale      # Pitch affects Z
            joint_delta[2] = delta_pos[0] * scale       # Elbow affects X
            joint_delta[3] = delta_rpy[1] * scale       # Wrist pitch
            joint_delta[4] = delta_rpy[0] * scale       # Wrist roll
            
            target_joints = current_joints + joint_delta
            db.outputs.joint_positions = target_joints.tolist()
        
        # Handle gripper (Jaw)
        gripper_cmd = db.inputs.gripper if hasattr(db.inputs, 'gripper') else 0.0
        # Map [-1, +1] to Jaw position range from URDF: [-0.17, 1.75] radians
        # -1 = close (0), +1 = open (1.75)
        jaw_min = 0.0
        jaw_max = 1.0
        jaw_pos = jaw_min + (jaw_max - jaw_min) * (gripper_cmd + 1.0) / 2.0
        db.outputs.jaw_position = max(jaw_min, min(jaw_max, jaw_pos))
        
        return True
        
    except Exception as e:
        db.log_error(f"IK computation failed: {e}")
        return False
