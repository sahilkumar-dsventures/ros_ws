import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni.graph.core as og

# Global cache for solver
_solver = None
_robot = None

def setup(db: og.Database):
    global _solver, _robot
    
    robot_prim_path = db.inputs.robot_path
    
    # Initialize robot articulation if needed
    if _robot is None:
        _robot = Articulation(robot_prim_path)
        _robot.initialize()
    
    # Initialize Lula IK Solver for Franka
    if _solver is None:
        _solver = LulaKinematicsSolver(
            robot_description_path="",  # Auto-load default if empty
            urdf_path="",              # Uses default
            robot_name="Franka"        # Built-in config
        )

def cleanup(db: og.Database):
    global _solver, _robot
    _solver = None
    _robot = None

def compute(db: og.Database):
    global _solver, _robot
    
    if _solver is None or _robot is None:
        return False
        
    # 1. Get current EE pose
    # Note: 'panda_hand' is the default EE frame name for Franka
    ee_pose = _solver.compute_end_effector_pose()
    current_pos = ee_pose[:3]
    current_rot = ee_pose[3:]  # Quaternion (w, x, y, z)
    
    # 2. Apply Deltas
    # Twist message comes as (linear_velocity, angular_velocity)
    # Since we run at 60Hz loop, simple integration: new_pos = old + delta
    
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
    
    if np.allclose(delta_pos, 0) and np.allclose(delta_rpy, 0):
        # Optimization: no movement needed
        return True

    target_pos = current_pos + delta_pos
    
    # Rotation logic
    current_rpy = quat_to_euler_angles(current_rot)
    target_rpy = current_rpy + delta_rpy
    target_rot = euler_angles_to_quat(target_rpy) # Returns (w, x, y, z)
    
    # 3. Solve IK
    action, success = _solver.compute_inverse_kinematics(
        target_position=target_pos,
        target_orientation=target_rot
    )
    
    if success:
        # Output the joint positions
        db.outputs.joint_positions = action.joint_positions
        return True
    else:
        db.log_warning("IK Solution Failed")
        return False