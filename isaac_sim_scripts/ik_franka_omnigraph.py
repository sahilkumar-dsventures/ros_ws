"""
Isaac Sim OmniGraph IK Script for Franka Panda

Uses LulaKinematicsSolver directly with built-in Franka config.
Avoids ArticulationKinematicsSolver which has initialization timing issues.

Setup in OmniGraph:
1. Create a Python Script node
2. Set this file as the script path
3. Add inputs: 
   - linear (vector3d) - connect from ROS2 Subscribe Twist linearVelocity
   - angular (vector3d) - connect from ROS2 Subscribe Twist angularVelocity
4. Add outputs: joint_positions (double[9] - 7 arm + 2 fingers)
"""

import numpy as np
from isaacsim.core.prims import SingleArticulation
from omni.isaac.motion_generation import LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni.graph.core as og
import omni.timeline

# Global cache
_lula_solver = None
_robot = None
_physics_ready = False
_init_frame_count = 0

# Franka Configuration
_ROBOT_PRIM_PATH = "/franka"  # Change to match your USD scene
_EE_FRAME = "panda_hand"


def setup(db: og.Database):
    """Initialize Franka robot with Lula IK solver."""
    global _lula_solver, _robot, _physics_ready, _init_frame_count
    
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        _physics_ready = False
        _init_frame_count = 0
        return
    
    _init_frame_count += 1
    if _init_frame_count < 30:  # Wait 30 frames for physics to fully stabilize
        return
    
    # Initialize robot articulation
    if _robot is None:
        try:
            _robot = SingleArticulation(_ROBOT_PRIM_PATH)
            _robot.initialize()
            
            # Check if physics view is created before accessing joint positions
            if not hasattr(_robot, '_articulation_view') or _robot._articulation_view is None:
                _robot = None
                _init_frame_count = 0
                return
            
            test_joints = _robot.get_joint_positions()
            if test_joints is None:
                _robot = None
                _init_frame_count = 0
                return
            db.log_info(f"Initialized Franka robot at: {_ROBOT_PRIM_PATH}")
        except Exception as e:
            _robot = None
            _init_frame_count = 0
            return
    
    # Initialize LulaKinematicsSolver (directly, not wrapped)
    if _lula_solver is None and _robot is not None:
        try:
            franka_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
            _lula_solver = LulaKinematicsSolver(**franka_config)
            _physics_ready = True
            db.log_info("Initialized LulaKinematicsSolver for Franka (built-in config)")
        except Exception as e:
            db.log_warning(f"Lula solver init failed: {e}, using fallback")
            _lula_solver = "fallback"
            _physics_ready = True


def cleanup(db: og.Database):
    global _lula_solver, _robot, _physics_ready, _init_frame_count
    _lula_solver = None
    _robot = None
    _physics_ready = False
    _init_frame_count = 0


def compute(db: og.Database):
    """Compute IK from EE delta to joint positions."""
    global _lula_solver, _robot, _physics_ready
    
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        return True
    
    if not _physics_ready or _robot is None:
        setup(db)
        if not _physics_ready or _robot is None:
            return True
    
    # Get inputs
    linear = db.inputs.linear
    angular = db.inputs.angular
    
    if linear is None or angular is None:
        return True
    
    try:
        delta_pos = np.array([linear[0], linear[1], linear[2]])
        delta_rpy = np.array([angular[0], angular[1], angular[2]])
    except (TypeError, IndexError):
        return True
    
    # Clamp deltas to prevent extreme movements
    MAX_POS = 0.03  # 3cm max per frame
    MAX_ROT = 0.05  # 0.05 rad max per frame
    delta_pos = np.clip(delta_pos, -MAX_POS, MAX_POS)
    delta_rpy = np.clip(delta_rpy, -MAX_ROT, MAX_ROT)
    
    if np.allclose(delta_pos, 0) and np.allclose(delta_rpy, 0):
        return True
    
    try:
        # Get current joint positions
        current_q = _robot.get_joint_positions()
        if current_q is None:
            return True
        current_q = np.array(current_q[:7])  # 7 arm joints
        
        if _lula_solver is not None and _lula_solver != "fallback":
            # Use Lula for FK and IK
            ee_pos, ee_rot_matrix = _lula_solver.compute_forward_kinematics(
                frame_name=_EE_FRAME,
                joint_positions=current_q
            )
            
            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation
            current_quat_xyzw = Rotation.from_matrix(ee_rot_matrix).as_quat()
            current_quat = np.array([current_quat_xyzw[3], current_quat_xyzw[0], 
                                     current_quat_xyzw[1], current_quat_xyzw[2]])
            
            # Compute target pose
            target_pos = np.array(ee_pos) + delta_pos
            current_rpy = quat_to_euler_angles(current_quat)
            target_rpy = current_rpy + delta_rpy
            target_quat = euler_angles_to_quat(target_rpy)
            
            # Solve IK
            ik_result, success = _lula_solver.compute_inverse_kinematics(
                frame_name=_EE_FRAME,
                target_position=target_pos,
                target_orientation=target_quat
            )
            
            if success:
                arm_joints = ik_result
            else:
                return True  # IK failed, skip this frame
        else:
            return True  # No solver, skip
        
        # Gripper handling
        gripper_cmd = db.inputs.gripper if hasattr(db.inputs, 'gripper') else 0.0
        if gripper_cmd is None:
            gripper_cmd = 0.0
        finger_pos = 0.02 + 0.02 * (gripper_cmd + 1.0) / 2.0
        finger_pos = max(0.0, min(0.04, finger_pos))
        
        # Output 9 joints (7 arm + 2 fingers)
        all_joints = list(arm_joints) + [finger_pos, finger_pos]
        db.outputs.joint_positions = all_joints
        
        return True
        
    except Exception as e:
        return True  # Silently skip on error