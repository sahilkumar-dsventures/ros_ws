"""
Isaac Sim OmniGraph IK Script for Franka Panda.

This script runs as an OmniGraph Python node to:
1. Receive EE delta commands from ROS 2 via Twist message
2. Get current EE pose from robot using forward kinematics
3. Compute target pose = current + delta
4. Solve IK using Lula solver (or fallback)
5. Output joint positions for ArticulationController

Franka Panda Configuration:
- 7 arm joints: panda_joint1-7
- 2 finger joints: panda_finger_joint1-2
- End-effector frame: "panda_hand"

Setup in OmniGraph:
1. Create a Python Script node
2. Set this file as the script path
3. Add inputs: 
   - linear (vector3d) - connect from ROS2 Subscribe Twist linearVelocity output
   - angular (vector3d) - connect from ROS2 Subscribe Twist angularVelocity output
   - gripper (double) - optional, default 0.0
4. Add outputs: joint_positions (double[9] - 7 arm + 2 fingers)
5. Connect ROS2 Subscribe Twist node to this Python Script node
"""

import numpy as np
from isaacsim.core.prims import SingleArticulation
from omni.isaac.motion_generation import LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni.graph.core as og
import omni.timeline

# Global cache
_solver = None
_robot = None
_physics_ready = False
_init_frame_count = 0

# Franka Panda Configuration
_ROBOT_PRIM_PATH = "/franka"  # Change this to match your USD scene
_EE_FRAME = "panda_hand"  # End-effector frame name

# Joint names in order
_ARM_JOINTS = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", 
               "panda_joint5", "panda_joint6", "panda_joint7"]
_FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


def setup(db: og.Database):
    """Initialize Franka robot and Lula IK solver."""
    global _solver, _robot, _physics_ready, _init_frame_count
    
    # Check if simulation is playing
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        _physics_ready = False
        _init_frame_count = 0
        return  # Don't initialize until simulation starts
    
    # Wait a few frames for physics to fully initialize
    _init_frame_count += 1
    if _init_frame_count < 10:  # Wait ~10 frames
        return
    
    robot_prim_path = _ROBOT_PRIM_PATH
    
    # Initialize robot articulation
    if _robot is None:
        try:
            _robot = SingleArticulation(robot_prim_path)
            _robot.initialize()
            # Test if we can actually read joint positions
            test_joints = _robot.get_joint_positions()
            if test_joints is None:
                _robot = None
                return  # Physics not ready yet
            _physics_ready = True
            db.log_info(f"Initialized Franka robot at: {robot_prim_path}")
        except Exception as e:
            _robot = None
            _physics_ready = False
            return  # Silently retry next frame
    
    # Initialize Lula IK Solver using built-in Franka config
    if _solver is None:
        try:
            # Load built-in Franka kinematics config (no URDF path needed!)
            franka_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
            _solver = LulaKinematicsSolver(**franka_config)
            db.log_info("Initialized Lula IK solver for Franka Panda (built-in config)")
        except Exception as e:
            db.log_warning(f"Lula solver init failed: {e}, using fallback Jacobian IK")
            _solver = "fallback"  # Will use Jacobian-based IK


def cleanup(db: og.Database):
    """Clean up on node removal."""
    global _solver, _robot, _physics_ready, _init_frame_count
    _solver = None
    _robot = None
    _physics_ready = False
    _init_frame_count = 0


def compute(db: og.Database):
    """
    Compute IK from EE delta to joint positions for Franka Panda.
    
    Inputs (from ROS 2 Twist via OmniGraph - connect directly from ROS2 Subscribe Twist):
        - linear: vector3d (x, y, z) - EE position delta in meters
        - angular: vector3d (roll, pitch, GRIPPER) - roll/pitch rotation, gripper in z
    
    Outputs:
        - joint_positions: 9-element double array (7 arm + 2 fingers)
    """
    global _solver, _robot, _physics_ready
    
    # Check if simulation is playing
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        return True  # Silently skip if not playing
    
    # Check if physics is ready and robot is initialized
    if not _physics_ready or _robot is None:
        # Try to initialize now
        setup(db)
        if not _physics_ready or _robot is None:
            return True  # Still not ready, skip silently
    
    # Get delta from vector3 inputs (directly from ROS2 Subscribe Twist node)
    # linear = (x, y, z) position delta
    # angular = (roll, pitch, yaw) rotation delta
    linear = db.inputs.linear
    angular = db.inputs.angular
    
    # Check if inputs are valid (not None and have data)
    if linear is None or angular is None:
        return True  # No ROS2 message received yet, skip silently
    
    try:
        delta_pos = np.array([linear[0], linear[1], linear[2]])
        delta_rpy = np.array([angular[0], angular[1], angular[2]])  # Full rotation
    except (TypeError, IndexError):
        return True  # Invalid input format, skip silently
    
    # Clamp deltas to prevent extreme movements (policy sometimes outputs very large values)
    MAX_POS_DELTA = 0.05    # Max 5cm per frame
    MAX_ROT_DELTA = 0.1     # Max 0.1 radians per frame
    delta_pos = np.clip(delta_pos, -MAX_POS_DELTA, MAX_POS_DELTA)
    delta_rpy = np.clip(delta_rpy, -MAX_ROT_DELTA, MAX_ROT_DELTA)
    
    # Skip if no movement
    if np.allclose(delta_pos, 0) and np.allclose(delta_rpy, 0):
        return True
    
    try:
        target_joints = None
        use_fallback = (_solver is None or _solver == "fallback")
        
        if not use_fallback:
            # Try Lula solver
            try:
                # Get current joint positions from robot
                current_q = _robot.get_joint_positions()
                if current_q is None:
                    _physics_ready = False
                    return True
                current_q = np.array(current_q[:7])  # 7 arm joints
                
                # Get current EE pose using forward kinematics
                # Returns (position, rotation_matrix) - rotation is 3x3 matrix
                ee_pos, ee_rot_matrix = _solver.compute_forward_kinematics(
                    frame_name=_EE_FRAME,
                    joint_positions=current_q
                )
                
                # Convert rotation matrix to quaternion using scipy
                from scipy.spatial.transform import Rotation
                current_quat = Rotation.from_matrix(ee_rot_matrix).as_quat()  # [x,y,z,w]
                # Convert to [w,x,y,z] format expected by Isaac Sim
                current_quat = np.array([current_quat[3], current_quat[0], current_quat[1], current_quat[2]])
                
                # Compute target EE pose
                target_pos = np.array(ee_pos) + delta_pos
                current_rpy = quat_to_euler_angles(current_quat)
                target_rpy = current_rpy + delta_rpy
                target_quat = euler_angles_to_quat(target_rpy)
                
                # Solve IK
                ik_result, success = _solver.compute_inverse_kinematics(
                    frame_name=_EE_FRAME,
                    target_position=target_pos,
                    target_orientation=target_quat
                )
                
                if success:
                    target_joints = np.array(ik_result)
                else:
                    use_fallback = True
            except Exception as e:
                db.log_warning(f"Lula IK error: {e}, using fallback")
                use_fallback = True
        
        # Fallback: Simplified Jacobian-like mapping
        if use_fallback or target_joints is None:
            try:
                joint_positions = _robot.get_joint_positions()
                if joint_positions is None:
                    _physics_ready = False
                    return True
                current_joints = np.array(joint_positions[:7])  # 7 arm joints
            except Exception as e:
                _physics_ready = False
                return True
            
            # Scale down the delta - different scales for different axes
            pos_scale = 0.5    # Position scale for Franka (larger workspace)
            rot_scale = 0.3    # Rotation scale
            
            # Approximate Jacobian mapping for Franka Panda
            # Franka joints: [j1, j2, j3, j4, j5, j6, j7]
            joint_delta = np.zeros(7)
            
            # Joint 1: Base rotation (affects Y movement)
            joint_delta[0] = delta_pos[1] * pos_scale
            
            # Joint 2: Shoulder pitch (affects Z movement)
            joint_delta[1] = -delta_pos[2] * pos_scale * 0.5
            
            # Joint 3: Shoulder roll (affects X movement)
            joint_delta[2] = delta_pos[0] * pos_scale * 0.3
            
            # Joint 4: Elbow (affects reach/Z)
            joint_delta[3] = delta_pos[2] * pos_scale * 0.5
            
            # Joint 5: Wrist pitch
            joint_delta[4] = delta_rpy[1] * rot_scale
            
            # Joint 6: Wrist yaw
            joint_delta[5] = delta_rpy[0] * rot_scale
            
            # Joint 7: Wrist roll
            joint_delta[6] = 0.0  # No direct mapping
            
            target_joints = current_joints + joint_delta
            
            # Clamp joints to Franka limits
            joint_limits = np.array([
                [-2.8973, 2.8973],   # j1
                [-1.7628, 1.7628],   # j2
                [-2.8973, 2.8973],   # j3
                [-3.0718, -0.0698],  # j4
                [-2.8973, 2.8973],   # j5
                [-0.0175, 3.7525],   # j6
                [-2.8973, 2.8973]    # j7
            ])
            for i in range(7):
                target_joints[i] = np.clip(target_joints[i], joint_limits[i, 0], joint_limits[i, 1])
        
        # Handle gripper (fingers) - from separate gripper input
        gripper_cmd = db.inputs.gripper if hasattr(db.inputs, 'gripper') else 0.0
        if gripper_cmd is None:
            gripper_cmd = 0.0
        # Map [-1, +1] to finger position range: [0, 0.04] meters
        # -1 = close (0), +1 = open (0.04)
        finger_min = 0.0
        finger_max = 0.04
        finger_pos = finger_min + (finger_max - finger_min) * (gripper_cmd + 1.0) / 2.0
        finger_pos = max(finger_min, min(finger_max, finger_pos))
        
        # Output combined 9-element array (7 arm joints + 2 fingers)
        all_joints = np.append(target_joints, [finger_pos, finger_pos])
        db.outputs.joint_positions = all_joints.tolist()
        
        return True
        
    except Exception as e:
        db.log_error(f"IK computation failed: {e}")
        return False