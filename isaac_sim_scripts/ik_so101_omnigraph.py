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
3. Add inputs: 
   - linear (vector3d) - connect from ROS2 Subscribe Twist linearVelocity output
   - angular (vector3d) - connect from ROS2 Subscribe Twist angularVelocity output
     NOTE: angular.z carries GRIPPER value (-1 to +1), not yaw rotation!
4. Add outputs: joint_positions (double[6] - 5 arm + 1 gripper), jaw_position (double)
5. Connect ROS2 Subscribe Twist node to this Python Script node
"""

import numpy as np
from isaacsim.core.prims import SingleArticulation
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni.graph.core as og
import omni.timeline
import os

# Global cache
_solver = None
_robot = None
_physics_ready = False
_init_frame_count = 0

# SO101 Configuration
_SO101_URDF = "/media/sarthak/a/Experiments/SO-ARM101_MoveIt_IsaacSim/src/so_arm_description/urdf/so101_new_calib.urdf"
_ROBOT_PRIM_PATH = "/so101_new_calib/base"  # Robot articulation root prim path in USD stage
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
    
    # Lula YAML format based on Isaac Sim examples
    # Note: Using simple string format to ensure correct YAML structure
    yaml_content = f"""# SO101 Robot Description for Lula
api_version: 1.0

# Configuration space joints
cspace:
    - Rotation
    - Pitch
    - Elbow
    - Wrist_Pitch
    - Wrist_Roll

# Default joint positions (home position)
default_q: [0.0, 0.0, 0.0, 0.0, 0.0]

# Gripper joint is fixed for IK purposes
cspace_to_urdf_rules:
    - {{name: Jaw, rule: fixed, value: 0.5}}

# Joint limits and dynamics (optional but helpful)
acceleration_limits: [5.0, 5.0, 5.0, 5.0, 5.0]
jerk_limits: [2500.0, 2500.0, 2500.0, 2500.0, 2500.0]
"""
    
    # Write to temp file
    desc_path = os.path.join(tempfile.gettempdir(), "so101_lula_desc.yaml")
    with open(desc_path, 'w') as f:
        f.write(yaml_content)
    
    return desc_path


def setup(db: og.Database):
    """Initialize SO101 robot and Lula IK solver."""
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
            db.log_info(f"Initialized SO101 robot at: {robot_prim_path}")
        except Exception as e:
            _robot = None
            _physics_ready = False
            return  # Silently retry next frame
    
    # Initialize Lula IK Solver with custom SO101 config
    if _solver is None:
        try:
            # Try with custom description (Isaac Sim 5.1 API)
            robot_desc_path = _create_lula_robot_description()
            _solver = LulaKinematicsSolver(
                robot_description_path=robot_desc_path,
                urdf_path=_SO101_URDF
            )
            db.log_info("Initialized Lula IK solver for SO101 with custom URDF")
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
    
    Inputs (from ROS 2 Twist via OmniGraph - connect directly from ROS2 Subscribe Twist):
        - linear: vector3d (x, y, z) - EE position delta in meters
        - angular: vector3d (x, y, z) - EE rotation delta (roll, pitch, yaw) in radians
        - gripper: double - Gripper/Jaw command (-1 to +1)
    
    Outputs:
        - joint_positions: 5-element double array for SO101 arm joints
        - jaw_position: Mapped jaw position (double)
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
    # angular = (roll, pitch, GRIPPER) - angular.z carries gripper value, not yaw!
    linear = db.inputs.linear
    angular = db.inputs.angular
    
    # Check if inputs are valid (not None and have data)
    if linear is None or angular is None:
        return True  # No ROS2 message received yet, skip silently
    
    try:
        delta_pos = np.array([linear[0], linear[1], linear[2]])
        # Only use angular.x and angular.y for rotation (roll, pitch)
        # angular.z contains gripper value, not yaw
        delta_rpy = np.array([angular[0], angular[1], 0.0])  # No yaw input
        gripper_from_twist = angular[2]  # Gripper embedded in angular.z
    except (TypeError, IndexError):
        return True  # Invalid input format, skip silently
    
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
                current_q = np.array(current_q[:5])
                
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
                current_joints = np.array(joint_positions[:5])
            except Exception as e:
                _physics_ready = False
                return True
            
            # Scale down the delta - different scales for different axes
            # Z movements were causing spinning, so reduce Z sensitivity
            pos_scale = 0.05    # Position scale (meters to radians, approximate)
            rot_scale = 0.1     # Rotation scale
            
            # Improved Jacobian-like mapping for SO101
            # SO101 joints: [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll]
            # Mapping EE movements to joint movements (approximate):
            joint_delta = np.zeros(5)
            
            # Rotation joint: controls yaw (Y-axis movement + yaw rotation)
            joint_delta[0] = delta_pos[1] * pos_scale + delta_rpy[2] * rot_scale
            
            # Pitch joint: controls up/down reach (Z-axis) - REDUCED sensitivity
            joint_delta[1] = -delta_pos[2] * pos_scale * 0.5  # Halved Z sensitivity
            
            # Elbow joint: controls forward reach (X-axis)  
            joint_delta[2] = delta_pos[0] * pos_scale + delta_pos[2] * pos_scale * 0.3
            
            # Wrist Pitch: controls EE pitch orientation
            joint_delta[3] = delta_rpy[1] * rot_scale
            
            # Wrist Roll: controls EE roll orientation
            joint_delta[4] = delta_rpy[0] * rot_scale
            
            target_joints = current_joints + joint_delta
            
            # Clamp joints to reasonable limits to prevent runaway
            joint_limits = np.array([
                [-1.5, 1.5],    # Rotation
                [-1.5, 1.5],    # Pitch
                [-1.5, 1.5],    # Elbow
                [-1.5, 1.5],    # Wrist Pitch
                [-2.5, 2.5]     # Wrist Roll
            ])
            for i in range(5):
                target_joints[i] = np.clip(target_joints[i], joint_limits[i, 0], joint_limits[i, 1])
        
        # Handle gripper (Jaw) - extracted from angular.z of Twist message
        gripper_cmd = gripper_from_twist
        if gripper_cmd is None:
            gripper_cmd = 0.0
        # Map [-1, +1] to Jaw position range from URDF: [0, 1.0] radians
        # -1 = close (0), +1 = open (1.0)
        jaw_min = 0.0
        jaw_max = 1.0
        jaw_pos = jaw_min + (jaw_max - jaw_min) * (gripper_cmd + 1.0) / 2.0
        jaw_pos = max(jaw_min, min(jaw_max, jaw_pos))
        
        # Output combined 6-element array (5 arm joints + 1 gripper)
        all_joints = np.append(target_joints, jaw_pos)
        db.outputs.joint_positions = all_joints.tolist()
        db.outputs.jaw_position = jaw_pos
        
        return True
        
    except Exception as e:
        db.log_error(f"IK computation failed: {e}")
        return False
