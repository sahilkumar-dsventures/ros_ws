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
   - linear (vector3d) - connect from ROS2 Subscribe Twist linearVelocity output (EE position delta)
   - angular (vector3d) - connect from ROS2 Subscribe Twist angularVelocity output (EE rotation delta)
   - gripper_linear (vector3d) - connect from ROS2 Subscribe Twist (gripper topic) linearVelocity output
     (gripper value is in the z component: 0=close, 1=open)
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
            db.log_info(f"[DEBUG] Attempting Lula init with desc: {robot_desc_path}, urdf: {_SO101_URDF}")
            
            # Check if URDF exists
            if not os.path.exists(_SO101_URDF):
                db.log_error(f"URDF file not found: {_SO101_URDF}")
                _solver = "fallback"
            else:
                _solver = LulaKinematicsSolver(
                    robot_description_path=robot_desc_path,
                    urdf_path=_SO101_URDF
                )
                db.log_info("✓ Lula IK solver initialized for SO101")
        except Exception as e:
            import traceback
            db.log_warning(f"Lula solver init failed: {e}")
            db.log_warning(f"Traceback: {traceback.format_exc()}")
            db.log_info("Using fallback Jacobian IK - movements may be less accurate")
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
    # angular = (roll, pitch, yaw) rotation delta - standard Twist format
    linear = db.inputs.linear
    angular = db.inputs.angular
    
    # Check if inputs are valid (not None and have data)
    if linear is None or angular is None:
        return True  # No ROS2 message received yet, skip silently
    
    try:
        delta_pos = np.array([linear[0], linear[1], linear[2]])
        # Use all three rotation components (roll, pitch, yaw)
        delta_rpy = np.array([angular[0], angular[1], angular[2]])
        db.log_info(f"[DEBUG] Received EE delta - pos: {delta_pos}, rpy: {delta_rpy}")
    except (TypeError, IndexError):
        db.log_warning(f"[DEBUG] Invalid input format - linear: {linear}, angular: {angular}")
        return True  # Invalid input format, skip silently
    
    # Skip if no movement
    if np.allclose(delta_pos, 0) and np.allclose(delta_rpy, 0):
        db.log_info(f"[DEBUG] Skipping - delta is near-zero")
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
                
                # Clamp target position to reasonable workspace bounds for SO101
                # SO101 has ~30cm reach, so clamp to prevent unreachable targets
                workspace_bounds = {
                    'x': (-0.4, 0.4),
                    'y': (-0.4, 0.4),
                    'z': (0.0, 0.5)  # Above table only
                }
                target_pos[0] = np.clip(target_pos[0], workspace_bounds['x'][0], workspace_bounds['x'][1])
                target_pos[1] = np.clip(target_pos[1], workspace_bounds['y'][0], workspace_bounds['y'][1])
                target_pos[2] = np.clip(target_pos[2], workspace_bounds['z'][0], workspace_bounds['z'][1])
                
                db.log_info(f"[DEBUG] Current EE pos: {ee_pos}, Target EE pos: {target_pos}")
                
                # Set warm start with current joint positions to help IK converge
                _solver.set_robot_base_pose(
                    robot_position=np.array([0.0, 0.0, 0.0]),
                    robot_orientation=np.array([1.0, 0.0, 0.0, 0.0])
                )
                
                # Try IK - skip orientation for 5-DOF arm (can't control all 6 Cartesian DOFs)
                # Just use position target, let orientation float
                try:
                    ik_result, success = _solver.compute_inverse_kinematics(
                        frame_name=_EE_FRAME,
                        target_position=target_pos,
                        warm_start=current_q
                    )
                except TypeError:
                    # Some versions require orientation
                    ik_result, success = _solver.compute_inverse_kinematics(
                        frame_name=_EE_FRAME,
                        target_position=target_pos,
                        target_orientation=target_quat,
                        warm_start=current_q
                    )
                
                if success:
                    target_joints = np.array(ik_result)
                    db.log_info(f"[DEBUG] Lula IK SUCCESS - target_joints: {target_joints}")
                else:
                    db.log_info(f"[DEBUG] Lula IK failed, using fallback. Target: {target_pos}")
                    use_fallback = True
            except Exception as e:
                db.log_info(f"Lula IK error: {e}, using fallback")
                use_fallback = True
        
        # Fallback: Simplified Jacobian-like mapping
        if use_fallback or target_joints is None:
            db.log_info(f"[DEBUG] Using FALLBACK Jacobian IK")
            try:
                joint_positions = _robot.get_joint_positions()
                if joint_positions is None:
                    _physics_ready = False
                    return True
                current_joints = np.array(joint_positions[:5])
                db.log_info(f"[DEBUG] Current joints: {current_joints}")
            except Exception as e:
                _physics_ready = False
                return True
            
            # Scale factors for converting EE deltas to joint movements
            # Pi0 outputs small deltas (~0.01-0.02), so we need higher scaling
            # Tune these based on desired responsiveness:
            # - Lower values (0.5-1.0): Smooth, slower movements  
            # - Higher values (2.0-5.0): Fast, responsive movements
            pos_scale = 0.1    # Position scale (meters to radians, approximate)
            rot_scale = 0.1    # Rotation scale (radians to radians)
            
            # Jacobian-like mapping for SO101
            # SO101 joints: [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll]
            # Pi0 output: pos=[x, y, z], rot=[roll, pitch, yaw]
            # Mapping EE movements to joint movements (approximate):
            joint_delta = np.zeros(5)
            
            # Rotation joint (base): controls yaw - responds to Y movement + yaw rotation
            # Positive Y = move left = positive rotation, Positive yaw = rotate CCW
            joint_delta[0] = delta_pos[1] * pos_scale + delta_rpy[2] * rot_scale
            
            # Pitch joint (shoulder): controls reach height (Z-axis)
            # Positive Z = move up = negative pitch (shoulder goes back)
            joint_delta[1] = -delta_pos[2] * pos_scale
            
            # Elbow joint: controls forward/backward reach (X-axis)
            # Positive X = reach forward = elbow extends (positive)
            # Also couples with Z for proper arm extension
            joint_delta[2] = delta_pos[0] * pos_scale + delta_pos[2] * pos_scale * 0.5
            
            # Wrist Pitch: controls EE pitch (nodding up/down)
            # Maps directly from Pi0's pitch rotation
            joint_delta[3] = delta_rpy[1] * rot_scale
            
            # Wrist Roll: controls EE roll (rotating wrist)
            # Maps directly from Pi0's roll rotation
            joint_delta[4] = delta_rpy[0] * rot_scale
            
            db.log_info(f"[DEBUG] Joint delta: {joint_delta}")
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
            
            db.log_info(f"[DEBUG] Clamped target joints: {target_joints}")
        
        # Handle gripper (Jaw) - from separate Twist message's linear.z
        # Pi0 outputs gripper in ~0-1 range (0 = close, 1 = open)
        # Gripper comes from a Twist message where linear.z contains the gripper value
        gripper_linear = db.inputs.gripper_linear if hasattr(db.inputs, 'gripper_linear') else None
        if gripper_linear is not None:
            try:
                gripper_cmd = float(gripper_linear[2])  # z component
            except (TypeError, IndexError):
                gripper_cmd = 0.5
        else:
            gripper_cmd = 0.5
        
        # Clamp Pi0 gripper value to 0-1 range (it may slightly exceed, e.g., 1.01)
        gripper_cmd = np.clip(gripper_cmd, 0.0, 1.0)
        
        # Map [0, 1] directly to Jaw position range from URDF: [0, 1.0] radians
        # 0 = close (0), 1 = open (1.0)
        jaw_min = 0.0
        jaw_max = 1.0
        jaw_pos = jaw_min + (jaw_max - jaw_min) * gripper_cmd
        
        # Output combined 6-element array (5 arm joints + 1 gripper)
        all_joints = np.append(target_joints, jaw_pos)
        db.outputs.joint_positions = all_joints.tolist()
        db.outputs.jaw_position = jaw_pos
        
        # Log final output
        db.log_info(f"[OUTPUT] Joint target: {all_joints.tolist()}")
        
        return True
        
    except Exception as e:
        db.log_error(f"IK computation failed: {e}")
        return False
