import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
import time
import threading
import io
import json
import requests
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String, Bool, Float64MultiArray
from geometry_msgs.msg import Twist, Pose, PoseStamped
from cv_bridge import CvBridge
import cv2

class PolicyNode(Node):
    """
    ROS 2 Policy Node for Pi0 + Isaac Sim Franka control.
    
    This node:
    1. Subscribes to /isaac_joint_states to get current robot state
    2. Subscribes to camera images for Pi0 observation
    3. Sends observation to Pi0 server
    4. Pi0 returns Cartesian EE delta: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
    5. We apply a simple Jacobian-like mapping to convert to joint deltas
    6. Publishes JointState to /ee_delta_command for Isaac Sim ArticulationController
    
    NOTE: Pi0 LIBERO outputs Cartesian EE deltas. The mapping here is a simple
    approximation. For precise control, use a proper IK solver.
    """
    
    def __init__(self):
        super().__init__('policy_node')
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameter('joint_state_topic', '/isaac_joint_states')
        self.declare_parameter('env_camera_topic', '/env_perspective')
        self.declare_parameter('wrist_camera_topic', '/wrist_perspective')
        self.declare_parameter('width', 224)
        self.declare_parameter('height', 224)
        self.declare_parameter('policy_server_url', 'http://localhost:8000/predict')
        self.declare_parameter('publish_topic', '/ee_delta_command')  # JointState for Isaac Sim
        self.declare_parameter('gripper_topic', '/gripper_command')
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('action_scale', 1.0)
        
        # State
        self.joint_state = None
        self.env_image = None
        self.wrist_image = None
        self.state_lock = threading.Lock()
        
        # Last command for fast republishing while inference is slow
        self.last_command = None
        self.last_command_lock = threading.Lock()
        
        # Get parameters
        joint_topic = self.get_parameter('joint_state_topic').value
        env_topic = self.get_parameter('env_camera_topic').value
        wrist_topic = self.get_parameter('wrist_camera_topic').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.policy_url = self.get_parameter('policy_server_url').value
        self.publish_topic = self.get_parameter('publish_topic').value
        self.gripper_topic = self.get_parameter('gripper_topic').value
        control_freq = self.get_parameter('control_frequency').value
        self.action_scale = self.get_parameter('action_scale').value
        
        # Joint names (LIBERO Franka format: 7 arm joints + 1 gripper = 8 DOF)
        # LIBERO dataset format: [joint1-7, gripper_width]
        self.arm_joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        # For Isaac Sim, we map the single gripper to two finger joints
        self.finger_joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
        self.joint_names = self.arm_joint_names + self.finger_joint_names
        
        # Callback groups
        self.sub_cb_group = ReentrantCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.republish_cb_group = MutuallyExclusiveCallbackGroup()
        
        # Subscriptions
        self.joint_sub = self.create_subscription(
            JointState, joint_topic, self.joint_state_callback, 10, callback_group=self.sub_cb_group)
        self.env_sub = self.create_subscription(
            Image, env_topic, self.env_image_callback, 10, callback_group=self.sub_cb_group)
        self.wrist_sub = self.create_subscription(
            Image, wrist_topic, self.wrist_image_callback, 10, callback_group=self.sub_cb_group)
        
        # Publisher: JointState for Isaac Sim ArticulationController
        self.joint_command_publisher = self.create_publisher(JointState, self.publish_topic, 10)

        # Inference flag with lock to prevent race conditions
        self.inference_in_progress = False
        self.inference_lock = threading.Lock()
        
        # Timer for policy loop (inference trigger)
        timer_period = 1.0 / control_freq
        self.timer = self.create_timer(timer_period, self.policy_loop, callback_group=self.timer_cb_group)
        
        # Fast timer for republishing last command (100Hz to keep robot stable)
        self.republish_timer = self.create_timer(0.01, self.republish_last_command, callback_group=self.republish_cb_group)
        
        self.get_logger().info(f'Policy node initialized 🚀')
        self.get_logger().info(f'Subscribing: {joint_topic}, {env_topic}, {wrist_topic}')
        self.get_logger().info(f'Publishing JointState to: {self.publish_topic} at 100Hz')
        self.get_logger().info(f'Policy server: {self.policy_url}')
        self.get_logger().info(f'Action scale: {self.action_scale}')

    def joint_state_callback(self, msg):
        with self.state_lock:
            self.joint_state = {
                'position': list(msg.position),
                'name': list(msg.name)
            }

    def _process_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            resized = cv2.resize(img, (self.width, self.height), cv2.INTER_LINEAR)
            return resized
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            return None

    def env_image_callback(self, msg):
        img = self._process_image(msg)
        if img is not None:
            with self.state_lock:
                self.env_image = img

    def wrist_image_callback(self, msg):
        img = self._process_image(msg)
        if img is not None:
            with self.state_lock:
                self.wrist_image = img

    def republish_last_command(self):
        """Republish last command at high frequency to keep robot stable."""
        with self.last_command_lock:
            if self.last_command is not None:
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.joint_names
                msg.position = self.last_command
                self.joint_command_publisher.publish(msg)

    def policy_loop(self):
        # Check and set flag atomically to prevent race conditions
        with self.inference_lock:
            if self.inference_in_progress:
                return
            self.inference_in_progress = True

        with self.state_lock:
            local_joint_state = self.joint_state
            local_env_image = self.env_image
            local_wrist_image = self.wrist_image

        if local_joint_state is None or local_env_image is None or local_wrist_image is None:
            # Reset flag if we can't run inference
            with self.inference_lock:
                self.inference_in_progress = False
            return

        threading.Thread(
            target=self._run_inference_thread, 
            args=(local_joint_state['position'], local_env_image, local_wrist_image),
            daemon=True
        ).start()

    def _run_inference_thread(self, positions, env_img, wrist_img):
        try:
            action = self._get_policy_action(positions, env_img, wrist_img)
            if action is not None:
                self._publish_joint_command(action, positions)
        except Exception as e:
            self.get_logger().error(f'Threaded inference failed: {e}')
        finally:
            with self.inference_lock:
                self.inference_in_progress = False

    def _get_policy_action(self, positions, env_img, wrist_img):      
        """
        Send observation to PiZero FastAPI server and get action.
        
        Server expects JSON with:
        - images: List of base64 encoded JPEG images (first=env, second=wrist)
        - joint_state: List of joint positions (floats)
        - task: Task description string (optional)
        
        Server returns (updated structure for ee_delta_cartesian):
        - ee_delta: [dx, dy, dz] in meters (typical range: ~-0.5 to +0.5)
        - ee_rotation_delta: [roll, pitch, yaw] in radians (typical range: ~-0.1 to +0.1)
        - gripper: -1 (close) to +1 (open)
        - action_type: "ee_delta_cartesian"
        - inference_time_ms: Inference time
        """
        try:
            import base64
            
            # Encode images to base64 JPEG
            _, env_encoded = cv2.imencode('.jpg', env_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, wrist_encoded = cv2.imencode('.jpg', wrist_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            env_b64 = base64.b64encode(env_encoded.tobytes()).decode('utf-8')
            wrist_b64 = base64.b64encode(wrist_encoded.tobytes()).decode('utf-8')
            
            # Convert Isaac Sim DOF (9: 7 arm + 2 fingers) to LIBERO DOF (8: 7 arm + 1 gripper)
            # LIBERO expects gripper as single value, Isaac Sim has two finger joints
            if len(positions) >= 9:
                arm_joints = list(positions[:7])  # 7 arm joints
                # Average the two finger positions to get gripper width
                # Isaac Sim: finger1 + finger2 (each 0 to 0.04m)
                # LIBERO expects gripper width (combined opening)
                gripper_width = (float(positions[7]) + float(positions[8]))  # Total gripper opening
                libero_state = arm_joints + [gripper_width]
            elif len(positions) == 8:
                # Already in LIBERO format
                libero_state = list(positions)
            else:
                # Fallback: pad with zeros if needed
                libero_state = list(positions) + [0.0] * (8 - len(positions))
                self.get_logger().warning(f'Unexpected DOF count: {len(positions)}, expected 8 or 9')
            
            # Build JSON payload matching PredictRequest schema
            payload = {
                "images": [env_b64, wrist_b64],
                "joint_state": libero_state,  # 8 DOF for LIBERO
                "task": "Pick up the object"  # Default task, can be parameterized
            }
            
            # Send as JSON
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                self.policy_url, 
                json=payload, 
                headers=headers,
                timeout=10.0  # Increased timeout for inference
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract new response fields (ee_delta_cartesian format)
                ee_delta = result.get('ee_delta', [0.0, 0.0, 0.0])  # [dx, dy, dz] in meters
                ee_rotation_delta = result.get('ee_rotation_delta', [0.0, 0.0, 0.0])  # [roll, pitch, yaw] in radians
                gripper = result.get('gripper', 0.0)  # -1 (close) to +1 (open)
                action_type = result.get('action_type', 'ee_delta_cartesian')
                inference_time = result.get('inference_time_ms', 0)
                
                # Combine into single action vector: [dx, dy, dz, roll, pitch, yaw, gripper]
                action = ee_delta + ee_rotation_delta + [gripper]
                
                self.get_logger().debug(
                    f'Inference: {inference_time:.1f}ms, type: {action_type}, '
                    f'pos_delta: [{ee_delta[0]:.4f}, {ee_delta[1]:.4f}, {ee_delta[2]:.4f}], '
                    f'rot_delta: [{ee_rotation_delta[0]:.4f}, {ee_rotation_delta[1]:.4f}, {ee_rotation_delta[2]:.4f}], '
                    f'gripper: {gripper:.2f}',
                    throttle_duration_sec=1.0
                )
                return action
            else:
                self.get_logger().error(f'HTTP error {response.status_code}: {response.text}')
                
        except requests.exceptions.Timeout:
            self.get_logger().error('Policy server timeout', throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f'Policy request error: {e}', throttle_duration_sec=5.0)
        return None
    
    def _publish_joint_command(self, action_delta, current_positions):
        """
        Convert Pi0 Cartesian EE delta to joint position commands and publish.
        
        Pi0 LIBERO outputs Cartesian EE deltas (action_type: ee_delta_cartesian):
        - [0-2]: Position delta (dx, dy, dz) in meters (~-0.5 to +0.5)
        - [3-5]: Rotation delta (roll, pitch, yaw) in radians (~-0.1 to +0.1)
        - [6]: Gripper command (-1=close, +1=open)
        
        This uses a simplified mapping (approximate Jacobian transpose).
        For precise control, use Isaac Sim's IK solver to convert:
          target_ee_pos = current_ee_pos + ee_delta
          target_ee_rot = apply_rotation_delta(current_ee_rot, ee_rotation_delta)
          target_joint_positions = robot.inverse_kinematics(target_ee_pos, target_ee_rot)
        """
        if len(action_delta) < 7:
            self.get_logger().warning(f'Action too short: {len(action_delta)}')
            return
        
        if len(current_positions) < 7:
            self.get_logger().warning(f'Current positions too short: {len(current_positions)}')
            return
        
        # Extract EE delta components
        dx = float(action_delta[0]) * self.action_scale
        dy = float(action_delta[1]) * self.action_scale
        dz = float(action_delta[2]) * self.action_scale
        droll = float(action_delta[3]) * self.action_scale
        dpitch = float(action_delta[4]) * self.action_scale
        dyaw = float(action_delta[5]) * self.action_scale
        gripper = float(action_delta[6])
        
        # Simple Jacobian-like mapping from EE delta to joint delta
        # This is a rough approximation for Franka Panda
        # Real IK would be more accurate
        scale = 1  # Damping factor for safety
        joint_delta = np.zeros(7)
        
        # Approximate mapping (simplified):
        # Joint 1 (base rotation) affects Y motion
        # Joint 2 (shoulder lift) affects Z motion  
        # Joint 3 (elbow) affects X motion
        # Joints 4-6 (wrist) affect rotation
        joint_delta[0] = dy * scale      # Base rotation
        joint_delta[1] = -dz * scale     # Shoulder (negative for intuitive control)
        joint_delta[2] = dx * scale      # Elbow
        joint_delta[3] = droll * scale   # Wrist roll
        joint_delta[4] = dpitch * scale  # Wrist pitch
        joint_delta[5] = dyaw * scale    # Wrist yaw
        joint_delta[6] = 0.0             # Last joint (minimal effect)
        
        # Compute target positions: current + delta
        target_positions = []
        for i in range(7):
            if i < len(current_positions):
                target = float(current_positions[i]) + joint_delta[i]
            else:
                target = joint_delta[i]
            target_positions.append(target)
        
        # Gripper: map [-1, +1] to Isaac Sim finger positions
        # LIBERO gripper: -1 = close, +1 = open
        # Isaac Sim fingers: 0.0 = close, 0.04 = open (max width)
        finger_pos = 0.02 * (gripper + 1.0)  # Map [-1,+1] to [0, 0.04]
        finger_pos = max(0.0, min(0.04, finger_pos))
        target_positions.append(finger_pos)  # finger_joint1
        target_positions.append(finger_pos)  # finger_joint2
        
        # Cache command for fast republishing
        with self.last_command_lock:
            self.last_command = target_positions.copy()
        
        # Publish as JointState for Isaac Sim ArticulationController
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names  # 7 arm + 2 fingers for Isaac Sim
        msg.position = target_positions
        
        self.joint_command_publisher.publish(msg)
        
        self.get_logger().info(
            f'Arm: [{target_positions[0]:.3f}, {target_positions[1]:.3f}, {target_positions[2]:.3f}, '
            f'{target_positions[3]:.3f}, {target_positions[4]:.3f}, {target_positions[5]:.3f}, {target_positions[6]:.3f}] '
            f'Grip: {finger_pos:.4f}',
            throttle_duration_sec=0.5
        )

def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()