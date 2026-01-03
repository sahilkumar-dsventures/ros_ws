"""
Generalized ROS 2 Policy Node for Pi0-based robot control.

This node works with ANY robot DOF by:
1. Subscribing to joint states and camera images
2. Sending observations to Pi0 server (padded to LIBERO 8-DOF format)
3. Receiving 7D Cartesian EE delta actions
4. Publishing raw EE deltas (Twist) + gripper (Float64) for Isaac Sim IK

The actual IK is handled by Isaac Sim OmniGraph scripts.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
import threading
import requests
import base64
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2


class PolicyNode(Node):
    """
    Generalized ROS 2 Policy Node for Pi0 + Isaac Sim control.
    
    Works with any robot by:
    1. Reading joint states (any DOF) → padding/converting to LIBERO 8-DOF
    2. Sending to Pi0 server → receiving 7D Cartesian EE delta
    3. Publishing Twist (EE delta) + Float64 (gripper) for Isaac Sim IK
    
    Supported robots (configured via YAML):
    - Franka Panda: 9 DOF (7 arm + 2 fingers) → 8 DOF LIBERO
    - SO101: 6 DOF (5 arm + 1 jaw) → 8 DOF LIBERO
    - Any other robot with proper DOF mapping
    """
    
    def __init__(self):
        super().__init__('policy_node')
        self.bridge = CvBridge()
        
        # ===== Parameters =====
        # Robot configuration
        self.declare_parameter('robot_type', 'generic')
        self.declare_parameter('input_dof', 6)   # DOF of incoming joint state
        self.declare_parameter('libero_dof', 8)  # DOF to send to server (LIBERO format)
        
        # ROS topics
        self.declare_parameter('joint_state_topic', '/isaac_joint_states')
        self.declare_parameter('env_camera_topic', '/env_perspective')
        self.declare_parameter('wrist_camera_topic', '/wrist_perspective')
        self.declare_parameter('ee_delta_topic', '/ee_delta_command')
        self.declare_parameter('gripper_topic', '/gripper_command')
        
        # Image processing
        self.declare_parameter('width', 224)
        self.declare_parameter('height', 224)
        
        # Policy server
        self.declare_parameter('policy_server_url', 'http://localhost:8000/predict')
        self.declare_parameter('task_description', 'Pick up the object')
        
        # Control
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('action_scale', 1.0)
        
        # ===== Get Parameters =====
        self.robot_type = self.get_parameter('robot_type').value
        self.input_dof = self.get_parameter('input_dof').value
        self.libero_dof = self.get_parameter('libero_dof').value
        
        joint_topic = self.get_parameter('joint_state_topic').value
        env_topic = self.get_parameter('env_camera_topic').value
        wrist_topic = self.get_parameter('wrist_camera_topic').value
        self.ee_delta_topic = self.get_parameter('ee_delta_topic').value
        self.gripper_topic = self.get_parameter('gripper_topic').value
        
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        
        self.policy_url = self.get_parameter('policy_server_url').value
        self.task_description = self.get_parameter('task_description').value
        
        control_freq = self.get_parameter('control_frequency').value
        self.action_scale = self.get_parameter('action_scale').value
        
        # ===== State =====
        self.joint_state = None
        self.env_image = None
        self.wrist_image = None
        self.state_lock = threading.Lock()
        
        # Last command for republishing
        self.last_ee_delta = None
        self.last_gripper = None
        self.last_command_lock = threading.Lock()
        
        # ===== Callback Groups =====
        self.sub_cb_group = ReentrantCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.republish_cb_group = MutuallyExclusiveCallbackGroup()
        
        # ===== Subscriptions =====
        self.joint_sub = self.create_subscription(
            JointState, joint_topic, self.joint_state_callback, 10,
            callback_group=self.sub_cb_group)
        self.env_sub = self.create_subscription(
            Image, env_topic, self.env_image_callback, 10,
            callback_group=self.sub_cb_group)
        self.wrist_sub = self.create_subscription(
            Image, wrist_topic, self.wrist_image_callback, 10,
            callback_group=self.sub_cb_group)
        
        # ===== Publishers =====
        # EE Delta as Twist: linear=(dx,dy,dz), angular=(droll,dpitch,dyaw)
        self.ee_delta_publisher = self.create_publisher(Twist, self.ee_delta_topic, 10)
        # Gripper as Float64: -1 (close) to +1 (open)
        self.gripper_publisher = self.create_publisher(Float64, self.gripper_topic, 10)
        
        # ===== Inference Control =====
        self.inference_in_progress = False
        self.inference_lock = threading.Lock()
        
        # ===== Timers =====
        timer_period = 1.0 / control_freq
        self.timer = self.create_timer(timer_period, self.policy_loop,
                                        callback_group=self.timer_cb_group)
        # Fast republish at 100Hz for smooth control
        self.republish_timer = self.create_timer(0.01, self.republish_last_command,
                                                   callback_group=self.republish_cb_group)
        
        # ===== Logging =====
        self.get_logger().info(f'🤖 Policy node initialized for robot: {self.robot_type}')
        self.get_logger().info(f'   Input DOF: {self.input_dof} → LIBERO DOF: {self.libero_dof}')
        self.get_logger().info(f'   Subscribing: {joint_topic}, {env_topic}, {wrist_topic}')
        self.get_logger().info(f'   Publishing EE delta to: {self.ee_delta_topic}')
        self.get_logger().info(f'   Publishing gripper to: {self.gripper_topic}')
        self.get_logger().info(f'   Policy server: {self.policy_url}')
        self.get_logger().info(f'   Action scale: {self.action_scale}')

    # ==================== Callbacks ====================
    
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
            self.get_logger().error(f"Image processing error: {e}")
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
        """Republish last command at high frequency for smooth control."""
        with self.last_command_lock:
            if self.last_ee_delta is not None:
                self.ee_delta_publisher.publish(self.last_ee_delta)
            if self.last_gripper is not None:
                self.gripper_publisher.publish(self.last_gripper)

    # ==================== Policy Loop ====================
    
    def policy_loop(self):
        with self.inference_lock:
            if self.inference_in_progress:
                return
            self.inference_in_progress = True

        with self.state_lock:
            local_joint_state = self.joint_state
            local_env_image = self.env_image
            local_wrist_image = self.wrist_image

        if local_joint_state is None or local_env_image is None or local_wrist_image is None:
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
                self._publish_ee_delta(action)
        except Exception as e:
            self.get_logger().error(f'Inference thread failed: {e}')
        finally:
            with self.inference_lock:
                self.inference_in_progress = False

    # ==================== DOF Conversion ====================
    
    def _convert_to_libero_dof(self, positions: list) -> list:
        """
        Convert robot-specific joint positions to LIBERO 8-DOF format.
        
        LIBERO format: [7 arm joints] + [1 gripper width]
        
        Robot-specific mappings:
        - Franka (9 DOF): [7 arm] + [finger1, finger2] → [7 arm] + [finger1+finger2]
        - SO101 (6 DOF):  [5 arm] + [jaw] → [5 arm, 0, 0] + [jaw]
        - Generic: pad with zeros to 8 DOF
        """
        n = len(positions)
        
        if self.robot_type == 'franka':
            # Franka: 7 arm + 2 fingers → 7 arm + gripper_width
            if n >= 9:
                arm_joints = list(positions[:7])
                gripper_width = float(positions[7]) + float(positions[8])
                return arm_joints + [gripper_width]
            elif n == 8:
                return list(positions)
            else:
                return list(positions) + [0.0] * (8 - n)
                
        elif self.robot_type == 'so101':
            # SO101: 5 arm + 1 jaw → pad arm to 7, then jaw
            if n >= 6:
                arm_joints = list(positions[:5])
                jaw = float(positions[5])
                # Pad arm to 7 joints, then add jaw
                return arm_joints + [0.0, 0.0] + [jaw]
            else:
                return list(positions) + [0.0] * (8 - n)
        
        else:
            # Generic: just pad to libero_dof
            if n >= self.libero_dof:
                return list(positions[:self.libero_dof])
            else:
                return list(positions) + [0.0] * (self.libero_dof - n)

    # ==================== Policy Request ====================
    
    def _get_policy_action(self, positions, env_img, wrist_img):
        """
        Send observation to Pi0 server and get EE delta action.
        
        Server expects:
        - images: [env_b64, wrist_b64] (base64 JPEG)
        - joint_state: 8-DOF LIBERO format
        - task: string
        
        Server returns:
        - ee_delta: [dx, dy, dz] meters
        - ee_rotation_delta: [roll, pitch, yaw] radians
        - gripper: -1 to +1
        """
        try:
            # Encode images
            _, env_encoded = cv2.imencode('.jpg', env_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, wrist_encoded = cv2.imencode('.jpg', wrist_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            env_b64 = base64.b64encode(env_encoded.tobytes()).decode('utf-8')
            wrist_b64 = base64.b64encode(wrist_encoded.tobytes()).decode('utf-8')
            
            # Convert to LIBERO format
            libero_state = self._convert_to_libero_dof(positions)
            
            payload = {
                "images": [env_b64, wrist_b64],
                "joint_state": libero_state,
                "task": self.task_description
            }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                self.policy_url,
                json=payload,
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                
                ee_delta = result.get('ee_delta', [0.0, 0.0, 0.0])
                ee_rotation_delta = result.get('ee_rotation_delta', [0.0, 0.0, 0.0])
                gripper = result.get('gripper', 0.0)
                inference_time = result.get('inference_time_ms', 0)
                
                action = ee_delta + ee_rotation_delta + [gripper]
                
                self.get_logger().debug(
                    f'[{self.robot_type}] Inference: {inference_time:.1f}ms, '
                    f'pos: [{ee_delta[0]:.4f}, {ee_delta[1]:.4f}, {ee_delta[2]:.4f}], '
                    f'rot: [{ee_rotation_delta[0]:.4f}, {ee_rotation_delta[1]:.4f}, {ee_rotation_delta[2]:.4f}], '
                    f'grip: {gripper:.2f}',
                    throttle_duration_sec=1.0
                )
                return action
            else:
                self.get_logger().error(f'HTTP {response.status_code}: {response.text}')
                
        except requests.exceptions.Timeout:
            self.get_logger().error('Policy server timeout', throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f'Policy request error: {e}', throttle_duration_sec=5.0)
        return None

    # ==================== Publish EE Delta ====================
    
    def _publish_ee_delta(self, action):
        """
        Publish Pi0's 7D EE delta as Twist + Float64 for Isaac Sim IK.
        
        Pi0 output: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        
        Isaac Sim should:
        1. Subscribe to /ee_delta_command (Twist)
        2. Get current EE pose from robot
        3. target_pose = current_pose + delta
        4. Solve IK → joint positions
        5. Apply to ArticulationController
        """
        if len(action) < 7:
            self.get_logger().warning(f'Action too short: {len(action)}')
            return
        
        # Apply action scale
        dx = float(action[0]) * self.action_scale
        dy = float(action[1]) * self.action_scale
        dz = float(action[2]) * self.action_scale
        droll = float(action[3]) * self.action_scale
        dpitch = float(action[4]) * self.action_scale
        dyaw = float(action[5]) * self.action_scale
        gripper = float(action[6])
        
        # Create Twist for EE delta
        ee_delta_msg = Twist()
        ee_delta_msg.linear.x = dx
        ee_delta_msg.linear.y = dy
        ee_delta_msg.linear.z = dz
        ee_delta_msg.angular.x = droll
        ee_delta_msg.angular.y = dpitch
        ee_delta_msg.angular.z = dyaw
        
        # Create gripper message
        gripper_msg = Float64()
        gripper_msg.data = gripper
        
        # Cache for republishing
        with self.last_command_lock:
            self.last_ee_delta = ee_delta_msg
            self.last_gripper = gripper_msg
        
        # Publish
        self.ee_delta_publisher.publish(ee_delta_msg)
        self.gripper_publisher.publish(gripper_msg)
        
        self.get_logger().info(
            f'EE Delta: pos=[{dx:.4f}, {dy:.4f}, {dz:.4f}]m '
            f'rot=[{droll:.4f}, {dpitch:.4f}, {dyaw:.4f}]rad '
            f'grip={gripper:.2f}',
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
