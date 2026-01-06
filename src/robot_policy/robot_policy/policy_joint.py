"""
ROS 2 Policy Node for Pi0-based robot control with JOINT STATE output.

This variant is for when the policy server returns joint state deltas
instead of Cartesian EE deltas. Publishes directly to Isaac Sim's
ArticulationController without IK.

1. Subscribing to joint states and camera images
2. Sending observations to Pi0 server (padded to LIBERO 8-DOF format)
3. Receiving joint position deltas from server
4. Publishing joint commands directly (no IK needed)
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
from cv_bridge import CvBridge
import cv2


class PolicyJointNode(Node):
    """
    ROS 2 Policy Node that outputs JOINT STATES directly.
    
    Use this when:
    - Policy server returns joint position deltas (not EE deltas)
    - No IK solver is needed in Isaac Sim
    - Direct joint control is desired
    
    Publishes: JointState to /ee_delta_command (or configured topic)
    """
    
    def __init__(self):
        super().__init__('policy_joint_node')
        self.bridge = CvBridge()
        
        # ===== Parameters =====
        # Robot configuration
        self.declare_parameter('robot_type', 'generic')
        self.declare_parameter('input_dof', 6)   # DOF of incoming joint state
        self.declare_parameter('output_dof', 6)  # DOF to output
        self.declare_parameter('libero_dof', 8)  # DOF to send to server (LIBERO format)
        
        # Joint names for output (default: SO101 joint names)
        self.declare_parameter('joint_names', ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw'])
        
        # ROS topics
        self.declare_parameter('joint_state_topic', '/isaac_joint_states')
        self.declare_parameter('env_camera_topic', '/env_perspective')
        self.declare_parameter('wrist_camera_topic', '/wrist_perspective')
        self.declare_parameter('joint_command_topic', '/ee_delta_command')  # Output topic
        
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
        self.output_dof = self.get_parameter('output_dof').value
        self.libero_dof = self.get_parameter('libero_dof').value
        self.joint_names = self.get_parameter('joint_names').value
        
        joint_topic = self.get_parameter('joint_state_topic').value
        env_topic = self.get_parameter('env_camera_topic').value
        wrist_topic = self.get_parameter('wrist_camera_topic').value
        self.joint_command_topic = self.get_parameter('joint_command_topic').value
        
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
        self.last_joint_command = None
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
        # Joint commands as JointState
        self.joint_publisher = self.create_publisher(JointState, self.joint_command_topic, 10)
        
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
        self.get_logger().info(f'🤖 Policy JOINT node initialized for robot: {self.robot_type}')
        self.get_logger().info(f'   Input DOF: {self.input_dof} → Output DOF: {self.output_dof}')
        self.get_logger().info(f'   Subscribing: {joint_topic}, {env_topic}, {wrist_topic}')
        self.get_logger().info(f'   Publishing joint commands to: {self.joint_command_topic}')
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
            if self.last_joint_command is not None:
                self.joint_publisher.publish(self.last_joint_command)

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
                self._publish_joint_command(action, positions)
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
        Send observation to Pi0 server and get joint action.
        
        Server expects:
        - images: [env_b64, wrist_b64] (base64 JPEG)
        - joint_state: 8-DOF LIBERO format
        - task: string
        
        Server returns:
        - action: joint position deltas or targets
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
                
                # Get action from server response
                # The server may return 'action' (joint deltas) or structured response
                action = result.get('action', None)
                if action is None:
                    # Fallback: try to get ee_delta and convert (if server returns EE format)
                    ee_delta = result.get('ee_delta', [0.0, 0.0, 0.0])
                    ee_rotation = result.get('ee_rotation_delta', [0.0, 0.0, 0.0])
                    gripper = result.get('gripper', 0.0)
                    action = ee_delta + ee_rotation + [gripper]
                
                inference_time = result.get('inference_time_ms', 0)
                
                self.get_logger().debug(
                    f'[{self.robot_type}] Inference: {inference_time:.1f}ms, action: {action[:min(4, len(action))]}...',
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

    # ==================== Publish Joint Command ====================
    
    def _publish_joint_command(self, action, current_positions):
        """
        Publish policy output as JointState command.
        
        Interprets action as joint position deltas and adds to current positions.
        """
        if len(action) < self.output_dof:
            self.get_logger().warning(f'Action too short: {len(action)} < {self.output_dof}')
            return
        
        # Create JointState message
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.name = self.joint_names[:self.output_dof]
        
        # Compute target positions: current + delta * scale
        target_positions = []
        for i in range(self.output_dof):
            delta = float(action[i]) * self.action_scale
            if i < len(current_positions):
                target = float(current_positions[i]) + delta
            else:
                target = delta
            target_positions.append(target)
        
        joint_cmd.position = target_positions
        
        # Cache for republishing
        with self.last_command_lock:
            self.last_joint_command = joint_cmd
        
        # Publish
        self.joint_publisher.publish(joint_cmd)
        
        self.get_logger().info(
            f'Joint Command: {[f"{p:.4f}" for p in target_positions[:min(4, len(target_positions))]]}...',
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyJointNode()
    
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
