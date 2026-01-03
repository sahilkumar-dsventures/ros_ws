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
import base64
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String, Bool, Float64MultiArray, Float64
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
    4. Pi0 returns Cartesian EE delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    5. Publishes raw EE deltas as Twist message for Isaac Sim's IK solver
    6. Publishes gripper command separately
    
    IMPORTANT: Pi0 LIBERO outputs Cartesian EE deltas. Isaac Sim should:
    1. Get current EE pose
    2. Apply the delta: target_pose = current_pose + delta
    3. Use IK to solve for joint positions
    4. Send joint positions to ArticulationController
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
        self.declare_parameter('ee_delta_topic', '/ee_delta_command')  # Twist for EE delta
        self.declare_parameter('gripper_topic', '/gripper_command')    # Float64 for gripper
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('action_scale', 1.0)
        self.declare_parameter('task_description', 'Pick up the object')
        
        # State
        self.joint_state = None
        self.env_image = None
        self.wrist_image = None
        self.state_lock = threading.Lock()
        
        # Last command for republishing
        self.last_ee_delta = None
        self.last_gripper = None
        self.last_command_lock = threading.Lock()
        
        # Get parameters
        joint_topic = self.get_parameter('joint_state_topic').value
        env_topic = self.get_parameter('env_camera_topic').value
        wrist_topic = self.get_parameter('wrist_camera_topic').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.policy_url = self.get_parameter('policy_server_url').value
        self.ee_delta_topic = self.get_parameter('ee_delta_topic').value
        self.gripper_topic = self.get_parameter('gripper_topic').value
        control_freq = self.get_parameter('control_frequency').value
        self.action_scale = self.get_parameter('action_scale').value
        self.task_description = self.get_parameter('task_description').value
        
        # Joint names (for reading state - LIBERO format: 7 arm + 1 gripper = 8 DOF)
        self.arm_joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        
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
        
        # Publishers
        # EE Delta as Twist: linear=(dx,dy,dz), angular=(droll,dpitch,dyaw)
        self.ee_delta_publisher = self.create_publisher(Twist, self.ee_delta_topic, 10)
        # Gripper as Float64: -1 (close) to +1 (open)
        self.gripper_publisher = self.create_publisher(Float64, self.gripper_topic, 10)

        # Inference flag
        self.inference_in_progress = False
        self.inference_lock = threading.Lock()
        
        # Timer for policy loop
        timer_period = 1.0 / control_freq
        self.timer = self.create_timer(timer_period, self.policy_loop, callback_group=self.timer_cb_group)
        
        # Fast republish timer (100Hz)
        self.republish_timer = self.create_timer(0.01, self.republish_last_command, callback_group=self.republish_cb_group)
        
        self.get_logger().info(f'Policy node initialized 🚀')
        self.get_logger().info(f'Subscribing: {joint_topic}, {env_topic}, {wrist_topic}')
        self.get_logger().info(f'Publishing EE delta (Twist) to: {self.ee_delta_topic}')
        self.get_logger().info(f'Publishing gripper (Float64) to: {self.gripper_topic}')
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
        """Republish last command at high frequency for smooth control."""
        with self.last_command_lock:
            if self.last_ee_delta is not None:
                self.ee_delta_publisher.publish(self.last_ee_delta)
            if self.last_gripper is not None:
                self.gripper_publisher.publish(self.last_gripper)

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
            self.get_logger().error(f'Threaded inference failed: {e}')
        finally:
            with self.inference_lock:
                self.inference_in_progress = False

    def _get_policy_action(self, positions, env_img, wrist_img):      
        """
        Send observation to PiZero FastAPI server and get action.
        
        Server expects JSON with:
        - images: List of base64 encoded JPEG images (first=env, second=wrist)
        - joint_state: List of joint positions (8 DOF for LIBERO)
        - task: Task description string
        
        Server returns (ee_delta_cartesian format):
        - ee_delta: [dx, dy, dz] in meters
        - ee_rotation_delta: [roll, pitch, yaw] in radians
        - gripper: -1 (close) to +1 (open)
        - action_type: "ee_delta_cartesian"
        - inference_time_ms: Inference time
        """
        try:
            # Encode images to base64 JPEG
            _, env_encoded = cv2.imencode('.jpg', env_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, wrist_encoded = cv2.imencode('.jpg', wrist_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            env_b64 = base64.b64encode(env_encoded.tobytes()).decode('utf-8')
            wrist_b64 = base64.b64encode(wrist_encoded.tobytes()).decode('utf-8')
            
            # Convert Isaac Sim DOF (9: 7 arm + 2 fingers) to LIBERO DOF (8: 7 arm + 1 gripper)
            if len(positions) >= 9:
                arm_joints = list(positions[:7])
                gripper_width = float(positions[7]) + float(positions[8])
                libero_state = arm_joints + [gripper_width]
            elif len(positions) == 8:
                libero_state = list(positions)
            else:
                libero_state = list(positions) + [0.0] * (8 - len(positions))
                self.get_logger().warning(f'Unexpected DOF count: {len(positions)}, expected 8 or 9')
            
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
                
                # Extract response fields (ee_delta_cartesian format)
                ee_delta = result.get('ee_delta', [0.0, 0.0, 0.0])
                ee_rotation_delta = result.get('ee_rotation_delta', [0.0, 0.0, 0.0])
                gripper = result.get('gripper', 0.0)
                action_type = result.get('action_type', 'ee_delta_cartesian')
                inference_time = result.get('inference_time_ms', 0)
                
                # Combine into action vector
                action = ee_delta + ee_rotation_delta + [gripper]
                
                self.get_logger().debug(
                    f'Inference: {inference_time:.1f}ms, type: {action_type}, '
                    f'pos: [{ee_delta[0]:.4f}, {ee_delta[1]:.4f}, {ee_delta[2]:.4f}], '
                    f'rot: [{ee_rotation_delta[0]:.4f}, {ee_rotation_delta[1]:.4f}, {ee_rotation_delta[2]:.4f}], '
                    f'grip: {gripper:.2f}',
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
    
    def _publish_ee_delta(self, action):
        """
        Publish Pi0's EE delta as Twist message for Isaac Sim.
        
        Pi0 outputs Cartesian EE deltas:
        - [0-2]: Position delta (dx, dy, dz) in meters
        - [3-5]: Rotation delta (roll, pitch, yaw) in radians
        - [6]: Gripper command (-1=close, +1=open)
        
        Isaac Sim should:
        1. Subscribe to /ee_delta_command (Twist)
        2. Get current EE pose
        3. Compute target: target_pose = current_pose + delta
        4. Use IK solver to get joint positions
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
        
        # Create Twist message for EE delta
        ee_delta_msg = Twist()
        ee_delta_msg.linear.x = dx   # meters
        ee_delta_msg.linear.y = dy   # meters
        ee_delta_msg.linear.z = dz   # meters
        ee_delta_msg.angular.x = droll   # radians (roll)
        ee_delta_msg.angular.y = dpitch  # radians (pitch)
        ee_delta_msg.angular.z = dyaw    # radians (yaw)
        
        # Create gripper message
        gripper_msg = Float64()
        gripper_msg.data = gripper  # -1 to +1
        
        # Cache for republishing
        with self.last_command_lock:
            self.last_ee_delta = ee_delta_msg
            self.last_gripper = gripper_msg
        
        # Publish
        self.ee_delta_publisher.publish(ee_delta_msg)
        self.gripper_publisher.publish(gripper_msg)
        
        self.get_logger().info(
            f'EE Delta: pos=[{dx:.4f}, {dy:.4f}, {dz:.4f}]m rot=[{droll:.4f}, {dpitch:.4f}, {dyaw:.4f}]rad grip={gripper:.2f}',
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