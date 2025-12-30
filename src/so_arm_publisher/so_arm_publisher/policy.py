import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
import time
import threading
import io
import requests
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2

class PolicyNode(Node):
    """
    Combined node for subscribing to data and running the policy controller.
    This avoids the complexity of multiple nodes and global variables.
    """
    
    def __init__(self):
        super().__init__('policy_node')
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameter('joint_state_topic', '/isaac_joint_states')
        self.declare_parameter('env_camera_topic', '/env_perspective')
        self.declare_parameter('wrist_camera_topic', '/wrist_perspective')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('policy_server_url', 'http://localhost:8000/predict')
        self.declare_parameter('publish_topic', '/joint_command')
        self.declare_parameter('control_frequency', 10.0)
        
        # State
        self.joint_state = None
        self.env_image = None
        self.wrist_image = None
        self.state_lock = threading.Lock()
        
        # Get parameters
        joint_topic = self.get_parameter('joint_state_topic').value
        env_topic = self.get_parameter('env_camera_topic').value
        wrist_topic = self.get_parameter('wrist_camera_topic').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.policy_url = self.get_parameter('policy_server_url').value
        self.publish_topic = self.get_parameter('publish_topic').value
        control_freq = self.get_parameter('control_frequency').value
        
        # Joint names
        self.joint_names = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']
        
        # Callback groups
        # Subscriptions can run in parallel with the timer, so we use ReentrantCallbackGroup
        # or separate MutuallyExclusiveCallbackGroups.
        self.sub_cb_group = ReentrantCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        
        # Subscriptions
        self.joint_sub = self.create_subscription(
            JointState, joint_topic, self.joint_state_callback, 10, callback_group=self.sub_cb_group)
        self.env_sub = self.create_subscription(
            Image, env_topic, self.env_image_callback, 10, callback_group=self.sub_cb_group)
        self.wrist_sub = self.create_subscription(
            Image, wrist_topic, self.wrist_image_callback, 10, callback_group=self.sub_cb_group)
        
        # Publisher
        self.joint_publisher = self.create_publisher(JointState, self.publish_topic, 10)
        
        # Timer for policy loop
        timer_period = 1.0 / control_freq
        self.timer = self.create_timer(timer_period, self.policy_loop, callback_group=self.timer_cb_group)
        
        self.get_logger().info(f'Policy node initialized 🚀')
        self.get_logger().info(f'Listening on: {joint_topic}, {env_topic}, {wrist_topic}')
        self.get_logger().info(f'Publishing to: {self.publish_topic} at {control_freq}Hz')
        self.get_logger().info(f'Policy server: {self.policy_url}')

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

    def policy_loop(self):
        # Capture state under lock to minimize lock duration
        with self.state_lock:
            local_joint_state = self.joint_state
            local_env_image = self.env_image
            local_wrist_image = self.wrist_image
        
        # Check if we have all required data
        if local_joint_state is None:
            self.get_logger().info('Waiting for joint state...', throttle_duration_sec=5.0)
            return
        
        if local_env_image is None:
            self.get_logger().info('Waiting for env image...', throttle_duration_sec=5.0)
            return
        
        if local_wrist_image is None:
            self.get_logger().info('Waiting for wrist image...', throttle_duration_sec=5.0)
            return
        
        current_positions = local_joint_state.get('position', [])
        if len(current_positions) < 6:
            self.get_logger().warning(f'Incomplete joint state: {len(current_positions)} joints', throttle_duration_sec=5.0)
            return
        
        # Send request to policy server
        try:
            action = self._get_policy_action(current_positions, local_env_image, local_wrist_image)
            if action is not None:
                self._publish_joint_state(action)
        except Exception as e:
            self.get_logger().error(f'Policy request failed: {e}')

    def _get_policy_action(self, positions, env_img, wrist_img):
        try:
            _, env_encoded = cv2.imencode('.jpg', env_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, wrist_encoded = cv2.imencode('.jpg', wrist_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            env_bytes = io.BytesIO(env_encoded.tobytes())
            wrist_bytes = io.BytesIO(wrist_encoded.tobytes())
            
            joint_state_str = str(list(positions[:6]))
            
            files = {
                'image': ('env_view.jpg', env_bytes, 'image/jpeg'),
                'wrist_image': ('wrist_view.jpg', wrist_bytes, 'image/jpeg'),
            }
            data = {'joint_state': joint_state_str}
            
            response = requests.post(self.policy_url, files=files, data=data, timeout=5.0)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    return result.get('action', [])
                else:
                    self.get_logger().warning(f'Policy server error: {result}', throttle_duration_sec=5.0)
            else:
                self.get_logger().warning(f'HTTP error: {response.status_code}', throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f'Policy request error: {e}', throttle_duration_sec=5.0)
        return None

    def _publish_joint_state(self, action):
        if len(action) < 6:
            return
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(a) for a in action[:6]]
        self.joint_publisher.publish(msg)
        self.get_logger().info(f'Published action: {msg.position}', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    
    # Use MultiThreadedExecutor so callbacks can run while timer is waiting for requests
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()