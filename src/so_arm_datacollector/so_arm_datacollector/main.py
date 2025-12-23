import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Bool
import threading
from pathlib import Path
import argparse
import os
import cv2
import time
from cv_bridge import CvBridge
import pandas as pd
import numpy as np

cv_bridge = CvBridge()

joint_states = None
# Dictionary to store latest frames from all cameras: {topic_name: frame}
latest_camera_frames = {}

class Get_Poses_Subscriber(Node):
   def __init__(self):
       super().__init__('get_poses_subscriber')
       self.declare_parameter('topic_name', '/poses')

       self.topic_name = self.get_parameter('topic_name').value
       self.subscription = self.create_subscription(JointState, self.topic_name, self.listener_callback, 10)

   def listener_callback(self, msg : JointState):
        global joint_states
       
        joint_states = {
            'name': msg.name,
            'position': list(msg.position),
            'velocity': list(msg.velocity),
            'effort': list(msg.effort)
        }

class Camera_Subscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        # Declare a list of topics
        self.declare_parameter('topics', ['/wrist_perspective', '/env_perspective', '/camera_array'])
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.topics = self.get_parameter('topics').value

        self.subscriptions_list = []

        for topic in self.topics:
            self.get_logger().info(f'Subscribing to camera topic: {topic}')
            # Need to capture topic in closure
            self.subscriptions_list.append(
                self.create_subscription(
                    Image, 
                    topic, 
                    lambda msg, t=topic: self.listener_callback(msg, t), 
                    10
                )
            )
            # Initialize entry in global dict
            latest_camera_frames[topic] = None
    
    def listener_callback(self, msg : Image , topic_name):
        global latest_camera_frames
        try:
            img = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            resized_image = cv2.resize(img, (self.width, self.height), cv2.INTER_LINEAR)
            
            # Ensure 3 channels
            if len(resized_image.shape) == 2:
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
            elif len(resized_image.shape) == 3 and resized_image.shape[2] == 1:
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
                
            latest_camera_frames[topic_name] = resized_image
        except Exception as e:
            self.get_logger().error(f"Error processing image from {topic_name}: {e}")

class Data_Recorder(Node):

    def __init__(self):
        super().__init__('data_recorder')

        self.declare_parameter('data_dir', '/media/sarthak/a/ros_ws/lerobot/SO-ARM101_MoveIt_IsaacSim/')
        self.declare_parameter('fps', 30)
        self.declare_parameter('num_joint', 6)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('is_done_topic', '/is_done')
        # Get the same topics as Camera_Subscriber (or define them again here if needed, but best shared)
        # For this script we assume the same parameters or we pass them. 
        # Since rclpy parameters are node-specific, we declare them here too matching Camera_Subscriber defaults
        self.declare_parameter('camera_topics', ['/wrist_perspective', '/env_perspective', '/camera_array'])

        self.data_dir = self.get_parameter('data_dir').value
        self.fps = self.get_parameter('fps').value
        self.num_joint = self.get_parameter('num_joint').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.is_done_topic = self.get_parameter('is_done_topic').value
        self.camera_topics = self.get_parameter('camera_topics').value

        self.create_subscription(Bool, self.is_done_topic, self.timer_callback, 10)
        
        self.total_steps = 0
        self.episode_idx = 0
        
        # Buffers
        # Dictionary for dynamic camera buffers: {topic_name: [list_of_frames]}
        self.camera_buffers = {topic: [] for topic in self.camera_topics}
        
        self.joint_states_name_buffer = []
        self.joint_states_position_buffer = []
        self.joint_states_velocity_buffer = []
        self.joint_states_effort_buffer = []
        self.timestamp_buffer = []

        self.size = (self.width, self.height)
        self.base_dir = Path(self.data_dir)
        
        # Structure: data/chunk-000/
        self.log_dir = self.base_dir / "data" / "chunk-000"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Video directories - Created dynamically
        self.video_dirs = {}
        base_vid_dir = self.base_dir / 'videos' / 'chunk-000'
        for topic in self.camera_topics:
            # Clean topic name for directory usage (remove leading slash, replace others with dot or underscore)
            clean_name = topic.lstrip('/').replace('/', '.')
            # If using lerobot convention, usually keys are like 'observation.images.wrist'
            # We map topic '/wrist_perspective' -> 'observation.images.wrist_perspective' (or simplified map if user wants specific names)
            # For "pass array", generic naming is safer: observation.images.<topic_suffix>
            
            # Simple heuristic: last part of topic
            key_name = clean_name.split('.')[-1] # fallback if it was dots? topic usually slashes
            key_name = topic.split('/')[-1]
            
            dir_name = f'observation.images.{key_name}'
            vid_dir = base_vid_dir / dir_name
            vid_dir.mkdir(parents=True, exist_ok=True)
            self.video_dirs[topic] = vid_dir

        # utility
        self.start_time = time.time()
        self.get_logger().info(f'Data recorder started with cameras: {self.camera_topics} 🚀')


    def timer_callback(self , msg):
        global joint_states, latest_camera_frames

        # Check wait condition
        if joint_states is None:
            return
        
        # Check if we have received frames for all cameras at least once (or current frame is not None)
        # Note: synchronizing perfectly is hard without message filters, but we take latest available
        for topic in self.camera_topics:
            if topic not in latest_camera_frames or latest_camera_frames[topic] is None:
                return

        # Record data
        for topic in self.camera_topics:
            self.camera_buffers[topic].append(latest_camera_frames[topic])
            
        self.timestamp_buffer.append(self.get_clock().now().nanoseconds / 1e9)
        self.joint_states_position_buffer.append(joint_states['position'])
        self.joint_states_velocity_buffer.append(joint_states['velocity'])
        self.joint_states_effort_buffer.append(joint_states['effort'])
        self.joint_states_name_buffer.append(joint_states['name'])
        
        self.total_steps += 1
        
        if msg.data == True:
            self.get_logger().info(f"🎬 Processing chunk at step {self.total_steps}")
            self.save_chunk()
            self.episode_idx += 1

        elif self.total_steps % 10 == 0:
            self.get_logger().info(f'Recording step {self.total_steps}...')

    def save_chunk(self):
        # 1. Save Videos
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        
        for topic in self.camera_topics:
            vid_dir = self.video_dirs[topic]
            vid_path = vid_dir / f'episode_{self.episode_idx:06d}.mp4'
            frames = self.camera_buffers[topic]
            self._write_video(str(vid_path), frames, fourcc)

        # 2. Save Parquet
        data_dict = {
            'timestamp': self.timestamp_buffer,
            'pose.name': [np.array(joint_state) for joint_state in self.joint_states_name_buffer],
            'pose.position': [np.array(joint_state) for joint_state in self.joint_states_position_buffer],
            'pose.velocity': [np.array(joint_state) for joint_state in self.joint_states_velocity_buffer],
            'pose.effort': [np.array(joint_state) for joint_state in self.joint_states_effort_buffer],
            'episode_index': [self.episode_idx] * len(self.timestamp_buffer),
            'frame_index': np.arange(0, len(self.timestamp_buffer)),
            'next.done': [False] * (len(self.timestamp_buffer) - 1) + [True]
        }
        
        df = pd.DataFrame(data_dict)
        
        parquet_path = self.log_dir / f'episode_{self.episode_idx:06d}.parquet'
        df.to_parquet(parquet_path)
        
        self.get_logger().info(f'Saved chunk {self.episode_idx} with {len(df)} frames! 🎬')
        self.get_logger().info(f'Total time taken: {(time.time() - self.start_time):.2f} seconds')

        # Clear buffers
        self.start_time = time.time()
        for topic in self.camera_topics:
            self.camera_buffers[topic].clear()
            
        self.joint_states_name_buffer.clear()
        self.joint_states_position_buffer.clear()
        self.joint_states_velocity_buffer.clear()
        self.joint_states_effort_buffer.clear()
        self.timestamp_buffer.clear()

    def _write_video(self, path, frames, fourcc):
        if not frames:
            return
        out = cv2.VideoWriter(path, fourcc, self.fps, self.size)
        for frame in frames:
            out.write(frame)
        out.release()

def main():

    rclpy.init(args=None)

    get_poses_subscriber = Get_Poses_Subscriber()
    camera_subscriber = Camera_Subscriber()
    data_recorder = Data_Recorder()

    # Adding the nodes to the executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_poses_subscriber)
    executor.add_node(camera_subscriber)
    executor.add_node(data_recorder)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        # A simple rate loop to keep main alive (or join thread)
        rate = get_poses_subscriber.create_rate(1)
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        print("Ctrl+C pressed.")
    except rclpy.exceptions.ROSInterruptException:
        print("ROS shutdown triggered — stopping loop safely.")
    finally:
        print(" 🚀 Shutting down 🚀 ")
        executor.shutdown()

        # Destroy nodes
        get_poses_subscriber.destroy_node()
        camera_subscriber.destroy_node()
        data_recorder.destroy_node()
        
        rclpy.shutdown()
        executor_thread.join()
        print("ROS nodes stopped.")
        exit()

if __name__ == '__main__':
    main()