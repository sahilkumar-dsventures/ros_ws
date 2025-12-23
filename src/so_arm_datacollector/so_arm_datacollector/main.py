import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState , Image
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
wrist_camera = None
env_camera = None

class Get_Poses_Subscriber(Node):
   def __init__(self):
       super().__init__('get_poses_subscriber')
       self.declare_parameter('topic_name', '/poses')

       self.topic_name = self.get_parameter('topic_name').value
       self.subscription = self.create_subscription(JointState, self.topic_name, self.listener_callback, 10)

   def listener_callback(self, msg : JointState):
        # print(f'I heard: "{msg.position}"')

        global joint_states
       
        # self.get_logger().info(f'Here is the topic name: {self.topic_name}')
        joint_states = {
            'name': msg.name,
            'position': list(msg.position),
            'velocity': list(msg.velocity),
            'effort': list(msg.effort)
        }

class Wrist_Camera_Subscriber(Node):
    def __init__(self):
        super().__init__('wrist_camera_subscriber')
        self.subscription = self.create_subscription(Image, 'wrist_perspective', self.listener_callback, 10)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        # self.wrist_camera = Image()
        # self.wrist_camera.header.frame_id = 'wrist_camera'
        # self.wrist_camera.header.stamp = self.get_clock().now().to_msg()
        # self.wrist_camera.height = 480
        # self.wrist_camera.width = 640
        # self.wrist_camera.encoding = 'bgr8'
        # self.wrist_camera.is_bigendian = False
    
    def listener_callback(self, msg : Image):

        global wrist_camera
        img = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

        resized_image = cv2.resize(img, (self.width, self.height), cv2.INTER_LINEAR)
        
        # Ensure 3 channels
        if len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        elif len(resized_image.shape) == 3 and resized_image.shape[2] == 1:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
            
        wrist_camera = resized_image


class Env_Camera_Subscriber(Node):
    def __init__(self):
        super().__init__('env_camera_subscriber')
        self.subscription = self.create_subscription(Image, 'env_perspective', self.listener_callback, 10)

        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value

    def listener_callback(self, msg : Image):
        global env_camera
        img = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        resized_image = cv2.resize(img, (self.width, self.height), cv2.INTER_LINEAR)
        
        # Ensure 3 channels
        if len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        elif len(resized_image.shape) == 3 and resized_image.shape[2] == 1:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
            
        env_camera = resized_image


        # self.get_logger().info(f'I heard: "{msg.data}"')


class Data_Recorder(Node):

    def __init__(self):
        super().__init__('data_recorder')

        self.declare_parameter('data_dir', '/media/sarthak/a/ros_ws/lerobot/SO-ARM101_MoveIt_IsaacSim/')
        self.declare_parameter('fps', 30)
        self.declare_parameter('num_joint', 6)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('is_done_topic', '/is_done')

        self.data_dir = self.get_parameter('data_dir').value
        self.fps = self.get_parameter('fps').value
        self.num_joint = self.get_parameter('num_joint').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.is_done_topic = self.get_parameter('is_done_topic').value

        self.create_subscription(Bool, self.is_done_topic, self.timer_callback, 10)
        # self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)



        self.total_steps = 0
        self.episode_idx = 0
        
        # Buffers
        self.wrist_camera_frames = []
        self.env_camera_frames = []
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

        # Video directories
        base_vid_dir = self.base_dir / 'videos' / 'chunk-000'
        self.wrist_vid_dir = base_vid_dir / 'observation.images.wrist'
        self.env_vid_dir = base_vid_dir / 'observation.images.env'

        self.wrist_vid_dir.mkdir(parents=True, exist_ok=True)
        self.env_vid_dir.mkdir(parents=True, exist_ok=True)

        # utility
        self.start_time = time.time()
        self.get_logger().info(f'Data recorder started 🚀')


    def timer_callback(self , msg):
        global joint_states, wrist_camera, env_camera


        # # Skip if no images received yet
        if wrist_camera is None or env_camera is None or joint_states is None:
            return

        self.wrist_camera_frames.append(wrist_camera)
        self.env_camera_frames.append(env_camera)
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
        wrist_vid_path = self.wrist_vid_dir / f'episode_{self.episode_idx:06d}.mp4'
        env_vid_path = self.env_vid_dir / f'episode_{self.episode_idx:06d}.mp4'

        self._write_video(str(wrist_vid_path), self.wrist_camera_frames, fourcc)
        self._write_video(str(env_vid_path), self.env_camera_frames, fourcc)


        # 2. Save Parquet
        df = pd.DataFrame({
            'timestamp': self.timestamp_buffer,
            'pose.name': [np.array(joint_state) for joint_state in self.joint_states_name_buffer],
            'pose.position': [np.array(joint_state) for joint_state in self.joint_states_position_buffer],
            'pose.velocity': [np.array(joint_state) for joint_state in self.joint_states_velocity_buffer],
            'pose.effort': [np.array(joint_state) for joint_state in self.joint_states_effort_buffer],
            'episode_index': [self.episode_idx] * len(self.timestamp_buffer),
            'frame_index': np.arange(0, len(self.timestamp_buffer)),
            'next.done': [False] * (len(self.timestamp_buffer) - 1) + [True]
        })
        
        parquet_path = self.log_dir / f'episode_{self.episode_idx:06d}.parquet'
        
        # DEBUG: Check if data exists before saving
        # self.get_logger().info("\n--- DataFrame Content Check ---")
        # # self.get_logger().info(df[["pose.position", "pose.name", "pose.velocity" , "pose.effort" , "episode_index", "next.done"]].head(10))
        # self.get_logger().info("--------------------------------------\n")
        
        df.to_parquet(parquet_path)
        
        self.get_logger().info(f'Saved chunk {self.episode_idx} with {len(df)} frames! 🎬')

        self.get_logger().info(f'Total time taken: {(time.time() - self.start_time):.2f} seconds')

        # Clear buffers
        self.start_time = time.time()
        self.wrist_camera_frames.clear()
        self.env_camera_frames.clear()
        self.joint_states_name_buffer.clear()
        self.joint_states_position_buffer.clear()
        self.joint_states_velocity_buffer.clear()
        self.joint_states_effort_buffer.clear()
        self.timestamp_buffer.clear()

    def _write_video(self, path, frames, fourcc):
        out = cv2.VideoWriter(path, fourcc, self.fps, self.size)
        for frame in frames:
            out.write(frame)
        out.release()

def main():

    rclpy.init(args=None)

    get_poses_subscriber = Get_Poses_Subscriber()
    wrist_camera_subscriber = Wrist_Camera_Subscriber()
    env_camera_subscriber = Env_Camera_Subscriber()
    data_recorder = Data_Recorder()

    # Adding the nodes to the executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_poses_subscriber)
    executor.add_node(wrist_camera_subscriber)
    executor.add_node(env_camera_subscriber)
    executor.add_node(data_recorder)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
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
        wrist_camera_subscriber.destroy_node()
        env_camera_subscriber.destroy_node()
        data_recorder.destroy_node()
        
        rclpy.shutdown()
        executor_thread.join()
        print("ROS nodes stopped.")
        exit()

if __name__ == '__main__':
    main()