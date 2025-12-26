

import rclpy
from rclpy.node import Node
import numpy as np
import time
from sensor_msgs.msg import JointState
from std_msgs.msg import String , Bool

# (Example publisher — replace with whatever SO100 data you want!)
class SOArmPublisher(Node):
    def __init__(self):
        super().__init__('so_arm_publisher')
        
        # Publish to your custom topic
        self.publisher_ = self.create_publisher(JointState, 'joint_command', 10)
        self.done_publisher = self.create_publisher(Bool , 'is_done' , 10)

        self.joint_state = JointState()

        self.joint_state.name = [
            'Rotation',
            'Pitch',
            'Elbow',
            'Wrist_Pitch',
            'Wrist_Roll',
            'Jaw'
        ]

        # Define keyframes for pick and place cycle
        # [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
        self.keyframes = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 0: Home
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],      # 1: Over Pick
            [0.3, 0.3, 0.3, 0.0, 0.0, 0.0],      # 2: Pick Down (Jaw open)
            [0.3, 0.3, 0.3, 0.0, 0.0, 0.5],      # 3: Close Jaw
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.5],      # 4: Lift
            [-0.3, 0.0, 0.0, 0.0, 0.0, 0.5],     # 5: Over Place
            [-0.3, 0.3, 0.3, 0.0, 0.0, 0.5],     # 6: Place Down
            [-0.3, 0.3, 0.3, 0.0, 0.0, 0.0],     # 7: Open Jaw
            [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0],     # 8: Lift
        ]

        self.current_keyframe_idx = 0
        self.next_keyframe_idx = 1
        self.interpolation_factor = 0.0
        self.interpolation_step = 0.05  # Adjust this for speed (0.05 means 20 steps between keyframes)

        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.done_timer = self.create_timer(timer_period, self.done_callback)

        self.get_logger().info("SO100 Pick and Place movement started 🚀")

    def timer_callback(self):
        self.joint_state.header.stamp = self.get_clock().now().to_msg()

        start_pose = np.array(self.keyframes[self.current_keyframe_idx])
        end_pose = np.array(self.keyframes[self.next_keyframe_idx])
        
        # Linear interpolation between keyframes
        current_pose = (1 - self.interpolation_factor) * start_pose + self.interpolation_factor * end_pose
        
        self.joint_state.position = current_pose.tolist()

        # Publish the message to the topic
        self.publisher_.publish(self.joint_state)

        # Advance interpolation
        self.interpolation_factor += self.interpolation_step
        if self.interpolation_factor >= 1.0:
            self.interpolation_factor = 0.0
            self.current_keyframe_idx = self.next_keyframe_idx
            self.next_keyframe_idx = (self.next_keyframe_idx + 1) % len(self.keyframes)

    def done_callback(self):
        msg = Bool()
        # Publish True if we just finished a full cycle (returned to home)
        if self.current_keyframe_idx == 0 and self.interpolation_factor < self.interpolation_step:
            msg.data = True
        else:
            msg.data = False
        
        self.done_publisher.publish(msg)

    def destroy_node(self):
        self.timer.cancel()
        self.done_timer.cancel()
        self.publisher_.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = SOArmPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


