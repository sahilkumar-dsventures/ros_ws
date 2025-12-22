

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

        num_joints = len(self.joint_state.name)

        # make sure kit's editor is playing for receiving messages
        self.joint_state.position = np.array([0.0] * num_joints, dtype=np.float64).tolist()
        self.default_joints = [0, 0, 0, 0, 0, 0]

        # Limiting the movements to a smaller range (this is not the range of the robot, just the range of the movement)
        self.max_joints = np.array(self.default_joints) + 0.3
        self.min_joints = np.array(self.default_joints) - 0.3

        # position control the robot to wiggle around each joint
        self.time_start = time.time()

        timer_period = 0.05  # seconds
        self.timer1 = self.create_timer(timer_period, self.timer_callback)
        self.timer2 = self.create_timer(timer_period, self.done_callback)

        # self.get_logger().info("SO100 custom publisher started 🚀")


    def timer_callback(self):
        self.joint_state.header.stamp = self.get_clock().now().to_msg()

        joint_position = (
            np.cos(time.time() - self.time_start) * (self.max_joints - self.min_joints) * 0.5 + self.default_joints
        )
        
        self.joint_state.position = joint_position.tolist()

        # Publish the message to the topic
        self.publisher_.publish(self.joint_state)

        # self.get_logger().info(f"Joint state: {self.joint_state}")

        # self.get_logger().info(f"Joint position: {joint_position}")


    def done_callback(self):
        # biased random number
        msg = Bool()
        msg.data = False

        random_int = np.random.randint(0, 1000)

        if random_int < 995:
            self.done_publisher.publish(msg)
            return

        msg.data = True
        self.done_publisher.publish(msg)

    def destroy_node(self):
        self.timer.cancel()
        self.publisher_.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = SOArmPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


