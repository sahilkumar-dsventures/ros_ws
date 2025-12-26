import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState # Added JointState 📥
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import torch

class Pi0IsaacBridge(Node):
    def __init__(self):
        super().__init__('pi0_isaac_bridge')
        
        # 1. Configuration ⚙️
        self.task_command = "open the top drawer" # Your text input! 📝
        self.current_joint_positions = None
        self.bridge = CvBridge()
        
        # 2. Subscriptions
        # We need the robot's current pose from Isaac
        self.joint_sub = self.create_subscription(
            JointState,
            '/issac_joint_states', 
            self.joint_state_callback,
            10)
            
        self.image_sub = self.create_subscription(
            Image,
            '/wrist_perspective',
            self.image_callback,
            10)
        
        # 3. Publisher
        self.action_pub = self.create_publisher(JointTrajectory, '/joint_commands', 10)

    def joint_state_callback(self, msg):
        # Store the initial/current positions from Isaac Sim 🤖
        self.current_joint_positions = torch.tensor(msg.position).float().unsqueeze(0)

    def image_callback(self, img_msg):
        # Wait until we have at least one joint state update 🛑
        if self.current_joint_positions is None:
            self.get_logger().info("Waiting for joint states...")
            return

        # 📸 Process Image
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        pixel_values = torch.from_numpy(cv_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # 🧠 Run Inference with ALL inputs
        with torch.no_grad():
            # The model needs: Image + State + Text Task
            inputs = {
                "observation.image": pixel_values.to("cuda"),
                "observation.state": self.current_joint_positions.to("cuda"),
                "task": self.task_command 
            }
            
            output_dict = self.policy.predict_action(inputs)
        
        self.publish_action(output_dict["action"])


def main(args=None):
    rclpy.init(args=args)
    pi0_isaac_bridge = Pi0IsaacBridge()
    rclpy.spin(pi0_isaac_bridge)
    pi0_isaac_bridge.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()