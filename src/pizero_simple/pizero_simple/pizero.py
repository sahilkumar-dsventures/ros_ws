import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import torch
import numpy as np
from transformers import AutoTokenizer

# Assuming you saved the previous code in a file named `modeling_pi0.py`
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

class Pi0IsaacBridge(Node):
    def __init__(self):
        super().__init__('pi0_isaac_bridge')
        
        # 1. Load the fine-tuned Pi0 model 🤖
        self.get_logger().info("Loading Pi0 model... 🧠")
        pretrained_path = "lerobot/pi0_libero_finetuned"
        self.policy = PI0Policy.from_pretrained(pretrained_path)
        self.policy.eval()
        self.policy.to("cuda") # Use GPU for speed! ⚡

        # 1.5 Load Tokenizer for the Text Instruction 📝
        # PI0 uses the PaliGemma tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.device = "cuda"
        
        # Store the task text (Fixed for now, can be made a ROS parameter)
        self.task_description = "open the drawer"
        self.get_logger().info(f"Current Task: {self.task_description}")
        
        # 2. ROS 2 Setup
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/wrist_perspective', 
            self.image_callback,
            10)
            
        # ⚠️ CRITICAL ADDITION: Subscribe to robot state
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states', # Standard ROS topic for robot joints
            self.joint_state_callback,
            10)
            
        self.latest_joint_state = None # Store latest state here

        # Publishers
        self.action_pub = self.create_publisher(
            JointTrajectory, 
            '/isaac_joint_states', 
            10)

    def joint_state_callback(self, msg):
        """Cache the latest joint positions from the robot."""
        # Ensure we just get the positions (you might need to filter for specific joints)
        self.latest_joint_state = torch.tensor(msg.position, dtype=torch.float32)

    def image_callback(self, msg):
        # 🛡️ Safety check: Don't run if we haven't heard from the robot yet
        if self.latest_joint_state is None:
            self.get_logger().warn("Waiting for joint states...", throttle_duration_sec=2.0)
            return

        # 📸 Step A: Convert ROS Image to Torch
        cv_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        pixel_values = self.preprocess_image(cv_img)
        
        # 🦾 Step B: Prepare Robot State
        # Add batch dimension [1, state_dim]
        state_tensor = self.latest_joint_state.unsqueeze(0).to(self.device)

        # 📝 Step C: Tokenize Text
        text_batch = self.tokenizer(
            self.task_description, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.policy.config.text_max_length # Use config limit
        ).to(self.device)

        # 🧠 Step D: Run Inference
        # We construct the batch dictionary exactly as LeRobot expects it
        batch = {
            "observation.images.wrist_image": pixel_values.to(self.device), # Check your policy config for exact key name!
            "observation.state": state_tensor,
            "observation.language_instruction_tokens": text_batch["input_ids"],
            "observation.language_instruction_mask": text_batch["attention_mask"],
        }

        with torch.no_grad():
            # Use select_action! It handles the chunking/queueing logic internally.
            action = self.policy.select_action(batch)
        
        # 🚀 Step E: Send the Action Back
        self.publish_action(action)

    def preprocess_image(self, img):
        # Resize to 224x224 if needed (PaliGemma standard)
        # Note: PI0 expects standard 0-1 float, its internal preprocessor handles normalization
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor.unsqueeze(0)

    def publish_action(self, action_tensor):
        # Convert model output to a ROS message Isaac can read
        msg = JointTrajectory()
        point = JointTrajectoryPoint()
        
        # action_tensor is [action_dim] because select_action returns a single step
        point.positions = action_tensor.cpu().numpy().tolist()
        msg.points.append(point)
        
        self.action_pub.publish(msg)
        # self.get_logger().info(f"Action sent: {point.positions[:3]}...") # Debug log

def main(args=None):
    rclpy.init(args=args)
    node = Pi0IsaacBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()