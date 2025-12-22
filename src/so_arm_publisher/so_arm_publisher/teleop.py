import sys
import termios
import tty
import select
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

STEP = 0.05  # radians per key press

JOINT_NAMES = [
    'Rotation',
    'Pitch',
    'Elbow',
    'Wrist_Pitch',
    'Wrist_Roll',
    'Jaw'
]

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # Use select to check if input is ready
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            ch = sys.stdin.read(1)
            print(ch)
            return ch
        else:
            return None  # No key pressed
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)



class Teleop(Node):
    def __init__(self):
        super().__init__("so_arm_teleop")
        self.joint_state = JointState()
        self.joint_state.name = JOINT_NAMES
        self.TIME_PERIOD = 0.05
        self.STEP = 0.05
        self.joint_state.position = [0.0] * len(JOINT_NAMES)
        self.publisher_ = self.create_publisher(JointState, "joint_command", 10)
        self.is_done = self.create_publisher(Bool , 'is_done' , 10)
        self.timer = self.create_timer(self.TIME_PERIOD, self.timer_callback)
        self.timer1 = self.create_timer(self.TIME_PERIOD, self.done_callback)


    def timer_callback(self):
        key = get_key()
        print("You pressed ", key)

        if key == '1': self.joint_state.position[0] += self.STEP
        elif key == 'q': self.joint_state.position[0] -= self.STEP
        elif key == '2': self.joint_state.position[1] += self.STEP
        elif key == 'w': self.joint_state.position[1] -= self.STEP
        elif key == '3': self.joint_state.position[2] += self.STEP
        elif key == 'e': self.joint_state.position[2] -= self.STEP
        elif key == '4': self.joint_state.position[3] += self.STEP
        elif key == 'r': self.joint_state.position[3] -= self.STEP
        elif key == '5': self.joint_state.position[4] += self.STEP
        elif key == 't': self.joint_state.position[4] -= self.STEP
        elif key == '6': self.joint_state.position[5] += self.STEP
        elif key == 'y': self.joint_state.position[5] -= self.STEP

        elif key == 'x':
            print("Exiting teleop 👋")
            self.destroy_node()
            return
    
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.joint_state)

    def done_callback(self):
        msg = Bool()
        key = get_key()
        if key == ' ':
            msg.data = True
            self.is_done.publish(msg)
            return
        else:
            msg.data = False
            self.is_done.publish(msg)

        
    def destroy_node(self):
        self.timer.cancel()
        self.publisher_.destroy()

def main(args=None):
    rclpy.init(args=args)
    teleop = Teleop()
    rclpy.spin(teleop)
    teleop.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
