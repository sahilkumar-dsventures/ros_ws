# Isaac Sim IK Scripts for Pi0 Policy

This directory contains OmniGraph Python scripts for computing Inverse Kinematics
from Pi0's Cartesian EE delta outputs.

## Files

| Script | Robot | Description |
|--------|-------|-------------|
| `ik_franka_omnigraph.py` | Franka Panda | Uses built-in Lula IK solver |
| `ik_so101_omnigraph.py` | SO101 | Uses Lula with custom URDF or fallback Jacobian |

## Data Flow

```
┌─────────────────┐         ┌──────────────┐         ┌────────────────┐
│   policy.py     │ ──Twist→│  OmniGraph   │──joints→│  Articulation  │
│  (ROS 2 Node)   │         │   IK Script  │         │   Controller   │
└─────────────────┘         └──────────────┘         └────────────────┘
```

## Setup in Isaac Sim

### 1. Create OmniGraph

1. Open your Isaac Sim scene
2. Window → Visual Scripting → Action Graph
3. Create new graph
4. Add nodes:
   - **ROS2 Subscribe Twist** → subscribe to `/so101/ee_delta_command` or `/franka/ee_delta_command`
   - **ROS2 Subscribe Float64** → subscribe to gripper topic
   - **Python Script** → load the IK script
   - **Articulation Controller** → apply joint positions

### 2. Configure Python Script Node

For the Python Script node:

**Inputs:**
- `robot_path` (string): Path to robot in stage, e.g., `/World/SO101`
- `linear_x`, `linear_y`, `linear_z` (float): From Twist.linear
- `angular_x`, `angular_y`, `angular_z` (float): From Twist.angular
- `gripper` (float): From Float64 gripper topic

**Outputs:**
- `joint_positions` (float[]): Arm joint positions
- `jaw_position` or `gripper_width` (float): Gripper position

### 3. Connect Nodes

```
ROS2 Subscribe Twist (linear/angular) → Python Script (linear_*/angular_*)
ROS2 Subscribe Float64 (data) → Python Script (gripper)
Python Script (joint_positions) → Articulation Controller (Position Target)
```

## Running

1. Start Pi0 server:
   ```bash
   cd /media/sarthak/a/lerobot/packages
   python pizeroServer.py
   ```

2. Run ROS 2 policy node:
   ```bash
   ros2 launch so_arm_publisher so101_policy.launch.py
   # OR
   ros2 launch so_arm_publisher franka_policy.launch.py
   ```

3. Start Isaac Sim with your configured stage

## Troubleshooting

- **"IK solution failed"**: Target pose may be unreachable. Try reducing `action_scale`
- **Slow movement**: Increase `action_scale` in the YAML config
- **No movement**: Check ROS 2 topic connections with `ros2 topic echo`
