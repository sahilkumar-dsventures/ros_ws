# SO-100 Arm ROS 2 Workspace

This workspace is designed for controlling the **SO-100 robotic arm** within **NVIDIA Isaac Sim** using **ROS 2 Humble**. It includes tools for teleoperation and a comprehensive data collection system formatted for imitation learning (compatible with frameworks like LeRobot).

---

## 📂 Project Structure

- **`src/`**: ROS 2 packages.
  - **`so_arm_publisher`**: Contains nodes for controlling the arm.
    - `teleop.py`: Keyboard-based control for individual joints.
    - `movement.py`: Logic for automated or scripted movements.
  - **`so_arm_datacollector`**: A robust data collection node.
    - `main.py`: Subscribes to joint states and multiple camera feeds, saving them as synchronized videos (`.mp4`) and metadata (`.parquet`).
- **`packages/`**:
  - **`environment_setup`**: Contains `main.py` which initializes the Isaac Sim environment, loads the robot USD, and enables necessary ROS 2 bridges.
- **`script/`**: Bash scripts for automating workflows.
  - `simulation.sh`: Launches Isaac Sim with the SO-100 arm.
  - `so_arm_teleop.sh`: Launches the keyboard teleoperation node.
  - `data_recorder.sh`: Starts the data collection process.
  - `omni_run.sh`: Utility script to run Python scripts using Isaac Sim's standalone Python environment.
- **`utils/`**: Helper tools for asset conversion and visualization.

---

## 🛠 Prerequisites

- **ROS 2 Humble**
- **NVIDIA Isaac Sim** (installed at `/media/sarthak/a/isaac_sim/` or as configured in scripts)
- **Python 3.10+**
- **Dependencies**: `opencv-python`, `pandas`, `pyarrow`, `cv_bridge`, `rclpy`.

---

## 🚀 Getting Started

### 1. Environment Setup
Ensure you have sourced your ROS 2 environment:
```bash
source /opt/ros/humble/setup.bash
```

### 2. Build the Workspace
From the root of `ros_ws`:
```bash
colcon build
source install/setup.bash
```

### 3. Running the Simulation
Execute the simulation script to start Isaac Sim and load the robot:
```bash
./script/simulation.sh
```

### 4. Teleoperation
Control the arm using your keyboard:
```bash
./script/so_arm_teleop.sh
```
**Keyboard Controls:**
- `1/q`: Joint 1 (Rotation) +/-
- `2/w`: Joint 2 (Pitch) +/-
- `3/e`: Joint 3 (Elbow) +/-
- `4/r`: Joint 4 (Wrist Pitch) +/-
- `5/t`: Joint 5 (Wrist Roll) +/-
- `6/y`: Joint 6 (Jaw) +/-
- `Space`: Mark episode as "Done" (for data collection)
- `x`: Exit teleop

---

## 📊 Data Collection

The `so_arm_datacollector` package captures high-quality datasets for training imitation learning models.

### Recorded Observations:
- **Joint States**: Position, velocity, and effort for all 6 joints.
- **Visuals**: Synchronized feeds from:
  - `/wrist_perspective`
  - `/env_perspective`
  - `/camera_array`

### Output Format:
Data is saved in the directory specified in `so_arm_datacollector/main.py` (default: `/media/sarthak/a/ros_ws/lerobot/...`):
- **Videos**: `videos/chunk-000/observation.images.<camera_name>/episode_XXXXXX.mp4`
- **Metadata**: `data/chunk-000/episode_XXXXXX.parquet`

---

## 📜 Scripts Reference

| Script | Description |
| :--- | :--- |
| `simulation.sh` | Starts Isaac Sim with the SO-100 arm model and ROS 2 Bridge. |
| `so_arm_teleop.sh` | Runs the keyboard teleoperation node to command `joint_command`. |
| `data_recorder.sh` | Starts the node that records joint states and camera frames. |
| `omni_run.sh` | Internal helper to run Python scripts within the Isaac Sim Python context. |
| `rerun_viz.sh` | (Optional) Visualization helper using Rerun. |

---

## 🔧 Configuration
- **USD Path**: The robot model path is defined in `packages/environment_setup/main.py`.
- **Data Directory**: Change the `data_dir` parameter in `src/so_arm_datacollector/so_arm_datacollector/main.py` to your preferred storage location.
