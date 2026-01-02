# Initialize SimulationApp
# It is important to initialize this before other omni imports
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.kit.app
import omni.timeline
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, UsdGeom
import numpy as np
import time

# Define the USD path variable
USD_PATH = "/media/sarthak/a/Experiments/franka.usd"
SIMULATION_PLAY = True

# Franka joint names
FRANKA_JOINT_NAMES = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
    'panda_joint5', 'panda_joint6', 'panda_joint7',
    'panda_finger_joint1', 'panda_finger_joint2'
]


def setup_ros2_omnigraph(franka_prim_path="/franka"):
    """
    Set up ROS 2 Joint State Publisher using OmniGraph.
    Uses Isaac Sim's native ROS 2 bridge - no external rclpy needed.
    """
    try:
        keys = og.Controller.Keys
        
        # Use the isaacsim.ros2.bridge node types (Isaac Sim 2024+)
        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": "/ActionGraph/ROS2Bridge", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ROS2Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                ],
                keys.SET_VALUES: [
                    ("ROS2Context.inputs:useDomainIDEnvVar", True),
                    ("PublishJointState.inputs:topicName", "/isaac_joint_states"),
                    ("PublishJointState.inputs:targetPrim", [franka_prim_path]),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
                ],
            }
        )
        print("✅ Created ROS 2 Joint State Publisher -> /isaac_joint_states")
        return True
        
    except Exception as e:
        print(f"⚠️ OmniGraph ROS 2 setup failed: {e}")
        print("\n📌 Alternative: Add ROS 2 components directly in your USD file:")
        print("   1. Open your franka.usd in Isaac Sim GUI")
        print("   2. Create -> Isaac -> ROS2 -> JointState Publisher")
        print("   3. Configure it to publish /isaac_joint_states")
        print("   4. Save the USD file")
        return False


def main():
    app = omni.kit.app.get_app()
    ext_mgr = app.get_extension_manager()
    
    # Enable ROS 2 bridge extensions
    print("Enabling ROS 2 bridge extensions...")
    ext_mgr.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

    # Let extensions initialize
    for _ in range(5):
        simulation_app.update()

    # Load USD
    print(f"Loading USD: {USD_PATH}")
    usd_context = omni.usd.get_context()
    usd_context.open_stage(USD_PATH)

    # Wait for stage to load
    for _ in range(5):
        simulation_app.update()
    
    # Create World
    world = World(stage_units_in_meters=1.0)
    simulation_app.update()
    
    # Get Franka robot from stage
    franka_prim_path = "/franka"
    
    try:
        franka = world.scene.add(
            Articulation(prim_path=franka_prim_path, name="franka")
        )
        print(f"✅ Found Franka at {franka_prim_path}")
    except Exception as e:
        print(f"❌ Could not find Franka at {franka_prim_path}: {e}")
        stage = usd_context.get_stage()
        print("\nAvailable root prims:")
        for prim in stage.GetPseudoRoot().GetChildren():
            print(f"  - {prim.GetPath()}")
        simulation_app.close()
        return
    
    # Reset world to initialize articulation
    world.reset()
    
    # Try to set up ROS 2 OmniGraph (may fail if bridge not fully loaded)
    setup_ros2_omnigraph(franka_prim_path)

    if SIMULATION_PLAY:
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
    else:
        print("Simulation is not playing")

    print("")
    print("=" * 60)
    print("🤖 Franka Isaac Sim Environment Running")
    print("=" * 60)
    print("")
    print("If ROS 2 bridge is working, topics should be available:")
    print("  /isaac_joint_states (JointState)")
    print("")
    print("To verify: ros2 topic list")
    print("")
    print("=" * 60)
    print("Press Ctrl+C to exit")
    print("")

    # Main simulation loop
    while simulation_app.is_running():
        simulation_app.update()
        world.step(render=True)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
