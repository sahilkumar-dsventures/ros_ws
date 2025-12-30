# Initialize SimulationApp
# It is important to initialize this before other omni imports
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.kit.app
import omni.timeline
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
import numpy as np

# Define the USD path variable
USD_PATH = "/media/sarthak/a/Experiments/so_100_arm.usd"
SIMULATION_PLAY = True

def main():
    app = omni.kit.app.get_app()
    ext_mgr = app.get_extension_manager()
    
    ext_mgr.set_extension_enabled("isaacsim.ros2.bridge", True)

    ext_mgr.set_extension_enabled("isaacsim.ros2.sim_control", True)

    # Let extensions initialize
    simulation_app.update()
    simulation_app.update()

    # Load USD
    usd_context = omni.usd.get_context()
    usd_context.open_stage(USD_PATH)

    # Wait for stage
    simulation_app.update()

    # Load Cup and Ball from Isaac Sim assets
    assets_root_path = get_assets_root_path()
    if assets_root_path:
        # Load Cup (YCB Mug)
        cup_path = assets_root_path + "/Isaac/Props/YCB/Axis_Aligned/025_mug.usd"
        add_reference_to_stage(usd_path=cup_path, prim_path="/World/Cup")
        cup = XFormPrim(prim_path="/World/Cup", name="cup")
        cup.set_world_pose(position=np.array([0.5, 0.0, 0.1]))
        
        # Load Plate (YCB Plate)
        plate_path = assets_root_path + "/Isaac/Props/YCB/Axis_Aligned/029_plate.usd"
        add_reference_to_stage(usd_path=plate_path, prim_path="/World/Plate")
        plate = XFormPrim(prim_path="/World/Plate", name="plate")
        plate.set_world_pose(position=np.array([0.5, 0.2, 0.05]))
    else:
        print("Could not find Isaac Sim assets folder")

    if SIMULATION_PLAY:
        # Start simulation (physics + /clock)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
    else:
        print("Simulation is not playing")

    print("Simulation is running")
    print("Press Ctrl+C to exit")

    # Keep app running
    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()

if __name__ == "__main__":
    main()
