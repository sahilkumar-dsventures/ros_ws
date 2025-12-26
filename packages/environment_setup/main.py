import argparse
import os
import sys

# Initialize SimulationApp
# It is important to initialize this before other omni imports
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.kit.app
import omni.timeline

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

    # 3️Load USD
    usd_context = omni.usd.get_context()
    usd_context.open_stage(USD_PATH)

    # Wait for stage
    simulation_app.update()

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
