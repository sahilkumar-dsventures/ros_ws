import argparse
import sys
import os
import time
import asyncio

# Initialize SimulationApp with headless mode
# This must be run before any other omni imports
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core.utils.extensions import enable_extension

# Enable the asset converter extension explicitly
enable_extension("omni.kit.asset_converter")

import omni.kit.asset_converter as converter

def convert(input_path, output_path):
    conv_instance = converter.get_instance()
    

    # Ensure output directory exists (now that we have full path)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Starting conversion: {input_path} -> {output_path}")
    

    # task = conv_instance.create_converter_task(input_path, output_path, progress_callback, context)
    task = conv_instance.create_converter_task(input_path, output_path, None, None)
    
    # DEBUG: Inspect the task object
    print(f"DEBUG: Task dir: {dir(task)}")
    
    print("⏳ Waiting for conversion to finish...")
    # Blocking wait - checking if this avoids the loop issue
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(task.wait_until_finished())
    except Exception as e:
        print(f"⚠️ Warning during await: {e}")
    
    # Check final status
    if task.get_status(): # Assuming non-zero or truthy is done, relying on error msg
        error_msg = task.get_error_message()
        if error_msg:
             print(f"❌ Error converting asset: {error_msg}")
             success = False
        else:
             print(f"✅ Conversion complete: {output_path}")
             success = True
    else:
        # Fallback if get_status is 0/OK but finished
        print(f"✅ Conversion finished (Status OK): {output_path}")
        success = True

    return success

def main():
    parser = argparse.ArgumentParser(description="Convert assets to USD using Isaac Sim Asset Converter")
    parser.add_argument("--input", required=True, help="Path to input file (obj, fbx, urdf, etc.)")
    parser.add_argument("--output", required=True, help="Path to output USD file")
    
    # Filter out arguments that might be passed by the kit launcher (like internal flags)
    # We only care about known args
    args, unknown = parser.parse_known_args()

    input_path = str('/media/sarthak/a/ros_ws/' + args.input)
    output_path = str('/media/sarthak/a/ros_ws/' + args.output)

    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        simulation_app.close()
        sys.exit(1)

    # Run conversion
    convert(input_path, output_path)
    
    # Clean up
    simulation_app.close()

if __name__ == "__main__":
    main()