import argparse
import rerun as rr
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import os
import sys

def visualize_dataset(data_dir, episode_idx=None):
    base_path = Path(data_dir)
    data_path = base_path / "data" / "chunk-000"
    videos_path = base_path / "videos" / "chunk-000"

    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist.")
        print("Please check your --data-dir argument.")
        return

    # Find parquet files
    parquet_files = sorted(list(data_path.glob("episode_*.parquet")))
    
    if not parquet_files:
        print(f"No parquet files found in {data_path}")
        return

    print(f"Found {len(parquet_files)} episodes.")
    
    

    for pq_file in parquet_files:
        # Extract episode index from filename "episode_XXXXXX.parquet"
        try:
            current_ep_idx = int(pq_file.stem.split('_')[1])
        except (IndexError, ValueError):
            print(f"Skipping malformed filename: {pq_file.name}")
            continue
        
        if episode_idx is not None and current_ep_idx != episode_idx:
            continue
            
        print(f"Visualizing Episode {current_ep_idx}...")

         # Initialize a new recording for this episode
        rr.init("Lerobot_Dataset_Viz", recording_id=f"episode_{current_ep_idx:06d}" , spawn=True)
        
        # Load Parquet
        try:
            df = pd.read_parquet(pq_file)
        except Exception as e:
            print(f"Failed to read {pq_file}: {e}")
            continue
        
        # Load Video Paths
        wrist_vid_path = videos_path / "observation.images.wrist" / f"episode_{current_ep_idx:06d}.mp4"
        env_vid_path = videos_path / "observation.images.env" / f"episode_{current_ep_idx:06d}.mp4"
        
        if not wrist_vid_path.exists():
            print(f"Warning: Video not found {wrist_vid_path}")
        if not env_vid_path.exists():
            print(f"Warning: Video not found {env_vid_path}")

        # Open Video Captures
        cap_wrist = cv2.VideoCapture(str(wrist_vid_path))
        cap_env = cv2.VideoCapture(str(env_vid_path))
        
        # Get unique timestamps to iterate time steps
        if 'timestamp' not in df.columns:
            print(f"Error: 'timestamp' column missing in {pq_file}")
            continue
            
        unique_timestamps = sorted(df['timestamp'].unique())
        if not unique_timestamps:
            continue
            
        start_time = unique_timestamps[0]
        
        print(f"Processing {len(unique_timestamps)} frames...")

        for i, ts in enumerate(unique_timestamps):
            # Normalize time to start at 0
            rel_time = ts - start_time

            # Use rr.set_time instead of deprecated set_time_seconds
            rr.set_time("stable_time", duration=rel_time)
            rr.set_time("frame_idx", sequence=i)
            
            # --- Log Joint Data ---
            # Get data for this timestamp
            step_data = df[df['timestamp'] == ts]

            for _, row in step_data.iterrows():
                p_names = row['pose.name']
                p_pos = row['pose.position']
                p_vel = row['pose.velocity']
                p_eff = row['pose.effort']
                
                # Check if the data is iterable (e.g. numpy array) and not just a string
                if isinstance(p_names, (np.ndarray, list)):
                    # Iterate through the arrays
                    for name, pos, vel, eff in zip(p_names, p_pos, p_vel, p_eff):
                        # Clean up name if it's a byte string or numpy Scalars
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                        name = str(name).strip()
                        
                        rr.log(f"joint_position/{name}", rr.Scalars(pos))
                        rr.log(f"joint_velocity/{name}", rr.Scalars(vel))
                        rr.log(f"joint_effort/{name}", rr.Scalars(eff))
                else:
                    # Scalars case
                    rr.log(f"joint_position/{p_names}", rr.Scalars(p_pos))
                    rr.log(f"joint_velocity/{p_names}", rr.Scalars(p_vel))
                    rr.log(f"joint_effort/{p_names}", rr.Scalars(p_eff))
            
            # --- Log Video Frames ---
            # Read next frame
            ret_w, frame_w = cap_wrist.read()
            ret_e, frame_e = cap_env.read()
            
            if ret_w:
                # Convert BGR to RGB
                frame_w = cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB)
                rr.log("camera/wrist", rr.Image(frame_w))
            
            if ret_e:
                frame_e = cv2.cvtColor(frame_e, cv2.COLOR_BGR2RGB)
                rr.log("camera/env", rr.Image(frame_e))
                
        cap_wrist.release()
        cap_env.release()
        
    print("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Lerobot Dataset with Rerun")
    parser.add_argument("--data-dir", type=str, default="/media/sarthak/a/ros_ws/lerobot/SO-ARM101_MoveIt_IsaacSim/", help="Path to dataset root")
    parser.add_argument("--episode-index", type=int, default=None, help="Specific episode index to visualize")
    
    args = parser.parse_args()
    
    visualize_dataset(args.data_dir, args.episode_index)
