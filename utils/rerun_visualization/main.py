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
    
    # Discover available video sources dynamically
    # Look for directories in videos_path
    if videos_path.exists():
        video_source_dirs = [d for d in videos_path.iterdir() if d.is_dir()]
        print(f"Discovered {len(video_source_dirs)} camera sources: {[d.name for d in video_source_dirs]}")
    else:
        print(f"Warning: Video path {videos_path} does not exist.")
        video_source_dirs = []

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
            pq_mtime = pq_file.stat().st_mtime
        except Exception as e:
            print(f"Failed to read {pq_file}: {e}")
            continue
        
        # Open Video Captures for all sources
        caps = {}
        for source_dir in video_source_dirs:
            # Source dir name example: 'observation.images.wrist'
            # We can extract a simpler name like 'wrist' or keep full name
            camera_name = source_dir.name.replace('observation.images.', '')
            
            vid_path = source_dir / f"episode_{current_ep_idx:06d}.mp4"
            if vid_path.exists():
                # Check for staleness: Video and Parquet should be modified around the same time
                vid_mtime = vid_path.stat().st_mtime
                if abs(vid_mtime - pq_mtime) > 120:  # 2 minute tolerance
                    # print(f"Skipping stale video {vid_path} (diff: {abs(vid_mtime - pq_mtime):.1f}s)")
                    continue

                cap = cv2.VideoCapture(str(vid_path))
                if cap.isOpened():
                    caps[camera_name] = cap
                else:
                    print(f"Warning: Could not open video {vid_path}")
            # else:
            #     print(f"Warning: Video not found {vid_path}")
        
        # Get unique timestamps to iterate time steps
        if 'timestamp' not in df.columns:
            print(f"Error: 'timestamp' column missing in {pq_file}")
            # Release any caps if we skip
            for cap in caps.values(): cap.release()
            continue
            
        unique_timestamps = sorted(df['timestamp'].unique())
        if not unique_timestamps:
            for cap in caps.values(): cap.release()
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
            # Read next frame for each camera
            for cam_name, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rr.log(f"camera/{cam_name}", rr.Image(frame))
                else:
                    # Might indicate video is shorter than parquet data or dropped frames
                    pass

        # Cleanup
        for cap in caps.values():
            cap.release()
        
    print("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Lerobot Dataset with Rerun")
    parser.add_argument("--data-dir", type=str, default="/media/sarthak/a/ros_ws/lerobot/SO-ARM101_MoveIt_IsaacSim/", help="Path to dataset root")
    parser.add_argument("--episode-index", type=int, default=None, help="Specific episode index to visualize")
    
    args = parser.parse_args()
    
    visualize_dataset(args.data_dir, args.episode_index)
