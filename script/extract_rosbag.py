import rosbag2_py
from cv_bridge import CvBridge
import logging
import cv2
import json
import os

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri='rosbag2_2025_11_19-16_39_22', storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions('', '')
reader.open(storage_options, converter_options)

bridge = CvBridge()

i = 0
while reader.has_next():
    print(f'It\'s row no {i}')
    (topic, data, t) = reader.read_next()
    

    
    if topic == "/wrist_perspective":
        print('Image Topic got wrist')
        img = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        cv2.imwrite(f"dataset/wrist/{i:06d}.png", img)


    
    if topic == "/env_perspective":
        print('Image Topic got env')
        print(i)
        img = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        cv2.imwrite(f"dataset/wrist/{i:06d}.png", img)

    # if topic == "/odom":
    #     pose = {
    #         "position": {
    #             "x": data.pose.pose.position.x,
    #             "y": data.pose.pose.position.y,
    #             "z": data.pose.pose.position.z,
    #         },
    #         "orientation": {
    #             "x": data.pose.pose.orientation.x,
    #             "y": data.pose.pose.orientation.y,
    #             "z": data.pose.pose.orientation.z,
    #             "w": data.pose.pose.orientation.w,
    #         },
    #         "timestamp": t
    #     }
    #     with open(f"dataset/pose/{i:06d}.json", "w") as f:
    #         json.dump(pose, f, indent=2)

    i += 1
