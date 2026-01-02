import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('so_arm_publisher')
    
    # Path to the config file
    config_file = os.path.join(pkg_dir, 'config', 'franka_param.yaml')
    
    return LaunchDescription([
        Node(
            package='so_arm_publisher',
            executable='franka_policy',
            output='screen',
            parameters=[config_file],
        )
    ])
