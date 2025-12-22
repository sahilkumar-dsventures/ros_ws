
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():

    pkg_share = get_package_share_directory('so_arm_datacollector')
    config_file = os.path.join(pkg_share, 'config', 'collector.yaml')

    return LaunchDescription([
        Node(
            package='so_arm_datacollector',
            executable='so_arm_datacollector',
            # name='data_recorder',
            parameters=[config_file]
        )
    ])