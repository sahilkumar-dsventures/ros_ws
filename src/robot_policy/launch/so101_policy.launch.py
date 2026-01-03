import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Launch generalized policy node with SO101 configuration."""
    pkg_dir = get_package_share_directory('robot_policy')
    config_file = os.path.join(pkg_dir, 'config', 'so101_param.yaml')
    
    return LaunchDescription([
        Node(
            package='robot_policy',
            executable='policy',
            name='policy_node',
            output='screen',
            parameters=[config_file],
        )
    ])
