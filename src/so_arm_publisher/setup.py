from setuptools import find_packages, setup

package_name = 'so_arm_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/policy_launch.py']),
        ('share/' + package_name + '/config', ['config/policy_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dhruv2',
    maintainer_email='dhruv2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'so_arm_movement = so_arm_publisher.movement:main',
            'so_arm_teleop = so_arm_publisher.teleop:main',
            'so_arm_policy = so_arm_publisher.policy:main',
        ],
    },
)
