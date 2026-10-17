"""Starts Gazebo with the yard, spawns the robot, bridges topics, publishes TF.
   ros2 launch box_sorter sim.launch.py            (with GUI)
   ros2 launch box_sorter sim.launch.py gui:=false (headless)"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from box_sorter import robot_gen, world_gen

WORLD_PATH = '/tmp/box_sorter_yard.sdf'


def generate_launch_description():
    pkg = get_package_share_directory('box_sorter')
    world_gen.write_world(WORLD_PATH)
    urdf = robot_gen.build_urdf()
    gui = LaunchConfiguration('gui')

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        ExecuteProcess(cmd=['gz', 'sim', '-r', WORLD_PATH], output='screen', condition=IfCondition(gui)),
        ExecuteProcess(cmd=['gz', 'sim', '-r', '-s', WORLD_PATH], output='screen', condition=UnlessCondition(gui)),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': ParameterValue(urdf, value_type=str), 'use_sim_time': True}]),
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             parameters=[{'config_file': os.path.join(pkg, 'config', 'bridge.yaml'), 'use_sim_time': True}]),
        # give gz a moment to load the world before spawning
        TimerAction(period=4.0, actions=[
            Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-world', 'yard', '-name', robot_gen.ROBOT, '-string', urdf,
                            '-x', '0', '-y', '0', '-z', '0.02']),
        ]),
    ])
