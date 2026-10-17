"""Launch file for Part F: Full system integration for box sorting.
Starts Gazebo, robot, bridge, box spawner, navigator, and brain.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, LogInfo
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
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Set to "true" for Gazebo 3D GUI, or "false" for headless'),
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation (Gazebo) clock'),

        LogInfo(msg=['Starting Gazebo world: ', WORLD_PATH]),

        # Gazebo simulation (headless if gui:=false, GUI window if gui:=true)
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', '-s', WORLD_PATH],
            output='screen',
            condition=UnlessCondition(gui)),
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', WORLD_PATH],
            output='screen',
            condition=IfCondition(gui)),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': ParameterValue(urdf, value_type=str),
                         'use_sim_time': use_sim_time}]),

        # ROS-Gazebo bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            parameters=[{'config_file': os.path.join(pkg, 'config', 'bridge.yaml'),
                         'use_sim_time': use_sim_time}]),

        # Spawn the robot into the yard (7s delay ensures Gazebo GUI has initialized the world in WSL)
        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    output='screen',
                    arguments=['-world', 'yard', '-name', robot_gen.ROBOT, '-string', urdf,
                               '-x', '0', '-y', '0', '-z', '0.05']),
            ]),

        # Box spawner (spawns boxes at input conveyors)
        TimerAction(
            period=9.0,
            actions=[
                Node(
                    package='box_sorter',
                    executable='box_spawner',
                    name='box_spawner',
                    output='screen',
                    parameters=[{'use_sim_time': use_sim_time,
                                 'spawn_period': 4.0}]),
            ]),

        # Navigator (Part D: A* navigation & obstacle avoidance)
        TimerAction(
            period=11.0,
            actions=[
                Node(
                    package='box_sorter',
                    executable='navigator',
                    name='navigator',
                    output='screen',
                    parameters=[{'use_sim_time': use_sim_time}]),
            ]),

        # Brain (Part E: Autonomous decision state machine)
        TimerAction(
            period=13.0,
            actions=[
                Node(
                    package='box_sorter',
                    executable='sorter_brain',
                    name='sorter_brain',
                    output='screen',
                    parameters=[{'use_sim_time': use_sim_time}]),
            ]),
    ])