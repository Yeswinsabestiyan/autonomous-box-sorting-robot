"""Launch file for Autonomous Box Sorter in RViz.
Runs complete 3D simulation with robot kinematics, live box feeder,
A* navigation, 3D autonomous sorting brain, and RViz 2 visualization.
Zero Gazebo overhead or dependencies.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from box_sorter import robot_gen


def generate_launch_description():
    pkg = get_package_share_directory('box_sorter')
    urdf = robot_gen.build_urdf()
    rviz_config = os.path.join(pkg, 'rviz', 'box_sorter.rviz')

    spawn_period = LaunchConfiguration('spawn_period')
    open_rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        # Ensure gz subprocesses are bypassed
        SetEnvironmentVariable('SIM_MODE', 'rviz'),

        DeclareLaunchArgument('spawn_period', default_value='3.5',
                              description='Period in seconds between box arrivals'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Launch RViz 2 visualization window'),

        LogInfo(msg='🤖 Launching Autonomous Robotic Box Sorter in RViz Simulation...'),

        # 1. Robot State Publisher (Publishes 3D URDF Robot Tree to /tf)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': ParameterValue(urdf, value_type=str),
                         'use_sim_time': False}]),

        # 2. RViz Simulator Node (Diff-Drive Kinematics, Arm Controllers, 3D Box Management, Yard Markers)
        Node(
            package='box_sorter',
            executable='rviz_sim_node',
            name='rviz_sim_node',
            output='screen',
            parameters=[{'use_sim_time': False}]),

        # 3. Box Spawner (Feeder: Spawns grippable boxes on conveyors)
        Node(
            package='box_sorter',
            executable='box_spawner',
            name='box_spawner',
            output='screen',
            parameters=[{
                'sim_mode': 'rviz',
                'p_big': 0.0,
                'p_heavy': 0.0,
                'p_damaged': 0.10,
                'spawn_period': spawn_period,
                'use_sim_time': False
            }]),

        # 4. Navigator (Pure-Python A* Planner & Obstacle Avoidance)
        Node(
            package='box_sorter',
            executable='navigator',
            name='navigator',
            output='screen',
            parameters=[{'use_sim_time': False}]),

        # 5. Sorter Brain (Autonomous 3D Grasp & Sorting State Machine)
        Node(
            package='box_sorter',
            executable='sorter_brain',
            name='sorter_brain',
            output='screen',
            parameters=[{'use_sim_time': False}]),

        # 6. RViz 2 (Visual 3D Workspace, Robot, Bins, Conveyors, Boxes, and Live HUD)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            condition=IfCondition(open_rviz)),
    ])
