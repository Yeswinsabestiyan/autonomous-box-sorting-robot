# Autonomous Box Sorting Robot 

An autonomous robotic box sorting system built with **ROS 2 (Humble)**. The system features a custom mobile manipulator robot equipped with a differential-drive base, push blade, 4-DOF robotic arm, and parallel gripper with linear guide rails. It autonomously detects, navigates to, classifies, grips, transports, and dispatches incoming warehouse boxes into designated sorting bins in an enclosed factory yard.

![ROS 2](https://img.shields.io/badge/ROS%202-Humble-blue)
![Python](https://img.shields.io/badge/Python-3.10-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Tests-81%20Passed-brightgreen)

---

## 🌟 Key Features

- **Kinematics & Arm IK**: Closed-form analytical inverse kinematics (`arm_ik.py`) with automatic wrist pitch compensation to keep the gripper vertical.
- **Realistic Parallel Gripper**: Heavy-duty transverse linear guide rail beam with polished runner tracks and sliding carriage brackets (`robot_gen.py`).
- **Autonomous Decision Engine**: Sorting rules state machine (`rules.py`, `sorter_brain.py`) that classifies boxes based on dimensions, mass, color (Red, Green, Blue, Other), and damage status.
- **A* Navigation & Obstacle Avoidance**: Pure-Python grid-based A* path planner (`navigator.py`) with dynamic obstacle repulsion and standoff approach calculations.
- **Interactive 3D RViz Simulation (`rviz_sim_node.py`)**:
  - Live diff-drive kinematics and rotating wheel visualization.
  - Animated incoming conveyor belts feeding boxes from outer wall openings.
  - Animated outfeed conveyor chutes conveying sorted boxes through the wall into dispatch.
  - Live 3D box length dimension labels (e.g., `0.40m`).
  - Bin labels showing real-time dispatched counts (`RED | Dispatched: X`).
  - Live overhead floating robot status label (`BOT: NAVIGATING`, `BOT: CLAMPING BOX`, `BOT: TRANSPORTING TO RED BIN`).
  - Realistic gripper lock LED and targeting beams.
  - Enclosed factory perimeter safety barriers with hazard trim and corner bollards.
  - Physics collision handling (boxes push aside or stack realistically).
- **Gazebo Simulation Support**: Complete SDF world and model generation (`world_gen.py`, `gz_utils.py`) with ROS-Gz bridges.

---

## 🏗️ System Architecture

```
box_sorter/
├── box_sorter/
│   ├── arm_ik.py          # Analytical inverse and forward kinematics
│   ├── rules.py           # Classification rules (Grip/Push to designated bins)
│   ├── yard.py            # Single source of truth for warehouse geometry
│   ├── robot_gen.py       # Algorithmic URDF generation (diff-drive + 4-DOF arm)
│   ├── world_gen.py       # SDF world generation for Gazebo
│   ├── gz_utils.py        # Gazebo subprocess and SDF generation helpers
│   ├── box_spawner.py     # Procedural box arrival feeder
│   ├── navigator.py       # A* path planning with obstacle clearance
│   ├── sorter_brain.py    # Autonomous finite state machine
│   └── rviz_sim_node.py   # Complete 3D RViz factory physics & animation simulator
├── launch/
│   ├── rviz_sim.launch.py # Lightweight RViz 3D factory simulation (Recommended)
│   ├── part_f.launch.py   # Full Gazebo + ROS 2 integration launch
│   └── sim.launch.py      # Base Gazebo simulation
├── test/                  # Comprehensive test suite (81 unit tests across Parts A-E)
├── urdf/                  # Checked-in robot URDF
└── rviz/                  # RViz display configuration
```

---

## 🚀 Getting Started

### Prerequisites

- **Ubuntu 22.04 LTS** (or WSL 2 on Windows)
- **ROS 2 Humble Hawksbill**
- `python3-colcon-common-extensions`

### Build the Workspace

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select box_sorter
source install/setup.bash
```

### Run the RViz Factory Simulation

To launch the autonomous robot in the interactive 3D RViz simulation:

```bash
ros2 launch box_sorter rviz_sim.launch.py
```

### Run Full Gazebo Integration

To launch with Gazebo Sim:

```bash
ros2 launch box_sorter part_f.launch.py
```

---

## 🧪 Testing

The repository includes a comprehensive 81-test unit suite:

```bash
cd box_sorter
python3 test/test_part_a.py  # Rules, IK roundtrips, yard geometry
python3 test/test_part_b.py  # URDF structure, joint limits, kinematics
python3 test/test_part_c.py  # Procedural box spawning, conveyor delays
python3 test/test_part_d.py  # A* navigator, path planning, obstacle avoidance
python3 test/test_part_e.py  # Sorter brain autonomous decision state machine
```

---

## 📜 License

This project is licensed under the MIT License.
