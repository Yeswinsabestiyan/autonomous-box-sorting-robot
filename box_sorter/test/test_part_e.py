"""Part E tests: Sorter brain state machine.
Run:  python3 -m unittest test.test_part_e -v"""
import json
import math
import os
import sys
import unittest
from unittest.mock import Mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from box_sorter import sorter_brain as SB
from box_sorter import rules


class TestSorterBrain(unittest.TestCase):
    def setUp(self):
        self.brain = SB.SorterBrain.__new__(SB.SorterBrain)
        # Manually initialize the attributes we need for testing
        self.brain.state = SB.State.IDLE
        self.brain.box_queue = []
        self.brain.current_box = None
        self.brain.action = None
        self.brain.retries = 0
        self.brain.robot_pose = None
        self.brain.robot_pose_valid = True
        # Mock publishers and subscribers
        self.brain.box_ready_sub = Mock()
        self.brain.nav_status_sub = Mock()
        self.brain.world_poses_sub = Mock()
        self.brain.nav_goal_pub = Mock()
        self.brain.turret_pub = Mock()
        self.brain.shoulder_pub = Mock()
        self.brain.elbow_pub = Mock()
        self.brain.wrist_pub = Mock()
        self.brain.finger_pub = Mock()
        self.brain.blade_pub = Mock()
        self.brain.timer = Mock()

    def test_initial_state(self):
        self.assertEqual(self.brain.state, SB.State.IDLE)

    def test_handle_idle_no_boxes(self):
        self.brain.handle_idle()
        self.assertEqual(self.brain.state, SB.State.IDLE)

    def test_handle_idle_with_boxes(self):
        # Add a box to queue
        self.brain.box_queue = [{'size': 0.5, 'mass': 1.0, 'color': 'red', 'damaged': False, 'x': 0.0, 'y': 0.0, 'z': 0.0}]
        self.brain.robot_pose = (0.0, 0.0, 0.0, 0.0)  # fake pose
        self.brain.handle_idle()
        self.assertEqual(self.brain.state, SB.State.SELECT_BOX)

    def test_handle_select_box(self):
        box = {'size': 0.5, 'mass': 1.0, 'color': 'red', 'damaged': False, 'x': 1.0, 'y': 2.0, 'z': 0.0}
        self.brain.box_queue = [box]
        self.brain.handle_select_box()
        self.assertEqual(self.brain.current_box, box)
        self.assertEqual(len(self.brain.box_queue), 0)
        self.assertEqual(self.brain.state, SB.State.DECIDE)

    def test_handle_decide_grip(self):
        self.brain.current_box = {'size': 0.5, 'mass': 1.0, 'color': 'red', 'damaged': False}
        self.brain.handle_decide()
        self.assertIsInstance(self.brain.action, SB.Action)
        self.assertEqual(self.brain.action.mode, 'grip')
        self.assertEqual(self.brain.action.bin, 'red')
        self.assertEqual(self.brain.state, SB.State.NAVIGATE_TO_STANDOFF)

    def test_handle_decide_push_big(self):
        self.brain.current_box = {'size': 1.5, 'mass': 1.0, 'color': 'red', 'damaged': False}  # size > GRIP_MAX
        self.brain.handle_decide()
        self.assertEqual(self.brain.action.mode, 'push')
        self.assertEqual(self.brain.action.bin, 'big')
        self.assertEqual(self.brain.state, SB.State.NAVIGATE_TO_STANDOFF)

    def test_handle_decide_push_heavy(self):
        self.brain.current_box = {'size': 0.5, 'mass': 15.0, 'color': 'red', 'damaged': False}  # mass > MAX_GRIP_KG
        self.brain.handle_decide()
        self.assertEqual(self.brain.action.mode, 'push')
        self.assertEqual(self.brain.action.bin, 'big')
        self.assertEqual(self.brain.state, SB.State.NAVIGATE_TO_STANDOFF)

    def test_handle_decide_damaged(self):
        self.brain.current_box = {'size': 0.5, 'mass': 1.0, 'color': 'red', 'damaged': True}
        self.brain.handle_decide()
        self.assertEqual(self.brain.action.mode, 'grip')
        self.assertEqual(self.brain.action.bin, 'scrap')
        self.assertEqual(self.brain.state, SB.State.NAVIGATE_TO_STANDOFF)

    def test_distance_to_robot(self):
        self.brain.robot_pose = (0.0, 0.0, 0.0, 0.0)
        box = {'x': 3.0, 'y': 4.0, 'z': 0.0}
        dist = self.brain.distance_to_robot(box)
        self.assertAlmostEqual(dist, 5.0)

    def test_box_ready_callback(self):
        # Test that callback adds to queue and sorts
        self.brain.robot_pose = (0.0, 0.0, 0.0, 0.0)
        box1 = {'size': 0.5, 'mass': 1.0, 'color': 'red', 'damaged': False, 'x': 5.0, 'y': 0.0, 'z': 0.0, 'name': 'box1', 'input': 0}
        box2 = {'size': 0.5, 'mass': 1.0, 'color': 'green', 'damaged': False, 'x': 1.0, 'y': 0.0, 'z': 0.0, 'name': 'box2', 'input': 1}
        msg1 = Mock()
        msg1.data = json.dumps(box1)
        msg2 = Mock()
        msg2.data = json.dumps(box2)
        self.brain.box_ready_callback(msg1)
        self.brain.box_ready_callback(msg2)
        # Should have two boxes
        self.assertEqual(len(self.brain.box_queue), 2)
        # Should be sorted by distance (box2 is closer)
        self.assertEqual(self.brain.box_queue[0]['name'], 'box2')
        self.assertEqual(self.brain.box_queue[1]['name'], 'box1')


if __name__ == '__main__':
    unittest.main()