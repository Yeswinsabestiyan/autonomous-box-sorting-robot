"""Part D tests: Navigation system.
Run:  python3 -m unittest test.test_part_d -v"""
import math, os, sys, unittest
import xml.etree.ElementTree as ET
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from box_sorter import navigator as Nav, yard as Y


class TestNavigator(unittest.TestCase):
    def setUp(self):
        self.nav = Nav.Navigator()

    def test_initial_state(self):
        self.assertEqual(self.nav.get_status().state, 'idle')
        self.assertEqual(self.nav.pose.x, 0.0)
        self.assertEqual(self.nav.pose.y, 0.0)
        self.assertEqual(self.nav.pose.theta, 0.0)

    def test_set_goal_idle_status(self):
        goal = Nav.Goal(5.0, 5.0)
        self.nav.set_goal(goal)
        # Should start planning, so status might be idle or moving depending on timing
        # But goal should be set
        self.assertIsNotNone(self.nav.goal)
        self.assertEqual(self.nav.goal.x, 5.0)
        self.assertEqual(self.nav.goal.y, 5.0)

    def test_occupancy_grid_bounds(self):
        # Test that grid covers the yard area
        grid = self.nav.grid
        # Yard is 20x12 metres, origin at centre
        # So x from -10 to 10, y from -6 to 6
        # With border for inflation

        # Test a point inside yard
        self.assertTrue(grid.is_free(0.0, 0.0))  # Centre should be free

        # Test a point in wall (should be occupied)
        self.assertFalse(grid.is_free(-10.5, 0.0))  # Outside left wall

        # Test a point in bin area (should be occupied)
        bin_x, bin_y, bin_size = Y.BINS['red']
        self.assertFalse(grid.is_free(bin_x, bin_y))  # Inside bin should be occupied

    def test_astar_simple_path(self):
        grid = self.nav.grid
        # Test path planning in free space - vertical line up the middle (should be clear of poles)
        start_world = (0.0, -5.0)
        goal_world = (0.0, 5.0)
        start = grid.world_to_grid(*start_world)
        goal = grid.world_to_grid(*goal_world)

        path = Nav.astar(grid, start, goal)
        self.assertIsNotNone(path, "Should find a path around poles")
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)

    def test_astar_blocked_path(self):
        grid = self.nav.grid
        # Block a cell and verify path finding handles it
        start = grid.world_to_grid(1.0, 1.0)
        goal = grid.world_to_grid(2.0, 2.0)

        # Block the direct path
        gx, gy = grid.world_to_grid(1.5, 1.5)
        if grid.is_valid_cell(gx, gy):
            grid.set_occupied(*grid.grid_to_world(gx, gy))

        # Should still find a path around (or None if completely blocked)
        path = Nav.astar(grid, start, goal)
        # Either finds a path or returns None if blocked - both are valid outcomes
        # For this simple test, we just verify it doesn't crash

    def test_heuristic_function(self):
        self.assertEqual(Nav.heuristic((0, 0), (3, 4)), 5.0)  # 3-4-5 triangle
        self.assertEqual(Nav.heuristic((1, 1), (1, 1)), 0.0)
        self.assertEqual(Nav.heuristic((0, 0), (0, 5)), 5.0)

    def test_smooth_path_no_change_needed(self):
        # Test smoothing on a straight line (should remain straight)
        path = [(0, 0), (1, 0), (2, 0), (3, 0)]
        grid = self.nav.grid
        smoothed = Nav.smooth_path(path, grid)
        # Should be similar or same length
        self.assertGreaterEqual(len(smoothed), 2)
        self.assertLessEqual(len(smoothed), len(path))

    def test_world_to_grid_conversion(self):
        grid = self.nav.grid
        # Test conversion round-trip
        wx, wy = 0.0, 0.0
        gx, gy = grid.world_to_grid(wx, wy)
        wx2, wy2 = grid.grid_to_world(gx, gy)
        # Should be close (within resolution)
        self.assertAlmostEqual(wx, wx2, delta=grid.resolution)
        self.assertAlmostEqual(wy, wy2, delta=grid.resolution)

    def test_is_free_function(self):
        grid = self.nav.grid
        # Test free space detection
        self.assertTrue(grid.is_free(0.0, 0.0))  # Centre should be free
        # Test out of bounds (should return False)
        self.assertFalse(grid.is_free(100.0, 100.0))  # Far outside map

    def test_blade_helper(self):
        self.assertEqual(Nav.blade_helper('raise', 'raise'), 'hold')
        self.assertEqual(Nav.blade_helper('raise', 'lower'), 'lower')
        self.assertEqual(Nav.blade_helper('lower', 'raise'), 'raise')
        self.assertEqual(Nav.blade_helper('lower', 'lower'), 'hold')

    def test_push_box_in_line(self):
        waypoints = Nav.push_box_in_line(0.0, 0.0, 3.0, 4.0)
        self.assertGreaterEqual(len(waypoints), 2)
        self.assertAlmostEqual(waypoints[0][0], 0.0, places=5)
        self.assertAlmostEqual(waypoints[0][1], 0.0, places=5)
        self.assertAlmostEqual(waypoints[-1][0], 3.0, places=5)
        self.assertAlmostEqual(waypoints[-1][1], 4.0, places=5)

        # Check that points lie on the line
        for wx, wy in waypoints:
            # For line from (0,0) to (3,4), expected ratio y/x = 4/3 (when x!=0)
            if abs(wx) > 1e-6:
                self.assertAlmostEqual(wy/wx, 4.0/3.0, places=3)
            elif abs(wx) < 1e-6:  # Near x=0
                self.assertAlmostEqual(wx, 0.0, places=5)


class TestNavigatorUpdate(unittest.TestCase):
    def test_update_with_no_goal(self):
        nav = Nav.Navigator()
        lin_vel, ang_vel, blade_cmd = nav.update(0.1)
        self.assertEqual(lin_vel, 0.0)
        self.assertEqual(ang_vel, 0.0)
        self.assertEqual(blade_cmd, 'hold')

    def test_update_with_goal_no_path(self):
        nav = Nav.Navigator()
        goal = Nav.Goal(5.0, 5.0)
        nav.set_goal(goal)
        # Manually clear path to simulate planning failure
        nav.path = []
        lin_vel, ang_vel, blade_cmd = nav.update(0.1)
        self.assertEqual(lin_vel, 0.0)
        self.assertEqual(ang_vel, 0.0)
        self.assertEqual(blade_cmd, 'hold')

    def test_update_reached_goal(self):
        nav = Nav.Navigator()
        # Set up a scenario where we're already at the goal
        nav.pose = Nav.Pose(5.0, 5.0, 0.0)
        goal = Nav.Goal(5.0, 5.0)
        nav.set_goal(goal)
        # Manually set a short path
        nav.path = [(5.0, 5.0)]
        nav.current_waypoint_index = 0

        lin_vel, ang_vel, blade_cmd = nav.update(0.1)
        # Should be stopping or nearly stopped
        self.assertLessEqual(abs(lin_vel), 0.1)
        self.assertLessEqual(abs(ang_vel), 0.1)


if __name__ == '__main__':
    unittest.main()