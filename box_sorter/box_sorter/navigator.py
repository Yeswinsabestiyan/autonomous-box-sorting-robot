"""Navigation system for box_sorter robot.
Pure-Python planner with ROS node interface.
"""

import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

# Import from our own modules (no ROS imports at module level)
from . import yard as Y
from . import arm_ik

# Robot physical parameters
ROBOT_RADIUS = 0.475  # Half of chassis width (1.2m/2) + safety margin
BLADE_LOW = -0.18     # metres, blade lowered position
BLADE_RAISED = 0.0    # metres, blade raised position
WHEEL_SEPARATION = 1.0  # metres
MAX_LINEAR_VEL = 1.0  # m/s
MAX_ANGULAR_VEL = 1.5  # rad/s
MAP_RESOLUTION = 0.1  # metres per grid cell


@dataclass
class Pose:
    """Robot pose in world frame."""
    x: float
    y: float
    theta: float  # radians


@dataclass
class Goal:
    """Navigation goal."""
    x: float
    y: float
    theta: float = 0.0  # optional orientation


@dataclass
class NavStatus:
    """Navigation status."""
    state: str  # 'idle', 'moving', 'reached', 'failed'
    message: str = ""


class OccupancyGrid:
    """2D occupancy grid for path planning."""

    def __init__(self, width: int, height: int, resolution: float,
                 origin_x: float, origin_y: float):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x  # world coordinates of grid[0,0]
        self.origin_y = origin_y
        self.grid = [[0] * width for _ in range(height)]

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        gx = int(round((wx - self.origin_x) / self.resolution))
        gy = int(round((wy - self.origin_y) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        wx = gx * self.resolution + self.origin_x
        wy = gy * self.resolution + self.origin_y
        return wx, wy

    def is_valid_cell(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are within bounds."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def is_free_cell(self, gx: int, gy: int) -> bool:
        """Check if integer grid cell (gx, gy) is free."""
        if not self.is_valid_cell(gx, gy):
            return False
        return self.grid[gy][gx] == 0

    def set_occupied_cell(self, gx: int, gy: int):
        """Mark integer grid cell (gx, gy) as occupied."""
        if self.is_valid_cell(gx, gy):
            self.grid[gy][gx] = 1

    def is_free(self, x, y) -> bool:
        """Check if a position is free.
        Accepts either float world coordinates (wx, wy) or int grid coordinates (gx, gy).
        """
        if isinstance(x, float) or isinstance(y, float):
            gx, gy = self.world_to_grid(float(x), float(y))
        else:
            gx, gy = int(x), int(y)
        return self.is_free_cell(gx, gy)

    def is_free_world(self, wx: float, wy: float) -> bool:
        """Check if a world position is free."""
        gx, gy = self.world_to_grid(wx, wy)
        return self.is_free_cell(gx, gy)

    def set_occupied_world(self, wx: float, wy: float):
        """Mark world position as occupied."""
        gx, gy = self.world_to_grid(wx, wy)
        self.set_occupied_cell(gx, gy)

    def set_occupied(self, x, y):
        """Mark a cell as occupied.
        Accepts either float world coordinates (wx, wy) or int grid coordinates (gx, gy).
        """
        if isinstance(x, float) or isinstance(y, float):
            gx, gy = self.world_to_grid(float(x), float(y))
        else:
            gx, gy = int(x), int(y)
        self.set_occupied_cell(gx, gy)

    def clear(self):
        """Clear all occupancy data."""
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = 0


# ---- A* Path Planning -------------------------------------------------------
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Heuristic function for A* (Euclidean distance)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(grid: OccupancyGrid, start: Tuple[int, int],
          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* path planning algorithm on the grid."""
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if not grid.is_valid_cell(*start) or not grid.is_valid_cell(*goal):
        return None

    if not grid.is_free_cell(*start) or not grid.is_free_cell(*goal):
        return None

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        # Check neighbors (8-connected)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                neighbor = (current[0] + dx, current[1] + dy)

                if not grid.is_valid_cell(*neighbor):
                    continue

                if not grid.is_free_cell(*neighbor):
                    continue

                # Diagonal movement cost
                move_cost = math.sqrt(2) if (dx != 0 and dy != 0) else 1.0
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

    return None  # No path found


def _is_line_clear(grid: OccupancyGrid,
                   start: Tuple[int, int],
                   end: Tuple[int, int]) -> bool:
    """Check if a straight line between two grid cells is obstacle-free
    using Bresenham's line algorithm."""
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if not grid.is_free_cell(x0, y0):
            return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return True


def smooth_path(grid_path: List[Tuple[int, int]],
                grid: OccupancyGrid) -> List[Tuple[int, int]]:
    """Smooth a grid-coordinate path by shortcutting clear segments."""
    if len(grid_path) < 3:
        return list(grid_path)

    smoothed = [(int(p[0]), int(p[1])) for p in grid_path]
    changed = True
    max_iters = 10
    while changed and max_iters > 0:
        changed = False
        max_iters -= 1
        i = 0
        while i < len(smoothed) - 2:
            # Try to shortcut from i directly to the farthest reachable point
            for j in range(len(smoothed) - 1, i + 1, -1):
                if _is_line_clear(grid, smoothed[i], smoothed[j]):
                    # Remove intermediate points
                    smoothed = smoothed[:i + 1] + smoothed[j:]
                    changed = True
                    break
            i += 1

    return smoothed


class Navigator:
    """Main navigation controller."""

    def __init__(self):
        self.pose = Pose(0.0, 0.0, 0.0)  # Start at origin
        self.goal: Optional[Goal] = None
        self.status = NavStatus('idle')
        self.path: List[Tuple[float, float]] = []  # World coordinate path
        self.current_waypoint_index = 0

        # Initialize occupancy grid
        grid_border = 1  # extra cells on each side
        self.map_origin_x = -Y.HALF_X - grid_border * MAP_RESOLUTION
        self.map_origin_y = -Y.HALF_Y - grid_border * MAP_RESOLUTION
        self.map_width = int(2 * Y.HALF_X / MAP_RESOLUTION) + 2 * grid_border
        self.map_height = int(2 * Y.HALF_Y / MAP_RESOLUTION) + 2 * grid_border
        self.grid = OccupancyGrid(
            self.map_width, self.map_height,
            MAP_RESOLUTION, self.map_origin_x, self.map_origin_y
        )

        # Build static obstacles from yard data
        self._build_static_obstacles()

    def _inflate_rect(self, cx: float, cy: float,
                      half_w: float, half_h: float):
        """Mark all grid cells within an axis-aligned rectangle as occupied."""
        gx_min, gy_min = self.grid.world_to_grid(cx - half_w, cy - half_h)
        gx_max, gy_max = self.grid.world_to_grid(cx + half_w, cy + half_h)
        for gy in range(max(0, gy_min), min(self.grid.height, gy_max + 1)):
            for gx in range(max(0, gx_min), min(self.grid.width, gx_max + 1)):
                self.grid.set_occupied_cell(gx, gy)

    def _inflate_circle(self, cx: float, cy: float, radius: float):
        """Mark all grid cells within a circle as occupied."""
        gx_min, gy_min = self.grid.world_to_grid(cx - radius, cy - radius)
        gx_max, gy_max = self.grid.world_to_grid(cx + radius, cy + radius)
        for gy in range(max(0, gy_min), min(self.grid.height, gy_max + 1)):
            for gx in range(max(0, gx_min), min(self.grid.width, gx_max + 1)):
                wx, wy = self.grid.grid_to_world(gx, gy)
                if math.hypot(wx - cx, wy - cy) <= radius:
                    self.grid.set_occupied_cell(gx, gy)

    def _build_static_obstacles(self):
        """Build occupancy grid from static yard obstacles (walls, bins, poles)."""
        self.grid.clear()
        infl = ROBOT_RADIUS  # inflation radius

        # Walls — mark cells that are outside the yard boundary
        for gy in range(self.grid.height):
            for gx in range(self.grid.width):
                wx, wy = self.grid.grid_to_world(gx, gy)
                if (wx <= -Y.HALF_X + infl or wx >= Y.HALF_X - infl or
                        wy <= -Y.HALF_Y + infl or wy >= Y.HALF_Y - infl):
                    self.grid.set_occupied_cell(gx, gy)

        # Bins — each is a square of side `bsize`, inflate by robot radius
        for bin_name, (bx, by, bsize) in Y.BINS.items():
            half = bsize / 2 + infl
            self._inflate_rect(bx, by, half, half)

        # Poles — cylinders, inflate by generous clearance so robot (with blade/carried box) never touches poles
        pole_infl = max(infl, 1.05)
        for px, py in Y.POLES:
            self._inflate_circle(px, py, Y.POLE_R + pole_infl)

    def update_dynamic_obstacles(self,
                                 box_positions: List[Tuple[float, float, str]]):
        """Rebuild grid with current box positions as extra obstacles."""
        self._build_static_obstacles()

        for bx, by, box_name in box_positions:
            if bx < -Y.HALF_X + 2.0:
                continue  # Already in bin column on left wall
            if math.hypot(bx - self.pose.x, by - self.pose.y) < 1.45:
                continue  # Target box being grasped/carried next to robot
            box_radius = 0.40
            self._inflate_circle(bx, by, box_radius + ROBOT_RADIUS)

    def set_goal(self, goal: Goal):
        """Set a new navigation goal and plan a path."""
        self.goal = goal
        self.status = NavStatus('idle')
        self._plan_path()

    def _plan_path(self):
        """Plan a path to the current goal using A*."""
        if not self.goal:
            self.status = NavStatus('failed', 'No goal set')
            return

        start = self.grid.world_to_grid(self.pose.x, self.pose.y)
        goal = self.grid.world_to_grid(self.goal.x, self.goal.y)

        if not self.grid.is_free_cell(*start):
            start = self._find_nearest_free(start)
            if start is None:
                self.status = NavStatus('failed', 'Start position inside obstacle')
                self.path = []
                return

        if not self.grid.is_free_cell(*goal):
            goal = self._find_nearest_free(goal)
            if goal is None:
                self.status = NavStatus('failed', 'Goal inside obstacle, no nearby free cell')
                self.path = []
                return

        grid_path = astar(self.grid, start, goal)

        if grid_path is None:
            self.status = NavStatus('failed', 'No path found to goal')
            self.path = []
            return

        grid_path = smooth_path(grid_path, self.grid)

        self.path = []
        for gx, gy in grid_path:
            wx, wy = self.grid.grid_to_world(gx, gy)
            self.path.append((wx, wy))

        self.current_waypoint_index = 0
        if self.path:
            self.status = NavStatus(
                'moving', f'Path planned with {len(self.path)} waypoints')
        else:
            self.status = NavStatus('failed', 'Empty path')

    def _find_nearest_free(self, cell: Tuple[int, int],
                           max_radius: int = 25) -> Optional[Tuple[int, int]]:
        """BFS outward from cell to find the nearest free grid cell."""
        cell = (int(cell[0]), int(cell[1]))
        visited = {cell}
        queue = deque([cell])
        while queue:
            cx, cy = queue.popleft()
            if self.grid.is_free_cell(cx, cy):
                return (cx, cy)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) not in visited and self.grid.is_valid_cell(nx, ny):
                        dist = max(abs(nx - cell[0]), abs(ny - cell[1]))
                        if dist <= max_radius:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        return None

    def update(self, dt: float = 0.05) -> Tuple[float, float, str]:
        """
        Update navigation and return velocity commands.
        Returns: (linear_vel, angular_vel, blade_command)
        """
        if not self.goal or not self.path:
            return 0.0, 0.0, 'hold'

        if self.current_waypoint_index >= len(self.path):
            self.status = NavStatus('reached', 'Goal reached')
            return 0.0, 0.0, 'hold'

        target_x, target_y = self.path[self.current_waypoint_index]

        dx = target_x - self.pose.x
        dy = target_y - self.pose.y
        distance = math.hypot(dx, dy)

        is_final = (self.current_waypoint_index == len(self.path) - 1)
        waypoint_tolerance = 0.10 if is_final else 0.18
        if distance < waypoint_tolerance:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.path):
                self.status = NavStatus('reached', 'Goal reached')
                return 0.0, 0.0, 'hold'
            target_x, target_y = self.path[self.current_waypoint_index]
            dx = target_x - self.pose.x
            dy = target_y - self.pose.y
            distance = math.hypot(dx, dy)

        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.pose.theta
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        if abs(angle_diff) > 0.28:
            linear_vel = 0.0
            angular_vel = max(-MAX_ANGULAR_VEL,
                              min(MAX_ANGULAR_VEL, angle_diff * 2.2))
        else:
            align_factor = max(0.25, math.cos(angle_diff))
            linear_vel = min(MAX_LINEAR_VEL, distance * 1.6) * align_factor
            linear_vel = max(0.0, linear_vel)
            angular_vel = max(-MAX_ANGULAR_VEL,
                              min(MAX_ANGULAR_VEL, angle_diff * 3.0))

        return linear_vel, angular_vel, 'hold'

    def get_status(self) -> NavStatus:
        """Get current navigation status."""
        return self.status

    def update_robot_pose(self, x: float, y: float, theta: float):
        """Update the robot's estimated pose (called from ROS odometry)."""
        self.pose = Pose(x, y, theta)


# ---- Utility functions -------------------------------------------------------

def blade_helper(current_state: str, target_state: str) -> str:
    if current_state == target_state:
        return 'hold'
    elif target_state == 'lower':
        return 'lower'
    elif target_state == 'raise':
        return 'raise'
    else:
        return 'hold'


def push_box_in_line(start_x: float, start_y: float,
                     target_x: float, target_y: float
                     ) -> List[Tuple[float, float]]:
    dx = target_x - start_x
    dy = target_y - start_y
    distance = math.hypot(dx, dy)

    if distance < 0.01:
        return [(start_x, start_y)]

    num_waypoints = max(2, int(distance / 0.1))
    waypoints = []

    for i in range(num_waypoints + 1):
        t = i / num_waypoints
        wx = start_x + t * dx
        wy = start_y + t * dy
        waypoints.append((wx, wy))

    return waypoints


# ---- ROS Node ---------------------------------------------------------------
def main(args=None):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Twist
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    class NavigatorNode(Node):
        def __init__(self):
            super().__init__('navigator')
            self.nav = Navigator()
            self._has_pose = False

            # Subscriptions
            self.create_subscription(
                PoseStamped, '/nav/goal', self.on_goal, 10)
            self.create_subscription(
                TFMessage, '/world_poses', self.on_poses, 10)
            self.create_subscription(
                TFMessage, '/tf', self.on_tf, 10)

            # Publishers
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.status_pub = self.create_publisher(String, '/nav/status', 10)
            from nav_msgs.msg import Path
            self.path_pub = self.create_publisher(Path, '/nav/path', 10)

            # Control loop timer (20 Hz)
            self.timer = self.create_timer(0.05, self.control_loop)
            self.get_logger().info('Navigator node initialized')

        def on_goal(self, msg):
            gx = msg.pose.position.x
            gy = msg.pose.position.y

            if not self._has_pose:
                self.get_logger().warn(
                    'Rejecting nav goal — no robot pose received yet')
                self.nav.status = NavStatus('failed', 'No robot pose yet')
                self.publish_status()
                return

            if not self.nav.grid.is_free_world(gx, gy):
                self.get_logger().warn(
                    f'Goal ({gx:.2f}, {gy:.2f}) inside obstacle, '
                    f'finding nearest free cell...')
                cell = self.nav.grid.world_to_grid(gx, gy)
                free = self.nav._find_nearest_free(cell)
                if free is None:
                    self.get_logger().error(
                        f'No free cell near goal ({gx:.2f}, {gy:.2f})')
                    self.nav.status = NavStatus(
                        'failed', 'Goal inside obstacle')
                    self.publish_status()
                    return
                gx, gy = self.nav.grid.grid_to_world(*free)
                self.get_logger().info(
                    f'Adjusted goal to ({gx:.2f}, {gy:.2f})')

            self.get_logger().info(
                f'Received new goal: ({gx:.2f}, {gy:.2f})')
            self.nav.set_goal(Goal(gx, gy))
            self.publish_status()

        def on_tf(self, msg):
            """Extract robot pose from /tf."""
            from . import robot_gen
            for tf in msg.transforms:
                name = tf.child_frame_id
                if name in ('base_link', '/base_link', robot_gen.ROBOT, '/' + robot_gen.ROBOT):
                    tx = tf.transform.translation.x
                    ty = tf.transform.translation.y
                    rot = tf.transform.rotation
                    siny = 2.0 * (rot.w * rot.z + rot.x * rot.y)
                    cosy = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
                    yaw = math.atan2(siny, cosy)
                    self.nav.update_robot_pose(tx, ty, yaw)
                    self._has_pose = True

        def on_poses(self, msg):
            """Extract box positions from /world_poses."""
            from . import robot_gen
            box_positions = []
            for tf in msg.transforms:
                name = tf.child_frame_id
                tx = tf.transform.translation.x
                ty = tf.transform.translation.y
                if name in ('base_link', '/base_link', robot_gen.ROBOT, '/' + robot_gen.ROBOT):
                    rot = tf.transform.rotation
                    siny = 2.0 * (rot.w * rot.z + rot.x * rot.y)
                    cosy = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
                    yaw = math.atan2(siny, cosy)
                    self.nav.update_robot_pose(tx, ty, yaw)
                    self._has_pose = True
                elif name.startswith('box_') and '::' not in name:
                    box_positions.append((tx, ty, name))

            self.nav.update_dynamic_obstacles(box_positions)

        def control_loop(self):
            v, w, _blade_cmd = self.nav.update(0.05)  # 20 Hz
            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(w)
            self.cmd_vel_pub.publish(twist)
            self.publish_status()

        def publish_status(self):
            status = self.nav.get_status()
            self.status_pub.publish(String(data=status.state))
            self.publish_path()

        def publish_path(self):
            if hasattr(self, 'path_pub') and getattr(self.nav, 'path', None):
                try:
                    from nav_msgs.msg import Path
                    from geometry_msgs.msg import PoseStamped
                    msg = Path()
                    msg.header.frame_id = 'world'
                    if hasattr(self, 'get_clock'):
                        msg.header.stamp = self.get_clock().now().to_msg()
                    for wx, wy in self.nav.path:
                        p = PoseStamped()
                        p.header.frame_id = 'world'
                        p.pose.position.x = float(wx)
                        p.pose.position.y = float(wy)
                        p.pose.position.z = 0.05
                        p.pose.orientation.w = 1.0
                        msg.poses.append(p)
                    self.path_pub.publish(msg)
                except Exception:
                    pass

    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()