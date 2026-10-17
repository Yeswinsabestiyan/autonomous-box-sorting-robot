#!/usr/bin/env python3
"""
Box sorter brain: state machine for picking up boxes with full 3D pose (x, y, z)
and 3D dimension (width, depth, height) awareness and placing them into their
designated bins in RViz / Gazebo.
"""
import json
import math
import time
from enum import Enum, auto

# Import our own modules (no ROS imports at module level)
from . import rules
from .rules import Action
from . import arm_ik
from . import robot_gen
from . import yard as Y
from . import gz_utils
from . import navigator as Nav


class State(Enum):
    IDLE = auto()
    SELECT_BOX = auto()
    DECIDE = auto()
    NAVIGATE_TO_STANDOFF = auto()
    NAVIGATE_TO_STANDOFF_WAIT = auto()
    GRIP_YAW = auto()
    GRIP_LOWER = auto()
    GRIP_LOWER_WAIT = auto()
    GRIP_IK = auto()
    GRIP_IK_WAIT = auto()
    GRIP_CLOSE = auto()
    GRIP_CLOSE_WAIT = auto()
    GRIP_LIFT = auto()
    GRIP_LIFT_WAIT = auto()
    NAVIGATE_TO_BIN = auto()
    NAVIGATE_TO_BIN_WAIT = auto()
    GRIP_RELEASE = auto()
    GRIP_POSITION_BIN_WAIT = auto()
    GRIP_RELEASE_WAIT = auto()
    RETRACT = auto()
    SCRAP = auto()
    FAILED = auto()


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_box_grasp_profile(box: dict) -> dict:
    """
    Given a box dict with 3D coordinates (x, y, z) and dimensions (size, width, depth, height),
    compute the exact geometric parameters needed to approach, open, lower, grip, and lift it.

    Geometry reference (from robot_gen.py & arm_ik.py):
      - Wrist link origin is the palm point (r, z) targeted by arm_ik.solve(r, z).
      - Palm thickness is PALM[2] = 0.05 m -> bottom of palm is at (palm_z - 0.05).
      - Fingers have height FINGER[2] = 0.20 m below palm -> finger tips are at (palm_z - 0.25).
      - Finger inner gap for command q in [0.0, 0.5] is: gap = 0.10 + 2 * q.
    """
    size = float(box.get('size', 0.4))
    width = float(box.get('width', size))
    depth = float(box.get('depth', size))
    height = float(box.get('height', size))
    bx = float(box.get('x', 0.0))
    by = float(box.get('y', 0.0))
    bz = float(box.get('z', height / 2.0))

    # Box bottom and top surfaces in world Z (metres above floor z=0)
    box_bottom_z = max(0.0, bz - height / 2.0)
    box_top_z = box_bottom_z + height

    # 1. Hover height: finger tips (palm_z - 0.25) clear top of box (box_top_z) while rotating/opening
    hover_palm_z = min(
        robot_gen.ENVELOPE_Z[1] - 0.02,
        max(0.48, box_top_z + 0.28)
    )

    # 2. Grasp height:
    #    - Bottom of palm (palm_z - 0.05) stays >= box_top_z + 0.005 so palm never crushes top of box
    #    - Finger tips (palm_z - 0.25) stay >= 0.025 m above floor so fingers never scrape ground
    grasp_palm_z = min(
        robot_gen.ENVELOPE_Z[1] - 0.05,
        max(0.275, box_top_z + 0.055)
    )

    # 3. Lift height: lift box clear off the floor while staying inside ENVELOPE_Z = (0.2, 1.3)
    lift_palm_z = min(
        robot_gen.ENVELOPE_Z[1] - 0.02,
        max(grasp_palm_z + 0.18, 0.60)
    )

    # 4. Finger opening (before descent) and closing (to clamp box width)
    #    gap = 0.10 + 2 * q  =>  q_touch = (width - 0.10) / 2.0
    q_touch = (width - 0.10) / 2.0
    finger_open = min(
        robot_gen.FINGER_TRAVEL,
        max(0.35, q_touch + 0.15)
    )
    finger_close = max(
        0.0,
        min(robot_gen.FINGER_TRAVEL, q_touch - 0.015)
    )

    # 5. Standoff distance from box center (bx, by):
    #    Centered inside ENVELOPE_R = (0.9, 1.9) and clearing front of chassis
    standoff_r = min(
        1.50,
        max(1.25, 0.85 + max(width, depth) / 2.0)
    )

    return {
        'x': bx,
        'y': by,
        'z': bz,
        'size': size,
        'width': width,
        'depth': depth,
        'height': height,
        'box_bottom_z': box_bottom_z,
        'box_top_z': box_top_z,
        'hover_palm_z': hover_palm_z,
        'grasp_palm_z': grasp_palm_z,
        'lift_palm_z': lift_palm_z,
        'finger_open': finger_open,
        'finger_close': finger_close,
        'standoff_r': standoff_r,
    }


class SorterBrain:
    State = State
    Action = Action

    def __init__(self):
        # State machine
        self.state = State.IDLE

        # Box queue: list of dicts from /box_ready
        self.box_queue = []
        self.current_box = None  # dict of box being processed
        self.action = None       # Action from rules.decide
        self.retries = 0         # grip retries (max 1)

        # Live box poses from /world_poses: name -> (x, y, z, yaw)
        self.live_box_poses = {}

        # Sorted box counters
        self.sorted_count = 0
        self.sorted_by_bin = {k: 0 for k in Y.BINS}

        # Robot pose from /tf or /world_poses
        self.robot_pose = None   # (x, y, z, yaw)
        self.robot_pose_valid = False

        # Navigation tracking
        self.nav_goal_sent_time = None
        self.active_nav_goal = None  # (goal_x, goal_y)
        self.nav_timeout_seconds = 55.0  # Generous timeout for traveling across 20m yard

        # Action tracking for waiting
        self.action_start_time = None
        self.action_timeout_seconds = 6.0  # Timeout for actions like finger/arm movement

        # State tracking for timeouts
        self.state_start_time = None
        self.state_timeout_seconds = 60.0

        # Constants for standoff distances
        self.GRIP_STANDOFF_DISTANCE = 1.35  # meters, centered in ENVELOPE_R (0.9, 1.9)
        self.brain_status_pub = None

        # If subclassed by Node or ROS environment
        if hasattr(self, 'create_subscription'):
            self._setup_ros()

    def get_logger(self):
        class _FallbackLogger:
            def info(self, msg): pass
            def debug(self, msg): pass
            def warn(self, msg): pass
            def error(self, msg): pass
        return _FallbackLogger()

    def _setup_ros(self):
        from std_msgs.msg import String, Float64
        from geometry_msgs.msg import PoseStamped, Twist
        from tf2_msgs.msg import TFMessage

        # Subscribers
        self.box_ready_sub = self.create_subscription(
            String, '/box_ready', self.box_ready_callback, 10)
        self.nav_status_sub = self.create_subscription(
            String, '/nav/status', self.nav_status_callback, 10)
        self.world_poses_sub = self.create_subscription(
            TFMessage, '/world_poses', self.world_poses_callback, 10)
        self.tf_sub = self.create_subscription(
            TFMessage, '/tf', self.tf_callback, 10)

        # Publishers
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/nav/goal', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.turret_pub = self.create_publisher(Float64, '/cmd/turret', 10)
        self.shoulder_pub = self.create_publisher(Float64, '/cmd/shoulder', 10)
        self.elbow_pub = self.create_publisher(Float64, '/cmd/elbow', 10)
        self.wrist_pub = self.create_publisher(Float64, '/cmd/wrist', 10)
        self.finger_pub = self.create_publisher(Float64, '/cmd/finger', 10)
        self.blade_pub = self.create_publisher(Float64, '/cmd/blade', 10)
        self.box_pose_pub = self.create_publisher(String, '/box_pose', 10)
        self.sorted_pub = self.create_publisher(String, '/sorted_status', 10)
        self.brain_status_pub = self.create_publisher(String, '/brain_status', 10)

        # Timer for state machine updates
        self.timer = self.create_timer(0.1, self.update_state_machine)  # 10 Hz
        self.get_logger().info('SorterBrain node started! Monitoring /box_ready and /world_poses...')
        self._idle_ticks = 0

    def _sort_queue(self):
        """Sort box queue balancing distance with waiting time to ensure far boxes are never starved."""
        if not self.box_queue:
            return
        now = time.time()
        def _score(b):
            dist = self.distance_to_robot(b)
            wait = now - b.get('_arrival_time', now)
            return dist - (wait * 2.5)
        self.box_queue.sort(key=_score)

    def box_ready_callback(self, msg):
        """Handle incoming box message."""
        try:
            box_data = json.loads(msg.data)
            size = float(box_data.get('size', 0.4))
            box_data.setdefault('width', size)
            box_data.setdefault('depth', size)
            box_data.setdefault('height', size)
            box_data.setdefault('z', box_data['height'] / 2.0)
            box_data.setdefault('_arrival_time', time.time())

            self.get_logger().info(
                f'Detected {box_data.get("name")}: pos=({box_data["x"]:.2f}, {box_data["y"]:.2f}, {box_data["z"]:.2f}), '
                f'dims=({box_data["width"]:.2f}x{box_data["depth"]:.2f}x{box_data["height"]:.2f}m), '
                f'mass={box_data.get("mass")}kg, color={box_data.get("color")}, damaged={box_data.get("damaged")}'
            )
            self.box_queue.append(box_data)
            if self.robot_pose is not None:
                self._sort_queue()
        except Exception as e:
            self.get_logger().error(f'Failed to parse box_ready: {e}')

    def nav_status_callback(self, msg):
        """Handle navigation status updates.
        Strictly guards against stale 'reached' messages from previous goals!
        """
        status_text = msg.data.lower()

        # 1. Only process nav status when we are in an explicit *_WAIT state
        if self.state not in (State.NAVIGATE_TO_STANDOFF_WAIT, State.NAVIGATE_TO_BIN_WAIT):
            return

        # 2. Must have actually sent a navigation goal for this state
        if self.nav_goal_sent_time is None:
            return

        # 3. Ignore stale status messages for the first 0.6s after sending a new goal
        elapsed = time.time() - self.nav_goal_sent_time
        if elapsed < 0.6:
            return

        if 'reached' in status_text:
            # 4. Verify robot pose is actually near the active navigation goal
            if self.robot_pose is not None and getattr(self, 'active_nav_goal', None) is not None:
                rx, ry = self.robot_pose[0], self.robot_pose[1]
                gx, gy = self.active_nav_goal
                dist_to_goal = math.hypot(rx - gx, ry - gy)
                if dist_to_goal > 1.15:
                    # Stale 'reached' from previous goal; re-send current goal if needed
                    if elapsed > 2.0 and int(elapsed * 10) % 20 == 0:
                        self.publish_nav_goal(gx, gy, 0.0)
                    return

            self.get_logger().info('Navigation goal reached')
            self.nav_goal_sent_time = None
            self.active_nav_goal = None
            if self.state == State.NAVIGATE_TO_STANDOFF_WAIT:
                self.state = State.GRIP_YAW
            elif self.state == State.NAVIGATE_TO_BIN_WAIT:
                self.state = State.GRIP_RELEASE

        elif 'failed' in status_text:
            self.get_logger().warn('Navigation goal failed')
            self.nav_goal_sent_time = None
            self.active_nav_goal = None
            self.state = State.FAILED

    def tf_callback(self, msg):
        """Extract robot pose from /tf (diff_drive / rviz_sim_node publishes odom->base_link here)."""
        for tf in msg.transforms:
            if tf.child_frame_id in ('base_link', '/base_link', robot_gen.ROBOT, '/' + robot_gen.ROBOT):
                x = tf.transform.rotation.x
                y = tf.transform.rotation.y
                z = tf.transform.rotation.z
                w = tf.transform.rotation.w
                siny_cosp = 2 * (w * z + x * y)
                cosy_cosp = 1 - 2 * (y * y + z * z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                self.robot_pose = (
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                    yaw
                )
                self.robot_pose_valid = True
                break

    def world_poses_callback(self, msg):
        """Extract robot pose and live 3D box poses from /world_poses TFMessage."""
        if not hasattr(self, '_debug_printed'):
            frames = [tf.child_frame_id for tf in msg.transforms]
            self.get_logger().info(f"DEBUG /world_poses frames: {frames}")
            self._debug_printed = True

        if not hasattr(self, 'live_box_poses'):
            self.live_box_poses = {}

        for tf in msg.transforms:
            name = tf.child_frame_id
            x = tf.transform.rotation.x
            y = tf.transform.rotation.y
            z = tf.transform.rotation.z
            w = tf.transform.rotation.w
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            if name in ('base_link', '/base_link', robot_gen.ROBOT, '/' + robot_gen.ROBOT):
                self.robot_pose = (
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                    yaw
                )
                self.robot_pose_valid = True
            elif name.startswith('box_') and '::' not in name:
                self.live_box_poses[name] = (
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                    yaw
                )

    def _refresh_current_box_pose(self):
        """Update current_box (x, y, z) from live poses if available before grasping."""
        if not self.current_box:
            return
        size = float(self.current_box.get('size', 0.4))
        self.current_box.setdefault('width', size)
        self.current_box.setdefault('depth', size)
        self.current_box.setdefault('height', size)
        self.current_box.setdefault('z', self.current_box['height'] / 2.0)

        live_poses = getattr(self, 'live_box_poses', None)
        name = self.current_box.get('name')
        if live_poses and name in live_poses:
            lx, ly, lz, lyaw = live_poses[name]
            self.current_box['x'] = lx
            self.current_box['y'] = ly
            self.current_box['z'] = lz
            self.current_box['yaw'] = lyaw

    def _get_bin_pos(self, bin_name: str):
        """Resolve bin coordinates safely, falling back if a bin name is missing in yard.BINS."""
        if bin_name in Y.BINS:
            return Y.BINS[bin_name]
        if 'other' in Y.BINS:
            return Y.BINS['other']
        if 'scrap' in Y.BINS:
            return Y.BINS['scrap']
        return (-Y.HALF_X + 0.8, -1.0, 1.6)

    def _is_standoff_clear(self, gx: float, gy: float) -> bool:
        """Check that a candidate standoff point is inside yard bounds and clear of poles/bins/other boxes."""
        margin = 0.70
        if not (-Y.HALF_X + margin <= gx <= Y.HALF_X - margin and
                -Y.HALF_Y + margin <= gy <= Y.HALF_Y - margin):
            return False
        for px, py in Y.POLES:
            if math.hypot(gx - px, gy - py) < (Y.POLE_R + 1.15):
                return False
        for _, (bx, by, bsize) in Y.BINS.items():
            half = bsize / 2.0 + 0.65
            if abs(gx - bx) < half and abs(gy - by) < half:
                return False
        # Avoid standing right on top of another waiting box
        cur_name = self.current_box.get('name') if self.current_box else None
        for b in self.box_queue:
            if b.get('name') != cur_name:
                if math.hypot(gx - b.get('x', 999.0), gy - b.get('y', 999.0)) < 0.85:
                    return False
        return True

    def _choose_grip_standoff(self, rx: float, ry: float,
                              bx: float, by: float, standoff_dist: float):
        """
        Choose an obstacle-free standoff point around the box at distance `standoff_dist`.
        """
        dx = bx - rx
        dy = by - ry
        dist_to_box = math.hypot(dx, dy)
        direct_angle = math.atan2(-dy, -dx) if dist_to_box > 1e-6 else math.pi

        candidate_angles = [
            direct_angle,
            0.0, math.pi / 2.0, math.pi, -math.pi / 2.0,
        ]
        for k in range(32):
            candidate_angles.append(k * (2.0 * math.pi / 32.0) - math.pi)

        best_pt = None
        best_cost = float('inf')
        for ang in candidate_angles:
            gx = bx + standoff_dist * math.cos(ang)
            gy = by + standoff_dist * math.sin(ang)
            if self._is_standoff_clear(gx, gy):
                is_cardinal = any(abs(_normalize_angle(ang - c)) < 0.05
                                  for c in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0))
                cost = math.hypot(gx - rx, gy - ry) - (0.15 if is_cardinal else 0.0)
                if cost < best_cost:
                    best_cost = cost
                    best_pt = (gx, gy)

        if best_pt is not None:
            return best_pt

        if dist_to_box < 1e-6:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / dist_to_box, dy / dist_to_box
        gx = max(-Y.HALF_X + 0.65, min(Y.HALF_X - 0.65, bx - standoff_dist * ux))
        gy = max(-Y.HALF_Y + 0.65, min(Y.HALF_Y - 0.65, by - standoff_dist * uy))
        return gx, gy

    # -------------------- State Machine --------------------
    def update_state_machine(self):
        """State machine tick."""
        current_time = time.time()
        if self.state_start_time is None:
            self.state_start_time = current_time
        elif hasattr(self, '_last_state') and self._last_state != self.state:
            self.state_start_time = current_time
        self._last_state = self.state

        if self.state_start_time is not None and current_time - self.state_start_time > self.state_timeout_seconds:
            self.get_logger().error(f'Timeout in state {self.state}')
            self.state = State.FAILED
            self.state_start_time = None
            return

        self._publish_brain_status()

        try:
            # Carrying logic: continuously sync box position with palm while carrying
            if (self.state in (State.GRIP_LIFT_WAIT, State.NAVIGATE_TO_BIN,
                               State.NAVIGATE_TO_BIN_WAIT, State.GRIP_POSITION_BIN_WAIT)
                    and self.current_box is not None):
                self._update_carrying_box_pose()

            if self.state == State.IDLE:
                self.handle_idle()
            elif self.state == State.SELECT_BOX:
                self.handle_select_box()
            elif self.state == State.DECIDE:
                self.handle_decide()
            elif self.state == State.NAVIGATE_TO_STANDOFF:
                self.handle_navigate_to_standoff()
            elif self.state == State.NAVIGATE_TO_STANDOFF_WAIT:
                self.handle_navigate_to_standoff_wait()
            elif self.state == State.GRIP_YAW:
                self.handle_grip_yaw()
            elif self.state == State.GRIP_LOWER:
                self.handle_grip_lower()
            elif self.state == State.GRIP_LOWER_WAIT:
                self.handle_grip_lower_wait()
            elif self.state == State.GRIP_IK:
                self.handle_grip_ik()
            elif self.state == State.GRIP_IK_WAIT:
                self.handle_grip_ik_wait()
            elif self.state == State.GRIP_CLOSE:
                self.handle_grip_close()
            elif self.state == State.GRIP_CLOSE_WAIT:
                self.handle_grip_close_wait()
            elif self.state == State.GRIP_LIFT:
                self.handle_grip_lift()
            elif self.state == State.GRIP_LIFT_WAIT:
                self.handle_grip_lift_wait()
            elif self.state == State.NAVIGATE_TO_BIN:
                self.handle_navigate_to_bin()
            elif self.state == State.NAVIGATE_TO_BIN_WAIT:
                self.handle_navigate_to_bin_wait()
            elif self.state == State.GRIP_RELEASE:
                self.handle_grip_release()
            elif self.state == State.GRIP_POSITION_BIN_WAIT:
                self.handle_grip_position_bin_wait()
            elif self.state == State.GRIP_RELEASE_WAIT:
                self.handle_grip_release_wait()
            elif self.state == State.RETRACT:
                self.handle_retract()
            elif self.state == State.SCRAP:
                self.handle_scrap()
            elif self.state == State.FAILED:
                self.handle_failed()
            else:
                self.get_logger().warn(f'Unknown state: {self.state}')

        except Exception as e:
            import traceback
            self.get_logger().error(f'CRASH in state {self.state}: {e}\n{traceback.format_exc()}')
            self.state = State.FAILED

    def _publish_brain_status(self):
        """Publish concise human-readable brain status for RViz 3D HUD."""
        if not hasattr(self, 'brain_status_pub') or self.brain_status_pub is None:
            return
        bin_str = self.action.bin.upper() if getattr(self, 'action', None) else ''
        box_name = self.current_box.get('name', 'BOX').upper() if getattr(self, 'current_box', None) else ''

        status_map = {
            State.IDLE: "STANDBY | WAITING FOR BOX",
            State.SELECT_BOX: f"TARGETING {box_name}",
            State.DECIDE: f"DECIDING ROUTE -> {bin_str} BIN",
            State.NAVIGATE_TO_STANDOFF: f"NAVIGATING TO {box_name}",
            State.NAVIGATE_TO_STANDOFF_WAIT: f"APPROACHING {box_name}",
            State.GRIP_YAW: "ALIGNING ARM & HOVER",
            State.GRIP_LOWER: "OPENING CLAW TO GRASP",
            State.GRIP_LOWER_WAIT: "OPENING CLAW TO GRASP",
            State.GRIP_IK: "DESCENDING OVER BOX",
            State.GRIP_IK_WAIT: "POSITIONING ON BOX",
            State.GRIP_CLOSE: "CLAMPING BOX",
            State.GRIP_CLOSE_WAIT: "CLAMPING BOX",
            State.GRIP_LIFT: "LIFTING BOX",
            State.GRIP_LIFT_WAIT: "LIFTING BOX",
            State.NAVIGATE_TO_BIN: f"TRANSPORTING TO {bin_str} BIN",
            State.NAVIGATE_TO_BIN_WAIT: f"TRANSPORTING TO {bin_str} BIN",
            State.GRIP_RELEASE: f"ALIGNING TO {bin_str} BIN",
            State.GRIP_POSITION_BIN_WAIT: f"PLACING INTO {bin_str} BIN",
            State.GRIP_RELEASE_WAIT: f"DISPATCHING TO {bin_str}",
            State.RETRACT: "RETRACTING ARM",
            State.SCRAP: "ROUTING SCRAP BOX",
            State.FAILED: "RECOVERING",
        }
        status_text = status_map.get(self.state, str(self.state))
        try:
            from std_msgs.msg import String
            self.brain_status_pub.publish(String(data=status_text))
        except Exception:
            pass

    # -------------------- State Handlers --------------------
    def handle_idle(self):
        """Wait for boxes. Don't advance until we have a valid robot pose."""
        if not self.robot_pose_valid:
            self._idle_ticks = getattr(self, '_idle_ticks', 0) + 1
            if self._idle_ticks % 50 == 0:
                self.get_logger().warn('No robot pose yet, waiting...')
            self.state_start_time = None
            return

        if self.box_queue:
            if self.robot_pose is not None:
                self._sort_queue()
            self.get_logger().info('Boxes in queue, selecting next prioritized box...')
            self.state = State.SELECT_BOX
        else:
            self._idle_ticks = getattr(self, '_idle_ticks', 0) + 1
            if self._idle_ticks % 50 == 0:
                self.get_logger().info(
                    f'SorterBrain IDLE: waiting for boxes (sorted so far: {getattr(self, "sorted_count", 0)})'
                )

    def handle_select_box(self):
        """Select next box from queue and reset retry counter."""
        if not self.box_queue:
            self.state = State.IDLE
            return
        if self.robot_pose is not None:
            self._sort_queue()
        self.current_box = self.box_queue.pop(0)
        self.retries = 0
        self.nav_goal_sent_time = None
        self.active_nav_goal = None
        self._refresh_current_box_pose()
        self.get_logger().info(f'Selected box: {self.current_box.get("name")}')
        self.state = State.DECIDE

    def handle_decide(self):
        """Decide target bin using rules and compute 3D grasp profile."""
        size = self.current_box['size']
        mass = self.current_box['mass']
        color = self.current_box['color']
        damaged = self.current_box.get('damaged', False)
        self.action = rules.decide(size, mass, color, damaged)
        profile = compute_box_grasp_profile(self.current_box)
        self.grasp_profile = profile
        self.get_logger().info(
            f'Box {self.current_box.get("name")} @ '
            f'(x={profile["x"]:.2f}, y={profile["y"]:.2f}, z={profile["z"]:.2f}) | '
            f'width={profile["width"]:.2f}m, depth={profile["depth"]:.2f}m, height={profile["height"]:.2f}m, '
            f'mass={mass}kg, color={color}, damaged={damaged} -> '
            f'Decision: {self.action.mode.upper()} to bin "{self.action.bin}" '
            f'(hover_z={profile["hover_palm_z"]:.3f}m, grasp_z={profile["grasp_palm_z"]:.3f}m, '
            f'lift_z={profile["lift_palm_z"]:.3f}m, finger_close={profile["finger_close"]:.3f})'
        )
        self.nav_goal_sent_time = None
        self.active_nav_goal = None
        self.state = State.NAVIGATE_TO_STANDOFF

    def handle_navigate_to_standoff(self):
        """Compute standoff point around the target box and publish /nav/goal."""
        if not self.robot_pose:
            self.get_logger().warn('No robot pose yet, waiting...')
            self.state_start_time = time.time()
            return

        self._refresh_current_box_pose()
        profile = compute_box_grasp_profile(self.current_box)
        self.grasp_profile = profile

        rx, ry, rz, ryaw = self.robot_pose
        bx = profile['x']
        by = profile['y']

        # Hold blade raised, claw closed in neat travel pose, and arm tucked while approaching
        self.publish_blade(0.0)
        self.publish_finger(0.0)
        try:
            s_safe, e_safe, w_safe = arm_ik.solve(1.10, 0.75)
            self.publish_shoulder(s_safe)
            self.publish_elbow(e_safe)
            self.publish_wrist(w_safe)
        except ValueError:
            pass

        standoff_dist = profile['standoff_r']
        goal_x, goal_y = self._choose_grip_standoff(rx, ry, bx, by, standoff_dist)
        self.active_nav_goal = (goal_x, goal_y)
        self.nav_goal_sent_time = time.time()
        self.grasp_palm_z = profile['grasp_palm_z']
        self.grasp_r = standoff_dist
        self.publish_nav_goal(goal_x, goal_y, 0.0)
        self.state = State.NAVIGATE_TO_STANDOFF_WAIT

    def handle_navigate_to_standoff_wait(self):
        """Wait for navigation to standoff to complete; check timeout."""
        if self.nav_goal_sent_time is not None:
            if time.time() - self.nav_goal_sent_time > self.nav_timeout_seconds:
                self.get_logger().error('Navigation timeout in NAVIGATE_TO_STANDOFF_WAIT')
                self.nav_goal_sent_time = None
                self.active_nav_goal = None
                self.state = State.FAILED

    def handle_grip_yaw(self):
        """Yaw turret to face the box and raise arm to hover height above the box."""
        if not self.robot_pose or not self.current_box:
            self.get_logger().warn('Missing data for yaw')
            return
        self._refresh_current_box_pose()
        profile = compute_box_grasp_profile(self.current_box)
        self.grasp_profile = profile

        rx, ry, rz, ryaw = self.robot_pose
        bx, by = profile['x'], profile['y']
        dx = bx - rx
        dy = by - ry
        r = math.hypot(dx, dy)

        # If robot is not yet within reach of the box, drive to standoff first!
        if r > 2.15 or r < 0.70:
            self.get_logger().warn(
                f'Robot is at r={r:.2f}m from {self.current_box.get("name")}, navigating to standoff first...'
            )
            self.nav_goal_sent_time = None
            self.active_nav_goal = None
            self.state = State.NAVIGATE_TO_STANDOFF
            return

        angle_to_box = math.atan2(dy, dx)
        turret_angle = _normalize_angle(angle_to_box - ryaw)
        self.publish_turret(turret_angle)

        r_hover = max(robot_gen.ENVELOPE_R[0], min(robot_gen.ENVELOPE_R[1], r))
        try:
            s_hov, e_hov, w_hov = arm_ik.solve(r_hover, profile['hover_palm_z'])
            self.publish_shoulder(s_hov)
            self.publish_elbow(e_hov)
            self.publish_wrist(w_hov)
        except ValueError:
            pass

        self.get_logger().info(
            f'Yawing turret to {turret_angle:.2f} rad and hovering at z={profile["hover_palm_z"]:.2f}m '
            f'over {self.current_box.get("name")} (width={profile["width"]:.2f}m, height={profile["height"]:.2f}m)'
        )
        self.state = State.GRIP_LOWER

    def handle_grip_lower(self):
        """Open the gripper fingers wide above the box before descending to grasp."""
        self.get_logger().info('Opening fingers wide before lowering onto box...')
        self.publish_finger(0.5)
        self.action_start_time = time.time()
        self.state = State.GRIP_LOWER_WAIT

    def handle_grip_lower_wait(self):
        """Wait for fingers and hover pose to settle before descending onto the box."""
        if time.time() - self.action_start_time > 0.8:
            self.get_logger().info('Fingers opened and hover settled, descending to grasp position...')
            self.action_start_time = None
            self.state = State.GRIP_IK
        elif time.time() - self.action_start_time > self.action_timeout_seconds:
            self.get_logger().error('Timeout waiting for fingers to settle')
            self.state = State.FAILED
            self.action_start_time = None

    def handle_grip_ik(self):
        """Compute IK to lower the open gripper vertically onto the box based on its 3D dimensions."""
        if not self.robot_pose or not self.current_box:
            self.get_logger().warn('Missing data for IK')
            return
        self._refresh_current_box_pose()
        profile = compute_box_grasp_profile(self.current_box)
        self.grasp_profile = profile

        rx, ry, rz, ryaw = self.robot_pose
        bx, by = profile['x'], profile['y']
        dx = bx - rx
        dy = by - ry
        r = math.hypot(dx, dy)
        z = profile['grasp_palm_z']

        # Re-align turret with exact box center
        turret_angle = _normalize_angle(math.atan2(dy, dx) - ryaw)
        self.publish_turret(turret_angle)

        if r < 0.70 or r > 2.20:
            self.get_logger().warn(f'Palm r={r:.2f} far outside envelope {robot_gen.ENVELOPE_R}')
            self.state = State.FAILED
            return
        r_ik = max(robot_gen.ENVELOPE_R[0], min(robot_gen.ENVELOPE_R[1], r))

        if not (robot_gen.ENVELOPE_Z[0] <= z <= robot_gen.ENVELOPE_Z[1]):
            self.get_logger().warn(f'Palm z={z:.2f} outside envelope {robot_gen.ENVELOPE_Z}')
            self.state = State.FAILED
            return
        try:
            shoulder, elbow, wrist = arm_ik.solve(r_ik, z)
            self.get_logger().info(
                f'Lowering open gripper onto box (r={r_ik:.2f}m, z={z:.3f}m for box height={profile["height"]:.2f}m): '
                f'shoulder={shoulder:.2f}, elbow={elbow:.2f}, wrist={wrist:.2f}'
            )
            self.publish_shoulder(shoulder)
            self.publish_elbow(elbow)
            self.publish_wrist(wrist)
        except ValueError as e:
            self.get_logger().warn(f'IK failed: {e}.')
            self.state = State.FAILED
            return
        self.action_start_time = time.time()
        self.state = State.GRIP_IK_WAIT

    def handle_grip_ik_wait(self):
        """Wait for arm to settle at grasp height around the box."""
        if time.time() - self.action_start_time > 0.8:
            self.get_logger().info('Arm settled around box, closing fingers...')
            self.action_start_time = None
            self.state = State.GRIP_CLOSE
        elif time.time() - self.action_start_time > self.action_timeout_seconds:
            self.get_logger().error('Timeout waiting for arm to settle')
            self.state = State.FAILED
            self.action_start_time = None

    def handle_grip_close(self):
        """Close fingers to tightly grip the box according to its exact width."""
        profile = compute_box_grasp_profile(self.current_box)
        finger_opening = profile['finger_close']
        self.get_logger().info(
            f'Closing fingers to {finger_opening:.3f}m for box width={profile["width"]:.2f}m, height={profile["height"]:.2f}m'
        )
        self.publish_finger(finger_opening)
        self.action_start_time = time.time()
        self.state = State.GRIP_CLOSE_WAIT

    def handle_grip_close_wait(self):
        """Wait for fingers to settle after closing onto the box."""
        if time.time() - self.action_start_time > 0.8:
            self.get_logger().info('Fingers clamped onto box, lifting box...')
            self.action_start_time = None
            self.state = State.GRIP_LIFT
        elif time.time() - self.action_start_time > self.action_timeout_seconds:
            self.get_logger().error('Timeout waiting for fingers to settle')
            self.state = State.FAILED
            self.action_start_time = None

    def handle_grip_lift(self):
        """Lift the box after gripping."""
        self.get_logger().info('Lifting box...')
        if not self.robot_pose or not self.current_box:
            self.get_logger().warn('Missing data for lift')
            return
        profile = compute_box_grasp_profile(self.current_box)
        rx, ry, rz, ryaw = self.robot_pose
        bx, by = profile['x'], profile['y']
        r = math.hypot(bx - rx, by - ry)
        r_ik = max(robot_gen.ENVELOPE_R[0], min(robot_gen.ENVELOPE_R[1], r))
        z = profile['lift_palm_z']

        # Hold turret centered (0.0 rad) and tuck box close to chassis while carrying
        self.carry_r = 1.05
        self.carry_theta = 0.0
        self.carry_palm_z = z
        self.publish_turret(0.0)

        if not (robot_gen.ENVELOPE_Z[0] <= z <= robot_gen.ENVELOPE_Z[1]):
            self.get_logger().warn(f'Lift z={z:.2f} outside envelope')
            self.state = State.FAILED
            return
        try:
            shoulder, elbow, wrist = arm_ik.solve(self.carry_r, z)
            self.get_logger().info(
                f'IK lift (r={self.carry_r:.2f}m, z={z:.3f}m): shoulder={shoulder:.2f}, elbow={elbow:.2f}, wrist={wrist:.2f}'
            )
            self.publish_shoulder(shoulder)
            self.publish_elbow(elbow)
            self.publish_wrist(wrist)
        except ValueError as e:
            self.get_logger().warn(f'IK lift failed: {e}')
            self.state = State.FAILED
            return

        self._update_carrying_box_pose()
        self.nav_goal_sent_time = None
        self.active_nav_goal = None
        self.state = State.NAVIGATE_TO_BIN

    def handle_grip_lift_wait(self):
        self.state = State.NAVIGATE_TO_BIN

    def handle_navigate_to_bin(self):
        """Navigate to the target bin on the left wall."""
        if not self.robot_pose:
            self.get_logger().warn('No robot pose yet, waiting...')
            self.state_start_time = time.time()
            return

        bin_name = self.action.bin
        bx, by, bsize = self._get_bin_pos(bin_name)

        # Stand off 1.4m in front of the bin center (outside inflated bin obstacle and centered in ENVELOPE_R)
        goal_x = bx + 1.4
        goal_y = by
        self.active_nav_goal = (goal_x, goal_y)
        self.nav_goal_sent_time = time.time()
        self.publish_nav_goal(goal_x, goal_y, 0.0)
        self.state = State.NAVIGATE_TO_BIN_WAIT

    def handle_navigate_to_bin_wait(self):
        """Wait for navigation to bin to complete; check timeout."""
        if self.nav_goal_sent_time is not None:
            if time.time() - self.nav_goal_sent_time > self.nav_timeout_seconds:
                self.get_logger().error('Navigation timeout in NAVIGATE_TO_BIN_WAIT')
                self.nav_goal_sent_time = None
                self.active_nav_goal = None
                self.state = State.FAILED

    def handle_grip_release(self):
        """Phase 1 of release: rotate turret to face bin and extend arm over bin center while keeping fingers closed."""
        if not self.robot_pose:
            self.get_logger().warn('No robot pose yet, waiting...')
            self.state_start_time = time.time()
            return
        bin_name = self.action.bin
        bx, by, bsize = self._get_bin_pos(bin_name)
        profile = compute_box_grasp_profile(self.current_box) if self.current_box else {'height': 0.4}
        height = profile['height']

        palm_x = bx
        palm_y = by
        palm_z = min(robot_gen.ENVELOPE_Z[1] - 0.05, max(0.55, height + 0.25))

        rx, ry, rz, ryaw = self.robot_pose
        dx = palm_x - rx
        dy = palm_y - ry
        r = math.hypot(dx, dy)
        r_ik = max(robot_gen.ENVELOPE_R[0], min(robot_gen.ENVELOPE_R[1], r))
        z = palm_z

        # Rotate turret directly toward the bin center
        turret_to_bin = _normalize_angle(math.atan2(dy, dx) - ryaw)
        self.publish_turret(turret_to_bin)

        # Update carry coordinates so box swings over the bin center with the arm
        self.carry_r = r_ik
        self.carry_theta = turret_to_bin
        self.carry_palm_z = z

        self.get_logger().info(
            f'Positioning palm over bin "{bin_name}" at ({palm_x:.2f}, {palm_y:.2f}, {palm_z:.2f}), '
            f'r={r_ik:.2f}m, turret={turret_to_bin:.2f} rad for release...'
        )

        if not (robot_gen.ENVELOPE_Z[0] <= z <= robot_gen.ENVELOPE_Z[1]):
            self.get_logger().warn(f'Release palm z={z:.2f} outside envelope {robot_gen.ENVELOPE_Z}')
            self.state = State.FAILED
            return
        try:
            shoulder, elbow, wrist = arm_ik.solve(r_ik, z)
            self.get_logger().info(
                f'IK release position: shoulder={shoulder:.2f}, elbow={elbow:.2f}, wrist={wrist:.2f}'
            )
            self.publish_shoulder(shoulder)
            self.publish_elbow(elbow)
            self.publish_wrist(wrist)
        except ValueError as e:
            self.get_logger().warn(f'IK release failed: {e}')
            self.state = State.FAILED
            return

        self.action_start_time = time.time()
        self.state = State.GRIP_POSITION_BIN_WAIT

    def handle_grip_position_bin_wait(self):
        """Phase 2 of release: once arm has swung over the bin center, place box in bin and open fingers."""
        if time.time() - self.action_start_time > 0.8:
            bin_name = self.action.bin
            bx, by, bsize = self._get_bin_pos(bin_name)
            profile = compute_box_grasp_profile(self.current_box) if self.current_box else {'height': 0.4}
            height = profile['height']

            if self.current_box and 'name' in self.current_box:
                box_name = self.current_box['name']
                drop_z = height / 2.0 + 0.02
                self._publish_box_pose(box_name, bx, by, drop_z, state='released')
                if hasattr(self, 'live_box_poses'):
                    self.live_box_poses[box_name] = (bx, by, drop_z, 0.0)

            self.get_logger().info('Opening fingers to release box into bin...')
            self.publish_finger(0.5)
            self.action_start_time = time.time()
            self.state = State.GRIP_RELEASE_WAIT

    def handle_grip_release_wait(self):
        """Wait for fingers to settle after opening for release."""
        if time.time() - self.action_start_time > 0.6:
            self.get_logger().info('Fingers settled, retracting after release...')
            self.action_start_time = None
            self.state = State.RETRACT
        elif time.time() - self.action_start_time > self.action_timeout_seconds:
            self.get_logger().error('Timeout waiting for fingers to settle after release')
            self.state = State.FAILED
            self.action_start_time = None

    def handle_retract(self):
        """Retract arm to safe position inside envelope and record sorted box."""
        self.get_logger().info('Retracting arm to safe position...')
        safe_r = (robot_gen.ENVELOPE_R[0] + robot_gen.ENVELOPE_R[1]) / 2.0
        safe_z = (robot_gen.ENVELOPE_Z[0] + robot_gen.ENVELOPE_Z[1]) / 2.0

        try:
            shoulder, elbow, wrist = arm_ik.solve(safe_r, safe_z)
            self.publish_turret(0.0)
            self.publish_shoulder(shoulder)
            self.publish_elbow(elbow)
            self.publish_wrist(wrist)
        except ValueError as e:
            self.get_logger().warn(f'IK retract failed: {e}. Falling back to zero position.')
            self.publish_shoulder(0.0)
            self.publish_elbow(0.0)
            self.publish_wrist(0.0)
        self.publish_finger(0.0)

        self._record_box_sorted()
        self.current_box = None
        self.action = None
        self.state = State.IDLE

    def _record_box_sorted(self):
        """Increment and log sorted box counters."""
        self.sorted_count = getattr(self, 'sorted_count', 0) + 1
        if not hasattr(self, 'sorted_by_bin'):
            self.sorted_by_bin = {k: 0 for k in Y.BINS}
        bin_name = self.action.bin if self.action else 'unknown'
        self.sorted_by_bin[bin_name] = self.sorted_by_bin.get(bin_name, 0) + 1
        box_name = self.current_box.get('name') if self.current_box else 'box'
        self.get_logger().info(
            f'Sorted {box_name} into bin "{bin_name}"! '
            f'Total sorted: {self.sorted_count} | Breakdown: {self.sorted_by_bin}'
        )
        if hasattr(self, 'sorted_pub') and self.sorted_pub:
            try:
                from std_msgs.msg import String
                self.sorted_pub.publish(String(data=json.dumps({
                    'total': self.sorted_count,
                    'bins': self.sorted_by_bin,
                    'last_box': box_name,
                    'last_bin': bin_name,
                })))
            except Exception:
                pass

    def handle_scrap(self):
        """Handle damaged box (scrap)."""
        self.get_logger().info('Handling damaged box as scrap...')
        self.state = State.NAVIGATE_TO_STANDOFF

    def handle_failed(self):
        """Handle failure (e.g., IK unreachable, navigation failed)."""
        self.get_logger().warn(f'Failure in state {self.state}, retrying...')
        self.publish_blade(0.0)
        if self.current_box is not None:
            self.publish_finger(0.5)
        if self.retries < 2:
            self.retries += 1
            self.get_logger().info(f'Retrying box ({self.retries})...')
            self.nav_goal_sent_time = None
            self.active_nav_goal = None
            self.state = State.NAVIGATE_TO_STANDOFF
        else:
            self.get_logger().info('Re-queuing box to pick up on next pass so no box is left behind.')
            if self.current_box is not None:
                self.box_queue.append(self.current_box)
            self.current_box = None
            self.action = None
            self.nav_goal_sent_time = None
            self.active_nav_goal = None
            self.state = State.IDLE

    # -------------------- Helper Publishers --------------------
    def publish_nav_goal(self, x, y, z):
        """Publish a navigation goal."""
        if hasattr(self, 'nav_goal_pub') and self.nav_goal_pub:
            try:
                from geometry_msgs.msg import PoseStamped
                msg = PoseStamped()
                if hasattr(self, 'get_clock'):
                    msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'world'
                msg.pose.position.x = float(x)
                msg.pose.position.y = float(y)
                msg.pose.position.z = float(z)
                msg.pose.orientation.w = 1.0
                self.nav_goal_pub.publish(msg)
                self.get_logger().info(f'Published nav goal: ({x:.2f}, {y:.2f}, {z:.2f})')
            except Exception:
                pass

    def publish_turret(self, angle):
        if hasattr(self, 'turret_pub') and self.turret_pub:
            try:
                from std_msgs.msg import Float64
                msg = Float64()
                msg.data = float(angle)
                self.turret_pub.publish(msg)
            except Exception:
                pass

    def publish_shoulder(self, angle):
        if hasattr(self, 'shoulder_pub') and self.shoulder_pub:
            try:
                from std_msgs.msg import Float64
                msg = Float64()
                msg.data = float(angle)
                self.shoulder_pub.publish(msg)
            except Exception:
                pass

    def publish_elbow(self, angle):
        if hasattr(self, 'elbow_pub') and self.elbow_pub:
            try:
                from std_msgs.msg import Float64
                msg = Float64()
                msg.data = float(angle)
                self.elbow_pub.publish(msg)
            except Exception:
                pass

    def publish_wrist(self, angle):
        if hasattr(self, 'wrist_pub') and self.wrist_pub:
            try:
                from std_msgs.msg import Float64
                msg = Float64()
                msg.data = float(angle)
                self.wrist_pub.publish(msg)
            except Exception:
                pass

    def publish_finger(self, opening):
        if hasattr(self, 'finger_pub') and self.finger_pub:
            try:
                from std_msgs.msg import Float64
                msg = Float64()
                msg.data = float(opening)
                self.finger_pub.publish(msg)
            except Exception:
                pass

    def publish_blade(self, position):
        if hasattr(self, 'blade_pub') and self.blade_pub:
            try:
                from std_msgs.msg import Float64
                msg = Float64()
                msg.data = float(position)
                self.blade_pub.publish(msg)
            except Exception:
                pass

    def _publish_box_pose(self, box_name: str, x: float, y: float, z: float, state: str = 'carried'):
        """Publish box pose to /box_pose for RViz simulator node."""
        if hasattr(self, 'box_pose_pub') and self.box_pose_pub:
            try:
                from std_msgs.msg import String
                self.box_pose_pub.publish(String(data=json.dumps({
                    'name': box_name,
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'state': state,
                })))
            except Exception:
                pass

    def distance_to_robot(self, box):
        """Compute distance from robot to box (ignoring z)."""
        if self.robot_pose is None:
            return float('inf')
        rx, ry = self.robot_pose[0], self.robot_pose[1]
        dx = box['x'] - rx
        dy = box['y'] - ry
        return math.hypot(dx, dy)

    def _update_carrying_box_pose(self, blocking=False):
        """Update the box pose in RViz (/box_pose) to follow the gripper palm while carrying."""
        if not self.robot_pose or not self.current_box:
            return

        rx, ry, rz, ryaw = self.robot_pose
        size = float(self.current_box.get('size', 0.4))
        height = float(self.current_box.get('height', size))

        if hasattr(self, 'carry_r') and hasattr(self, 'carry_theta'):
            world_angle = ryaw + self.carry_theta
            palm_x = rx + self.carry_r * math.cos(world_angle)
            palm_y = ry + self.carry_r * math.sin(world_angle)
        else:
            palm_x = self.current_box['x']
            palm_y = self.current_box['y']

        palm_z = getattr(self, 'carry_palm_z', max(0.60, height + 0.25))
        box_center_z = max(height / 2.0 + 0.08, palm_z - 0.055 - height / 2.0)

        box_name = self.current_box['name']
        self._publish_box_pose(box_name, palm_x, palm_y, box_center_z, state='carried')


def main(args=None):
    import rclpy
    from rclpy.node import Node

    class SorterBrainNode(Node, SorterBrain):
        def __init__(self):
            Node.__init__(self, 'sorter_brain')
            SorterBrain.__init__(self)

    rclpy.init(args=args)
    node = SorterBrainNode()
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