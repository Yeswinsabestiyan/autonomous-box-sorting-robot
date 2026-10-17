#!/usr/bin/env python3
"""
RViz 3D Simulator Node for Autonomous Robotic Box Sorter.
Replaces Gazebo with a fast, deterministic 3D kinematic & visual simulation in RViz2:
  - Integrates /cmd_vel to drive the robot and publishes odom->base_link on /tf and /world_poses
  - Smoothly interpolates arm & gripper joint commands (/cmd/*) and publishes /joint_states
  - Tracks 3D boxes from /box_ready, attaches a grasped box to the 3D gripper palm when fingers
    close (and/or via /box_pose), carries it across the floor, and drops it into its designated bin
  - Publishes 3D visualization markers (/visualization_marker_array) for floor, walls, conveyor
    belts, obstacle poles, the 5 sorting bins (red, green, blue, other, scrap), and all boxes.
"""
import json
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from . import yard as Y
from . import arm_ik
from . import robot_gen


def _yaw_to_quat(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class RvizSimNode(Node):
    def __init__(self):
        super().__init__('rviz_sim_node')

        # Robot state in odom/world frame
        self.rx = 0.0
        self.ry = 0.0
        self.ryaw = 0.0
        self.vx = 0.0
        self.vyaw = 0.0

        # Joint positions & targets
        self.joint_pos = {
            'left_wheel_joint': 0.0,
            'right_wheel_joint': 0.0,
            'blade_joint': 0.0,
            'turret_joint': 0.0,
            'shoulder_joint': 0.0,
            'elbow_joint': 0.0,
            'wrist_joint': 0.0,
            'finger_left_joint': 0.0,
            'finger_right_joint': 0.0,
        }
        self.joint_target = dict(self.joint_pos)

        # Boxes: name -> dict(name, x, y, z, width, depth, height, size, color, mass, damaged, state)
        self.boxes = {}
        self.carried_box_name = None
        # Stacking slots per bin: 4 slots arranged in 2x2 grid inside each 1.6m bin
        self.SLOT_OFFSETS = [
            (-0.38, -0.38),  # Slot 0: bottom-left
            ( 0.38, -0.38),  # Slot 1: bottom-right
            (-0.38,  0.38),  # Slot 2: top-left
            ( 0.38,  0.38),  # Slot 3: top-right
        ]
        self.bin_stacks = {k: [[], [], [], []] for k in Y.BINS}
        self.dispatched_counts = {k: 0 for k in Y.BINS}

        # Subscribers
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.create_subscription(Float64, '/cmd/turret', lambda m: self._set_target('turret_joint', m.data), 10)
        self.create_subscription(Float64, '/cmd/shoulder', lambda m: self._set_target('shoulder_joint', m.data), 10)
        self.create_subscription(Float64, '/cmd/elbow', lambda m: self._set_target('elbow_joint', m.data), 10)
        self.create_subscription(Float64, '/cmd/wrist', lambda m: self._set_target('wrist_joint', m.data), 10)
        self.create_subscription(Float64, '/cmd/blade', lambda m: self._set_target('blade_joint', m.data), 10)
        self.create_subscription(Float64, '/cmd/finger', self.on_cmd_finger, 10)
        self.create_subscription(String, '/box_ready', self.on_box_ready, 50)
        self.create_subscription(String, '/box_pose', self.on_box_pose, 50)
        self.brain_status = "STANDBY | WAITING FOR BOX"
        self.create_subscription(String, '/brain_status', self.on_brain_status, 10)

        # Publishers
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 20)
        self.world_poses_pub = self.create_publisher(TFMessage, '/world_poses', 20)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 20)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 20)

        # 30 Hz simulation tick
        self.dt = 1.0 / 30.0
        self.create_timer(self.dt, self.sim_step)
        self.get_logger().info('🚀 RViz Simulator Node started! Running 3D simulation with zero Gazebo overhead.')

    def on_brain_status(self, msg: String):
        self.brain_status = msg.data

    def _set_target(self, joint_name: str, val: float):
        if joint_name in robot_gen.LIMITS:
            lo, hi = robot_gen.LIMITS[joint_name][:2]
            val = max(lo, min(hi, float(val)))
        self.joint_target[joint_name] = float(val)

    def on_cmd_vel(self, msg: Twist):
        self.vx = float(msg.linear.x)
        self.vyaw = float(msg.angular.z)

    def place_box_in_bin(self, b: dict, bname: str):
        """Place box in bin: lands on outfeed conveyor, moves ahead, clears space, and disappears into dispatch."""
        if b.get('state') == 'discharging':
            return
        if bname not in Y.BINS:
            bname = 'other' if 'other' in Y.BINS else list(Y.BINS.keys())[0]

        bx, by, bsize = Y.BINS[bname]

        # Place at the front intake of the outfeed belt (in front of the robot at -8.65m)
        b['x'] = float(bx + 0.55)  # -8.65m
        b['y'] = float(by)
        b['z'] = float(b['height'] / 2.0 + 0.02)
        b['state'] = 'discharging'
        b['bin'] = bname
        b['convey_speed'] = 0.85  # m/s conveying ahead towards and through the wall

        self.get_logger().info(
            f'📦 Placed {b["name"]} ({b.get("color", "").upper()}) onto {bname.upper()} outfeed belt! '
            f'Conveying ahead to clear space and dispatching...'
        )

    def place_box_on_floor(self, b: dict, x: float, y: float):
        """Place box on the floor: stack up if placed directly on an existing box, or move aside."""
        r_b = max(b.get('width', 0.4), b.get('depth', 0.4)) * 0.5
        top_box = None
        highest_z = 0.0
        for name, other in self.boxes.items():
            if other is b or other.get('state') == 'carried':
                continue
            r_o = max(other.get('width', 0.4), other.get('depth', 0.4)) * 0.5
            dist = math.hypot(other['x'] - x, other['y'] - y)
            if dist < (r_b + r_o) * 0.45:
                top_surface = other['z'] + other['height'] / 2.0
                if top_surface > highest_z:
                    highest_z = top_surface
                    top_box = other

        if top_box is not None:
            b['x'] = top_box['x']
            b['y'] = top_box['y']
            b['z'] = float(highest_z + b['height'] / 2.0)
            b['state'] = 'floor'
            self.get_logger().info(f'🧱 Placed {b["name"]} stacked up on top of {top_box["name"]} at z={b["z"]:.2f}m')
        else:
            b['x'] = float(x)
            b['y'] = float(y)
            b['z'] = float(b['height'] / 2.0 + 0.01)
            b['state'] = 'floor'

    def _resolve_box_collisions(self):
        """Physics collision resolution between boxes on the floor:
        1. If one box is stacked vertically on another, keep it resting on top.
        2. If boxes overlap horizontally at the same height, push them smoothly aside (move aside).
        3. If a box is dropped directly on top of another box, stack up like real life.
        """
        floor_boxes = [b for b in self.boxes.values() if b.get('state') == 'floor']
        n = len(floor_boxes)
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = floor_boxes[i], floor_boxes[j]

                dx = b2['x'] - b1['x']
                dy = b2['y'] - b1['y']
                dist_xy = math.hypot(dx, dy)
                r1 = max(b1['width'], b1['depth']) * 0.5
                r2 = max(b2['width'], b2['depth']) * 0.5
                min_xy = (r1 + r2) * 1.02

                if dist_xy >= min_xy:
                    continue

                z1, z2 = b1['z'], b2['z']
                h1, h2 = b1['height'], b2['height']
                z_dist = abs(z1 - z2)
                stacked_z_dist = (h1 + h2) * 0.5

                # Case 1: Already stably stacked vertically
                if abs(z_dist - stacked_z_dist) < 0.05:
                    continue

                # Case 2: Direct drop on top of another box -> stack up!
                if dist_xy < min_xy * 0.45 and z_dist < stacked_z_dist:
                    if z2 >= z1:
                        b2['z'] = float(z1 + stacked_z_dist)
                    else:
                        b1['z'] = float(z2 + stacked_z_dist)
                    continue

                # Case 3: Horizontal collision at same elevation -> move aside!
                if z_dist < stacked_z_dist:
                    if dist_xy < 0.001:
                        dx, dy = 0.02, 0.0
                        dist_xy = 0.02
                    overlap = min_xy - dist_xy
                    nx = dx / dist_xy
                    ny = dy / dist_xy
                    push_step = min(overlap, 0.05)
                    b1['x'] -= float(0.5 * push_step * nx)
                    b1['y'] -= float(0.5 * push_step * ny)
                    b2['x'] += float(0.5 * push_step * nx)
                    b2['y'] += float(0.5 * push_step * ny)

                    b1['x'] = max(-Y.HALF_X + 0.4, min(Y.HALF_X - 0.4, b1['x']))
                    b1['y'] = max(-Y.HALF_Y + 0.4, min(Y.HALF_Y - 0.4, b1['y']))
                    b2['x'] = max(-Y.HALF_X + 0.4, min(Y.HALF_X - 0.4, b2['x']))
                    b2['y'] = max(-Y.HALF_Y + 0.4, min(Y.HALF_Y - 0.4, b2['y']))

    def on_cmd_finger(self, msg: Float64):
        val = max(0.0, min(robot_gen.FINGER_TRAVEL, float(msg.data)))
        self.joint_target['finger_left_joint'] = val
        self.joint_target['finger_right_joint'] = val

        # Check kinematic grasp / release based on finger command
        palm_x, palm_y, palm_z = self._compute_world_palm()
        if val < 0.42 and self.carried_box_name is None:
            best_name = None
            best_dist = 0.85
            for name, b in self.boxes.items():
                if b.get('state') in ('sorted', 'discharging', 'conveying_in'):
                    continue
                d = math.hypot(b['x'] - palm_x, b['y'] - palm_y)
                if d < best_dist:
                    best_dist = d
                    best_name = name
            if best_name is not None:
                self.carried_box_name = best_name
                self.boxes[best_name]['state'] = 'carried'
                self.get_logger().info(f'🦾 Gripper attached to {best_name} at ({palm_x:.2f}, {palm_y:.2f}, {palm_z:.2f})')
        elif val >= 0.42 and self.carried_box_name is not None:
            b = self.boxes.get(self.carried_box_name)
            if b is not None and b.get('state') != 'sorted':
                best_bin = None
                best_d = 2.2
                for bin_name, (bx, by, bsize) in Y.BINS.items():
                    d = math.hypot(palm_x - bx, palm_y - by)
                    if d < best_d:
                        best_d = d
                        best_bin = bin_name
                if best_bin is not None:
                    self.place_box_in_bin(b, best_bin)
                else:
                    self.place_box_on_floor(b, palm_x, palm_y)
            self.carried_box_name = None

    def on_box_ready(self, msg: String):
        try:
            d = json.loads(msg.data)
            size = float(d.get('size', 0.4))
            width = float(d.get('width', size))
            depth = float(d.get('depth', size))
            height = float(d.get('height', size))
            name = d['name']
            target_x = float(d['x'])
            target_y = float(d['y'])

            # Conveyor infeed animation: box starts from outer wall feeder and glides in along conveyor
            feed_dir = -1.0 if target_y > 0 else 1.0
            start_y = float(5.95 if target_y > 0 else -5.95)

            # Nudge any existing box on this landing pad forward along the staging lane
            for other_name, other in list(self.boxes.items()):
                if other.get('state') != 'floor':
                    continue
                d_drop = math.hypot(other['x'] - target_x, other['y'] - target_y)
                if d_drop < 1.10:
                    other['y'] = float(target_y + feed_dir * 1.15)
                    self.get_logger().info(
                        f'➡️ Advanced existing {other_name} forward along conveyor lane to ({other["x"]:.2f}, {other["y"]:.2f}) '
                        f'to make room for incoming {name}!'
                    )

            b = {
                'name': name,
                'x': target_x,
                'y': start_y,  # Start at conveyor wall entrance
                'z': float(height / 2.0 + 0.115),  # Elevated on conveyor belt track
                'target_y': target_y,
                'feed_dir': feed_dir,
                'convey_speed': 0.85,
                'size': size,
                'width': width,
                'depth': depth,
                'height': height,
                'mass': float(d.get('mass', 1.0)),
                'color': d.get('color', 'red'),
                'damaged': bool(d.get('damaged', False)),
                'state': 'conveying_in',  # Smooth incoming conveyor belt animation
            }

            self.boxes[name] = b
            self.get_logger().info(
                f'📦 Registered box {name} ({d.get("color")}, {d.get("mass"):.1f}kg, '
                f'{width:.2f}x{depth:.2f}x{height:.2f}m) gliding IN along conveyor belt -> landing pad ({target_x:.2f}, {target_y:.2f})'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to register box: {e}')

    def on_box_pose(self, msg: String):
        """Handle explicit box pose updates from sorter_brain."""
        try:
            d = json.loads(msg.data)
            name = d.get('name')
            if name in self.boxes:
                b = self.boxes[name]
                state = d.get('state', 'carried')
                if state == 'released':
                    bx, by = float(d['x']), float(d['y'])
                    best_bin = None
                    best_d = 2.5
                    for bin_name, (bin_x, bin_y, bsize) in Y.BINS.items():
                        d_bin = math.hypot(bx - bin_x, by - bin_y)
                        if d_bin < best_d:
                            best_d = d_bin
                            best_bin = bin_name
                    if best_bin is not None:
                        self.place_box_in_bin(b, best_bin)
                    else:
                        self.place_box_on_floor(b, bx, by)
                    if self.carried_box_name == name:
                        self.carried_box_name = None
                else:
                    self.carried_box_name = name
                    b['state'] = 'carried'
                    b['x'] = float(d['x'])
                    b['y'] = float(d['y'])
                    b['z'] = float(d['z'])
        except Exception:
            pass

    def _compute_world_palm(self):
        r, z = arm_ik.fk(self.joint_pos['shoulder_joint'], self.joint_pos['elbow_joint'])
        world_ang = self.ryaw + self.joint_pos['turret_joint']
        px = self.rx + r * math.cos(world_ang)
        py = self.ry + r * math.sin(world_ang)
        return px, py, z

    def sim_step(self):
        now = self.get_clock().now().to_msg()

        # 1. Integrate diff-drive robot pose
        self.rx += self.vx * math.cos(self.ryaw) * self.dt
        self.ry += self.vx * math.sin(self.ryaw) * self.dt
        self.ryaw = math.atan2(
            math.sin(self.ryaw + self.vyaw * self.dt),
            math.cos(self.ryaw + self.vyaw * self.dt)
        )

        # Real-world solid obstacle collision physics:
        # A) Obstacle Poles (repel robot on physical contact so it never penetrates or collides inside poles)
        robot_col_radius = 0.65  # chassis collision radius
        for px, py in Y.POLES:
            dx = self.rx - px
            dy = self.ry - py
            dist = math.hypot(dx, dy)
            min_dist = Y.POLE_R + robot_col_radius  # 0.30 + 0.65 = 0.95m
            if dist < min_dist and dist > 1e-4:
                overlap = min_dist - dist
                nx = dx / dist
                ny = dy / dist
                self.rx += float(overlap * nx)
                self.ry += float(overlap * ny)

        # B) Sorting Bins (left wall lip at x = -8.4m; standoff at -7.8m; robot cannot penetrate into bin walls)
        self.rx = max(-7.82, min(Y.HALF_X - 0.60, self.rx))
        self.ry = max(-Y.HALF_Y + 0.60, min(Y.HALF_Y - 0.60, self.ry))

        # C) Front bumper push: if robot contacts loose floor boxes, nudge them aside like real bumper
        for name, b in self.boxes.items():
            if b.get('state') == 'floor':
                dx = b['x'] - self.rx
                dy = b['y'] - self.ry
                dist = math.hypot(dx, dy)
                min_clear = robot_col_radius + max(b['width'], b['depth']) * 0.5
                if dist < min_clear and dist > 1e-4:
                    overlap = min_clear - dist
                    nx = dx / dist
                    ny = dy / dist
                    b['x'] += float(overlap * nx)
                    b['y'] += float(overlap * ny)

        # Spin wheels visually
        v_left = self.vx - 0.5 * robot_gen.WHEEL_SEP * self.vyaw
        v_right = self.vx + 0.5 * robot_gen.WHEEL_SEP * self.vyaw
        self.joint_pos['left_wheel_joint'] += (v_left / robot_gen.WHEEL_R) * self.dt
        self.joint_pos['right_wheel_joint'] += (v_right / robot_gen.WHEEL_R) * self.dt

        # 2. Smoothly step controlled joints toward targets
        for j in ('blade_joint', 'turret_joint', 'shoulder_joint', 'elbow_joint',
                  'wrist_joint', 'finger_left_joint', 'finger_right_joint'):
            diff = self.joint_target[j] - self.joint_pos[j]
            max_step = 3.5 * self.dt
            if abs(diff) <= max_step:
                self.joint_pos[j] = self.joint_target[j]
            else:
                self.joint_pos[j] += math.copysign(max_step, diff)

        # 3. If carrying a box, keep it attached to the live 3D gripper palm
        if self.carried_box_name and self.carried_box_name in self.boxes:
            b = self.boxes[self.carried_box_name]
            px, py, pz = self._compute_world_palm()
            b['x'] = px
            b['y'] = py
            b['z'] = max(b['height'] / 2.0 + 0.05, pz - 0.055 - b['height'] / 2.0)

        # 3a. Infeed conveyor: incoming boxes glide in along the belt from outer wall feeder onto landing pad
        for name, b in list(self.boxes.items()):
            if b.get('state') == 'conveying_in':
                speed = float(b.get('convey_speed', 0.85))
                feed_dir = float(b.get('feed_dir', -1.0 if b.get('target_y', 0.0) > 0 else 1.0))
                target_y = float(b['target_y'])
                b['y'] += float(feed_dir * speed * self.dt)
                b['z'] = float(b['height'] / 2.0 + 0.115)
                # Check if arrived at landing pad
                if (feed_dir < 0 and b['y'] <= target_y) or (feed_dir > 0 and b['y'] >= target_y):
                    b['y'] = target_y
                    b['z'] = float(b['height'] / 2.0 + 0.02)
                    b['state'] = 'floor'
                    self.get_logger().info(
                        f'✅ Box {name} arrived at landing pad ({b["x"]:.2f}, {b["y"]:.2f}) ready for sorting!'
                    )

        # 3b. Physics collision resolution: boxes move aside or stack up like real life
        self._resolve_box_collisions()

        # 3c. Outfeed conveyor: boxes placed in sorting bins move ahead like a belt and disappear
        to_remove = []
        for name, b in list(self.boxes.items()):
            if b.get('state') == 'discharging':
                speed = float(b.get('convey_speed', 0.85))
                # Move ahead in -x direction (toward and through the wall into dispatch)
                b['x'] -= float(speed * self.dt)
                # Once it moves through the wall and chute, it disappears!
                if b['x'] <= -10.6:
                    to_remove.append(name)
                    bname = b.get('bin', 'other')
                    self.dispatched_counts[bname] = self.dispatched_counts.get(bname, 0) + 1
                    self.get_logger().info(
                        f'🚀 Box {name} ({b.get("color", "").upper()}) conveyed ahead and DISAPPEARED into {bname.upper()} dispatch! '
                        f'(Total {bname} dispatched: {self.dispatched_counts[bname]})'
                    )

        for name in to_remove:
            self.boxes.pop(name, None)
            if self.carried_box_name == name:
                self.carried_box_name = None

        # 4. Publish /tf (odom -> base_link and world -> odom)
        qx, qy, qz, qw = _yaw_to_quat(self.ryaw)

        t_world = TransformStamped()
        t_world.header.stamp = now
        t_world.header.frame_id = 'world'
        t_world.child_frame_id = 'odom'
        t_world.transform.rotation.w = 1.0

        t_base = TransformStamped()
        t_base.header.stamp = now
        t_base.header.frame_id = 'odom'
        t_base.child_frame_id = 'base_link'
        t_base.transform.translation.x = self.rx
        t_base.transform.translation.y = self.ry
        t_base.transform.translation.z = 0.0
        t_base.transform.rotation.x = qx
        t_base.transform.rotation.y = qy
        t_base.transform.rotation.z = qz
        t_base.transform.rotation.w = qw

        self.tf_pub.publish(TFMessage(transforms=[t_world, t_base]))

        # 5. Publish /world_poses (base_link + all boxes)
        wp = [t_base]
        for name, b in self.boxes.items():
            tb = TransformStamped()
            tb.header.stamp = now
            tb.header.frame_id = 'odom'
            tb.child_frame_id = name
            tb.transform.translation.x = b['x']
            tb.transform.translation.y = b['y']
            tb.transform.translation.z = b['z']
            tb.transform.rotation.w = 1.0
            wp.append(tb)
        self.world_poses_pub.publish(TFMessage(transforms=wp))

        # 6. Publish /joint_states
        js = JointState()
        js.header.stamp = now
        js.name = list(self.joint_pos.keys())
        js.position = [self.joint_pos[k] for k in js.name]
        self.joint_pub.publish(js)

        # 7. Publish 3D markers for RViz
        self._publish_markers(now)

    def _make_marker(self, now, ns: str, mid: int, mtype: int,
                     x: float, y: float, z: float,
                     sx: float, sy: float, sz: float,
                     rgb, alpha: float = 1.0, text: str = '') -> Marker:
        m = Marker()
        m.header.stamp = now
        m.header.frame_id = 'odom'
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = float(sx)
        m.scale.y = float(sy)
        m.scale.z = float(sz)
        m.color.r = float(rgb[0])
        m.color.g = float(rgb[1])
        m.color.b = float(rgb[2])
        m.color.a = float(alpha)
        if text:
            m.text = text
        return m

    def _publish_markers(self, now):
        ma = MarkerArray()
        # Reset markers each frame so disappearing boxes vanish cleanly in RViz
        del_all = Marker()
        del_all.action = Marker.DELETEALL
        ma.markers.append(del_all)

        # Warehouse floor
        ma.markers.append(self._make_marker(
            now, 'yard', 0, Marker.CUBE,
            0.0, 0.0, -0.01, 2 * Y.HALF_X, 2 * Y.HALF_Y, 0.02,
            (0.72, 0.72, 0.70), 1.0
        ))
        # Factory Perimeter Safety Barriers with Hazard Trim (20m x 12m facility enclosure)
        # 1. Right exterior wall
        ma.markers.append(self._make_marker(
            now, 'walls', 0, Marker.CUBE,
            Y.HALF_X, 0.0, 0.22, 0.16, 2 * Y.HALF_Y, 0.44, (0.22, 0.24, 0.28), 1.0
        ))
        ma.markers.append(self._make_marker(
            now, 'wall_trim', 0, Marker.CUBE,
            Y.HALF_X, 0.0, 0.45, 0.18, 2 * Y.HALF_Y, 0.03, (0.95, 0.70, 0.05), 1.0
        ))
        # 2. Top & Bottom Wall Barrier Segments (between conveyor openings at x=-5 and x=5)
        wall_segments = [(-7.85, 4.3), (0.0, 8.6), (7.85, 4.3)]
        for idx, (sx_c, sx_w) in enumerate(wall_segments):
            # Top wall segments
            ma.markers.append(self._make_marker(
                now, 'walls', 10 + idx, Marker.CUBE,
                sx_c, Y.HALF_Y, 0.22, sx_w, 0.16, 0.44, (0.22, 0.24, 0.28), 1.0
            ))
            ma.markers.append(self._make_marker(
                now, 'wall_trim', 10 + idx, Marker.CUBE,
                sx_c, Y.HALF_Y, 0.45, sx_w, 0.18, 0.03, (0.95, 0.70, 0.05), 1.0
            ))
            # Bottom wall segments
            ma.markers.append(self._make_marker(
                now, 'walls', 20 + idx, Marker.CUBE,
                sx_c, -Y.HALF_Y, 0.22, sx_w, 0.16, 0.44, (0.22, 0.24, 0.28), 1.0
            ))
            ma.markers.append(self._make_marker(
                now, 'wall_trim', 20 + idx, Marker.CUBE,
                sx_c, -Y.HALF_Y, 0.45, sx_w, 0.18, 0.03, (0.95, 0.70, 0.05), 1.0
            ))
        # 3. Corner Safety Bollards (4 corners)
        corners = [(Y.HALF_X - 0.14, Y.HALF_Y - 0.14),
                   (Y.HALF_X - 0.14, -Y.HALF_Y + 0.14),
                   (-Y.HALF_X + 0.14, Y.HALF_Y - 0.14),
                   (-Y.HALF_X + 0.14, -Y.HALF_Y + 0.14)]
        for idx, (cx, cy) in enumerate(corners):
            ma.markers.append(self._make_marker(
                now, 'bollards', idx, Marker.CYLINDER,
                cx, cy, 0.30, 0.24, 0.24, 0.60, (0.95, 0.65, 0.05), 1.0
            ))
        # Sorting bins / outfeed conveyor chutes on the left wall
        t_now = self.get_clock().now().nanoseconds * 1e-9
        belt_speed = 0.85
        bin_tread_offset = (t_now * belt_speed) % 0.28

        for i, (bname, (bx, by, bsize)) in enumerate(Y.BINS.items()):
            rgb = Y.BIN_RGB.get(bname, (0.5, 0.5, 0.5))
            # Outfeed Conveyor Frame (Matching bin color)
            ma.markers.append(self._make_marker(
                now, 'bins', i, Marker.CUBE,
                bx, by, 0.015, bsize, bsize, 0.03, rgb, 0.95
            ))
            # Conveyor Track Bed (Dark Carbon Tread conveying ahead through wall)
            ma.markers.append(self._make_marker(
                now, 'bin_tracks', i, Marker.CUBE,
                bx - 0.20, by, 0.035, bsize + 0.40, bsize * 0.75, 0.015, (0.12, 0.13, 0.16), 1.0
            ))
            # Animated moving conveyor treads gliding into the wall (-x)
            for ti in range(4):
                rib_x = (bx + 0.35) - ((ti * 0.28 + bin_tread_offset) % 1.2)
                ma.markers.append(self._make_marker(
                    now, 'bin_treads', i * 10 + ti, Marker.CUBE,
                    rib_x, by, 0.045, 0.05, bsize * 0.65, 0.005,
                    (0.25, 0.27, 0.32), 0.9
                ))
            # Clean bin label: DISPATCHED COUNT (NO 'chute', clearly visible font, no overlap)
            count = self.dispatched_counts.get(bname, 0)
            label_text = f'{bname.upper()} | Dispatched: {count}'
            ma.markers.append(self._make_marker(
                now, 'bin_labels', i, Marker.TEXT_VIEW_FACING,
                bx + 0.15, by, 0.38, 0.0, 0.0, 0.17, (1.0, 1.0, 1.0), 1.0,
                text=label_text
            ))

        # Obstacle poles
        for i, (px, py) in enumerate(Y.POLES):
            ma.markers.append(self._make_marker(
                now, 'poles', i, Marker.CYLINDER,
                px, py, 1.0, 2 * Y.POLE_R, 2 * Y.POLE_R, 2.0,
                (0.9, 0.6, 0.0), 1.0
            ))

        # 4 High-Visibility Input Conveyor Belts (Vivid Industrial Orange, Track, and Landing Pads)
        in_tread_offset = (t_now * belt_speed) % 0.32
        for i, (ix, iy) in enumerate(getattr(Y, 'BELTS_4', Y.INPUTS[:4])):
            sy = 1.0 if iy > 0 else -1.0
            # Main Conveyor Frame (Vivid Safety Orange - never blends with grey floor!)
            ma.markers.append(self._make_marker(
                now, 'conveyor_frame', i, Marker.CUBE,
                ix, sy * (Y.HALF_Y - 0.65), 0.05, 1.15, 1.7, 0.10,
                (1.0, 0.42, 0.04), 1.0
            ))
            # Conveyor Track Bed (Dark Carbon Tread)
            ma.markers.append(self._make_marker(
                now, 'conveyor_track', i, Marker.CUBE,
                ix, sy * (Y.HALF_Y - 0.65), 0.105, 0.88, 1.6, 0.02,
                (0.12, 0.13, 0.16), 1.0
            ))
            # Animated conveyor treads gliding incoming boxes from wall toward landing pad
            for ti in range(4):
                rib_y = sy * (Y.HALF_Y - 0.20 - ((ti * 0.32 + in_tread_offset) % 1.25))
                ma.markers.append(self._make_marker(
                    now, 'conveyor_treads', i * 10 + ti, Marker.CUBE,
                    ix, rib_y, 0.116, 0.78, 0.05, 0.005,
                    (0.25, 0.27, 0.32), 0.95
                ))
            # Landing Pad at Drop Point (Glowing Cyan/Teal)
            ma.markers.append(self._make_marker(
                now, 'drop_pads', i, Marker.CUBE,
                ix, iy, 0.02, 0.95, 0.95, 0.04,
                (0.05, 0.85, 0.95), 0.9
            ))
            # Belt label at back frame
            ma.markers.append(self._make_marker(
                now, 'conveyor_labels', i, Marker.TEXT_VIEW_FACING,
                ix, sy * (Y.HALF_Y - 0.20), 0.38, 0.0, 0.0, 0.18,
                (1.0, 0.92, 0.2), 1.0,
                text=f"BELT {i+1}"
            ))

        # Dynamic boxes: display live box cube and pure length dimension label (no box name)
        for i, (name, b) in enumerate(self.boxes.items()):
            rgb = Y.BOX_RGB.get(b['color'], (0.8, 0.8, 0.8))
            ma.markers.append(self._make_marker(
                now, 'boxes', i, Marker.CUBE,
                b['x'], b['y'], b['z'],
                b['width'], b['depth'], b['height'],
                rgb, 1.0
            ))
            # Clean length dimension only (e.g. 0.40m), increased font, clearly visible
            dim_label = f"{b['width']:.2f}m"
            ma.markers.append(self._make_marker(
                now, 'box_labels', i, Marker.TEXT_VIEW_FACING,
                b['x'], b['y'], b['z'] + b['height'] / 2.0 + 0.16,
                0.0, 0.0, 0.17, (1.0, 1.0, 1.0), 1.0,
                text=dim_label
            ))

        # Live Robot State Status Label (follows directly above robot chassis)
        status_color = (0.10, 1.0, 0.45) if self.carried_box_name else (
            (0.15, 0.88, 1.0) if any(k in self.brain_status for k in ('NAVIGATING', 'APPROACHING', 'TRANSPORTING')) else (1.0, 0.85, 0.2)
        )
        ma.markers.append(self._make_marker(
            now, 'robot_status', 0, Marker.TEXT_VIEW_FACING,
            self.rx, self.ry, 1.48, 0.0, 0.0, 0.16, status_color, 1.0,
            text=f"BOT: {self.brain_status}"
        ))

        # Gripper Lock LED & Targeting Beams
        px, py, pz = self._compute_world_palm()
        if self.carried_box_name and self.carried_box_name in self.boxes:
            b_carried = self.boxes[self.carried_box_name]
            # Active Lock LED Indicator (Bright Glowing Neon Green on wrist palm)
            ma.markers.append(self._make_marker(
                now, 'gripper_led', 0, Marker.SPHERE,
                px, py, pz + 0.035, 0.07, 0.07, 0.07, (0.05, 1.0, 0.30), 0.95
            ))
            # Left & Right locking pins / visual alignment beams down to box
            for side_idx, sy in enumerate((1, -1)):
                ma.markers.append(self._make_marker(
                    now, 'gripper_beam', side_idx, Marker.CYLINDER,
                    px, py + sy * (b_carried['width'] / 2.0 + 0.02), pz - 0.07,
                    0.02, 0.02, 0.14, (0.0, 1.0, 0.4), 0.85
                ))
        else:
            # Standby Status LED Indicator (Amber/Cyan standby glow)
            ma.markers.append(self._make_marker(
                now, 'gripper_led', 0, Marker.SPHERE,
                px, py, pz + 0.035, 0.045, 0.045, 0.045, (0.2, 0.6, 0.85), 0.70
            ))
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = RvizSimNode()
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
