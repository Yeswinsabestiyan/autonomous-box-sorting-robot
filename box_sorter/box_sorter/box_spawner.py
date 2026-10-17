"""Part C: random box feeder for the 8 input conveyors (RViz & Gazebo compatible).

Every `spawn_period` seconds it picks a free input, makes a random grippable box (size / mass / colour),
waits for the belt travel time, then spawns the box at that input's drop point and publishes its
metadata as JSON on /box_ready (std_msgs/String) so the brain and RViz know size, width, depth, height,
mass, colour, damaged, and 3D coordinates (x, y, z).
"""
import json
import math
import os
import random
import re
import threading
from dataclasses import dataclass, asdict

from . import yard as Y
from . import gz_utils

BELT_LEN = 3.0               # belt visuals are 3 m long
BELT_SPEED = 0.5             # m/s -> 6 s from belt start to drop point
SDF_DIR = '/tmp/box_sorter_boxes'
BOX_NAME = re.compile(r'^box_\d+$')

COLORS = [c for c in Y.BOX_RGB if c != 'gray']      # gray is reserved for damaged boxes
SORT_WEIGHT = {'red': 3, 'green': 3, 'blue': 3}     # sortable colours twice as likely as the rest


@dataclass(frozen=True)
class BoxSpec:
    name: str
    size: float
    mass: float
    color: str
    damaged: bool
    input: int
    x: float
    y: float
    z: float
    width: float = 0.0
    depth: float = 0.0
    height: float = 0.0

    def __post_init__(self):
        if self.width <= 0.0:
            object.__setattr__(self, 'width', self.size)
        if self.depth <= 0.0:
            object.__setattr__(self, 'depth', self.size)
        if self.height <= 0.0:
            object.__setattr__(self, 'height', self.size)

    def to_json(self):
        return json.dumps(asdict(self))


# ---- pure helpers ---------------------------------------------------------------------------
def belt_delay(speed=BELT_SPEED):
    return BELT_LEN / speed


def random_box(rng, p_big=0.15, p_heavy=0.10, p_damaged=0.10):
    """-> (size, mass, color, damaged). Sizes: 6 cm..1.4 m, mass 0.5..25 kg."""
    if rng.random() < p_big:
        size = rng.uniform(1.05, 1.4) if rng.random() < 0.6 else rng.uniform(0.06, 0.095)
    else:
        size = rng.uniform(0.18, 0.80)
    mass = rng.uniform(10.5, 25.0) if rng.random() < p_heavy else rng.uniform(0.8, 8.5)
    damaged = rng.random() < p_damaged
    if damaged:
        color = 'gray'
    else:
        color = rng.choices(COLORS, weights=[SORT_WEIGHT.get(c, 1) for c in COLORS])[0]
    return round(size, 3), round(mass, 2), color, damaged


def count_boxes(poses):
    """poses: [(frame_name, x, y)]. Number of distinct box models currently in the world."""
    return len({n for n, _, _ in poses if BOX_NAME.match(n)})


def drop_zone_clear(poses, drop, size, margin=0.6):
    """True if nothing (robot, another box) is within size/2+margin of the drop point."""
    r = size / 2 + margin
    return all(math.hypot(x - drop[0], y - drop[1]) > r for _, x, y in poses)


class SpawnPlanner:
    """Decides what to spawn next. No ROS, no gz, fully seedable."""

    def __init__(self, seed=None, p_big=0.15, p_heavy=0.10, p_damaged=0.10,
                 max_active=8, max_boxes=0, inputs=None):
        self.rng = random.Random(seed)
        self.p = (p_big, p_heavy, p_damaged)
        self.max_active, self.max_boxes = max_active, max_boxes
        self.inputs = inputs if inputs is not None else Y.INPUTS
        self.count = 0                      # boxes planned so far
        self.pending = {}                   # input index -> BoxSpec still on the belt
        self.lock = threading.Lock()

    def done(self):
        return self.max_boxes > 0 and self.count >= self.max_boxes

    def plan(self, poses):
        """Return the next BoxSpec (and reserve its input) or None if nothing should spawn now."""
        with self.lock:
            if self.done() or count_boxes(poses) + len(self.pending) >= self.max_active:
                return None
            size, mass, color, damaged = random_box(self.rng, *self.p)
            free = [i for i, (x, y) in enumerate(self.inputs)
                    if i not in self.pending and drop_zone_clear(poses, (x, y), size)]
            if not free:
                return None
            i = self.rng.choice(free)
            x, y = self.inputs[i]
            spec = BoxSpec(
                f'box_{self.count}', size, mass, color, damaged,
                i, x, y, size / 2 + 0.01,
                width=size, depth=size, height=size
            )
            self.count += 1
            self.pending[i] = spec
            return spec

    def delivered(self, spec):
        with self.lock:
            self.pending.pop(spec.input, None)


def write_box_sdf(spec, directory=SDF_DIR):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, spec.name + '.sdf')
    with open(path, 'w') as f:
        f.write(gz_utils.box_sdf(
            spec.name, spec.size, spec.mass, Y.BOX_RGB[spec.color],
            width=spec.width, depth=spec.depth, height=spec.height
        ))
    return path


def deliver(spec):
    """Spawn the box in Gazebo. -> (ok, message)."""
    r = gz_utils.spawn_file(write_box_sdf(spec), spec.name, spec.x, spec.y, spec.z)
    ok = r.returncode == 0 and 'true' in r.stdout
    return ok, (r.stdout + r.stderr).strip()


# ---- ROS node --------------------------------------------------------------------------------
def main(args=None):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    class BoxSpawner(Node):
        def __init__(self):
            super().__init__('box_spawner')
            g = self.declare_parameter
            self.period = g('spawn_period', 3.5).value
            seed = g('seed', -1).value
            self.speed = g('belt_speed', 1.0).value
            self.mode = g('mode', 'rviz').value
            num_belts = g('num_belts', 4).value
            active_inputs = Y.BELTS_4 if num_belts == 4 else Y.INPUTS
            self.planner = SpawnPlanner(
                seed=None if seed < 0 else seed,
                p_big=0.0,
                p_heavy=0.0,
                p_damaged=g('p_damaged', 0.10).value,
                max_active=g('max_active', 50).value,
                max_boxes=g('max_boxes', 0).value,
                inputs=active_inputs)
            self.poses = []
            self.spawn_timers = []
            self.pub = self.create_publisher(String, '/box_ready', 50)
            self.create_subscription(TFMessage, '/world_poses', self.on_poses, 10)
            self.create_timer(self.period, self.tick)
            self.get_logger().info(
                f'box_spawner: running with {len(active_inputs)} conveyor belts, continuous feeder period={self.period}s, '
                f'mode={self.mode}, belt travel time={belt_delay(self.speed):.1f}s'
            )

        def on_poses(self, msg):
            # Only count boxes still sitting at drop points (leave drop points clear as soon as carried)
            self.poses = [
                (t.child_frame_id, t.transform.translation.x, t.transform.translation.y)
                for t in msg.transforms
                if t.transform.translation.x > -Y.HALF_X + 2.0
            ]

        def tick(self):
            spec = self.planner.plan(self.poses)
            if spec is None:
                if self.planner.done() and not self.planner.pending:
                    self.get_logger().info('max_boxes reached, spawner idle')
                return
            belt_num = spec.input + 1
            self.get_logger().info(
                f'📦 [BELT {belt_num}] Spawning {spec.name} ({spec.color.upper()}, {spec.mass:.1f}kg, '
                f'{spec.width:.2f}x{spec.depth:.2f}x{spec.height:.2f}m) -> drop point ({spec.x:.2f}, {spec.y:.2f})'
                + (' [DAMAGED]' if spec.damaged else '')
            )
            t = threading.Timer(belt_delay(self.speed), self.arrive, [spec])
            t.daemon = True
            self.spawn_timers.append(t)
            t.start()

        def arrive(self, spec):
            if self.mode == 'gz' or self.mode == 'gazebo':
                ok, msg = deliver(spec)
                if not ok:
                    self.get_logger().warn(f'Gazebo spawn of {spec.name} failed ({msg})')
            self.planner.delivered(spec)
            self.pub.publish(String(data=spec.to_json()))
            belt_num = spec.input + 1
            self.get_logger().info(
                f'✅ [BELT {belt_num}] {spec.name} ARRIVED at ({spec.x:.2f}, {spec.y:.2f}) ready for pickup!'
            )

        def destroy_node(self):
            for t in self.spawn_timers:
                t.cancel()
            super().destroy_node()

    rclpy.init(args=args)
    node = BoxSpawner()
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
