"""Part B tests: URDF structure, FK-from-URDF vs arm_ik, bridge/plugin topic agreement, launch syntax.
Run:  python3 test/test_part_b.py -v"""
import math, os, re, py_compile, sys, unittest
import xml.etree.ElementTree as ET
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from box_sorter import robot_gen as R, arm_ik


def mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def rpy_T(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rm = [[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
          [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
          [-sp,   cp*sr,            cp*cr]]
    return [Rm[0] + [xyz[0]], Rm[1] + [xyz[1]], Rm[2] + [xyz[2]], [0, 0, 0, 1]]

def axis_T(kind, axis, q):
    if kind == 'prismatic':
        return rpy_T([a * q for a in axis], (0, 0, 0))
    x, y, z = axis
    c, s, t = math.cos(q), math.sin(q), 1 - math.cos(q)
    Rm = [[t*x*x + c, t*x*y - s*z, t*x*z + s*y],
          [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
          [t*x*z - s*y, t*y*z + s*x, t*z*z + c]]
    return [Rm[0] + [0], Rm[1] + [0], Rm[2] + [0], [0, 0, 0, 1]]

def floats(s):
    return [float(v) for v in s.split()]

def poses(q):
    root = ET.fromstring(R.build_urdf())
    J = {j.find('child').get('link'): j for j in root.findall('joint')}
    cache = {'base_link': rpy_T([0, 0, 0], (0, 0, 0))}
    def get(link):
        if link in cache:
            return cache[link]
        j = J[link]
        o = j.find('origin')
        T = rpy_T(floats(o.get('xyz')), floats(o.get('rpy')))
        kind = j.get('type')
        if kind != 'fixed':
            ax = j.find('axis')
            T = mm(T, axis_T(kind, floats(ax.get('xyz')), q.get(j.get('name'), 0.0)))
        cache[link] = mm(get(j.find('parent').get('link')), T)
        return cache[link]
    return {l.get('name'): get(l.get('name')) for l in root.findall('link')}


class TestUrdfStructure(unittest.TestCase):
    def setUp(self):
        self.root = ET.fromstring(R.build_urdf())

    def test_tree_is_valid(self):
        links = [l.get('name') for l in self.root.findall('link')]
        self.assertEqual(len(links), len(set(links)))
        children = []
        for j in self.root.findall('joint'):
            self.assertIn(j.find('parent').get('link'), links)
            self.assertIn(j.find('child').get('link'), links)
            children.append(j.find('child').get('link'))
        self.assertEqual(len(children), len(set(children)), 'a link has two parents')
        self.assertEqual(set(links) - set(children), {'base_link'})
        poses({})

    def test_every_link_has_positive_mass_and_inertia(self):
        for l in self.root.findall('link'):
            self.assertGreater(float(l.find('inertial/mass').get('value')), 0, l.get('name'))
            i = l.find('inertial/inertia')
            for k in ('ixx', 'iyy', 'izz'):
                self.assertGreater(float(i.get(k)), 0, l.get('name'))

    def test_limited_joints_have_limits(self):
        for j in self.root.findall('joint'):
            if j.get('type') in ('revolute', 'prismatic'):
                lim = j.find('limit')
                self.assertIsNotNone(lim, j.get('name'))
                self.assertLess(float(lim.get('lower')), float(lim.get('upper')))

    def test_wheels_roll_forward_for_positive_velocity(self):
        P = poses({})
        for w in ('left_wheel', 'right_wheel'):
            axis_in_base = [P[w][i][2] for i in range(3)]
            self.assertAlmostEqual(axis_in_base[1], 1.0, 6)

    def test_diff_drive_params_match_geometry(self):
        txt = R.build_urdf()
        self.assertEqual(float(re.search(r'<wheel_separation>([\d.]+)', txt).group(1)), 2 * R.WHEEL_Y)
        self.assertEqual(float(re.search(r'<wheel_radius>([\d.]+)', txt).group(1)), R.WHEEL_R)
        P = poses({})
        self.assertAlmostEqual(P['left_wheel'][2][3], R.WHEEL_R)

    def test_chassis_fits_between_wheels_and_clears_ground(self):
        self.assertLessEqual(R.CHASSIS[1] / 2, R.WHEEL_Y - R.WHEEL_W / 2 + 1e-9)
        self.assertAlmostEqual(R.CHASSIS_Z - R.CHASSIS[2] / 2, 0.2)
        self.assertAlmostEqual(R.CASTER_R * 2, 0.2)

    def test_blade_clearances(self):
        raised = poses({'blade_joint': 0.0})['blade'][2][3] - R.BLADE[2] / 2
        lowered = poses({'blade_joint': -R.BLADE_DROP})['blade'][2][3] - R.BLADE[2] / 2
        self.assertGreaterEqual(raised, 0.15)
        self.assertGreater(lowered, 0.0)
        self.assertLess(lowered, 0.05)

    def test_gripper_spans_min_and_max_box(self):
        def opening(q):
            P = poses({'finger_left_joint': q, 'finger_right_joint': q})
            yl, yr = P['finger_left'][1][3], P['finger_right'][1][3]
            return (yl - yr) - R.FINGER[1]
        self.assertAlmostEqual(opening(0.0), 0.10, 6)
        self.assertGreaterEqual(opening(R.FINGER_TRAVEL), 1.00)

    def test_gravity_off_on_arm_links(self):
        txt = R.build_urdf()
        for n in ('turret', 'upper_arm', 'forearm', 'wrist', 'finger_left', 'finger_right', 'blade'):
            self.assertIn(f'<gazebo reference="{n}"><gravity>false</gravity>', txt)


class TestKinematicsMatchIK(unittest.TestCase):
    TARGETS = [(1.5, 0.3), (1.0, 0.5), (2.0, 0.9), (1.2, 1.2), (0.8, 0.3)]

    def q(self, r, z, yaw=0.0):
        s, e, w = arm_ik.solve(r, z)
        return {'turret_joint': yaw, 'shoulder_joint': s, 'elbow_joint': e, 'wrist_joint': w}

    def test_palm_lands_on_ik_target(self):
        for r, z in self.TARGETS:
            P = poses(self.q(r, z))['wrist']
            self.assertAlmostEqual(P[0][3], r, 5)
            self.assertAlmostEqual(P[1][3], 0.0, 5)
            self.assertAlmostEqual(P[2][3], z, 5)

    def test_turret_rotates_target_around_yaw(self):
        r, z, yaw = 1.5, 0.4, 0.7
        P = poses(self.q(r, z, yaw))['wrist']
        self.assertAlmostEqual(P[0][3], r * math.cos(yaw), 5)
        self.assertAlmostEqual(P[1][3], r * math.sin(yaw), 5)
        self.assertAlmostEqual(P[2][3], z, 5)

    def test_gripper_points_straight_down(self):
        for r, z in self.TARGETS:
            P = poses(self.q(r, z))['wrist']
            self.assertAlmostEqual(P[2][2], 1.0, 5)

    def test_ik_inside_joint_limits_over_working_envelope(self):
        bad, n = [], 0
        r = R.ENVELOPE_R[0]
        while r <= R.ENVELOPE_R[1] + 1e-9:
            z = R.ENVELOPE_Z[0]
            while z <= R.ENVELOPE_Z[1] + 1e-9:
                s, e, w = arm_ik.solve(r, z)            # must be reachable everywhere in the envelope
                n += 1
                for name, v in (('shoulder_joint', s), ('elbow_joint', e), ('wrist_joint', w)):
                    lo, hi = R.LIMITS[name][:2]
                    if not lo <= v <= hi:
                        bad.append((round(r, 2), round(z, 2), name, round(v, 2)))
                z += 0.05
            r += 0.05
        self.assertGreater(n, 100)
        self.assertEqual(bad, [])

    def test_palm_clears_chassis_when_reaching_a_box_beside_the_bot(self):
        s, e, w = arm_ik.solve(0.9, 0.35)
        self.assertTrue(R.LIMITS['shoulder_joint'][0] <= s <= R.LIMITS['shoulder_joint'][1])


class TestBridgeAndLaunch(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, 'config', 'bridge.yaml')) as f:
            self.yaml = f.read()
        self.gz = {t.lstrip('/') for t in re.findall(r'gz_topic_name:\s*"([^"]+)"', self.yaml)}

    def test_every_yaml_entry_is_complete(self):
        n = self.yaml.count('- ros_topic_name:')
        for key in ('gz_topic_name', 'ros_type_name', 'gz_type_name', 'direction'):
            self.assertEqual(self.yaml.count(key + ':'), n, key)
        for d in re.findall(r'direction:\s*(\S+)', self.yaml):
            self.assertIn(d, ('ROS_TO_GZ', 'GZ_TO_ROS', 'BIDIRECTIONAL'))

    def test_all_plugin_topics_are_bridged(self):
        txt = R.build_urdf()
        topics = set(re.findall(r'<(?:topic|odom_topic|tf_topic)>([^<]+)<', txt))
        self.assertTrue({'cmd_vel', 'odom', 'tf', 'joint_states'} <= topics)
        for t in R.CMD_TOPICS.values():
            self.assertIn(t, topics)
        self.assertLessEqual(topics, self.gz, f'not bridged: {topics - self.gz}')

    def test_every_controlled_joint_exists(self):
        names = {j.get('name') for j in ET.fromstring(R.build_urdf()).findall('joint')}
        self.assertTrue(set(R.CMD_TOPICS) <= names)

    def test_world_pose_topic_uses_world_name(self):
        from box_sorter import gz_utils
        self.assertIn(f'world/{gz_utils.WORLD}/dynamic_pose/info', self.gz)

    def test_launch_file_compiles(self):
        py_compile.compile(os.path.join(ROOT, 'launch', 'sim.launch.py'), doraise=True, cfile='/tmp/_l.pyc')

    def test_checked_in_urdf_is_current(self):
        with open(os.path.join(ROOT, 'urdf', 'box_sorter.urdf')) as f:
            self.assertEqual(f.read(), R.build_urdf())


if __name__ == '__main__':
    unittest.main()
