"""Builds the robot URDF (+ Gazebo plugins) from Python so dimensions share arm_ik.py.

Frames (all metres, ground = z 0, +x forward):
  base_link          chassis, origin on the ground under the chassis centre
   |- left/right_wheel_joint   continuous, diff drive
   |- front/rear caster        fixed frictionless balls
   |- blade_joint              prismatic z, push pad at the front (0 = raised, negative = lowered)
   '- turret_joint             revolute z, on the chassis top
       '- shoulder_joint      pitch (+ = arm pitches DOWN, same sign convention as arm_ik)
           '- elbow_joint     pitch
               '- wrist_joint pitch, keeps the gripper vertical
                   '- finger_left/right_joint   prismatic, symmetric opening
The wrist link origin IS the palm point that arm_ik.solve() targets.
"""
import math
import os
from . import arm_ik

# ---- dimensions ------------------------------------------------------------------------------
CHASSIS = (1.2, 0.9, 0.3)
CHASSIS_Z = 0.35                              # centre height (bottom 0.2, top 0.5)
CHASSIS_TOP = CHASSIS_Z + CHASSIS[2] / 2
WHEEL_R, WHEEL_W, WHEEL_Y = 0.2, 0.1, 0.5
WHEEL_SEP = 2 * WHEEL_Y
CASTER_R, CASTER_X = 0.1, 0.45
TURRET_R = 0.15
TURRET_H = arm_ik.SHOULDER_Z - CHASSIS_TOP    # shoulder sits exactly SHOULDER_Z above ground
LINK_T = 0.12
PALM = (0.3, 0.2, 0.05)
FINGER = (0.3, 0.03, 0.2)                     # x, y, z
FINGER_Z = -(PALM[2] + FINGER[2] / 2)
FINGER_Y0 = 0.05 + FINGER[1] / 2              # closed: inner faces 0.10 m apart (min box size)
FINGER_TRAVEL = 0.5                           # open: inner faces 1.10 m apart (max box size 1.0)
BLADE = (0.05, 1.2, 0.4)
BLADE_X, BLADE_Z, BLADE_DROP = 0.75, 0.40, 0.18   # lowered: bottom edge 2 cm off the floor

LIMITS = {   # (lower, upper, effort, velocity)
    'turret_joint': (-math.pi, math.pi, 500.0, 1.5),
    'shoulder_joint': (-1.8, 1.8, 800.0, 1.0),
    'elbow_joint': (0.0, 3.0, 500.0, 1.0),
    'wrist_joint': (-math.pi, math.pi, 100.0, 2.0),
    'finger_left_joint': (0.0, FINGER_TRAVEL, 200.0, 0.5),
    'finger_right_joint': (0.0, FINGER_TRAVEL, 200.0, 0.5),
    'blade_joint': (-BLADE_DROP, 0.0, 1000.0, 0.3),
}
# Region the brain is allowed to ask arm_ik for (palm radius from turret axis, palm height).
# Outside it (close in AND high) the shoulder would need < -1.8 rad.
ENVELOPE_R = (0.9, 1.9)
ENVELOPE_Z = (0.2, 1.3)

# command topic (gz side) for each controlled joint. Both fingers share one topic.
CMD_TOPICS = {
    'turret_joint': 'cmd/turret', 'shoulder_joint': 'cmd/shoulder', 'elbow_joint': 'cmd/elbow',
    'wrist_joint': 'cmd/wrist', 'finger_left_joint': 'cmd/finger', 'finger_right_joint': 'cmd/finger',
    'blade_joint': 'cmd/blade',
}
ROBOT = 'box_sorter'


# ---- small helpers ---------------------------------------------------------------------------
def _f(*v):
    return ' '.join(f'{x:.6g}' for x in v)


def _box_I(m, x, y, z):
    return (m / 12 * (y * y + z * z), m / 12 * (x * x + z * z), m / 12 * (x * x + y * y))


def _cyl_I(m, r, l):
    return (m / 12 * (3 * r * r + l * l), m / 12 * (3 * r * r + l * l), m * r * r / 2)


def _sph_I(m, r):
    return (2 * m * r * r / 5,) * 3


def _geom(g):
    kind = g[0]
    if kind == 'box':
        return f'<box size="{_f(*g[1])}"/>'
    if kind == 'cyl':
        return f'<cylinder radius="{g[1]}" length="{g[2]}"/>'
    return f'<sphere radius="{g[1]}"/>'


def _inertia(g, m):
    if g[0] == 'box':
        return _box_I(m, *g[1])
    if g[0] == 'cyl':
        return _cyl_I(m, g[1], g[2])
    return _sph_I(m, g[1])


def link(name, mass, geom, at=(0, 0, 0), rgb=(.6, .6, .6), collide=False, extra_visuals=None):
    ix, iy, iz = _inertia(geom, mass)
    col = f'<collision><origin xyz="{_f(*at)}"/><geometry>{_geom(geom)}</geometry></collision>' if collide else ''
    vis = (f'<visual><origin xyz="{_f(*at)}"/><geometry>{_geom(geom)}</geometry>'
           f'<material name="m_{name}"><color rgba="{_f(*rgb)} 1"/></material></visual>')
    if extra_visuals:
        for idx, (ev_geom, ev_at, ev_rgb) in enumerate(extra_visuals):
            vis += (f'<visual><origin xyz="{_f(*ev_at)}"/><geometry>{_geom(ev_geom)}</geometry>'
                    f'<material name="m_{name}_sub{idx}"><color rgba="{_f(*ev_rgb)} 1"/></material></visual>')
    return (f'<link name="{name}">'
            f'<inertial><origin xyz="{_f(*at)}"/><mass value="{mass}"/>'
            f'<inertia ixx="{ix:.6g}" iyy="{iy:.6g}" izz="{iz:.6g}" ixy="0" ixz="0" iyz="0"/></inertial>'
            f'{vis}{col}</link>')


def joint(name, kind, parent, child, xyz, rpy=(0, 0, 0), axis=None):
    ax = f'<axis xyz="{_f(*axis)}"/>' if axis else ''
    lim = ''
    if name in LIMITS:
        lo, hi, eff, vel = LIMITS[name]
        lim = f'<limit lower="{lo:.6g}" upper="{hi:.6g}" effort="{eff}" velocity="{vel}"/>'
    elif kind == 'continuous':
        lim = '<limit effort="300" velocity="20"/>'
    return (f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{_f(*xyz)}" rpy="{_f(*rpy)}"/>{ax}{lim}'
            + ('<dynamics damping="1.0" friction="0.1"/>' if kind != 'fixed' else '') + '</joint>')


def _jpc(joint_name, topic, p=120.0, d=12.0, fmax=500.0):
    return ('<plugin filename="gz-sim-joint-position-controller-system" '
            'name="gz::sim::systems::JointPositionController">'
            f'<joint_name>{joint_name}</joint_name><topic>{topic}</topic>'
            f'<p_gain>{p}</p_gain><i_gain>0</i_gain><d_gain>{d}</d_gain>'
            f'<cmd_max>{fmax}</cmd_max><cmd_min>{-fmax}</cmd_min></plugin>')


# ---- the robot -------------------------------------------------------------------------------
def build_urdf():
    L1, L2 = arm_ik.L1, arm_ik.L2
    P = []
    # chassis (root). Visual/collision box is lifted so base_link sits on the ground.
    P.append(link('base_link', 40.0, ('box', CHASSIS), (0, 0, CHASSIS_Z), (.9, .65, .1), collide=True))
    # wheels: joint rpy (-pi/2,0,0) turns link +z into world +y, so +velocity drives forward
    for side, sy in (('left', 1), ('right', -1)):
        P.append(link(f'{side}_wheel', 3.0, ('cyl', WHEEL_R, WHEEL_W), rgb=(.1, .1, .1), collide=True))
        P.append(joint(f'{side}_wheel_joint', 'continuous', 'base_link', f'{side}_wheel',
                       (0, sy * WHEEL_Y, WHEEL_R), (-math.pi / 2, 0, 0), (0, 0, 1)))
    for n, sx in (('front', 1), ('rear', -1)):
        P.append(link(f'{n}_caster', 0.5, ('sph', CASTER_R), rgb=(.2, .2, .2), collide=True))
        P.append(joint(f'{n}_caster_joint', 'fixed', 'base_link', f'{n}_caster', (sx * CASTER_X, 0, CASTER_R)))
    # push blade
    P.append(link('blade', 5.0, ('box', BLADE), rgb=(.8, .2, .1), collide=True))
    P.append(joint('blade_joint', 'prismatic', 'base_link', 'blade', (BLADE_X, 0, BLADE_Z), axis=(0, 0, 1)))
    # turret + arm
    P.append(link('turret', 6.0, ('cyl', TURRET_R, TURRET_H), (0, 0, TURRET_H / 2), (.3, .3, .35)))
    P.append(joint('turret_joint', 'revolute', 'base_link', 'turret', (0, 0, CHASSIS_TOP), axis=(0, 0, 1)))
    P.append(link('upper_arm', 4.0, ('box', (L1, LINK_T, LINK_T)), (L1 / 2, 0, 0), (.9, .65, .1)))
    P.append(joint('shoulder_joint', 'revolute', 'turret', 'upper_arm', (0, 0, TURRET_H), axis=(0, 1, 0)))
    P.append(link('forearm', 3.0, ('box', (L2, LINK_T, LINK_T)), (L2 / 2, 0, 0), (.9, .65, .1)))
    P.append(joint('elbow_joint', 'revolute', 'upper_arm', 'forearm', (L1, 0, 0), axis=(0, 1, 0)))
    # Wrist palm + heavy-duty linear guide rail bar across the full finger travel range
    wrist_extras = [
        # Transverse guide rail beam (1.18m wide along Y, spans the full finger travel)
        (('box', (0.15, 1.18, 0.035)), (0, 0, -PALM[2] / 2), (0.25, 0.26, 0.30)),
        # Dual polished steel guide shafts / runner tracks
        (('box', (0.05, 1.18, 0.018)), (0.04, 0, -PALM[2] + 0.008), (0.55, 0.60, 0.68)),
        (('box', (0.05, 1.18, 0.018)), (-0.04, 0, -PALM[2] + 0.008), (0.55, 0.60, 0.68)),
    ]
    P.append(link('wrist', 1.0, ('box', PALM), (0, 0, -PALM[2] / 2), (.25, .25, .3), extra_visuals=wrist_extras))
    P.append(joint('wrist_joint', 'revolute', 'forearm', 'wrist', (L2, 0, 0), axis=(0, 1, 0)))
    for side, sy in (('left', 1), ('right', -1)):
        # Realistic mechanical slider carriage and mounting bracket
        finger_extras = [
            # Slider carriage block that rides along the wrist guide rail
            (('box', (0.18, 0.065, 0.045)), (0, 0, 0.125), (0.35, 0.38, 0.44)),
            # Rigid vertical connecting bracket joining slider carriage down to clamping pad
            (('box', (0.22, 0.038, 0.075)), (0, 0, 0.065), (0.18, 0.20, 0.24)),
        ]
        P.append(link(f'finger_{side}', 0.5, ('box', FINGER), rgb=(.15, .15, .15), collide=True, extra_visuals=finger_extras))
        P.append(joint(f'finger_{side}_joint', 'prismatic', 'wrist', f'finger_{side}',
                       (0, sy * FINGER_Y0, FINGER_Z), axis=(0, sy, 0)))
    # gazebo extensions
    G = []
    for n in ('blade', 'turret', 'upper_arm', 'forearm', 'wrist', 'finger_left', 'finger_right'):
        G.append(f'<gazebo reference="{n}"><gravity>false</gravity></gazebo>')   # arm held by controllers
    for n in ('left_wheel', 'right_wheel'):
        G.append(f'<gazebo reference="{n}"><mu1>1.0</mu1><mu2>1.0</mu2></gazebo>')
    for n in ('front_caster', 'rear_caster'):
        G.append(f'<gazebo reference="{n}"><mu1>0.01</mu1><mu2>0.01</mu2></gazebo>')
    for n in ('finger_left', 'finger_right'):
        G.append(f'<gazebo reference="{n}"><mu1>5.0</mu1><mu2>5.0</mu2></gazebo>')
    plugins = [
        '<plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">'
        '<left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>'
        f'<wheel_separation>{WHEEL_SEP}</wheel_separation><wheel_radius>{WHEEL_R}</wheel_radius>'
        '<max_linear_velocity>1.0</max_linear_velocity><max_angular_velocity>1.5</max_angular_velocity>'
        '<topic>cmd_vel</topic><odom_topic>odom</odom_topic><tf_topic>tf</tf_topic>'
        '<frame_id>odom</frame_id><child_frame_id>base_link</child_frame_id><odom_publish_frequency>30</odom_publish_frequency>'
        '</plugin>',
        '<plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">'
        '<topic>joint_states</topic></plugin>',
    ]
    for j, t in CMD_TOPICS.items():
        eff = LIMITS[j][2]
        is_prismatic = j.startswith(('finger', 'blade'))
        plugins.append(_jpc(j, t, p=250.0 if is_prismatic else 150.0, d=15.0, fmax=eff))
    G.append('<gazebo>' + ''.join(plugins) + '</gazebo>')
    return (f'<?xml version="1.0"?>\n<robot name="{ROBOT}">\n' + '\n'.join(P) + '\n' + '\n'.join(G) + '\n</robot>\n')


def write_urdf(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write(build_urdf())
    return path


if __name__ == '__main__':
    import sys
    print(write_urdf(sys.argv[1] if len(sys.argv) > 1 else 'urdf/box_sorter.urdf'))
