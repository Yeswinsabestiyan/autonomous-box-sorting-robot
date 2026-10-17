"""Thin wrappers around `gz service` (spawn / teleport). Phase 1 shortcut, fine for sim."""
import subprocess
import threading

WORLD = 'yard'
_pose_lock = threading.Lock()
_pose_busy = False


def _call(service, reqtype, req, timeout=3000):
    import os
    if os.environ.get('SIM_MODE') == 'rviz':
        class MockResult:
            returncode = 0
            stdout = 'true'
            stderr = ''
        return MockResult()
    return subprocess.run(
        ['gz', 'service', '-s', service, '--reqtype', reqtype, '--reptype', 'gz.msgs.Boolean',
         '--timeout', str(timeout), '--req', req], capture_output=True, text=True)


def spawn_file(path, name, x, y, z):
    req = f'sdf_filename: "{path}" name: "{name}" pose {{position {{x: {x} y: {y} z: {z}}}}}'
    return _call(f'/world/{WORLD}/create', 'gz.msgs.EntityFactory', req)


def _extract_xyz_quat(x_or_pose, y=None, z=None, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    """Support both set_pose(name, x, y, z) and set_pose(name, pose_msg)."""
    if y is None and z is None:
        if hasattr(x_or_pose, 'position'):
            pos = x_or_pose.position
            ori = getattr(x_or_pose, 'orientation', None)
            if ori is not None:
                return pos.x, pos.y, pos.z, ori.w, ori.x, ori.y, ori.z
            return pos.x, pos.y, pos.z, 1.0, 0.0, 0.0, 0.0
        elif isinstance(x_or_pose, dict):
            return (x_or_pose['x'], x_or_pose['y'], x_or_pose['z'],
                    x_or_pose.get('qw', 1.0), x_or_pose.get('qx', 0.0),
                    x_or_pose.get('qy', 0.0), x_or_pose.get('qz', 0.0))
        elif isinstance(x_or_pose, (tuple, list)) and len(x_or_pose) >= 3:
            return x_or_pose[0], x_or_pose[1], x_or_pose[2], 1.0, 0.0, 0.0, 0.0
    return float(x_or_pose), float(y), float(z), float(qw), float(qx), float(qy), float(qz)


def set_pose(name, x_or_pose, y=None, z=None, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    px, py, pz, w, rx, ry, rz = _extract_xyz_quat(x_or_pose, y, z, qw, qx, qy, qz)
    req = (f'name: "{name}" '
           f'position {{x: {px:.4f} y: {py:.4f} z: {pz:.4f}}} '
           f'orientation {{x: {rx:.4f} y: {ry:.4f} z: {rz:.4f} w: {w:.4f}}}')
    return _call(f'/world/{WORLD}/set_pose', 'gz.msgs.Pose', req, timeout=1000)


def set_pose_nonblocking(name, x_or_pose, y=None, z=None, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    """Fire-and-forget pose update so the ROS 2 timer loop is never blocked by CLI startup."""
    global _pose_busy
    with _pose_lock:
        if _pose_busy:
            return
        _pose_busy = True

    def _worker():
        global _pose_busy
        try:
            set_pose(name, x_or_pose, y, z, qw, qx, qy, qz)
        finally:
            with _pose_lock:
                _pose_busy = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def box_sdf(name, size, mass, rgb, width=None, depth=None, height=None):
    sx = float(width if width is not None else size)
    sy = float(depth if depth is not None else size)
    sz = float(height if height is not None else size)
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    r, g, b = rgb
    return f'''<?xml version="1.0"?>
<sdf version="1.9"><model name="{name}"><link name="link">
<inertial><mass>{mass}</mass><inertia><ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz></inertia></inertial>
<collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
<surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision>
<visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material></visual>
</link></model></sdf>'''
