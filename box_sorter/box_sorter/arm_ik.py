"""Planar IK for turret(yaw) + shoulder + elbow. Wrist joint just keeps the gripper vertical.
Targets are the palm point: r = horizontal distance from the turret axis, z = height above ground."""
import math

L1 = L2 = 1.1
SHOULDER_Z = 0.75      # shoulder joint height above ground


def solve(r, z):
    x, y = r, z - SHOULDER_Z
    d2 = x * x + y * y
    c = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1:
        raise ValueError(f'unreachable r={r:.2f} z={z:.2f}')
    a2 = -math.acos(c)                                   # elbow up
    a1 = math.atan2(y, x) - math.atan2(L2 * math.sin(a2), L1 + L2 * math.cos(a2))
    # +y rotation pitches the arm DOWN, so joint = -angle_up
    return -a1, -a2, a1 + a2                             # shoulder, elbow, wrist


def fk(shoulder, elbow):
    a1, a2 = -shoulder, -elbow
    return (L1 * math.cos(a1) + L2 * math.cos(a1 + a2),
            SHOULDER_Z + L1 * math.sin(a1) + L2 * math.sin(a1 + a2))
