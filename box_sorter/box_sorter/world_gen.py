"""Builds the Gazebo world from yard.py so layout lives in one place."""
import os
from . import yard as Y


def _visual_box(name, x, y, z, sx, sy, sz, rgb):
    r, g, b = rgb
    return (f'<model name="{name}"><static>true</static><pose>{x} {y} {z} 0 0 0</pose><link name="l">'
            f'<visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
            f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material></visual>'
            f'</link></model>')


def build_world():
    items = [_visual_box('yard_floor', 0, 0, 0.005, 2 * Y.HALF_X, 2 * Y.HALF_Y, 0.01, (.75, .75, .72))]
    for n, (x, y, s) in Y.BINS.items():
        items.append(_visual_box(f'bin_{n}', x, y, 0.015, s, s, 0.01, Y.BIN_RGB[n]))
    for i, (x, y) in enumerate(Y.INPUTS):          # belts are visual only for now
        if abs(y) > 4:
            sy = 1 if y > 0 else -1
            items.append(_visual_box(f'belt_{i}', x, sy * (Y.HALF_Y + 1.5), 0.1, 1.2, 3.0, 0.2, (.15, .15, .15)))
        else:
            items.append(_visual_box(f'belt_{i}', Y.HALF_X + 1.5, y, 0.1, 3.0, 1.2, 0.2, (.15, .15, .15)))
    for i, (x, y) in enumerate(Y.POLES):
        items.append(
            f'<model name="pole_{i}"><static>true</static><pose>{x} {y} 1 0 0 0</pose><link name="l">'
            f'<collision name="c"><geometry><cylinder><radius>{Y.POLE_R}</radius><length>2</length></cylinder></geometry></collision>'
            f'<visual name="v"><geometry><cylinder><radius>{Y.POLE_R}</radius><length>2</length></cylinder></geometry>'
            f'<material><ambient>.9 .6 0 1</ambient><diffuse>.9 .6 0 1</diffuse></material></visual></link></model>')
    return f'''<?xml version="1.0"?>
<sdf version="1.9"><world name="yard">
<physics name="p" type="ode"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
<plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
<plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
<light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
<diffuse>.9 .9 .9 1</diffuse><direction>-.4 .2 -1</direction></light>
<model name="ground"><static>true</static><link name="l">
<collision name="c"><geometry><plane><normal>0 0 1</normal><size>80 80</size></plane></geometry>
<surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
<visual name="v"><geometry><plane><normal>0 0 1</normal><size>80 80</size></plane></geometry>
<material><ambient>.45 .45 .45 1</ambient><diffuse>.45 .45 .45 1</diffuse></material></visual></link></model>
{chr(10).join(items)}
</world></sdf>'''


def write_world(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(build_world())
    return path
