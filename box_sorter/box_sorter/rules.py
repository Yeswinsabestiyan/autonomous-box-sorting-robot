"""Pure decision logic: grip or push, and which bin."""
from dataclasses import dataclass

GRIP_MIN, GRIP_MAX, MAX_GRIP_KG = 0.10, 1.00, 10.0
SORT_COLORS = ('red', 'green', 'blue')


@dataclass(frozen=True)
class Action:
    mode: str   # 'grip' or 'push'
    bin: str


def decide(size, mass, color, damaged=False):
    out_of_range = size > GRIP_MAX or size < GRIP_MIN
    mode = 'push' if (out_of_range or mass > MAX_GRIP_KG) else 'grip'
    if damaged:
        return Action(mode, 'scrap')
    if mode == 'push':
        return Action(mode, 'big')
    return Action(mode, color if color in SORT_COLORS else 'other')
