"""Single source of truth for the yard layout. Metres, origin at yard centre, +x right."""
HALF_X, HALF_Y = 10.0, 6.0

# 5 designated placement bins on the LEFT wall, one column. (x, y, size)
BIN_X = -HALF_X + 0.8
BINS = {
    'red':   (BIN_X,  4.5, 1.6),
    'green': (BIN_X,  2.25, 1.6),
    'blue':  (BIN_X,  0.0, 1.6),
    'other': (BIN_X, -2.25, 1.6),
    'scrap': (BIN_X, -4.5, 1.6),
}
BIN_RGB = {
    'red':   (.9, .1, .1),
    'green': (.1, .7, .2),
    'blue':  (.1, .2, .9),
    'other': (.85, .75, .15),
    'scrap': (.25, .25, .25),
}

POLES = [(-4.0, 3.0), (-3.0, -3.0), (2.0, 2.0), (5.0, -2.0), (6.0, 4.0), (2.5, -3.5)]
POLE_R = 0.3

# 8 input conveyors. Drop point = where the belt ends inside the yard.
INPUTS = [(-5, 5.0), (0, 5.0), (5, 5.0), (-5, -5.0), (0, -5.0), (5, -5.0), (9.0, -2.0), (9.0, 2.0)]
# 4 designated main conveyor belts for streamlined simulation
BELTS_4 = [(-5.0, 5.0), (5.0, 5.0), (-5.0, -5.0), (5.0, -5.0)]

BOX_RGB = {'red': (.9, .1, .1), 'green': (.1, .7, .2), 'blue': (.1, .2, .9),
           'yellow': (.95, .85, .1), 'purple': (.6, .2, .7), 'orange': (.95, .5, .1),
           'gray': (.35, .35, .35)}   # gray = damaged
