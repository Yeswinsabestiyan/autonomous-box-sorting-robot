"""Part C tests: box generation, spawn planning, SDF output, entry point.
Run:  python3 test/test_part_c.py -v"""
import math, os, py_compile, random, re, sys, unittest, json
import xml.etree.ElementTree as ET
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from box_sorter import box_spawner as S, yard as Y, rules


class TestRandomBox(unittest.TestCase):
    def setUp(self):
        rng = random.Random(1)
        self.boxes = [S.random_box(rng) for _ in range(3000)]

    def test_ranges(self):
        for size, mass, color, damaged in self.boxes:
            self.assertTrue(0.06 <= size <= 1.4)
            self.assertTrue(0.5 <= mass <= 25.0)
            self.assertIn(color, Y.BOX_RGB)

    def test_damaged_is_gray_and_only_damaged_is_gray(self):
        for _, _, color, damaged in self.boxes:
            self.assertEqual(damaged, color == 'gray')

    def test_every_rule_outcome_occurs(self):
        seen = {rules.decide(s, m, c, d) for s, m, c, d in self.boxes}
        for mode in ('grip', 'push'):
            self.assertTrue(any(a.mode == mode for a in seen), mode)
        for b in ('red', 'green', 'blue', 'other', 'scrap', 'big'):
            self.assertTrue(any(a.bin == b for a in seen), b)

    def test_mostly_grippable(self):
        grip = sum(rules.decide(s, m, c, d).mode == 'grip' for s, m, c, d in self.boxes)
        self.assertGreater(grip / len(self.boxes), 0.6)

    def test_probabilities_zero_gives_plain_boxes(self):
        rng = random.Random(2)
        for _ in range(500):
            size, mass, color, damaged = S.random_box(rng, 0, 0, 0)
            self.assertTrue(0.12 <= size <= 0.95 and mass <= 9.5 and not damaged)

    def test_seed_is_repeatable(self):
        a = [S.random_box(random.Random(7)) for _ in range(3)]
        self.assertEqual(a, [S.random_box(random.Random(7)) for _ in range(3)])


class TestGeometry(unittest.TestCase):
    def test_biggest_box_fits_at_every_drop_point(self):
        half_diag = 1.4 * math.sqrt(2) / 2
        for x, y in Y.INPUTS:
            self.assertLess(abs(x) + 0.7, Y.HALF_X)
            self.assertLess(abs(y) + 0.7, Y.HALF_Y)
            for px, py in Y.POLES:
                self.assertGreater(math.hypot(x - px, y - py), Y.POLE_R + half_diag, (x, y, px, py))

    def test_belt_delay(self):
        self.assertAlmostEqual(S.belt_delay(0.5), 6.0)
        self.assertAlmostEqual(S.belt_delay(1.0), 3.0)


class TestPlanner(unittest.TestCase):
    def test_spawns_at_input_drop_points_on_floor(self):
        p = S.SpawnPlanner(seed=3)
        spec = p.plan([])
        self.assertEqual((spec.x, spec.y), Y.INPUTS[spec.input])
        self.assertAlmostEqual(spec.z, spec.size / 2 + 0.01)
        self.assertEqual(spec.name, 'box_0')

    def test_names_unique_and_counted(self):
        p = S.SpawnPlanner(seed=3, max_active=100)
        names = []
        for _ in range(8):
            names.append(p.plan([]).name)
        self.assertEqual(len(set(names)), 8)
        self.assertEqual(p.count, 8)

    def test_pending_input_is_not_reused_until_delivered(self):
        p = S.SpawnPlanner(seed=4, max_active=100)
        specs = [p.plan([]) for _ in range(8)]
        self.assertEqual(sorted(s.input for s in specs), list(range(8)))   # all 8 inputs, each once
        self.assertIsNone(p.plan([]))                                      # none free
        p.delivered(specs[0])
        again = p.plan([])
        self.assertEqual(again.input, specs[0].input)

    def test_blocked_drop_point_is_skipped(self):
        p = S.SpawnPlanner(seed=5, max_active=100)
        blocker = [('box_99', Y.INPUTS[0][0], Y.INPUTS[0][1])]
        for _ in range(7):
            spec = p.plan(blocker)
            self.assertNotEqual(spec.input, 0)

    def test_robot_on_drop_point_blocks_it(self):
        self.assertFalse(S.drop_zone_clear([('box_sorter', 0.3, 5.2)], Y.INPUTS[1], 0.5))
        self.assertTrue(S.drop_zone_clear([('box_sorter', 0.3, 0.0)], Y.INPUTS[1], 0.5))

    def test_max_active_limit(self):
        p = S.SpawnPlanner(seed=6, max_active=3)
        floor = [('box_%d' % i, 100 + i, 100) for i in range(3)]
        self.assertIsNone(p.plan(floor))
        self.assertIsNotNone(S.SpawnPlanner(seed=6, max_active=3).plan(floor[:2] + [('box_1::link', 0, 100)]))  # link frames are not extra boxes
        self.assertEqual(S.count_boxes(floor + [('box_0::link', 0, 0), ('box_sorter', 0, 0)]), 3)

    def test_max_boxes_limit(self):
        p = S.SpawnPlanner(seed=7, max_active=100, max_boxes=2)
        a, b = p.plan([]), p.plan([])
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertIsNone(p.plan([]))
        self.assertTrue(p.done())

    def test_unlimited_when_max_boxes_zero(self):
        p = S.SpawnPlanner(seed=8, max_active=1000)
        for _ in range(50):
            spec = p.plan([])
            if spec:
                p.delivered(spec)
        self.assertEqual(p.count, 50)
        self.assertFalse(p.done())

    def test_big_box_never_planned_on_top_of_neighbour(self):
        # a 1.4 m box needs 1.3 m clear: a box 1.0 m from the drop point blocks it, 2.0 m does not
        d = Y.INPUTS[0]
        self.assertFalse(S.drop_zone_clear([('box_1', d[0] + 1.0, d[1])], d, 1.4))
        self.assertTrue(S.drop_zone_clear([('box_1', d[0] + 2.0, d[1])], d, 1.4))


class TestOutput(unittest.TestCase):
    def test_json_has_everything_the_brain_needs(self):
        spec = S.SpawnPlanner(seed=9).plan([])
        d = json.loads(spec.to_json())
        for k in ('name', 'size', 'mass', 'color', 'damaged', 'input', 'x', 'y', 'z'):
            self.assertIn(k, d)
        self.assertEqual(rules.decide(d['size'], d['mass'], d['color'], d['damaged']),
                         rules.decide(spec.size, spec.mass, spec.color, spec.damaged))

    def test_sdf_is_valid_and_matches_spec(self):
        spec = S.SpawnPlanner(seed=10).plan([])
        path = S.write_box_sdf(spec, '/tmp/_box_sdf_test')
        root = ET.parse(path).getroot()
        self.assertEqual(root.find('model').get('name'), spec.name)
        self.assertEqual(float(root.find('.//mass').text), spec.mass)
        self.assertEqual(root.find('.//collision/geometry/box/size').text, f'{spec.size} {spec.size} {spec.size}')

    def test_module_compiles_and_entry_point_exists(self):
        py_compile.compile(os.path.join(ROOT, 'box_sorter', 'box_spawner.py'), doraise=True, cfile='/tmp/_s.pyc')
        with open(os.path.join(ROOT, 'setup.py')) as f:
            self.assertIn('box_spawner = box_sorter.box_spawner:main', f.read())

    def test_importing_needs_no_ros(self):
        self.assertNotIn('rclpy', sys.modules)


if __name__ == '__main__':
    unittest.main()
