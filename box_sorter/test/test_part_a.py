"""Part A tests. Plain unittest, no pytest needed:  python3 -m unittest test.test_part_a -v"""
import math, os, sys, unittest
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from box_sorter import rules, arm_ik, yard as Y, world_gen, gz_utils


class TestRules(unittest.TestCase):
    def test_grip_sorted_by_colour(self):
        for c in ('red', 'green', 'blue'):
            self.assertEqual(rules.decide(0.4, 5, c), rules.Action('grip', c))

    def test_other_colour(self):
        self.assertEqual(rules.decide(0.4, 5, 'yellow'), rules.Action('grip', 'other'))

    def test_heavy_pushed(self):
        self.assertEqual(rules.decide(0.4, 10.5, 'red'), rules.Action('push', 'big'))

    def test_ten_kg_exactly_is_gripped(self):
        self.assertEqual(rules.decide(0.4, 10.0, 'red').mode, 'grip')

    def test_size_limits(self):
        self.assertEqual(rules.decide(1.2, 2, 'red').mode, 'push')
        self.assertEqual(rules.decide(0.05, 1, 'red').mode, 'push')
        self.assertEqual(rules.decide(1.0, 2, 'red').mode, 'grip')
        self.assertEqual(rules.decide(0.10, 1, 'red').mode, 'grip')

    def test_damaged_goes_to_scrap_keeping_mode(self):
        self.assertEqual(rules.decide(0.3, 2, 'gray', damaged=True), rules.Action('grip', 'scrap'))
        self.assertEqual(rules.decide(1.3, 2, 'gray', damaged=True), rules.Action('push', 'scrap'))


class TestIK(unittest.TestCase):
    def test_round_trip(self):
        for r, z in [(1.5, 0.3), (1.0, 0.5), (2.0, 0.9), (1.2, 1.2), (0.8, 0.2)]:
            s, e, _ = arm_ik.solve(r, z)
            fr, fz = arm_ik.fk(s, e)
            self.assertAlmostEqual(fr, r, 6)
            self.assertAlmostEqual(fz, z, 6)

    def test_unreachable(self):
        with self.assertRaises(ValueError):
            arm_ik.solve(5.0, 0.3)

    def test_wrist_keeps_gripper_vertical(self):
        s, e, w = arm_ik.solve(1.5, 0.3)
        self.assertAlmostEqual(w, -(s + e), 6)


class TestYard(unittest.TestCase):
    def test_bins_inside_yard(self):
        for n, (x, y, s) in Y.BINS.items():
            self.assertGreaterEqual(x - s / 2, -Y.HALF_X - 1e-9, n)
            self.assertLessEqual(abs(y) + s / 2, Y.HALF_Y + 1e-9, n)

    def test_bins_do_not_overlap(self):
        b = list(Y.BINS.items())
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                (_, (x1, y1, s1)), (_, (x2, y2, s2)) = b[i], b[j]
                self.assertTrue(abs(x1 - x2) >= (s1 + s2) / 2 or abs(y1 - y2) >= (s1 + s2) / 2,
                                f'{b[i][0]} overlaps {b[j][0]}')

    def test_poles_clear_of_bins_and_inputs(self):
        for px, py in Y.POLES:
            for n, (x, y, s) in Y.BINS.items():
                self.assertFalse(abs(px - x) < s / 2 + Y.POLE_R and abs(py - y) < s / 2 + Y.POLE_R, n)
            for ix, iy in Y.INPUTS:
                self.assertGreater(math.hypot(px - ix, py - iy), 1.0)

    def test_eight_inputs_inside_yard(self):
        self.assertEqual(len(Y.INPUTS), 8)
        for x, y in Y.INPUTS:
            self.assertLessEqual(abs(x), Y.HALF_X)
            self.assertLessEqual(abs(y), Y.HALF_Y)

    def test_every_bin_has_colour(self):
        self.assertEqual(set(Y.BINS), set(Y.BIN_RGB))


class TestWorld(unittest.TestCase):
    def test_world_is_valid_xml_with_all_models(self):
        root = ET.fromstring(world_gen.build_world())
        names = {m.get('name') for m in root.iter('model')}
        for n in Y.BINS:
            self.assertIn(f'bin_{n}', names)
        for i in range(len(Y.POLES)):
            self.assertIn(f'pole_{i}', names)
        for i in range(len(Y.INPUTS)):
            self.assertIn(f'belt_{i}', names)
        self.assertEqual(root.find('world').get('name'), gz_utils.WORLD)

    def test_box_sdf_is_valid_and_mass_inertia(self):
        root = ET.fromstring(gz_utils.box_sdf('b1', 0.5, 3.0, (1, 0, 0)))
        self.assertEqual(float(root.find('.//mass').text), 3.0)
        self.assertAlmostEqual(float(root.find('.//ixx').text), 3.0 * 2 * 0.25 / 12)


if __name__ == '__main__':
    unittest.main()
