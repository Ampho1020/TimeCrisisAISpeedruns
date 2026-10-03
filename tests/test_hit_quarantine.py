"""Pure-logic unit tests for hit_quarantine.HitQuarantine."""

import unittest
from dataclasses import dataclass

from hit_quarantine import HitQuarantine


@dataclass
class FakeDet:
    x: int
    y: int
    w: int
    h: int
    class_id: int = 0
    aim_x_norm: float | None = None
    aim_y_norm: float | None = None
    cx_norm: float = 0.0
    cy_norm: float = 0.0


def enemy(x, y, w=40, h=80, class_id=0):
    # 320x240 frame; aim point 26% down the box, like ENEMY_AIM_Y_FRACTION.
    return FakeDet(
        x=x, y=y, w=w, h=h, class_id=class_id,
        cx_norm=(x + w / 2) / 320.0, cy_norm=(y + h / 2) / 240.0,
        aim_x_norm=(x + w / 2) / 320.0, aim_y_norm=(y + 0.26 * h) / 240.0,
    )


def make(max_ticks=12, absent_ticks=2):
    return HitQuarantine(
        max_ticks=max_ticks, absent_ticks=absent_ticks,
        match_frac=0.5, point_radius=0.05,
    )


class QuarantineCreationSuite(unittest.TestCase):
    def test_hit_quarantines_the_box_and_hides_it_from_the_policy(self):
        q = make()
        hit_box = enemy(100, 50)
        other = enemy(240, 50)
        self.assertTrue(q.add(0, hit_box.aim_x_norm, hit_box.aim_y_norm, [hit_box, other]))
        visible = q.filter([hit_box, other])
        self.assertEqual(visible, [other])

    def test_hit_with_no_box_nearby_creates_nothing(self):
        q = make()
        far = enemy(240, 50)
        self.assertFalse(q.add(0, 0.1, 0.1, [far]))
        self.assertEqual(q.unboxed_hits, 1)
        self.assertFalse(q.active)

    def test_non_enemy_classes_are_never_quarantined_or_hidden(self):
        q = make()
        grenade = enemy(100, 50, class_id=1)
        self.assertFalse(q.add(0, grenade.aim_x_norm, grenade.aim_y_norm, [grenade]))
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        self.assertEqual(q.filter([grenade]), [grenade])

    def test_second_hit_on_same_box_does_not_duplicate(self):
        q = make()
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        self.assertFalse(q.add(0, e.aim_x_norm, e.aim_y_norm, [e]))
        self.assertEqual(q.created, 1)


class QuarantineReleaseSuite(unittest.TestCase):
    def test_released_after_box_absent_for_absent_ticks(self):
        q = make(absent_ticks=2)
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        q.update(1, [e])
        self.assertTrue(q.active)
        q.update(2, [])
        self.assertTrue(q.active)  # only 1 absent tick so far
        q.update(3, [])
        self.assertFalse(q.active)
        self.assertEqual(q.released_absent, 1)

    def test_detector_flicker_does_not_release_early(self):
        q = make(absent_ticks=2)
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        q.update(1, [])          # one missed frame
        q.update(2, [e])         # sprite is back: still the same enemy
        q.update(3, [])
        self.assertTrue(q.active)
        self.assertEqual(q.released_absent, 0)

    def test_timeout_releases_even_if_box_never_leaves(self):
        q = make(max_ticks=5)
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        for tick in range(1, 5):
            q.update(tick, [e])
        self.assertTrue(q.active)
        q.update(5, [e])
        self.assertFalse(q.active)
        self.assertEqual(q.released_timeout, 1)

    def test_enemy_reappearing_after_release_is_visible_again(self):
        q = make(absent_ticks=1)
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        q.update(1, [])  # gone -> released
        newcomer = enemy(100, 50)
        self.assertEqual(q.filter([newcomer]), [newcomer])


class QuarantineMatchingSuite(unittest.TestCase):
    def test_follows_a_box_that_drifts_between_ticks(self):
        q = make()
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        drifted = enemy(108, 53)
        q.update(1, [drifted])
        self.assertEqual(q.entries[0]["absent"], 0)
        self.assertEqual(q.filter([drifted]), [])

    def test_adjacent_enemy_one_box_width_away_is_not_hidden(self):
        q = make()
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        neighbour = enemy(145, 50)  # centres 45px apart, box width 40
        self.assertEqual(q.filter([neighbour]), [neighbour])


class QuarantineTriggerBlockSuite(unittest.TestCase):
    def test_blocks_aim_on_the_quarantined_enemy_only(self):
        q = make()
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        self.assertTrue(q.blocks(e.aim_x_norm, e.aim_y_norm))
        self.assertFalse(q.blocks(0.9, 0.9))

    def test_blocks_nothing_when_empty(self):
        self.assertFalse(make().blocks(0.5, 0.5))


class QuarantineResetSuite(unittest.TestCase):
    def test_reset_clears_entries_and_counters(self):
        q = make()
        e = enemy(100, 50)
        q.add(0, e.aim_x_norm, e.aim_y_norm, [e])
        q.filter([e])
        q.reset()
        self.assertFalse(q.active)
        self.assertEqual((q.created, q.filtered, q.released_absent, q.released_timeout), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
