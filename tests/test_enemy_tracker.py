"""Pure-logic unit tests for enemy_tracker.EnemyTracker.

No BizHawk/detector dependency -- fast, isolated validation of the
nearest-centroid matching, expiry, and one-shot "done" credit logic before
it's trusted inside a real (expensive) eval run.
"""

import unittest
from dataclasses import dataclass

from enemy_tracker import EnemyTracker


@dataclass
class FakeDet:
    cx_norm: float
    cy_norm: float
    class_id: int = 0


class EnemyTrackerBasicsSuite(unittest.TestCase):
    def test_first_detection_creates_one_track(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=5)
        t.update(0, [FakeDet(0.5, 0.5)])
        self.assertEqual(t.track_count, 1)
        self.assertEqual(t.total_created, 1)

    def test_same_spot_next_tick_reuses_the_same_id(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=5)
        t.update(0, [FakeDet(0.5, 0.5)])
        tid_before = t.nearest_track_id(0.5, 0.5)
        t.update(1, [FakeDet(0.51, 0.50)])  # tiny jitter, still within radius
        tid_after = t.nearest_track_id(0.51, 0.50)
        self.assertEqual(tid_before, tid_after)
        self.assertEqual(t.track_count, 1)
        self.assertEqual(t.total_created, 1)  # no new track created

    def test_two_enemies_get_distinct_ids(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=5)
        t.update(0, [FakeDet(0.2, 0.5), FakeDet(0.8, 0.5)])
        self.assertEqual(t.track_count, 2)
        id_a = t.nearest_track_id(0.2, 0.5)
        id_b = t.nearest_track_id(0.8, 0.5)
        self.assertIsNotNone(id_a)
        self.assertIsNotNone(id_b)
        self.assertNotEqual(id_a, id_b)

    def test_non_enemy_class_is_ignored(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=5)
        t.update(0, [FakeDet(0.5, 0.5, class_id=1)])  # GRENADE, not ENEMY
        self.assertEqual(t.track_count, 0)

    def test_nearest_track_id_outside_radius_returns_none(self):
        t = EnemyTracker(match_radius=0.05, expire_ticks=5)
        t.update(0, [FakeDet(0.5, 0.5)])
        self.assertIsNone(t.nearest_track_id(0.9, 0.9))


class EnemyTrackerExpirySuite(unittest.TestCase):
    def test_track_survives_brief_absence_within_expire_window(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=3)
        t.update(0, [FakeDet(0.5, 0.5)])
        tid = t.nearest_track_id(0.5, 0.5)
        t.update(1, [])  # detector flicker, no boxes this tick
        t.update(2, [])
        # Still within expire_ticks=3 of last_seen_tick=0.
        self.assertEqual(t.track_count, 1)
        t.update(3, [FakeDet(0.5, 0.5)])  # reappears before expiry
        self.assertEqual(t.nearest_track_id(0.5, 0.5), tid)
        self.assertEqual(t.total_created, 1)  # same id, no new track

    def test_track_is_dropped_after_expire_window(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=2)
        t.update(0, [FakeDet(0.5, 0.5)])
        t.update(1, [])
        t.update(2, [])
        t.update(3, [])  # 3 ticks since last_seen_tick=0 > expire_ticks=2
        self.assertEqual(t.track_count, 0)

    def test_new_enemy_at_same_spot_after_expiry_gets_a_fresh_id_and_is_not_done(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=2)
        t.update(0, [FakeDet(0.5, 0.5)])
        old_id = t.nearest_track_id(0.5, 0.5)
        t.credit_hit(old_id)
        self.assertTrue(t.is_done(old_id))
        for tick in (1, 2, 3, 4):
            t.update(tick, [])  # old track expires
        t.update(5, [FakeDet(0.5, 0.5)])  # a new enemy walks into the same spot
        new_id = t.nearest_track_id(0.5, 0.5)
        self.assertNotEqual(old_id, new_id)
        self.assertFalse(t.is_done(new_id))


class EnemyTrackerOneShotCreditSuite(unittest.TestCase):
    def test_single_hit_marks_a_track_done(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=5)
        t.update(0, [FakeDet(0.5, 0.5)])
        tid = t.nearest_track_id(0.5, 0.5)
        self.assertFalse(t.is_done(tid))
        t.credit_hit(tid)
        self.assertTrue(t.is_done(tid))
        self.assertEqual(t.total_done, 1)

    def test_crediting_an_already_done_track_does_not_double_count(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=5)
        t.update(0, [FakeDet(0.5, 0.5)])
        tid = t.nearest_track_id(0.5, 0.5)
        t.credit_hit(tid)
        t.credit_hit(tid)
        t.credit_hit(tid)
        self.assertEqual(t.total_done, 1)
        self.assertEqual(t.tracks[tid]["hits"], 3)

    def test_crediting_none_or_unknown_id_is_a_safe_no_op(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=5)
        t.credit_hit(None)
        t.credit_hit(999)
        self.assertEqual(t.total_done, 0)
        self.assertEqual(t.total_credit_misses, 2)

    def test_is_done_for_untracked_location_is_false(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=5)
        t.update(0, [FakeDet(0.5, 0.5)])
        self.assertFalse(t.is_done(t.nearest_track_id(0.9, 0.9)))


class EnemyTrackerResetSuite(unittest.TestCase):
    def test_reset_clears_all_state(self):
        t = EnemyTracker(match_radius=0.08, expire_ticks=5)
        t.update(0, [FakeDet(0.5, 0.5), FakeDet(0.2, 0.2)])
        t.credit_hit(t.nearest_track_id(0.5, 0.5))
        t.reset()
        self.assertEqual(t.track_count, 0)
        self.assertEqual(t.total_created, 0)
        self.assertEqual(t.total_done, 0)
        # IDs restart from 1 after reset.
        t.update(0, [FakeDet(0.5, 0.5)])
        self.assertEqual(t.nearest_track_id(0.5, 0.5), 1)


if __name__ == "__main__":
    unittest.main()
