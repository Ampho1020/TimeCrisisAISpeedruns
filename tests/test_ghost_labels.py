"""Pure-logic tests for ghost_labels: label migration, box tracking, hit
attribution and GHOST proposals. No emulator or model needed."""

import unittest

import ghost_labels as gl
from ghost_labels import Box, ENEMY_ID, GHOST_ID

W, H = 320, 240


def rec(i, boxes=(), hits=()):
    """events.jsonl-style record. ``boxes``: (x, y, w, h); ``hits``: index into
    ``boxes`` of the box each hit lands on (aim = that box's aim point)."""
    dets = [{"c": 0, "conf": 0.9, "x": x, "y": y, "w": w, "h": h} for x, y, w, h in boxes]
    events = []
    for k in hits:
        x, y, w, h = boxes[k]
        events.append([0, (x + w / 2) / W, (y + gl.ENEMY_AIM_Y_FRACTION * h) / H])
    return {"frame": i, "tick": i, "w": W, "h": H, "dets": dets, "hit_events": events}


def classes(labels_for_record):
    return sorted(c for c, _ in labels_for_record)


class MigrationSuite(unittest.TestCase):
    def test_keeps_enemy_and_drops_grenade_and_projectile(self):
        text = "0 0.5 0.5 0.1 0.2\n1 0.3 0.3 0.05 0.05\n2 0.1 0.1 0.01 0.01\n0 0.7 0.6 0.1 0.2\n"
        self.assertEqual(gl.migrate_label_text(text),
                         "0 0.5 0.5 0.1 0.2\n0 0.7 0.6 0.1 0.2\n")

    def test_file_with_only_dropped_classes_becomes_empty(self):
        self.assertEqual(gl.migrate_label_text("1 0.3 0.3 0.05 0.05\n"), "")

    def test_empty_input_stays_empty(self):
        self.assertEqual(gl.migrate_label_text(""), "")


class TrackingSuite(unittest.TestCase):
    def test_a_slowly_moving_box_is_one_track(self):
        frames = [[Box(100 + 3 * i, 50, 40, 80)] for i in range(5)]
        self.assertEqual(len(gl.track_boxes(frames)), 1)

    def test_two_separate_enemies_are_two_tracks(self):
        frames = [[Box(60, 50, 40, 80), Box(220, 50, 40, 80)] for _ in range(4)]
        self.assertEqual(len(gl.track_boxes(frames)), 2)

    def test_one_missed_frame_does_not_split_a_track(self):
        frames = [[Box(100, 50, 40, 80)], [], [Box(100, 50, 40, 80)]]
        self.assertEqual(len(gl.track_boxes(frames, max_gap=1)), 1)

    def test_a_long_gap_starts_a_new_track_for_a_new_enemy(self):
        frames = [[Box(100, 50, 40, 80)], [], [], [], [Box(100, 50, 40, 80)]]
        self.assertEqual(len(gl.track_boxes(frames, max_gap=1)), 2)


class AttributionSuite(unittest.TestCase):
    def test_hit_goes_to_the_box_containing_the_aim_point(self):
        boxes = [Box(60, 50, 40, 80), Box(220, 50, 40, 80)]
        ax, ay = 240 / W, (50 + 0.26 * 80) / H
        self.assertEqual(gl.attribute_hit(ax, ay, boxes, W, H), 1)

    def test_hit_outside_every_box_is_unattributed(self):
        self.assertIsNone(gl.attribute_hit(0.5, 0.95, [Box(60, 50, 40, 80)], W, H))


class ProposalSuite(unittest.TestCase):
    BOX = (100, 50, 40, 80)

    def episode(self, present, hits, total=14):
        """Box present in records ``present``; a hit lands in each of ``hits``."""
        out = []
        for i in range(total):
            out.append(rec(i, [self.BOX] if i in present else [],
                           hits=[0] if i in hits else []))
        return out

    def test_kill_marks_frames_between_last_hit_and_vanishing_as_ghost(self):
        labels, report = gl.propose(self.episode(present=range(0, 7), hits={3}))
        self.assertEqual([classes(labels[i]) for i in range(0, 4)], [[ENEMY_ID]] * 4)
        self.assertEqual([classes(labels[i]) for i in (4, 5, 6)], [[GHOST_ID]] * 3)
        self.assertEqual(classes(labels[7]), [])
        row = next(r for r in report if r.get("outcome"))
        self.assertEqual((row["outcome"], row["ghost_frames"]), ("kill", 3))

    def test_multi_hit_enemy_is_ghost_only_after_its_last_hit(self):
        labels, _ = gl.propose(self.episode(present=range(0, 9), hits={3, 5}))
        self.assertEqual(classes(labels[5]), [ENEMY_ID])
        self.assertEqual(classes(labels[6]), [GHOST_ID])

    def test_a_box_reappearing_after_a_gap_is_not_the_dying_enemy(self):
        # Enemy hit at 3 and gone at 4; something (e.g. a fire false positive)
        # is boxed in the same spot at 5. Only the unbroken run is GHOST.
        labels, _ = gl.propose(self.episode(present={0, 1, 2, 3, 5, 6}, hits={3}))
        self.assertEqual(classes(labels[4]), [])
        self.assertTrue(all(c != GHOST_ID for i in (5, 6) for c, _ in labels[i]))

    def test_box_that_never_vanishes_is_a_survivor_not_ghost(self):
        labels, report = gl.propose(self.episode(present=range(0, 14), hits={3}))
        self.assertTrue(all(classes(labels[i]) == [ENEMY_ID] for i in range(14)))
        self.assertEqual(next(r for r in report if r.get("outcome"))["outcome"], "cut_off")

    def test_box_lingering_far_past_the_window_is_not_proposed_as_ghost(self):
        labels, report = gl.propose(self.episode(present=range(0, 12), hits={1}, total=16),
                                    max_ghost_ticks=4)
        self.assertTrue(all(c != GHOST_ID for lab in labels for c, _ in lab))
        self.assertEqual(next(r for r in report if r.get("outcome"))["outcome"],
                         "survivor_or_slow_death")

    def test_instant_kill_when_the_box_is_gone_in_the_hit_tick_image(self):
        # Box present through record 2; the hit lands in record 3 whose image
        # already has no box. Attributed via the previous record, no GHOST frames.
        eps = self.episode(present=range(0, 3), hits=set())
        eps[3] = rec(3, [], hits=[])
        x, y, w, h = self.BOX
        eps[3]["hit_events"] = [[0, (x + w / 2) / W, (y + 0.26 * h) / H]]
        labels, report = gl.propose(eps)
        self.assertTrue(all(c != GHOST_ID for lab in labels for c, _ in lab))
        self.assertEqual(next(r for r in report if r.get("outcome"))["outcome"], "instant_kill")

    def test_a_hit_with_no_box_anywhere_is_counted_unattributed(self):
        eps = [rec(i) for i in range(6)]
        eps[2]["hit_events"] = [[0, 0.5, 0.5]]
        _, report = gl.propose(eps)
        self.assertEqual(report[-1]["unattributed_hits"], 1)

    def test_unhit_enemy_is_never_ghost(self):
        other = (220, 50, 40, 80)
        eps = [rec(i, [self.BOX, other] if i < 7 else [], hits=[0] if i == 3 else [])
               for i in range(14)]
        labels, _ = gl.propose(eps)
        for i in (4, 5, 6):
            self.assertEqual(classes(labels[i]), sorted([GHOST_ID, ENEMY_ID]))

    def test_low_confidence_boxes_are_ignored(self):
        eps = [rec(i, [self.BOX]) for i in range(4)]
        for r in eps:
            r["dets"][0]["conf"] = 0.2
        labels, _ = gl.propose(eps, min_conf=0.5)
        self.assertTrue(all(lab == [] for lab in labels))


class YoloLineSuite(unittest.TestCase):
    def test_line_is_normalised_and_clipped(self):
        line = gl.to_yolo_line(1, Box(100, 50, 40, 80), W, H)
        cls, cx, cy, w, h = line.split()
        self.assertEqual(cls, "1")
        self.assertAlmostEqual(float(cx), 120 / W, places=5)
        self.assertAlmostEqual(float(h), 80 / H, places=5)

    def test_box_past_the_frame_edge_is_clipped_to_one(self):
        _, cx, _, _, _ = gl.to_yolo_line(0, Box(300, 50, 80, 80), W, H).split()
        self.assertLessEqual(float(cx), 1.0)


if __name__ == "__main__":
    unittest.main()
