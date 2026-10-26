import unittest

import ghost_consistency as gc

W, H = 280, 240
BOX = dict(x=100, y=100, w=40, h=60)
HIT = [[0, 120 / W, 130 / H]]


def rec(i, *, e=False, g=False, hit=False, conf=0.9, **over):
    box = {**BOX, **over, "conf": conf}
    return {"frame": i, "tick": i, "w": W, "h": H,
            "dets": [{**box, "c": 0}] if e else [],
            "ghosts": [{**box, "c": 3}] if g else [],
            "hit_events": HIT if hit else []}


def seq(spec, hit_at=()):
    out = [rec(i, e=ch in "EC", g=ch in "GC", hit=i in hit_at) for i, ch in enumerate(spec)]
    return out + [rec(len(out) + k) for k in range(4)]  # empty ticks so the track counts as ended


def reasons(records):
    return {r for t in gc.analyze(records)["tracks"] for r in t["reasons"]}


class ConsistencyTests(unittest.TestCase):
    def test_clean_kill_has_no_flags(self):
        res = gc.analyze(seq("EEEGG", hit_at=(2,)))
        self.assertEqual(res["tracks"][0]["labels"], "EEEGG")
        self.assertEqual(reasons(seq("EEEGG", hit_at=(2,))), set())

    def test_resurrection_and_flicker(self):
        r = reasons(seq("EEGGEE", hit_at=(1,)))
        self.assertIn("resurrection", r)
        self.assertIn("flicker", r)

    def test_missed_ghost_when_sprite_lingers_as_enemy_after_kill(self):
        self.assertIn("missed_ghost", reasons(seq("EEEEE", hit_at=(2,))))

    def test_instant_kill_is_not_a_missed_ghost(self):
        self.assertNotIn("missed_ghost", reasons(seq("EEE", hit_at=(2,))))

    def test_born_ghost_without_hit_is_unconfirmed(self):
        r = reasons(seq("GGG"))
        self.assertIn("born_ghost", r)
        self.assertIn("unconfirmed_ghost", r)

    def test_overlapping_enemy_and_ghost_box_is_one_contested_sprite(self):
        res = gc.analyze(seq("CC"))
        self.assertEqual(len(res["tracks"]), 1)
        self.assertEqual(res["tracks"][0]["labels"], "CC")
        self.assertIn("contested", res["tracks"][0]["reasons"])

    def test_low_confidence_boxes_are_ignored(self):
        records = [rec(0, e=True, conf=0.2), rec(1, e=True, conf=0.2)]
        self.assertEqual(gc.analyze(records)["tracks"], [])


if __name__ == "__main__":
    unittest.main()
