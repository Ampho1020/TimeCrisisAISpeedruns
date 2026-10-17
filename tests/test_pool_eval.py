import unittest

import audit_misses as am
import pool_eval as pe


def _ep(fired, hit, shots=None, cleared=False):
    return {"shots_fired": fired, "shots_hit": hit, "cleared": cleared, "shots": shots or []}


class PoolEvalTests(unittest.TestCase):
    def test_pooled_accuracy_is_shot_weighted(self):
        eps = [_ep(10, 5), _ep(90, 18)]
        self.assertAlmostEqual(pe._acc(eps), 0.23)

    def test_bootstrap_ci_brackets_estimate_and_is_deterministic(self):
        eps = [_ep(100, 40), _ep(100, 35), _ep(100, 48), _ep(100, 41)]
        lo, hi = pe.bootstrap_ci(eps)
        self.assertLessEqual(lo, pe._acc(eps))
        self.assertGreaterEqual(hi, pe._acc(eps))
        self.assertEqual((lo, hi), pe.bootstrap_ci(eps))

    def test_diff_ci_flags_a_clear_gap(self):
        a = [_ep(100, 30), _ep(100, 31), _ep(100, 29)]
        b = [_ep(100, 50), _ep(100, 52), _ep(100, 49)]
        d, lo, hi = pe.diff_ci(a, b)
        self.assertGreater(lo, 0)
        self.assertAlmostEqual(d, 0.2033, places=3)

    def test_diff_ci_straddles_zero_for_noise(self):
        a = [_ep(100, 30), _ep(100, 50), _ep(100, 40)]
        b = [_ep(100, 35), _ep(100, 45), _ep(100, 40)]
        _, lo, hi = pe.diff_ci(a, b)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_split_buckets(self):
        shots = [
            {"fired": 1, "hit": 0, "had_detection": False, "nearest_dist": -1.0},
            {"fired": 1, "hit": 0, "had_detection": True, "nearest_dist": 0.2},
            {"fired": 2, "hit": 1, "had_detection": True, "nearest_dist": 0.01},
        ]
        self.assertEqual(
            pe.split_buckets(shots, 0.08),
            {"blind": (1, 0), "off": (1, 0), "on": (2, 1)},
        )

    def test_group_episodes_keeps_order(self):
        recs = [{"tag": "b"}, {"tag": "a"}, {"tag": "b"}]
        g = pe.group_episodes(recs, "tag")
        self.assertEqual(list(g), ["b", "a"])
        self.assertEqual(len(g["b"]), 2)


def _rec(tick, aim, dets, fired=1, hits=0, hit_events=()):
    return {"tick": tick, "aim": list(aim), "w": 100, "h": 100, "fired": fired,
            "hits": hits, "hit_events": [list(e) for e in hit_events], "dets": dets}


def _det(cx, top, conf=0.9, w=10, h=20):
    return {"x": cx - w // 2, "y": top, "w": w, "h": h, "conf": conf, "c": 0}


class AuditMissTests(unittest.TestCase):
    def aim_of(self, d):
        return (d["x"] + d["w"] / 2) / 100, (d["y"] + am.ENEMY_AIM_Y_FRACTION * d["h"]) / 100

    def test_blind(self):
        self.assertEqual(am.classify_miss(_rec(1, (0.5, 0.5), []), [])[0], "blind")

    def test_off_target_and_fresh_and_low_conf(self):
        d = _det(50, 40)
        ax, ay = self.aim_of(d)
        self.assertEqual(am.classify_miss(_rec(1, (ax + 0.3, ay), [d]), [])[0], "off_target")
        self.assertEqual(am.classify_miss(_rec(1, (ax, ay), [d]), [])[0], "fresh")
        low = _det(50, 40, conf=0.4)
        self.assertEqual(am.classify_miss(_rec(1, (ax, ay), [low]), [])[0], "low_conf")

    def test_ghost_window_needs_recent_nearby_hit(self):
        d = _det(50, 40)
        ax, ay = self.aim_of(d)
        rec = _rec(10, (ax, ay), [d])
        self.assertEqual(am.classify_miss(rec, [(8, ax, ay)])[0], "ghost_window")
        self.assertEqual(am.classify_miss(rec, [(2, ax, ay)])[0], "fresh")
        self.assertEqual(am.classify_miss(rec, [(8, ax + 0.5, ay)])[0], "fresh")

    def test_audit_only_missed_ticks_and_earlier_hits(self):
        d = _det(50, 40)
        ax, ay = self.aim_of(d)
        recs = [
            _rec(1, (ax, ay), [d], fired=1, hits=1, hit_events=[(0, ax, ay)]),
            _rec(2, (ax, ay), [d], fired=1, hits=0),
            _rec(3, (ax, ay), [d], fired=0, hits=0),
        ]
        out = am.audit(recs)
        self.assertEqual([e["bucket"] for e in out], ["ghost_window"])


if __name__ == "__main__":
    unittest.main()
