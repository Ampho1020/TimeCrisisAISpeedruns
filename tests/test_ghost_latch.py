import unittest

from detector import GHOST_CLASS_ID, Detection, EnemyClass
from ghost_latch import GhostLatch


def det(x=100, y=100, w=40, h=60, conf=0.9, cls=int(EnemyClass.ENEMY)):
    return Detection(x, y, w, h, cls, conf, 0.0, 0.0)


class GhostLatchTests(unittest.TestCase):
    def test_enemy_after_ghost_on_same_sprite_stays_ghost(self):
        latch = GhostLatch()
        latch.apply([det()], [], 0)
        latch.apply([], [det(conf=0.8, cls=GHOST_CLASS_ID)], 5)
        dets, ghosts = latch.apply([det(conf=0.5)], [], 10)
        self.assertEqual(dets, [])
        self.assertEqual(len(ghosts), 1)
        self.assertEqual(ghosts[0].class_id, GHOST_CLASS_ID)
        self.assertEqual(latch.suppressed, 1)

    def test_confident_enemy_clears_the_latch(self):
        latch = GhostLatch()
        latch.apply([], [det(conf=0.8, cls=GHOST_CLASS_ID)], 0)
        dets, ghosts = latch.apply([det(conf=0.95)], [], 5)
        self.assertEqual(len(dets), 1)
        self.assertEqual(ghosts, [])

    def test_latch_expires_after_ttl(self):
        latch = GhostLatch(ttl_frames=20)
        latch.apply([], [det(conf=0.8, cls=GHOST_CLASS_ID)], 0)
        dets, _ = latch.apply([det(conf=0.5)], [], 100)
        self.assertEqual(len(dets), 1)

    def test_other_sprite_elsewhere_is_untouched(self):
        latch = GhostLatch()
        latch.apply([], [det(conf=0.8, cls=GHOST_CLASS_ID)], 0)
        dets, ghosts = latch.apply([det(x=10, conf=0.5)], [], 5)
        self.assertEqual(len(dets), 1)
        self.assertEqual(ghosts, [])

    def test_low_confidence_ghost_does_not_latch(self):
        latch = GhostLatch(ghost_conf=0.4)
        latch.apply([], [det(conf=0.36, cls=GHOST_CLASS_ID)], 0)
        dets, _ = latch.apply([det(conf=0.5)], [], 5)
        self.assertEqual(len(dets), 1)

    def test_non_enemy_detections_pass_through(self):
        latch = GhostLatch()
        proj = det(cls=int(EnemyClass.PROJECTILE))
        dets, _ = latch.apply([proj], [], 0)
        self.assertEqual(dets, [proj])


if __name__ == "__main__":
    unittest.main()
