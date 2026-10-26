"""Remember which sprite is already dead: once a sprite's box has been called GHOST, later ENEMY calls on it stay GHOST.

The detector judges every frame on its own, so a collapsed sprite can flip GHOST -> ENEMY again
(typically at low confidence). A dead sprite never recovers, so this keeps a small identity
track per box and latches the GHOST call on it.

``apply`` is called once per detection with the clock the env uses (frames since the episode
start), so the expiry is the same wall time whether vision runs every frame (eval) or every
tick (training). A confident ENEMY box clears the latch: a new enemy that appears where a corpse
just was must not be suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from detector import GHOST_CLASS_ID, Detection, EnemyClass
from ghost_labels import Box, _match_score


@dataclass
class _Track:
    box: Box
    last_seen: int
    ghost: bool


class GhostLatch:
    def __init__(self, ttl_frames: int = 20, release_conf: float = 0.8, ghost_conf: float = 0.4,
                 min_iou: float = 0.3):
        self.ttl_frames = ttl_frames
        self.release_conf = release_conf
        self.ghost_conf = ghost_conf
        self.min_iou = min_iou
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_Track] = []
        self.suppressed = 0  # ENEMY boxes turned into GHOST since reset (diagnostic)

    def apply(self, dets: list[Detection], ghosts: list[Detection], now: int) -> tuple[list[Detection], list[Detection]]:
        self._tracks = [t for t in self._tracks if now - t.last_seen <= self.ttl_frames]
        items = [(d, False) for d in dets if int(d.class_id) == int(EnemyClass.ENEMY)] + [(g, True) for g in ghosts]
        boxes = [Box(d.x, d.y, d.w, d.h, d.confidence) for d, _ in items]

        cands = []
        for ti, t in enumerate(self._tracks):
            for bi, b in enumerate(boxes):
                s = _match_score(t.box, b, self.min_iou)
                if s > 0:
                    cands.append((s, ti, bi))
        cands.sort(reverse=True)
        track_of: dict[int, _Track] = {}
        used_t: set[int] = set()
        for _s, ti, bi in cands:
            if ti in used_t or bi in track_of:
                continue
            used_t.add(ti)
            track_of[bi] = self._tracks[ti]
        for bi, b in enumerate(boxes):
            if bi not in track_of:
                t = _Track(b, now, False)
                self._tracks.append(t)
                track_of[bi] = t

        out_dets = [d for d in dets if int(d.class_id) != int(EnemyClass.ENEMY)]
        out_ghosts = list(ghosts)
        for bi, (d, is_ghost) in enumerate(items):
            t = track_of[bi]
            t.box, t.last_seen = boxes[bi], now
            if is_ghost:
                if d.confidence >= self.ghost_conf:
                    t.ghost = True
            elif t.ghost and d.confidence >= self.release_conf:
                t.ghost = False
            if not is_ghost:
                if t.ghost:
                    out_ghosts.append(replace(d, class_id=GHOST_CLASS_ID))
                    self.suppressed += 1
                else:
                    out_dets.append(d)
        return out_dets, out_ghosts
