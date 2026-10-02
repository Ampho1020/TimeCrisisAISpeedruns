"""Lightweight per-enemy identity tracker for the vision_schedule policy.

Experimental (branch 1-punchman-(1-shot-on-target-only)): assigns a stable
integer ID to each detected ENEMY-class box across ticks via nearest-centroid
matching, so kill suppression can be keyed on enemy IDENTITY instead of a
fixed screen-space hit-location + a short time window (see
env_timecrisis.py's `_kill_stamps` / `KILL_REFRACTORY_*` machinery, which
stays untouched and independently togglable via ENABLE_KILL_REFRACTORY).

Hypothesis under test: a location-based refractory can't tell "this exact
enemy is already dead" from "a new enemy just walked into the same spot", and
its window is a guess at how long a death animation lasts. ID tracking
removes both problems: a track follows ITS enemy's box (not a fixed screen
coordinate), and is only dropped once ITS box has been absent long enough to
be confident the enemy is really gone -- not a fixed tick count tuned against
nothing more than an eval-run anecdote.

This module has no dependency on config.py or env_timecrisis.py so it can be
unit tested in complete isolation (see tests/test_enemy_tracker.py).
"""

from __future__ import annotations


class EnemyTracker:
    """Nearest-centroid multi-object tracker for ENEMY-class detections.

    Not a Kalman filter / proper MOT algorithm -- enemies in this game move
    slowly relative to the ~83ms decision cadence, so greedy nearest-centroid
    matching (closest pairs first, each track and box used at most once per
    update) is simple and sufficient.
    """

    def __init__(self, match_radius: float, expire_ticks: int):
        self.match_radius = float(match_radius)
        self.expire_ticks = int(expire_ticks)
        self._next_id = 1
        # id -> {"x", "y", "hits", "done", "last_seen_tick"}
        self.tracks: dict[int, dict] = {}
        # Cumulative diagnostics (survive track purges).
        self.total_created = 0
        self.total_done = 0
        self.total_credit_misses = 0  # credit_hit() called with a None/expired id

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self.total_created = 0
        self.total_done = 0
        self.total_credit_misses = 0

    def update(self, tick: int, detections) -> None:
        """Match this tick's ENEMY-class detections to existing tracks,
        create new tracks for unmatched boxes, and purge tracks that have
        been absent for more than ``expire_ticks``."""
        boxes = [
            (float(det.cx_norm), float(det.cy_norm))
            for det in (detections or [])
            if int(getattr(det, "class_id", -1)) == 0
        ]

        candidates = []
        for tid, s in self.tracks.items():
            for bi, (bx, by) in enumerate(boxes):
                d = ((s["x"] - bx) ** 2 + (s["y"] - by) ** 2) ** 0.5
                if d <= self.match_radius:
                    candidates.append((d, tid, bi))
        candidates.sort(key=lambda c: c[0])

        matched_tracks: set[int] = set()
        matched_boxes: set[int] = set()
        for _d, tid, bi in candidates:
            if tid in matched_tracks or bi in matched_boxes:
                continue
            matched_tracks.add(tid)
            matched_boxes.add(bi)
            bx, by = boxes[bi]
            s = self.tracks[tid]
            s["x"], s["y"] = bx, by
            s["last_seen_tick"] = tick

        for bi, (bx, by) in enumerate(boxes):
            if bi in matched_boxes:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {
                "x": bx, "y": by, "hits": 0, "done": False,
                "last_seen_tick": tick,
            }
            self.total_created += 1

        self.tracks = {
            tid: s for tid, s in self.tracks.items()
            if (tick - s["last_seen_tick"]) <= self.expire_ticks
        }

    def nearest_track_id(self, x: float, y: float) -> int | None:
        """Return the id of the track nearest (x, y) within match_radius, or
        None if no track is close enough (e.g. no detection ever seen there)."""
        best_id = None
        best_d = self.match_radius
        for tid, s in self.tracks.items():
            d = ((s["x"] - x) ** 2 + (s["y"] - y) ** 2) ** 0.5
            if d <= best_d:
                best_d = d
                best_id = tid
        return best_id

    def credit_hit(self, track_id: int | None) -> None:
        """Record one confirmed RAM hit against a track. Marks it permanently
        "done" on the FIRST hit -- the one-shot-only experiment this branch
        exists to test, regardless of how many hits the enemy actually needs
        to die in-game."""
        if track_id is None or track_id not in self.tracks:
            self.total_credit_misses += 1
            return
        s = self.tracks[track_id]
        s["hits"] += 1
        if not s["done"]:
            s["done"] = True
            self.total_done += 1

    def is_done(self, track_id: int | None) -> bool:
        if track_id is None:
            return False
        s = self.tracks.get(track_id)
        return bool(s and s["done"])

    @property
    def track_count(self) -> int:
        return len(self.tracks)
