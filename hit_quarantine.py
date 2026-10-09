"""Hit quarantine: stop engaging an enemy once a confirmed hit has landed on it.

When RAM reports a hit, the ENEMY box the shot was aimed at is quarantined:
it is hidden from the policy (so aim and trigger move on to other enemies
instead of dumping rounds into a dying sprite) and the trigger is blocked on
its aim point. A quarantine ends when that box has been gone for
``absent_ticks`` consecutive ticks (the enemy is really gone), or after
``max_ticks`` as a safety net so a survivor is never ignored forever.

Identity is never tracked across gaps: a box that disappears and later
reappears at the same spot is a new enemy, because the quarantine is already
released by then. Stacked enemies sharing one box are deliberately not handled.

Pure logic, no config/env imports, so it can be unit tested without the
emulator (see tests/test_hit_quarantine.py).
"""

from __future__ import annotations


def _aim_point(det) -> tuple[float, float]:
    ax = getattr(det, "aim_x_norm", None)
    ay = getattr(det, "aim_y_norm", None)
    return (
        float(det.cx_norm if ax is None else ax),
        float(det.cy_norm if ay is None else ay),
    )


class HitQuarantine:
    def __init__(
        self,
        max_ticks: int,
        absent_ticks: int,
        match_frac: float,
        point_radius: float,
        origin_radius: float = 0.08,
    ):
        self.max_ticks = int(max_ticks)
        self.absent_ticks = int(absent_ticks)
        self.match_frac = float(match_frac)
        self.point_radius = float(point_radius)
        self.origin_radius = float(origin_radius)
        self.entries: list[dict] = []
        self.created = 0
        self.released_absent = 0
        self.released_timeout = 0
        self.filtered = 0  # detection-ticks hidden from the policy
        self.unboxed_hits = 0  # hits with no ENEMY box near the aim point

    def reset(self) -> None:
        self.entries = []
        self.created = 0
        self.released_absent = 0
        self.released_timeout = 0
        self.filtered = 0
        self.unboxed_hits = 0

    @property
    def active(self) -> bool:
        return bool(self.entries)

    def _same_box(self, entry: dict, det) -> bool:
        cx = det.x + det.w / 2.0
        cy = det.y + det.h / 2.0
        return (
            abs(cx - entry["cx"]) <= self.match_frac * max(entry["w"], det.w)
            and abs(cy - entry["cy"]) <= self.match_frac * max(entry["h"], det.h)
        )

    def add(self, tick: int, x: float, y: float, detections) -> bool:
        """Quarantine the ENEMY box nearest the hit's aim point (x, y).

        Returns True if a new quarantine was created."""
        best = None
        best_d = self.origin_radius
        for det in detections or []:
            if int(det.class_id) != 0:
                continue
            ax, ay = _aim_point(det)
            d = ((ax - x) ** 2 + (ay - y) ** 2) ** 0.5
            if d <= best_d:
                best_d = d
                best = det
        if best is None:
            self.unboxed_hits += 1
            return False
        for e in self.entries:
            if self._same_box(e, best):
                return False  # already quarantined (a round already in flight)
        ax, ay = _aim_point(best)
        self.entries.append({
            "cx": best.x + best.w / 2.0, "cy": best.y + best.h / 2.0,
            "w": float(best.w), "h": float(best.h),
            "ax": ax, "ay": ay,
            "start_tick": int(tick), "absent": 0,
        })
        self.created += 1
        return True

    def update(self, tick: int, detections) -> None:
        """Once per decision tick: follow each quarantined box, and release it
        after it has been gone ``absent_ticks`` ticks or lived ``max_ticks``."""
        if not self.entries:
            return
        enemies = [d for d in (detections or []) if int(d.class_id) == 0]
        keep = []
        for e in self.entries:
            match = next((d for d in enemies if self._same_box(e, d)), None)
            if match is not None:
                e["cx"] = match.x + match.w / 2.0
                e["cy"] = match.y + match.h / 2.0
                e["w"], e["h"] = float(match.w), float(match.h)
                e["ax"], e["ay"] = _aim_point(match)
                e["absent"] = 0
            else:
                e["absent"] += 1
            if e["absent"] >= self.absent_ticks:
                self.released_absent += 1
                continue
            if tick - e["start_tick"] >= self.max_ticks:
                self.released_timeout += 1
                continue
            keep.append(e)
        self.entries = keep

    def filter(self, detections) -> list:
        """Detections the policy may see: quarantined ENEMY boxes removed."""
        dets = detections or []
        if not self.entries:
            return list(dets)
        out = []
        for d in dets:
            if int(d.class_id) == 0 and any(self._same_box(e, d) for e in self.entries):
                self.filtered += 1
                continue
            out.append(d)
        return out

    def blocks(self, x: float, y: float) -> bool:
        """True if aim point (x, y) sits on a quarantined enemy."""
        for e in self.entries:
            if ((e["ax"] - x) ** 2 + (e["ay"] - y) ** 2) ** 0.5 <= self.point_radius:
                return True
        return False
