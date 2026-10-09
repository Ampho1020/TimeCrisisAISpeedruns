"""Label tooling for the 2-class ENEMY / GHOST detector.

GHOST = an enemy sprite that is already dead or dying (its death / fall
animation after the fatal hit). Shooting it is wasted ammo, and the detector
currently boxes it as ENEMY, which is what we want to fix.

The old corpus (human gameplay video) has no RAM signal, so GHOST cannot be
derived there. The agent's own play does: every RAM-confirmed hit says exactly
which enemy was hit and when. ``propose`` follows that enemy's box after its
last hit; if the box then vanishes within ``max_ghost_ticks`` the enemy was
killed and the frames in between are GHOST. A box that never vanishes is a
survivor (multi-hit enemy) and stays ENEMY. These are PROPOSALS: ENEMY boxes
are the current detector's own output, so a human pass in CVAT is expected.

Pure logic (config is only read for the aim-point constant) so it is unit
tested without the emulator -- see tests/test_ghost_labels.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import ENEMY_AIM_Y_FRACTION

ENEMY_ID = 0
GHOST_ID = 1
CLASS_NAMES = ["ENEMY", "GHOST"]


# ---------------------------------------------------------------------------
# Migrating the old 3-class corpus
# ---------------------------------------------------------------------------


def migrate_label_text(text: str) -> str:
    """3-class (ENEMY/GRENADE/PROJECTILE) YOLO label text -> 2-class.

    ENEMY (0) is kept unchanged; GRENADE (1) and PROJECTILE (2) are dropped
    (together they were ~2.7% of boxes). Returns "" when nothing is left."""
    keep = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(float(parts[0])) == 0:
            keep.append(" ".join(parts))
    return "\n".join(keep) + ("\n" if keep else "")


# ---------------------------------------------------------------------------
# Boxes and tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def aim(self) -> tuple[float, float]:
        return self.cx, self.y + ENEMY_AIM_Y_FRACTION * self.h


def iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    ih = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _match_score(a: Box, b: Box, min_iou: float) -> float:
    """IoU if boxes overlap enough, else a small score when the centres are
    close (fast movers barely overlap between ticks), else 0."""
    v = iou(a, b)
    if v >= min_iou:
        return v
    if abs(a.cx - b.cx) <= 0.5 * max(a.w, b.w) and abs(a.cy - b.cy) <= 0.5 * max(a.h, b.h):
        return 0.01 + 0.5 * v
    return 0.0


@dataclass
class Track:
    id: int
    obs: dict[int, Box] = field(default_factory=dict)  # record index -> box
    bidx: dict[int, int] = field(default_factory=dict)  # record index -> index in that record's box list

    @property
    def first(self) -> int:
        return min(self.obs)

    @property
    def last(self) -> int:
        return max(self.obs)


def track_boxes(frames: list[list[Box]], min_iou: float = 0.3, max_gap: int = 1) -> list[Track]:
    """Greedy frame-to-frame tracking. ``max_gap`` missed records are tolerated
    (detector flicker) before a track is considered ended."""
    tracks: list[Track] = []
    for i, boxes in enumerate(frames):
        active = [t for t in tracks if i - t.last <= max_gap + 1]
        cands = []
        for t in active:
            last_box = t.obs[t.last]
            for bi, b in enumerate(boxes):
                s = _match_score(last_box, b, min_iou)
                if s > 0:
                    cands.append((s, t.id, bi))
        cands.sort(reverse=True)
        used_t: set[int] = set()
        used_b: set[int] = set()
        by_id = {t.id: t for t in tracks}
        for _s, tid, bi in cands:
            if tid in used_t or bi in used_b:
                continue
            used_t.add(tid)
            used_b.add(bi)
            by_id[tid].obs[i] = boxes[bi]
            by_id[tid].bidx[i] = bi
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                tracks.append(Track(id=len(tracks), obs={i: b}, bidx={i: bi}))
    return tracks


def attribute_hit(aim_x: float, aim_y: float, boxes: list[Box], img_w: float,
                  img_h: float, margin_frac: float = 0.03) -> int | None:
    """Index of the box a hit at normalised aim (aim_x, aim_y) landed on: the
    nearest box (by its aim point) that contains the aim point plus a margin."""
    px, py = aim_x * img_w, aim_y * img_h
    mx, my = margin_frac * img_w, margin_frac * img_h
    best, best_d = None, None
    for i, b in enumerate(boxes):
        if not (b.x - mx <= px <= b.x + b.w + mx and b.y - my <= py <= b.y + b.h + my):
            continue
        ax, ay = b.aim
        d = ((ax - px) ** 2 + (ay - py) ** 2) ** 0.5
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


def _boxes_of(rec: dict, min_conf: float) -> list[Box]:
    return [
        Box(d["x"], d["y"], d["w"], d["h"], d.get("conf", 1.0))
        for d in rec.get("dets", [])
        if int(d["c"]) == ENEMY_ID and d.get("conf", 1.0) >= min_conf
    ]


def propose(records: list[dict], *, max_ghost_ticks: int = 12, min_conf: float = 0.5,
            vanish_gap: int = 2):
    """Derive ENEMY/GHOST label proposals from one episode's events.jsonl.

    Returns ``(labels, report)``: ``labels[i]`` is a list of ``(class_id, Box)``
    for record ``i``; ``report`` has one row per hit-on track plus unattributed
    hits. ``vanish_gap`` records must remain after a track ends for it to count
    as vanished rather than cut off by the end of the episode."""
    frames = [_boxes_of(r, min_conf) for r in records]
    tracks = track_boxes(frames)
    n = len(records)

    owner: dict[tuple[int, int], int] = {}  # (record, box index) -> track id
    for t in tracks:
        for i, bi in t.bidx.items():
            owner[(i, bi)] = t.id

    hits_by_track: dict[int, list[int]] = {}
    unattributed = 0
    for i, rec in enumerate(records):
        for ev in rec.get("hit_events", []):
            ax, ay = float(ev[1]), float(ev[2])
            w, h = rec["w"], rec["h"]
            bi = attribute_hit(ax, ay, frames[i], w, h)
            ri = i
            if bi is None and i > 0:  # fatal hit: box already gone from this tick's image
                bi = attribute_hit(ax, ay, frames[i - 1], w, h)
                ri = i - 1
            if bi is None:
                unattributed += 1
                continue
            hits_by_track.setdefault(owner[(ri, bi)], []).append(i)

    ghost_obs: set[tuple[int, int]] = set()  # (track id, record) proposed GHOST
    report: list[dict] = []
    by_id = {t.id: t for t in tracks}
    for tid, hit_recs in sorted(hits_by_track.items()):
        t = by_id[tid]
        last_hit = max(hit_recs)
        ticks_after = t.last - last_hit
        ended = t.last < n - 1 - vanish_gap
        if ended and ticks_after <= 0:
            outcome, ghosts = "instant_kill", []
        elif ended and ticks_after <= max_ghost_ticks:
            outcome = "kill"
            ghosts = [i for i in sorted(t.obs) if i > last_hit]
        elif ended:
            outcome, ghosts = "survivor_or_slow_death", []
        else:
            outcome, ghosts = "cut_off", []
        ghost_obs.update((tid, i) for i in ghosts)
        report.append({
            "track": tid, "first_frame": records[t.first]["frame"],
            "last_frame": records[t.last]["frame"], "hits": len(hit_recs),
            "last_hit_frame": records[last_hit]["frame"], "outcome": outcome,
            "ghost_frames": len(ghosts), "ticks_after_last_hit": max(0, ticks_after),
        })

    labels: list[list[tuple[int, Box]]] = [[] for _ in range(n)]
    for t in tracks:
        for i, b in t.obs.items():
            cls = GHOST_ID if (t.id, i) in ghost_obs else ENEMY_ID
            labels[i].append((cls, b))
    report.append({"track": None, "unattributed_hits": unattributed})
    return labels, report


def to_yolo_line(cls: int, b: Box, img_w: float, img_h: float) -> str:
    def clip(v: float) -> float:
        return min(1.0, max(0.0, v))
    return (f"{cls} {clip(b.cx / img_w):.6f} {clip(b.cy / img_h):.6f} "
            f"{clip(b.w / img_w):.6f} {clip(b.h / img_h):.6f}")
