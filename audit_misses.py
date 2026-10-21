"""Bucket the missed shots of a ``run_eval.py --dump-frames`` capture and render contact sheets.

    python audit_misses.py captures/ep1 [more dirs] [--on-target 0.08] [--ghost-ticks 6] [--per-sheet 12] [--stats]

Each decision tick that fired but did not hit is put into one bucket:
  blind        no detection on the frame
  ghost_window a hit landed near the aim point within --ghost-ticks ticks earlier
  off_target   nearest box aim point is >= --on-target from the aim
  low_conf     aimed at a box with confidence < 0.5
  fresh        aimed at a confident box with no recent hit nearby (the unexplained misses)

Sheets (<dump_dir>/audit/<bucket>.png) draw detector boxes (green), the aim
(red cross) and the nearest box's aim point (yellow), so a human can tell false
positives from unhittable or mis-aimed enemies. The aim is the end-of-tick aim,
so the red cross can sit a little off from where a mid-tick pulse landed.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

from config import ENEMY_AIM_Y_FRACTION, KILL_REFRACTORY_RADIUS

BUCKETS = ["blind", "ghost_window", "off_target", "low_conf", "fresh"]


def classify_miss(rec: dict, earlier_hits: list[tuple[int, float, float]], *,
                  on_target: float = 0.08, ghost_ticks: int = 6,
                  conf_floor: float = 0.5) -> tuple[str, float, float]:
    """Return (bucket, nearest_dist, nearest_conf) for one fired-but-missed tick.

    ``earlier_hits`` is (tick, x, y) of every RAM hit before this record."""
    dets = rec.get("dets", [])
    if not dets:
        return "blind", -1.0, 0.0
    ax, ay = rec["aim"]
    w, h = rec["w"], rec["h"]
    best = None
    for d in dets:
        px = (d["x"] + d["w"] / 2.0) / w
        py = (d["y"] + ENEMY_AIM_Y_FRACTION * d["h"]) / h
        dist = math.hypot(px - ax, py - ay)
        if best is None or dist < best[0]:
            best = (dist, d["conf"])
    dist, conf = best
    for tick, hx, hy in reversed(earlier_hits):
        if rec["tick"] - tick > ghost_ticks:
            break
        if math.hypot(hx - ax, hy - ay) <= KILL_REFRACTORY_RADIUS:
            return "ghost_window", dist, conf
    if dist >= on_target:
        return "off_target", dist, conf
    if conf < conf_floor:
        return "low_conf", dist, conf
    return "fresh", dist, conf


def audit(records: list[dict], **kw) -> list[dict]:
    """One entry per missed shot tick: {"rec", "bucket", "dist", "conf"}."""
    hits: list[tuple[int, float, float]] = []
    out = []
    for rec in records:
        if rec.get("fired", 0) > 0 and rec.get("hits", 0) == 0:
            bucket, dist, conf = classify_miss(rec, hits, **kw)
            out.append({"rec": rec, "bucket": bucket, "dist": dist, "conf": conf})
        for _, hx, hy in rec.get("hit_events", []):
            hits.append((rec["tick"], hx, hy))
    return out


def fired_tick_features(records: list[dict]) -> list[dict]:
    """One entry per fired tick (hit or miss) with the nearest box's size/conf and the
    count of consecutive earlier missed ticks aimed at the same spot."""
    out = []
    streak = 0
    last_pos = None
    last_tick = None
    for rec in records:
        if rec.get("fired", 0) <= 0:
            continue
        dets = rec.get("dets", [])
        w, h = rec["w"], rec["h"]
        ax, ay = rec["aim"]
        best = None
        for d in dets:
            px = (d["x"] + d["w"] / 2.0) / w
            py = (d["y"] + ENEMY_AIM_Y_FRACTION * d["h"]) / h
            dist = math.hypot(px - ax, py - ay)
            if best is None or dist < best[0]:
                best = (dist, d, px, py)
        pos = (best[2], best[3]) if best else None
        near = (pos is not None and last_pos is not None and last_tick is not None
                and rec["tick"] - last_tick <= 2
                and math.hypot(pos[0] - last_pos[0], pos[1] - last_pos[1]) <= 0.06)
        streak = streak + 1 if near else 0
        hit = rec.get("hits", 0) > 0
        out.append({
            "hit": hit, "has_box": best is not None,
            "h_norm": best[1]["h"] / h if best else -1.0,
            "conf": best[1]["conf"] if best else -1.0,
            "streak": streak,
        })
        if hit:
            streak = 0
        last_pos, last_tick = (None if hit else pos), rec["tick"]
    return out


def print_hit_rates(feats: list[dict]) -> None:
    """Hit rate of fired ticks by nearest-box height, confidence and miss streak."""
    def table(title, key, edges):
        print(f"  by {title}:")
        for lo, hi in edges:
            sel = [f for f in feats if f["has_box"] and lo <= key(f) < hi]
            if sel:
                h = sum(f["hit"] for f in sel)
                print(f"    [{lo:g},{hi:g}) : {len(sel):3d} fired ticks, hit {100.0 * h / len(sel):4.1f}%")
    n = len(feats)
    print(f"pooled: {n} fired ticks, hit {100.0 * sum(f['hit'] for f in feats) / n if n else 0:.1f}%")
    table("box height (fraction of frame)", lambda f: f["h_norm"], [(0, 0.08), (0.08, 0.14), (0.14, 0.22), (0.22, 9)])
    table("box confidence", lambda f: f["conf"], [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)])
    table("earlier consecutive misses on the same spot", lambda f: f["streak"], [(0, 1), (1, 2), (2, 4), (4, 99)])


def _annotate(img, rec: dict, label: str):
    import cv2
    w, h = rec["w"], rec["h"]
    for d in rec.get("dets", []):
        cv2.rectangle(img, (d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"]), (0, 255, 0), 1)
        cv2.putText(img, f"{d['conf']:.2f}", (d["x"], max(8, d["y"] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        px = int(d["x"] + d["w"] / 2.0)
        py = int(d["y"] + ENEMY_AIM_Y_FRACTION * d["h"])
        cv2.circle(img, (px, py), 3, (0, 255, 255), -1)
    ax, ay = int(rec["aim"][0] * w), int(rec["aim"][1] * h)
    cv2.drawMarker(img, (ax, ay), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    cv2.putText(img, label, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def render_sheet(dump_dir: str, entries: list[dict], path: str, per_sheet: int) -> None:
    import cv2
    import numpy as np
    tiles = []
    for e in entries[:per_sheet]:
        rec = e["rec"]
        img = cv2.imread(os.path.join(dump_dir, f"frame_{rec['frame']:06d}.png"))
        if img is None:
            continue
        tiles.append(_annotate(img, rec, f"f{rec['frame']} d={e['dist']:.2f} c={e['conf']:.2f}"))
    if not tiles:
        return
    h, w = tiles[0].shape[:2]
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(t, (w, h))
    cv2.imwrite(path, sheet)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dirs", nargs="+")
    ap.add_argument("--on-target", type=float, default=0.08)
    ap.add_argument("--ghost-ticks", type=int, default=6)
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--stats", action="store_true",
                    help="also print hit rate by box size / confidence / miss streak, pooled over all dirs")
    args = ap.parse_args()

    pooled_counts: Counter = Counter()
    pooled_feats: list[dict] = []
    for dump_dir in args.dump_dirs:
        events = os.path.join(dump_dir, "events.jsonl")
        if not os.path.isfile(events):
            print(f"{events} not found; capture with run_eval.py --dump-frames")
            return 1
        with open(events) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        if records and "aim" not in records[0]:
            print(f"{dump_dir}: capture predates the 'aim' field; recapture with the current run_eval.py")
            return 1

        fired_ticks = sum(1 for r in records if r.get("fired", 0) > 0)
        entries = audit(records, on_target=args.on_target, ghost_ticks=args.ghost_ticks)
        counts = Counter(e["bucket"] for e in entries)
        pooled_counts.update(counts)
        pooled_feats += fired_tick_features(records)
        print(f"{dump_dir}: {len(records)} frames, {fired_ticks} ticks with shots, {len(entries)} missed")
        for b in BUCKETS:
            n = counts.get(b, 0)
            print(f"  {b:13s}: {n:3d} ({100.0 * n / len(entries) if entries else 0:4.1f}% of misses)")

        if not args.no_sheets and entries:
            out_dir = os.path.join(dump_dir, "audit")
            os.makedirs(out_dir, exist_ok=True)
            by_bucket = defaultdict(list)
            for e in entries:
                by_bucket[e["bucket"]].append(e)
            for b, es in by_bucket.items():
                render_sheet(dump_dir, es, os.path.join(out_dir, f"{b}.png"), args.per_sheet)
            print(f"sheets written to {out_dir}")

    if len(args.dump_dirs) > 1:
        total = sum(pooled_counts.values())
        print(f"POOLED misses: {total}")
        for b in BUCKETS:
            n = pooled_counts.get(b, 0)
            print(f"  {b:13s}: {n:3d} ({100.0 * n / total if total else 0:4.1f}%)")
    if args.stats:
        print_hit_rates(pooled_feats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
