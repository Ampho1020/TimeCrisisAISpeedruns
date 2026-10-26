"""How stable are the detector's ENEMY / GHOST calls over time, and do they match the RAM?

    python ghost_consistency.py captures/ghostlive/ep1 [more dirs] [--flags-out flags.json] [--sheets DIR]

Boxes of each captured tick (ENEMY and GHOST, as the live detector returned them) are
tracked across ticks, so every sprite gets a label sequence such as "EEEGG". Reported:
  resurrections   a track that is GHOST and later ENEMY again (a dead sprite must never recover)
  flicker         ENEMY -> GHOST -> ENEMY inside one track
  contested       ENEMY and GHOST boxes on the same sprite in one tick (the detector is unsure)
  born ghost      a track whose first call is already GHOST
  unconfirmed     a GHOST track with no RAM hit on it before its first GHOST tick
  missed ghost    a RAM kill whose sprite lingered after the fatal hit but was never called GHOST
Hits come from RAM (events.jsonl hit_events), so the last two compare the detector to the game.
--flags-out writes the frames worth a human look for export_review_set.py --flags.
"""

import argparse
import json
import os
import sys
from collections import Counter

import ghost_labels as gl
from ghost_labels import Box


def frame_boxes(rec: dict, min_conf: float = 0.35) -> list[tuple[str, Box]]:
    """(label, box) per box of one tick; an overlapping ENEMY+GHOST pair becomes one 'C' box."""
    enemies = [Box(d["x"], d["y"], d["w"], d["h"], d.get("conf", 1.0))
               for d in rec.get("dets", []) if int(d["c"]) == gl.ENEMY_ID and d.get("conf", 1.0) >= min_conf]
    ghosts = [Box(d["x"], d["y"], d["w"], d["h"], d.get("conf", 1.0))
              for d in rec.get("ghosts", []) if d.get("conf", 1.0) >= min_conf]
    out: list[tuple[str, Box]] = []
    used_g: set[int] = set()
    for e in enemies:
        pair = next((j for j, g in enumerate(ghosts) if j not in used_g and gl.iou(e, g) >= 0.5), None)
        if pair is None:
            out.append(("E", e))
        else:
            used_g.add(pair)
            out.append(("C", e if e.conf >= ghosts[pair].conf else ghosts[pair]))
    out += [("G", g) for j, g in enumerate(ghosts) if j not in used_g]
    return out


def label_tracks(records: list[dict], min_conf: float = 0.35) -> tuple[list[dict], list[list[tuple[str, Box]]]]:
    """Track all boxes; each track carries its per-tick label string and confidences."""
    fb = [frame_boxes(r, min_conf) for r in records]
    tracks = gl.track_boxes([[b for _, b in f] for f in fb])
    out = []
    for t in tracks:
        idxs = sorted(t.obs)
        out.append({
            "id": t.id, "ticks": idxs,
            "labels": "".join(fb[i][t.bidx[i]][0] for i in idxs),
            "conf": [round(fb[i][t.bidx[i]][1].conf, 3) for i in idxs],
            "box_of": {i: fb[i][t.bidx[i]][1] for i in idxs},
            "bidx": dict(t.bidx),
        })
    return out, fb


def _runs(s: str) -> str:
    return "".join(c for k, c in enumerate(s) if k == 0 or s[k - 1] != c)


def analyze(records: list[dict], min_conf: float = 0.35, vanish_gap: int = 2) -> dict:
    tracks, fb = label_tracks(records, min_conf)
    n = len(records)
    owner = {(i, t["bidx"][i]): t["id"] for t in tracks for i in t["ticks"]}
    hits: dict[int, list[int]] = {}
    for i, rec in enumerate(records):
        for ev in rec.get("hit_events", []):
            boxes = [b for _, b in fb[i]]
            bi = gl.attribute_hit(float(ev[1]), float(ev[2]), boxes, rec["w"], rec["h"])
            ri = i
            if bi is None and i > 0:
                bi = gl.attribute_hit(float(ev[1]), float(ev[2]), [b for _, b in fb[i - 1]], rec["w"], rec["h"])
                ri = i - 1
            if bi is not None:
                hits.setdefault(owner[(ri, bi)], []).append(i)

    flags: list[dict] = []
    counts: Counter = Counter()
    for t in tracks:
        runs = _runs(t["labels"])
        t["hits"] = hits.get(t["id"], [])
        reasons = []
        if "G" in runs and "E" in runs[runs.index("G"):]:
            reasons.append("resurrection")
        if "EGE" in runs:
            reasons.append("flicker")
        if "C" in runs:
            reasons.append("contested")
        has_g = "G" in runs
        if has_g and runs[0] == "G":
            reasons.append("born_ghost")
        if has_g:
            first_g = t["ticks"][t["labels"].index("G")]
            if not any(h <= first_g for h in t["hits"]):
                reasons.append("unconfirmed_ghost")
        if t["hits"]:
            last_hit = max(t["hits"])
            ended = t["ticks"][-1] < n - 1 - vanish_gap
            after = [t["labels"][k] for k, i in enumerate(t["ticks"]) if i > last_hit]
            if ended and after and not any(c in "GC" for c in after):
                reasons.append("missed_ghost")
        t["reasons"] = reasons
        counts["tracks"] += 1
        counts["tracks_with_ghost"] += has_g
        counts["tracks_with_hit"] += bool(t["hits"])
        for r in reasons:
            counts[r] += 1
        for r in reasons:
            for i in t["ticks"]:
                flags.append({"record": i, "frame": records[i]["frame"], "track": t["id"],
                              "reason": r, "labels": t["labels"]})
    ghost_conf = [c for t in tracks for c, l in zip(t["conf"], t["labels"]) if l == "G"]
    return {"tracks": tracks, "counts": dict(counts), "flags": flags, "ghost_conf": ghost_conf}


def replay_latch(records: list[dict], latch=None, frames_per_record: int = 5) -> tuple[list[dict], int]:
    """Run the records' raw detector output through GhostLatch; returns (new records, boxes suppressed)."""
    from detector import Detection
    from ghost_latch import GhostLatch
    latch = latch or GhostLatch()
    latch.reset()

    def to_det(d, cls):
        return Detection(d["x"], d["y"], d["w"], d["h"], cls, d.get("conf", 1.0), 0.0, 0.0)

    def to_dict(d):
        return {"c": int(d.class_id), "conf": d.confidence, "x": d.x, "y": d.y, "w": d.w, "h": d.h}
    out = []
    for i, r in enumerate(records):
        dets, ghosts = latch.apply([to_det(d, int(d["c"])) for d in r.get("dets", [])],
                                   [to_det(g, 3) for g in r.get("ghosts", [])], i * frames_per_record)
        out.append({**r, "dets": [to_dict(d) for d in dets], "ghosts": [to_dict(g) for g in ghosts]})
    return out, latch.suppressed


def _print(name: str, res: dict) -> None:
    c = res["counts"]
    print(f"{name}: {c.get('tracks', 0)} sprites tracked, {c.get('tracks_with_ghost', 0)} called GHOST, "
          f"{c.get('tracks_with_hit', 0)} with a RAM hit")
    for r in ["resurrection", "flicker", "contested", "born_ghost", "unconfirmed_ghost", "missed_ghost"]:
        print(f"  {r:18s}: {c.get(r, 0)}")
    gc = res["ghost_conf"]
    if gc:
        gc = sorted(gc)
        print(f"  GHOST confidence: min {gc[0]:.2f}  median {gc[len(gc) // 2]:.2f}  "
              f"below 0.5: {sum(1 for x in gc if x < 0.5)} of {len(gc)}")


def _sheet(dump_dir: str, records: list[dict], flags: list[dict], tracks: list[dict], path: str, limit: int) -> None:
    import cv2
    import numpy as np
    by_id = {t["id"]: t for t in tracks}
    seen, tiles = set(), []
    for f in flags:
        if (f["track"], f["reason"]) in seen:
            continue
        seen.add((f["track"], f["reason"]))
        t = by_id[f["track"]]
        mid = t["ticks"][len(t["ticks"]) // 2]
        img = cv2.imread(os.path.join(dump_dir, f"frame_{records[mid]['frame']:06d}.png"))
        if img is None:
            continue
        b = t["box_of"][mid]
        cv2.rectangle(img, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), (0, 255, 255), 2)
        cv2.putText(img, f"{f['reason']} {t['labels'][:14]}", (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(img)
        if len(tiles) >= limit:
            break
    if not tiles:
        return
    h, w = tiles[0].shape[:2]
    cols = 4
    rows = -(-len(tiles) // cols)
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(im, (w, h))
    cv2.imwrite(path, sheet)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dirs", nargs="+")
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument("--flags-out")
    ap.add_argument("--sheets", help="directory for one contact sheet per flag reason")
    ap.add_argument("--latch", action="store_true", help="replay the captures through GhostLatch and analyse that")
    ap.add_argument("--flag-unconfirmed", action="store_true",
                    help="also put unconfirmed_ghost frames in --flags-out (weak signal: RAM hit attribution often fails)")
    args = ap.parse_args()

    total: Counter = Counter()
    all_flags: dict[str, list[dict]] = {}
    for d in args.dump_dirs:
        ev = os.path.join(d, "events.jsonl")
        if not os.path.isfile(ev):
            print(f"skip {d}: no events.jsonl")
            continue
        with open(ev) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        res = analyze(records, args.min_conf)
        if args.latch:
            records, n_sup = replay_latch(records)
            res = analyze(records, args.min_conf)
            print(f"{d}: latch turned {n_sup} ENEMY boxes into GHOST")
        _print(d, res)
        total.update(res["counts"])
        all_flags[d] = [f for f in res["flags"] if args.flag_unconfirmed or f["reason"] != "unconfirmed_ghost"]
        if args.sheets:
            os.makedirs(args.sheets, exist_ok=True)
            tag = "_".join(os.path.normpath(d).split(os.sep)[-2:])
            for reason in {f["reason"] for f in res["flags"]}:
                _sheet(d, records, [f for f in res["flags"] if f["reason"] == reason], res["tracks"],
                       os.path.join(args.sheets, f"{tag}_{reason}.png"), 12)
    if len(args.dump_dirs) > 1:
        print("POOLED " + "  ".join(f"{k}={v}" for k, v in sorted(total.items())))
    if args.flags_out:
        with open(args.flags_out, "w") as fh:
            json.dump(all_flags, fh)
        print(f"flags -> {args.flags_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
