"""Turn a ``run_eval.py --dump-frames`` capture into reviewable GHOST proposals.

    python run_eval.py theta_interrupt.npy --dump-frames captures/ep1
    python propose_ghost_labels.py captures/ep1 [--zip]

Writes ``captures/ep1_cvat/`` in CVAT's "YOLO 1.1" layout (obj.names, obj.data,
train.txt, obj_train_data/frame_*.png + .txt). Import it into CVAT as a new
task, fix the boxes (GHOST proposals especially), then export "YOLO 1.1" again
and feed the exported folder to build_ghost_dataset.py.

ENEMY boxes are the current detector's own output and GHOST boxes come from RAM
hit events (see ghost_labels.py), so treat every box as a proposal. The summary
tells you where to look: kills with many GHOST frames are the interesting ones.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter

import ghost_labels as gl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir", help="directory written by run_eval.py --dump-frames")
    ap.add_argument("--out", help="output directory (default: <dump_dir>_cvat)")
    ap.add_argument("--min-conf", type=float, default=0.35,
                    help="ignore detector boxes below this confidence (default 0.35, the "
                         "policy's VISION_DETECTION_CONFIDENCE)")
    ap.add_argument("--max-ghost-ticks", type=int, default=12,
                    help="a box that vanishes within this many ticks of its last hit is a kill "
                         "(default 12); longer is treated as a survivor")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip for CVAT upload")
    ap.add_argument("--context", type=int, nargs=2, default=[2, 4], metavar=("BEFORE", "AFTER"),
                    help="export only frames from BEFORE ticks before to AFTER ticks after each "
                         "RAM hit (default 2 4): the frames where ENEMY vs GHOST matters")
    ap.add_argument("--all-frames", action="store_true", help="export every frame, not just near hits")
    args = ap.parse_args()

    events = os.path.join(args.dump_dir, "events.jsonl")
    if not os.path.isfile(events):
        print(f"{events} not found. Capture with: run_eval.py <checkpoint> --dump-frames {args.dump_dir}")
        return 1
    with open(events) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    records = [r for r in records if os.path.isfile(os.path.join(args.dump_dir, f"frame_{r['frame']:06d}.png"))]
    if not records:
        print("no records with matching PNGs")
        return 1

    labels, report = gl.propose(records, max_ghost_ticks=args.max_ghost_ticks, min_conf=args.min_conf)

    n_all = len(records)
    if args.all_frames:
        keep = set(range(n_all))
    else:
        before, after = args.context
        keep = {j for i, r in enumerate(records) if r.get("hit_events")
                for j in range(max(0, i - before), min(n_all, i + after + 1))}
    sel = [i for i in range(n_all) if i in keep]
    if not sel:
        print("no RAM hits in this capture, so nothing to export (use --all-frames)")
        return 1
    records = [records[i] for i in sel]
    labels = [labels[i] for i in sel]

    out = args.out or args.dump_dir.rstrip("/\\") + "_cvat"
    data_dir = os.path.join(out, "obj_train_data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(out, "obj.names"), "w") as fh:
        fh.write("\n".join(gl.CLASS_NAMES) + "\n")
    with open(os.path.join(out, "obj.data"), "w") as fh:
        fh.write(f"classes = {len(gl.CLASS_NAMES)}\ntrain = data/train.txt\n"
                 "names = data/obj.names\nbackup = backup/\n")
    names = []
    for rec, lab in zip(records, labels):
        stem = f"frame_{rec['frame']:06d}"
        shutil.copy2(os.path.join(args.dump_dir, stem + ".png"), os.path.join(data_dir, stem + ".png"))
        with open(os.path.join(data_dir, stem + ".txt"), "w") as fh:
            fh.write("".join(gl.to_yolo_line(c, b, rec["w"], rec["h"]) + "\n" for c, b in lab))
        names.append(f"data/obj_train_data/{stem}.png")
    with open(os.path.join(out, "train.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    rows = [r for r in report if r.get("outcome")]
    outcomes = Counter(r["outcome"] for r in rows)
    ghost_frames = sum(1 for lab in labels if any(c == gl.GHOST_ID for c, _ in lab))
    total_hits = sum(len(r.get("hit_events", [])) for r in records)
    print(f"{len(records)} of {n_all} frames exported, {total_hits} RAM hits -> {out}")
    print(f"  enemies hit: {len(rows)}   " + "  ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    print(f"  hits with no box under the aim: {report[-1]['unattributed_hits']}")
    print(f"  frames with a GHOST box: {ghost_frames}   "
          f"GHOST boxes: {sum(r['ghost_frames'] for r in rows)}")
    if rows:
        ticks = sorted(r["ticks_after_last_hit"] for r in rows if r["outcome"] == "kill")
        if ticks:
            print(f"  ticks the box lingers after the killing hit: min {ticks[0]}, "
                  f"median {ticks[len(ticks) // 2]}, max {ticks[-1]}")
    if args.zip:
        print("  zip:", shutil.make_archive(out, "zip", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
