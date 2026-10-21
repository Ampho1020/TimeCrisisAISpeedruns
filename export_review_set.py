"""Build one CVAT review set from several ``run_eval.py --dump-frames`` captures, centred on missed shots.

    python export_review_set.py captures/base/ep1 captures/base_more/ep1 ... --out captures/review1 [--zip]

Frames exported per capture: every tick that fired without a hit (plus --before ticks
earlier for context) and --random unbiased extra frames, so false positives (sparks,
muzzle flashes), missing boxes and wrong sprite states can all be corrected in CVAT.
Boxes are the detector's own proposals (ENEMY, plus GHOST from RAM hit events), so
review every frame. Output is CVAT "YOLO 1.1" like propose_ghost_labels.py; feed the
re-exported folder to build_ghost_dataset.py --reviewed.
"""

import argparse
import json
import os
import random
import shutil
import sys

import ghost_labels as gl


def select_frames(records: list[dict], before: int = 1, n_random: int = 20, seed: int = 0) -> dict[int, str]:
    """Record index -> reason ("miss", "context", "random")."""
    chosen: dict[int, str] = {}
    for i, r in enumerate(records):
        if r.get("fired", 0) > 0 and r.get("hits", 0) == 0:
            chosen[i] = "miss"
    for i in list(chosen):
        for j in range(max(0, i - before), i):
            chosen.setdefault(j, "context")
    rest = [i for i in range(len(records)) if i not in chosen]
    random.Random(seed).shuffle(rest)
    for i in rest[:n_random]:
        chosen[i] = "random"
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--before", type=int, default=1)
    ap.add_argument("--random", type=int, default=20, help="unbiased extra frames per capture")
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    data_dir = os.path.join(args.out, "obj_train_data")
    if os.path.exists(os.path.join(args.out, "train.txt")):
        sys.exit(f"{args.out} already holds an export; use a fresh directory")
    os.makedirs(data_dir, exist_ok=True)

    names, report, per_reason = [], [], {"miss": 0, "context": 0, "random": 0}
    for dump_dir in args.dump_dirs:
        events = os.path.join(dump_dir, "events.jsonl")
        if not os.path.isfile(events):
            print(f"skip {dump_dir}: no events.jsonl")
            continue
        with open(events) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        records = [r for r in records if os.path.isfile(os.path.join(dump_dir, f"frame_{r['frame']:06d}.png"))]
        labels, _ = gl.propose(records, min_conf=args.min_conf)
        tag = "_".join(os.path.normpath(dump_dir).split(os.sep)[-2:])
        for i, reason in sorted(select_frames(records, args.before, args.random).items()):
            if len(names) >= args.max_frames:
                break
            rec = records[i]
            stem = f"{tag}_f{rec['frame']:06d}"
            shutil.copy2(os.path.join(dump_dir, f"frame_{rec['frame']:06d}.png"), os.path.join(data_dir, stem + ".png"))
            with open(os.path.join(data_dir, stem + ".txt"), "w") as fh:
                fh.write("".join(gl.to_yolo_line(c, b, rec["w"], rec["h"]) + "\n" for c, b in labels[i]))
            names.append(f"data/obj_train_data/{stem}.png")
            report.append({"image": stem, "reason": reason})
            per_reason[reason] += 1

    with open(os.path.join(args.out, "obj.names"), "w") as fh:
        fh.write("\n".join(gl.CLASS_NAMES) + "\n")
    with open(os.path.join(args.out, "obj.data"), "w") as fh:
        fh.write(f"classes = {len(gl.CLASS_NAMES)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
    with open(os.path.join(args.out, "train.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"{len(names)} frames -> {args.out}  ({per_reason['miss']} missed-shot, "
          f"{per_reason['context']} context, {per_reason['random']} random)")
    if args.zip:
        print("zip:", shutil.make_archive(args.out, "zip", args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
