"""Build the 2-class (ENEMY / GHOST) YOLO dataset for ultralytics.

    python build_ghost_dataset.py --out ghost_dataset \
        --reviewed captures/ep1_reviewed captures/ep2_reviewed \
        --val-reviewed captures/ep3_reviewed

* The old 3-class corpus (maingame_fulltrain) is migrated: ENEMY is kept,
  GRENADE and PROJECTILE are dropped, frames with no label file stay
  background. It cannot contain GHOST (human gameplay video has no RAM signal).
* --reviewed / --val-reviewed are CVAT "YOLO 1.1" exports of reviewed capture
  proposals (a folder holding obj_train_data/, or the images+labels directly).
  Whole captures go to val, never single frames, so train/val do not leak.
* Captured frames are repeated --repeat times in train.txt: they are the only
  source of GHOST and the old corpus labels dying sprites ENEMY, so they need
  weight to win that conflict.

Images are symlinked, labels written fresh; the old dataset is not modified.
"""

import argparse
import os
import sys

import ghost_labels as gl

IMG_EXTS = (".png", ".jpg", ".jpeg")


def _link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def _write(path: str, text: str) -> None:
    with open(path, "w") as fh:
        fh.write(text)


def add_old_corpus(root: str, out: str) -> dict[str, list[str]]:
    """Migrate maingame_fulltrain's train/val/test lists. Returns split -> image paths."""
    img_dir = os.path.join(out, "images", "old")
    lab_dir = os.path.join(out, "labels", "old")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    splits: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        lst = os.path.join(root, f"autosplit_{split}.txt")
        if not os.path.isfile(lst):
            continue
        paths = []
        for line in open(lst):
            line = line.strip()
            if not line:
                continue
            src = os.path.normpath(os.path.join(root, line))
            stem = os.path.splitext(os.path.basename(src))[0]
            src_label = os.path.join(root, "labels", os.path.relpath(os.path.dirname(src), os.path.join(root, "images")), stem + ".txt")
            text = open(src_label).read() if os.path.isfile(src_label) else ""
            name = f"{split}_{os.path.basename(src)}"
            _link(src, os.path.join(img_dir, name))
            _write(os.path.join(lab_dir, os.path.splitext(name)[0] + ".txt"), gl.migrate_label_text(text))
            paths.append(os.path.abspath(os.path.join(img_dir, name)))
        splits[split] = paths
    return splits


def add_reviewed(folder: str, tag: str, out: str) -> list[str]:
    """Add one reviewed capture (CVAT YOLO 1.1 export). Returns image paths."""
    src_dir = os.path.join(folder, "obj_train_data")
    if not os.path.isdir(src_dir):
        src_dir = folder
    img_dir = os.path.join(out, "images", tag)
    lab_dir = os.path.join(out, "labels", tag)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    paths = []
    for fname in sorted(os.listdir(src_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMG_EXTS:
            continue
        label_src = os.path.join(src_dir, stem + ".txt")
        text = open(label_src).read() if os.path.isfile(label_src) else ""
        for line in text.splitlines():
            if line.split() and int(float(line.split()[0])) not in (gl.ENEMY_ID, gl.GHOST_ID):
                raise SystemExit(f"{label_src}: class {line.split()[0]} is not 0 (ENEMY) or 1 (GHOST); "
                                 "check the CVAT label order matches obj.names")
        _link(os.path.join(src_dir, fname), os.path.join(img_dir, fname))
        _write(os.path.join(lab_dir, stem + ".txt"), text)
        paths.append(os.path.abspath(os.path.join(img_dir, fname)))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ghost_dataset")
    ap.add_argument("--old", default="maingame_fulltrain", help="old 3-class dataset root")
    ap.add_argument("--no-old", action="store_true", help="do not include the migrated old corpus")
    ap.add_argument("--reviewed", nargs="*", default=[], help="reviewed capture folders -> train")
    ap.add_argument("--val-reviewed", nargs="*", default=[], help="reviewed capture folders -> val")
    ap.add_argument("--repeat", type=int, default=3, help="times each train capture frame is listed")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    if not args.no_old:
        if not os.path.isdir(args.old):
            print(f"old dataset {args.old} not found (use --no-old to skip)")
            return 1
        old = add_old_corpus(args.old, out)
        train += old.get("train", [])
        val += old.get("val", [])
        test += old.get("test", [])
    n_cap_train = n_cap_val = 0
    for i, folder in enumerate(args.reviewed):
        imgs = add_reviewed(folder, f"cap_train_{i}", out)
        train += imgs * max(1, args.repeat)
        n_cap_train += len(imgs)
    for i, folder in enumerate(args.val_reviewed):
        imgs = add_reviewed(folder, f"cap_val_{i}", out)
        val += imgs
        n_cap_val += len(imgs)
    if not val:
        print("no validation images: pass --val-reviewed or keep the old corpus")
        return 1

    _write(os.path.join(out, "train.txt"), "\n".join(train) + "\n")
    _write(os.path.join(out, "val.txt"), "\n".join(val) + "\n")
    yaml = f"path: {out}\ntrain: train.txt\nval: val.txt\n"
    if test:
        _write(os.path.join(out, "test.txt"), "\n".join(test) + "\n")
        yaml += "test: test.txt\n"
    yaml += "\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(gl.CLASS_NAMES))
    _write(os.path.join(out, "dataset.yaml"), yaml)

    print(f"{out}\n  train lines: {len(train)} (capture frames: {n_cap_train} x{max(1, args.repeat)})"
          f"\n  val images: {len(val)} (capture frames: {n_cap_val})")
    if n_cap_train == 0:
        print("  NOTE: no reviewed captures, so there are no GHOST labels yet.")
    print(f"\n  yolo detect train model=best.pt data={out}/dataset.yaml epochs=50 imgsz=320 batch=16 device=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
