"""Compare detector weights on reviewed in-domain frames the way the policy sees them.

    python eval_detector.py ghost_dataset/val_indomain.txt best.pt runs/detect/ghost1/weights/best.pt

Runs detector.build_detector (same GHOST hiding / confidence floor as the live env) on
every image of a YOLO list file and scores the ENEMY boxes it returns at IoU >= 0.5:
  recall      GT ENEMY boxes matched by a returned box
  precision   returned boxes matching a GT ENEMY box
  ghost_fp    returned boxes that match a GT GHOST box and no GT ENEMY (what the policy
              would waste shots on); small = restricted to GT boxes under 14% of frame height
"""

import argparse
import os

import cv2

from config import VISION_DETECTION_CONFIDENCE
from detector import build_detector


def _iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _gt(path, w, h):
    label = os.path.join(os.path.dirname(path).replace("/images/", "/labels/"), os.path.basename(path)[:-4] + ".txt")
    out = []
    if os.path.isfile(label):
        for line in open(label):
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, bw, bh = int(p[0]), *(float(v) for v in p[1:5])
                out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h)))
    return out


def score(weights: str, paths: list[str], iou: float = 0.5, small_h: float = 0.14) -> dict:
    det = build_detector(None, torch_model_path=weights, device="cuda",
                         confidence_threshold=VISION_DETECTION_CONFIDENCE)
    s = dict(gt_enemy=0, hit_enemy=0, pred=0, pred_ok=0, ghost_fp=0, gt_small=0, hit_small=0, ghost_gt=0)
    for p in paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        preds = [(d.x, d.y, d.w, d.h) for d in det.detect(img[:, :, ::-1].copy())]
        gt = _gt(p, w, h)
        enemies = [b for c, b in gt if c == 0]
        ghosts = [b for c, b in gt if c == 1]
        s["gt_enemy"] += len(enemies)
        s["ghost_gt"] += len(ghosts)
        for b in enemies:
            ok = any(_iou(b, q) >= iou for q in preds)
            s["hit_enemy"] += ok
            if b[3] / h < small_h:
                s["gt_small"] += 1
                s["hit_small"] += ok
        for q in preds:
            s["pred"] += 1
            if any(_iou(q, b) >= iou for b in enemies):
                s["pred_ok"] += 1
            elif any(_iou(q, b) >= iou for b in ghosts):
                s["ghost_fp"] += 1
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("list_file")
    ap.add_argument("weights", nargs="+")
    args = ap.parse_args()
    paths = [l.strip() for l in open(args.list_file) if l.strip()]
    print(f"{len(paths)} images, conf floor {VISION_DETECTION_CONFIDENCE}")
    for wts in args.weights:
        s = score(wts, paths)
        r = lambda a, b: f"{100.0 * a / b:5.1f}%" if b else "  -- "
        print(f"{wts}\n  recall {r(s['hit_enemy'], s['gt_enemy'])} ({s['hit_enemy']}/{s['gt_enemy']})   "
              f"small-box recall {r(s['hit_small'], s['gt_small'])} ({s['hit_small']}/{s['gt_small']})\n"
              f"  precision {r(s['pred_ok'], s['pred'])} ({s['pred_ok']}/{s['pred']})   "
              f"returned boxes on GHOST sprites: {s['ghost_fp']} of {s['ghost_gt']} GHOST")


if __name__ == "__main__":
    main()
