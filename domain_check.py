"""Compare the training corpus with emulator captures (the video -> emulator domain gap).

    python domain_check.py maingame_fulltrain/images/train captures/ep1 [--n 150]

For a random sample of each folder prints image size, mean brightness, and what
the live detector sees: boxes per frame, normalised box size and confidence. A
large mismatch (different aspect ratio, brightness, box size) means detector
metrics on the video corpus will not transfer to the game.
"""

import argparse
import glob
import os
import random
import statistics
import sys

import cv2

from config import (
    VISION_DETECTION_CONFIDENCE, VISION_DETECTOR_DEVICE, VISION_ONNX_MODEL_PATH,
    VISION_TORCH_MODEL_PATH,
)


def _images(folder: str) -> list[str]:
    paths = []
    for ext in ("png", "jpg", "jpeg"):
        paths += glob.glob(os.path.join(folder, f"*.{ext}"))
    return sorted(paths)


def summarize(det, folder: str, n: int, seed: int = 0) -> dict:
    paths = _images(folder)
    random.Random(seed).shuffle(paths)
    sizes, bright, counts, ws, hs, confs = set(), [], [], [], [], []
    for p in paths[:n]:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        sizes.add((w, h))
        bright.append(float(bgr.mean()))
        dets = det.detect(bgr[:, :, ::-1].copy())
        counts.append(len(dets))
        for d in dets:
            ws.append(d.w / w)
            hs.append(d.h / h)
            confs.append(float(d.confidence))

    def mean(xs):
        return statistics.fmean(xs) if xs else float("nan")

    return {
        "folder": folder, "images": len(counts), "sizes": sorted(sizes),
        "brightness": mean(bright), "boxes_per_frame": mean(counts),
        "frames_with_box": mean([1.0 if c else 0.0 for c in counts]),
        "box_w": mean(ws), "box_h": mean(hs), "conf": mean(confs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus")
    ap.add_argument("capture")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    from detector import build_detector
    det = build_detector(
        VISION_ONNX_MODEL_PATH or None,
        torch_model_path=VISION_TORCH_MODEL_PATH or None,
        device=VISION_DETECTOR_DEVICE,
        confidence_threshold=VISION_DETECTION_CONFIDENCE,
    )
    for folder in (args.corpus, args.capture):
        if not _images(folder):
            sys.exit(f"no images in {folder}")
        s = summarize(det, folder, args.n)
        print(f"{s['folder']}: {s['images']} images, sizes {s['sizes']}")
        print(f"    brightness {s['brightness']:.1f}  boxes/frame {s['boxes_per_frame']:.2f}  "
              f"frames with a box {100 * s['frames_with_box']:.0f}%")
        print(f"    box w {s['box_w']:.3f}  h {s['box_h']:.3f} (of frame)  conf {s['conf']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
