"""Vision diagnostic tool -- see what the detector sees and how confident it is.

Two modes:

  LIVE (default) -- connects to a running BizHawk + Lua bridge exactly like
  es_train.py does, captures frames in a loop, runs ClassicalDetector on each,
  and saves annotated PNGs under ``--out-dir`` (default: /tmp/vision_inspect/).

    python inspect_vision.py [--n 20] [--every 3] [--out-dir /tmp/vis]

  FILE -- annotates a directory of raw PNG frames produced by
  ``run_eval.py --dump-frames <dir>`` (no BizHawk required):

    python inspect_vision.py --frames <dir> [--out-dir /tmp/vis]

Each saved PNG has:
  * A coloured bounding box per detection, colour-coded by class:
      ENEMY       = red
            GRENADE     = orange
      PROJECTILE  = yellow
  * A label showing   CLASS  conf=0.83   e.g. "ENEMY  conf=0.83"
  * A small green cross-hair at the detector centroid (cx_norm, cy_norm)
  * A header bar at the top listing all detections in text form so you can
    read confidences without zooming into the image
  * A dashed cross-hair at the NEAREST detection centroid (the one
    act_vision_schedule would pick with uniform class-priority) so you can
    immediately see where the gain would steer the reticle

Unannotated frames are written alongside as <name>_raw.png so you can
compare side-by-side.
"""

import argparse
import os
import sys
import time

import numpy as np

# Lazy-import cv2 so the error message is clear on machines without it.
try:
    import cv2
except ImportError:
    print("ERROR: cv2 not found. Install opencv-python-headless from requirements.txt.")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))


def _build_detector(args):
    """Return the detector backend requested on the CLI.

    ``--classical`` forces ``ClassicalDetector`` regardless of config.
    Otherwise delegates to ``detector.build_detector`` with ``--onnx`` (or,
    if omitted, ``config.VISION_ONNX_MODEL_PATH``) -- same selection logic
    ``TimeCrisisEnv`` uses, so what you see here matches what training sees.
    """
    from detector import build_detector, ClassicalDetector

    if args.classical:
        return ClassicalDetector()
    from config import VISION_ONNX_MODEL_PATH
    onnx_path = args.onnx if args.onnx is not None else VISION_ONNX_MODEL_PATH
    det = build_detector(onnx_path or None)
    print(f"[inspect] Using {type(det).__name__}"
          + (f" ({onnx_path})" if type(det).__name__ == "ONNXDetector" else ""))
    return det


# ---------------------------------------------------------------------------
# Class colours (BGR for cv2 drawing)
# ---------------------------------------------------------------------------
_CLASS_BGR = {
    0: (0,   0,   255),   # ENEMY        -- red
    1: (210, 210, 210),   # GRENADE      -- light grey
    2: (0,   230, 230),   # PROJECTILE   -- yellow
}
_CLASS_NAMES = {0: "ENEMY", 1: "GRENADE", 2: "PROJECTILE"}


def _annotate(frame_rgb: np.ndarray, detections, frame_idx: int) -> np.ndarray:
    """Draw bounding boxes + centroid markers + header bar on ``frame_rgb``.

    Returns the annotated BGR image (cv2 convention) at 3x scale so boxes
    and text are legible even on a 256x224 PS1 frame.
    """
    h, w = frame_rgb.shape[:2]
    scale = 3
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    sh, sw = bgr.shape[:2]

    # Header bar (dark grey, 22px tall) listing detections in text.
    bar_h = 22
    header = np.full((bar_h, sw, 3), 30, dtype=np.uint8)
    det_summary = (
        f"frame {frame_idx:04d}  |  {len(detections)} detection(s)  |  "
        + "  ".join(
            f"{_CLASS_NAMES.get(int(d.class_id), '?')} {d.confidence:.2f}"
            for d in detections
        )
        if detections
        else f"frame {frame_idx:04d}  |  NO detections"
    )
    cv2.putText(
        header, det_summary, (4, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
    )
    bgr = np.vstack([header, bgr])
    sh, sw = bgr.shape[:2]
    offset_y = bar_h  # all pixel coords below must be shifted by this

    # Pick the "best" detection using uniform class priority (what
    # act_vision_schedule does at gen-0 before ES has tuned priorities).
    best_det = max(detections, key=lambda d: d.confidence) if detections else None

    for det in detections:
        cls_bgr = _CLASS_BGR.get(int(det.class_id), (128, 128, 128))
        # Scaled pixel coords.
        bx = det.x * scale
        by = det.y * scale + offset_y
        bw = det.w * scale
        bh_ = det.h * scale

        # Bounding box.
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh_), cls_bgr, 1)

        # Label: "ENEMY  0.73"
        label = f"{_CLASS_NAMES.get(int(det.class_id), '?')}  {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        # Small filled rect behind text for contrast.
        lx, ly = bx, max(by - th - 2, offset_y)
        cv2.rectangle(bgr, (lx, ly), (lx + tw + 4, ly + th + 4), cls_bgr, cv2.FILLED)
        cv2.putText(
            bgr, label, (lx + 2, ly + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA,
        )

        # Cross-hair at centroid.
        cx = int(det.cx_norm * w * scale)
        cy = int(det.cy_norm * h * scale) + offset_y
        r = 4
        cv2.line(bgr, (cx - r, cy), (cx + r, cy), cls_bgr, 1, cv2.LINE_AA)
        cv2.line(bgr, (cx, cy - r), (cx, cy + r), cls_bgr, 1, cv2.LINE_AA)

    # Larger dashed-circle marker on the "winning" detection centroid.
    if best_det is not None:
        cx = int(best_det.cx_norm * w * scale)
        cy = int(best_det.cy_norm * h * scale) + offset_y
        cv2.drawMarker(bgr, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        cv2.circle(bgr, (cx, cy), 8, (0, 255, 0), 1, cv2.LINE_AA)
        # "AIM" label
        cv2.putText(
            bgr, "AIM", (cx + 10, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA,
        )

    return bgr


def _save_pair(out_dir: str, idx: int, frame_rgb: np.ndarray, annotated_bgr: np.ndarray):
    """Write both the raw frame and the annotated frame."""
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"frame_{idx:04d}_raw.png")
    ann_path = os.path.join(out_dir, f"frame_{idx:04d}_ann.png")
    cv2.imwrite(raw_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(ann_path, annotated_bgr)
    return raw_path, ann_path


def _print_detections(idx: int, detections):
    if not detections:
        print(f"  frame {idx:04d}: no detections")
        return
    for d in detections:
        cname = _CLASS_NAMES.get(int(d.class_id), f"cls{d.class_id}")
        print(
            f"  frame {idx:04d}: {cname:<12s} conf={d.confidence:.3f}  "
            f"box=({d.x},{d.y},{d.w},{d.h})  "
            f"centroid=({d.cx_norm:.3f},{d.cy_norm:.3f})"
        )


# ---------------------------------------------------------------------------
# LIVE mode
# ---------------------------------------------------------------------------

def run_live(args):
    from config import HOST, PORT, STATE_SLOT
    from bridge_client import BridgeClient

    det = _build_detector(args)
    client = BridgeClient(HOST, PORT)
    print(f"[inspect] Listening on {HOST}:{PORT} -- launch BizHawk with the Lua bridge.")
    client.connect()
    print("[inspect] Connected.")

    out_dir = args.out_dir
    n = args.n
    every = args.every
    print(f"[inspect] Capturing {n} frames (1 every {every} ticks) -> {out_dir}")

    captured = 0
    tick = 0
    t_last = time.time()

    while captured < n:
        # Advance a tick to give the game a chance to change state.
        try:
            client.step_frames(every)
        except Exception as exc:
            print(f"[inspect] step failed: {exc}")
            break

        frame_rgb = None
        try:
            frame_rgb = client.get_screenshot()
        except Exception as exc:
            print(f"[inspect] screenshot failed: {exc}")
            tick += 1
            continue

        detections = det.detect(frame_rgb)
        annotated = _annotate(frame_rgb, detections, captured)

        raw_p, ann_p = _save_pair(out_dir, captured, frame_rgb, annotated)
        _print_detections(captured, detections)
        print(f"    saved: {ann_p}")

        captured += 1
        tick += every

    print(f"\n[inspect] Done. {captured} annotated frames in {out_dir}/")
    client.close()


# ---------------------------------------------------------------------------
# FILE mode
# ---------------------------------------------------------------------------

def run_file(args):
    src_dir = args.frames
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Find all PNG/JPG files that DON'T already have _ann / _raw suffix
    # (so we don't accidentally re-annotate our own output if out_dir == src_dir).
    exts = {".png", ".jpg", ".jpeg"}
    files = sorted([
        f for f in os.listdir(src_dir)
        if os.path.splitext(f)[1].lower() in exts
        and not f.endswith("_ann.png")
        and not f.endswith("_raw.png")
    ])
    if not files:
        print(f"[inspect] No images found in {src_dir}")
        return

    print(f"[inspect] Annotating {len(files)} image(s) from {src_dir} -> {out_dir}")
    det = _build_detector(args)

    for idx, fname in enumerate(files):
        path = os.path.join(src_dir, fname)
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  [{idx:04d}] could not read {fname}, skipping")
            continue
        frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Reset detector between files so MOG2 doesn't carry state across
        # unrelated captures in the directory.  Keep it consistent for a
        # run of frames that ARE from the same episode by NOT resetting
        # inside the loop -- users can always pass a single-file directory.
        if idx == 0:
            det.reset()

        detections = det.detect(frame_rgb)
        annotated = _annotate(frame_rgb, detections, idx)

        stem = os.path.splitext(fname)[0]
        ann_path = os.path.join(out_dir, f"{stem}_ann.png")
        raw_path = os.path.join(out_dir, f"{stem}_raw.png")
        cv2.imwrite(raw_path, bgr)
        cv2.imwrite(ann_path, annotated)

        _print_detections(idx, detections)
        print(f"    -> {ann_path}")

    print(f"\n[inspect] Done. Annotated files in {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--frames", metavar="DIR", default=None,
        help="FILE mode: directory of raw PNGs from --dump-frames (no BizHawk needed).",
    )
    parser.add_argument(
        "--out-dir", default="/tmp/vision_inspect",
        help="Where to write annotated PNGs (default: /tmp/vision_inspect).",
    )
    parser.add_argument(
        "--onnx", metavar="PATH", default=None,
        help="Path to an ONNX model to use instead of config.VISION_ONNX_MODEL_PATH "
             "(default: whatever config.VISION_ONNX_MODEL_PATH points at).",
    )
    parser.add_argument(
        "--classical", action="store_true",
        help="Force the classical MOG2 detector, ignoring --onnx / config.",
    )
    # Live-mode flags
    parser.add_argument(
        "--n", type=int, default=20,
        help="[LIVE] Number of frames to capture (default: 20).",
    )
    parser.add_argument(
        "--every", type=int, default=3,
        help="[LIVE] Decision ticks to advance between captures (default: 3, "
             "matches config.VISION_CAPTURE_EVERY_N_TICKS).",
    )
    args = parser.parse_args()

    if args.frames:
        run_file(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
