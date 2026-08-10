"""Multi-class enemy detector for Time Crisis frames.

This module is Phase 2 of the vision-conditioned schedule plan
(``/memories/session/plan.md``). It replaces the naive single-color
centroid finder in ``vision.py`` with a real, proven-algorithm detector that
outputs bounding boxes + class labels per frame.

Two backends, one contract:

* ``ClassicalDetector`` -- day-1 default, no learned model required. Uses
  OpenCV's proven ``BackgroundSubtractorMOG2`` (Zivkovic 2004/2006) to
  isolate moving foreground, then per-class color-palette masking and
  ``cv2.connectedComponentsWithStats`` blob analysis to recover bounding
  boxes. Classification is a two-tier lookup: (a) palette-color match
  decides the enemy vs. civilian split, (b) heuristic size/aspect rules
  distinguish projectiles (small, fast, off-cursor) and muzzle flashes
  (very small, very bright, near-cursor) from full-body sprites.
  Deterministic given a frame history -- MOG2 mutates its own state
  across calls, so each ``TimeCrisisEnv`` owns exactly one instance.

* ``ONNXDetector`` -- slot for a future fine-tuned YOLO/RT-DETR model.
  ``onnxruntime`` (CPU provider) runs the pre-trained ONNX graph and
  decodes its per-anchor output tensor into the same
  ``list[Detection]`` contract. Kept as a hook because no Time Crisis-
  specific fine-tuned model exists yet -- the day-1 offline workflow to
  produce one is documented in the module footer.

``build_detector()`` picks between the two based on whether
``config.VISION_ONNX_MODEL_PATH`` points at an existing file, so a
downstream call site never has to branch.

Class IDs are stable across backends (see ``EnemyClass`` below). The
class-priority vector learned by ``policy.act_vision_schedule`` uses
these IDs as indices, so the order MUST NOT change without also
retraining every checkpoint that consumed the old order.

Numpy is the only hard dependency at import time; cv2 / onnxruntime are
imported lazily inside the classes that need them so unit tests that only
touch ``Detection`` / ``EnemyClass`` still work in a numpy-only env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Public data contract
# ---------------------------------------------------------------------------


class EnemyClass(IntEnum):
    """Stable class-ID enum used by every detector backend.

    Values are consumed as indices into ``policy.act_vision_schedule``'s
    global class-priority vector. Do NOT reorder or renumber -- old
    checkpoints would silently start weighting the wrong class.
    """

    ENEMY = 0
    CIVILIAN = 1
    PROJECTILE = 2
    MUZZLE_FLASH = 3


NUM_CLASSES = len(EnemyClass)


@dataclass(frozen=True)
class Detection:
    """One detected object in a frame.

    Coordinates are stored in TWO forms so downstream code never has to
    know the frame resolution:

    * ``x, y, w, h`` -- pixel-space top-left corner + size (ints, matches
      cv2.connectedComponentsWithStats output). Useful for overlays and
      debugging dumps.
    * ``cx_norm, cy_norm`` -- centroid in [0, 1] frame-normalized space,
      same convention as ``vision.detect_enemy`` / ``config`` cursor
      normalization (0 = left/top). This is what ``act_vision_schedule``
      consumes to compute the aim blend, so it can never be omitted.

    ``confidence`` is in [0, 1] and is backend-defined:
      * ClassicalDetector: blob-area-based (larger blob = more confident,
        clipped to 1.0 at a per-class saturation area).
      * ONNXDetector: raw sigmoid output of the model's confidence head.
    """

    x: int
    y: int
    w: int
    h: int
    class_id: int
    confidence: float
    cx_norm: float
    cy_norm: float

    @property
    def area(self) -> int:
        return int(self.w) * int(self.h)


# ---------------------------------------------------------------------------
# Default detection tuning (public: callers can override per-instance)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassPalette:
    """Per-class color palette + morphology tuning for the classical detector.

    Every entry is a tuple of representative RGB colors sampled from real
    Time Crisis sprites. A pixel counts as a candidate for a class when it
    falls within ``tolerance`` on every channel of at least one color in
    the palette. Multiple colors per class handle sprite variants
    (different enemy costumes, hit-flash frames) without needing a
    trained classifier.

    ``min_blob_area`` filters out speckle noise; ``saturation_area`` is
    the blob size at which ``confidence`` saturates to 1.0. Both are in
    pixels and must be tuned per FRAME RESOLUTION -- the defaults below
    target BizHawk's ~256x224 or ~320x240 PS1 output.
    """

    colors: tuple[tuple[int, int, int], ...]
    tolerance: int = 40
    min_blob_area: int = 12
    saturation_area: int = 400


# Default palettes are conservative placeholders -- they encode the
# category structure the plan requires, not real per-sprite calibration.
# Phase 2 of the plan explicitly calls out a hand-tuning pass against
# real captures via ``run_eval.py --dump-frames`` before promoting the
# detector to production; the values below are safe defaults for the
# probe sim (see tests/test_simulation.py's TimedSpotMotionColorGame,
# which uses a palette of 5 distinct spot colors we approximate here).
DEFAULT_PALETTES: dict[EnemyClass, ClassPalette] = {
    EnemyClass.ENEMY: ClassPalette(
        colors=(
            (220, 30, 30),   # sim red enemy (matches TimedSpotMotionColorGame)
            (30, 30, 220),   # sim blue enemy
            (30, 220, 30),   # sim green enemy
            (220, 220, 30),  # sim yellow enemy
            (220, 30, 220),  # sim magenta enemy
        ),
        tolerance=40,
        min_blob_area=12,
        saturation_area=400,
    ),
    EnemyClass.CIVILIAN: ClassPalette(
        colors=(
            (240, 220, 180),  # skin-toned civilian placeholder
        ),
        tolerance=25,
        min_blob_area=20,   # civilians typically full-body, larger
        saturation_area=600,
    ),
    EnemyClass.PROJECTILE: ClassPalette(
        colors=(
            (255, 240, 120),  # muzzle-lit bullet/tracer
        ),
        tolerance=30,
        min_blob_area=3,    # bullets are tiny
        saturation_area=40,
    ),
    EnemyClass.MUZZLE_FLASH: ClassPalette(
        colors=(
            (255, 255, 200),  # very bright yellow-white flash
        ),
        tolerance=30,
        min_blob_area=4,
        saturation_area=60,
    ),
}


# ---------------------------------------------------------------------------
# ClassicalDetector -- MOG2 + palette + connected components
# ---------------------------------------------------------------------------


class ClassicalDetector:
    """Proven-algorithm baseline detector: BackgroundSubtractorMOG2 + palette.

    Runtime shape per ``detect(frame)`` call:

      1. MOG2 (Zivkovic's Gaussian mixture background subtractor) is fed
         the incoming frame and produces a foreground mask -- pixels that
         differ meaningfully from the running background model. This
         handles the "static scenery vs. real animating sprite" split
         (memory's failed multi-color probe hit this limitation with a
         hand-rolled frame differencer).
      2. Optional morphological open (kernel 3x3) removes single-pixel
         speckle from the mask -- cheap and standard.
      3. For each ``EnemyClass``, the class's palette-color mask is
         computed as a logical-OR over its palette entries, then ANDed
         with the MOG2 foreground mask (except for MUZZLE_FLASH, which is
         so brief MOG2 may still classify it as background -- for that
         class we use color alone).
      4. ``cv2.connectedComponentsWithStats`` extracts per-class blobs;
         each blob above ``min_blob_area`` becomes a ``Detection`` with
         confidence based on its pixel area capped at
         ``saturation_area``.

    Determinism: MOG2 mutates its internal Gaussian mixture across
    calls, so ``ClassicalDetector`` instances are stateful. Each
    ``TimeCrisisEnv`` owns exactly one instance and calls ``reset()`` on
    it whenever the episode's savestate is reloaded (i.e. when the
    background definitionally changes).
    """

    def __init__(
        self,
        palettes: dict[EnemyClass, ClassPalette] | None = None,
        mog_history: int = 200,
        mog_var_threshold: float = 16.0,
        motion_required: Sequence[EnemyClass] = (
            EnemyClass.ENEMY,
            EnemyClass.CIVILIAN,
            EnemyClass.PROJECTILE,
        ),
    ):
        import cv2  # lazy import: keeps import cost off pure-data test paths

        self._cv2 = cv2
        self.palettes = palettes if palettes is not None else DEFAULT_PALETTES
        # detectShadows=False: we do not need cv2's shadow-classification
        # channel (grey pixels in the mask) -- it just adds a value we'd have
        # to threshold back out. Setting False keeps the output strictly
        # binary and slightly faster.
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=mog_history,
            varThreshold=mog_var_threshold,
            detectShadows=False,
        )
        self._motion_required = set(motion_required)
        # 3x3 rectangular kernel for the morphological open -- smallest
        # value that reliably kills 1-pixel MOG2 speckle without eroding
        # thin sprite features.
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def reset(self) -> None:
        """Discard the accumulated background model.

        Called by ``TimeCrisisEnv.reset()`` right after ``load_state`` so
        the previous episode's background doesn't leak into this one's
        foreground mask.
        """
        cv2 = self._cv2
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=self._mog.getHistory(),
            varThreshold=self._mog.getVarThreshold(),
            detectShadows=False,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detected objects for one HxWx3 uint8 RGB frame.

        ``frame`` must be in the SAME RGB layout that ``vision.decode_bmp``
        produces (row 0 = top of image, channel order R,G,B), so the same
        code path works against sim-synthesized frames and real BizHawk
        captures.
        """
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(
                f"Expected HxWx3(+) uint8 RGB frame, got shape {frame.shape}"
            )
        cv2 = self._cv2
        h, w = frame.shape[:2]
        # MOG2 expects BGR by convention (uses only luma anyway, but the
        # docs assume BGR). Convert once.
        bgr = frame[:, :, [2, 1, 0]].astype(np.uint8, copy=False)
        fg_mask = self._mog.apply(bgr)
        # Morphological open to kill speckle.
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_kernel)

        frame_i16 = frame[:, :, :3].astype(np.int16)
        detections: list[Detection] = []
        for class_id, palette in self.palettes.items():
            color_mask = np.zeros((h, w), dtype=np.uint8)
            for color in palette.colors:
                diff = np.abs(frame_i16 - np.array(color, dtype=np.int16))
                match = np.all(diff <= palette.tolerance, axis=-1)
                color_mask |= match.astype(np.uint8)
            if class_id in self._motion_required:
                # Require BOTH the class color AND MOG2 foreground.
                combined = color_mask & (fg_mask > 0).astype(np.uint8)
            else:
                combined = color_mask
            if not combined.any():
                continue

            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
                combined, connectivity=8,
            )
            # Label 0 is the background component -- skip it.
            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < palette.min_blob_area:
                    continue
                bx = int(stats[i, cv2.CC_STAT_LEFT])
                by = int(stats[i, cv2.CC_STAT_TOP])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx = float(centroids[i, 0])
                cy = float(centroids[i, 1])
                confidence = float(min(1.0, area / max(1, palette.saturation_area)))
                detections.append(
                    Detection(
                        x=bx,
                        y=by,
                        w=bw,
                        h=bh,
                        class_id=int(class_id),
                        confidence=confidence,
                        cx_norm=cx / w,
                        cy_norm=cy / h,
                    )
                )
        return detections


# ---------------------------------------------------------------------------
# ONNXDetector -- slot for a future fine-tuned YOLO/RT-DETR model
# ---------------------------------------------------------------------------


class ONNXDetector:
    """CPU ONNX inference wrapper matching the ``Detection`` contract.

    Assumes a YOLOv8-style ONNX export (single float32 input tensor
    named 'images' of shape (1, 3, H, W) in [0, 1], single output tensor
    of shape (1, 4 + NUM_CLASSES, N_ANCHORS) with per-anchor
    (cx, cy, w, h, score_class0, score_class1, ...)). If your fine-tune
    exports a different layout, wrap the decode logic here rather than
    changing the callers.

    NOTE: no Time Crisis-specific fine-tuned model ships with the repo.
    Producing one is an offline workflow (see the module footer). This
    class only loads a model that already exists at
    ``config.VISION_ONNX_MODEL_PATH``; ``build_detector`` handles the
    absent-model fallback.
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (320, 320),
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
    ):
        import onnxruntime as ort  # lazy import
        import cv2

        self._cv2 = cv2
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        # CPUExecutionProvider is the only guaranteed provider on this
        # dev box; add CUDAExecutionProvider first here if targeting GPU.
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def reset(self) -> None:
        """No stateful history -- ONNX detectors are pure functions of one
        frame -- but implement ``reset()`` for backend-parity with
        ``ClassicalDetector`` so callers don't have to branch."""
        return

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(
                f"Expected HxWx3(+) uint8 RGB frame, got shape {frame.shape}"
            )
        cv2 = self._cv2
        src_h, src_w = frame.shape[:2]
        in_w, in_h = self.input_size
        # Letterbox-free resize: fast, but distorts aspect ratio slightly.
        # Time Crisis PS1 output aspect is close to 4:3 so a 1:1 model
        # input's distortion is small; if training the model at a
        # different aspect, swap this for a letterbox pad.
        resized = cv2.resize(frame[:, :, :3], (in_w, in_h))
        # RGB->CHW float32 [0, 1].
        blob = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]

        raw = self._session.run(None, {self._input_name: blob})[0]
        # Expected shape (1, 4 + NUM_CLASSES, N). Squeeze batch.
        arr = np.squeeze(raw, axis=0)
        if arr.shape[0] < 4 + NUM_CLASSES:
            raise RuntimeError(
                f"ONNX model output has too few channels: got shape "
                f"{arr.shape}, need at least 4 + NUM_CLASSES rows."
            )
        boxes_cxcywh = arr[0:4, :].T  # (N, 4)
        class_scores = arr[4:4 + NUM_CLASSES, :].T  # (N, NUM_CLASSES)
        best_class = np.argmax(class_scores, axis=1)
        best_conf = class_scores[np.arange(class_scores.shape[0]), best_class]

        keep = best_conf >= self.confidence_threshold
        if not keep.any():
            return []
        boxes_cxcywh = boxes_cxcywh[keep]
        best_class = best_class[keep]
        best_conf = best_conf[keep]

        # Convert cxcywh (model-input coords, 0..in_w/in_h) to xyxy in
        # source-image coords for NMS + emit.
        scale_x = src_w / in_w
        scale_y = src_h / in_h
        cx = boxes_cxcywh[:, 0] * scale_x
        cy = boxes_cxcywh[:, 1] * scale_y
        bw = boxes_cxcywh[:, 2] * scale_x
        bh = boxes_cxcywh[:, 3] * scale_y
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        # cv2.dnn.NMSBoxes wants xywh int lists.
        nms_boxes = [
            [int(x1[i]), int(y1[i]), int(bw[i]), int(bh[i])]
            for i in range(x1.shape[0])
        ]
        keep_idx = cv2.dnn.NMSBoxes(
            nms_boxes,
            best_conf.astype(np.float32).tolist(),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_iou_threshold,
        )
        if len(keep_idx) == 0:
            return []
        keep_idx = np.array(keep_idx).flatten()

        detections: list[Detection] = []
        for i in keep_idx:
            bx = max(0, int(x1[i]))
            by = max(0, int(y1[i]))
            w_i = max(1, int(bw[i]))
            h_i = max(1, int(bh[i]))
            cxi = float(cx[i])
            cyi = float(cy[i])
            detections.append(
                Detection(
                    x=bx,
                    y=by,
                    w=w_i,
                    h=h_i,
                    class_id=int(best_class[i]),
                    confidence=float(best_conf[i]),
                    cx_norm=float(cxi / src_w),
                    cy_norm=float(cyi / src_h),
                )
            )
        return detections


# ---------------------------------------------------------------------------
# Factory + selection helper
# ---------------------------------------------------------------------------


def build_detector(onnx_model_path: str | None = None):
    """Return an ONNXDetector if the model file exists, else ClassicalDetector.

    Called once per ``TimeCrisisEnv`` in Phase 3 (never in a hot loop). If
    ``onnx_model_path`` is None or the file is missing, we deliberately
    fall back to the classical detector rather than raising -- day-1
    production has no ONNX model, and forcing a crash on absent-model
    would prevent training from starting.
    """
    if onnx_model_path and os.path.isfile(onnx_model_path):
        return ONNXDetector(onnx_model_path)
    return ClassicalDetector()


# ---------------------------------------------------------------------------
# Offline fine-tune workflow (documentation only; no runnable code here)
# ---------------------------------------------------------------------------
# The ONNXDetector slot above is designed to load a fine-tuned YOLOv8-nano
# model trained on labelled Time Crisis captures. The recommended offline
# workflow (deliberately OUT OF SCOPE for this module -- see
# /memories/session/plan.md, Phase 2 explicit exclusions) is:
#
#   1. Collect frames: `python run_eval.py --dump-frames <dir> <theta.npy>`.
#      Dumps one PNG per decision tick during evaluation.
#   2. Label a small set (~200-500 frames) with your labelling tool of
#      choice (e.g. Label Studio, LabelImg). Class IDs MUST match
#      EnemyClass above (0=enemy, 1=civilian, 2=projectile,
#      3=muzzle_flash).
#   3. Fine-tune yolov8n on your labels (offline, in a separate
#      environment): `yolo detect train data=timecrisis.yaml
#      model=yolov8n.pt imgsz=320 epochs=100`.
#   4. Export to ONNX: `yolo export model=runs/detect/train/weights/best.pt
#      format=onnx imgsz=320`. Set
#      `config.VISION_ONNX_MODEL_PATH = "<path to best.onnx>"`; next
#      `build_detector()` call picks it up automatically.
