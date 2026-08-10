"""Multi-class enemy detector for Time Crisis frames.

This module is Phase 2 of the vision-conditioned schedule plan
(``/memories/session/plan.md``). It replaces the naive single-color
centroid finder in ``vision.py`` with a real, proven-algorithm detector that
outputs bounding boxes + class labels per frame.

Two backends, one contract:

* ``ClassicalDetector`` -- day-1 default, no learned model required.

  Uses OpenCV's proven ``BackgroundSubtractorMOG2`` (Zivkovic 2004/2006)
  to isolate moving foreground, then **pure geometry** (blob area +
  aspect ratio) to split blobs into ENEMY vs. PROJECTILE.  No palette /
  color matching at all (2026-08-10 redesign -- the earlier color-AND
  approach failed in real BizHawk captures because enemy uniforms vary
  and share colors with background HUD elements).

  Key insight (user diagnosis, 2026-08-10):
  ``"enemies are the only things that are remotely humanoid; compared to
  the background they should be easy to take out -- what is moving and
  what is not"``
  MOG2 nails the motion side; humanoid-vs-projectile is purely a blob
  geometry decision:
    * Large, roughly upright blob  → ENEMY (sprite body, h > w typically)
    * Small blob                   → PROJECTILE (bullet/tracer)
    * Very small or very round     → noise, discarded
  CIVILIAN and MUZZLE_FLASH are NOT emitted by this backend -- they
  require either a trained model or reliable color cues we don't have.
  Downstream code (``act_vision_schedule`` scoring) just assigns them
  zero weight via the softmax when their class never appears, so this is
  a clean omission, not a silent bug.

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
# ClassicalDetector -- MOG2 + geometry (no color)
# ---------------------------------------------------------------------------

# Geometry thresholds for ``ClassicalDetector._classify_blob``.  All areas
# are in pixels; they assume a 256×224 (or 320×240) PS1 frame. If BizHawk
# is rendering at a higher internal resolution, multiply by the square of
# the scale factor.
#
# TUNING GUIDE (use ``python inspect_vision.py`` to iterate):
#   * If real enemies are missed → lower ENEMY_MIN_AREA.
#   * If the crosshair / HUD badges pollute results → raise ENEMY_MIN_AREA.
#   * If bullets are classified as ENEMY → lower PROJECTILE_MAX_AREA or
#     raise ENEMY_MIN_AREA so they don't overlap.
#   * If enemies are split into two blobs (top+bottom halves) → use a
#     larger morphological close kernel (see _CLOSE_KERNEL_SIZE below).

# Minimum blob pixel area to report any detection (below = discarded as noise).
_NOISE_MIN_AREA   = 10
# Blobs larger than this are enemies (full humanoid body).
_ENEMY_MIN_AREA   = 80
# Blobs below this and above _NOISE_MIN_AREA are projectiles (bullets/tracers).
_PROJECTILE_MAX_AREA = 79
# Confidence saturates (= 1.0) at this area for each class.
_ENEMY_SAT_AREA   = 1200
_PROJ_SAT_AREA    = 40

# Morphological kernels.
# OPEN (3×3): kill single-pixel MOG2 speckle.
# CLOSE (5×5): bridge small gaps inside a humanoid silhouette so a walking
#              enemy doesn't get split into a head-blob and a body-blob.
_OPEN_KERNEL_SIZE  = (3, 3)
_CLOSE_KERNEL_SIZE = (5, 5)

# HUD exclusion -- bottom band.
# Time Crisis renders ALL of its HUD at the BOTTOM of the frame:
#   * Ejected-shell animation (casing bounces after each shot)
#   * Countdown timer digits
#   * HP/life counter
# MOG2 correctly detects these as motion (the timer counts down, the casing
# animates), but we never want to aim there.  Zeroing out the bottom
# _HUD_BOTTOM_FRAC of the foreground mask before blob analysis removes all
# three sources of false positives deterministically without touching the
# mid-screen enemy region at all.
# Set to 0.0 to disable (e.g. if your savestate uses a different HUD layout).
_HUD_BOTTOM_FRAC = 0.15

# Text-overlay aspect-ratio filter.
# Time Crisis displays full-width text banners mid-screen in two situations:
#   "Hurry up!" -- flashes when the timer drops below ~10 s.
#   "DANGER!"   -- flashes when the player is forced to stay in cover.
# Both banners are MUCH wider than they are tall (aspect ratio w/h >> 1),
# which is the opposite of a humanoid sprite (h >= w typically).  Any blob
# whose bounding-box width-to-height ratio exceeds this threshold is assumed
# to be a text overlay and is discarded from detection entirely.
# Humanoid enemies rarely exceed w/h ~ 1.5 even when crouching, so 3.0
# gives comfortable headroom while still killing the full-width banners.
# Set to float('inf') to disable.
_MAX_BLOB_ASPECT = 3.0


class ClassicalDetector:
    """Motion-based detector: BackgroundSubtractorMOG2 + geometry classification.

    Runtime shape per ``detect(frame)`` call:

      1. MOG2 is fed the incoming frame and produces a foreground mask --
         pixels that differ meaningfully from the running background model.
         This separates static scenery from animating sprites reliably even
         when enemy uniforms vary.
      2. Morphological open (3×3) removes single-pixel speckle, then a
         morphological close (5×5) bridges gaps inside humanoid bodies.
      3. ``cv2.connectedComponentsWithStats`` extracts all moving blobs.
      4. Blobs are classified purely by area:
           area ≥ ENEMY_MIN_AREA               → ENEMY
           NOISE_MIN_AREA ≤ area < ENEMY_MIN_AREA → PROJECTILE
           area < NOISE_MIN_AREA              → discarded
         No color test is performed.  CIVILIAN and MUZZLE_FLASH are not
         emitted (need a trained model for reliable inference).

    Determinism: MOG2 mutates its internal Gaussian mixture across
    calls, so ``ClassicalDetector`` instances are stateful. Each
    ``TimeCrisisEnv`` owns exactly one instance and calls ``reset()`` on
    it whenever the episode's savestate is reloaded.
    """

    def __init__(
        self,
        mog_history: int = 200,
        mog_var_threshold: float = 16.0,
    ):
        import cv2  # lazy import

        self._cv2 = cv2
        self._mog_history = mog_history
        self._mog_var_threshold = mog_var_threshold
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=mog_history,
            varThreshold=mog_var_threshold,
            detectShadows=False,
        )
        self._open_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, _OPEN_KERNEL_SIZE)
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, _CLOSE_KERNEL_SIZE)

    def reset(self) -> None:
        """Discard the accumulated background model.

        Called by ``TimeCrisisEnv.reset()`` right after ``load_state`` so
        the previous episode's background doesn't leak into this one's
        foreground mask.
        """
        cv2 = self._cv2
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=self._mog_history,
            varThreshold=self._mog_var_threshold,
            detectShadows=False,
        )

    @staticmethod
    def _classify_blob(area: int, bw: int, bh: int) -> tuple[int, float] | None:
        """Return (class_id, confidence) or None if the blob should be discarded.

        Two-stage filter:
          1. Aspect-ratio check: blobs that are much wider than they are tall
             are text banners ("Hurry up!", "DANGER!"), not sprites.
          2. Area-based class assignment: large = ENEMY, small = PROJECTILE.
        """
        # Discard wide text banners -- they are never enemies.
        if bh > 0 and (bw / bh) > _MAX_BLOB_ASPECT:
            return None
        if area >= _ENEMY_MIN_AREA:
            conf = float(min(1.0, area / _ENEMY_SAT_AREA))
            return int(EnemyClass.ENEMY), conf
        if area >= _NOISE_MIN_AREA:
            conf = float(min(1.0, area / _PROJ_SAT_AREA))
            return int(EnemyClass.PROJECTILE), conf
        return None  # discard noise

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detected objects for one HxWx3 uint8 RGB frame.

        ``frame`` must be in the RGB layout that ``bridge_client.get_screenshot``
        produces (row 0 = top of image, channel order R,G,B).
        """
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(
                f"Expected HxWx3(+) uint8 RGB frame, got shape {frame.shape}"
            )
        cv2 = self._cv2
        h, w = frame.shape[:2]
        # MOG2 uses luma internally (accepts BGR or gray). Convert RGB→BGR once.
        bgr = frame[:, :, [2, 1, 0]].astype(np.uint8, copy=False)
        fg_mask = self._mog.apply(bgr)

        # Clean the mask: open kills speckle, close bridges body gaps.
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self._open_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._close_kernel)

        # Blank the bottom HUD band so the timer, HP bar, and ejected-shell
        # animation never produce false detections.  Done AFTER morphology so
        # close operations near the boundary don't bleed upward into play area.
        if _HUD_BOTTOM_FRAC > 0.0:
            hud_row = max(0, h - int(h * _HUD_BOTTOM_FRAC))
            fg_mask[hud_row:, :] = 0

        if not fg_mask.any():
            return []

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            fg_mask, connectivity=8,
        )
        detections: list[Detection] = []
        # Label 0 is the background component -- skip it.
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bx   = int(stats[i, cv2.CC_STAT_LEFT])
            by   = int(stats[i, cv2.CC_STAT_TOP])
            bw   = int(stats[i, cv2.CC_STAT_WIDTH])
            bh   = int(stats[i, cv2.CC_STAT_HEIGHT])
            result = ClassicalDetector._classify_blob(area, bw, bh)
            if result is None:
                continue
            class_id, confidence = result
            cx = float(centroids[i, 0])
            cy = float(centroids[i, 1])
            detections.append(
                Detection(
                    x=bx,
                    y=by,
                    w=bw,
                    h=bh,
                    class_id=class_id,
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
