"""Regression tests for detector.py's Detection contract, ClassicalDetector
warm-up + motion-only blob analysis, and the build_detector() factory
fallback when no ONNX model is on disk.

2026-08-10 redesign: ClassicalDetector no longer uses color palettes.
All detection is purely MOG2 motion + geometry (area thresholds).
"""

import os
import tempfile
import unittest

import numpy as np

from detector import (
    ClassicalDetector,
    Detection,
    EnemyClass,
    NUM_CLASSES,
    _ENEMY_MIN_AREA,
    _NOISE_MIN_AREA,
    _PROJECTILE_MAX_AREA,
    _HUD_BOTTOM_FRAC,
    _MAX_BLOB_ASPECT,
    _PRE_BLUR_KERNEL,
    _UPSCALE_FACTOR,
    build_detector,
)


def _warmup(det: ClassicalDetector, frame: np.ndarray, n: int = 20) -> None:
    """Feed ``n`` copies of ``frame`` into MOG2 so it treats the content as
    established background before real detection tests run.

    Necessary because the very first MOG2 output is nearly all-foreground
    (the model has no history yet), which would swamp the blob analysis with
    false positives on frame 1.
    """
    for _ in range(n):
        det.detect(frame)


class DetectionContractSuite(unittest.TestCase):
    """Lock in Detection dataclass invariants that downstream policy code
    (act_vision_schedule) relies on."""

    def test_class_ids_are_stable_ints_in_expected_order(self):
        self.assertEqual(int(EnemyClass.ENEMY), 0)
        self.assertEqual(int(EnemyClass.CIVILIAN), 1)
        self.assertEqual(int(EnemyClass.PROJECTILE), 2)
        self.assertEqual(int(EnemyClass.MUZZLE_FLASH), 3)
        self.assertEqual(NUM_CLASSES, 4)

    def test_detection_area_is_wxh(self):
        det = Detection(
            x=10, y=20, w=8, h=6,
            class_id=0, confidence=0.5,
            cx_norm=0.5, cy_norm=0.5,
        )
        self.assertEqual(det.area, 48)


class ClassicalDetectorSuite(unittest.TestCase):
    """Behavioural tests for the motion-only (no-color) ClassicalDetector.

    Synthetic frames: solid dark background + bright moving blob.  The
    blob's class is determined solely by its pixel area, which makes these
    tests resolution-independent and stable across color-scheme changes.
    """

    def _make_enemy_blob(self, h=96, w=128, bx=30, by=20, bw=12, bh=16):
        """Return a frame with a large-enough blob to classify as ENEMY."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Any color works -- detector is color-agnostic.
        frame[by:by + bh, bx:bx + bw] = (180, 40, 60)
        return frame

    def test_large_moving_blob_is_detected_as_enemy(self):
        det = ClassicalDetector()
        h, w = 96, 128
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _warmup(det, bg)

        frame = self._make_enemy_blob(h=h, w=w, bx=30, by=20, bw=12, bh=16)
        # bw*bh = 192 px  >>  _ENEMY_MIN_AREA (80 px)
        dets = det.detect(frame)
        enemy = [d for d in dets if d.class_id == int(EnemyClass.ENEMY)]
        self.assertGreater(len(enemy), 0, msg=f"no ENEMY detected; all: {dets}")
        d = enemy[0]
        # Bounding box should contain the planted blob.
        self.assertLessEqual(d.x, 30)
        self.assertLessEqual(d.y, 20)
        self.assertGreaterEqual(d.x + d.w, 30 + 12)
        self.assertGreaterEqual(d.y + d.h, 20 + 16)

    def test_small_moving_blob_is_detected_as_projectile(self):
        det = ClassicalDetector()
        h, w = 96, 128
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _warmup(det, bg)

        frame = bg.copy()
        # 4x4 = 16 px, inside [_NOISE_MIN_AREA, _PROJECTILE_MAX_AREA].
        frame[50:54, 60:64] = (255, 255, 100)
        dets = det.detect(frame)
        proj = [d for d in dets if d.class_id == int(EnemyClass.PROJECTILE)]
        self.assertGreater(len(proj), 0, msg=f"no PROJECTILE detected; all: {dets}")

    def test_static_scenery_is_ignored_after_warmup(self):
        """MOG2 must classify persistently-static content as background so
        decorative objects in the game scene don't register as enemies."""
        det = ClassicalDetector()
        h, w = 96, 128
        static = np.zeros((h, w, 3), dtype=np.uint8)
        # Large bright block present in every warm-up frame.
        static[10:26, 10:30] = (200, 150, 50)
        _warmup(det, static, n=40)

        dets = det.detect(static)
        self.assertEqual(
            dets, [],
            msg=(
                "Fully static foreground must NOT produce detections after "
                f"MOG2 warm-up; got: {dets}"
            ),
        )

    def test_noise_below_enemy_area_is_not_classified_as_enemy(self):
        """With 2× upscale + Gaussian blur, a tiny blob (3×3 original pixels)
        can spread into the PROJECTILE range -- that's expected and fine.
        What must never happen is a sub-body blob being aim-targeted as an
        ENEMY (which vision_schedule steers toward).  Only ENEMY detections
        drive the aim blend; a stray PROJECTILE in an otherwise clear frame
        will never win the argmax against a real enemy."""
        det = ClassicalDetector()
        h, w = 96, 128
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _warmup(det, bg)

        frame = bg.copy()
        # 3×3 = 9 px original -- too small for a humanoid body.
        frame[10:13, 10:13] = (200, 200, 200)
        dets = det.detect(frame)
        enemy = [d for d in dets if d.class_id == int(EnemyClass.ENEMY)]
        self.assertEqual(
            enemy, [],
            msg=f"tiny blob must not classify as ENEMY; got: {dets}",
        )

    def test_enemy_class_is_color_agnostic(self):
        """The detection must fire for a blob of ANY color -- not just the
        old red/blue/green palette entries."""
        det = ClassicalDetector()
        h, w = 96, 128
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _warmup(det, bg)

        for color in [(80, 140, 200), (50, 200, 80), (240, 230, 10), (10, 10, 200)]:
            det2 = ClassicalDetector()
            _warmup(det2, bg)
            frame = bg.copy()
            frame[20:36, 30:46] = color  # 16x16 = 256 px  >>  ENEMY_MIN_AREA
            dets = det2.detect(frame)
            enemy = [d for d in dets if d.class_id == int(EnemyClass.ENEMY)]
            self.assertGreater(
                len(enemy), 0,
                msg=f"color {color} should detect as ENEMY regardless of hue; got: {dets}",
            )

    def test_reset_discards_learned_background(self):
        """After reset(), a previously-learned-as-background blob must be
        detectable again (the background model was recreated)."""
        det = ClassicalDetector()
        h, w = 96, 128
        static = np.zeros((h, w, 3), dtype=np.uint8)
        static[20:36, 20:36] = (180, 100, 60)  # 16x16 -- enemy-sized
        _warmup(det, static, n=40)
        pre_reset = [d for d in det.detect(static) if d.class_id == int(EnemyClass.ENEMY)]
        self.assertEqual(pre_reset, [], msg="should be background after warmup")

        det.reset()
        post_reset = [d for d in det.detect(static) if d.class_id == int(EnemyClass.ENEMY)]
        self.assertGreater(
            len(post_reset), 0,
            msg="reset() must clear MOG2 so a previously-static blob detects again",
        )

    def test_geometry_thresholds_are_consistent(self):
        """Sanity check that the module-level constants form a coherent
        partition: NOISE < PROJECTILE <= ENEMY, and HUD/aspect params are valid."""
        self.assertLess(_NOISE_MIN_AREA, _ENEMY_MIN_AREA)
        self.assertEqual(_PROJECTILE_MAX_AREA, _ENEMY_MIN_AREA - 1)
        self.assertGreaterEqual(_HUD_BOTTOM_FRAC, 0.0)
        self.assertLess(_HUD_BOTTOM_FRAC, 0.5)  # sanity: never more than half
        self.assertGreater(_MAX_BLOB_ASPECT, 1.0)
        # Pre-processing constants must be valid.
        self.assertGreaterEqual(_UPSCALE_FACTOR, 1)
        # Blur kernel must be 0 (disabled), 1 (noop), or a positive odd int.
        self.assertTrue(
            _PRE_BLUR_KERNEL <= 1 or _PRE_BLUR_KERNEL % 2 == 1,
            msg=f"_PRE_BLUR_KERNEL must be <=1 or odd, got {_PRE_BLUR_KERNEL}",
        )

    def test_wide_text_banner_blob_is_rejected(self):
        """A blob with w/h > _MAX_BLOB_ASPECT must be discarded even if its
        area qualifies as ENEMY -- this is how 'Hurry up!' and 'DANGER!'
        overlays are filtered out."""
        det = ClassicalDetector()
        h, w = 96, 256
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _warmup(det, bg)

        frame = bg.copy()
        # Wide banner: 200 px wide × 12 px tall = aspect 16.7 >> _MAX_BLOB_ASPECT.
        # Area = 2400, well above _ENEMY_MIN_AREA -- area alone would classify it.
        bw_, bh_ = 200, 12
        frame[20:20 + bh_, 20:20 + bw_] = (200, 200, 60)
        dets = det.detect(frame)
        self.assertEqual(
            dets, [],
            msg=f"wide text-banner blob must be filtered by aspect ratio; got: {dets}",
        )

    def test_bottom_hud_band_is_excluded(self):
        """A moving blob entirely within the bottom _HUD_BOTTOM_FRAC of the
        frame must not produce any detections -- this covers the ejected-shell
        animation, timer digits, and HP counter."""
        det = ClassicalDetector()
        h, w = 224, 256
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _warmup(det, bg)

        frame = bg.copy()
        # Place a large enemy-sized blob just inside the HUD zone.
        hud_row = h - int(h * _HUD_BOTTOM_FRAC) + 2  # 2 rows inside the band
        bw_, bh_ = 12, 16  # 192 px, well above _ENEMY_MIN_AREA
        frame[hud_row:hud_row + bh_, 30:30 + bw_] = (180, 40, 60)
        dets = det.detect(frame)
        self.assertEqual(
            dets, [],
            msg=f"blob inside bottom HUD band must be suppressed; got: {dets}",
        )


class BuildDetectorFactorySuite(unittest.TestCase):
    def test_no_model_path_returns_classical_detector(self):
        det = build_detector(onnx_model_path=None)
        self.assertIsInstance(det, ClassicalDetector)

    def test_missing_model_file_falls_back_to_classical(self):
        det = build_detector(onnx_model_path="/definitely/does/not/exist.onnx")
        self.assertIsInstance(det, ClassicalDetector)

    def test_empty_file_at_model_path_attempts_onnx_and_would_raise(self):
        """Sanity check that build_detector actually TRIES the ONNX path
        when the file exists. We create an empty temp file; onnxruntime
        must raise on the invalid model. We deliberately do not stub
        onnxruntime here -- catching the raise is the strongest evidence
        that the factory took the ONNX branch."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        with self.assertRaises(Exception):
            build_detector(onnx_model_path=path)


if __name__ == "__main__":
    unittest.main()
