"""Regression tests for detector.py's Detection contract, ClassicalDetector
warm-up + palette+MOG2 blob analysis, and the build_detector() factory
fallback when no ONNX model is on disk."""

import os
import tempfile
import unittest

import numpy as np

from detector import (
    ClassPalette,
    ClassicalDetector,
    Detection,
    EnemyClass,
    NUM_CLASSES,
    build_detector,
)


def _warmup(det: ClassicalDetector, frame: np.ndarray, n: int = 20) -> None:
    """Feed ``n`` copies of ``frame`` into MOG2 so the model treats it as
    established background before real detection tests run.

    Necessary because the very first MOG2 output is nearly all-foreground
    (the background model has no history yet), which would swamp the
    per-class blob analysis with false positives on frame 1.
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
    """Behavioural tests for the classical MOG2 + palette detector.

    The scenarios are deliberately synthetic (solid backgrounds + solid
    color rectangles) so the tests exercise contract shape, not pixel-
    perfect blob accuracy on real Time Crisis frames -- that calibration
    step is explicitly Phase 2 hand-tuning against real captures.
    """

    def test_moving_red_blob_is_detected_as_enemy(self):
        det = ClassicalDetector()
        h, w = 64, 96
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _warmup(det, bg)

        frame = bg.copy()
        # 8x8 solid ENEMY-red block at pixel (30, 40).
        frame[40:48, 30:38] = (220, 30, 30)
        dets = det.detect(frame)
        # Should see exactly one detection: the red block. Class is ENEMY.
        self.assertEqual(len(dets), 1, msg=f"unexpected detections: {dets}")
        d = dets[0]
        self.assertEqual(d.class_id, int(EnemyClass.ENEMY))
        self.assertEqual((d.x, d.y, d.w, d.h), (30, 40, 8, 8))
        # Centroid roughly at the block's center in normalized space.
        # cv2.connectedComponentsWithStats returns pixel-INDEX centroids (a
        # block spanning columns 30..37 inclusive has cx=33.5, not 34.5), so
        # normalize by (start + (size-1)/2) not (start + size/2).
        self.assertAlmostEqual(d.cx_norm, (30 + (8 - 1) / 2) / w, places=2)
        self.assertAlmostEqual(d.cy_norm, (40 + (8 - 1) / 2) / h, places=2)
        # Area 64, saturation_area 400 -> confidence 0.16.
        self.assertGreater(d.confidence, 0.1)
        self.assertLess(d.confidence, 0.5)

    def test_static_scenery_does_not_produce_enemy_detections(self):
        """MOG2 must classify persistently-unchanging same-color scenery as
        background -- this is exactly the failure mode the memory's naive
        multi-color probe hit with a hand-rolled frame differencer."""
        det = ClassicalDetector()
        h, w = 64, 96
        # A red blob that has been present in EVERY frame including the
        # warm-up frames -- MOG2 must learn it as background.
        static = np.zeros((h, w, 3), dtype=np.uint8)
        static[10:14, 10:14] = (220, 30, 30)
        _warmup(det, static, n=40)

        dets = det.detect(static)
        enemy_dets = [d for d in dets if d.class_id == int(EnemyClass.ENEMY)]
        self.assertEqual(
            enemy_dets, [],
            msg=(
                "MOG2-conditioned enemy palette must not fire on a fully "
                "static same-color patch after warm-up; got "
                f"{enemy_dets}"
            ),
        )

    def test_small_speckle_below_min_blob_area_is_rejected(self):
        det = ClassicalDetector()
        h, w = 64, 96
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _warmup(det, bg)

        frame = bg.copy()
        # Just 2x2 -- well below DEFAULT_PALETTES ENEMY.min_blob_area (12),
        # AND it will additionally be eroded by the morphological open.
        frame[10:12, 10:12] = (220, 30, 30)
        dets = det.detect(frame)
        self.assertEqual(
            [d for d in dets if d.class_id == int(EnemyClass.ENEMY)],
            [],
        )

    def test_reset_discards_learned_background(self):
        """After reset(), a previously-learned-as-background blob must be
        detectable again (because the background model was recreated)."""
        det = ClassicalDetector()
        h, w = 64, 96
        static = np.zeros((h, w, 3), dtype=np.uint8)
        static[20:32, 20:32] = (220, 30, 30)
        _warmup(det, static, n=40)
        pre_reset = [d for d in det.detect(static) if d.class_id == 0]
        self.assertEqual(pre_reset, [])

        det.reset()
        # First-after-reset frame is essentially all-foreground for MOG2 --
        # the palette-matched blob will re-detect.
        post_reset = [d for d in det.detect(static) if d.class_id == 0]
        self.assertGreater(
            len(post_reset), 0,
            msg="reset() must clear the MOG2 background so a previously-"
                "learned blob detects again",
        )

    def test_custom_palette_maps_a_new_color_to_the_expected_class(self):
        # A palette with just one color under CIVILIAN so we can prove the
        # class-id routing works end-to-end, independent of the defaults.
        palettes = {
            EnemyClass.CIVILIAN: ClassPalette(
                colors=((10, 200, 250),),   # unmistakable cyan
                tolerance=5,
                min_blob_area=4,
                saturation_area=100,
            ),
        }
        det = ClassicalDetector(palettes=palettes)
        h, w = 64, 96
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _warmup(det, bg)

        frame = bg.copy()
        frame[10:20, 10:20] = (10, 200, 250)
        dets = det.detect(frame)
        civ = [d for d in dets if d.class_id == int(EnemyClass.CIVILIAN)]
        self.assertEqual(len(civ), 1)
        self.assertEqual(civ[0].class_id, int(EnemyClass.CIVILIAN))


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
