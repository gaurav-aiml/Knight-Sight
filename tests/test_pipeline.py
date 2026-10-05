import unittest
import sys
import types
from unittest.mock import patch

import numpy as np

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)
easyocr_stub = types.ModuleType("easyocr")
easyocr_stub.Reader = object
sys.modules.setdefault("easyocr", easyocr_stub)

from anpr.pipeline import VehicleIntelligencePipeline


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, _image):
        return self.detections


class FakeOCR:
    def extract_text(self, _image):
        return "MH-12-AB-1234", 0.9, True


class PipelineTests(unittest.TestCase):
    def test_pipeline_construction_does_not_initialize_ocr(self):
        with (
            patch("anpr.pipeline.VehicleDetector"),
            patch("anpr.pipeline.PlateDetector"),
            patch("anpr.pipeline.ANPREngine") as ocr_engine,
        ):
            VehicleIntelligencePipeline()

        ocr_engine.assert_not_called()

    def setUp(self):
        self.pipeline = VehicleIntelligencePipeline.__new__(VehicleIntelligencePipeline)
        self.pipeline.vehicle_detector = FakeDetector([
            {"box": [10, 10, 90, 90], "confidence": 0.8}
        ])
        self.pipeline.plate_detector = FakeDetector([
            {"box": [30, 60, 70, 75], "confidence": 0.95}
        ])
        self.pipeline.anpr_engine = FakeOCR()

    def test_ocr_engine_is_not_created_when_no_plate_is_detected(self):
        self.pipeline.plate_detector = FakeDetector([])
        self.pipeline.anpr_engine = None

        with patch("anpr.pipeline.ANPREngine") as ocr_engine:
            results, _, _, crops = self.pipeline.process_image(
                image_array=np.zeros((100, 100, 3), dtype=np.uint8)
            )

        self.assertEqual(results, [])
        self.assertEqual(crops, [])
        ocr_engine.assert_not_called()

    def test_ocr_engine_is_created_when_a_valid_plate_crop_is_found(self):
        self.pipeline.anpr_engine = None

        with patch("anpr.pipeline.ANPREngine", return_value=FakeOCR()) as ocr_engine:
            results, _, _, crops = self.pipeline.process_image(
                image_array=np.zeros((100, 100, 3), dtype=np.uint8)
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(crops), 1)
        ocr_engine.assert_called_once_with()

    def test_process_image_returns_plate_and_vehicle_results(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        results, vehicles, plates, crops = self.pipeline.process_image(image_array=image)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["vehicle_box"], [10, 10, 90, 90])
        self.assertEqual(results[0]["plate_box"], [30, 60, 70, 75])
        self.assertEqual(len(vehicles), 1)
        self.assertEqual(len(plates), 1)
        self.assertEqual(len(crops), 1)

    def test_annotate_image_draws_plate_box(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [{"plate_box": [30, 60, 70, 75], "plate_text": "TEST"}]

        annotated = self.pipeline.annotate_image(image, results, [])

        self.assertTrue(np.any(annotated[60, 30] != 0))

if __name__ == "__main__":
    unittest.main()