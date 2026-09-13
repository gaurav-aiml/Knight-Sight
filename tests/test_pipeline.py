import unittest
import sys
import types
from unittest.mock import patch

import cv2
import numpy as np

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)
easyocr_stub = types.ModuleType("easyocr")
easyocr_stub.Reader = object
sys.modules.setdefault("easyocr", easyocr_stub)

from pipeline import VehicleIntelligencePipeline


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, track=False):
        return self.detections


class FakeOCR:
    def extract_text(self, image):
        return "MH-12-AB-1234", 0.9, True


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = VehicleIntelligencePipeline.__new__(VehicleIntelligencePipeline)
        self.pipeline.vehicle_detector = FakeDetector([
            {"box": [10, 10, 90, 90], "confidence": 0.8, "track_id": 4}
        ])
        self.pipeline.plate_detector = FakeDetector([
            {"box": [30, 60, 70, 75], "confidence": 0.95, "track_id": 8}
        ])
        self.pipeline.anpr_engine = FakeOCR()

    def test_process_image_returns_plate_and_vehicle_results(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        results, vehicles, plates, crops = self.pipeline.process_image(image_array=image)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["vehicle_track_id"], 4)
        self.assertEqual(results[0]["plate_box"], [30, 60, 70, 75])
        self.assertEqual(len(vehicles), 1)
        self.assertEqual(len(plates), 1)
        self.assertEqual(len(crops), 1)

    def test_annotate_image_draws_plate_box(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [{"plate_box": [30, 60, 70, 75], "plate_text": "TEST"}]

        annotated = self.pipeline.annotate_image(image, results, [])

        self.assertTrue(np.any(annotated[60, 30] != 0))

    def test_video_path_uses_four_value_image_contract(self):
        frames = [np.zeros((32, 32, 3), dtype=np.uint8)]

        class FakeCapture:
            def __init__(self):
                self.index = 0

            def isOpened(self):
                return True

            def read(self):
                if self.index == 0:
                    self.index += 1
                    return True, frames[0]
                return False, None

            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_WIDTH: 32,
                        cv2.CAP_PROP_FRAME_HEIGHT: 32, cv2.CAP_PROP_FRAME_COUNT: 1}.get(prop, 0)

            def release(self):
                pass

        class FakeWriter:
            def isOpened(self):
                return True

            def write(self, frame):
                pass

            def release(self):
                pass

        with patch("pipeline.cv2.VideoCapture", return_value=FakeCapture()), \
             patch("pipeline.cv2.VideoWriter", return_value=FakeWriter()):
            output = self.pipeline.process_video("input.mp4", "output.mp4")

        self.assertEqual(len(output), 0)


if __name__ == "__main__":
    unittest.main()