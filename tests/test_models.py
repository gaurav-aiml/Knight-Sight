import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anpr.models.anpr_engine import ANPREngine
from anpr.models.plate_detector import PlateDetector
from anpr.models.vehicle_detector import VehicleDetector


class PlateDetectorTests(unittest.TestCase):
    def test_vehicle_detector_passes_missing_pretrained_name_for_resolution(self):
        with patch("anpr.models.vehicle_detector.YOLO") as yolo:
            VehicleDetector()

        yolo.assert_called_once_with("yolov8n.pt")

    def test_easyocr_storage_paths_are_anchored_to_project_root(self):
        project_root = Path(__file__).resolve().parents[1]

        with patch("anpr.models.anpr_engine.easyocr.Reader") as reader:
            ANPREngine()

        reader.assert_called_once_with(
            ["en"],
            gpu=False,
            model_storage_directory=str(project_root / "artifacts" / "ocr" / "models"),
            user_network_directory=str(project_root / "artifacts" / "ocr" / "user_network"),
            download_enabled=True,
        )

    def test_onnx_checkpoint_loads_when_pt_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            onnx_path = Path(temporary_directory) / "plate_detector.onnx"
            onnx_path.touch()

            with patch("anpr.models.plate_detector.YOLO") as yolo:
                detector = PlateDetector(str(onnx_path.with_suffix(".pt")))

            yolo.assert_called_once_with(str(onnx_path), task="detect")
            self.assertIsNotNone(detector.model)

    def test_onnx_checkpoint_is_preferred_to_sibling_pt_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            pt_path = Path(temporary_directory) / "plate_detector.pt"
            onnx_path = pt_path.with_suffix(".onnx")
            pt_path.touch()
            onnx_path.touch()

            with patch("anpr.models.plate_detector.YOLO") as yolo:
                PlateDetector(str(pt_path))

            yolo.assert_called_once_with(str(onnx_path), task="detect")


if __name__ == "__main__":
    unittest.main()
