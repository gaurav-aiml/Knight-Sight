import unittest

from anpr.app import _result_rows


class StreamlitResultTests(unittest.TestCase):
    def test_image_results_are_rendered_as_detection_rows(self):
        results = [{
            "plate_text": "MH-12-AB-1234",
            "ocr_confidence": 0.9,
            "plate_confidence": 0.95,
            "vehicle_box": [10, 10, 90, 90],
            "plate_box": [30, 60, 70, 75],
        }]

        self.assertEqual(
            _result_rows(results),
            [{
                "plate_text": "MH-12-AB-1234",
                "ocr_confidence": 0.9,
                "plate_confidence": 0.95,
                "vehicle_box": [10, 10, 90, 90],
                "plate_box": [30, 60, 70, 75],
            }],
        )

if __name__ == "__main__":
    unittest.main()
