import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.train_detector import split_labeled_images, validate_label_file


class TrainingTests(unittest.TestCase):
    def test_dataset_config_points_to_reorganized_dataset(self):
        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "configs" / "plate_detection.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset_root = (config_path.parent / config["path"]).resolve()

        self.assertEqual(dataset_root, project_root / "data" / "plate_detection")
        self.assertTrue((dataset_root / "images").is_dir())
        self.assertTrue((dataset_root / "labels").is_dir())
        self.assertTrue((dataset_root / "test_images").is_dir())

    def test_split_is_reproducible_disjoint_and_excludes_unlabeled_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            for index in range(10):
                (images_dir / f"{index}.jpg").touch()
                (labels_dir / f"{index}.txt").write_text(
                    "0 0.5 0.5 0.25 0.25\n",
                    encoding="utf-8",
                )
            (images_dir / "unlabeled.jpg").touch()

            train, validation, unlabeled_count = split_labeled_images(
                images_dir,
                labels_dir,
                val_fraction=0.2,
                seed=7,
            )
            repeated_train, repeated_validation, _ = split_labeled_images(
                images_dir,
                labels_dir,
                val_fraction=0.2,
                seed=7,
            )

            self.assertEqual(len(train), 8)
            self.assertEqual(len(validation), 2)
            self.assertEqual(unlabeled_count, 1)
            self.assertFalse(set(train) & set(validation))
            self.assertEqual(train, repeated_train)
            self.assertEqual(validation, repeated_validation)
            self.assertNotIn(images_dir / "unlabeled.jpg", train + validation)

            smoke_train, smoke_validation, _ = split_labeled_images(
                images_dir,
                labels_dir,
                val_fraction=0.2,
                seed=7,
                max_train_images=3,
                max_val_images=1,
            )
            self.assertEqual(len(smoke_train), 3)
            self.assertEqual(len(smoke_validation), 1)
            self.assertFalse(set(smoke_train) & set(smoke_validation))

    def test_invalid_normalized_box_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "bad.txt"
            label_path.write_text("0 0.5 0.5 1.2 0.25\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "box sizes"):
                validate_label_file(label_path, num_classes=1)

    def test_unknown_class_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "bad.txt"
            label_path.write_text("1 0.5 0.5 0.25 0.25\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "class ID"):
                validate_label_file(label_path, num_classes=1)


if __name__ == "__main__":
    unittest.main()
