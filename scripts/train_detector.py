"""Fine-tune a YOLOv8 detector using a reproducible split of labeled images."""

import argparse
import math
import random
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def validate_label_file(label_path, num_classes):
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected class_id x_center y_center width height")

        try:
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: label values must be numeric") from error

        if not 0 <= class_id < num_classes:
            raise ValueError(f"{label_path}:{line_number}: class ID {class_id} is outside [0, {num_classes})")
        if not all(map(math.isfinite, (x_center, y_center, width, height))):
            raise ValueError(f"{label_path}:{line_number}: box coordinates must be finite")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{label_path}:{line_number}: centers must be in [0, 1] and box sizes in (0, 1]")


def split_labeled_images(
    images_dir,
    labels_dir,
    val_fraction=0.2,
    seed=42,
    num_classes=1,
    max_train_images=None,
    max_val_images=None,
):
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")

    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labeled_images = [image for image in images if (labels_dir / f"{image.stem}.txt").is_file()]
    unlabeled_count = len(images) - len(labeled_images)
    if not labeled_images:
        raise ValueError(f"No labeled images found in {images_dir}.")
    if len(labeled_images) < 2:
        raise ValueError("At least two labeled images are required for a train/validation split.")

    for image in labeled_images:
        validate_label_file(labels_dir / f"{image.stem}.txt", num_classes)

    random.Random(seed).shuffle(labeled_images)
    validation_count = min(len(labeled_images) - 1, max(1, round(len(labeled_images) * val_fraction)))
    train_images = labeled_images[validation_count:]
    validation_images = labeled_images[:validation_count]
    if max_train_images is not None:
        train_images = train_images[:max_train_images]
    if max_val_images is not None:
        validation_images = validation_images[:max_val_images]
    return train_images, validation_images, unlabeled_count


def write_training_data_config(data_config, dataset_root, train_images, validation_images, split_dir):
    import yaml

    train_list = split_dir / "train.txt"
    validation_list = split_dir / "val.txt"
    train_list.write_text(
        "\n".join(path.resolve().as_posix() for path in train_images) + "\n",
        encoding="utf-8",
    )
    validation_list.write_text(
        "\n".join(path.resolve().as_posix() for path in validation_images) + "\n",
        encoding="utf-8",
    )

    split_config = dict(data_config)
    split_config["path"] = dataset_root.resolve().as_posix()
    split_config["train"] = train_list.resolve().as_posix()
    split_config["val"] = validation_list.resolve().as_posix()
    data_path = split_dir / "data.yaml"
    data_path.write_text(yaml.safe_dump(split_config, sort_keys=False), encoding="utf-8")
    return data_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on a YOLO-format dataset.")
    parser.add_argument(
        "--data",
        default="configs/plate_detection.yaml",
        help="Dataset YAML, resolved relative to the project root",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="artifacts/training", help="Directory for training output")
    parser.add_argument("--name", default="yolov8_train", help="Training run name")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained weights or a YOLO architecture YAML")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize from an architecture YAML, not pretrained weights")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of labeled images held out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the reproducible train/validation split")
    parser.add_argument("--max-train-images", type=int, help="Limit training images, for example during a smoke test")
    parser.add_argument("--max-val-images", type=int, help="Limit validation images, for example during a smoke test")
    args = parser.parse_args(argv)

    if (
        args.epochs < 1
        or args.imgsz < 1
        or args.batch < 1
        or (args.max_train_images is not None and args.max_train_images < 1)
        or (args.max_val_images is not None and args.max_val_images < 1)
    ):
        parser.error("Epochs, image size, batch size, and image limits must be positive integers.")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.is_file():
        parser.error(f"Dataset YAML does not exist: {data_path}")

    import yaml

    with data_path.open(encoding="utf-8") as data_file:
        data_config = yaml.safe_load(data_file)
    if not isinstance(data_config, dict):
        parser.error(f"Dataset YAML must contain a mapping: {data_path}")

    dataset_root = Path(data_config.get("path", data_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_path.parent / dataset_root

    train_source = data_config.get("train", "images")
    if not isinstance(train_source, str):
        parser.error("The dataset 'train' entry must name an image directory.")
    images_dir = Path(train_source)
    if not images_dir.is_absolute():
        images_dir = dataset_root / images_dir
    if not images_dir.is_dir():
        parser.error(f"Training image directory does not exist: {images_dir}")

    labels_dir = dataset_root / "labels"
    if not labels_dir.is_dir():
        parser.error(f"YOLO label directory does not exist: {labels_dir}")

    names = data_config.get("names", {})
    num_classes = data_config.get("nc", len(names))
    if not isinstance(num_classes, int) or num_classes < 1:
        parser.error("Dataset YAML must specify a positive 'nc' or non-empty 'names'.")

    model_path = Path(args.model)
    if not model_path.is_absolute() and (PROJECT_ROOT / model_path).exists():
        model_path = PROJECT_ROOT / model_path
    if args.from_scratch and model_path.suffix.lower() != ".yaml":
        parser.error("--from-scratch requires --model to name a YOLO architecture YAML, not pretrained weights.")
    if not args.from_scratch and model_path.suffix.lower() == ".yaml":
        parser.error("An architecture YAML initializes a model from scratch; remove --model YAML or use --from-scratch.")

    train_images, validation_images, unlabeled_count = split_labeled_images(
        images_dir,
        labels_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_classes=num_classes,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
    )
    print(
        f"Dataset split: {len(train_images)} train, {len(validation_images)} validation"
        f" ({unlabeled_count} images without labels excluded)."
    )

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = PROJECT_ROOT / project_path

    print(f"Starting {'from-scratch training' if args.from_scratch else 'fine-tuning'} with {model_path}...")
    with tempfile.TemporaryDirectory(prefix="knight-sight-yolo-") as temporary_directory:
        split_dir = Path(temporary_directory)
        split_data_path = write_training_data_config(
            data_config,
            dataset_root,
            train_images,
            validation_images,
            split_dir,
        )
        model.train(
            data=str(split_data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(project_path),
            name=args.name,
            seed=args.seed,
            exist_ok=False,
        )

    print(f"Training complete. Weights saved under {project_path / args.name / 'weights'}.")


if __name__ == "__main__":
    main()
