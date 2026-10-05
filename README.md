# Knight-Sight ANPR

Image-based Automatic Number Plate Recognition (ANPR) for vehicle detection, license plate localization, and OCR. The project includes a Streamlit dashboard and a YOLOv8 training utility.

## Features

- Detect cars, motorcycles, buses, and trucks with a YOLO model.
- Detect license plates with custom-trained YOLOv8 weights.
- Read plate text with EasyOCR.
- Apply CLAHE low-light enhancement and glare mitigation.
- Load ONNX detector weights when available.
- Upload JPG, JPEG, and PNG images through the Streamlit dashboard.

## Project layout

```text
anpr/
  app.py                    Streamlit dashboard
  pipeline.py               Image-processing and detection pipeline
  models/                   Vehicle, plate, and OCR components
scripts/
  train_detector.py         YOLOv8 fine-tuning command
configs/
  plate_detection.yaml      Dataset configuration
data/
  plate_detection/
    images/                 Training image pool
    labels/                 YOLO labels and source annotations
    test_images/            Separate images not used for training
artifacts/
  ocr/                      Downloaded EasyOCR models
  training/                 Training runs and generated checkpoints
weights/                    Optional custom plate detector weights
tests/                      Unit and regression tests
```

## Requirements

- Python 3.10 or newer.
- Internet access the first time pretrained YOLO or EasyOCR models need to be downloaded.
- A CUDA-capable GPU is recommended for full training. CPU inference and small smoke tests are supported.

## Install

Run these commands from the repository root.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell does not allow activation, invoke the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`requirements.txt` is also provided for pip-based installs:

```bash
python -m pip install -r requirements.txt
```

## Run the project

Start the image-upload dashboard from the repository root:

```bash
python -m streamlit run anpr/app.py
```

Open <http://localhost:8501> and upload a JPG, JPEG, or PNG image. The dashboard displays the annotated image, detected plate crops, OCR results, and confidence values.

Streamlit usage statistics are disabled by `.streamlit/config.toml`.

### Models required for inference

The vehicle detector uses the pretrained COCO `yolov8n.pt` model. If that file is not in the repository root, Ultralytics attempts to download it on first use; an internet connection is needed unless the weights are already in its local cache.

Plate localization and OCR require a custom plate-detector checkpoint. The current checkout does not include either YOLO weight file, so train a checkpoint using the instructions below or provide one of your own. You can place a checkpoint at either of these locations:

```text
weights/plate_detector.pt
weights/plate_detector.onnx
```

The detector also searches for training checkpoints in this order:

```text
artifacts/training/yolov8_train/weights/best.pt
artifacts/training/license_plate_detector/weights/best.pt
```

For a `.pt` checkpoint, a sibling `.onnx` file is preferred when available. If no plate checkpoint is found, the dashboard displays a warning: vehicle detection can still run, but plate localization and OCR are unavailable. EasyOCR recognition models are downloaded only when a plate crop needs OCR and are stored under `artifacts/ocr/`.

## Train the plate detector

The dataset is configured in `configs/plate_detection.yaml`. It currently contains 6,326 training-pool images and 6,177 matching YOLO `.txt` labels. The training script validates label contents, excludes images without matching `.txt` labels, and creates a reproducible, non-overlapping train/validation split from the annotated images. The 149 images in `data/plate_detection/test_images/` are not used or modified by the trainer.

Run from the repository root after installation. Fine-tune pretrained YOLOv8n weights for 25 epochs:

```bash
python scripts/train_detector.py
```

The defaults are equivalent to:

```bash
python scripts/train_detector.py --data configs/plate_detection.yaml --model yolov8n.pt --epochs 25 --batch 16 --imgsz 640 --val-fraction 0.2 --seed 42 --project artifacts/training --name yolov8_train
```

Training writes the best and last checkpoints, metrics, and plots to `artifacts/training/yolov8_train/`. The default `yolov8n.pt` weights are not included in this checkout; Ultralytics will try to download them if they are not already available locally. A CUDA GPU is recommended for a full run; training on CPU can take a long time.

To run a small one-epoch smoke test on CPU (128 training images and 32 validation images):

```bash
python scripts/train_detector.py --epochs 1 --batch 2 --max-train-images 128 --max-val-images 32 --name yolov8_smoke
```

The `yolov8_smoke` checkpoint is saved separately and is not one of the detector's default checkpoint locations. Use a production run name such as `yolov8_train` for a checkpoint that the app will discover automatically.

Common options:

```text
--data configs/plate_detection.yaml   Dataset configuration
--model yolov8n.pt                    Pretrained weights or model file
--epochs 25                           Number of training epochs
--imgsz 640                           Input image size
--batch 16                            Batch size
--val-fraction 0.2                    Fraction held out for validation
--seed 42                             Seed for the reproducible split
--name yolov8_train                   Name of the output run
--project artifacts/training          Parent directory for training output
```

To intentionally initialize from a YOLO architecture YAML instead of pretrained weights, use `--from-scratch`:

```bash
python scripts/train_detector.py --from-scratch --model yolov8n.yaml --epochs 25
```

After the default training command finishes, the best checkpoint is saved at `artifacts/training/yolov8_train/weights/best.pt`, which the app searches for automatically.

## Use the pipeline from Python

```python
import cv2
from anpr import VehicleIntelligencePipeline

pipeline = VehicleIntelligencePipeline()
image = cv2.imread("path/to/image.jpg")

if image is None:
    raise ValueError("Could not read input image")

results, vehicles, plates, plate_crops = pipeline.process_image(image_array=image)
annotated = pipeline.annotate_image(image, results, vehicles)
cv2.imwrite("output.jpg", annotated)

for detection in results:
    print(detection["plate_text"], detection["ocr_confidence"])
```

## Run tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```
