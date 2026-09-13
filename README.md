# Knight-Sight — ANPR & Vehicle Intelligence Pipeline

A modular, edge-friendly Automatic Number Plate Recognition (ANPR) pipeline for research and small deployments. Combines vehicle detection, plate localization, and OCR-based text extraction, with a Streamlit demo and YOLOv8 training utilities.

## Features

- Vehicle detection with a base YOLO model, with optional tracking for video input
- License-plate detection with a YOLOv8 model, custom-trainable on your own dataset
- Text extraction via EasyOCR
- Low-light enhancement (CLAHE) and glare mitigation (adaptive thresholding) applied to plate crops before OCR
- ONNX fallback: both detectors will load an optimized `.onnx` model if present, falling back to `.pt` weights otherwise
- Streamlit demo for interactive upload-and-inspect inference
- YOLOv8 training script for the plate detector, with a ready-to-use dataset in YOLO format

## Tech Stack

- **Detection:** Ultralytics YOLOv8 (`ultralytics`, `torch`, `torchvision`)
- **OCR:** EasyOCR
- **Image processing:** OpenCV (`opencv-python-headless`)
- **UI:** Streamlit
- **Data handling:** `numpy`, `Pillow`, `pandas`

## Pipeline

```
Image / Frame
     │
     ▼
CLAHE low-light enhancement (pipeline.py)
     │
     ▼
Vehicle Detector   (models/vehicle_detector.py)   ← YOLO, optional tracking for video
     │
     ▼
Plate Detector     (models/plate_detector.py)     ← YOLOv8, custom-trained or fallback weights
     │
     ▼
Adaptive-threshold glare mitigation on plate crop
     │
     ▼
ANPR Engine        (models/anpr_engine.py)        ← EasyOCR text extraction
     │
     ▼
Annotated image + plate text + OCR confidence
```

Both detectors prefer an optimized `.onnx` model if one exists alongside the configured `.pt` weights, and fall back to the `.pt` file (or a base `yolov8n.pt`) otherwise.

## Installation

```bash
git clone https://github.com/chmodgaurav/Knight-Sight.git
cd Knight-Sight
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- A CUDA-capable GPU is recommended for training (CPU works for inference and quick tests)
- EasyOCR downloads its recognition models on first run — expect a delay and some disk usage the first time the ANPR engine initializes

## Usage

### Streamlit demo

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, upload an image, and view detected vehicles, plates, and OCR results.

### Programmatic use

```python
import cv2
from pipeline import VehicleIntelligencePipeline

pipeline = VehicleIntelligencePipeline()
image = cv2.imread("path/to/image.jpg")
     results, vehicles, plates, plate_crops = pipeline.process_image(image_array=image)

for r in results:
    print(r.get("plate_text"), r.get("ocr_confidence"))

annotated = pipeline.annotate_image(image, results, vehicles)
cv2.imwrite("output.jpg", annotated)
```

## Data & Training

Dataset lives under `Dataset/images` and `Dataset/labels` in YOLO format, referenced by `data.yaml`:

```yaml
path: ./Dataset
train: images
val: images
nc: 1
names:
  0: 'license_plate'
```

Train the plate detector:

```bash
# GPU
python train_yolov8.py --data data.yaml --epochs 50 --batch 8 --imgsz 640 --model yolov8n.pt

# CPU quick test
python train_yolov8.py --data data.yaml --epochs 1 --batch 2 --imgsz 640
```

Trained weights are saved under `runs/train/<run-name>/weights/`. The application searches for a plate checkpoint in `models/plate_detector.pt`, then `runs/train/yolov8_train/weights/best.pt`, and then `runs/license_plate_detector/weights/best.pt`. If none exists, plate detection is disabled rather than using a generic object detector as a false plate detector. `models/vehicle_detector.py` defaults to `yolov8n.pt` for the vehicle-detection stage.

## Project Structure

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Web demo for upload-and-inspect inference |
| `pipeline.py` | `VehicleIntelligencePipeline` — end-to-end orchestration, CLAHE and glare mitigation |
| `models/vehicle_detector.py` | Vehicle detection (YOLO, optional ONNX) |
| `models/plate_detector.py` | License-plate detection (YOLOv8, custom or fallback weights) |
| `models/anpr_engine.py` | EasyOCR-based text extraction |
| `train_yolov8.py` | YOLOv8 training script for the plate detector |
| `data.yaml` | Ultralytics dataset config (single class: `license_plate`) |
| `Dataset/` | Training images and YOLO-format labels |
| `yolov8n.pt` | Base YOLOv8 nano weights used as the default/fallback |
| `yolo26n.pt` | Additional pretrained weights included in the repo |

## Notes

- No Tesseract dependency — OCR is handled entirely by EasyOCR (`easyocr` in `requirements.txt`), which downloads its own recognition models on first use.
- A trained plate checkpoint is required for plate localization and OCR. The app reports when that checkpoint is missing; it never treats a generic vehicle detector as a plate detector.

## Future Improvements

- Bundle or document a pretrained plate-detector checkpoint so the pipeline doesn't silently fall back to the generic base model
- Add batch/video processing examples beyond the single-image API shown above
- Add automated tests for the detection and OCR stages

## Contributing

Issues and pull requests are welcome.

## License

Built with Ultralytics YOLOv8, Streamlit, and EasyOCR. See repository for license details.
