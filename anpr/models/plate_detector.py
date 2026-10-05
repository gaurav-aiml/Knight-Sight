from pathlib import Path
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path=None):
        """
        Initialize the plate detector with the custom trained YOLO model or optimized ONNX model.
        Falls back to base YOLOv8 model if the custom weights are not found.
        """
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        candidates = [Path(model_path)] if model_path else []
        candidates.extend([
            PROJECT_ROOT / 'weights/plate_detector.pt',
            PROJECT_ROOT / 'artifacts/training/yolov8_train/weights/best.pt',
            PROJECT_ROOT / 'artifacts/training/license_plate_detector/weights/best.pt',
        ])
        candidates = [path if path.is_absolute() else PROJECT_ROOT / path for path in candidates]
        selected = None
        for candidate in candidates:
            if candidate.suffix.lower() == '.onnx':
                if candidate.is_file():
                    selected = candidate
                    break
                continue

            onnx_path = candidate.with_suffix('.onnx')
            if onnx_path.is_file():
                selected = onnx_path
                break
            if candidate.is_file():
                selected = candidate
                break

        if selected is None:
            self.model = None
            self.is_custom = False
            print('Warning: no trained plate model found; plate detection is disabled.')
            return

        if selected.suffix.lower() == '.onnx':
            print(f"Loading optimized ONNX plate detector: {selected}")
            self.model = YOLO(str(selected), task='detect')
        else:
            self.model = YOLO(str(selected))
        self.is_custom = True
            
    def detect(self, image):
        """
        Detect license plates in an image.
        Returns a list of dictionaries with bounding box and confidence.
        """
        if self.model is None:
            return []

        results = self.model(image, verbose=False)
            
        plates = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                
                if self.is_custom:
                    # Custom model: only allow class 0 (license plate)
                    if cls_id != 0:
                        continue
                else:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                plates.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                })
                
        return plates
