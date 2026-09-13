from pathlib import Path
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path=None):
        """
        Initialize the plate detector with the custom trained YOLO model or optimized ONNX model.
        Falls back to base YOLOv8 model if the custom weights are not found.
        """
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        candidates = [Path(model_path)] if model_path else []
        candidates.extend([
            PROJECT_ROOT / 'models/plate_detector.pt',
            PROJECT_ROOT / 'runs/train/yolov8_train/weights/best.pt',
            PROJECT_ROOT / 'runs/license_plate_detector/weights/best.pt',
        ])
        candidates = [path if path.is_absolute() else PROJECT_ROOT / path for path in candidates]
        selected = next((path for path in candidates if path and path.exists()), None)
        if selected is None:
            self.model = None
            self.is_custom = False
            print('Warning: no trained plate model found; plate detection is disabled.')
            return

        onnx_path = selected.with_suffix('.onnx')
        if onnx_path.exists():
            print(f"Loading optimized ONNX plate detector: {onnx_path}")
            self.model = YOLO(str(onnx_path), task='detect')
        else:
            self.model = YOLO(str(selected))
        self.is_custom = True
            
    def detect(self, image, track=False):
        """
        Detect license plates in an image.
        If track=True, uses YOLO's native tracking to return object IDs.
        Returns a list of dictionaries with bounding box, confidence, and track_id.
        """
        if self.model is None:
            return []

        if track:
            results = self.model.track(image, persist=True, verbose=False)
        else:
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
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                
                plates.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'track_id': track_id
                })
                
        return plates
