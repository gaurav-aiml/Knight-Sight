import cv2
from anpr.models.vehicle_detector import VehicleDetector
from anpr.models.plate_detector import PlateDetector
from anpr.models.anpr_engine import ANPREngine

class VehicleIntelligencePipeline:
    def __init__(self, vehicle_model_path='yolov8n.pt', plate_model_path=None):
        self.vehicle_detector = VehicleDetector(vehicle_model_path)
        self.plate_detector = PlateDetector(plate_model_path)
        self.anpr_engine = None

    def preprocess_image_clahe(self, image):
        """Apply CLAHE for low-light enhancement without amplifying noise."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def mitigate_glare(self, plate_crop):
        """Reduce glare using adaptive thresholding and morphological operations."""
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def process_image(self, image_path=None, image_array=None):
        """
        Process an image end-to-end.
        """
        if image_array is not None:
            image = image_array
        elif image_path:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image.")
        else:
            raise ValueError("Must provide image_path or image_array")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR image with three color channels.")
        if image.size == 0:
            raise ValueError("Image is empty.")

        # Low light enhancement
        enhanced_image = self.preprocess_image_clahe(image)

        output_data = []
        plate_crops = []

        # 1. Detect vehicles
        vehicles = self.vehicle_detector.detect(enhanced_image)

        # 2. Detect plates
        plates = self.plate_detector.detect(enhanced_image)
        
        for plate in plates:
            px1, py1, px2, py2 = plate['box']
            plate_conf = plate['confidence']
            
            # Ensure valid bounds
            py1, py2 = max(0, int(py1)), min(image.shape[0], int(py2))
            px1, px2 = max(0, int(px1)), min(image.shape[1], int(px2))
            
            plate_crop = image[py1:py2, px1:px2]
            
            # Avoid processing empty or extremely small crops
            if plate_crop.size == 0 or plate_crop.shape[0] < 5 or plate_crop.shape[1] < 5:
                continue
                
            # Apply glare mitigation
            processed_crop = self.mitigate_glare(plate_crop)
            
            # 3. Read Text
            if self.anpr_engine is None:
                self.anpr_engine = ANPREngine()
            text, ocr_conf, is_indian = self.anpr_engine.extract_text(processed_crop)
            
            # Associate the plate with a vehicle containing its center.
            associated_vehicle_box = None
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v['box']
                # If plate center is inside vehicle box
                pc_x = (px1 + px2) / 2
                pc_y = (py1 + py2) / 2
                if vx1 <= pc_x <= vx2 and vy1 <= pc_y <= vy2:
                    associated_vehicle_box = v['box']
                    break
            
            output_data.append({
                "vehicle_box": associated_vehicle_box,
                "plate_box": [px1, py1, px2, py2],
                "plate_confidence": float(plate_conf),
                "plate_text": text,
                "is_indian_plate": is_indian,
                "ocr_confidence": float(ocr_conf)
            })

            crop_annot = plate_crop.copy()
            h, w = crop_annot.shape[:2]
            border_thickness = max(2, int(round(min(w, h) * 0.03)))
            cv2.rectangle(crop_annot, (0, 0), (w - 1, h - 1), (0, 255, 0), border_thickness)
            text_label = f"{text} ({ocr_conf:.2f})"
            cv2.putText(crop_annot, text_label, (5, max(15, border_thickness + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            plate_crops.append({
                "crop": plate_crop,
                "annotated_crop": crop_annot,
                "plate_text": text,
                "ocr_confidence": float(ocr_conf),
                "plate_confidence": float(plate_conf),
                "plate_box": [px1, py1, px2, py2]
            })

        return output_data, vehicles, plates, plate_crops

    def annotate_image(self, image, results, vehicles):
        """
        Draw bounding boxes and text on the image for visualization.
        """
        annotated = image.copy()
        
        # Draw vehicle boxes
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v['box']
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
            label = f"Veh {v['confidence']:.2f}"
            cv2.putText(annotated, label, (vx1, max(10, vy1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for result in results:
            px1, py1, px2, py2 = result['plate_box']
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
            text = result.get('plate_text') or 'plate'
            cv2.putText(annotated, text, (px1, max(10, py1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                        
        return annotated
