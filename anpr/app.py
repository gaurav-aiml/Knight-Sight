import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from anpr.pipeline import VehicleIntelligencePipeline


def _result_rows(results):
    return [
        {
            "plate_text": result.get("plate_text"),
            "ocr_confidence": result.get("ocr_confidence"),
            "plate_confidence": result.get("plate_confidence"),
            "vehicle_box": result.get("vehicle_box"),
            "plate_box": result.get("plate_box"),
        }
        for result in results
    ]


# Ensure models load only once
@st.cache_resource
def load_pipeline():
    return VehicleIntelligencePipeline()


def main():
    st.set_page_config(page_title="KnightSight ANPR Dashboard", layout="wide")
    st.title("🚗 YOLOv8 ANPR Dashboard")
    st.markdown("Upload an image to run vehicle detection, plate localization, and OCR.")

    st.sidebar.header("Pipeline Settings")
    st.sidebar.markdown(
        "This dashboard uses a lightweight YOLOv8-based detection pipeline with edge-friendly plate crop highlighting."
    )
    st.sidebar.info("Model: YOLOv8n / YOLOv8 configuration\nOCR: EasyOCR\nEnhancement: CLAHE + glare mitigation")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        return

    try:
        st.markdown("### Processing...")

        with st.spinner("Running inference pipeline..."):
            pipeline = load_pipeline()
            if pipeline.plate_detector.model is None:
                st.warning(
                    "No trained license-plate detector found. Vehicle detection will still run, "
                    "but plate localization and OCR are unavailable. Fine-tune the model with "
                    "`python scripts/train_detector.py` to enable them."
                )

            image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            results, vehicles, _, plate_crops = pipeline.process_image(image_array=img_bgr)
            annotated_img = pipeline.annotate_image(img_bgr, results, vehicles)
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            left, mid, right = st.columns([1, 1, 1])
            with left:
                st.image(image, caption="Original Image", width="stretch")
            with mid:
                st.image(annotated_rgb, caption="Vehicle and plate detections", width="stretch")
            with right:
                st.markdown("### Detected Plates")
                if not plate_crops:
                    st.info("No plates detected in this image.")
                else:
                    cols = st.columns(2)
                    for idx, plate_crop in enumerate(plate_crops):
                        crop_rgb = cv2.cvtColor(plate_crop["annotated_crop"], cv2.COLOR_BGR2RGB)
                        cols[idx % len(cols)].image(
                            crop_rgb,
                            caption=(
                                f"{plate_crop['plate_text']} | "
                                f"OCR {plate_crop['ocr_confidence']:.2f}"
                            ),
                            width="stretch",
                        )

                    st.markdown("#### Plate Crop Details")
                    crop_rows = [
                        {
                            "Plate Text": plate_crop["plate_text"],
                            "OCR Confidence": f"{plate_crop['ocr_confidence']:.2f}",
                            "Plate Confidence": f"{plate_crop['plate_confidence']:.2f}",
                            "Plate Box": plate_crop["plate_box"],
                        }
                        for plate_crop in plate_crops
                    ]
                    st.table(crop_rows)

        st.markdown("### Results Summary")
        rows = _result_rows(results)
        if rows:
            st.table(rows)
        else:
            st.info("No detections to show.")

    except Exception as error:
        st.error(f"Error processing file: {error}")
        st.exception(error)


if __name__ == "__main__":
    main()
