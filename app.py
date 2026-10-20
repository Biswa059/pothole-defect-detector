
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Pothole & Defect Detector",
    page_icon="🚧",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🚧 Pothole & Defect Detector")
st.write("Upload a road image to detect potholes and estimate severity.")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    with st.spinner("Detecting potholes..."):
        results = model.predict(
            source=image_array,
            conf=0.25,
            verbose=False
        )

    result = results[0]
    height, width = image_array.shape[:2]

    total_box_area = 0

    for box in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        total_box_area += (
            max(0, x2 - x1) * max(0, y2 - y1)
        )

    coverage = (total_box_area / (width * height)) * 100

    if len(result.boxes) == 0:
        severity = "NONE"
    elif coverage <= 3:
        severity = "LOW"
    elif coverage <= 10:
        severity = "MEDIUM"
    elif coverage <= 20:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    annotated_image = result.plot()[:, :, ::-1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_array, use_container_width=True)

    with col2:
        st.subheader("Detection Results")
        st.image(annotated_image, use_container_width=True)

    st.subheader("Analysis")

    m1, m2, m3 = st.columns(3)
    m1.metric("Potholes Detected", len(result.boxes))
    m2.metric("Defect Coverage", f"{coverage:.2f}%")
    m3.metric("Severity", severity)

    if len(result.boxes) > 0:
        st.subheader("Detection Confidence Scores")
        for i, box in enumerate(result.boxes):
            confidence = float(box.conf[0])
            st.write(
                f"Pothole {i + 1}: {confidence:.2%} confidence"
            )

    st.caption(
        "Severity is estimated from detected bounding-box coverage. "
        "It does not measure pothole depth or guarantee road safety."
    )
