import streamlit as st
import requests
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

API_URL = "http://127.0.0.1:8000/detect"  # Adjust if backend is hosted elsewhere

st.title("Border Surveillance - AI Detection Dashboard")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    with st.spinner("Running detection..."):
        response = requests.post(API_URL, files={"image": ("image.jpg", img_bytes, "image/jpeg")})

    if response.status_code == 200:
        data = response.json()
        detections = data.get("detections", [])
        st.write(f"Detections: {len(detections)}")

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["class"]
            conf = det["confidence"]
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Detection Results", use_column_width=True)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
