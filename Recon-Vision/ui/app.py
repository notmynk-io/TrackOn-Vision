import streamlit as st
import requests
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import pandas as pd
import time

# ----------- Settings and Styles ------------
st.set_page_config(
    page_title="Border Surveillance AI Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://127.0.0.1:8000"

# ----------- Sidebar Design -----------------
with st.sidebar:
    st.title("🛰️ Surveillance Monitor")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"]{
            background: linear-gradient(135deg, #22223b 40%, #4a4e69 100%);
            color: #f2e9e4;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.subheader("Recent Alerts")
    try:
        alerts_resp = requests.get(f"{API_BASE}/alerts")
        if alerts_resp.status_code == 200:
            alerts_data = alerts_resp.json()
            alerts_list = alerts_data.get("alerts", [])
            if alerts_list:
                st.dataframe(pd.DataFrame(alerts_list), use_container_width=True)
            else:
                st.markdown("No alerts available.")
        else:
            st.error(f"Failed to fetch alerts ({alerts_resp.status_code})")
    except Exception as e:
        st.write("Error fetching alerts:", e)

    st.subheader("Detection Hotspots")
    try:
        hotspots_resp = requests.get(f"{API_BASE}/hotspots")
        if hotspots_resp.status_code == 200:
            hotspots_data = hotspots_resp.json().get("hotspots", [])
            if hotspots_data:
                st.bar_chart(
                    pd.DataFrame(hotspots_data).set_index("hour")["count"],
                    use_container_width=True
                )
            else:
                st.markdown("No hotspot data yet.")
        else:
            st.error(f"Failed to fetch hotspots ({hotspots_resp.status_code})")
    except Exception as e:
        st.write("Error fetching hotspot data:", e)

# ----------- Main App Layout -----------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(90deg, #f2e9e4 0%, #c9ada7 40%, #4a4e69 100%);
        color: #22223b;
        border-radius: 12px;
        padding: 2em;
    }
    </style>
    """, unsafe_allow_html=True
)
st.title("Border Surveillance Detection Dashboard")
st.write("Monitor live camera stream or analyze uploaded images for AI-powered object/person detection.")

tab_image, tab_camera = st.tabs(["📷 Image Detection", "🎥 Camera Stream"])

# ----------- Image Detection Tab -------------
with tab_image:
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"],
        help="Supported formats: jpg, jpeg, png"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

        with st.spinner("Running detection..."):
            resp = requests.post(
                f"{API_BASE}/detect",
                files={"image": ("image.jpg", img_bytes, "image/jpeg")}
            )

        if resp.status_code == 200:
            data = resp.json()
            detections = data.get("detections", [])
            alerts = data.get("alerts", [])
            st.success(f"Detections: {len(detections)} | Alerts: {len(alerts)}")

            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                label = det["class"]
                conf = det["confidence"]
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (14, 118, 105), 3)
                cv2.putText(
                    img_cv, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (69, 78, 105), 3
                )
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Detection Results", use_column_width=True)
        else:
            st.error(f"Detection failed: {resp.status_code} - {resp.text}")

# ----------- Camera Stream Tab ---------------
with tab_camera:
    st.info("Start monitoring your webcam. Only 1 frame every few seconds is processed to ensure performance.")
    IP_CAMERA_URL = "rtsp:192.168.201.191:1935/"
    run_stream = st.button("Start Camera Stream", type="primary")
    FRAME_SKIP = 10  # Process every 10th frame for speed

    if run_stream:
        cap = cv2.VideoCapture(IP_CAMERA_URL)
        frame_count = 0
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera stream ended or unavailable.")
                break
            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()
                # Send frame to backend
                response = requests.post(
                    f"{API_BASE}/detect",
                    files={"image": ("frame.jpg", img_bytes, "image/jpeg")}
                )
                if response.status_code == 200:
                    data = response.json()
                    detections = data.get("detections", [])
                    for det in detections:
                        x1, y1, x2, y2 = det["box"]
                        label = det["class"]
                        conf = det["confidence"]
