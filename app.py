# ==============================================================================
#  NAMA FILE: app_webrtc.py
#  VERSI: Real-Time Streaming (Canggih)
#  DESKRIPSI: Menggunakan streamlit-webrtc untuk pengalaman deteksi yang
#             sangat mulus. Mungkin tidak berfungsi di beberapa jaringan
#             yang ketat (kampus/kantor).
# ==============================================================================

import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Sampah Real-Time",
    page_icon="♻️",
    layout="wide"
)

# --- Judul dan Deskripsi Aplikasi ---
st.title("♻️ Aplikasi Deteksi Sampah Real-Time (WebRTC)")
st.write("Aplikasi ini menggunakan model YOLO dan teknologi WebRTC untuk deteksi sampah yang mulus. Pilih mode di sidebar.")

# --- Sidebar untuk Opsi dan Informasi ---
st.sidebar.header("Pengaturan")

# --- Fungsi untuk Memuat Model (dengan cache agar lebih efisien) ---
@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- Path ke file model 'best.pt' ---
MODEL_PATH = 'best.pt'

# --- Muat Model ---
model = load_yolo_model(MODEL_PATH)

if not model:
    st.stop()

# Dapatkan nama kelas dari model
CLASS_NAMES = model.names

# --- Pilihan Mode di Sidebar ---
app_mode = st.sidebar.radio(
    "Pilih Mode Aplikasi",
    ["Tentang Aplikasi", "Deteksi dari Gambar", "Deteksi Real-Time (Webcam)"]
)

# --- Slider untuk Confidence Threshold ---
confidence_threshold = st.sidebar.slider(
    "Tingkat Keyakinan Deteksi", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

# ==============================================================================
# --- Logika untuk Setiap Mode Aplikasi ---
# ==============================================================================

if app_mode == "Tentang Aplikasi":
    st.header("Tentang Aplikasi Ini")
    st.markdown(
        """
        Aplikasi ini adalah prototipe canggih yang menggunakan **YOLO** untuk deteksi objek dan **Streamlit-WebRTC** untuk streaming video real-time.
        
        **Fitur Utama:**
        - **Streaming Real-Time (WebRTC):** Memberikan pengalaman deteksi yang sangat mulus dan responsif, layaknya video call.
        - **Deteksi dari Gambar:** Pengguna dapat mengunggah gambar untuk dianalisis.
        - **Klasifikasi Objek:** Model dilatih untuk membedakan antara sampah **organik** dan **anorganik**.
        - **Pengaturan Fleksibel:** Pengguna dapat menyesuaikan tingkat keyakinan deteksi.
        """
    )

elif app_mode == "Deteksi dari Gambar":
    st.header("Unggah Gambar untuk Deteksi")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np_rgb = np.array(image)
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
        
        results = model.predict(image_np_bgr, conf=confidence_threshold)
        annotated_image_bgr = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
        with col2:
            st.image(annotated_image_rgb, caption="Gambar Hasil Deteksi", use_container_width=True)

        st.subheader("Ringkasan Hasil Deteksi")
        if len(results[0].boxes) > 0:
            detection_counts = {}
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = CLASS_NAMES[class_id]
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
            for class_name, count in detection_counts.items():
                st.write(f"- Ditemukan **{count}** objek **{class_name}**")
        else:
            st.write("Tidak ada objek yang terdeteksi pada gambar ini.")

elif app_mode == "Deteksi Real-Time (Webcam)":
    st.header("Deteksi Real-Time Menggunakan WebRTC")
    st.write("Klik 'START' di bawah untuk menyalakan kamera Anda.")

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            results = model.predict(image, conf=confidence_threshold)
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="yolo-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                
                {
                    "urls": ["turn:openrelay.metered.ca:80"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                 },
    ]
}
