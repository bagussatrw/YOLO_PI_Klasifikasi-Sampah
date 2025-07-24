# ==============================================================================
#  NAMA FILE: app.py (Versi Refactored)
#  DESKRIPSI: Kode yang telah dirapikan untuk kejelasan, keterbacaan,
#             dan praktik terbaik.
# ==============================================================================

# --------------------------------- IMPORTS ------------------------------------
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ------------------------- KONFIGURASI HALAMAN & JUDUL ------------------------
st.set_page_config(
    page_title="Deteksi Sampah Real-Time",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Aplikasi Deteksi Sampah Real-Time")

# -------------------------- FUNGSI PEMUATAN MODEL -----------------------------
@st.cache_resource
def load_yolo_model(model_path):
    """
    Memuat model YOLO dari path yang diberikan dan menyimpannya di cache.
    Ini mencegah model dimuat ulang setiap kali ada interaksi.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# -------------------------- PENGATURAN SIDEBAR --------------------------------
st.sidebar.header("Menu")

# Inisialisasi session_state jika belum ada
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Tentang Aplikasi"

# Tombol-tombol untuk navigasi mode
if st.sidebar.button("Tentang Aplikasi", use_container_width=True):
    st.session_state.app_mode = "Tentang Aplikasi"
if st.sidebar.button("Deteksi dari Gambar", use_container_width=True):
    st.session_state.app_mode = "Deteksi dari Gambar"
if st.sidebar.button("Deteksi Real-Time (Webcam)", use_container_width=True):
    st.session_state.app_mode = "Deteksi Real-Time (Webcam)"

st.sidebar.divider()

confidence_threshold = st.sidebar.slider(
    "Tingkat Keyakinan Deteksi",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# ------------------------ PEMUATAN MODEL & KELAS ------------------------------
MODEL_PATH = 'best.pt'
model = load_yolo_model(MODEL_PATH)
if not model:
    st.warning("Model tidak dapat dimuat. Pastikan file 'best.pt' ada di folder yang sama.")
    st.stop()

CLASS_NAMES = model.names

# ------------------------- KELAS PROSESOR VIDEO WEBRTC ------------------------
class YOLOVideoProcessor(VideoProcessorBase):
    """
    Class untuk memproses setiap frame video dari WebRTC.
    Deteksi YOLO dilakukan di sini.
    """
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        results = model.predict(image, conf=confidence_threshold)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ==============================================================================
# ------------------------- LOGIKA UTAMA APLIKASI ------------------------------
# ==============================================================================

# --- Halaman Tentang Aplikasi ---
if st.session_state.app_mode == "Tentang Aplikasi":
    st.header("Tentang Aplikasi Ini")
    st.markdown(
        """
        Aplikasi ini adalah prototipe yang menggunakan model **YOLO** untuk mendeteksi
        dan mengklasifikasikan sampah secara real-time.

        **Fitur Utama:**
        - **Streaming Real-Time (WebRTC):** Memberikan pengalaman deteksi yang mulus dan responsif.
        - **Deteksi Gambar:** Pengguna dapat mengunggah gambar untuk dianalisis.
        - **Klasifikasi Jenis Sampah:** Model dilatih untuk mengklasifikasikan antara sampah **organik** dan **anorganik**.
        - **Mengatur Tingkat Keyakinan:** Pengguna dapat menyesuaikan tingkat keyakinan deteksi untuk menyaring hasil.
        """
    )

# --- Halaman Deteksi dari Gambar ---
elif st.session_state.app_mode == "Deteksi dari Gambar":
    st.header("Unggah Gambar untuk Deteksi")
    uploaded_file = st.file_uploader(
        "Pilih sebuah gambar...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Konversi gambar untuk diproses
        image_np_rgb = np.array(image)
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        # Lakukan prediksi
        results = model.predict(image_np_bgr, conf=confidence_threshold)
        annotated_image_bgr = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

        # Tampilkan gambar berdampingan
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
        with col2:
            st.image(annotated_image_rgb, caption="Gambar Hasil Deteksi", use_container_width=True)

        # Tampilkan ringkasan hasil
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

# --- Halaman Deteksi Real-Time ---
elif st.session_state.app_mode == "Deteksi Real-Time (Webcam)":
    st.header("Deteksi Real-Time Menggunakan Kamera")
    st.write("Klik 'START' untuk menyalakan kamera.")

    webrtc_streamer(
        key="yolo-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:freestun.net:3478"]},
                {
                    "urls": ["turn:freestun.net:3478"],
                    "username": "free",
                    "credential": "free",
                },
            ]
        }
    )
