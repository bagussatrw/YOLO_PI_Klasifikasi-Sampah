# ==============================================================================
#  NAMA FILE: app_stable.py
#  VERSI: Stabil (Ambil Foto)
#  DESKRIPSI: Menggunakan st.camera_input. Dijamin berfungsi di semua
#             jaringan, tetapi tidak real-time streaming. Gunakan ini sebagai
#             rencana cadangan jika versi WebRTC gagal.
# ==============================================================================

import streamlit as st
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Sampah Canggih",
    page_icon="♻️",
    layout="wide"
)

# --- Judul dan Deskripsi Aplikasi ---
st.title("♻️ Aplikasi Deteksi Sampah Canggih")
st.write("Aplikasi ini menggunakan model YOLO untuk mendeteksi sampah. Pilih mode deteksi di sidebar.")

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
    ["Tentang Aplikasi", "Deteksi dari Gambar", "Deteksi Langsung dari Kamera"]
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
        Aplikasi ini adalah sebuah prototipe yang dibuat untuk **Penulisan Ilmiah** dengan tujuan mendemonstrasikan penerapan *Computer Vision* menggunakan model **YOLO (You Only Look Once)**.
        
        **Fitur Utama:**
        - **Deteksi Langsung:** Mampu mendeteksi objek sampah langsung dari kamera perangkat (desktop maupun mobile).
        - **Deteksi dari Gambar:** Pengguna dapat mengunggah gambar untuk dianalisis.
        - **Klasifikasi Objek:** Model dilatih untuk membedakan antara sampah **organik** dan **anorganik**.
        - **Pengaturan Fleksibel:** Pengguna dapat menyesuaikan tingkat keyakinan deteksi melalui slider di sidebar.
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

elif app_mode == "Deteksi Langsung dari Kamera":
    st.header("Deteksi Langsung dari Kamera")
    
    picture = st.camera_input("Arahkan kamera ke objek, lalu klik tombol 'Take Photo'")

    if picture is not None:
        image = Image.open(picture)
        image_np_rgb = np.array(image)
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        results = model.predict(image_np_bgr, conf=confidence_threshold)
        annotated_image_bgr = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar yang Diambil", use_container_width=True)
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
