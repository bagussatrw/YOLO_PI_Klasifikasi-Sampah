# ==============================================================================
#  APLIKASI DETEKSI SAMPAH DENGAN FITUR LENGKAP
# ==============================================================================
#  Penulis: Gemini (Asisten Riset Virtual Anda)
#  Tujuan: Membuat antarmuka web interaktif untuk deteksi sampah dari webcam
#           atau gambar yang diunggah, dengan kontrol parameter.
# ==============================================================================

# --- Import Library yang Dibutuhkan ---
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
st.write("Aplikasi ini menggunakan model YOLO untuk mendeteksi sampah. Pilih mode deteksi di sidebar: dari kamera webcam secara real-time atau dengan mengunggah gambar.")

# --- Sidebar untuk Opsi dan Informasi ---
st.sidebar.header("Pengaturan")

# --- Fungsi untuk Memuat Model (dengan cache agar lebih efisien) ---
@st.cache_resource
def load_yolo_model(model_path):
    """
    Memuat model YOLO dari path yang diberikan.
    Menggunakan cache agar model tidak perlu dimuat ulang.
    """
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
    st.stop() # Hentikan aplikasi jika model gagal dimuat

# Dapatkan nama kelas dari model
CLASS_NAMES = model.names

# --- Pilihan Mode di Sidebar ---
app_mode = st.sidebar.radio(
    "Pilih Mode Aplikasi",
    ["Tentang Aplikasi", "Deteksi dari Gambar", "Deteksi dari Webcam"]
)

# --- Slider untuk Confidence Threshold ---
confidence_threshold = st.sidebar.slider(
    "Tingkat Keyakinan Deteksi", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

# --- Fungsi untuk Mendeteksi Kamera yang Tersedia ---
@st.cache_data
def get_available_cameras():
    """
    Memeriksa dan mendapatkan daftar indeks kamera yang tersedia.
    """
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

# ==============================================================================
# --- Logika untuk Setiap Mode Aplikasi ---
# ==============================================================================

# --- Mode: Tentang Aplikasi ---
if app_mode == "Tentang Aplikasi":
    st.header("Tentang Aplikasi Ini")
    st.markdown(
        """
        Aplikasi ini adalah sebuah prototipe yang dibuat untuk **Penulisan Ilmiah** dengan tujuan mendemonstrasikan penerapan *Computer Vision* menggunakan model **YOLO (You Only Look Once)**.
        
        **Fitur Utama:**
        - **Deteksi Real-Time:** Mampu mendeteksi objek sampah langsung dari kamera webcam.
        - **Deteksi dari Gambar:** Pengguna dapat mengunggah gambar untuk dianalisis.
        - **Klasifikasi Objek:** Model dilatih untuk membedakan antara sampah **organik** dan **anorganik**.
        - **Pengaturan Fleksibel:** Pengguna dapat menyesuaikan tingkat keyakinan deteksi melalui slider di sidebar.
        
        Silakan pilih mode lain di sidebar untuk mulai menggunakan aplikasi.
        """
    )
    # DIHAPUS: Baris kode untuk menampilkan gambar placeholder telah dihapus sesuai permintaan.
    # st.image("https://placehold.co/800x400/2B3467/FFFFFF?text=Contoh+Gambar+Aplikasi", caption="Ilustrasi Aplikasi Deteksi Sampah")

# --- Mode: Deteksi dari Gambar ---
elif app_mode == "Deteksi dari Gambar":
    st.header("Unggah Gambar untuk Deteksi")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar yang diunggah
        image = Image.open(uploaded_file)
        
        # Konversi gambar PIL ke array NumPy (formatnya masih RGB)
        image_np_rgb = np.array(image)
        
        # Konversi dari RGB ke BGR sebelum diprediksi
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
        
        # Lakukan deteksi pada gambar BGR
        results = model.predict(image_np_bgr, conf=confidence_threshold)
        
        # Metode .plot() akan menggambar pada gambar BGR dan mengembalikan gambar BGR
        annotated_image_bgr = results[0].plot()
        
        # Konversi kembali ke RGB untuk ditampilkan dengan benar di Streamlit
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

        # Gunakan kolom untuk membuat tata letak responsif
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
        with col2:
            st.image(annotated_image_rgb, caption="Gambar Hasil Deteksi", use_container_width=True)

        # Tampilkan ringkasan hasil deteksi
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


# --- Mode: Deteksi dari Webcam ---
elif app_mode == "Deteksi dari Webcam":
    st.header("Deteksi Real-Time dari Kamera")
    available_cameras = get_available_cameras()

    if not available_cameras:
        st.error("Tidak ada kamera yang ditemukan!")
    else:
        camera_options = {f"Kamera {i}": i for i in available_cameras}
        selected_camera_name = st.selectbox("Pilih Kamera:", options=list(camera_options.keys()))
        selected_camera_index = camera_options[selected_camera_name]

        # Menggunakan session_state untuk mengontrol status kamera
        if 'run_camera' not in st.session_state:
            st.session_state.run_camera = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button('Mulai Kamera'):
                st.session_state.run_camera = True
        with col2:
            if st.button('Hentikan Kamera'):
                st.session_state.run_camera = False
        
        FRAME_WINDOW = st.image([])

        if st.session_state.run_camera:
            camera = cv2.VideoCapture(selected_camera_index)
            
            while st.session_state.run_camera:
                success, frame = camera.read()
                if not success:
                    st.write("Gagal mengakses kamera.")
                    st.session_state.run_camera = False # Hentikan loop jika kamera error
                    break

                results = model.predict(frame, imgsz=640, conf=confidence_threshold)
                annotated_frame = results[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                FRAME_WINDOW.image(annotated_frame_rgb)
            
            camera.release()
            st.write('Kamera dimatikan.')
        else:
            st.info("Klik 'Mulai Kamera' untuk memulai deteksi.")

