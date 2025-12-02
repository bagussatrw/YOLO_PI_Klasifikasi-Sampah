# YOLO_PI_Klasifikasi-Sampah

**Deskripsi singkat**

Repositori ini berisi implementasi sederhana untuk **klasifikasi/pendeteksian sampah** menggunakan model YOLO yang sudah dilatih (`best.pt`) dan sebuah aplikasi Python (`app.py`) untuk menjalankan inferensi—dirancang untuk dijalankan pada perangkat seperti **Raspberry Pi** atau komputer kecil lain yang terhubung ke kamera.

> Catatan: isi README ini dibuat berdasarkan file-file yang tersedia di repositori (mis. `app.py`, `best.pt`, `requirements.txt`, `packages.txt`, dan konfigurasi `.devcontainer`). Sesuaikan instruksi berikut jika ada perubahan pada kode atau lingkungan target.

---

## Fitur

* Inference deteksi objek berbasis YOLO menggunakan model `best.pt`.
* Aplikasi Python (`app.py`) untuk menangkap frame dari kamera dan melakukan prediksi (menampilkan bounding box, label, dan confidence).
* Berkas dependensi (`requirements.txt`) untuk menginstall paket Python yang dibutuhkan.
* `packages.txt` berisi paket sistem / OS yang mungkin diperlukan pada Raspberry Pi (opsional, jika disediakan).

---

## Struktur repositori (ringkasan)

```
.YOLO_PI_Klasifikasi-Sampah/
├─ .devcontainer/           # Konfigurasi container development (opsional)
├─ app.py                   # Aplikasi utama untuk menjalankan inferensi
├─ best.pt                  # Model YOLO terlatih (file model)
├─ requirements.txt         # Dependensi Python
├─ packages.txt             # Paket sistem (opsional)
└─ README.md                # (file ini)
```

---

## Persyaratan

* Python 3.7+ (disarankan 3.8 atau 3.9 tergantung kompatibilitas paket)
* Pip
* Kamera yang terhubung (USB camera atau kamera modul pada Raspberry Pi)
* Jika dijalankan pada Raspberry Pi: pastikan paket dan driver kamera sudah terpasang, serta dukungan untuk versi PyTorch/torchvision pada arsitektur ARM (lihat catatan di bawah).

---

## Instalasi cepat (lokal)

1. Clone repositori:

```bash
git clone https://github.com/bagussatrw/YOLO_PI_Klasifikasi-Sampah.git
cd YOLO_PI_Klasifikasi-Sampah
```

2. Siapkan environment (opsional — disarankan menggunakan virtualenv/venv):

```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate     # Windows (PowerShell/CMD)
```

3. Instal dependensi Python:

```bash
pip install -r requirements.txt
```

> Jika sistem Anda adalah Raspberry Pi, pemasangan `torch` pada ARM memerlukan langkah khusus — gunakan rujukan resmi PyTorch untuk ARM atau gunakan wheel yang kompatibel. Jika ada `packages.txt`, jalankan terlebih dahulu (jika berisi paket apt):

```bash
# contoh jika packages.txt berisi paket apt
sudo xargs -a packages.txt apt-get install -y
```

---

## Menjalankan aplikasi

1. Pastikan model `best.pt` berada di folder yang sama dengan `app.py` (atau sesuaikan path di `app.py`).
2. Jalankan aplikasi:

```bash
python app.py
```

Aplikasi biasanya akan membuka sambungan ke kamera dan mulai melakukan inferensi pada frame yang diterima. Perhatikan keluaran terminal untuk informasi tentang deteksi (label, confidence) dan cara interaksi (mis. menekan `q` untuk keluar) — periksa `app.py` untuk parameter tambahan seperti threshold, ukuran input, atau path model.

---

## Catatan penting untuk Raspberry Pi

* Versi PyTorch resmi untuk Raspberry Pi (ARM) sering berbeda; pemasangan `pip install torch` pada Pi dapat gagal. Lihat dokumentasi PyTorch atau gunakan build yang sudah disediakan komunitas (wheel binary) untuk arsitektur yang sesuai.
* Untuk performa lebih baik, pertimbangkan:

  * Menggunakan model ringan (mis. YOLO-nano / YOLOv5s yang dikonversi atau distilasi model) atau mengkuantisasi model.
  * Menggunakan akselerator seperti Coral TPU, NPU, atau kamera dengan kemampuan onboard processing.
* Jika ingin menjalankan inference real-time, sesuaikan ukuran input/deteksi dan frekuensi pengambilan frame agar dapat dijalankan pada perangkat low-power.

---

## Mengganti / Melatih Ulang Model

* File `best.pt` adalah model terlatih. Jika Anda ingin melatih ulang atau memperbaiki akurasi:

  * Siapkan dataset berlabel (format sesuai YOLO: gambar + file label `.txt` per gambar dengan format kelas x_center y_center width height normalisasi)
  * Gunakan pipeline pelatihan YOLO (misalnya Ultralytics YOLOv5/YOLOv8) pada mesin dengan GPU.
  * Setelah pelatihan selesai, letakkan model hasil training di path `best.pt` atau sesuaikan pathnya di `app.py`.

---

## Troubleshooting

* `ModuleNotFoundError` atau kegagalan instal paket: pastikan `requirements.txt` terinstall pada virtual environment yang aktif.
* Kamera tidak terdeteksi: coba cek apakah kamera berfungsi pada OS (mis. `v4l2-ctl --list-devices` pada Linux), periksa permission, dan pastikan path kamera benar pada `app.py`.
* Masalah PyTorch di Raspberry Pi: cari wheel PyTorch yang cocok untuk versi Python + arsitektur Anda atau gunakan alternatif inference runtime (mis. ONNX Runtime, TensorRT jika tersedia).

