# Sistem Web Deteksi Penyakit Daun Terung

Antarmuka web untuk mendemonstrasikan model YOLOv11s final pada foto daun terung.

---

## Menjalankan

Prasyarat: **Python 3.10–3.13** (diuji pada 3.12.8 dan 3.13.14) dan koneksi internet untuk instalasi pertama.

> **Windows: clone ke folder yang pendek**, misalnya `C:\eggplant-leaf-disease-detection-yolov11`. Beberapa berkas
> PyTorch memiliki path sangat panjang; bila folder proyek terlalu dalam, `pip install`
> gagal dengan `OSError: [Errno 2] No such file or directory ... Long Path support`.

Dari **akar proyek** (`eggplant-leaf-disease-detection-yolov11/` atau folder hasil clone), bukan dari folder ini:

```powershell
git clone https://github.com/Dwiyuda/eggplant-leaf-disease-detection-yolov11.git
cd eggplant-leaf-disease-detection-yolov11

# sekali saja: buat lingkungan Python lalu pasang dependensi
# (ultralytics ikut mengunduh PyTorch, ±1–2 GB)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r 07_SISTEM_WEB\requirements.txt

# jalankan
.venv\Scripts\python.exe 07_SISTEM_WEB\app.py
```

Di macOS/Linux ganti `.venv\Scripts\python.exe` dengan `.venv/bin/python`.

Bobot model (`04_KAGGLE/.../best_yolo11s_img1024.pt`) sudah ikut di repo, jadi tidak perlu diunduh terpisah.

Lalu buka **http://localhost:5000**

Port lain bila 5000 sudah dipakai:

```powershell
.venv\Scripts\python.exe 07_SISTEM_WEB\app.py --port 5001
```

Server memuat model **saat start**, bukan saat permintaan pertama. Bila ada prasyarat yang kurang, ia berhenti seketika dengan pesan jelas — bukan gagal di tengah demo.

---

## Cara memakai

**Unggah foto** — klik *Unggah Foto* atau seret berkas ke area gambar.

**Kamera** — klik *Nyalakan Kamera*, arahkan ke daun, lalu klik *Jepret*.

- **Mode live** menampilkan kotak yang mengikuti kamera. Frame dikecilkan ke 640 px agar ringan, jadi ini **bantuan membidik saja**.
- **Jepret** mengambil frame resolusi penuh dan memprosesnya pada imgsz 1024. **Inilah hasil yang otoritatif** dan yang sejalan dengan angka di penelitian.

**Panel detail teknis** tertutup secara default. Buka bila ingin melihat daftar kotak, skor tiap deteksi, dan ambang yang dipakai.

---

## Foto cadangan

Folder `statik/contoh/` berisi empat citra — satu per kelas — untuk berjaga bila kamera bermasalah saat demo.

Keempatnya diambil dari **subset test**, bukan train. Jadi bila ditanya apakah model sudah pernah melihat citra ini saat dilatih, jawabannya: **belum**.

| Berkas | Kelas sebenarnya |
|---|---|
| `Healthy Leaf Image 126.jpg` | healthy_leaf |
| `Healthy Leaf Image 283.jpg` | mosaic_virus_disease |
| `Healthy Leaf Image 245.jpg` | leaf_spot_disease |
| `Healthy Leaf Image 254.jpg` | insect_pest_disease |

> Nama berkas tidak selalu mencerminkan kelasnya. Banyak citra berpindah kelas saat perbaikan label (377 label diperbaiki, 18% dataset), sedangkan nama berkas aslinya dipertahankan agar dapat ditelusuri kembali ke dataset.

---

## Memeriksa kesehatan sistem

```powershell
.venv\Scripts\python.exe 07_SISTEM_WEB\deteksi.py --uji
.venv\Scripts\python.exe 07_SISTEM_WEB\app.py --uji
.venv\Scripts\python.exe 07_SISTEM_WEB\uji_konsistensi.py
```

Pada hasil clone, hanya `app.py --uji` yang bisa dijalankan: dua uji lainnya membutuhkan `02_DATASET/`, yang tidak ikut di repo karena ukurannya.

`uji_konsistensi.py` adalah yang terpenting: ia menjalankan model pada citra subset test dan membandingkan hasilnya dengan label. **Hasil terakhir: 18/24 = 0,750**, sejalan dengan akurasi tingkat-citra 0,798 yang dilaporkan di penelitian.

Bila uji ini gagal, penyebab paling mungkin adalah pipeline inferensi berubah — periksa bahwa `predict` dipanggil dengan `conf=0.01` dan penyaringan `0.05` dilakukan **setelahnya**, bukan di dalam `predict`.

---

## Susunan berkas

| Berkas | Tugasnya |
|---|---|
| `deteksi.py` | muat model, prediksi, gabung deteksi jadi satu kelas |
| `app.py` | rute HTTP, validasi berkas, balas JSON |
| `konten/penyakit.json` | informasi penyakit dan penanganan, bersitasi |
| `templat/index.html` | halaman |
| `statik/gaya.css` | seluruh token visual dari `MASTER.md` |
| `statik/app.js` | unggah, kamera, menggambar kotak |
| `statik/contoh/` | foto cadangan, disajikan Flask sebagai berkas statis |
| `uji_konsistensi.py` | bukti sistem sejalan dengan penelitian |

Bobot model **tidak disalin** ke folder ini. `deteksi.py` menunjuk ke `04_KAGGLE/FOLDER_HASIL_KAGGLE/hasil_yolo11_ablasi_resolusi_dataset_baru/img1024/best_yolo11s_img1024.pt` lewat satu konstanta.

Bobot itu **ikut disertakan di repositori** (18 MB), jadi setelah `git clone` sistem langsung bisa dijalankan tanpa menyalin berkas apa pun. Seluruh dataset dan artefak training sengaja tidak disertakan — lihat `.gitignore` di akar proyek.

---

## Keterbatasan — sebutkan sebelum ditanya

1. **Model hanya mengenal empat kelas** dan hanya dilatih pada citra **daun terung tunggal berlatar polos**. Diarahkan ke wajah atau dinding, ia tetap akan menjawab salah satu dari empat kelas. Karena itu sistem menampilkan peringatan bila `conf_maks < 0,25`.

2. **Mode live bukan hasil otoritatif.** Frame dikecilkan demi kecepatan. Hasil final selalu dari jepret atau unggah.

3. **Inferensi berjalan di CPU**, sekitar 343 ms per citra. Ini alat demo, bukan sistem produksi.

4. **Kualitas kotak dibatasi plafon anotasi otomatis.** Kotak lesi dibuat aturan warna `autolabel.py`, dan audit mengukur 46% kotak palsu serta 56% kotak asli terlewat. Itu sebabnya mAP kelas lesi rendah sementara akurasi tingkat-citra mencapai 79,8%. Rinciannya di `06_AUDIT_KOTAK/`.

5. **Angka "dominasi deteksi" bukan akurasi dan bukan probabilitas.** Ia adalah porsi kepercayaan kelas pemenang terhadap seluruh deteksi pada citra itu.

---

## Sumber konten penyakit

Gejala dan penyebab diambil dari tinjauan literatur (Pahlevi dkk. 2025; Nabillah dkk. 2024; Maulidia dkk. 2023; Shk dkk.; Putri Elsa dkk.). Saran penanganan dilengkapi dari CABI BioProtection Portal, Dinas Pertanian Kabupaten Buleleng, dan repositori Kementerian Pertanian. Sitasi lengkap tercantum di setiap entri `konten/penyakit.json` dan ditampilkan di antarmuka.

Tidak ada anjuran agronomi yang dikarang tanpa sumber.
