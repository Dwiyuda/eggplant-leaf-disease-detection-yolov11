# Instruksi Kaggle — Pilot Koreksi Kotak (yolo11s @1024) —
Dibuat 2026-09-11.

## Apa ini

Latih ulang **yolo11s @1024** pada dataset yang 89 kotak lesinya sudah dikoreksi
manual (Roboflow). Setelan **identik** run label-bersih sebelumnya — satu-satunya
yang beda adalah kualitas kotak. Ini **pilot** untuk menguji sinyal, bukan run final.

## Dua berkas yang dipakai

| Berkas | Lokasi lokal | Fungsi |
|---|---|---|
| `dataset_80_10_10_koreksi.zip` | `02_DATASET/` | dataset yang di-upload ke Kaggle |
| `train_yolo11_koreksi_kotak.py` | `04_KAGGLE/` | skrip training (salin ke notebook) |

## Langkah demi langkah

### 1. Upload dataset baru ke Kaggle
1. Buka https://www.kaggle.com/datasets → **New Dataset**
2. Upload `02_DATASET/dataset_80_10_10_koreksi.zip`
3. Beri nama jelas, mis. **`terung-koreksi-kotak-80-10-10`**
4. Create. Tunggu Kaggle selesai meng-unzip.

> **JANGAN** timpa/gunakan dataset lama. Ini dataset baru supaya hasil bisa
> dibandingkan dengan run sebelumnya.

### 2. Buat notebook
1. Dari halaman dataset itu → **New Notebook**, atau Code → New Notebook lalu
   **Add Input** → pilih dataset `terung-koreksi-kotak-80-10-10`
2. **Settings → Accelerator → GPU T4 x2** (atau P100) — **WAJIB, JANGAN SAMPAI TERLEWAT.**
   Run pertama (2026-09-12) gagal justru karena ini: Accelerator = None → jalan di
   CPU (~1,8 jam/epoch) → dibunuh Kaggle di ~epoch 7 saat batas 12 jam. Skrip sekarang
   berhenti seketika kalau GPU mati. Pastikan sel pertama mencetak `CUDA: True | Tesla T4`.
3. **Settings → Internet → On** (wajib, untuk `pip install ultralytics`)

### 3. Masukkan skrip
- Buka `04_KAGGLE/train_yolo11_koreksi_kotak.py`, salin seluruh isinya ke sel
  notebook (skrip sudah bersel `# %%`, boleh disalin utuh atau per blok), **atau**
- File → Import Notebook → pilih file `.py` itu.

Tidak ada yang perlu diubah di dalam skrip — ia mencari folder `dataset_80_10_10/`
otomatis di `/kaggle/input`.

### 4. Jalankan
- **Save Version → Save & Run All (Commit)** — jangan Draft Session (biar tetap
  jalan walau tab ditutup).
- Perkiraan **±3,5 jam** di T4.

### 5. Unduh hasil
- Panel kanan → **Output** → `hasil_yolo11_koreksi_kotak.zip` → Download
- Simpan ke: **`04_KAGGLE/FOLDER_HASIL_KAGGLE/`**
- Beri tahu saya kalau sudah — saya lanjut **langkah 8 (evaluasi)** di lokal.

## Yang akan keluar

Isi zip: `img1024/best_yolo11s_img1024.pt`, kurva, confusion matrix, `metrik_lengkap.json`,
`perbandingan_koreksi_kotak.csv`, `per_kelas_koreksi_kotak.csv`.

Angka val/test langsung dibandingkan dengan pembanding kotak-lama:
**val mAP50-95 0,5656 · test mAP50 0,6736**.

## ⚠️ Ingat saat membaca hasil

Test set **tidak** dikoreksi (dipertahankan agar penggaris tetap). Jadi **test mAP
mungkin tidak naik penuh** walau model membaik — karena kunci jawaban test masih
autolabel lama yang rusak. Pembacaan utama pilot ini nanti (saya kerjakan lokal):
**akurasi tingkat-citra** (`evaluasi_sistem.py`) + **recall per kelas lesi** +
pemeriksaan kualitatif. Kalau leaf_spot/insect naik di situ → sinyal hijau untuk
scale-up anotasi kotak (termasuk kotak test) menuju target mAP50 >75%.
