# Panduan Audit Kotak — Roboflow

Untuk dua kelas terlemah: `leaf_spot_disease` dan `insect_pest_disease`.

## Pertanyaan yang sedang dijawab

Recall `leaf_spot` cuma **0,2297** — model menemukan kurang dari seperempat bercak yang dilabeli.

Selama ini kita memperbaiki **kelas mana** yang benar untuk tiap citra. Kita **belum pernah** memeriksa apakah **kotaknya sendiri** masuk akal.

| | Kalau ini yang benar |
|---|---|
| **(a)** Bercaknya nyata tapi terlalu kecil untuk dideteksi | recall rendah = keterbatasan model |
| **(b)** Sebagian kotak dibuat aturan warna padahal **tidak ada bercaknya** | recall rendah itu **wajar** — model tidak bisa menemukan yang tidak ada. Yang perlu diperbaiki aturan `autolabel.py`, bukan modelnya |

## Menyiapkan berkasnya

```powershell
cd eggplant-leaf-disease-detection-yolov11
.\.venv\Scripts\Activate.ps1
python 01_KODE\siapkan_audit_kotak.py --uji
python 01_KODE\siapkan_audit_kotak.py
```

Menghasilkan **50 citra per kelas, total 100**, dalam satu folder siap unggah.

## Kenapa separuh acak — ini penting

Tiap kelas dibagi dua kelompok:

| Kelompok | Isi | Gunanya |
|---|---|---|
| **terburuk** (25) | paling banyak kotak yang tak ditemukan model | membuktikan gejalanya **nyata** |
| **acak** (25) | diambil acak dari sisanya | memberi **angka yang boleh dikutip** |

Run sebelumnya memberi "71% kotak tak ditemukan" — angka itu **tidak boleh dikutip**, karena 30 citranya sengaja dipilih sebagai kasus terburuk. Tinggi karena cara memilihnya, bukan karena datasetnya begitu.

Kolom `kelompok` di `prioritas.csv` menandai tiap citra. **Saat menghitung persentase nanti, pakai kelompok `acak` saja.**

Jebakan yang sama pernah terjadi pada skrining label, dan waktu itu diselesaikan dengan cara yang sama.

## Struktur yang dihasilkan

```
06_AUDIT_KOTAK\
├── prioritas.csv              <- lembar vonis
├── classes.txt
└── roboflow_upload\           <- INI yang diunggah
    ├── <citra>.jpg
    ├── <citra>.txt            label YOLO yang ada sekarang
    ├── classes.txt
    └── _darknet.labels
```

**Kedua kelas dalam satu folder, bukan dipisah.** Roboflow memetakan ulang id kelas menurut apa yang ditemukannya saat unggah — kalau diunggah sebagai dua proyek terpisah, keduanya jadi id 0 dan tercampur waktu digabungkan kembali.

## Langkah di Roboflow

**1. Buat proyek**
roboflow.com → **Create New Project** → tipe **Object Detection** → nama bebas, misalnya `audit-kotak-terung`.

**2. Unggah**
Drag seluruh isi folder `roboflow_upload` ke area unggah — **citra dan berkas `.txt` sekaligus**. Roboflow mendeteksi format YOLO Darknet dan langsung memasangkan tiap label ke citranya.

Pastikan setelah unggah muncul keterangan bahwa anotasi terbaca. Kalau tertulis "0 annotations", berarti berkas `.txt` tidak ikut terunggah — ulangi dan pastikan ikut terpilih.

**3. Periksa nama kelas**
Harus muncul **4 kelas**: `healthy_leaf`, `mosaic_virus_disease`, `leaf_spot_disease`, `insect_pest_disease`. Kalau cuma 2 yang muncul itu wajar — kelas lain memang tidak ada di 100 citra ini. Yang penting **namanya jangan diubah**.

**4. Jangan pakai Preprocessing atau Augmentation**
Saat membuat Version, matikan semuanya — resize, auto-orient, augmentasi. Kita cuma butuh anotasinya, bukan dataset olahan. Augmentasi dari Roboflow akan merusak kecocokan dengan pipeline kita.

**5. Anotasi**
Masuk ke **Annotate**, buka tiap citra. Kotak yang sekarang sudah tampil — tinggal dikoreksi.

**6. Ekspor**
Setelah selesai: **Versions** → **Create New Version** (tanpa preprocessing/augmentasi) → **Export Dataset** → format **YOLOv8** → **download zip**.

Simpan zip-nya ke `06_AUDIT_KOTAK\hasil_roboflow\`.

## Yang dikerjakan pada tiap citra

| Yang terlihat | Tindakan |
|---|---|
| Kotak menandai bercak/lubang **nyata**, letaknya pas | biarkan |
| Bercak nyata tapi kotaknya **meleset** | geser / ubah ukurannya |
| Kotak **tidak ada apa-apanya** — cuma urat daun, bayangan, gradasi warna | **HAPUS** |
| Ada bercak nyata tapi **tidak ada kotaknya** | tambahkan |

Aturan kelas, sama seperti sejak awal proyek:

| Yang terlihat | Kelas |
|---|---|
| lubang **tanpa** warna di tepinya | `insect_pest_disease` |
| lubang **dengan** halo kuning/coklat/hitam | `leaf_spot_disease` |
| bercak berwarna **tanpa** lubang | `leaf_spot_disease` |

## Isi lembar vonis — ini yang paling menentukan

Buka `prioritas.csv`, isi kolom `vonis_kotak`:

| Vonis | Artinya |
|---|---|
| `banyak_palsu` | sebagian besar kotak lama tidak ada bercaknya |
| `sebagian_palsu` | campur |
| `sudah_benar` | kotaknya wajar, cuma perlu digeser |
| `kurang_kotak` | bercaknya justru lebih banyak dari yang dilabeli |

**Vonis ini lebih penting daripada gambar kotaknya.** Dia yang menentukan apakah `autolabel.py` perlu diperbaiki, dan bagian mana yang salah.

## Berapa lama

Seratus citra, sekitar **1,5–2 menit** per citra kalau cuma mengoreksi. Sekitar **3 jam**, bisa dicicil.

Kalau waktunya mepet, kerjakan **kelompok `acak` dulu** — itu yang memberi angka. Kelompok `terburuk` boleh menyusul.

## Setelah selesai

Kirim dua hal:

1. `prioritas.csv` yang sudah terisi vonisnya
2. zip ekspor dari Roboflow

Dari situ dihitung berapa persen kotak yang ternyata palsu — **dari kelompok acak saja** — lalu diputuskan:

- **Banyak palsu** → perbaiki ambang di `autolabel.py` supaya berhenti membuat kotak pada variasi warna biasa, buat ulang label seluruh dataset, latih ulang. Jauh lebih murah daripada menggambar 12.693 kotak dengan tangan.
- **Kotaknya sudah wajar** → recall rendah memang keterbatasan deteksi objek kecil. Laporkan apa adanya di hasil evaluasi, dan berhenti di situ.

Anotasi manual 100 citra ini juga jadi **acuan kebenaran**: bisa dipakai mengukur seberapa jauh `autolabel.py` menyimpang, dan itu angka yang bagus untuk dilaporkan.
