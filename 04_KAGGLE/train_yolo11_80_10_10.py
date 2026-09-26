# %% [markdown]
# # Training YOLOv11 — Deteksi Penyakit Daun Terung (split 80:10:10)
#
# —
#
# Melatih tiga varian: **yolo11n, yolo11s, yolo11m** dengan setelan identik,
# early stopping, lalu mengemas seluruh hasil jadi satu file `.zip`.
#
# ### Sebelum menjalankan
# 1. **Add Data** → tambahkan dataset yang berisi folder `dataset_80_10_10/`
# 2. **Settings → Accelerator → GPU T4 x2** (atau P100)
# 3. **Settings → Internet → On** (perlu untuk `pip install ultralytics` + unduh bobot pretrained)
# 4. Run All
#
# ### Perkiraan waktu di T4
# yolo11n 1–1,5 jam · yolo11s 1,5–2,5 jam · yolo11m 3–4,5 jam → total ±6–8 jam
# (batas sesi Kaggle 12 jam; early stopping sering memotong lebih awal)

# %%
# =============================================================================
# KONFIGURASI  --  ubah hanya di sel ini
# =============================================================================

SPLIT_DIR   = "dataset_80_10_10"   # nama folder dataset di dalam /kaggle/input/...
SPLIT_LABEL = "80_10_10"           # dipakai untuk penamaan hasil

MODELS   = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]   # dilatih berurutan
EPOCHS   = 150      # batas atas; early stopping biasanya berhenti lebih awal
PATIENCE = 30       # berhenti kalau 30 epoch berturut-turut tanpa perbaikan mAP val
IMGSZ    = 640      # standar YOLO
BATCH    = 16       # aman untuk T4 16 GB; otomatis turun kalau CUDA OOM
SEED     = 0        # reproducibility
OPTIMIZER = "auto"  # biarkan Ultralytics memilih (SGD/AdamW) sesuai panjang training

# Pre-resize citra sebelum training.
# Citra asli rata-rata 12,7 MP (sisi terpanjang ~4.000 px). Men-decode JPEG sebesar itu
# tiap epoch membuat CPU jadi bottleneck dan GPU menganggur. Label YOLO memakai koordinat
# ternormalisasi, jadi resize proporsional TIDAK mengubah isi label sama sekali.
RESIZE_MAX = 1024   # sisi terpanjang setelah resize; 0 = matikan pre-resize

# Augmentasi tambahan di luar default Ultralytics.
# Citra dataset ini daun tunggal di latar putih dan tidak punya orientasi baku,
# jadi rotasi dan flip vertikal masuk akal secara domain.
# Set False kalau ingin memakai default Ultralytics apa adanya sebagai baseline.
EXTRA_AUG = True

TIME_BUDGET_H = 11.0   # jangan mulai model baru kalau sesi sudah lewat ini

print(f"Split  : {SPLIT_LABEL}  ({SPLIT_DIR})")
print(f"Model  : {MODELS}")
print(f"Epochs : {EPOCHS} (patience {PATIENCE}) | imgsz {IMGSZ} | batch {BATCH} | seed {SEED}")

# %%
# =============================================================================
# 1. Setup lingkungan
# =============================================================================
import os, sys, subprocess

os.environ["WANDB_MODE"]     = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["COMET_MODE"]     = "DISABLED"

# Untuk reproducibility ketat di penelitian, pin versinya, misal:
#   "ultralytics==8.3.100"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import torch, ultralytics
from ultralytics import YOLO

try:
    from ultralytics import settings as ul_settings
    ul_settings.update({"wandb": False, "comet": False, "mlflow": False, "clearml": False})
except Exception as e:
    print("catatan: gagal mematikan logger pihak ketiga:", e)

CUDA = torch.cuda.is_available()
print("ultralytics :", ultralytics.__version__)
print("torch       :", torch.__version__)
print("CUDA        :", CUDA, "|", torch.cuda.get_device_name(0) if CUDA else "TIDAK ADA GPU")
if CUDA:
    print("VRAM        : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print("PERINGATAN: GPU tidak aktif. Settings -> Accelerator -> GPU T4 x2 (atau P100).")

WORKERS = min(8, os.cpu_count() or 4)
print("dataloader workers:", WORKERS)

# %%
# =============================================================================
# 2. Temukan dataset di /kaggle/input dan periksa keutuhannya
# =============================================================================
from pathlib import Path
import collections, yaml

INPUT_ROOT = Path("/kaggle/input")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cari_dataset(nama_folder):
    """Cari folder yang punya struktur images/train + labels/train."""
    kandidat = []
    for p in INPUT_ROOT.rglob("images"):
        akar = p.parent
        if (akar / "images" / "train").is_dir() and (akar / "labels" / "train").is_dir():
            kandidat.append(akar)
    if not kandidat:
        raise FileNotFoundError(
            "Tidak menemukan folder dengan struktur images/train + labels/train di "
            "/kaggle/input. Pastikan dataset sudah di-Add ke notebook ini."
        )
    cocok = [c for c in kandidat if c.name == nama_folder]
    terpilih = cocok[0] if cocok else kandidat[0]
    if not cocok:
        print(f"PERINGATAN: folder '{nama_folder}' tidak ketemu, memakai '{terpilih}'")
    return terpilih


SRC = cari_dataset(SPLIT_DIR)
print("dataset sumber:", SRC)

NAMES = {0: "healthy_leaf", 1: "mosaic_virus_disease",
         2: "leaf_spot_disease", 3: "insect_pest_disease"}
src_yaml = SRC / "data.yaml"
if src_yaml.exists():
    y = yaml.safe_load(src_yaml.read_text())
    if isinstance(y.get("names"), dict):
        NAMES = {int(k): v for k, v in y["names"].items()}
    elif isinstance(y.get("names"), list):
        NAMES = {i: n for i, n in enumerate(y["names"])}
print("kelas:", NAMES)


def periksa(split):
    imgs = sorted(p for p in (SRC / "images" / split).iterdir() if p.suffix.lower() in IMG_EXT)
    lbls = {p.stem for p in (SRC / "labels" / split).glob("*.txt")}
    tanpa_label = [p.name for p in imgs if p.stem not in lbls]
    n_img, n_box, kosong = collections.Counter(), collections.Counter(), 0
    for p in imgs:
        t = SRC / "labels" / split / f"{p.stem}.txt"
        baris = [b for b in t.read_text().splitlines() if b.strip()] if t.exists() else []
        if not baris:
            kosong += 1
        for c in {int(b.split()[0]) for b in baris}:
            n_img[c] += 1
        for b in baris:
            n_box[int(b.split()[0])] += 1
    return imgs, tanpa_label, kosong, n_img, n_box


total_img = 0
print(f"\n{'split':6s} {'citra':>6s} {'box':>7s}   sebaran per kelas")
for split in ["train", "val", "test"]:
    imgs, tanpa_label, kosong, n_img, n_box = periksa(split)
    total_img += len(imgs)
    rinci = " | ".join(f"{NAMES.get(c, c)}: {n_img[c]}c/{n_box[c]}b" for c in sorted(NAMES))
    print(f"{split:6s} {len(imgs):6d} {sum(n_box.values()):7d}   {rinci}")
    assert not tanpa_label, f"{split}: {len(tanpa_label)} citra tanpa label -> {tanpa_label[:5]}"
    if kosong:
        print(f"       PERINGATAN: {kosong} label kosong di {split}")

print(f"\ntotal {total_img} citra, semua punya pasangan label.")

# %%
# =============================================================================
# 3. Pre-resize citra ke /kaggle/working + tulis data.yaml
# =============================================================================
# Memakai cv2 (bukan PIL) supaya perlakuan orientasi EXIF sama persis dengan
# autolabel.py yang membuat labelnya. Hasil resize ditulis tanpa tag EXIF, jadi
# tidak ada lagi ambiguitas orientasi antara citra dan label.
import cv2, shutil, time
from concurrent.futures import ThreadPoolExecutor

WORK = Path("/kaggle/working")
DATA = WORK / "data" / SPLIT_DIR
OUT = WORK / "hasil"
for d in (DATA, OUT):
    d.mkdir(parents=True, exist_ok=True)


def olah_satu(args):
    src_img, dst_img = args
    im = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    if im is None:
        return src_img.name
    h, w = im.shape[:2]
    if RESIZE_MAX and max(h, w) > RESIZE_MAX:
        s = RESIZE_MAX / max(h, w)
        im = cv2.resize(im, (max(1, round(w * s)), max(1, round(h * s))),
                        interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_img), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return None


t0 = time.time()
for split in ["train", "val", "test"]:
    di, dl = DATA / "images" / split, DATA / "labels" / split
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)

    for t in (SRC / "labels" / split).glob("*.txt"):        # label disalin apa adanya
        shutil.copy2(t, dl / t.name)

    tugas = [(p, di / (p.stem + ".jpg"))
             for p in sorted((SRC / "images" / split).iterdir())
             if p.suffix.lower() in IMG_EXT]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        gagal = [g for g in ex.map(olah_satu, tugas) if g]
    assert not gagal, f"gagal membaca: {gagal[:5]}"
    print(f"{split:6s} {len(tugas):5d} citra siap")

print("selesai dalam %.1f menit" % ((time.time() - t0) / 60))

DATA_YAML = DATA / "data.yaml"
DATA_YAML.write_text(yaml.safe_dump({
    "path": str(DATA), "train": "images/train", "val": "images/val", "test": "images/test",
    "nc": len(NAMES), "names": {int(k): v for k, v in NAMES.items()},
}, sort_keys=False, allow_unicode=True))
print("\n" + DATA_YAML.read_text())

# %%
# =============================================================================
# 4. Training tiga varian YOLOv11 (n, s, m)
# =============================================================================
# Ketiganya memakai konfigurasi identik supaya perbandingannya sah. Yang menentukan
# kapan berhenti adalah early stopping (patience), bukan jumlah epoch tetap.
import gc, json, time

AUG = {}
if EXTRA_AUG:
    AUG = {"degrees": 15.0, "flipud": 0.5}   # daun tidak punya orientasi baku

ARGS_DASAR = dict(
    data=str(DATA_YAML), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ,
    seed=SEED, deterministic=True, optimizer=OPTIMIZER, pretrained=True,
    workers=WORKERS, device=0 if CUDA else "cpu",
    project=str(OUT / "runs"), exist_ok=True, plots=True, val=True,
    save_period=-1, verbose=True, **AUG,
)


def ambil_metrik(m):
    """Ringkasan metrik dari objek hasil model.val()."""
    b = m.box
    return {"precision": float(b.mp), "recall": float(b.mr),
            "mAP50": float(b.map50), "mAP50_95": float(b.map)}


def metrik_per_kelas(m):
    """Baris per kelas. Memakai atribut array yang stabil lintas versi Ultralytics."""
    b = m.box
    idx = [int(c) for c in getattr(b, "ap_class_index", list(NAMES))]
    rows = []
    for i, c in enumerate(idx):
        try:
            rows.append({"kelas": NAMES.get(c, str(c)),
                         "precision": float(b.p[i]), "recall": float(b.r[i]),
                         "mAP50": float(b.ap50[i]), "mAP50_95": float(b.maps[c])})
        except Exception:
            p, r, a50, a = list(b.class_result(i))[:4]
            rows.append({"kelas": NAMES.get(c, str(c)), "precision": float(p),
                         "recall": float(r), "mAP50": float(a50), "mAP50_95": float(a)})
    return rows


HASIL = []
t_mulai = time.time()

for nama_model in MODELS:
    tag = Path(nama_model).stem
    lewat = (time.time() - t_mulai) / 3600
    if lewat > TIME_BUDGET_H:
        print(f"\n>>> {tag} DILEWATI: sudah {lewat:.1f} jam, mendekati batas sesi Kaggle.")
        print(f">>> Jalankan ulang dengan MODELS = ['{nama_model}'] di sesi terpisah.")
        break

    print("\n" + "=" * 78)
    print(f"  {tag}   (split {SPLIT_LABEL})")
    print("=" * 78)

    batch = BATCH
    for _ in range(3):
        try:
            model = YOLO(nama_model)
            t0 = time.time()
            model.train(name=f"{tag}_{SPLIT_LABEL}", batch=batch, **ARGS_DASAR)
            durasi = (time.time() - t0) / 60
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            batch = max(2, batch // 2)
            print(f"!! CUDA OOM -> ulangi dengan batch={batch}")
    else:
        print(f"!! {tag} gagal dilatih, dilewati.")
        continue

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    epoch_terpakai = len(
        [l for l in (run_dir / "results.csv").read_text().splitlines()[1:] if l.strip()]
    )

    model_best = YOLO(str(best))
    m_val = model_best.val(data=str(DATA_YAML), split="val", imgsz=IMGSZ,
                           project=str(OUT / "runs"), name=f"{tag}_{SPLIT_LABEL}_val",
                           exist_ok=True, plots=True, verbose=False)
    m_test = model_best.val(data=str(DATA_YAML), split="test", imgsz=IMGSZ,
                            project=str(OUT / "runs"), name=f"{tag}_{SPLIT_LABEL}_test",
                            exist_ok=True, plots=True, verbose=False)

    n_param = sum(p.numel() for p in model_best.model.parameters())
    HASIL.append({
        "model": tag, "split": SPLIT_LABEL, "batch": batch,
        "epoch_terpakai": epoch_terpakai, "epoch_maks": EPOCHS, "patience": PATIENCE,
        "menit_training": round(durasi, 1), "parameter_juta": round(n_param / 1e6, 2),
        "run_dir": str(run_dir), "best_pt": str(best),
        "val": ambil_metrik(m_val), "test": ambil_metrik(m_test),
        "val_per_kelas": metrik_per_kelas(m_val),
        "test_per_kelas": metrik_per_kelas(m_test),
    })

    print(f"\n{tag}: berhenti di epoch {epoch_terpakai}/{EPOCHS} setelah {durasi:.1f} menit")
    print("  val  ", {k: round(v, 4) for k, v in HASIL[-1]["val"].items()})
    print("  test ", {k: round(v, 4) for k, v in HASIL[-1]["test"].items()})

    del model, model_best
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()

print(f"\nTotal waktu: {(time.time() - t_mulai) / 3600:.2f} jam")

# %%
# =============================================================================
# 5. Perbandingan model dan pemilihan yang terbaik
# =============================================================================
# Aturan: model terbaik dipilih dari skor VALIDASI, angka yang dilaporkan di penelitian
# diambil dari TEST. Memilih berdasarkan test sama saja membocorkan data uji ke
# proses pemilihan model, dan angkanya jadi optimistis.
import pandas as pd

assert HASIL, "Tidak ada model yang berhasil dilatih."

ringkas = pd.DataFrame([{
    "model": h["model"], "params (juta)": h["parameter_juta"],
    "epoch": f'{h["epoch_terpakai"]}/{h["epoch_maks"]}',
    "menit": h["menit_training"],
    "val mAP50": round(h["val"]["mAP50"], 4),
    "val mAP50-95": round(h["val"]["mAP50_95"], 4),
    "test P": round(h["test"]["precision"], 4),
    "test R": round(h["test"]["recall"], 4),
    "test mAP50": round(h["test"]["mAP50"], 4),
    "test mAP50-95": round(h["test"]["mAP50_95"], 4),
} for h in HASIL]).sort_values("val mAP50-95", ascending=False)

print(f"PERBANDINGAN VARIAN YOLOv11  --  split {SPLIT_LABEL}\n")
print(ringkas.to_string(index=False))

terbaik = max(HASIL, key=lambda h: h["val"]["mAP50_95"])
print(f"\nModel terpilih (mAP50-95 validasi tertinggi): {terbaik['model']}")
print(f"  bobot: {terbaik['best_pt']}")

per_kelas = pd.DataFrame([
    {"model": h["model"], "subset": subset, **row}
    for h in HASIL
    for subset, rows in (("val", h["val_per_kelas"]), ("test", h["test_per_kelas"]))
    for row in rows
]).round(4)

print(f"\nMETRIK PER KELAS (test) -- {terbaik['model']}\n")
print(per_kelas[(per_kelas.model == terbaik["model"])
                & (per_kelas.subset == "test")].to_string(index=False))

print("\nCatatan Bab IV: dataset ini timpang di level box (~11:1). healthy_leaf dan "
      "mosaic_virus_disease hanya 1 box per citra, leaf_spot dan insect ~10 box. "
      "Karena itu mAP per kelas di atas lebih informatif daripada rata-ratanya saja.")

ringkas.to_csv(OUT / f"perbandingan_{SPLIT_LABEL}.csv", index=False)
per_kelas.to_csv(OUT / f"per_kelas_{SPLIT_LABEL}.csv", index=False)

# %%
# =============================================================================
# 6. Kemas hasil jadi satu file .zip untuk diunduh
# =============================================================================
import platform

PAKET = WORK / f"hasil_yolo11_{SPLIT_LABEL}"
if PAKET.exists():
    shutil.rmtree(PAKET)
PAKET.mkdir(parents=True)

BERKAS_RUN = ["results.csv", "results.png", "args.yaml", "labels.jpg",
              "confusion_matrix.png", "confusion_matrix_normalized.png",
              "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png",
              "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"]

for h in HASIL:
    tag = h["model"]
    for sub, asal in [("train", Path(h["run_dir"])),
                      ("val", OUT / "runs" / f"{tag}_{SPLIT_LABEL}_val"),
                      ("test", OUT / "runs" / f"{tag}_{SPLIT_LABEL}_test")]:
        tujuan = PAKET / tag / sub
        tujuan.mkdir(parents=True, exist_ok=True)
        if not asal.exists():
            continue
        for nama in BERKAS_RUN:
            if (asal / nama).exists():
                shutil.copy2(asal / nama, tujuan / nama)
    # hanya best.pt yang ikut; last.pt sengaja ditinggal supaya zip tidak membengkak
    if Path(h["best_pt"]).exists():
        shutil.copy2(h["best_pt"], PAKET / tag / f"best_{tag}_{SPLIT_LABEL}.pt")

for f in [f"perbandingan_{SPLIT_LABEL}.csv", f"per_kelas_{SPLIT_LABEL}.csv"]:
    if (OUT / f).exists():
        shutil.copy2(OUT / f, PAKET / f)

(PAKET / "metrik_lengkap.json").write_text(json.dumps({
    "split": SPLIT_LABEL,
    "konfigurasi": {"models": MODELS, "epochs": EPOCHS, "patience": PATIENCE,
                    "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
                    "optimizer": OPTIMIZER, "resize_max": RESIZE_MAX,
                    "extra_aug": AUG, "kelas": NAMES},
    "lingkungan": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                   "python": platform.python_version(),
                   "gpu": torch.cuda.get_device_name(0) if CUDA else None},
    "model_terpilih": terbaik["model"],
    "dasar_pemilihan": "mAP50-95 tertinggi pada subset validasi",
    "hasil": HASIL,
}, indent=2, ensure_ascii=False))

ZIPFILE = shutil.make_archive(str(WORK / f"hasil_yolo11_{SPLIT_LABEL}"), "zip", root_dir=PAKET)
print("ZIP siap :", ZIPFILE)
print("ukuran   : %.1f MB" % (Path(ZIPFILE).stat().st_size / 1e6))
print("\nIsi:")
for p in sorted(PAKET.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(PAKET)}  ({p.stat().st_size / 1e3:.0f} KB)")

print("\nCara mengunduh: panel kanan Kaggle -> Output -> "
      f"hasil_yolo11_{SPLIT_LABEL}.zip -> tombol download.")

# Bersihkan salinan dataset supaya folder Output tidak ikut membengkak
shutil.rmtree(WORK / "data", ignore_errors=True)
print("\nsalinan dataset kerja dihapus.")

# %% [markdown]
# ---
# ## Catatan metodologi
#
# **Kenapa early stopping, bukan epoch tetap.** `patience=30` menghentikan training
# kalau mAP validasi tidak membaik selama 30 epoch berturut-turut, lalu `best.pt`
# diambil dari epoch terbaik — bukan epoch terakhir. Ini penjaga overfitting standar
# dan bisa langsung dilaporkan di Bab III. Jumlah epoch yang benar-benar terpakai
# tercatat di kolom `epoch` pada tabel perbandingan.
#
# **Kenapa ketiga varian pakai setelan identik.** Kalau n, s, dan m dilatih dengan
# epoch, batch, atau augmentasi berbeda, perbandingannya tidak sah — tidak bisa
# dipisahkan mana efek ukuran model dan mana efek setelan. Satu-satunya yang boleh
# berubah otomatis adalah batch, itu pun hanya kalau CUDA OOM, dan nilainya dicatat.
#
# **Kenapa memilih dari val, melaporkan dari test.** Memilih model berdasarkan skor
# test berarti data uji ikut menentukan model, sehingga angkanya optimistis dan tidak
# lagi mengukur generalisasi.
#
# **Kenapa pre-resize aman.** Label YOLO berupa `class cx cy w h` dalam koordinat
# ternormalisasi 0–1. Resize proporsional tidak mengubah nilainya. Resize memakai cv2,
# sama seperti autolabel.py yang membuat labelnya, jadi perlakuan EXIF konsisten.
#
# **Yang wajib dilaporkan per kelas.** Dataset timpang di level bounding box ~11:1.
# YOLO menghitung loss per box, jadi AP healthy dan mosaic cenderung lebih
# berfluktuasi. Laporkan mAP per kelas, jangan hanya rata-ratanya.
#
# ## Isi file zip
#
# ```
# hasil_yolo11_80_10_10.zip
# ├── perbandingan_80_10_10.csv     ringkasan antar-varian
# ├── per_kelas_80_10_10.csv        metrik tiap kelas, val dan test
# ├── metrik_lengkap.json           semua angka + konfigurasi + versi library
# ├── yolo11n/
# │   ├── best_yolo11n_80_10_10.pt  bobot terbaik
# │   ├── train/   results.csv, results.png, confusion matrix, kurva PR/P/R/F1
# │   ├── val/     hasil validasi
# │   └── test/    hasil pengujian
# ├── yolo11s/  (sama)
# └── yolo11m/  (sama)
# ```
