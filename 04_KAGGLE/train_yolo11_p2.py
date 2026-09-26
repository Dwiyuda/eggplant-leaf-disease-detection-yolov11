# %% [markdown]
# # Kepala deteksi P2 — YOLOv11 untuk lesi kecil
#
# —
#
# Menambahkan kepala deteksi **P2/4** pada YOLOv11s, lalu membandingkannya
# dengan baseline yang setelannya identik.
#
# ### Kenapa ini dicoba
#
# Pengukuran 8.989 kotak lesi (2026-08-07): median ukurannya **0,0248 × sisi citra**.
# Kepala terkecil YOLO11 bawaan adalah P3 dengan stride 8.
#
# | imgsz | median lesi | sel di P3 (stride 8) | sel di P2 (stride 4) |
# |---:|---:|---:|---:|
# | 640 | 15,9 px | 2,0 | 4,0 |
# | **1024** | **25,4 px** | **3,2** | **6,3** |
# | 1280 | 31,7 px | 4,0 | 7,9 |
#
# Ambang praktis deteksi sekitar 2 sel. P2 **menggandakan** angka itu tanpa
# menaikkan resolusi — dan menaikkan resolusi sudah terbukti mentok di 1024
# (1280 justru lebih buruk).
#
# ### Pembanding
#
# yolo11s / 80:10:10 / **imgsz 640** / batch 8 / epoch 150 / seed 0, dari ablasi
# resolusi 2026-08-08:
#
# | | val mAP50-95 | leaf_spot | insect | healthy | mosaic |
# |---|---:|---:|---:|---:|---:|
# | baseline | **0,5640** | 0,1986 | 0,3252 | 0,8621 | 0,8700 |
#
# Setelan di sini disamakan persis dengan run itu, jadi satu-satunya yang
# berubah adalah **ada tidaknya kepala P2**.
#
# ### Kenapa 640 dan bukan 1024
#
# Kepala P2 berstride 4, peta fiturnya 4x lebih besar dari P3 dan mendominasi
# VRAM. Dari angka terukur (yolo11s @1280 batch 8 = 7,81 GB pada T4 15 GB):
#
# | konfigurasi | perkiraan VRAM | |
# |---|---:|---|
# | baseline @1024 | 5,0 GB | aman |
# | **P2 @1024** | **20,2 GB** | **pasti OOM** |
# | **P2 @640** | **7,9 GB** | **aman** |
#
# Memaksa 1024 akan menurunkan batch otomatis jadi 4 atau 2, dan saat itu
# perbandingannya tidak sah lagi. Di 640 hipotesisnya tetap teruji penuh —
# lesi naik dari 2 sel jadi 4 sel, penggandaan yang sama seperti di 1024.
#
# ### Hipotesis dan cara menggugurkannya
#
# **Hipotesis:** mAP50-95 `leaf_spot` dan `insect` naik berarti (>0,03).
#
# **Gugur bila:** kelas lesi tidak naik, atau yang naik justru healthy dan
# mosaic. Kedua kelas besar itu sudah 700+ px — kepala stride 4 seharusnya
# tidak memberi mereka apa-apa.
#
# ### Biaya
#
# Perkiraan **3–4 jam** di T4. Kalau batch tetap 8, perbandingannya sah dan
# vonisnya bisa langsung dipakai. Kalau CUDA OOM tetap terjadi dan batch turun,
# skrip akan menahan vonisnya — baseline wajib diulang pada batch yang sama
# sebelum angkanya boleh dibandingkan.
#
# ### Sebelum menjalankan
# 1. **Add Data** → dataset berisi `dataset_80_10_10/`
# 2. **Settings → Accelerator → GPU T4 x2** (atau P100)
# 3. **Settings → Internet → On**
# 4. **Save Version → Save & Run All (Commit)** — jangan Draft Session, ini panjang

# %%
# =============================================================================
# KONFIGURASI
# =============================================================================
SPLIT_DIR   = "dataset_80_10_10"
SPLIT_LABEL = "p2"

SCALE    = "s"        # n|s|m  -- dikunci ke s agar sebanding dengan baseline
EPOCHS   = 150
PATIENCE = 30

# KENAPA 640, BUKAN 1024 (titik terbaik ablasi resolusi).
# Kepala P2 berjalan pada stride 4, jadi peta fiturnya 4x lebih besar dari P3
# dan mendominasi pemakaian VRAM. Perhitungan dari angka terukur (yolo11s @1280
# batch 8 = 7,81 GB pada T4 15 GB):
#     P2 @1024 -> perkiraan 20,2 GB  = PASTI OOM
#     P2 @640  -> perkiraan  7,9 GB  = aman
# Kalau dipaksa 1024, batch akan turun otomatis jadi 4 atau 2, dan saat itu
# perbandingannya TIDAK SAH lagi karena baseline memakai batch 8.
#
# Hipotesisnya tetap teruji penuh di 640: lesi naik dari 2 sel jadi 4 sel,
# penggandaan yang sama seperti di 1024. Dan kita PUNYA baseline batch 8 di 640.
# Kalau P2 terbukti menolong, barulah 1024 dicoba -- dengan batch diturunkan
# untuk kedua sisi supaya tetap sebanding.
IMGSZ    = 640
BATCH    = 8          # sama dengan baseline. Turun otomatis kalau OOM -- CATAT!
SEED     = 0
OPTIMIZER = "auto"
RESIZE_MAX = 1280     # sama dengan ablasi resolusi
EXTRA_AUG = True      # degrees=15, flipud=0.5

# Baseline: yolo11s @640, batch 8, dari ablasi resolusi 2026-08-08 (val mAP50-95)
BASELINE = {"total": 0.5640, "leaf_spot_disease": 0.1986, "insect_pest_disease": 0.3252,
            "healthy_leaf": 0.8621, "mosaic_virus_disease": 0.8700}

TIME_BUDGET_H = 11.0

print(f"Model  : yolo11{SCALE}-p2  (kepala P2/4 + P3/8 + P4/16 + P5/32)")
print(f"Epochs : {EPOCHS} (patience {PATIENCE}) | imgsz {IMGSZ} | batch {BATCH} | seed {SEED}")

# %%
# =============================================================================
# 1. Setup lingkungan
# =============================================================================
import os, sys, subprocess

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["COMET_MODE"] = "DISABLED"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import torch, ultralytics
from ultralytics import YOLO

try:
    from ultralytics import settings as ul_settings
    ul_settings.update({"wandb": False, "comet": False, "mlflow": False, "clearml": False})
except Exception as e:
    print("catatan:", e)

CUDA = torch.cuda.is_available()
print("ultralytics :", ultralytics.__version__)
print("torch       :", torch.__version__)
print("CUDA        :", CUDA, "|", torch.cuda.get_device_name(0) if CUDA else "TIDAK ADA GPU")
if CUDA:
    print("VRAM        : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
WORKERS = min(8, os.cpu_count() or 4)

# %%
# =============================================================================
# 2. Tulis arsitektur yolo11<scale>-p2.yaml
# =============================================================================
# Ditulis di sini supaya tidak perlu mengunggah berkas terpisah ke Kaggle.
# Nama berkas HARUS memuat skalanya (yolo11s-p2), karena Ultralytics membaca
# skala model dari nama berkas, bukan dari isinya.
from pathlib import Path

YAML_P2 = """
nc: 4
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]              # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]             # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]      # 2   <- keluaran P2/4
  - [-1, 1, Conv, [256, 3, 2]]             # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]      # 4
  - [-1, 1, Conv, [512, 3, 2]]             # 5-P4/16
  - [-1, 2, C3k2, [512, True]]             # 6
  - [-1, 1, Conv, [1024, 3, 2]]            # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]            # 8
  - [-1, 1, SPPF, [1024, 5]]               # 9
  - [-1, 2, C2PSA, [1024]]                 # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11
  - [[-1, 6], 1, Concat, [1]]                    # 12
  - [-1, 2, C3k2, [512, False]]                  # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 14
  - [[-1, 4], 1, Concat, [1]]                    # 15
  - [-1, 2, C3k2, [256, False]]                  # 16

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 17
  - [[-1, 2], 1, Concat, [1]]                    # 18  cat backbone P2
  - [-1, 2, C3k2, [128, False]]                  # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]                   # 20
  - [[-1, 16], 1, Concat, [1]]                   # 21
  - [-1, 2, C3k2, [256, False]]                  # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]                   # 23
  - [[-1, 13], 1, Concat, [1]]                   # 24
  - [-1, 2, C3k2, [512, False]]                  # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]                   # 26
  - [[-1, 10], 1, Concat, [1]]                   # 27
  - [-1, 2, C3k2, [1024, True]]                  # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]]          # Detect(P2, P3, P4, P5)
"""

WORK = Path("/kaggle/working")
CFG = WORK / f"yolo11{SCALE}-p2.yaml"
CFG.write_text(YAML_P2)
print("arsitektur ditulis:", CFG.name)

# Periksa sebelum training: bangun modelnya, hitung parameter dan stride kepala.
uji = YOLO(str(CFG))
n_par = sum(p.numel() for p in uji.model.parameters())
strides = [int(s) for s in uji.model.stride]
print(f"parameter : {n_par/1e6:.2f} juta")
print(f"stride kepala : {strides}")
assert 4 in strides, "Kepala P2 (stride 4) tidak terbentuk — periksa arsitektur."
assert len(strides) == 4, f"Diharapkan 4 kepala, dapat {len(strides)}"
print("OK: kepala P2 terbentuk.\n")
del uji

# %%
# =============================================================================
# 3. Dataset
# =============================================================================
import collections, yaml, shutil, time, cv2
from concurrent.futures import ThreadPoolExecutor

INPUT_ROOT = Path("/kaggle/input")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cari_dataset(nama):
    kand = [p.parent for p in INPUT_ROOT.rglob("images")
            if (p.parent / "images" / "train").is_dir() and (p.parent / "labels" / "train").is_dir()]
    if not kand:
        raise FileNotFoundError("Dataset tidak ketemu di /kaggle/input. Sudah di-Add Data?")
    cocok = [c for c in kand if c.name == nama]
    return cocok[0] if cocok else kand[0]


SRC = cari_dataset(SPLIT_DIR)
NAMES = {0: "healthy_leaf", 1: "mosaic_virus_disease",
         2: "leaf_spot_disease", 3: "insect_pest_disease"}
if (SRC / "data.yaml").exists():
    y = yaml.safe_load((SRC / "data.yaml").read_text())
    if isinstance(y.get("names"), dict):
        NAMES = {int(k): v for k, v in y["names"].items()}
    elif isinstance(y.get("names"), list):
        NAMES = {i: n for i, n in enumerate(y["names"])}
print("dataset:", SRC, "| kelas:", NAMES)

DATA = WORK / "data" / SPLIT_DIR
OUT = WORK / "hasil"
for d in (DATA, OUT):
    d.mkdir(parents=True, exist_ok=True)


def olah(a):
    s, t = a
    im = cv2.imread(str(s), cv2.IMREAD_COLOR)
    if im is None:
        return s.name
    h, w = im.shape[:2]
    if RESIZE_MAX and max(h, w) > RESIZE_MAX:
        k = RESIZE_MAX / max(h, w)
        im = cv2.resize(im, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(t), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return None


t0 = time.time()
for sp in ["train", "val", "test"]:
    di, dl = DATA / "images" / sp, DATA / "labels" / sp
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    for t in (SRC / "labels" / sp).glob("*.txt"):
        shutil.copy2(t, dl / t.name)
    tug = [(p, di / (p.stem + ".jpg")) for p in sorted((SRC / "images" / sp).iterdir())
           if p.suffix.lower() in IMG_EXT]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        gagal = [g for g in ex.map(olah, tug) if g]
    assert not gagal, f"gagal baca: {gagal[:5]}"
    print(f"{sp:6s} {len(tug):5d} citra")
print("pre-resize %.1f menit" % ((time.time() - t0) / 60))

DATA_YAML = DATA / "data.yaml"
DATA_YAML.write_text(yaml.safe_dump({
    "path": str(DATA), "train": "images/train", "val": "images/val", "test": "images/test",
    "nc": len(NAMES), "names": {int(k): v for k, v in NAMES.items()},
}, sort_keys=False, allow_unicode=True))

# %%
# =============================================================================
# 4. Training
# =============================================================================
import gc, json

AUG = {"degrees": 15.0, "flipud": 0.5} if EXTRA_AUG else {}
ARGS = dict(data=str(DATA_YAML), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ,
            seed=SEED, deterministic=True, optimizer=OPTIMIZER, workers=WORKERS,
            device=0 if CUDA else "cpu", project=str(OUT / "runs"), exist_ok=True,
            plots=True, val=True, save_period=-1, verbose=True, **AUG)


def metrik(m):
    b = m.box
    return {"precision": float(b.mp), "recall": float(b.mr),
            "mAP50": float(b.map50), "mAP50_95": float(b.map)}


def per_kelas(m):
    b = m.box
    idx = [int(c) for c in getattr(b, "ap_class_index", list(NAMES))]
    out = []
    for i, c in enumerate(idx):
        try:
            out.append({"kelas": NAMES.get(c, str(c)), "precision": float(b.p[i]),
                        "recall": float(b.r[i]), "mAP50": float(b.ap50[i]),
                        "mAP50_95": float(b.maps[c])})
        except Exception:
            p, r, a50, a = list(b.class_result(i))[:4]
            out.append({"kelas": NAMES.get(c, str(c)), "precision": float(p),
                        "recall": float(r), "mAP50": float(a50), "mAP50_95": float(a)})
    return out


t_mulai = time.time()
batch = BATCH
for _ in range(4):
    try:
        # Bobot pretrained dimuat ke backbone; kepala P2 mulai dari nol karena
        # memang tidak ada padanannya di bobot 3-kepala. Ini perilaku normal.
        model = YOLO(str(CFG)).load(f"yolo11{SCALE}.pt")
        t0 = time.time()
        model.train(name=f"p2_{SCALE}", batch=batch, **ARGS)
        durasi = (time.time() - t0) / 60
        break
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        batch = max(2, batch // 2)
        print(f"!! CUDA OOM -> batch={batch}")
        print("!! CATAT: batch berbeda dari baseline (8). Baseline WAJIB diulang")
        print("!! pada batch ini sebelum angkanya dibandingkan.")
else:
    raise RuntimeError("gagal melatih bahkan pada batch terkecil")

run_dir = Path(model.trainer.save_dir)
best = run_dir / "weights" / "best.pt"
epoch_pakai = len([l for l in (run_dir / "results.csv").read_text().splitlines()[1:] if l.strip()])

mb = YOLO(str(best))
m_val = mb.val(data=str(DATA_YAML), split="val", imgsz=IMGSZ, project=str(OUT / "runs"),
               name="p2_val", exist_ok=True, plots=True, verbose=False)
m_test = mb.val(data=str(DATA_YAML), split="test", imgsz=IMGSZ, project=str(OUT / "runs"),
                name="p2_test", exist_ok=True, plots=True, verbose=False)

HASIL = {"model": f"yolo11{SCALE}-p2", "batch": batch, "imgsz": IMGSZ,
         "epoch_terpakai": epoch_pakai, "epoch_maks": EPOCHS,
         "menit": round(durasi, 1),
         "parameter_juta": round(sum(p.numel() for p in mb.model.parameters()) / 1e6, 2),
         "run_dir": str(run_dir), "best_pt": str(best),
         "val": metrik(m_val), "test": metrik(m_test),
         "val_per_kelas": per_kelas(m_val), "test_per_kelas": per_kelas(m_test)}
print(f"\nselesai epoch {epoch_pakai}/{EPOCHS} dalam {durasi:.1f} menit")

# %%
# =============================================================================
# 5. Uji hipotesis terhadap baseline
# =============================================================================
import pandas as pd

vk = {r["kelas"]: r["mAP50_95"] for r in HASIL["val_per_kelas"]}
tab = pd.DataFrame([{
    "kelas": k, "baseline": round(BASELINE[k], 4), "p2": round(vk.get(k, float("nan")), 4),
    "selisih": round(vk.get(k, float("nan")) - BASELINE[k], 4),
} for k in ["leaf_spot_disease", "insect_pest_disease", "healthy_leaf", "mosaic_virus_disease"]])

print("val mAP50-95 per kelas  --  yolo11s-p2 vs baseline yolo11s @1024\n")
print(tab.to_string(index=False))
print(f"\nTOTAL   baseline {BASELINE['total']:.4f}  ->  p2 {HASIL['val']['mAP50_95']:.4f}"
      f"   ({HASIL['val']['mAP50_95'] - BASELINE['total']:+.4f})")

d_kecil = tab[tab.kelas.isin(["leaf_spot_disease", "insect_pest_disease"])].selisih.mean()
d_besar = tab[tab.kelas.isin(["healthy_leaf", "mosaic_virus_disease"])].selisih.mean()
print(f"\nrata-rata kelas KECIL : {d_kecil:+.4f}")
print(f"rata-rata kelas BESAR : {d_besar:+.4f}")

if batch != BATCH:
    print(f"\n!! PERINGATAN: batch turun jadi {batch}, tidak sebanding dengan baseline "
          f"(batch {BATCH}). Vonis di bawah TIDAK SAH sampai baseline diulang di batch {batch}.")
elif d_kecil > 0.03 and d_kecil > d_besar:
    print("\n=> HIPOTESIS DIDUKUNG. Kepala P2 menolong kelas lesi. Pakai untuk model final.")
elif d_kecil > 0.0:
    print("\n=> KENAIKAN KECIL (<0,03). P2 menolong tapi tidak sebanding biaya komputasinya.")
    print("   Timbang ulang: apakah waktu training 2-3x lipat sepadan.")
else:
    print("\n=> HIPOTESIS GUGUR. P2 tidak menolong kelas lesi.")
    print("   Batasnya bukan resolusi peta fitur. Arahkan usaha ke perbaikan label.")

print("\nCatatan: perbandingan memakai VAL, bukan test. Protokol test-sekali.")

# %%
# =============================================================================
# 6. Kemas hasil
# =============================================================================
import platform

PAKET = WORK / f"hasil_yolo11_{SPLIT_LABEL}"
if PAKET.exists():
    shutil.rmtree(PAKET)
PAKET.mkdir(parents=True)

BERKAS = ["results.csv", "results.png", "args.yaml", "labels.jpg",
          "confusion_matrix.png", "confusion_matrix_normalized.png",
          "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png",
          "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"]
for sub, asal in [("train", run_dir), ("val", OUT / "runs" / "p2_val"),
                  ("test", OUT / "runs" / "p2_test")]:
    tuj = PAKET / sub
    tuj.mkdir(parents=True, exist_ok=True)
    if asal.exists():
        for n in BERKAS:
            if (asal / n).exists():
                shutil.copy2(asal / n, tuj / n)
if Path(HASIL["best_pt"]).exists():
    shutil.copy2(HASIL["best_pt"], PAKET / f"best_yolo11{SCALE}-p2.pt")
shutil.copy2(CFG, PAKET / CFG.name)
tab.to_csv(PAKET / "perbandingan_p2.csv", index=False)

(PAKET / "metrik_lengkap.json").write_text(json.dumps({
    "percobaan": "kepala deteksi P2/4",
    "konfigurasi": {"scale": SCALE, "imgsz": IMGSZ, "batch": batch, "batch_diminta": BATCH,
                    "epochs": EPOCHS, "patience": PATIENCE, "seed": SEED,
                    "resize_max": RESIZE_MAX, "extra_aug": AUG},
    "lingkungan": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                   "python": platform.python_version(),
                   "gpu": torch.cuda.get_device_name(0) if CUDA else None},
    "baseline_val_mAP50_95": BASELINE,
    "hasil": HASIL,
}, indent=2, ensure_ascii=False))

ZIP = shutil.make_archive(str(WORK / f"hasil_yolo11_{SPLIT_LABEL}"), "zip", root_dir=PAKET)
print("ZIP :", ZIP, "| %.1f MB" % (Path(ZIP).stat().st_size / 1e6))
shutil.rmtree(WORK / "data", ignore_errors=True)
print("Simpan ke kaggle/FOLDER_HASIL_KAGGLE/ seperti sebelumnya.")
