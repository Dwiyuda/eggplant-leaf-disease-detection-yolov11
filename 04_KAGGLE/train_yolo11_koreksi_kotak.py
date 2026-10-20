# %% [markdown]
# # Pilot koreksi kotak — YOLOv11s @1024 pada split 80:10:10
#
# —
#
# Melatih ulang **yolo11s @1024** pada dataset yang **89 kotak lesinya sudah
# dikoreksi manual** (ground-truth Roboflow, 75 train + 14 val). Setelan model,
# epoch, batch, seed, augmentasi, dan pre-resize **dikunci identik** dengan run
# label-bersih sebelumnya (`train_yolo11_ablasi_resolusi.py`, imgsz 1024). Satu-
# satunya yang berubah adalah **kualitas kotaknya** — jadi selisih hasil bisa
# dikaitkan langsung ke koreksi kotak, bukan ke setelan.
#
# ### Kenapa ini dijalankan
#
# Fase A memperbaiki KELAS tiap citra, tapi kotaknya tetap buatan `autolabel.py`
# (aturan warna). Audit membuktikan kotak autolabel ~46% palsu dan ~56% kotak
# asli terlewat, dan kalibrasi membuktikan autolabel sudah mentok. Ini menyerang
# akar rendahnya mAP kelas lesi (leaf_spot 0,34 · insect 0,58 test mAP50).
#
# **Ini PILOT 100 citra (biased sample), bukan anotasi ulang penuh.** Tujuannya
# menguji apakah kotak bersih memberi sinyal, sebelum memutuskan investasi
# anotasi berskala besar.
#
# ### Keterbatasan tafsir yang HARUS diingat
#
# 11 citra test **tidak** ikut dikoreksi (test dipertahankan agar penggaris
# banding tidak berubah). Konsekuensinya **test mAP dinilai dengan kotak test
# autolabel lama yang kita tahu rusak** — jadi kalau test mAP lesi tidak naik,
# itu BELUM tentu berarti koreksi sia-sia. Baca sinyal dari: **akurasi tingkat-
# citra** (`evaluasi_sistem.py`, lokal), **recall per kelas**, dan pemeriksaan
# kualitatif. Evaluasi A/B/C dikerjakan lokal via `bandingkan_model.py`.
#
# ### Pembanding (label bersih, kotak LAMA autolabel)
#
# yolo11s / 80:10:10 / imgsz 1024 / batch 8 — **val mAP50-95 0,5656 · test mAP50 0,6736**.
# Setelan di sini identik, jadi angkanya langsung sebanding.
#
# ### Sebelum menjalankan
# 1. **Add Data** → dataset berisi folder `dataset_80_10_10/` (versi **koreksi**,
#    yaitu `dataset_80_10_10_koreksi.zip` yang baru diunggah — JANGAN yang lama)
# 2. **Settings → Accelerator → GPU T4 x2** (atau P100)
# 3. **Settings → Internet → On**
# 4. Run All (Save Version → Save & Run All / Commit)
#
# ### Perkiraan waktu di T4
# imgsz 1024 saja → **±3,5 jam**.

# %%
# =============================================================================
# KONFIGURASI  --  ubah hanya di sel ini
# =============================================================================

SPLIT_DIR   = "dataset_80_10_10"
SPLIT_LABEL = "koreksi_kotak"

MODEL      = "yolo11s.pt"        # DIKUNCI, identik run label-bersih sebelumnya.
IMGSZ_LIST = [1024]              # pilot: hanya 1024 (titik terbaik yang sudah terbukti)

EPOCHS   = 150
PATIENCE = 30
SEED     = 0
OPTIMIZER = "auto"

# Batch 8, bukan 16. Alasannya: pada imgsz 1280 batch 16 kemungkinan besar kena
# CUDA OOM di T4 16 GB. Kalau batch dibiarkan 16 lalu turun otomatis hanya pada
# run 1280, ketiga run tidak lagi sebanding -- kenaikan yang muncul bisa jadi
# efek batch, bukan efek resolusi. Lebih baik dikunci rendah sejak awal untuk
# ketiganya. Konsekuensinya titik 640 harus dijalankan ulang di sini.
BATCH = 8

# Pre-resize dilakukan SEKALI pada nilai tertinggi, lalu dipakai ketiga run.
# Wajib >= max(IMGSZ_LIST), kalau tidak, melatih pada 1280 hanya memperbesar
# piksel yang detailnya sudah dibuang. Citra asli 4000 px, jadi ruangnya lega.
RESIZE_MAX = 1280

EXTRA_AUG = True          # degrees=15, flipud=0.5 -- sama seperti run sebelumnya
TIME_BUDGET_H = 11.0

assert RESIZE_MAX >= max(IMGSZ_LIST), "RESIZE_MAX harus >= imgsz terbesar"

print(f"Model     : {MODEL}  (dikunci)")
print(f"Resolusi  : {IMGSZ_LIST}")
print(f"Epochs    : {EPOCHS} (patience {PATIENCE}) | batch {BATCH} | seed {SEED}")
print(f"Pre-resize: {RESIZE_MAX} px")

# %%
# =============================================================================
# 1. Setup lingkungan
# =============================================================================
import os, sys, subprocess

os.environ["WANDB_MODE"]     = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["COMET_MODE"]     = "DISABLED"

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
    # BERHENTI SEKETIKA. Melatih di CPU makan ~1,8 jam/epoch -> 150 epoch tidak
    # akan selesai dalam batas 12 jam Kaggle (pernah kejadian: dibunuh di epoch 7).
    raise SystemExit(
        "\n!!! GPU TIDAK AKTIF -- BERHENTI. Jangan lanjut di CPU.\n"
        "    Perbaiki: Settings -> Accelerator -> GPU T4 x2 (atau P100),\n"
        "    lalu Save Version -> Save & Run All (Commit) lagi."
    )

WORKERS = min(8, os.cpu_count() or 4)
print("dataloader workers:", WORKERS)

# %%
# =============================================================================
# 2. Temukan dataset dan periksa keutuhannya
# =============================================================================
from pathlib import Path
import collections, yaml

INPUT_ROOT = Path("/kaggle/input")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cari_dataset(nama_folder):
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
# 3. Pre-resize SEKALI ke RESIZE_MAX, dipakai ketiga run
# =============================================================================
# Memakai cv2 supaya perlakuan orientasi EXIF sama persis dengan autolabel.py
# yang membuat labelnya. Label ternormalisasi, jadi resize proporsional tidak
# mengubah isinya sama sekali.
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

    for t in (SRC / "labels" / split).glob("*.txt"):
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
# 4. Training satu model, tiga resolusi
# =============================================================================
import gc, json, time

AUG = {"degrees": 15.0, "flipud": 0.5} if EXTRA_AUG else {}

ARGS_DASAR = dict(
    data=str(DATA_YAML), epochs=EPOCHS, patience=PATIENCE,
    seed=SEED, deterministic=True, optimizer=OPTIMIZER, pretrained=True,
    workers=WORKERS, device=0 if CUDA else "cpu",
    project=str(OUT / "runs"), exist_ok=True, plots=True, val=True,
    save_period=-1, verbose=True, **AUG,
)


def ambil_metrik(m):
    b = m.box
    return {"precision": float(b.mp), "recall": float(b.mr),
            "mAP50": float(b.map50), "mAP50_95": float(b.map)}


def metrik_per_kelas(m):
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

for imgsz in IMGSZ_LIST:
    tag = f"img{imgsz}"
    lewat = (time.time() - t_mulai) / 3600
    if lewat > TIME_BUDGET_H:
        print(f"\n>>> imgsz {imgsz} DILEWATI: sudah {lewat:.1f} jam, mendekati batas sesi.")
        print(f">>> Jalankan ulang dengan IMGSZ_LIST = [{imgsz}] di sesi terpisah.")
        break

    print("\n" + "=" * 78)
    print(f"  {MODEL}  @  imgsz {imgsz}   (batch {BATCH})")
    print("=" * 78)

    batch = BATCH
    for _ in range(3):
        try:
            model = YOLO(MODEL)
            t0 = time.time()
            model.train(name=tag, imgsz=imgsz, batch=batch, **ARGS_DASAR)
            durasi = (time.time() - t0) / 60
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            batch = max(2, batch // 2)
            print(f"!! CUDA OOM -> ulangi dengan batch={batch}")
            print("!! CATAT INI: batch berbeda membuat titik ini tidak sebanding "
                  "dengan resolusi lain.")
    else:
        print(f"!! imgsz {imgsz} gagal dilatih, dilewati.")
        continue

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    epoch_terpakai = len(
        [l for l in (run_dir / "results.csv").read_text().splitlines()[1:] if l.strip()]
    )

    model_best = YOLO(str(best))
    m_val = model_best.val(data=str(DATA_YAML), split="val", imgsz=imgsz,
                           project=str(OUT / "runs"), name=f"{tag}_val",
                           exist_ok=True, plots=True, verbose=False)
    m_test = model_best.val(data=str(DATA_YAML), split="test", imgsz=imgsz,
                            project=str(OUT / "runs"), name=f"{tag}_test",
                            exist_ok=True, plots=True, verbose=False)

    HASIL.append({
        "imgsz": imgsz, "model": MODEL, "batch": batch,
        "epoch_terpakai": epoch_terpakai, "epoch_maks": EPOCHS, "patience": PATIENCE,
        "menit_training": round(durasi, 1),
        "run_dir": str(run_dir), "best_pt": str(best),
        "val": ambil_metrik(m_val), "test": ambil_metrik(m_test),
        "val_per_kelas": metrik_per_kelas(m_val),
        "test_per_kelas": metrik_per_kelas(m_test),
    })

    print(f"\nimgsz {imgsz}: berhenti di epoch {epoch_terpakai}/{EPOCHS} "
          f"setelah {durasi:.1f} menit")
    print("  val  ", {k: round(v, 4) for k, v in HASIL[-1]["val"].items()})
    print("  test ", {k: round(v, 4) for k, v in HASIL[-1]["test"].items()})

    del model, model_best
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()

print(f"\nTotal waktu: {(time.time() - t_mulai) / 3600:.2f} jam")

# %%
# =============================================================================
# 5. Kurva resolusi dan uji hipotesis
# =============================================================================
import pandas as pd

assert HASIL, "Tidak ada run yang berhasil."
HASIL.sort(key=lambda h: h["imgsz"])

ringkas = pd.DataFrame([{
    "imgsz": h["imgsz"], "batch": h["batch"],
    "epoch": f'{h["epoch_terpakai"]}/{h["epoch_maks"]}',
    "menit": h["menit_training"],
    "val mAP50": round(h["val"]["mAP50"], 4),
    "val mAP50-95": round(h["val"]["mAP50_95"], 4),
    "test mAP50": round(h["test"]["mAP50"], 4),
    "test mAP50-95": round(h["test"]["mAP50_95"], 4),
} for h in HASIL])

print("KURVA RESOLUSI  --  yolo11s, split 80:10:10\n")
print(ringkas.to_string(index=False))

per_kelas = pd.DataFrame([
    {"imgsz": h["imgsz"], "subset": subset, **row}
    for h in HASIL
    for subset, rows in (("val", h["val_per_kelas"]), ("test", h["test_per_kelas"]))
    for row in rows
]).round(4)

# --- Uji hipotesis, memakai VAL (protokol test-sekali) ---
print("\n" + "=" * 78)
print("  UJI HIPOTESIS  --  mAP50-95 validasi per kelas terhadap resolusi")
print("=" * 78 + "\n")

pivot = (per_kelas[per_kelas.subset == "val"]
         .pivot(index="kelas", columns="imgsz", values="mAP50_95"))
print(pivot.to_string())

KECIL = ["leaf_spot_disease", "insect_pest_disease"]
BESAR = ["healthy_leaf", "mosaic_virus_disease"]
kolom = sorted(pivot.columns)

if len(kolom) >= 2:
    lo, hi = kolom[0], kolom[-1]
    d_kecil = (pivot.loc[[k for k in KECIL if k in pivot.index], hi]
               - pivot.loc[[k for k in KECIL if k in pivot.index], lo]).mean()
    d_besar = (pivot.loc[[k for k in BESAR if k in pivot.index], hi]
               - pivot.loc[[k for k in BESAR if k in pivot.index], lo]).mean()

    print(f"\nPerubahan rata-rata mAP50-95 val, imgsz {lo} -> {hi}:")
    print(f"  kelas objek KECIL (leaf_spot, insect) : {d_kecil:+.4f}")
    print(f"  kelas objek BESAR (healthy, mosaic)   : {d_besar:+.4f}")

    if d_kecil > 0.02 and d_kecil > d_besar:
        print("\n  => HIPOTESIS DIDUKUNG. Kelas objek kecil naik lebih banyak.")
        print("     Resolusi memang mekanismenya. Pertimbangkan naik ke 1536.")
    elif d_kecil <= 0.02 and d_besar <= 0.02:
        print("\n  => HIPOTESIS GUGUR. Tidak ada yang naik berarti.")
        print("     Batasnya bukan resolusi. Arahkan usaha ke perbaikan label.")
    else:
        print("\n  => HASIL CAMPURAN. Kenaikan tidak terkonsentrasi di kelas kecil,")
        print("     jadi mekanismenya bukan objek-kecil. Tafsirkan hati-hati dan")
        print("     jangan klaim resolusi sebagai sebabnya di laporan hasil.")
else:
    print("\nHanya satu resolusi selesai — kurva tidak bisa dibentuk.")

print("\nCatatan: angka test sengaja TIDAK dipakai memilih. Protokol test-sekali, "
      "")

ringkas.to_csv(OUT / f"perbandingan_{SPLIT_LABEL}.csv", index=False)
per_kelas.to_csv(OUT / f"per_kelas_{SPLIT_LABEL}.csv", index=False)

# %%
# =============================================================================
# 6. Kemas hasil jadi satu file .zip
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
    tag = f"img{h['imgsz']}"
    for sub, asal in [("train", Path(h["run_dir"])),
                      ("val", OUT / "runs" / f"{tag}_val"),
                      ("test", OUT / "runs" / f"{tag}_test")]:
        tujuan = PAKET / tag / sub
        tujuan.mkdir(parents=True, exist_ok=True)
        if not asal.exists():
            continue
        for nama in BERKAS_RUN:
            if (asal / nama).exists():
                shutil.copy2(asal / nama, tujuan / nama)
    if Path(h["best_pt"]).exists():
        shutil.copy2(h["best_pt"], PAKET / tag / f"best_yolo11s_{tag}.pt")

for f in [f"perbandingan_{SPLIT_LABEL}.csv", f"per_kelas_{SPLIT_LABEL}.csv"]:
    if (OUT / f).exists():
        shutil.copy2(OUT / f, PAKET / f)

(PAKET / "metrik_lengkap.json").write_text(json.dumps({
    "percobaan": "ablasi resolusi",
    "split": SPLIT_DIR,
    "konfigurasi": {"model": MODEL, "imgsz_list": IMGSZ_LIST, "epochs": EPOCHS,
                    "patience": PATIENCE, "batch": BATCH, "seed": SEED,
                    "optimizer": OPTIMIZER, "resize_max": RESIZE_MAX,
                    "extra_aug": AUG, "kelas": NAMES},
    "lingkungan": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                   "python": platform.python_version(),
                   "gpu": torch.cuda.get_device_name(0) if CUDA else None},
    "hipotesis": "mAP50-95 leaf_spot dan insect naik seiring imgsz; "
                 "healthy dan mosaic tidak berubah berarti",
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
print("Simpan ke kaggle/FOLDER_HASIL_KAGGLE/ seperti sebelumnya.")

shutil.rmtree(WORK / "data", ignore_errors=True)
print("\nsalinan dataset kerja dihapus.")
