#!/usr/bin/env python3
"""
Audit visual prediksi YOLOv11 vs label, untuk menjawab satu pertanyaan:
apakah false positive yang besar itu model mengarang, atau justru menemukan
lesi asli yang terlewat oleh autolabel.py?

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

Tiap deteksi digolongkan jadi empat, dan penggolongan inilah inti auditnya:

    TP        kotak cocok (IoU >= ambang) DAN kelasnya benar
    FP_latar  kotak tidak cocok dengan label mana pun
              -> INI yang harus dilihat mata. Kalau di situ memang ada bercak
                 atau gigitan, berarti LABEL yang salah, bukan model.
    FP_kelas  kotak cocok posisinya tapi kelasnya keliru
              -> ini kekacauan leaf_spot vs insect, bukan soal lokalisasi
    FN        label yang tidak tersentuh prediksi mana pun

Keluaran:
    audit_hasil/<kelas>/<nama>.jpg   gambar bersisian: label | prediksi
    audit_hasil/ringkasan.csv        satu baris per citra
    audit_hasil/rekap.csv            agregat per kelas

Pakai:
    python audit_prediksi.py --uji                 # self-check, tanpa ultralytics
    python audit_prediksi.py                       # 10 citra/kelas, split 80_10_10
    python audit_prediksi.py --sampel 20
    python audit_prediksi.py --split 70_15_15
    python audit_prediksi.py --semua               # statistik seluruh test set
    python audit_prediksi.py --terparah            # pilih citra dengan FP_latar terbanyak

Prasyarat:
    pip install ultralytics
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# ======================================================================
# KONFIGURASI
# ======================================================================
AKAR = Path(__file__).resolve().parent.parent   # skrip ada di 01_KODE/
DIR_DATASET = AKAR / "02_DATASET"
DIR_SUMBER  = AKAR / "03_CITRA_SUMBER"
DIR_KAGGLE  = AKAR / "04_KAGGLE" / "FOLDER_HASIL_KAGGLE"
DIR_HASIL   = AKAR / "05_HASIL"
HASIL_KAGGLE = DIR_KAGGLE
KELUARAN = DIR_HASIL / "audit_hasil"

NAMA_KELAS = {
    0: "healthy_leaf",
    1: "mosaic_virus_disease",
    2: "leaf_spot_disease",
    3: "insect_pest_disease",
}

# Model terpilih tiap split, dari mAP50-95 validasi tertinggi.
# 
MODEL_TERPILIH = {
    "80_10_10": "yolo11m",
    "70_15_15": "yolo11s",
}

# Ambang. conf 0.25 disamakan dengan default confusion matrix Ultralytics
# supaya hasil audit ini bisa dibaca berdampingan dengan confusion_matrix.png.
CONF = 0.25
IOU_COCOK = 0.50

SISI_TAMPIL = 900          # tinggi tiap panel pada gambar bersisian
TEBAL = 2

# BGR
WARNA = {
    "label":    (200, 200, 60),    # cyan  - kotak label
    "FN":       (255, 120, 0),     # biru  - label yang terlewat model
    "TP":       (60, 200, 60),     # hijau - benar
    "FP_kelas": (0, 165, 255),     # oranye- posisi benar, kelas salah
    "FP_latar": (60, 60, 235),     # merah - tidak ada labelnya sama sekali
}


# ======================================================================
# GEOMETRI DAN PENCOCOKAN
# ======================================================================
def iou(a, b):
    """IoU dua kotak format (x1, y1, x2, y2)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    irisan = (x2 - x1) * (y2 - y1)
    luas_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    luas_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    gabungan = luas_a + luas_b - irisan
    return irisan / gabungan if gabungan > 0 else 0.0


def muat_label(berkas, lebar, tinggi):
    """Baca label YOLO ternormalisasi jadi (kelas, x1, y1, x2, y2) piksel."""
    kotak = []
    if not Path(berkas).exists():
        return kotak
    for baris in Path(berkas).read_text().splitlines():
        bagian = baris.split()
        if len(bagian) < 5:
            continue
        k = int(float(bagian[0]))
        cx, cy, w, h = (float(v) for v in bagian[1:5])
        kotak.append((k,
                      (cx - w / 2) * lebar, (cy - h / 2) * tinggi,
                      (cx + w / 2) * lebar, (cy + h / 2) * tinggi))
    return kotak


def cocokkan(label, prediksi, iou_min=IOU_COCOK):
    """
    Cocokkan prediksi ke label secara greedy, urut dari conf tertinggi.
    Satu label hanya boleh dipakai sekali -- deteksi kembar jadi FP.

    label    : [(kelas, x1, y1, x2, y2)]
    prediksi : [(kelas, x1, y1, x2, y2, conf)]

    Kembalian: (vonis_prediksi, indeks_FN)
        vonis_prediksi[i] = "TP" | "FP_kelas" | "FP_latar"
    """
    terpakai = [False] * len(label)
    vonis = [None] * len(prediksi)

    urutan = sorted(range(len(prediksi)), key=lambda i: -prediksi[i][5])
    for i in urutan:
        pk = prediksi[i][0]
        pkotak = prediksi[i][1:5]

        terbaik, skor_terbaik = -1, iou_min
        for j, (lk, *lkotak) in enumerate(label):
            if terpakai[j]:
                continue
            s = iou(pkotak, lkotak)
            if s >= skor_terbaik:
                terbaik, skor_terbaik = j, s

        if terbaik < 0:
            vonis[i] = "FP_latar"
        else:
            terpakai[terbaik] = True
            vonis[i] = "TP" if label[terbaik][0] == pk else "FP_kelas"

    fn = [j for j, t in enumerate(terpakai) if not t]
    return vonis, fn


# ======================================================================
# GAMBAR
# ======================================================================
def _skala(img, tinggi_target):
    h, w = img.shape[:2]
    s = tinggi_target / h
    return cv2.resize(img, (max(1, round(w * s)), tinggi_target),
                      interpolation=cv2.INTER_AREA), s


def _kotak(img, x1, y1, x2, y2, warna, teks=None, tebal=TEBAL):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), warna, tebal)
    if teks:
        (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(int(y1), th + 4)
        cv2.rectangle(img, (int(x1), ty - th - 4), (int(x1) + tw + 4, ty), warna, -1)
        cv2.putText(img, teks, (int(x1) + 2, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _judul(img, teks, tinggi=34):
    pita = np.full((tinggi, img.shape[1], 3), 32, np.uint8)
    cv2.putText(pita, teks, (8, tinggi - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([pita, img])


def bersisian(citra, label, prediksi, vonis, fn, nama, kelas_folder):
    """Panel kiri = label, panel kanan = prediksi berwarna menurut vonis."""
    dasar, s = _skala(citra, SISI_TAMPIL)

    kiri = dasar.copy()
    for j, (k, x1, y1, x2, y2) in enumerate(label):
        terlewat = j in fn
        _kotak(kiri, x1 * s, y1 * s, x2 * s, y2 * s,
               WARNA["FN"] if terlewat else WARNA["label"],
               f"{NAMA_KELAS.get(k, k)}{' TERLEWAT' if terlewat else ''}",
               TEBAL + 1 if terlewat else TEBAL)

    kanan = dasar.copy()
    for i, (k, x1, y1, x2, y2, conf) in enumerate(prediksi):
        _kotak(kanan, x1 * s, y1 * s, x2 * s, y2 * s, WARNA[vonis[i]],
               f"{NAMA_KELAS.get(k, k)} {conf:.2f} [{vonis[i]}]")

    c = Counter(vonis)
    kiri = _judul(kiri, f"LABEL  n={len(label)}   terlewat(FN)={len(fn)}")
    kanan = _judul(kanan, "PREDIKSI  n=%d   TP=%d  FP_kelas=%d  FP_latar=%d"
                          % (len(prediksi), c["TP"], c["FP_kelas"], c["FP_latar"]))

    pemisah = np.full((kiri.shape[0], 6, 3), 90, np.uint8)
    gabung = np.hstack([kiri, pemisah, kanan])
    return _judul(gabung, f"{kelas_folder}  |  {nama}", 30)


# ======================================================================
# ALUR UTAMA
# ======================================================================
def cari_bobot(split, model):
    pola = f"best_{model}_{split}.pt"
    for p in HASIL_KAGGLE.rglob(pola):
        return p
    return None


def jalankan(args):
    from ultralytics import YOLO          # diimpor di sini supaya --uji tak butuh

    split = args.split
    model_tag = args.model or MODEL_TERPILIH[split]
    bobot = cari_bobot(split, model_tag)
    if bobot is None:
        sys.exit(f"Bobot 'best_{model_tag}_{split}.pt' tidak ketemu di {HASIL_KAGGLE}")

    akar_data = DIR_DATASET / f"dataset_{split}"
    dir_img = akar_data / "images" / "test"
    dir_lbl = akar_data / "labels" / "test"
    if not dir_img.is_dir():
        sys.exit(f"Folder tidak ada: {dir_img}")

    semua = sorted(p for p in dir_img.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"bobot  : {bobot.name}")
    print(f"data   : {dir_img}  ({len(semua)} citra test)")
    print(f"ambang : conf={CONF}  IoU={IOU_COCOK}\n")

    model = YOLO(str(bobot))

    # Kelas sebuah citra diambil dari label mayoritasnya -- dataset ini satu
    # kondisi dominan per citra, jadi ini setara dengan folder asalnya.
    def kelas_citra(p):
        k = [b[0] for b in muat_label(dir_lbl / f"{p.stem}.txt", 1, 1)]
        return Counter(k).most_common(1)[0][0] if k else -1

    per_kelas = defaultdict(list)
    for p in semua:
        per_kelas[kelas_citra(p)].append(p)

    baris = []
    rekap = defaultdict(Counter)

    # Tahap 1: hitung statistik. Kalau --semua, seluruh test set dipakai.
    kandidat = semua if (args.semua or args.terparah) else None
    if kandidat is None:
        rng = random.Random(args.seed)
        kandidat = []
        for k in sorted(per_kelas):
            if k < 0:
                continue
            grup = per_kelas[k][:]
            rng.shuffle(grup)
            kandidat.extend(grup[:args.sampel])

    print(f"menjalankan inferensi pada {len(kandidat)} citra...")
    simpanan = {}
    for n, p in enumerate(kandidat, 1):
        citra = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if citra is None:
            print(f"  ! gagal baca {p.name}")
            continue
        h, w = citra.shape[:2]

        label = muat_label(dir_lbl / f"{p.stem}.txt", w, h)
        r = model.predict(str(p), conf=CONF, imgsz=args.imgsz, verbose=False)[0]
        prediksi = [(int(c), *map(float, xyxy), float(cf))
                    for c, xyxy, cf in zip(r.boxes.cls.tolist(),
                                           r.boxes.xyxy.tolist(),
                                           r.boxes.conf.tolist())]

        vonis, fn = cocokkan(label, prediksi)
        c = Counter(vonis)
        kf = NAMA_KELAS.get(kelas_citra(p), "tak_berlabel")

        baris.append({
            "citra": p.name, "kelas": kf,
            "n_label": len(label), "n_prediksi": len(prediksi),
            "TP": c["TP"], "FP_kelas": c["FP_kelas"],
            "FP_latar": c["FP_latar"], "FN": len(fn),
        })
        rekap[kf].update({"citra": 1, "n_label": len(label),
                          "n_prediksi": len(prediksi), "TP": c["TP"],
                          "FP_kelas": c["FP_kelas"], "FP_latar": c["FP_latar"],
                          "FN": len(fn)})
        simpanan[p] = (citra, label, prediksi, vonis, fn, kf)

        if n % 25 == 0:
            print(f"  {n}/{len(kandidat)}")

    # Tahap 2: render. Kalau --terparah, ambil yang FP_latar-nya terbanyak.
    urut = sorted(baris, key=lambda b: -b["FP_latar"]) if args.terparah else baris
    batas = args.sampel * len(per_kelas) if (args.semua or args.terparah) else len(urut)
    dirender = {b["citra"] for b in urut[:batas]}

    KELUARAN.mkdir(parents=True, exist_ok=True)
    n_render = 0
    for p, (citra, label, prediksi, vonis, fn, kf) in simpanan.items():
        if p.name not in dirender:
            continue
        tujuan = KELUARAN / kf
        tujuan.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tujuan / f"{p.stem}.jpg"),
                    bersisian(citra, label, prediksi, vonis, fn, p.name, kf),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        n_render += 1

    # Tahap 3: tulis CSV
    with open(KELUARAN / "ringkasan.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(baris[0]))
        wtr.writeheader()
        wtr.writerows(sorted(baris, key=lambda b: -b["FP_latar"]))

    kolom = ["citra", "n_label", "n_prediksi", "TP", "FP_kelas", "FP_latar", "FN"]
    with open(KELUARAN / "rekap.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["kelas"] + kolom + ["FP_latar/citra", "recall", "presisi"])
        for kf in sorted(rekap):
            r = rekap[kf]
            tp, fp = r["TP"], r["FP_kelas"] + r["FP_latar"]
            wtr.writerow([kf] + [r[k] for k in kolom] + [
                round(r["FP_latar"] / max(1, r["citra"]), 2),
                round(tp / max(1, tp + r["FN"]), 4),
                round(tp / max(1, tp + fp), 4),
            ])

    # Ringkasan layar
    print(f"\n{'kelas':22s} {'citra':>6s} {'label':>6s} {'pred':>6s} "
          f"{'TP':>5s} {'FPkls':>6s} {'FPltr':>6s} {'FN':>5s}  FPltr/citra")
    for kf in sorted(rekap):
        r = rekap[kf]
        print(f"{kf:22s} {r['citra']:6d} {r['n_label']:6d} {r['n_prediksi']:6d} "
              f"{r['TP']:5d} {r['FP_kelas']:6d} {r['FP_latar']:6d} {r['FN']:5d}  "
              f"{r['FP_latar'] / max(1, r['citra']):.2f}")

    print(f"\n{n_render} gambar ditulis ke {KELUARAN}")
    print("""
Cara membacanya:

  Buka folder leaf_spot_disease dan insect_pest_disease lebih dulu.
  Lihat kotak MERAH (FP_latar) di panel kanan, lalu cek titik yang sama
  di panel kiri.

    ada bercak/gigitan nyata di situ  -> LABEL yang kurang, model benar.
                                         Perbaikannya di autolabel.py.
    tidak ada apa-apa di situ         -> model memang mengarang.
                                         Perbaikannya di training.

  Kotak ORANYE (FP_kelas) = posisi benar tapi tertukar leaf_spot <-> insect.
  Kalau ini banyak, yang perlu ditinjau adalah aturan pembeda halo di
  autolabel.py, bukan modelnya.

  Kotak BIRU di panel kiri = label yang tidak terdeteksi sama sekali.
""")


# ======================================================================
# SELF-CHECK
# ======================================================================
def uji():
    lolos = True

    def cek(nama, syarat):
        nonlocal lolos
        print(f"  {'ok  ' if syarat else 'GAGAL'}  {nama}")
        lolos = lolos and syarat

    print("iou()")
    cek("kotak identik = 1.0", abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9)
    cek("tidak beririsan = 0.0", iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0)
    cek("bersentuhan tepi = 0.0", iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0)
    cek("tumpang setengah = 1/3",
        abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9)
    cek("kotak nol luas aman", iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0)

    print("muat_label()")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("2 0.5 0.5 0.5 0.5\n\n3 0.25 0.25 0.5 0.5\n")
        jalur = f.name
    kotak = muat_label(jalur, 100, 100)
    cek("dua baris terbaca, baris kosong dilewati", len(kotak) == 2)
    cek("koordinat tengah benar", kotak[0] == (2, 25.0, 25.0, 75.0, 75.0))
    cek("berkas tidak ada = daftar kosong", muat_label("/tidak/ada.txt", 10, 10) == [])

    print("cocokkan()")
    L = [(2, 0, 0, 10, 10), (3, 50, 50, 60, 60)]

    v, fn = cocokkan(L, [(2, 0, 0, 10, 10, 0.9)])
    cek("cocok + kelas benar = TP", v == ["TP"] and fn == [1])

    v, fn = cocokkan(L, [(3, 0, 0, 10, 10, 0.9)])
    cek("cocok + kelas salah = FP_kelas", v == ["FP_kelas"] and fn == [1])

    v, fn = cocokkan(L, [(2, 200, 200, 210, 210, 0.9)])
    cek("jauh dari label = FP_latar", v == ["FP_latar"] and fn == [0, 1])

    v, fn = cocokkan(L, [(2, 0, 0, 10, 10, 0.9), (2, 0, 0, 10, 10, 0.8)])
    cek("deteksi kembar: satu TP satu FP_latar",
        v == ["TP", "FP_latar"] and fn == [1])

    v, fn = cocokkan(L, [(2, 0, 0, 10, 10, 0.3), (2, 1, 1, 11, 11, 0.95)])
    cek("conf tertinggi dapat label lebih dulu",
        v[1] == "TP" and v[0] == "FP_latar")

    v, fn = cocokkan(L, [])
    cek("tanpa prediksi = semua FN", v == [] and fn == [0, 1])

    v, fn = cocokkan([], [(2, 0, 0, 10, 10, 0.9)])
    cek("tanpa label = semua FP_latar", v == ["FP_latar"] and fn == [])

    v, fn = cocokkan(L, [(2, 0, 0, 7, 7, 0.9)])
    cek("IoU 0.49 di bawah ambang = FP_latar", v == ["FP_latar"])

    print("bersisian()")
    citra = np.full((400, 300, 3), 120, np.uint8)
    L2 = [(2, 10, 10, 100, 100), (3, 150, 150, 200, 200)]
    P2 = [(2, 12, 12, 98, 98, 0.88), (3, 250, 250, 280, 280, 0.41)]
    v2, fn2 = cocokkan(L2, P2)
    keluar = bersisian(citra, L2, P2, v2, fn2, "uji.jpg", "leaf_spot_disease")
    cek("keluaran 3 kanal", keluar.ndim == 3 and keluar.shape[2] == 3)
    cek("lebar = 2 panel + pemisah", keluar.shape[1] == 2 * round(300 * (SISI_TAMPIL / 400)) + 6)
    cek("tinggi = panel + 2 pita judul", keluar.shape[0] == SISI_TAMPIL + 34 + 30)

    keluar = bersisian(citra, [], [], [], [], "kosong.jpg", "healthy_leaf")
    cek("citra tanpa kotak tidak error", keluar.shape[0] > 0)

    print("\n" + ("SEMUA UJI LOLOS" if lolos else "ADA YANG GAGAL"))
    return 0 if lolos else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="80_10_10", choices=["80_10_10", "70_15_15"])
    p.add_argument("--model", default=None,
                   help="yolo11n|yolo11s|yolo11m; default = terpilih tiap split")
    p.add_argument("--sampel", type=int, default=10, help="citra per kelas (default 10)")
    p.add_argument("--semua", action="store_true",
                   help="inferensi seluruh test set untuk statistik")
    p.add_argument("--terparah", action="store_true",
                   help="render citra dengan FP_latar terbanyak (implisit --semua)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="harus sama dengan imgsz saat training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--uji", action="store_true", help="self-check lalu keluar")
    args = p.parse_args()

    if args.uji:
        sys.exit(uji())
    jalankan(args)


if __name__ == "__main__":
    main()
