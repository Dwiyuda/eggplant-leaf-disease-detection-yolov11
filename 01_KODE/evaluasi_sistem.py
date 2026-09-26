#!/usr/bin/env python3
"""
Evaluasi sistem dari sudut pandang PEMAKAI, bukan dari sudut pandang kotak.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

mAP menjawab "seberapa tepat kotaknya". Pemakai bertanya "daun ini sakit apa".
Dua pertanyaan berbeda, dan selama ini kita cuma mengukur yang pertama.

Tiga hal dikerjakan sekaligus:

  1. AKURASI TINGKAT-CITRA
     Deteksi digabung jadi satu tebakan penyakit per citra, lalu dibandingkan
     dengan kelas sebenarnya. Menghasilkan akurasi, macro-F1, dan confusion
     matrix -- angka yang bisa dibandingkan dengan penelitian klasifikasi.

  2. AMBANG CONF TERBAIK
     Disapu dari 0,05 sampai 0,95. Dipilih dari VAL, dilaporkan di TEST.

  3. TTA (Test-Time Augmentation)
     Citra diprediksi dalam beberapa versi (skala + cermin) lalu digabung.
     Biasanya menaikkan mAP 1-2% tanpa training ulang.

PROTOKOL. Aturan penggabungan dan ambang conf dipilih dari subset VAL.
Subset TEST cuma dipakai sekali di akhir untuk melaporkan. Memilih dari test
sama saja membocorkan data uji ke proses pemilihan.

CATATAN. Label kelas di sini diambil dari label YOLO yang ada, yang kita tahu
15-19% keliru. Jadi angka di bawah adalah "kesepakatan dengan kunci jawaban
yang sebagian salah", bukan kemampuan sebenarnya.

Pakai:
    python evaluasi_sistem.py --uji            # self-check, tanpa ultralytics
    python evaluasi_sistem.py                  # model terbaik, val + test
    python evaluasi_sistem.py --tanpa-tta      # lewati TTA (lebih cepat)
    python evaluasi_sistem.py --bobot <path.pt> --imgsz 640
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parent.parent   # skrip ada di 01_KODE/
DIR_DATASET = AKAR / "02_DATASET"
DIR_SUMBER  = AKAR / "03_CITRA_SUMBER"
DIR_KAGGLE  = AKAR / "04_KAGGLE" / "FOLDER_HASIL_KAGGLE"
DIR_HASIL   = AKAR / "05_HASIL"
HASIL = DIR_KAGGLE
KELUARAN = DIR_HASIL / "evaluasi_hasil"

NAMA = {0: "healthy_leaf", 1: "mosaic_virus_disease",
        2: "leaf_spot_disease", 3: "insect_pest_disease"}

# Model terbaik menurut val mAP50-95 (0,5726) dari ablasi resolusi.
BOBOT_DEFAULT = "hasil_yolo11_ablasi_resolusi/img1024/best_yolo11s_img1024.pt"
IMGSZ_DEFAULT = 1024
SPLIT_DIR = "dataset_80_10_10"

AMBANG = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]


# ======================================================================
# ATURAN PENGGABUNGAN  --  deteksi (banyak) -> satu tebakan per citra
# ======================================================================
# Empat aturan diuji karena tidak ada satu pun yang jelas benar di muka.
# healthy dan mosaic punya 1 kotak per citra; leaf_spot dan insect punya
# belasan. Aturan yang menghitung jumlah kotak akan condong ke kelas lesi,
# aturan yang melihat conf tertinggi tidak. Biar data yang memilih.

def gab_conf_maks(det):
    """Kelas milik deteksi ber-conf tertinggi."""
    return max(det, key=lambda d: d[1])[0]


def gab_jumlah(det):
    """Kelas dengan kotak terbanyak; seri dipecah oleh conf tertinggi."""
    c = Counter(k for k, _ in det)
    n = max(c.values())
    kandidat = {k for k, v in c.items() if v == n}
    return max((d for d in det if d[0] in kandidat), key=lambda d: d[1])[0]


def gab_jumlah_conf(det):
    """Kelas dengan total conf terbesar."""
    s = defaultdict(float)
    for k, c in det:
        s[k] += c
    return max(s, key=s.get)


def gab_rata_conf(det):
    """Kelas dengan rata-rata conf tertinggi."""
    s, n = defaultdict(float), Counter()
    for k, c in det:
        s[k] += c
        n[k] += 1
    return max(s, key=lambda k: s[k] / n[k])


# --- Dua aturan tambahan, ditambahkan 2026-08-15 ---
#
# Empat aturan di atas mengabaikan struktur masalahnya. Diukur dari data latih,
# jumlah kotak per citra: healthy 1, mosaic 1, leaf_spot 8, insect 8.
#
# Akibatnya 'jumlah_conf' condong berat ke kelas lesi: daun leaf_spot dengan
# 8 kotak berkepercayaan 0,4 mendapat total 3,2, sedangkan daun healthy dengan
# 1 kotak berkepercayaan 0,9 cuma 0,9. healthy kalah bukan karena model ragu,
# tapi karena kotaknya memang cuma satu.
#
# Terlihat pada hasil 2026-08-11: insect recall 0,909 tapi presisi 0,694
# (kebanyakan menebak insect), healthy justru sebaliknya.

KOTAK_KHAS = {0: 1, 1: 1, 2: 8, 3: 8}   # median kotak per citra, dari data latih


def gab_ternormalisasi(det):
    """Total conf dibagi jumlah kotak khas kelasnya, supaya adil."""
    s = defaultdict(float)
    for k, c in det:
        s[k] += c
    return max(s, key=lambda k: s[k] / KOTAK_KHAS.get(k, 1))


def gab_bertingkat(det):
    """
    Dua langkah, mengikuti bentuk asli masalahnya.

    1. Kondisi daun utuh (healthy/mosaic) atau lesi (leaf_spot/insect)?
       Diputuskan dari conf TERTINGGI tiap kelompok, bukan jumlah -- supaya
       jumlah kotak tidak ikut menentukan.
    2. Di dalam kelompok yang menang, pilih kelas ber-conf tertinggi.
    """
    utuh = [d for d in det if d[0] in (0, 1)]
    lesi = [d for d in det if d[0] in (2, 3)]
    if not utuh:
        return max(lesi, key=lambda d: d[1])[0]
    if not lesi:
        return max(utuh, key=lambda d: d[1])[0]
    kelompok = utuh if max(d[1] for d in utuh) >= max(d[1] for d in lesi) else lesi
    return max(kelompok, key=lambda d: d[1])[0]


ATURAN = {"conf_maks": gab_conf_maks, "jumlah_kotak": gab_jumlah,
          "jumlah_conf": gab_jumlah_conf, "rata_conf": gab_rata_conf,
          "ternormalisasi": gab_ternormalisasi, "bertingkat": gab_bertingkat}


# ======================================================================
# METRIK
# ======================================================================
def hitung(benar, tebak, kelas=tuple(NAMA)):
    """Akurasi, makro-F1, dan metrik per kelas. tebak == -1 berarti tanpa deteksi."""
    n = len(benar)
    akurasi = sum(1 for b, t in zip(benar, tebak) if b == t) / n if n else 0.0
    per = {}
    f1s = []
    for k in kelas:
        tp = sum(1 for b, t in zip(benar, tebak) if b == k and t == k)
        fp = sum(1 for b, t in zip(benar, tebak) if b != k and t == k)
        fn = sum(1 for b, t in zip(benar, tebak) if b == k and t != k)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per[NAMA[k]] = {"precision": p, "recall": r, "f1": f1, "dukungan": tp + fn}
        f1s.append(f1)
    return {"akurasi": akurasi, "makro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "tanpa_deteksi": sum(1 for t in tebak if t == -1), "n": n, "per_kelas": per}


def matriks(benar, tebak):
    m = np.zeros((len(NAMA), len(NAMA) + 1), int)   # kolom terakhir = tanpa deteksi
    for b, t in zip(benar, tebak):
        m[b, len(NAMA) if t == -1 else t] += 1
    return m


def cetak_matriks(m):
    lbl = [NAMA[i][:14] for i in sorted(NAMA)]
    judul = "sebenarnya | tebakan"
    print("\n" + f"{judul:24s} " + " ".join(f"{l:>14s}" for l in lbl) + f" {'(kosong)':>9s}")
    for i in sorted(NAMA):
        print(f"{NAMA[i]:24s} " + " ".join(f"{m[i, j]:14d}" for j in sorted(NAMA))
              + f" {m[i, len(NAMA)]:9d}")


# ======================================================================
# ALUR UTAMA
# ======================================================================
def kelas_asli(dir_lbl, stem):
    t = dir_lbl / f"{stem}.txt"
    if not t.exists():
        return -1
    k = [int(float(b.split()[0])) for b in t.read_text().splitlines() if b.strip()]
    return Counter(k).most_common(1)[0][0] if k else -1


def kumpulkan(model, dir_img, dir_lbl, imgsz, tta, label=""):
    """Satu kali inferensi pada conf rendah; ambang disaring belakangan."""
    berkas = sorted(p for p in dir_img.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    benar, deteksi = [], []
    for i, p in enumerate(berkas, 1):
        k = kelas_asli(dir_lbl, p.stem)
        if k < 0:
            continue
        r = model.predict(str(p), conf=0.01, imgsz=imgsz, augment=tta, verbose=False)[0]
        benar.append(k)
        deteksi.append(list(zip((int(c) for c in r.boxes.cls.tolist()),
                                (float(c) for c in r.boxes.conf.tolist()))))
        if i % 50 == 0:
            print(f"    {label} {i}/{len(berkas)}")
    return benar, deteksi


def terapkan(deteksi, ambang, aturan):
    keluar = []
    for det in deteksi:
        d = [x for x in det if x[1] >= ambang]
        keluar.append(ATURAN[aturan](d) if d else -1)
    return keluar


def jalankan(args):
    from ultralytics import YOLO

    bobot = Path(args.bobot) if args.bobot else HASIL / BOBOT_DEFAULT
    if not bobot.exists():
        sys.exit(f"Bobot tidak ketemu: {bobot}")
    akar = DIR_DATASET / SPLIT_DIR
    KELUARAN.mkdir(parents=True, exist_ok=True)

    print(f"bobot : {bobot.name}")
    print(f"imgsz : {args.imgsz}")
    print(f"TTA   : {'ya' if not args.tanpa_tta else 'tidak'}\n")
    model = YOLO(str(bobot))

    # ---------- inferensi sekali per subset ----------
    data = {}
    for sp in ["val", "test"]:
        print(f"inferensi {sp}...")
        data[sp] = kumpulkan(model, akar / "images" / sp, akar / "labels" / sp,
                             args.imgsz, False, sp)
        print(f"  {len(data[sp][0])} citra")

    # ---------- 1+2. pilih aturan & ambang DARI VAL ----------
    print("\n" + "=" * 78)
    print("  MEMILIH ATURAN PENGGABUNGAN DAN AMBANG conf  --  dari VAL")
    print("=" * 78)
    bv, dv = data["val"]
    rekap = []
    for at in ATURAN:
        for am in AMBANG:
            h = hitung(bv, terapkan(dv, am, at))
            rekap.append({"aturan": at, "conf": am, "akurasi": h["akurasi"],
                          "makro_f1": h["makro_f1"], "tanpa_deteksi": h["tanpa_deteksi"]})
    rekap.sort(key=lambda r: -r["makro_f1"])
    terbaik = rekap[0]

    print(f"\n{'aturan':16s} {'conf':>5s} {'akurasi':>8s} {'makro-F1':>9s} {'tanpa deteksi':>14s}")
    for r in rekap[:10]:
        print(f"{r['aturan']:16s} {r['conf']:5.2f} {r['akurasi']:8.4f} "
              f"{r['makro_f1']:9.4f} {r['tanpa_deteksi']:14d}")
    print(f"\nTERPILIH: aturan '{terbaik['aturan']}' pada conf {terbaik['conf']:.2f}")
    print(f"          val akurasi {terbaik['akurasi']:.4f} | makro-F1 {terbaik['makro_f1']:.4f}")

    # kurva ambang untuk aturan terpilih -- bahan grafik Bab IV
    kurva = [r for r in rekap if r["aturan"] == terbaik["aturan"]]
    kurva.sort(key=lambda r: r["conf"])
    with open(KELUARAN / "kurva_ambang_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kurva[0]))
        w.writeheader(); w.writerows(kurva)

    # ---------- 3. laporan TEST, sekali ----------
    print("\n" + "=" * 78)
    print(f"  AKURASI TINGKAT-CITRA pada TEST  (aturan {terbaik['aturan']}, "
          f"conf {terbaik['conf']:.2f})")
    print("=" * 78)
    bt, dt = data["test"]
    tebak = terapkan(dt, terbaik["conf"], terbaik["aturan"])
    ht = hitung(bt, tebak)
    print(f"\n  AKURASI  : {ht['akurasi']:.4f}   ({int(ht['akurasi']*ht['n'])} dari {ht['n']} citra)")
    print(f"  makro-F1 : {ht['makro_f1']:.4f}")
    print(f"  tanpa deteksi sama sekali : {ht['tanpa_deteksi']} citra")
    print(f"\n{'kelas':24s} {'presisi':>8s} {'recall':>8s} {'F1':>8s} {'jumlah':>7s}")
    for k, v in ht["per_kelas"].items():
        print(f"{k:24s} {v['precision']:8.4f} {v['recall']:8.4f} {v['f1']:8.4f} {v['dukungan']:7d}")
    cetak_matriks(matriks(bt, tebak))

    with open(KELUARAN / "per_citra_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kelas_sebenarnya", "tebakan", "benar"])
        for b, t in zip(bt, tebak):
            w.writerow([NAMA[b], NAMA.get(t, "(tanpa deteksi)"), int(b == t)])

    # ---------- TTA ----------
    tta = None
    if not args.tanpa_tta:
        print("\n" + "=" * 78)
        print("  TTA  --  dibandingkan pada VAL")
        print("=" * 78)
        m0 = model.val(data=str(akar / "data.yaml"), split="val", imgsz=args.imgsz,
                       augment=False, verbose=False, plots=False)
        m1 = model.val(data=str(akar / "data.yaml"), split="val", imgsz=args.imgsz,
                       augment=True, verbose=False, plots=False)
        tta = {"tanpa_tta": {"mAP50": float(m0.box.map50), "mAP50_95": float(m0.box.map)},
               "dengan_tta": {"mAP50": float(m1.box.map50), "mAP50_95": float(m1.box.map)}}
        print(f"\n{'':12s} {'mAP50':>9s} {'mAP50-95':>10s}")
        print(f"{'tanpa TTA':12s} {tta['tanpa_tta']['mAP50']:9.4f} {tta['tanpa_tta']['mAP50_95']:10.4f}")
        print(f"{'dengan TTA':12s} {tta['dengan_tta']['mAP50']:9.4f} {tta['dengan_tta']['mAP50_95']:10.4f}")
        d = tta["dengan_tta"]["mAP50_95"] - tta["tanpa_tta"]["mAP50_95"]
        print(f"{'selisih':12s} {tta['dengan_tta']['mAP50']-tta['tanpa_tta']['mAP50']:+9.4f} {d:+10.4f}")
        print("\n=> TTA menolong, pakai saat inferensi." if d > 0.002 else
              "\n=> TTA tidak menolong berarti. Lewati, tidak sepadan 3x waktu inferensi.")

    (KELUARAN / "ringkasan.json").write_text(json.dumps({
        "bobot": bobot.name, "imgsz": args.imgsz,
        "dipilih_dari_val": terbaik, "test": ht, "tta_pada_val": tta,
        "catatan": "Label 15-19% keliru; angka ini kesepakatan dengan kunci jawaban "
                   "yang sebagian salah, bukan kemampuan sebenarnya.",
    }, indent=2, ensure_ascii=False))
    print(f"\nhasil ditulis ke {KELUARAN}")
    print("""
Cara membaca:

  AKURASI TINGKAT-CITRA menjawab "berapa persen daun yang penyakitnya
  ditebak benar" -- setara dengan akurasi pada penelitian klasifikasi,
  dan bisa dibandingkan langsung dengan mereka.

  mAP tetap dilaporkan sebagai metrik utama karena ini penelitian deteksi.
  Akurasi tingkat-citra melengkapinya, bukan menggantikannya.

  INGAT: 15-19% label dataset ini keliru. Angka di atas adalah kesepakatan
  dengan kunci jawaban yang sebagian salah.
""")


# ======================================================================
# SELF-CHECK
# ======================================================================
def uji():
    lolos = True

    def cek(n, s):
        nonlocal lolos
        print(f"  {'ok  ' if s else 'GAGAL'}  {n}")
        lolos = lolos and s

    print("aturan penggabungan")
    det = [(2, 0.9), (3, 0.4), (3, 0.5), (3, 0.45)]
    cek("conf_maks pilih conf tertinggi", gab_conf_maks(det) == 2)
    cek("jumlah_kotak pilih terbanyak", gab_jumlah(det) == 3)
    cek("jumlah_conf jumlahkan conf", gab_jumlah_conf(det) == 3)      # 1.35 > 0.9
    cek("rata_conf pakai rata-rata", gab_rata_conf(det) == 2)         # 0.9 > 0.45
    cek("satu deteksi", gab_conf_maks([(1, 0.3)]) == 1)
    d2 = [(0, 0.8), (1, 0.8)]
    cek("seri jumlah dipecah conf", gab_jumlah(d2) in (0, 1))

    print("aturan ternormalisasi")
    # 1 kotak healthy 0,9 -> 0,9/1 = 0,90 ; 8 kotak leaf_spot 0,4 -> 3,2/8 = 0,40
    dn = [(0, 0.9)] + [(2, 0.4)] * 8
    cek("jumlah_conf kalah oleh banyak kotak", gab_jumlah_conf(dn) == 2)
    cek("ternormalisasi membetulkannya", gab_ternormalisasi(dn) == 0)
    dn2 = [(0, 0.3)] + [(2, 0.6)] * 8      # 0,30 vs 0,60 -> lesi memang menang
    cek("lesi tetap menang kalau memang kuat", gab_ternormalisasi(dn2) == 2)

    print("aturan bertingkat")
    cek("tanpa kelas utuh -> pilih lesi", gab_bertingkat([(2, 0.4), (3, 0.6)]) == 3)
    cek("tanpa kelas lesi -> pilih utuh", gab_bertingkat([(0, 0.4), (1, 0.6)]) == 1)
    db = [(0, 0.9)] + [(2, 0.4)] * 8
    cek("utuh menang karena conf tertinggi", gab_bertingkat(db) == 0)
    db2 = [(0, 0.3)] + [(2, 0.8)] * 3
    cek("lesi menang karena conf tertinggi", gab_bertingkat(db2) == 2)
    cek("dalam kelompok pilih conf tertinggi",
        gab_bertingkat([(0, 0.5), (1, 0.7), (2, 0.2)]) == 1)
    cek("enam aturan terdaftar", len(ATURAN) == 6)

    print("terapkan()")
    dd = [[(2, 0.9), (3, 0.1)], [(1, 0.2)], []]
    cek("ambang menyaring", terapkan(dd, 0.5, "conf_maks") == [2, -1, -1])
    cek("ambang rendah meloloskan", terapkan(dd, 0.05, "conf_maks") == [2, 1, -1])

    print("hitung()")
    h = hitung([0, 0, 1, 1], [0, 0, 1, 1])
    cek("sempurna = akurasi 1", abs(h["akurasi"] - 1.0) < 1e-9)
    cek("sempurna = makro-F1 1 pada kelas terpakai",
        abs(h["per_kelas"]["healthy_leaf"]["f1"] - 1.0) < 1e-9)
    h = hitung([0, 0, 1, 1], [0, 1, 0, 1])
    cek("setengah benar", abs(h["akurasi"] - 0.5) < 1e-9)
    h = hitung([0, 1], [-1, -1])
    cek("tanpa deteksi dihitung", h["tanpa_deteksi"] == 2 and h["akurasi"] == 0.0)
    h = hitung([2, 2, 2], [2, 2, 3])
    cek("recall kelas 2 = 2/3", abs(h["per_kelas"]["leaf_spot_disease"]["recall"] - 2/3) < 1e-9)
    cek("dukungan benar", h["per_kelas"]["leaf_spot_disease"]["dukungan"] == 3)

    print("matriks()")
    m = matriks([0, 0, 1], [0, 1, -1])
    cek("diagonal", m[0, 0] == 1)
    cek("salah kelas", m[0, 1] == 1)
    cek("kolom tanpa deteksi", m[1, 4] == 1)
    cek("bentuk 4x5", m.shape == (4, 5))

    print("\n" + ("SEMUA UJI LOLOS" if lolos else "ADA YANG GAGAL"))
    return 0 if lolos else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bobot", default=None, help="path .pt; default = model terbaik")
    p.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT,
                   help="harus sama dengan imgsz saat training")
    p.add_argument("--tanpa-tta", action="store_true", help="lewati bagian TTA")
    p.add_argument("--uji", action="store_true", help="self-check lalu keluar")
    a = p.parse_args()
    if a.uji:
        sys.exit(uji())
    jalankan(a)


if __name__ == "__main__":
    main()
