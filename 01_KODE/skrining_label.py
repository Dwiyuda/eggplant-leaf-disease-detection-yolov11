#!/usr/bin/env python3
"""
Skrining label pakai model: cari citra yang kelas folder-nya bertentangan
dengan apa yang dilihat model, lalu susun antrian tinjauan terurut.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

Latar: audit 2026-08-07 menemukan citra di folder healthy_leaf yang ternyata
berlubang. Model mendeteksinya sebagai lesi lalu dihukum, padahal deteksinya
benar. Skrip ini mengukur seberapa luas masalah itu.

MODEL HANYA MENGUSULKAN. Keputusan memindahkan atau membuang citra tetap di
tangan manusia, berdasarkan bukti visual -- bukan berdasarkan skor. Ini
melanjutkan praktik tahap 4 pipeline, waktu autolabel.py mengusulkan lewat
perlu_ditinjau.csv dan 19 citra dipindah setelah ditinjau mata.

Empat pola yang dicari:

    healthy_berlubang   folder healthy/mosaic, tapi model melihat lesi
                        -> kandidat kontaminasi kelas sehat
    lesi_tampak_utuh    folder leaf_spot/insect, tapi model melihat daun utuh
                        -> kandidat gejala menyebar yang salah mode anotasi
    tertukar_utuh       healthy <-> mosaic
    tertukar_lesi       leaf_spot <-> insect

Catatan penting soal subset:
    Citra yang dipakai MELATIH model sudah dioptimasi supaya model setuju
    dengan labelnya. Jadi kalau model tetap tidak setuju di citra train, itu
    bukti yang JAUH lebih kuat daripada ketidaksetujuan di val/test.
    Sebaliknya, sedikitnya temuan di train tidak berarti train bersih.
    Kolom `subset` di antrian.csv menandai ini.

Pakai:
    python skrining_label.py --uji           # self-check, tanpa ultralytics
    python skrining_label.py                 # seluruh 2.089 citra
    python skrining_label.py --sampel 100    # uji cepat
    python skrining_label.py --model yolo11s # lebih cepat, sedikit lebih lemah

Prasyarat:
    pip install ultralytics
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from audit_prediksi import (NAMA_KELAS, MODEL_TERPILIH, cari_bobot,
                            muat_label, _skala, _kotak, _judul, WARNA)

from audit_prediksi import AKAR, DIR_DATASET, DIR_HASIL
KELUARAN = DIR_HASIL / "skrining_hasil"

UTUH = {0, 1}          # healthy_leaf, mosaic_virus_disease -- satu kotak seluruh daun
LESI = {2, 3}          # leaf_spot_disease, insect_pest_disease -- banyak kotak

CONF = 0.25
LUAS_UTUH = 0.25       # deteksi >25% luas citra dianggap "kotak daun utuh"

POLA = ["healthy_berlubang", "lesi_tampak_utuh", "tertukar_utuh", "tertukar_lesi"]


# ======================================================================
# PENILAIAN
# ======================================================================
def nilai(kelas_folder, prediksi, luas_citra):
    """
    prediksi: [(kelas, x1, y1, x2, y2, conf)]

    Kembalian: (skor, pola, rincian)
        skor  jumlah conf deteksi yang bertentangan dengan kelas folder
        pola  nama pola terkuat, atau None kalau tidak ada pertentangan
    """
    asing = [p for p in prediksi if p[0] != kelas_folder]
    if not asing:
        return 0.0, None, Counter()

    rincian = Counter()
    bobot = defaultdict(float)

    for k, x1, y1, x2, y2, conf in asing:
        frac = ((x2 - x1) * (y2 - y1)) / luas_citra if luas_citra > 0 else 0.0
        rincian[NAMA_KELAS.get(k, k)] += 1

        if kelas_folder in UTUH and k in LESI:
            bobot["healthy_berlubang"] += conf
        elif kelas_folder in LESI and k in UTUH and frac >= LUAS_UTUH:
            bobot["lesi_tampak_utuh"] += conf
        elif kelas_folder in UTUH and k in UTUH:
            bobot["tertukar_utuh"] += conf
        elif kelas_folder in LESI and k in LESI:
            bobot["tertukar_lesi"] += conf

    if not bobot:
        return 0.0, None, rincian
    pola = max(bobot, key=bobot.get)
    return round(bobot[pola], 4), pola, rincian


# ======================================================================
# GAMBAR
# ======================================================================
def kartu(citra, label, prediksi, kelas_folder, nama, pola, skor):
    """Satu panel: label tipis, deteksi tebal. Yang bertentangan diwarnai merah."""
    dasar, s = _skala(citra, 900)

    for k, x1, y1, x2, y2 in label:
        _kotak(dasar, x1 * s, y1 * s, x2 * s, y2 * s, WARNA["label"],
               f"label: {NAMA_KELAS.get(k, k)}", 1)

    for k, x1, y1, x2, y2, conf in prediksi:
        bentrok = k != kelas_folder
        _kotak(dasar, x1 * s, y1 * s, x2 * s, y2 * s,
               WARNA["FP_latar"] if bentrok else WARNA["TP"],
               f"{NAMA_KELAS.get(k, k)} {conf:.2f}", 3 if bentrok else 2)

    dasar = _judul(dasar, f"folder: {NAMA_KELAS.get(kelas_folder, kelas_folder)}"
                          f"   |   {pola}  (skor {skor:.2f})", 34)
    return _judul(dasar, nama, 30)


# ======================================================================
# ALUR UTAMA
# ======================================================================
def jalankan(args):
    from ultralytics import YOLO

    # Run sampel menulis ke folder terpisah. Kalau tidak, menjalankan --sampel
    # setelah run lengkap akan menimpa antrian.csv dan gambar bernomornya.
    keluaran = Path(str(KELUARAN) + "_sampel") if args.sampel else KELUARAN

    split = args.split
    model_tag = args.model or MODEL_TERPILIH[split]
    bobot = cari_bobot(split, model_tag)
    if bobot is None:
        sys.exit(f"Bobot 'best_{model_tag}_{split}.pt' tidak ketemu.")

    akar_data = DIR_DATASET / f"dataset_{split}"
    if not akar_data.is_dir():
        sys.exit(f"Folder tidak ada: {akar_data}")

    # Kumpulkan seluruh citra dari ketiga subset, sekalian catat subsetnya.
    tugas = []
    for sub in ["train", "val", "test"]:
        di, dl = akar_data / "images" / sub, akar_data / "labels" / sub
        if not di.is_dir():
            continue
        for p in sorted(di.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                tugas.append((p, dl / f"{p.stem}.txt", sub))

    if args.sampel:
        tugas = tugas[::max(1, len(tugas) // args.sampel)][:args.sampel]

    print(f"bobot  : {bobot.name}")
    print(f"citra  : {len(tugas)}")
    print(f"ambang : conf={CONF}\n")
    print("CATATAN: citra 'train' dipakai melatih model ini, jadi model sudah")
    print("dioptimasi untuk setuju dengan labelnya. Ketidaksetujuan di train")
    print("adalah bukti yang lebih kuat, bukan lebih lemah.\n")

    model = YOLO(str(bobot))
    baris = []
    rekap = defaultdict(Counter)
    simpanan = {}

    for n, (p, t, sub) in enumerate(tugas, 1):
        citra = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if citra is None:
            print(f"  ! gagal baca {p.name}")
            continue
        h, w = citra.shape[:2]

        label = muat_label(t, w, h)
        if not label:
            continue
        kelas_folder = Counter(b[0] for b in label).most_common(1)[0][0]

        r = model.predict(str(p), conf=CONF, imgsz=args.imgsz, verbose=False)[0]
        prediksi = [(int(c), *map(float, xy), float(cf))
                    for c, xy, cf in zip(r.boxes.cls.tolist(),
                                         r.boxes.xyxy.tolist(),
                                         r.boxes.conf.tolist())]

        skor, pola, rincian = nilai(kelas_folder, prediksi, float(w * h))
        kf = NAMA_KELAS.get(kelas_folder, str(kelas_folder))

        rekap[kf]["citra"] += 1
        if pola:
            rekap[kf][pola] += 1
            baris.append({
                "citra": p.name, "subset": sub, "kelas_folder": kf,
                "pola": pola, "skor": skor,
                "n_label": len(label), "n_prediksi": len(prediksi),
                "deteksi_asing": "; ".join(f"{k}x{v}" for k, v in rincian.items()),
            })
            simpanan[p.name] = (citra, label, prediksi, kf, pola, skor, p)

        if n % 100 == 0:
            print(f"  {n}/{len(tugas)}  ({len(baris)} bertanda)")

    if not baris:
        print("\nTidak ada pertentangan ditemukan.")
        return

    baris.sort(key=lambda b: -b["skor"])

    keluaran.mkdir(parents=True, exist_ok=True)
    with open(keluaran / "antrian.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(baris[0]) + ["vonis"])
        wtr.writeheader()
        wtr.writerows([{**b, "vonis": ""} for b in baris])

    # Render per pola. Dua mode, dan pilihannya menentukan apa yang boleh
    # disimpulkan dari hasil tinjauan:
    #   default  N teratas -- untuk memastikan polanya nyata. TIDAK boleh
    #            dipakai memperkirakan persentase, karena sengaja bias ke
    #            kasus yang paling meyakinkan.
    #   --acak   sampel acak dari yang bertanda -- boleh dipakai
    #            memperkirakan persentase dataset yang benar-benar salah.
    import random as _rnd
    n_render = 0
    for pola in POLA:
        semua_pola = [b for b in baris if b["pola"] == pola]
        if args.acak:
            rng = _rnd.Random(args.seed)
            antre = rng.sample(semua_pola, min(args.render, len(semua_pola)))
            antre.sort(key=lambda b: -b["skor"])
        else:
            antre = semua_pola[:args.render]
        if not antre:
            continue
        tujuan = keluaran / pola
        tujuan.mkdir(parents=True, exist_ok=True)
        for lama in tujuan.glob("*.jpg"):      # buang sisa run sebelumnya
            lama.unlink()
        for urut, b in enumerate(antre, 1):
            citra, label, prediksi, kf, pl, skor, p = simpanan[b["citra"]]
            cv2.imwrite(str(tujuan / f"{urut:03d}_{p.stem}.jpg"),
                        kartu(citra, label, prediksi,
                              [k for k, v in NAMA_KELAS.items() if v == kf][0],
                              b["citra"], pl, skor),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            n_render += 1

    # ---- Ringkasan ----
    print(f"\n{'kelas folder':22s} {'citra':>6s} " +
          " ".join(f"{p[:16]:>17s}" for p in POLA))
    for kf in sorted(rekap):
        r = rekap[kf]
        print(f"{kf:22s} {r['citra']:6d} " +
              " ".join(f"{r[p]:>17d}" for p in POLA))

    print(f"\nTotal bertanda: {len(baris)} dari {sum(r['citra'] for r in rekap.values())} citra "
          f"({100 * len(baris) / max(1, sum(r['citra'] for r in rekap.values())):.1f}%)")

    # Laju tanda per subset. train seharusnya PALING RENDAH karena model sudah
    # dilatih setuju di situ; selisihnya mengukur seberapa besar efek hafalan.
    n_sub = Counter(t[2] for t in tugas)
    per_sub = Counter(b["subset"] for b in baris)
    print(f"\n{'subset':8s} {'bertanda':>9s} {'citra':>7s} {'laju':>7s}")
    for sub in ["train", "val", "test"]:
        if n_sub[sub]:
            print(f"{sub:8s} {per_sub[sub]:9d} {n_sub[sub]:7d} "
                  f"{100 * per_sub[sub] / n_sub[sub]:6.1f}%")
    if n_sub["train"] and n_sub["test"]:
        lt = 100 * per_sub["train"] / n_sub["train"]
        ls = 100 * per_sub["test"] / n_sub["test"]
        print(f"\nSelisih test - train = {ls - lt:+.1f} poin. Karena model dilatih "
              f"pada train,\nlaju di test adalah perkiraan yang lebih jujur.")

    print(f"\n{n_render} gambar ditulis ke {keluaran}"
          + ("  (sampel ACAK, boleh dipakai memperkirakan persentase)" if args.acak
             else "  (N teratas, JANGAN dipakai memperkirakan persentase)"))
    print("""
Cara meninjau:

  1. Buka skrining_hasil/healthy_berlubang/ lebih dulu, urut dari 001.
     Kotak MERAH = yang dilihat model tapi bertentangan dengan folder.
     Pertanyaannya cuma satu: di situ ada gejala nyata atau tidak?

  2. Catat vonis anotator di kolom baru pada antrian.csv, misalnya:
        pindah    -> citra memang salah kelas
        biarkan   -> model yang keliru
        buang     -> meragukan, lebih aman disingkirkan

  3. Cukup tinjau 50-100 teratas dulu. Kalau proporsinya sudah jelas,
     skalanya bisa diperkirakan tanpa memeriksa semuanya.

  JANGAN memindahkan citra otomatis berdasarkan skor. Skor cuma mengurutkan
  antrian; yang memutuskan tetap mata manusia.
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

    L = 1000.0 * 1000.0
    besar = (0, 0, 700, 700)      # 49% luas
    kecil = (0, 0, 50, 50)        # 0,25% luas

    print("nilai()")
    s, p, _ = nilai(0, [(0, *besar, 0.9)], L)
    cek("sepakat = tanpa pola", p is None and s == 0.0)

    s, p, _ = nilai(0, [(2, *kecil, 0.8)], L)
    cek("healthy + lesi = healthy_berlubang", p == "healthy_berlubang" and s == 0.8)

    s, p, _ = nilai(1, [(3, *kecil, 0.5), (2, *kecil, 0.4)], L)
    cek("mosaic + dua lesi = skor dijumlah", p == "healthy_berlubang" and abs(s - 0.9) < 1e-9)

    s, p, _ = nilai(2, [(0, *besar, 0.7)], L)
    cek("leaf_spot + daun utuh besar = lesi_tampak_utuh", p == "lesi_tampak_utuh")

    s, p, _ = nilai(2, [(0, *kecil, 0.7)], L)
    cek("kotak healthy kecil bukan 'tampak utuh'", p is None)

    s, p, _ = nilai(0, [(1, *besar, 0.6)], L)
    cek("healthy <-> mosaic = tertukar_utuh", p == "tertukar_utuh")

    s, p, _ = nilai(2, [(3, *kecil, 0.6)], L)
    cek("leaf_spot <-> insect = tertukar_lesi", p == "tertukar_lesi")

    s, p, _ = nilai(0, [(2, *kecil, 0.9), (1, *besar, 0.3)], L)
    cek("pola terkuat yang menang", p == "healthy_berlubang")

    s, p, r = nilai(0, [(2, *kecil, 0.5), (3, *kecil, 0.5)], L)
    cek("rincian menghitung tiap kelas asing",
        r["leaf_spot_disease"] == 1 and r["insect_pest_disease"] == 1)

    s, p, _ = nilai(0, [], L)
    cek("tanpa deteksi = tanpa pola", p is None and s == 0.0)

    s, p, _ = nilai(0, [(2, *kecil, 0.8)], 0.0)
    cek("luas citra nol tidak error", p == "healthy_berlubang")

    print("kartu()")
    citra = np.full((400, 300, 3), 120, np.uint8)
    out = kartu(citra, [(0, 10, 10, 290, 390)],
                [(2, 20, 20, 60, 60, 0.77)], 0, "uji.jpg", "healthy_berlubang", 0.77)
    cek("keluaran 3 kanal", out.ndim == 3 and out.shape[2] == 3)
    cek("tinggi = panel + 2 pita", out.shape[0] == 900 + 34 + 30)
    out = kartu(citra, [], [], 0, "kosong.jpg", "tertukar_utuh", 0.0)
    cek("tanpa kotak tidak error", out.shape[0] > 0)

    print("\n" + ("SEMUA UJI LOLOS" if lolos else "ADA YANG GAGAL"))
    return 0 if lolos else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="80_10_10", choices=["80_10_10", "70_15_15"])
    p.add_argument("--model", default=None, help="yolo11n|yolo11s|yolo11m")
    p.add_argument("--sampel", type=int, default=0, help="0 = semua citra")
    p.add_argument("--render", type=int, default=40, help="gambar per pola (default 40)")
    p.add_argument("--acak", action="store_true",
                   help="render sampel ACAK dari yang bertanda, bukan N teratas. "
                        "Wajib dipakai kalau hasil tinjauan mau dijadikan persentase.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imgsz", type=int, default=640,
                   help="harus sama dengan imgsz saat training")
    p.add_argument("--uji", action="store_true", help="self-check lalu keluar")
    args = p.parse_args()

    if args.uji:
        sys.exit(uji())
    jalankan(args)


if __name__ == "__main__":
    main()
