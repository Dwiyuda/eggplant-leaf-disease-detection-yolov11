#!/usr/bin/env python3
"""
Membandingkan model LAMA dan model BARU pada dataset yang sama, untuk memisahkan
"model membaik" dari "pengukurannya yang membaik".

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

MASALAH YANG DIPECAHKAN. Angka lama (test mAP50-95 0,5208) diukur dengan kunci
jawaban yang 18% keliru. Kalau angka baru sekadar dibandingkan dengannya, naik
atau turunnya tidak bisa ditafsirkan -- yang berubah bukan cuma modelnya, tapi
juga alat ukurnya.

TIGA ANGKA yang memisahkan keduanya:

  A. model LAMA, label LAMA    -> 0,5208  (yang selama ini dilaporkan)
  B. model LAMA, label BARU    -> dihitung skrip ini
  C. model BARU, label BARU    -> dihitung skrip ini

  B - A  = seberapa banyak skor lama tertekan oleh label yang salah.
           Kalau B jauh di atas A, berarti model lama sebenarnya lebih baik
           dari yang tercatat, cuma dihukum kunci jawaban yang keliru.

  C - B  = kenaikan yang benar-benar berasal dari BELAJAR dari label bersih.
           Inilah manfaat sesungguhnya dari perbaikan label.

Perbandingan ini sah karena keanggotaan test tidak berubah -- 208 citra yang
sama persis, dibangun dengan mode --pertahankan-lama.

PROTOKOL. Default memakai subset VAL. Subset test hanya dibuka dengan --test,
dan itu dilakukan SEKALI di akhir, untuk kedua model sekaligus.

Pakai:
    python 01_KODE\\bandingkan_model.py --uji
    python 01_KODE\\bandingkan_model.py --baru <path_best.pt>
    python 01_KODE\\bandingkan_model.py --baru <path_best.pt> --test
"""

import argparse
import json
import sys
from pathlib import Path

from audit_prediksi import NAMA_KELAS, DIR_DATASET, DIR_KAGGLE, DIR_HASIL

LAMA_DEFAULT = DIR_KAGGLE / "hasil_yolo11_ablasi_resolusi" / "img1024" / "best_yolo11s_img1024.pt"
SPLIT_DIR = "dataset_80_10_10"
IMGSZ = 1024
KELUARAN = DIR_HASIL / "perbandingan_akhir"

# Dari run 2026-08-08, diukur pada dataset LAMA dengan label lama.
ACUAN_A = {"val_mAP50_95": 0.5726, "val_mAP50": 0.6632,
           "test_mAP50_95": 0.5208, "test_mAP50": 0.6205,
           "akurasi_citra_test": 0.7019}


def selisih(b, a):
    return f"{b - a:+.4f}"


def evaluasi(model, data_yaml, split, imgsz):
    """
    Kembalikan metrik total dan per kelas.

    Per kelas disimpan DUA-duanya -- mAP50 dan mAP50-95 -- di kamus terpisah.
    Versi pertama skrip ini cuma menyimpan mAP50-95 lalu memakainya untuk kedua
    tabel, sehingga baris per kelas pada tabel mAP50 menampilkan angka mAP50-95.
    Barisnya terlihat wajar dan tidak memicu error apa pun, jadi kekeliruan itu
    sempat lolos.
    """
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz,
                  verbose=False, plots=False)
    b = m.box
    per95, per50 = {}, {}
    idx = [int(c) for c in getattr(b, "ap_class_index", list(NAMA_KELAS))]
    for i, c in enumerate(idx):
        nama = NAMA_KELAS.get(c, str(c))
        try:
            per95[nama] = float(b.maps[c])
        except Exception:
            per95[nama] = float(list(b.class_result(i))[3])
        try:
            per50[nama] = float(b.ap50[i])
        except Exception:
            per50[nama] = float(list(b.class_result(i))[2])
    return {"mAP50": float(b.map50), "mAP50_95": float(b.map),
            "precision": float(b.mp), "recall": float(b.mr),
            "per_kelas": per95, "per_kelas_mAP50": per50}


def tabel(judul, lama, baru, kunci="mAP50_95"):
    kolom = "per_kelas" if kunci == "mAP50_95" else "per_kelas_mAP50"
    print(f"\n{judul}")
    print(f"  {'':24s} {'model LAMA':>12s} {'model BARU':>12s} {'selisih':>10s}")
    print(f"  {'TOTAL ' + kunci:24s} {lama[kunci]:12.4f} {baru[kunci]:12.4f} "
          f"{selisih(baru[kunci], lama[kunci]):>10s}")
    for k in NAMA_KELAS.values():
        if k in lama[kolom] and k in baru[kolom]:
            print(f"  {k:24s} {lama[kolom][k]:12.4f} {baru[kolom][k]:12.4f} "
                  f"{selisih(baru[kolom][k], lama[kolom][k]):>10s}")


def uji():
    lolos = True

    def cek(n, s):
        nonlocal lolos
        print(f"  {'ok  ' if s else 'GAGAL'}  {n}")
        lolos = lolos and s

    print("selisih()")
    cek("naik diberi tanda plus", selisih(0.60, 0.50) == "+0.1000")
    cek("turun diberi tanda minus", selisih(0.40, 0.50) == "-0.1000")
    cek("sama = nol bertanda", selisih(0.5, 0.5) == "+0.0000")

    print("acuan A")
    cek("test mAP50-95 lama tercatat", ACUAN_A["test_mAP50_95"] == 0.5208)
    cek("val mAP50-95 lama tercatat", ACUAN_A["val_mAP50_95"] == 0.5726)
    cek("akurasi citra lama tercatat", ACUAN_A["akurasi_citra_test"] == 0.7019)

    print("jalur")
    cek("bobot lama ditunjuk ke img1024", "img1024" in str(LAMA_DEFAULT))
    cek("split 80:10:10", SPLIT_DIR == "dataset_80_10_10")
    cek("imgsz 1024 sesuai model terbaik", IMGSZ == 1024)

    print("\n" + ("SEMUA UJI LOLOS" if lolos else "ADA YANG GAGAL"))
    return 0 if lolos else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lama", default=None, help="bobot model lama (.pt)")
    p.add_argument("--baru", default=None, help="bobot model baru (.pt) dari Kaggle")
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--test", action="store_true",
                   help="buka subset TEST. Lakukan SEKALI di akhir saja.")
    p.add_argument("--uji", action="store_true")
    a = p.parse_args()
    if a.uji:
        sys.exit(uji())

    from ultralytics import YOLO
    lama = Path(a.lama) if a.lama else LAMA_DEFAULT
    if not a.baru:
        sys.exit("Sebutkan bobot model baru:  --baru <path best.pt>")
    baru = Path(a.baru)
    for p_ in (lama, baru):
        if not p_.exists():
            sys.exit(f"Bobot tidak ketemu: {p_}")

    data_yaml = DIR_DATASET / SPLIT_DIR / "data.yaml"
    split = "test" if a.test else "val"
    KELUARAN.mkdir(parents=True, exist_ok=True)

    print(f"model lama : {lama.name}")
    print(f"model baru : {baru.name}")
    print(f"dataset    : {SPLIT_DIR} (label sudah diperbaiki)")
    print(f"subset     : {split.upper()}"
          + ("   <- protokol: dibuka sekali saja" if a.test else ""))
    print(f"imgsz      : {a.imgsz}\n")

    print("mengevaluasi model lama...")
    hl = evaluasi(YOLO(str(lama)), data_yaml, split, a.imgsz)
    print("mengevaluasi model baru...")
    hb = evaluasi(YOLO(str(baru)), data_yaml, split, a.imgsz)

    tabel(f"mAP50-95 per kelas  --  subset {split.upper()}, label BARU", hl, hb)
    tabel(f"mAP50 per kelas  --  subset {split.upper()}, label BARU", hl, hb, "mAP50")

    kunci = f"{split}_mAP50_95"
    A = ACUAN_A.get(kunci)
    print("\n" + "=" * 74)
    print("  MEMISAHKAN 'MODEL MEMBAIK' DARI 'PENGUKURAN MEMBAIK'")
    print("=" * 74)
    print(f"\n  A. model LAMA, label LAMA : {A:.4f}   (yang selama ini dilaporkan)")
    print(f"  B. model LAMA, label BARU : {hl['mAP50_95']:.4f}   ({selisih(hl['mAP50_95'], A)})")
    print(f"  C. model BARU, label BARU : {hb['mAP50_95']:.4f}   ({selisih(hb['mAP50_95'], hl['mAP50_95'])} dari B)")
    print(f"\n  B - A = {selisih(hl['mAP50_95'], A)}  seberapa banyak skor lama tertekan label salah")
    print(f"  C - B = {selisih(hb['mAP50_95'], hl['mAP50_95'])}  kenaikan dari BELAJAR pada label bersih")
    print(f"  C - A = {selisih(hb['mAP50_95'], A)}  selisih total apa adanya")

    print("\n  Cara membaca:")
    if hl["mAP50_95"] > A + 0.01:
        print("   B di atas A -> model lama sebenarnya lebih baik dari yang tercatat.")
        print("   Sebagian 'kegagalan' selama ini adalah kesalahan kunci jawaban.")
    elif hl["mAP50_95"] < A - 0.01:
        print("   B DI BAWAH A -> angka lama ternyata TERLALU TINGGI, bukan tertekan.")
        print("   Model lama ikut mempelajari label yang keliru, sehingga kunci")
        print("   jawaban yang sama-sama keliru itu justru menyanjungnya. Diuji")
        print("   dengan kunci yang benar, nilainya turun.")
        print(f"   -> Angka 'sebelum' yang jujur bukan {A:.4f}, melainkan {hl['mAP50_95']:.4f}.")
    if hb["mAP50_95"] > hl["mAP50_95"] + 0.01:
        print("   C di atas B -> melatih ulang pada label bersih memang berbuah.")
    elif hb["mAP50_95"] < hl["mAP50_95"] - 0.01:
        print("   C di bawah B -> label bersih TIDAK menolong training. Layak diselidiki:")
        print("   komposisi kelas berubah cukup jauh (healthy 23%->16%), bisa jadi itu sebabnya.")
    else:
        print("   C setara B -> kenaikan hampir seluruhnya dari pengukuran yang benar,")
        print("   bukan dari model yang belajar lebih baik. Ini pun temuan yang sah.")

    (KELUARAN / f"perbandingan_{split}.json").write_text(json.dumps({
        "model_lama": str(lama), "model_baru": str(baru),
        "dataset": SPLIT_DIR, "subset": split, "imgsz": a.imgsz,
        "A_lama_labelLama": A, "B_lama_labelBaru": hl, "C_baru_labelBaru": hb,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nhasil -> {KELUARAN}")
    if not a.test:
        print("\nIni subset VAL. Setelah konfigurasi final dikunci, jalankan sekali")
        print("dengan --test untuk angka yang masuk Bab IV.")


if __name__ == "__main__":
    main()
