#!/usr/bin/env python3
"""
Memperbaiki label 101 citra yang konvensinya tidak bisa ditangani autolabel.py.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

MASALAHNYA. autolabel.py menentukan mode anotasi dari FOLDER: daun_utuh untuk
healthy/mosaic, lesi untuk leaf_spot/insect. Dua vonis tinjauan melanggar itu.

  mode_utuh (28 citra)
      Citra kini berada di folder lesi, jadi autolabel memberinya banyak kotak.
      Padahal gejalanya MENYEBAR, bukan bercak diskret -- seharusnya SATU kotak
      seluruh daun, seperti konvensi healthy dan mosaic.

  campuran (73 citra)
      Satu daun memuat dua kondisi sekaligus: bercak berhalo DAN lubang bersih.
      autolabel memberi semua kotak satu kelas saja, mengikuti foldernya.
      Padahal YOLO mendukung banyak kelas dalam satu citra, dan memang begitu
      kenyataannya di daun tersebut.

CARA MEMBEDAKAN pada citra campuran, mengikuti aturan patologi proyek ini yang
sudah tertulis di autolabel.py:

      lubang TANPA warna di tepinya          -> insect_pest_disease
      lubang DENGAN halo kuning/hitam/coklat -> leaf_spot_disease
      bercak berwarna tanpa lubang           -> leaf_spot_disease

autolabel.py kini mencatat jenis tiap kotak secara terpisah (`lubang_bersih`
dan `lubang_berwarna`), jadi kelas per kotak bisa ditentukan dari situ.

DIJALANKAN SETELAH `autolabel.py --bersih`, bukan sebelum. Skrip ini menimpa
berkas label yang sudah ada.

Pakai:
    python 01_KODE\\terapkan_instruksi_label.py --uji        # self-check
    python 01_KODE\\terapkan_instruksi_label.py              # KERING: laporan
    python 01_KODE\\terapkan_instruksi_label.py --terapkan   # sungguhan
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from audit_prediksi import DIR_HASIL
from autolabel import CFG, KELAS, proses_citra, DIR_SUMBER, KELUARAN

INSTRUKSI = DIR_HASIL / "skrining_hasil" / "instruksi_label.csv"
DIR_LABEL = KELUARAN / "labels"
NAMA_KE_ID = {k: v[0] for k, v in KELAS.items()}
ID_KE_NAMA = {v: k for k, v in NAMA_KE_ID.items()}

# Jenis kotak -> kelas, mengikuti aturan patologi proyek.
JENIS_KE_KELAS = {
    "lubang_bersih": "insect_pest_disease",
    "lubang_berwarna": "leaf_spot_disease",
    "bercak": "leaf_spot_disease",
    "mottling": "leaf_spot_disease",
}


def baris_yolo(kotak, id_kelas, lebar, tinggi):
    x, y, w, h = kotak
    return (f"{id_kelas} {(x + w / 2) / lebar:.6f} {(y + h / 2) / tinggi:.6f} "
            f"{w / lebar:.6f} {h / tinggi:.6f}")


def kelas_per_kotak(jenis, kelas_utama, kelas_kedua):
    """
    Tentukan kelas tiap kotak pada citra campuran.

    Hanya dua kelas yang boleh muncul: kelas foldernya dan kelas kedua hasil
    tinjauan. Jenis kotak yang memetakan ke kelas di luar keduanya dikembalikan
    ke kelas utama -- lebih aman daripada memunculkan kelas ketiga yang tidak
    pernah divonis siapa pun.
    """
    boleh = {kelas_utama, kelas_kedua}
    keluar = []
    for j in jenis:
        k = JENIS_KE_KELAS.get(j, kelas_utama)
        keluar.append(k if k in boleh else kelas_utama)
    return keluar


def cari_citra(nama):
    for k in KELAS:
        p = DIR_SUMBER / k / nama
        if p.exists():
            return p, k
    return None, None


def olah(r, terapkan):
    """Kembalikan (status, rincian) untuk satu baris instruksi."""
    path, folder = cari_citra(r["citra"])
    if path is None:
        return "hilang", ""

    mode = "daun_utuh" if r["mode_label"] == "daun_utuh" else "lesi"
    hasil, bgr = proses_citra(path, NAMA_KE_ID[folder], mode, CFG)
    if bgr is None or not hasil["kotak"]:
        return "tanpa_kotak", ""
    T, L = bgr.shape[:2]

    if mode == "daun_utuh":
        baris = [baris_yolo(k, NAMA_KE_ID[folder], L, T) for k in hasil["kotak"]]
        rinci = f"{len(baris)} kotak, kelas {folder}"
    else:
        kelas = kelas_per_kotak(hasil["jenis"], folder, r["kelas_kedua"])
        baris = [baris_yolo(k, NAMA_KE_ID[c], L, T)
                 for k, c in zip(hasil["kotak"], kelas)]
        c = Counter(kelas)
        rinci = " + ".join(f"{n} {nm.split('_')[0]}" for nm, n in c.most_common())
        if len(c) < 2:
            return "tetap_satu_kelas", rinci

    if terapkan:
        tuj = DIR_LABEL / folder / f"{path.stem}.txt"
        tuj.parent.mkdir(parents=True, exist_ok=True)
        tuj.write_text("\n".join(baris) + "\n", encoding="utf-8")
    return "ok", rinci


def uji():
    lolos = True

    def cek(n, s):
        nonlocal lolos
        print(f"  {'ok  ' if s else 'GAGAL'}  {n}")
        lolos = lolos and s

    print("baris_yolo()")
    b = baris_yolo((10, 20, 40, 60), 2, 100, 200)
    bag = b.split()
    cek("id kelas benar", bag[0] == "2")
    cek("pusat x ternormalisasi", abs(float(bag[1]) - 0.30) < 1e-6)
    cek("pusat y ternormalisasi", abs(float(bag[2]) - 0.25) < 1e-6)
    cek("lebar ternormalisasi", abs(float(bag[3]) - 0.40) < 1e-6)
    cek("tinggi ternormalisasi", abs(float(bag[4]) - 0.30) < 1e-6)
    cek("semua dalam 0-1", all(0 <= float(v) <= 1 for v in bag[1:]))

    print("kelas_per_kotak()")
    j = ["lubang_bersih", "lubang_berwarna", "bercak"]
    k = kelas_per_kotak(j, "leaf_spot_disease", "insect_pest_disease")
    cek("lubang bersih -> insect", k[0] == "insect_pest_disease")
    cek("lubang berhalo -> leaf_spot", k[1] == "leaf_spot_disease")
    cek("bercak -> leaf_spot", k[2] == "leaf_spot_disease")

    k = kelas_per_kotak(["lubang_bersih"] * 3, "insect_pest_disease", "leaf_spot_disease")
    cek("folder insect: lubang bersih tetap insect", set(k) == {"insect_pest_disease"})

    k = kelas_per_kotak(["daun_utuh"], "leaf_spot_disease", "insect_pest_disease")
    cek("jenis tak dikenal jatuh ke kelas utama", k == ["leaf_spot_disease"])

    k = kelas_per_kotak(["mottling", "bercak"], "insect_pest_disease", "leaf_spot_disease")
    cek("mottling -> leaf_spot", set(k) == {"leaf_spot_disease"})

    k = kelas_per_kotak(["bercak"], "insect_pest_disease", "mosaic_virus_disease")
    cek("kelas di luar dua yang diizinkan ditolak", k == ["insect_pest_disease"])

    print("pemetaan jenis")
    cek("empat jenis terpetakan", len(JENIS_KE_KELAS) == 4)
    cek("lubang bersih != lubang berwarna",
        JENIS_KE_KELAS["lubang_bersih"] != JENIS_KE_KELAS["lubang_berwarna"])

    print("\n" + ("SEMUA UJI LOLOS" if lolos else "ADA YANG GAGAL"))
    return 0 if lolos else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--terapkan", action="store_true", help="sungguhan menimpa label")
    p.add_argument("--uji", action="store_true")
    a = p.parse_args()
    if a.uji:
        sys.exit(uji())

    if not INSTRUKSI.exists():
        sys.exit(f"Tidak ada {INSTRUKSI}. Jalankan terapkan_vonis.py --terapkan dulu.")
    rows = list(csv.DictReader(open(INSTRUKSI, encoding="utf-8-sig")))
    print(f"instruksi : {len(rows)} citra")
    print(f"  satu kotak daun utuh : {sum(1 for r in rows if r['mode_label']=='daun_utuh')}")
    print(f"  dua kelas (campuran) : {sum(1 for r in rows if r['kelas_kedua'])}")
    print(f"mode      : {'TERAPKAN' if a.terapkan else 'KERING (tidak menulis)'}\n")

    stat = Counter()
    contoh = []
    for i, r in enumerate(rows, 1):
        s, rinci = olah(r, a.terapkan)
        stat[s] += 1
        if len(contoh) < 8 and s in ("ok", "tetap_satu_kelas"):
            contoh.append(f"  {s:18s} {r['citra'][:38]:40s} {rinci}")
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}")

    print("\ncontoh hasil:")
    for c in contoh:
        print(c)
    print(f"\nringkasan: {dict(stat)}")
    if stat["tetap_satu_kelas"]:
        print(f"\n  {stat['tetap_satu_kelas']} citra campuran tetap satu kelas -- deteksi warna")
        print("  tidak menemukan dua jenis lesi. Labelnya dibiarkan apa adanya,")
        print("  bukan dipaksakan. Ini wajar dan tidak perlu diperbaiki manual.")

    if not a.terapkan:
        print("\n" + "-" * 72)
        print("MODE KERING. Label belum ditimpa. Kalau hasil di atas masuk akal:")
        print("    python 01_KODE\\terapkan_instruksi_label.py --terapkan")
        print("-" * 72)
    else:
        print("\nLabel diperbarui. Langkah berikutnya: bangun ulang split.")


if __name__ == "__main__":
    main()
