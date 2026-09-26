#!/usr/bin/env python3
"""
Menerapkan vonis tinjauan ke dataset: memindahkan citra ke kelas yang benar,
membuat ulang labelnya, lalu membangun kembali kedua split.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

INI SATU-SATUNYA SKRIP YANG MENGUBAH DATASET. Semua yang lain cuma membaca.
Karena itu:
  - default-nya KERING (tidak menulis apa pun), harus --terapkan untuk sungguhan
  - cadangan otomatis dibuat sebelum apa pun disentuh
  - laporan rencana selalu dicetak dulu supaya bisa diperiksa

Perlakuan tiap vonis:

  pindah     citra dipindah ke kelas yang ditunjukkan deteksi model,
             lalu labelnya dibuat ulang oleh autolabel.py mode kelas baru
  campuran   citra TETAP di kelasnya. Satu daun memuat dua kondisi, dan YOLO
             memang mendukung banyak kelas dalam satu citra -- jadi kotak
             kelas kedua DITAMBAHKAN, bukan menggantikan
  mode_utuh  labelnya jadi SATU kotak seluruh daun. Kelasnya tergantung asal:
               dari folder lesi (leaf_spot/insect)  -> kelas TETAP.
                 Gejalanya menyebar, bukan bercak diskret.
               dari folder utuh (healthy/mosaic)    -> kelas PINDAH ke kelas
                 lesi yang ditunjukkan deteksi. Daunnya memang bergejala, tapi
                 menyebar -- jadi pindah kelas sambil tetap satu kotak.
             Ditetapkan 2026-08-15 setelah 26 vonis mode_utuh muncul pada
             citra healthy/mosaic, yang tanpa aturan ini tidak punya arti.
  buang      citra dikeluarkan dari dataset ke 99_ARSIP/dibuang_tinjauan/
  biarkan    tidak ada perubahan

Setelah citra berpindah, split WAJIB dibangun ulang -- kalau tidak, komposisi
kelas tiap split jadi timpang dan citra bisa bocor antar-split.

Pakai:
    python 01_KODE\\terapkan_vonis.py --uji          # self-check
    python 01_KODE\\terapkan_vonis.py                # KERING: cuma laporan rencana
    python 01_KODE\\terapkan_vonis.py --terapkan     # sungguhan
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from audit_prediksi import NAMA_KELAS, DIR_DATASET, DIR_HASIL

AKAR = DIR_DATASET.parent
DIR_SUMBER = AKAR / "03_CITRA_SUMBER"
XLSX = DIR_HASIL / "skrining_hasil" / "Tinjauan Skrining Label.xlsx"
DIBUANG = AKAR / "99_ARSIP" / "dibuang_tinjauan"
CADANGAN = AKAR / "99_ARSIP" / "cadangan_sebelum_terapkan"

NAMA_KE_ID = {v: k for k, v in NAMA_KELAS.items()}
UTUH, LESI = {0, 1}, {2, 3}


def baca_vonis(xlsx):
    try:
        from openpyxl import load_workbook
    except ModuleNotFoundError:
        sys.exit("\nopenpyxl belum terpasang. Pasang dulu, lalu ulangi:\n"
                 "    python -m pip install openpyxl\n")
    if not xlsx.exists():
        sys.exit(f"Lembar tinjauan tidak ada: {xlsx}")
    ws = load_workbook(xlsx)["Tinjauan"]
    baris = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[3] and (r[8] or "").strip():
            baris.append({"citra": r[3], "pola": r[1], "folder": r[5],
                          "deteksi": r[7] or "", "vonis": r[8].strip().lower(),
                          "skor": r[6]})
    return baris


def kelas_tujuan(folder_asal, deteksi):
    """
    Tentukan kelas tujuan dari deteksi model. Deteksi berformat
    'insect_pest_disease x4  +  leaf_spot_disease x1' -- diambil yang terbanyak,
    dan HANYA di antara kelas yang berbeda dari folder asalnya.
    """
    asal = NAMA_KE_ID.get(folder_asal, -1)
    hit = Counter()
    for bagian in (deteksi or "").split("+"):
        b = bagian.strip()
        if not b or " x" not in b:
            continue
        nama, n = b.rsplit(" x", 1)
        k = NAMA_KE_ID.get(nama.strip())
        if k is not None and k != asal and n.strip().isdigit():
            hit[k] += int(n)
    return hit.most_common(1)[0][0] if hit else None


def rencana(baris):
    """Susun rencana tindakan. Tidak menyentuh berkas apa pun."""
    aksi = {"pindah": [], "campuran": [], "mode_utuh": [], "buang": [], "lewat": []}
    for b in baris:
        v = b["vonis"]
        if v == "biarkan":
            aksi["lewat"].append(b)
        elif v == "buang":
            aksi["buang"].append(b)
        elif v == "mode_utuh":
            # Label selalu jadi satu kotak seluruh daun. Kelasnya berubah hanya
            # kalau citra berasal dari folder daun-utuh -- di situ mode_utuh
            # berarti "memang bergejala, tapi menyebar", jadi harus pindah kelas.
            asal = NAMA_KE_ID.get(b["folder"], -1)
            if asal in UTUH:
                k2 = kelas_tujuan(b["folder"], b["deteksi"])
                if k2 is None:
                    aksi["lewat"].append({**b, "alasan": "mode_utuh tanpa kelas tujuan"})
                else:
                    aksi["mode_utuh"].append({**b, "tujuan": k2, "mode_label": "daun_utuh"})
            else:
                aksi["mode_utuh"].append({**b, "tujuan": asal, "mode_label": "daun_utuh"})
        elif v == "campuran":
            k2 = kelas_tujuan(b["folder"], b["deteksi"])
            (aksi["campuran"] if k2 is not None else aksi["lewat"]).append(
                {**b, "kelas_kedua": k2})
        elif v == "pindah":
            k2 = kelas_tujuan(b["folder"], b["deteksi"])
            if k2 is None:
                aksi["lewat"].append({**b, "alasan": "kelas tujuan tak terbaca"})
            else:
                aksi["pindah"].append({**b, "tujuan": k2})
    return aksi


def cetak_rencana(aksi):
    print("\n" + "=" * 78)
    print("  RENCANA TINDAKAN")
    print("=" * 78)
    print(f"\n  pindah kelas    : {len(aksi['pindah'])} citra")
    if aksi["pindah"]:
        arah = Counter(f"{b['folder']} -> {NAMA_KELAS[b['tujuan']]}" for b in aksi["pindah"])
        for a, n in arah.most_common():
            print(f"      {a:52s} {n:4d}")
    print(f"\n  tambah kelas kedua (campuran) : {len(aksi['campuran'])} citra")
    if aksi["campuran"]:
        for a, n in Counter(f"{b['folder']} + {NAMA_KELAS[b['kelas_kedua']]}"
                            for b in aksi["campuran"]).most_common():
            print(f"      {a:52s} {n:4d}")
    mu = aksi["mode_utuh"]
    pindah_mu = [b for b in mu if b["tujuan"] != NAMA_KE_ID.get(b["folder"], -1)]
    print(f"\n  satu kotak daun utuh          : {len(mu)} citra")
    if pindah_mu:
        print(f"      di antaranya {len(pindah_mu)} SEKALIGUS pindah kelas:")
        for a, n in Counter(f"{b['folder']} -> {NAMA_KELAS[b['tujuan']]}"
                            for b in pindah_mu).most_common():
            print(f"        {a:50s} {n:4d}")
    tetap_mu = len(mu) - len(pindah_mu)
    if tetap_mu:
        print(f"      {tetap_mu} tetap di kelasnya, hanya labelnya yang berubah")
    print(f"  keluarkan dari dataset        : {len(aksi['buang'])} citra")
    print(f"  tidak diubah                  : {len(aksi['lewat'])} citra")
    tanpa = [b for b in aksi["lewat"] if b.get("alasan")]
    if tanpa:
        print(f"      di antaranya {len(tanpa)} gagal ditentukan kelas tujuannya:")
        for b in tanpa[:5]:
            print(f"        {b['citra'][:50]}  deteksi='{b['deteksi'][:34]}'")

    total = sum(len(aksi[k]) for k in ["pindah", "campuran", "mode_utuh", "buang"])
    print(f"\n  TOTAL citra berubah : {total}")


def cadangkan():
    CADANGAN.mkdir(parents=True, exist_ok=True)
    cap = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuj = CADANGAN / cap
    tuj.mkdir()
    for f in sorted(DIR_SUMBER.iterdir()):
        if f.is_dir():
            (tuj / "daftar").mkdir(exist_ok=True)
            (tuj / "daftar" / f"{f.name}.txt").write_text(
                "\n".join(sorted(p.name for p in f.iterdir() if p.is_file())),
                encoding="utf-8")
    print(f"\ncadangan daftar isi folder kelas -> {tuj}")
    return tuj


def terapkan(aksi, tuj_cadangan):
    log = defaultdict(list)

    for b in aksi["pindah"]:
        asal = DIR_SUMBER / b["folder"] / b["citra"]
        tuj = DIR_SUMBER / NAMA_KELAS[b["tujuan"]] / b["citra"]
        if not asal.exists():
            log["hilang"].append(b["citra"]); continue
        tuj.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(asal), str(tuj))
        log["pindah"].append(f"{b['folder']}/{b['citra']} -> {NAMA_KELAS[b['tujuan']]}")

    for b in aksi["buang"]:
        asal = DIR_SUMBER / b["folder"] / b["citra"]
        if not asal.exists():
            log["hilang"].append(b["citra"]); continue
        (DIBUANG / b["folder"]).mkdir(parents=True, exist_ok=True)
        shutil.move(str(asal), str(DIBUANG / b["folder"] / b["citra"]))
        log["buang"].append(f"{b['folder']}/{b['citra']}")

    # mode_utuh dari folder daun-utuh IKUT berpindah berkas; yang dari folder
    # lesi tidak. Sisanya soal LABEL, dibuat ulang autolabel.py setelah ini.
    for b in aksi["mode_utuh"]:
        asal_id = NAMA_KE_ID.get(b["folder"], -1)
        if b["tujuan"] != asal_id:
            asal = DIR_SUMBER / b["folder"] / b["citra"]
            tuj = DIR_SUMBER / NAMA_KELAS[b["tujuan"]] / b["citra"]
            if not asal.exists():
                log["hilang"].append(b["citra"]); continue
            tuj.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(asal), str(tuj))
            log["mode_utuh_pindah"].append(
                f"{b['folder']}/{b['citra']} -> {NAMA_KELAS[b['tujuan']]} (1 kotak)")
        else:
            log["mode_utuh_tetap"].append(b["citra"])

    for b in aksi["campuran"]:
        log["campuran"].append(f"{b['citra']} + {NAMA_KELAS[b['kelas_kedua']]}")

    (tuj_cadangan / "log_perubahan.json").write_text(
        json.dumps({k: v for k, v in log.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    tulis_instruksi(aksi)
    return log


def tulis_instruksi(aksi):
    """
    Daftar citra yang butuh perlakuan KHUSUS saat label dibuat ulang.

    autolabel.py menentukan mode dari FOLDER -- daun_utuh untuk healthy/mosaic,
    lesi untuk leaf_spot/insect. Dua vonis melanggar aturan itu:

      mode_utuh  citra kini ada di folder lesi, tapi labelnya harus SATU kotak
      campuran   citra butuh kotak DUA kelas sekaligus, sedangkan autolabel
                 cuma memberi satu kelas per citra

    Tanpa berkas ini, 99 citra akan dilabeli dengan konvensi yang salah.
    """
    baris = []
    for b in aksi["mode_utuh"]:
        baris.append({"citra": b["citra"], "kelas_final": NAMA_KELAS[b["tujuan"]],
                      "mode_label": "daun_utuh", "kelas_kedua": "",
                      "alasan": "gejala menyebar, bukan bercak diskret"})
    for b in aksi["campuran"]:
        baris.append({"citra": b["citra"], "kelas_final": b["folder"],
                      "mode_label": "lesi", "kelas_kedua": NAMA_KELAS[b["kelas_kedua"]],
                      "alasan": "satu daun memuat dua kondisi"})
    if not baris:
        return
    p = DIR_HASIL / "skrining_hasil" / "instruksi_label.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0]))
        w.writeheader()
        w.writerows(baris)
    print(f"\ninstruksi label khusus -> {p}")
    print(f"  {sum(1 for b in baris if b['mode_label']=='daun_utuh')} citra: satu kotak seluruh daun")
    print(f"  {sum(1 for b in baris if b['kelas_kedua'])} citra: butuh kotak dua kelas")


def uji():
    lolos = True

    def cek(n, s):
        nonlocal lolos
        print(f"  {'ok  ' if s else 'GAGAL'}  {n}")
        lolos = lolos and s

    print("kelas_tujuan()")
    cek("ambil kelas asing terbanyak",
        kelas_tujuan("healthy_leaf", "insect_pest_disease x4  +  leaf_spot_disease x1") == 3)
    cek("abaikan kelas yang sama dengan asal",
        kelas_tujuan("healthy_leaf", "healthy_leaf x2  +  leaf_spot_disease x1") == 2)
    cek("semua sama dengan asal -> None",
        kelas_tujuan("healthy_leaf", "healthy_leaf x3") is None)
    cek("deteksi kosong -> None", kelas_tujuan("healthy_leaf", "") is None)
    cek("None aman", kelas_tujuan("healthy_leaf", None) is None)
    cek("format rusak diabaikan", kelas_tujuan("healthy_leaf", "sampah tanpa x") is None)
    cek("satu kelas asing", kelas_tujuan("leaf_spot_disease", "insect_pest_disease x7") == 3)

    print("rencana()")
    baris = [
        {"citra": "a.jpg", "pola": "healthy_berlubang", "folder": "healthy_leaf",
         "deteksi": "insect_pest_disease x4", "vonis": "pindah", "skor": 2.0},
        {"citra": "b.jpg", "pola": "tertukar_lesi", "folder": "leaf_spot_disease",
         "deteksi": "insect_pest_disease x3", "vonis": "campuran", "skor": 1.0},
        {"citra": "c.jpg", "pola": "lesi_tampak_utuh", "folder": "leaf_spot_disease",
         "deteksi": "healthy_leaf x1", "vonis": "mode_utuh", "skor": 0.5},
        {"citra": "g.jpg", "pola": "healthy_berlubang", "folder": "healthy_leaf",
         "deteksi": "leaf_spot_disease x6", "vonis": "mode_utuh", "skor": 1.5},
        {"citra": "d.jpg", "pola": "tertukar_utuh", "folder": "healthy_leaf",
         "deteksi": "mosaic_virus_disease x1", "vonis": "biarkan", "skor": 0.4},
        {"citra": "e.jpg", "pola": "healthy_berlubang", "folder": "healthy_leaf",
         "deteksi": "leaf_spot_disease x2", "vonis": "buang", "skor": 0.3},
        {"citra": "f.jpg", "pola": "healthy_berlubang", "folder": "healthy_leaf",
         "deteksi": "healthy_leaf x1", "vonis": "pindah", "skor": 0.3},
    ]
    a = rencana(baris)
    cek("pindah terhitung", len(a["pindah"]) == 1 and a["pindah"][0]["tujuan"] == 3)
    cek("campuran terhitung", len(a["campuran"]) == 1 and a["campuran"][0]["kelas_kedua"] == 3)
    cek("mode_utuh terhitung", len(a["mode_utuh"]) == 2)
    mu = {x["citra"]: x for x in a["mode_utuh"]}
    cek("mode_utuh dari folder lesi -> kelas TETAP",
        mu["c.jpg"]["tujuan"] == 2 and mu["c.jpg"]["mode_label"] == "daun_utuh")
    cek("mode_utuh dari folder utuh -> kelas PINDAH ke lesi",
        mu["g.jpg"]["tujuan"] == 2 and mu["g.jpg"]["mode_label"] == "daun_utuh")
    cek("buang terhitung", len(a["buang"]) == 1)
    cek("biarkan tidak diubah", any(b["citra"] == "d.jpg" for b in a["lewat"]))
    cek("pindah tanpa kelas tujuan jatuh ke lewat",
        any(b["citra"] == "f.jpg" and b.get("alasan") for b in a["lewat"]))
    cek("tiap citra masuk tepat satu aksi",
        sum(len(v) for v in a.values()) == len(baris))

    print("keamanan")
    cek("DIBUANG di dalam 99_ARSIP", "99_ARSIP" in str(DIBUANG))
    cek("CADANGAN di dalam 99_ARSIP", "99_ARSIP" in str(CADANGAN))

    print("\n" + ("SEMUA UJI LOLOS" if lolos else "ADA YANG GAGAL"))
    return 0 if lolos else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--terapkan", action="store_true",
                   help="SUNGGUHAN memindahkan berkas. Tanpa ini cuma laporan rencana.")
    p.add_argument("--uji", action="store_true")
    a = p.parse_args()
    if a.uji:
        sys.exit(uji())

    baris = baca_vonis(XLSX)
    print(f"vonis terbaca : {len(baris)}")
    print("sebaran       :", dict(Counter(b["vonis"] for b in baris)))
    aksi = rencana(baris)
    cetak_rencana(aksi)

    if not a.terapkan:
        print("\n" + "-" * 78)
        print("MODE KERING. Tidak ada berkas yang disentuh.")
        print("Periksa rencana di atas. Kalau sudah benar, jalankan ulang dengan:")
        print("    python 01_KODE\\terapkan_vonis.py --terapkan")
        print("-" * 78)
        return

    tuj = cadangkan()
    log = terapkan(aksi, tuj)
    print("\nDITERAPKAN:")
    for k, v in log.items():
        print(f"  {k:10s} {len(v)}")
    print(f"\nLangkah berikutnya WAJIB, jangan dilewati:")
    print("  1. Buat ulang label   : python 01_KODE\\autolabel.py")
    print("  2. Terapkan instruksi khusus untuk citra mode_utuh dan campuran")
    print("     -- autolabel.py menentukan mode dari FOLDER, jadi 99 citra ini")
    print("        akan salah konvensi kalau langkah ini dilewati")
    print("  3. Bangun ulang split : citra sudah berpindah kelas, komposisi split")
    print("     lama tidak sah lagi dan berpotensi bocor antar-split")
    print(f"\nCadangan daftar isi + log ada di {tuj}")


if __name__ == "__main__":
    main()
