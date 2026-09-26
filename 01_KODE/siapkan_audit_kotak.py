#!/usr/bin/env python3
"""
Menyiapkan citra kelas lemah untuk audit KOTAK secara manual.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

PERTANYAAN YANG DIJAWAB. Recall leaf_spot cuma 0,2297 -- model menemukan kurang
dari seperempat bercak yang dilabeli. Dua kemungkinan yang belum pernah
dibedakan:

  (a) bercaknya nyata tapi terlalu kecil untuk dideteksi model
  (b) sebagian kotak yang dibuat autolabel.py TIDAK ada bercaknya -- cuma
      variasi warna yang lolos ambang aturan

Kalau (b), recall rendah itu WAJAR: model tidak bisa menemukan sesuatu yang
memang tidak terlihat. Selama ini kita memperbaiki kelas mana yang benar untuk
tiap citra, tapi tidak pernah memeriksa apakah kotaknya sendiri masuk akal.

PRIORITAS. Citra diurutkan menurut jumlah kotak label yang TIDAK ditemukan
model (FN). Yang teratas adalah kasus paling ekstrem -- label mengklaim banyak
bercak, model tidak melihat satu pun. Di situlah jawabannya paling kentara.

LINGKUP. Menggambar ulang seluruh 1.307 citra kelas lemah berarti 12.693 kotak,
pekerjaan berminggu-minggu. Skrip ini menyiapkan sebagian dulu supaya bisa
diputuskan apakah sepadan sebelum menghabiskan waktu sebanyak itu.

KELUARAN. Struktur baku YOLO yang langsung bisa dibuka LabelImg, Roboflow,
atau CVAT -- lengkap dengan label yang ADA SEKARANG, sehingga anotator tinggal
MENGOREKSI, bukan menggambar dari nol.

Pakai:
    python 01_KODE\\siapkan_audit_kotak.py --uji
    python 01_KODE\\siapkan_audit_kotak.py                  # 30 citra per kelas
    python 01_KODE\\siapkan_audit_kotak.py --jumlah 100
    python 01_KODE\\siapkan_audit_kotak.py --semua           # seluruh 1.307 citra
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

from audit_prediksi import (NAMA_KELAS, MODEL_TERPILIH, cari_bobot, muat_label,
                            cocokkan, DIR_DATASET, DIR_HASIL)

AKAR = DIR_DATASET.parent
KELUARAN = AKAR / "06_AUDIT_KOTAK"
KELAS_LEMAH = ["leaf_spot_disease", "insect_pest_disease"]
SPLIT_DIR = "dataset_80_10_10"
CONF = 0.25


def kumpulkan_citra(akar):
    """{stem: (path_citra, path_label, split)} untuk kelas lemah saja."""
    peta = {}
    for sub in ["train", "val", "test"]:
        di, dl = akar / "images" / sub, akar / "labels" / sub
        if not di.is_dir():
            continue
        for p in sorted(di.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            t = dl / f"{p.stem}.txt"
            if not t.exists():
                continue
            ids = [int(float(b.split()[0])) for b in t.read_text().split("\n") if b.strip()]
            if not ids:
                continue
            dom = Counter(ids).most_common(1)[0][0]
            if NAMA_KELAS.get(dom) in KELAS_LEMAH:
                peta[p.stem] = (p, t, sub, NAMA_KELAS[dom])
    return peta


def tulis_classes(dst):
    """classes.txt wajib untuk LabelImg; urutannya harus sama dengan id kelas."""
    (dst / "classes.txt").write_text(
        "\n".join(NAMA_KELAS[i] for i in sorted(NAMA_KELAS)) + "\n", encoding="utf-8")


def uji():
    lolos = True

    def cek(n, s):
        nonlocal lolos
        print(f"  {'ok  ' if s else 'GAGAL'}  {n}")
        lolos = lolos and s

    print("konfigurasi")
    cek("dua kelas lemah dipilih", KELAS_LEMAH == ["leaf_spot_disease", "insect_pest_disease"])
    cek("keluaran di 06_AUDIT_KOTAK", KELUARAN.name == "06_AUDIT_KOTAK")
    cek("conf sama dengan audit sebelumnya", CONF == 0.25)

    print("pembagian terburuk vs acak")
    import random as _rnd
    g = [{"i": i, "tidak_ditemukan": 100 - i} for i in range(200)]
    n_buruk = 50 // 2
    buruk, sisa = g[:n_buruk], g[n_buruk:]
    acak = _rnd.Random(42).sample(sisa, 50 - n_buruk)
    cek("separuh dari yang terburuk", len(buruk) == 25)
    cek("separuh acak dari sisanya", len(acak) == 25)
    cek("tidak ada yang dobel", len({x["i"] for x in buruk} & {x["i"] for x in acak}) == 0)
    cek("acak tidak diambil dari 25 teratas",
        all(x["i"] >= n_buruk for x in acak))
    cek("acak deterministik",
        [x["i"] for x in acak] == [x["i"] for x in _rnd.Random(42).sample(sisa, 25)])

    print("tulis_classes()")
    import tempfile
    d = Path(tempfile.mkdtemp())
    tulis_classes(d)
    isi = (d / "classes.txt").read_text().strip().split("\n")
    cek("empat kelas tertulis", len(isi) == 4)
    cek("urutan sesuai id kelas", isi[0] == "healthy_leaf" and isi[3] == "insect_pest_disease")

    print("cocokkan() dipakai menghitung FN")
    L = [(2, 0, 0, 10, 10), (2, 50, 50, 60, 60), (2, 90, 90, 99, 99)]
    v, fn = cocokkan(L, [(2, 0, 0, 10, 10, 0.9)])
    cek("2 label terlewat dari 3", len(fn) == 2)
    v, fn = cocokkan(L, [])
    cek("tanpa prediksi = semua terlewat", len(fn) == 3)
    v, fn = cocokkan(L, [(2, 0, 0, 10, 10, .9), (2, 50, 50, 60, 60, .8), (2, 90, 90, 99, 99, .7)])
    cek("semua ketemu = nol terlewat", len(fn) == 0)

    print("\n" + ("SEMUA UJI LOLOS" if lolos else "ADA YANG GAGAL"))
    return 0 if lolos else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jumlah", type=int, default=50,
                   help="citra per kelas (default 50: separuh terburuk, separuh acak)")
    p.add_argument("--semua", action="store_true", help="siapkan seluruh citra kelas lemah")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--uji", action="store_true")
    a = p.parse_args()
    if a.uji:
        sys.exit(uji())

    from ultralytics import YOLO
    akar = DIR_DATASET / SPLIT_DIR
    peta = kumpulkan_citra(akar)
    print(f"citra kelas lemah : {len(peta)}")
    for k in KELAS_LEMAH:
        print(f"   {k:24s} {sum(1 for v in peta.values() if v[3] == k)}")

    bobot = cari_bobot("80_10_10", MODEL_TERPILIH["80_10_10"])
    # model terbaik ada di folder dataset baru; pakai itu kalau ada
    baru = (AKAR / "04_KAGGLE" / "FOLDER_HASIL_KAGGLE"
            / "hasil_yolo11_ablasi_resolusi_dataset_baru" / "img1024"
            / "best_yolo11s_img1024.pt")
    if baru.exists():
        bobot = baru
    print(f"\nbobot : {bobot.name}\nimgsz : {a.imgsz}")
    print("menghitung kotak label yang TIDAK ditemukan model...\n")
    model = YOLO(str(bobot))

    baris = []
    for n, (stem, (pc, pl, sub, kelas)) in enumerate(sorted(peta.items()), 1):
        citra = cv2.imread(str(pc), cv2.IMREAD_COLOR)
        if citra is None:
            continue
        h, w = citra.shape[:2]
        label = muat_label(pl, w, h)
        r = model.predict(str(pc), conf=CONF, imgsz=a.imgsz, verbose=False)[0]
        pred = [(int(c), *map(float, xy), float(cf))
                for c, xy, cf in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(),
                                     r.boxes.conf.tolist())]
        vonis, fn = cocokkan(label, pred)
        baris.append({"citra": pc.name, "kelas": kelas, "subset": sub,
                      "n_label": len(label), "n_prediksi": len(pred),
                      "tidak_ditemukan": len(fn),
                      "persen_terlewat": round(len(fn) / max(1, len(label)), 3),
                      "_pc": pc, "_pl": pl})
        if n % 100 == 0:
            print(f"  {n}/{len(peta)}")

    baris.sort(key=lambda b: (-b["tidak_ditemukan"], -b["persen_terlewat"]))

    if KELUARAN.exists():
        shutil.rmtree(KELUARAN)
    KELUARAN.mkdir(parents=True)

    # Setengah "terburuk", setengah ACAK.
    #
    # Kalau semuanya diambil dari yang terburuk, persentase kotak palsu yang
    # dihitung nanti pasti tinggi -- bukan karena datasetnya begitu, tapi karena
    # cara memilihnya. Jebakan yang sama pernah terjadi pada skrining label.
    #
    #   kelompok "terburuk" -> membuktikan gejalanya NYATA
    #   kelompok "acak"     -> memberi angka yang boleh dikutip
    import random as _rnd
    terpilih, per = [], Counter()
    if a.semua:
        for b in baris:
            b["kelompok"] = "semua"
            terpilih.append(b)
    else:
        for kelas in KELAS_LEMAH:
            g = [b for b in baris if b["kelas"] == kelas]
            n_buruk = a.jumlah // 2
            buruk = g[:n_buruk]
            sisa = g[n_buruk:]
            rng = _rnd.Random(a.seed)
            acak = rng.sample(sisa, min(a.jumlah - n_buruk, len(sisa)))
            for b in buruk:
                b["kelompok"] = "terburuk"
            for b in acak:
                b["kelompok"] = "acak"
            terpilih.extend(buruk + acak)

    # Satu folder berisi SEMUA kelas, bukan dipisah per kelas.
    # Roboflow memetakan ulang id kelas menurut apa yang ditemukannya saat
    # unggah; kalau dua kelas diunggah sebagai dua proyek terpisah, id-nya jadi
    # 0 dua kali dan tercampur saat digabungkan kembali.
    unggah = KELUARAN / "roboflow_upload"
    unggah.mkdir(parents=True, exist_ok=True)
    for b in terpilih:
        shutil.copy2(b["_pc"], unggah / b["citra"])
        shutil.copy2(b["_pl"], unggah / f"{Path(b['citra']).stem}.txt")
    tulis_classes(unggah)
    tulis_classes(KELUARAN)
    (unggah / "_darknet.labels").write_text(
        "\n".join(NAMA_KELAS[i] for i in sorted(NAMA_KELAS)) + "\n", encoding="utf-8")

    with open(KELUARAN / "prioritas.csv", "w", newline="", encoding="utf-8-sig") as f:
        kol = ["citra", "kelas", "kelompok", "subset", "n_label", "n_prediksi",
               "tidak_ditemukan", "persen_terlewat"]
        w = csv.DictWriter(f, fieldnames=kol + ["vonis_kotak", "catatan"])
        w.writeheader()
        for b in sorted(terpilih, key=lambda x: (x["kelas"], x["kelompok"],
                                                 -x["tidak_ditemukan"])):
            w.writerow({**{k: b[k] for k in kol}, "vonis_kotak": "", "catatan": ""})

    print(f"\n{len(terpilih)} citra disiapkan di {KELUARAN / 'roboflow_upload'}")
    print(f"\n{'kelas':24s} {'kelompok':10s} {'citra':>6s} {'kotak':>7s} {'tak ketemu':>11s}")
    for k in KELAS_LEMAH:
        for grp in ["terburuk", "acak", "semua"]:
            g = [b for b in terpilih if b["kelas"] == k and b.get("kelompok") == grp]
            if not g:
                continue
            tl = sum(b["n_label"] for b in g)
            tf = sum(b["tidak_ditemukan"] for b in g)
            print(f"{k:24s} {grp:10s} {len(g):6d} {tl:7d} {tf:6d} ({tf/max(1,tl):3.0%})")

    acak = [b for b in terpilih if b.get("kelompok") == "acak"]
    if acak:
        tl = sum(b["n_label"] for b in acak)
        tf = sum(b["tidak_ditemukan"] for b in acak)
        print(f"\nLaju 'tak ditemukan' pada kelompok ACAK: {tf}/{tl} = {tf/max(1,tl):.0%}")
        print("  <- ini angka yang boleh dikutip. Kelompok 'terburuk' sengaja bias,")
        print("     gunanya cuma membuktikan gejalanya nyata, bukan mengukur.")

    print(f"\nSeluruh citra kelas lemah: {len(baris)} "
          f"({sum(b['n_label'] for b in baris)} kotak) -- pakai --semua kalau mau semuanya.")


if __name__ == "__main__":
    main()
