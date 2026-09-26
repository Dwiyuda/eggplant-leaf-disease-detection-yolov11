#!/usr/bin/env python3
"""
Membangun ulang dataset_80_10_10 dan dataset_70_15_15 dari citra + label terkini.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

WAJIB dijalankan setelah citra berpindah kelas. Split lama dibuat Juli dari
komposisi kelas yang sudah tidak berlaku: 379 citra pindah, 2 dibuang. Kalau
tidak dibangun ulang, proporsi kelas tiap split jadi timpang dan sebagian citra
punya label yang tidak lagi cocok dengan foldernya.

TIGA SYARAT yang dijaga, sama seperti pembuatan pertama:

  1. STRATIFIED per kelas
     Proporsi keempat kelas dijaga sama di train, val, dan test. Tanpa ini,
     kelas kecil bisa nyaris tidak muncul di test dan angkanya jadi tak berarti.

  2. SADAR-KLASTER duplikat-mirip
     Citra yang nyaris identik WAJIB masuk split yang sama. Kalau satu di train
     dan kembarannya di test, model seolah sudah pernah melihat soal ujiannya
     dan skor test jadi lebih bagus dari yang sebenarnya. Klaster dibaca dari
     neardup_klaster.csv hasil deduplikasi Juli -- masih sah karena kemiripan
     citra tidak berubah oleh perpindahan kelas.

  3. NOL citra tanpa label
     Diperiksa sebelum menulis, bukan sesudah.

Kelas sebuah citra ditentukan dari isi LABEL-nya (kelas mayoritas), bukan dari
nama foldernya -- 44 citra kini memuat kotak dua kelas sekaligus.

Pakai:
    python 01_KODE\\bangun_split.py --uji        # self-check
    python 01_KODE\\bangun_split.py              # KERING: laporan rencana
    python 01_KODE\\bangun_split.py --terapkan   # sungguhan
"""

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from audit_prediksi import NAMA_KELAS, DIR_DATASET, DIR_HASIL

DIR_SUMBER = DIR_DATASET.parent / "03_CITRA_SUMBER"
DIR_LABEL = DIR_HASIL / "autolabel_hasil" / "labels"
KLASTER = DIR_HASIL / "autolabel_hasil" / "neardup_klaster.csv"

VARIAN = {"dataset_80_10_10": (0.80, 0.10, 0.10),
          "dataset_70_15_15": (0.70, 0.15, 0.15)}
SEED = 42


def kumpulkan():
    """Kembalikan [(stem, path_citra, path_label, kelas_mayoritas)] + daftar masalah."""
    data, masalah = [], []
    for kelas in sorted(NAMA_KELAS.values()):
        dc, dl = DIR_SUMBER / kelas, DIR_LABEL / kelas
        if not dc.is_dir():
            continue
        for p in sorted(dc.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            t = dl / f"{p.stem}.txt"
            if not t.exists():
                masalah.append(f"tanpa label: {kelas}/{p.name}")
                continue
            ids = [int(float(b.split()[0])) for b in t.read_text().split("\n") if b.strip()]
            if not ids:
                masalah.append(f"label kosong: {kelas}/{p.name}")
                continue
            data.append((p.stem, p, t, Counter(ids).most_common(1)[0][0]))
    return data, masalah


def baca_keanggotaan_lama(nama):
    """
    {stem: split} dari zip dataset lama di 99_ARSIP.

    Dipakai mode --pertahankan-lama, yang mempertahankan citra tetap di split
    asalnya. Gunanya untuk PERBANDINGAN yang sah: kalau keanggotaan test tidak
    berubah, satu-satunya yang berbeda antara model lama dan model baru adalah
    kualitas labelnya. Dengan split acak baru, 82% citra test baru ternyata dulu
    ada di train lama, sehingga model lama sudah pernah melihatnya dan angkanya
    tidak bisa dibandingkan.
    """
    import zipfile
    z = DIR_DATASET.parent / "99_ARSIP" / f"{nama}.zip"
    if not z.exists():
        return {}
    peta = {}
    with zipfile.ZipFile(z) as f:
        for n in f.namelist():
            for s in ("train", "val", "test"):
                if f"/images/{s}/" in n and n.lower().endswith((".jpg", ".jpeg", ".png")):
                    peta[Path(n).stem] = s
    return peta


def bagi_pertahankan(data, lama, peta_klaster=None, lapor_perbaikan=True):
    """
    Tempatkan tiap citra di split asalnya. Citra tak dikenal masuk train.

    Split lama tidak dijamin bebas kebocoran klaster. Split 70:15:15 Juli
    ternyata memuat satu klaster duplikat-mirip LINTAS KELAS yang anggotanya
    tersebar di train dan val -- cacat yang tidak pernah terdeteksi karena
    pembagian Juli mengelompokkan klaster per kelas.

    Karena itu keanggotaan lama dipertahankan KECUALI bila melanggar keutuhan
    klaster. Klaster yang tersebar dirapatkan ke split yang memuat anggota
    terbanyak, dan perpindahannya dilaporkan supaya tercatat, bukan diam-diam.
    """
    tempat = {d[0]: lama.get(d[0], "train") for d in data}

    if peta_klaster:
        anggota = defaultdict(list)
        for d in data:
            k = peta_klaster.get(d[0])
            if k:
                anggota[k].append(d[0])
        for k, stems in anggota.items():
            tersebar = {tempat[s] for s in stems}
            if len(tersebar) > 1:
                menang = Counter(tempat[s] for s in stems).most_common(1)[0][0]
                pindah = [s for s in stems if tempat[s] != menang]
                for s in pindah:
                    tempat[s] = menang
                if lapor_perbaikan:
                    print(f"  [perbaikan klaster] {k}: {len(pindah)} citra dirapatkan "
                          f"ke '{menang}' -- {', '.join(s[:34] for s in pindah)}")

    hasil = {"train": [], "val": [], "test": []}
    for d in data:
        hasil[tempat[d[0]]].append(d)
    return hasil


def baca_klaster():
    """{stem: id_klaster} dari neardup_klaster.csv. Dicocokkan lewat nama berkas."""
    peta = {}
    if not KLASTER.exists():
        return peta
    for r in csv.DictReader(open(KLASTER, encoding="utf-8")):
        peta[Path(r["file"].replace("\\", "/")).stem] = f"k{r['klaster_id']}"
    return peta


def bagi(data, rasio, peta_klaster, seed=SEED):
    """
    Bagi citra ke train/val/test.

    Klaster dikelompokkan SECARA GLOBAL, bukan per kelas. Ini penting: setelah
    379 citra berpindah kelas, anggota satu klaster duplikat-mirip bisa berada
    di kelas yang berbeda. Kalau pengelompokan dilakukan per kelas, anggota
    klaster yang sama bisa jatuh ke split berbeda -- persis kebocoran yang
    hendak dicegah.

    Karena satu klaster bisa memuat beberapa kelas sekaligus, proporsi kelas
    tidak bisa dipenuhi persis. Dipakai penempatan serakah: tiap klaster
    diletakkan di split yang saat itu paling kekurangan kelas dominan klaster
    tersebut. Klaster besar didahulukan supaya yang paling mengikat ditempatkan
    saat pilihan masih longgar.
    """
    rng = random.Random(seed)
    SPLIT = ["train", "val", "test"]

    grup = defaultdict(list)
    for d in data:
        grup[peta_klaster.get(d[0], f"tunggal:{d[0]}")].append(d)
    klaster = list(grup.values())
    rng.shuffle(klaster)
    klaster.sort(key=len, reverse=True)

    total = Counter(d[3] for d in data)
    target = {s: {k: total[k] * r for k in total} for s, r in zip(SPLIT, rasio)}
    kini = {s: Counter() for s in SPLIT}
    hasil = {s: [] for s in SPLIT}

    for k in klaster:
        dom = Counter(d[3] for d in k).most_common(1)[0][0]
        # kekurangan relatif, supaya split kecil tidak selalu kalah dari train
        pilih = max(SPLIT, key=lambda s: (target[s][dom] - kini[s][dom])
                    / max(1.0, target[s][dom]))
        hasil[pilih].extend(k)
        kini[pilih].update(d[3] for d in k)
    return hasil


def periksa(hasil, peta_klaster):
    """Kembalikan daftar pelanggaran. Kosong berarti aman."""
    salah = []
    milik = {}
    for s, isi in hasil.items():
        for d in isi:
            milik[d[0]] = s
    tempat = defaultdict(set)
    for stem, s in milik.items():
        k = peta_klaster.get(stem)
        if k:
            tempat[k].add(s)
    for k, s in tempat.items():
        if len(s) > 1:
            salah.append(f"klaster {k} tersebar di {sorted(s)}")
    semua = [d[0] for isi in hasil.values() for d in isi]
    if len(semua) != len(set(semua)):
        salah.append("ada citra muncul di lebih dari satu split")
    return salah


def tulis(nama, hasil, terapkan):
    akar = DIR_DATASET / nama
    if terapkan:
        if akar.exists():
            shutil.rmtree(akar)
        for s in hasil:
            (akar / "images" / s).mkdir(parents=True, exist_ok=True)
            (akar / "labels" / s).mkdir(parents=True, exist_ok=True)
        for s, isi in hasil.items():
            for stem, pc, pl, _ in isi:
                shutil.copy2(pc, akar / "images" / s / pc.name)
                shutil.copy2(pl, akar / "labels" / s / f"{stem}.txt")
        (akar / "data.yaml").write_text(yaml.safe_dump({
            "path": str(akar), "train": "images/train", "val": "images/val",
            "test": "images/test", "nc": len(NAMA_KELAS),
            "names": {int(k): v for k, v in NAMA_KELAS.items()},
        }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return akar


def lapor(nama, hasil):
    print(f"\n{nama}")
    print(f"  {'split':6s} {'citra':>6s} {'kotak':>7s}   " +
          "  ".join(f"{v[:12]:>12s}" for v in NAMA_KELAS.values()))
    for s in ["train", "val", "test"]:
        isi = hasil[s]
        n_kotak = 0
        per = Counter()
        for stem, pc, pl, km in isi:
            for b in pl.read_text().split("\n"):
                if b.strip():
                    per[int(float(b.split()[0]))] += 1
                    n_kotak += 1
        tot = len(isi)
        print(f"  {s:6s} {tot:6d} {n_kotak:7d}   " +
              "  ".join(f"{sum(1 for d in isi if d[3]==i):5d} "
                        f"({sum(1 for d in isi if d[3]==i)/max(1,tot):4.0%})"
                        for i in sorted(NAMA_KELAS)))


def uji():
    lolos = True

    def cek(n, s):
        nonlocal lolos
        print(f"  {'ok  ' if s else 'GAGAL'}  {n}")
        lolos = lolos and s

    P = Path("x")
    data = [(f"a{i}", P, P, i % 4) for i in range(400)]

    print("bagi() tanpa klaster")
    h = bagi(data, (0.8, 0.1, 0.1), {})
    tot = sum(len(v) for v in h.values())
    cek("semua citra terpakai", tot == 400)
    cek("tidak ada yang dobel", len({d[0] for v in h.values() for d in v}) == 400)
    cek("train sekitar 80%", 0.74 <= len(h["train"]) / 400 <= 0.86)
    cek("val tidak kosong", len(h["val"]) > 0)
    cek("test tidak kosong", len(h["test"]) > 0)
    for i in range(4):
        p = sum(1 for d in h["train"] if d[3] == i) / max(1, len(h["train"]))
        if not 0.15 <= p <= 0.35:
            cek(f"proporsi kelas {i} di train wajar", False)
            break
    else:
        cek("proporsi tiap kelas di train wajar", True)

    print("bagi() dengan klaster")
    peta = {f"a{i}": "kA" for i in range(0, 40, 4)}      # 10 citra sekelas, satu klaster
    h = bagi(data, (0.8, 0.1, 0.1), peta)
    cek("semua citra tetap terpakai", sum(len(v) for v in h.values()) == 400)
    cek("klaster tidak tersebar", periksa(h, peta) == [])

    print("bagi() dengan klaster LINTAS KELAS (kasus yang dulu bocor)")
    # a0 kelas 0, a1 kelas 1, a2 kelas 2 -- satu klaster, tiga kelas berbeda
    peta2 = {"a0": "kX", "a1": "kX", "a2": "kX"}
    h2 = bagi(data, (0.8, 0.1, 0.1), peta2)
    cek("klaster lintas kelas tetap utuh", periksa(h2, peta2) == [])
    tempat = {s for s, isi in h2.items() for d in isi if d[0] in peta2}
    cek("ketiganya di satu split", len(tempat) == 1)
    cek("tidak ada citra hilang", sum(len(v) for v in h2.values()) == 400)

    print("periksa() mendeteksi pelanggaran")
    rusak = {"train": [("a0", P, P, 0)], "val": [("a4", P, P, 0)], "test": []}
    cek("klaster tersebar terdeteksi", len(periksa(rusak, peta)) >= 1)
    dobel = {"train": [("z", P, P, 0)], "val": [("z", P, P, 0)], "test": []}
    cek("citra dobel terdeteksi", any("lebih dari satu split" in s
                                      for s in periksa(dobel, {})))
    cek("split bersih -> tanpa pelanggaran",
        periksa({"train": [("p", P, P, 0)], "val": [("q", P, P, 1)],
                 "test": [("r", P, P, 2)]}, {}) == [])

    print("bagi_pertahankan()")
    lama = {f"a{i}": ("val" if i < 40 else "test" if i < 80 else "train")
            for i in range(400)}
    h = bagi_pertahankan(data, lama, None, lapor_perbaikan=False)
    cek("keanggotaan lama dipertahankan", len(h["val"]) == 40 and len(h["test"]) == 40)
    cek("sisanya ke train", len(h["train"]) == 320)

    # a0 di val, a100 di train -- satu klaster, harus dirapatkan
    pk = {"a0": "kZ", "a100": "kZ", "a101": "kZ"}
    h = bagi_pertahankan(data, lama, pk, lapor_perbaikan=False)
    cek("klaster tersebar dirapatkan", periksa(h, pk) == [])
    letak = {s for s, isi in h.items() for d in isi if d[0] in pk}
    cek("ketiganya satu split", len(letak) == 1)
    cek("dirapatkan ke mayoritas (train, 2 dari 3)", letak == {"train"})
    cek("tidak ada citra hilang", sum(len(v) for v in h.values()) == 400)

    h = bagi_pertahankan(data, lama, {"a0": "kQ", "a1": "kQ"}, lapor_perbaikan=False)
    cek("klaster yang sudah utuh tidak diganggu",
        all(d[0] not in ("a0", "a1") for d in h["train"]))

    print("determinisme")
    a = bagi(data, (0.8, 0.1, 0.1), {})
    b = bagi(data, (0.8, 0.1, 0.1), {})
    cek("seed sama -> hasil sama",
        [d[0] for d in a["train"]] == [d[0] for d in b["train"]])

    print("\n" + ("SEMUA UJI LOLOS" if lolos else "ADA YANG GAGAL"))
    return 0 if lolos else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--terapkan", action="store_true", help="sungguhan menulis dataset")
    p.add_argument("--pertahankan-lama", action="store_true",
                   help="pertahankan keanggotaan split lama dari zip di 99_ARSIP. "
                        "Dipakai agar hasil model baru bisa dibandingkan langsung "
                        "dengan model lama pada test set yang sama.")
    p.add_argument("--hanya", default=None, choices=list(VARIAN),
                   help="bangun satu varian saja. Berguna supaya dataset yang "
                        "sedang diunggah ke Kaggle tidak ikut ditimpa.")
    p.add_argument("--uji", action="store_true")
    a = p.parse_args()
    if a.uji:
        sys.exit(uji())

    data, masalah = kumpulkan()
    print(f"citra berlabel : {len(data)}")
    if masalah:
        print(f"BERMASALAH     : {len(masalah)}")
        for m in masalah[:5]:
            print("   ", m)
        sys.exit("\nPerbaiki dulu sebelum membangun split.")

    peta = baca_klaster()
    terpakai = sum(1 for d in data if d[0] in peta)
    print(f"klaster mirip  : {len(set(peta.values()))} klaster, "
          f"{terpakai} citra cocok di dataset sekarang")
    print(f"sebaran kelas  : " +
          ", ".join(f"{NAMA_KELAS[i]}={sum(1 for d in data if d[3]==i)}"
                    for i in sorted(NAMA_KELAS)))

    dikerjakan = {a.hanya: VARIAN[a.hanya]} if a.hanya else VARIAN
    if a.hanya:
        print(f"\n[hanya] cuma {a.hanya} yang dibangun; varian lain tidak disentuh")

    for nama, rasio in dikerjakan.items():
        if a.pertahankan_lama:
            lama = baca_keanggotaan_lama(nama)
            if not lama:
                sys.exit(f"Zip lama tidak ada: 99_ARSIP/{nama}.zip")
            cocok = sum(1 for d in data if d[0] in lama)
            print(f"\n[pertahankan-lama] {cocok}/{len(data)} citra ditemukan di split lama")
            hasil = bagi_pertahankan(data, lama, peta)
        else:
            hasil = bagi(data, rasio, peta)
        salah = periksa(hasil, peta)
        lapor(nama, hasil)
        if salah:
            print("  PELANGGARAN:")
            for s in salah:
                print("    ", s)
            sys.exit("\nDihentikan, split tidak sah.")
        print("  aman: nol kebocoran klaster, nol citra dobel")
        if a.terapkan:
            akar = tulis(nama, hasil, True)
            print(f"  ditulis -> {akar}")

    if not a.terapkan:
        print("\n" + "-" * 74)
        print("MODE KERING. Belum ada berkas ditulis.")
        print("Kalau sebaran di atas masuk akal:")
        print("    python 01_KODE\\bangun_split.py --terapkan")
        print("-" * 74)
    else:
        print("\nSelesai. Berikutnya: unggah dataset_80_10_10 ke Kaggle, latih ulang.")


if __name__ == "__main__":
    main()
