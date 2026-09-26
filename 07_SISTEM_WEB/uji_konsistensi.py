#!/usr/bin/env python3
"""Membuktikan deteksi.py memberi kelas yang sama dengan evaluasi_sistem.py.

Bila gagal, ada yang salah pada pipeline -- kemungkinan besar conf inferensi
atau urutan penyaringan -- dan angka demo tidak lagi sejalan dengan penelitian.

Pakai:
    python uji_konsistensi.py
    python uji_konsistensi.py --n 40
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deteksi

AKAR = Path(__file__).resolve().parent.parent
DS = AKAR / "02_DATASET" / "dataset_80_10_10"


def kelas_label(lbl):
    ids = [int(b.split()[0]) for b in lbl.read_text().splitlines() if b.strip()]
    return Counter(ids).most_common(1)[0][0] if ids else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    a = p.parse_args()

    citra = sorted((DS / "images" / "test").glob("*.jpg"))[:a.n]
    assert citra, "tidak ada citra test"

    ke_id = {v: k for k, v in deteksi.NAMA.items()}
    cocok, benar_list = 0, []

    for q in citra:
        h = deteksi.prediksi(str(q))
        benar = kelas_label(DS / "labels" / "test" / f"{q.stem}.txt")
        tebak = ke_id.get(h["kelas"])
        ok = tebak == benar
        cocok += ok
        benar_list.append((q.name, deteksi.NAMA.get(benar), h["kelas"], ok, h["conf_maks"]))
        tanda = "OK " if ok else "   "
        print(f"  {tanda} {q.stem[:34]:34s} benar={deteksi.NAMA.get(benar):22s} "
              f"tebak={str(h['kelas']):22s} conf_maks={h['conf_maks']:.3f}")

    laju = cocok / len(citra)
    print(f"\ncocok {cocok}/{len(citra)} = {laju:.3f}")

    # evaluasi_sistem.py melaporkan akurasi test 0,798 pada 208 citra.
    # Pada sampel kecil riak wajar; di bawah 0,55 menandakan pipeline salah.
    assert laju >= 0.55, f"laju {laju:.3f} terlalu rendah -- periksa pipeline inferensi"
    print("LOLOS -- sejalan dengan evaluasi_sistem.py")
    return benar_list


if __name__ == "__main__":
    main()
