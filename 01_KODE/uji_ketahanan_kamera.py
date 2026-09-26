#!/usr/bin/env python3
"""
Uji ketahanan: bagaimana model bereaksi pada kondisi kamera yang berbeda dari data latih.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

LATAR. ahli domain melihat bahwa pada deteksi real-time, warna tertentu membuat
sistem menandai penyakit. Seluruh citra latih berlatar putih murni dengan cahaya
seragam; kamera laptop di ruangan memberi cahaya kekuningan, redup, dan latar
meja. Skrip ini MENGUKUR efeknya dengan mensimulasikan kondisi itu pada citra
test, lewat jalur inferensi sistem web yang sama (deteksi.prediksi).

Ini simulasi, bukan rekaman kamera sungguhan. Tujuannya mengukur arah dan
besarnya efek, untuk bahan paragraf keterbatasan -- bukan memperbaikinya.

Pakai:
    python 01_KODE\\uji_ketahanan_kamera.py --uji
    python 01_KODE\\uji_ketahanan_kamera.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "07_SISTEM_WEB"))

DS = AKAR / "02_DATASET" / "dataset_80_10_10"
KELUAR = AKAR / "05_HASIL" / "uji_ketahanan"
NAMA = {0: "healthy_leaf", 1: "mosaic_virus_disease",
        2: "leaf_spot_disease", 3: "insect_pest_disease"}
LESI = {"leaf_spot_disease", "insect_pest_disease"}
RNG = np.random.default_rng(0)


# ---------------- simulasi kondisi kamera (semua BGR uint8) ----------------
def latar_mask(im):
    """Piksel latar putih: saturasi rendah dan terang."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < 40) & (hsv[..., 2] > 200)


def hangat(im):
    """Cahaya lampu ruangan kekuningan (~3000 K): biru turun, merah naik."""
    f = np.array([0.72, 0.95, 1.08])            # B, G, R
    return np.clip(im.astype(np.float32) * f, 0, 255).astype(np.uint8)


def redup(im):
    return np.clip(im.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)


def latar_meja(im):
    """Latar putih diganti warna meja kayu bertekstur."""
    out = im.copy()
    m = latar_mask(im)
    meja = np.array([70, 105, 145], np.float32)      # BGR coklat kayu
    noise = RNG.normal(0, 14, im.shape).astype(np.float32)
    out[m] = np.clip(meja + noise[m], 0, 255).astype(np.uint8)
    return out


def live640(im):
    """Mode live: frame dikecilkan ke sisi terpanjang 640 px."""
    h, w = im.shape[:2]
    s = 640 / max(h, w)
    return cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


KONDISI = {
    "asli":            lambda im: im,
    "live_640":        live640,
    "cahaya_hangat":   hangat,
    "redup":           redup,
    "latar_meja":      latar_meja,
    "hangat+meja":     lambda im: hangat(latar_meja(im)),
}


def kelas_label(p):
    k = [int(float(b.split()[0])) for b in p.read_text().splitlines() if b.strip()]
    return NAMA[Counter(k).most_common(1)[0][0]] if k else None


def main():
    import deteksi
    KELUAR.mkdir(parents=True, exist_ok=True)
    citra = sorted((DS / "images" / "test").glob("*.jpg"))
    benar = {p.stem: kelas_label(DS / "labels" / "test" / f"{p.stem}.txt") for p in citra}

    hasil = {k: {} for k in KONDISI}
    for i, p in enumerate(citra, 1):
        im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        for nama, f in KONDISI.items():
            h = deteksi.prediksi(f(im))
            hasil[nama][p.stem] = {
                "tebak": h["kelas"], "conf_maks": h["conf_maks"],
                "n_lesi": sum(1 for b in h["kotak"] if b["kelas"] in LESI),
            }
        if i % 20 == 0:
            print(f"  {i}/{len(citra)}", flush=True)

    rekap = {}
    for nama, r in hasil.items():
        n = len(r)
        benar_n = sum(1 for s, v in r.items() if v["tebak"] == benar[s])
        utuh = [s for s in r if benar[s] in ("healthy_leaf", "mosaic_virus_disease")]
        sehat = [s for s in r if benar[s] == "healthy_leaf"]
        salah = [s for s in r if r[s]["tebak"] != benar[s]]
        rekap[nama] = {
            "akurasi": benar_n / n, "benar": benar_n, "n": n,
            # daun utuh (healthy/mosaic) yang berubah dianggap penyakit lesi
            "utuh_jadi_lesi": sum(1 for s in utuh if r[s]["tebak"] in LESI) / len(utuh),
            "sehat_jadi_lesi": sum(1 for s in sehat if r[s]["tebak"] in LESI) / len(sehat),
            "kotak_lesi_per_daun_sehat": float(np.mean([r[s]["n_lesi"] for s in sehat])),
            # apakah peringatan keyakinan rendah (<0,25) menyala saat tebakan salah?
            "peringatan_pada_salah": (sum(1 for s in salah if r[s]["conf_maks"] < 0.25) / len(salah))
                                     if salah else 0.0,
            "n_salah": len(salah), "n_utuh": len(utuh), "n_sehat": len(sehat),
        }

    print(f"\n{'kondisi':15s} {'akurasi':>9s} {'utuh→lesi':>10s} {'sehat→lesi':>11s} "
          f"{'kotak lesi/daun sehat':>22s} {'peringatan saat salah':>22s}")
    for nama, v in rekap.items():
        print(f"{nama:15s} {v['akurasi']:8.3f}  {100*v['utuh_jadi_lesi']:8.1f}%  "
              f"{100*v['sehat_jadi_lesi']:9.1f}%  {v['kotak_lesi_per_daun_sehat']:20.2f}  "
              f"{100*v['peringatan_pada_salah']:20.1f}%")

    (KELUAR / "hasil.json").write_text(json.dumps({"rekap": rekap, "per_citra": hasil,
                                                   "benar": benar}, indent=1), encoding="utf-8")
    for nama in ("asli", "cahaya_hangat", "latar_meja", "hangat+meja"):
        p = citra[0]
        im = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(KELUAR / f"contoh_{nama.replace('+', '_')}.jpg"), live640(KONDISI[nama](im)))
    print(f"\nhasil -> {KELUAR}")


def uji():
    im = np.full((100, 120, 3), 255, np.uint8)
    im[30:70, 40:80] = (40, 140, 60)                       # "daun" hijau di tengah
    assert latar_mask(im).sum() == 100 * 120 - 40 * 40
    m = latar_meja(im)
    assert (m[30:70, 40:80] == (40, 140, 60)).all(), "daun tidak boleh tersentuh"
    assert m[0, 0].mean() < 200, "latar putih harus terganti"
    hw = hangat(im)
    assert hw[0, 0, 0] < hw[0, 0, 2], "cahaya hangat: biru < merah"
    assert redup(im).max() <= 141
    assert max(live640(np.zeros((3000, 4000, 3), np.uint8)).shape[:2]) == 640
    assert set(KONDISI) >= {"asli", "live_640", "cahaya_hangat", "redup", "latar_meja"}
    print("SEMUA UJI LOLOS")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--uji", action="store_true")
    uji() if a.parse_args().uji else main()
