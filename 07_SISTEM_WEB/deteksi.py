#!/usr/bin/env python3
"""
Pembungkus model deteksi penyakit daun terung.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

Tidak tahu apa-apa soal HTTP atau Flask. Bisa dijalankan sendiri.

PIPELINE DIKUNCI, identik dengan 01_KODE/evaluasi_sistem.py:

    predict(conf=0.01, imgsz=1024) -> saring conf >= 0.05 -> gabung jumlah_conf

Memanggil predict langsung dengan conf=0.05 DILARANG: NMS berperilaku berbeda
dan hasilnya menyimpang dari angka 79,8% yang dilaporkan di penelitian.

Pakai:
    python deteksi.py --uji                 # self-check
    python deteksi.py --citra <path.jpg>    # prediksi satu citra
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
BOBOT = (AKAR / "04_KAGGLE" / "FOLDER_HASIL_KAGGLE"
         / "hasil_yolo11_ablasi_resolusi_dataset_baru" / "img1024"
         / "best_yolo11s_img1024.pt")

IMGSZ = 1024
CONF_INFERENSI = 0.01   # DIKUNCI -- sama seperti evaluasi_sistem.py baris 212
AMBANG = 0.05           # penyaring, diterapkan di Python SETELAH inferensi
AMBANG_RAGU = 0.25      # di bawah ini -> peringatan keyakinan rendah

NAMA = {0: "healthy_leaf", 1: "mosaic_virus_disease",
        2: "leaf_spot_disease", 3: "insect_pest_disease"}

_model = None


def gabung(det):
    """Aturan jumlah_conf: kelas dengan total kepercayaan terbesar.

    det: list of (id_kelas: int, conf: float)
    Disalin apa adanya dari gab_jumlah_conf() di evaluasi_sistem.py baris 88.
    """
    if not det:
        return None
    s = defaultdict(float)
    for k, c in det:
        s[k] += c
    return max(s, key=s.get)


def muat(bobot=None):
    """Muat model sekali saja. Idempoten -- panggilan berikutnya memakai cache."""
    global _model
    if _model is None:
        b = Path(bobot) if bobot else BOBOT
        if not b.exists():
            raise FileNotFoundError(
                f"Bobot model tidak ketemu:\n  {b}\n"
                "Periksa apakah folder 04_KAGGLE/FOLDER_HASIL_KAGGLE masih utuh."
            )
        from ultralytics import YOLO
        _model = YOLO(str(b))
    return _model


def prediksi(sumber):
    """Jalankan model pada satu citra, kembalikan hasil siap-JSON.

    sumber: path, bytes, atau ndarray -- apa pun yang diterima ultralytics.
    """
    m = muat()
    t0 = time.perf_counter()
    r = m.predict(sumber, conf=CONF_INFERENSI, imgsz=IMGSZ, verbose=False)[0]
    ms = round((time.perf_counter() - t0) * 1000)

    # xyxyn sudah ternormalisasi 0-1 oleh ultralytics -- tidak ada pembagian manual
    mentah = list(zip(r.boxes.xyxyn.tolist(),
                      (int(c) for c in r.boxes.cls.tolist()),
                      (float(c) for c in r.boxes.conf.tolist())))

    kotak, det = [], []
    for (x1, y1, x2, y2), k, cf in mentah:
        if cf < AMBANG:          # penyaring, BUKAN di dalam predict()
            continue
        det.append((k, cf))
        kotak.append({
            "kelas": NAMA[k], "conf": round(cf, 4),
            "x": round(x1, 4), "y": round(y1, 4),
            "w": round(x2 - x1, 4), "h": round(y2 - y1, 4),
        })

    menang = gabung(det)
    total = sum(cf for _, cf in det)
    milik = [cf for k, cf in det if k == menang]

    return {
        "kelas": NAMA[menang] if menang is not None else None,
        "dominasi": round(sum(milik) / total, 4) if total else 0.0,
        "conf_maks": round(max(milik), 4) if milik else 0.0,
        "kotak": kotak,
        "ms": ms,
    }


def uji():
    # gabung() -- aturan jumlah_conf
    assert gabung([]) is None
    assert gabung([(2, 0.9), (3, 0.4), (3, 0.5), (3, 0.45)]) == 3   # 1.35 > 0.9
    assert gabung([(2, 0.9), (3, 0.4)]) == 2
    assert gabung([(0, 0.7)]) == 0

    # konstanta terkunci
    assert BOBOT.exists(), f"bobot tidak ketemu: {BOBOT}"
    assert CONF_INFERENSI == 0.01 and AMBANG == 0.05 and IMGSZ == 1024
    assert set(NAMA) == {0, 1, 2, 3}

    # bentuk keluaran pada satu citra nyata dari subset test
    contoh = sorted((AKAR / "02_DATASET" / "dataset_80_10_10"
                     / "images" / "test").glob("*.jpg"))
    assert contoh, "tidak ada citra test untuk diuji"
    h = prediksi(str(contoh[0]))
    assert set(h) == {"kelas", "dominasi", "conf_maks", "kotak", "ms"}
    assert h["kelas"] is None or h["kelas"] in NAMA.values()
    assert 0.0 <= h["dominasi"] <= 1.0, h["dominasi"]
    assert 0.0 <= h["conf_maks"] <= 1.0, h["conf_maks"]
    for b in h["kotak"]:
        assert b["conf"] >= AMBANG
        for k in ("x", "y", "w", "h"):
            assert 0.0 <= b[k] <= 1.0, (k, b[k])

    print("SEMUA UJI LOLOS")
    print(f"  citra uji : {contoh[0].name}")
    print(f"  hasil     : {h['kelas']} · dominasi {h['dominasi']} · "
          f"{len(h['kotak'])} kotak · {h['ms']} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uji", action="store_true")
    p.add_argument("--citra", default=None)
    a = p.parse_args()
    if a.uji:
        uji()
    elif a.citra:
        import json
        print(json.dumps(prediksi(a.citra), indent=2, ensure_ascii=False))
    else:
        p.print_help()
