#!/usr/bin/env python3
"""
Tahap 1b SAHI: jalankan SAHI sekali (simpan deteksi mentah), lalu sapu aturan
penggabungan x ambang conf secara offline. Menjawab: bisakah SAHI di-tuning
mengalahkan inferensi standar pada akurasi tingkat-citra?

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

Temuan Tahap 1: SAHI menaikkan recall lesi (0,74->0,97) tapi over-deteksi (4x
kotak) sehingga aturan jumlah_conf pada conf 0,05 salah menebak healthy/mosaic
sebagai lesi. Dugaan: pasangan (conf longgar) x (aturan peka-jumlah) yang salah.
Skrip ini mengujinya -- aturan conf_maks tahan terhadap jumlah kotak.

PROTOKOL: aturan+ambang dipilih dari VAL, dilaporkan di TEST.

Deteksi mahal (SAHI ubin di CPU ~75 mnt) di-cache ke JSON. Sweep offline instan.

Pakai:
    python 01_KODE\\uji_sahi_sweep.py --uji
    python 01_KODE\\uji_sahi_sweep.py                # infer (val+test) lalu sweep
    python 01_KODE\\uji_sahi_sweep.py --pakai-cache  # sweep saja dari cache
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
DIR_DATASET = AKAR / "02_DATASET" / "dataset_80_10_10"
DIR_KAGGLE = AKAR / "04_KAGGLE" / "FOLDER_HASIL_KAGGLE"
CACHE = AKAR / "05_HASIL" / "sahi_deteksi_cache.json"

NAMA = {0: "healthy_leaf", 1: "mosaic_virus_disease",
        2: "leaf_spot_disease", 3: "insect_pest_disease"}
MODEL_DEFAULT = DIR_KAGGLE / "hasil_yolo11_ablasi_resolusi_dataset_baru" / "img1024" / "best_yolo11s_img1024.pt"
IMGSZ = 1024
SLICE = 1024
OVERLAP = 0.2
CONF_INFER = 0.05      # ambang saat inferensi; sweep offline hanya bisa naik dari sini
AMBANG = [round(0.05 * k, 2) for k in range(1, 11)]   # 0.05..0.50


def kelas_mayoritas(ids):
    return Counter(ids).most_common(1)[0][0] if ids else None


# ---- aturan penggabungan: (kelas, skor) deteksi -> satu tebakan ----
def r_jumlah_conf(dets):
    if not dets:
        return None
    t = defaultdict(float)
    for c, s in dets:
        t[c] += s
    return max(t, key=t.get)


def r_conf_maks(dets):
    return max(dets, key=lambda d: d[1])[0] if dets else None


def r_jumlah_kotak(dets):
    if not dets:
        return None
    t = Counter(c for c, _ in dets)
    return t.most_common(1)[0][0]


ATURAN = {"jumlah_conf": r_jumlah_conf, "conf_maks": r_conf_maks,
          "jumlah_kotak": r_jumlah_kotak}


def akurasi(sampel, mode, aturan, ambang):
    """sampel: list dict {y, standar:[[c,s]..], sahi:[..]}. Kembalikan (akur, tanpa)."""
    fn = ATURAN[aturan]
    benar = tanpa = total = 0
    for s in sampel:
        y = s["y"]
        if y is None:
            continue
        total += 1
        dets = [(c, sc) for c, sc in s[mode] if sc >= ambang]
        pred = fn(dets)
        if pred is None:
            tanpa += 1
        elif pred == y:
            benar += 1
    return (benar / total if total else 0.0), tanpa, total


def pilih_dari_val(val, mode):
    """Sapu aturan x ambang, pilih akurasi val tertinggi."""
    terbaik = None
    for aturan in ATURAN:
        for amb in AMBANG:
            ak, _, _ = akurasi(val, mode, aturan, amb)
            if terbaik is None or ak > terbaik[0]:
                terbaik = (ak, aturan, amb)
    return terbaik  # (akur_val, aturan, ambang)


def recall_deteksi(sampel, mode, ambang):
    num, den = defaultdict(int), defaultdict(int)
    for s in sampel:
        lbl = Counter(s["lbl_kelas"])
        det = Counter(c for c, sc in s[mode] if sc >= ambang)
        for c, j in lbl.items():
            den[c] += j
            num[c] += min(j, det.get(c, 0))
    return {c: (num[c] / den[c] if den[c] else 0.0) for c in NAMA}


def infer(split, model):
    from sahi.predict import get_prediction, get_sliced_prediction
    imgs = sorted((DIR_DATASET / "images" / split).glob("*.jpg"))
    out = []
    for i, img in enumerate(imgs, 1):
        lbl = DIR_DATASET / "labels" / split / f"{img.stem}.txt"
        ids = [int(b.split()[0]) for b in lbl.read_text().splitlines() if b.strip()] if lbl.exists() else []
        r_std = get_prediction(str(img), model)
        r_slc = get_sliced_prediction(str(img), model, slice_height=SLICE, slice_width=SLICE,
                                      overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, verbose=0)
        out.append({
            "y": kelas_mayoritas(ids), "lbl_kelas": ids,
            "standar": [[int(o.category.id), round(float(o.score.value), 4)] for o in r_std.object_prediction_list],
            "sahi": [[int(o.category.id), round(float(o.score.value), 4)] for o in r_slc.object_prediction_list],
        })
        if i % 25 == 0:
            print(f"  {split} {i}/{len(imgs)}")
    return out


def uji():
    assert r_conf_maks([(2, 0.9), (3, 0.1), (3, 0.1)]) == 2
    assert r_jumlah_conf([(2, 0.9), (3, 0.1), (3, 0.1)]) == 2
    assert r_jumlah_conf([(2, 0.3), (3, 0.3), (3, 0.3)]) == 3
    assert r_jumlah_kotak([(2, 0.9), (3, 0.1), (3, 0.1)]) == 3
    assert r_conf_maks([]) is None
    s = [{"y": 0, "lbl_kelas": [0], "standar": [[0, 0.9]], "sahi": [[0, 0.9], [2, 0.06]]}]
    assert akurasi(s, "standar", "conf_maks", 0.05)[0] == 1.0
    assert akurasi(s, "sahi", "jumlah_kotak", 0.05)[0] == 1.0   # 0 muncul 1x, 2 muncul 1x -> most_common ambil 0 (pertama)
    # kunci bug 2026-09-13: nilai dari sahi berupa numpy, harus JSON-serializable
    import numpy as np
    contoh = {"standar": [[int(np.int64(2)), round(float(np.float32(0.5)), 4)]], "sahi": []}
    json.dumps(contoh)   # meledak kalau lupa cast
    print("SEMUA UJI LOLOS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uji", action="store_true")
    ap.add_argument("--pakai-cache", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if args.uji:
        uji()
        return

    if args.pakai_cache and CACHE.exists():
        data = json.loads(CACHE.read_text())
        print(f"cache dipakai: {CACHE}")
    else:
        model_path = Path(args.model) if args.model else MODEL_DEFAULT
        from sahi import AutoDetectionModel
        try:
            model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(model_path),
                                                       confidence_threshold=CONF_INFER, image_size=IMGSZ, device="cpu")
        except Exception:
            model = AutoDetectionModel.from_pretrained(model_type="yolov8", model_path=str(model_path),
                                                       confidence_threshold=CONF_INFER, image_size=IMGSZ, device="cpu")
        print("inferensi val...")
        val = infer("val", model)
        print("inferensi test...")
        test = infer("test", model)
        data = {"val": val, "test": test}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data))
        print(f"cache ditulis: {CACHE}")

    val, test = data["val"], data["test"]

    print("\n" + "=" * 74)
    print("  PILIH ATURAN+AMBANG DARI VAL, LAPORKAN DI TEST")
    print("=" * 74)
    for mode in ("standar", "sahi"):
        akv, aturan, amb = pilih_dari_val(val, mode)
        akt, tanpa, total = akurasi(test, mode, aturan, amb)
        print(f"\n### {mode.upper()}")
        print(f"  terpilih dari val : aturan '{aturan}', conf {amb}  (val akurasi {akv:.4f})")
        print(f"  akurasi TEST      : {akt:.4f}  ({round(akt*total)}/{total}) | tanpa deteksi {tanpa}")
        rec = recall_deteksi(test, mode, amb)
        print("  recall deteksi test per kelas:")
        for c in sorted(NAMA):
            print(f"    {NAMA[c]:22s} {rec[c]:.3f}")

    # tabel ringkas conf_maks lintas ambang di TEST (untuk melihat kurva)
    print("\n" + "-" * 74)
    print("  conf_maks di TEST lintas ambang (diagnostik)")
    print("-" * 74)
    print(f"  {'conf':>5s}  {'std akur':>9s}  {'sahi akur':>9s}")
    for amb in AMBANG:
        a_s, _, _ = akurasi(test, "standar", "conf_maks", amb)
        a_h, _, _ = akurasi(test, "sahi", "conf_maks", amb)
        print(f"  {amb:5.2f}  {a_s:9.4f}  {a_h:9.4f}")


if __name__ == "__main__":
    main()
