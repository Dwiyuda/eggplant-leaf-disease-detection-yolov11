"""
kalibrasi_autolabel.py — Cari parameter optimal autolabel.py terhadap ground truth Roboflow

Sweep parameter deteksi pada 100 citra audit, skor tiap kombinasi terhadap
anotasi manual Roboflow, cetak tabel hasil.

Jalankan dari akar proyek:
    python 01_KODE/kalibrasi_autolabel.py --uji       # self-check import
    python 01_KODE/kalibrasi_autolabel.py              # sweep penuh (~15-25 menit)
    python 01_KODE/kalibrasi_autolabel.py --cepat      # sweep kecil (~5 menit)
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

# Import autolabel
sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolabel

AKAR = Path(__file__).resolve().parent.parent
GT_DIR = AKAR / "06_AUDIT_KOTAK" / "hasil_analisis" / "label_koreksi_yolo"
PRIORITAS = AKAR / "06_AUDIT_KOTAK" / "prioritas.csv"
HASIL_CSV = AKAR / "06_AUDIT_KOTAK" / "hasil_analisis" / "kalibrasi_hasil.csv"


def muat_ground_truth():
    """Muat label YOLO ground truth dari Roboflow."""
    gt = {}
    for fp in GT_DIR.glob("*.txt"):
        nama = fp.stem + ".jpg"
        kotak = []
        with open(fp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    kotak.append((
                        int(parts[0]),
                        float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]),
                    ))
        gt[nama] = kotak
    return gt


def muat_daftar_citra():
    """Muat daftar citra audit beserta kelas dan kelompoknya."""
    daftar = []
    with open(PRIORITAS, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            nama = row["citra"]
            kelas = row["kelas"]
            kelompok = row["kelompok"]
            # Cari path sumber
            for folder in [kelas, "healthy_leaf", "mosaic_virus_disease",
                           "leaf_spot_disease", "insect_pest_disease"]:
                src = AKAR / "03_CITRA_SUMBER" / folder / nama
                if src.exists():
                    break
            else:
                continue
            id_kelas, mode = autolabel.KELAS[kelas]
            daftar.append({
                "nama": nama, "path": src, "kelas": kelas,
                "id_kelas": id_kelas, "mode": mode, "kelompok": kelompok,
            })
    return daftar


def hitung_iou(a, b):
    """IoU dua kotak YOLO (kelas, cx, cy, w, h)."""
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def skor_satu_citra(pred_kotak, gt_kotak, ambang_iou=0.3):
    """Hitung TP, FP, FN antara prediksi dan ground truth."""
    cocok_gt = set()
    cocok_pred = set()
    pasangan = []
    for i, p in enumerate(pred_kotak):
        for j, g in enumerate(gt_kotak):
            iou = hitung_iou(p, g)
            if iou >= ambang_iou:
                pasangan.append((iou, i, j))
    pasangan.sort(reverse=True)
    for iou, i, j in pasangan:
        if i in cocok_pred or j in cocok_gt:
            continue
        cocok_pred.add(i)
        cocok_gt.add(j)
    tp = len(cocok_pred)
    fp = len(pred_kotak) - tp
    fn = len(gt_kotak) - len(cocok_gt)
    return tp, fp, fn


def jalankan_satu_konfigurasi(daftar_citra, gt, cfg):
    """Proses semua citra dengan satu set parameter, kembalikan skor."""
    total_tp, total_fp, total_fn = 0, 0, 0
    acak_tp, acak_fp, acak_fn = 0, 0, 0

    for item in daftar_citra:
        hasil, _ = autolabel.proses_citra(
            item["path"], item["id_kelas"], item["mode"], cfg)
        if hasil["ukuran"] is None:
            continue

        L, T = hasil["ukuran"]
        pred = []
        for (x, y, w, h) in hasil["kotak"]:
            cx = (x + w / 2) / L
            cy = (y + h / 2) / T
            nw = w / L
            nh = h / T
            pred.append((item["id_kelas"], cx, cy, nw, nh))

        gt_boxes = gt.get(item["nama"], [])
        tp, fp, fn = skor_satu_citra(pred, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if item["kelompok"] == "acak":
            acak_tp += tp
            acak_fp += fp
            acak_fn += fn

    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "acak_tp": acak_tp, "acak_fp": acak_fp, "acak_fn": acak_fn,
    }


def metrik(s):
    """Hitung precision, recall, F1 dari dict skor."""
    p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
    r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


def metrik_acak(s):
    p = s["acak_tp"] / (s["acak_tp"] + s["acak_fp"]) if (s["acak_tp"] + s["acak_fp"]) > 0 else 0
    r = s["acak_tp"] / (s["acak_tp"] + s["acak_fn"]) if (s["acak_tp"] + s["acak_fn"]) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


# ======================================================================
# PARAMETER SWEEP
# ======================================================================

# Tahap 1: parameter deteksi warna — sweep DUA arah
# Temuan sweep cepat: menaikkan threshold membunuh recall.
# Coba juga delta_merah LEBIH RENDAH + min_luas LEBIH TINGGI (tangkap lebih
# banyak lesi tapi buang noise kecil).
SWEEP_WARNA = {
    "delta_merah":    [8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
    "k_merah":        [1.5, 2.0, 2.5, 3.0],
    "min_luas_frac":  [0.00040, 0.00060, 0.0010, 0.0015, 0.0020],
}

# Tahap 2: parameter lubang (sumber palsu kedua)
SWEEP_LUBANG = {
    "min_luas_lubang_frac": [0.00008, 0.00020, 0.00050, 0.0010],
    "maks_sat_lubang":      [12, 16, 20, 25],
}

SWEEP_CEPAT = {
    "delta_merah":    [8.0, 12.0, 18.0],
    "k_merah":        [1.5, 2.0, 3.0],
    "min_luas_frac":  [0.00040, 0.00060, 0.0015],
}


def sweep(daftar, gt, param_grid, cfg_base, label=""):
    """Jalankan grid search, kembalikan daftar hasil terurut F1."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    kombo = list(product(*values))
    print(f"\n{'='*70}")
    print(f"SWEEP {label}: {len(kombo)} kombinasi × {len(daftar)} citra")
    print(f"Parameter: {keys}")
    print(f"{'='*70}")

    hasil_list = []
    t0 = time.time()
    for idx, vals in enumerate(kombo):
        cfg = dict(cfg_base)
        for k, v in zip(keys, vals):
            cfg[k] = v

        skor = jalankan_satu_konfigurasi(daftar, gt, cfg)
        p, r, f1 = metrik(skor)
        pa, ra, f1a = metrik_acak(skor)

        entry = {k: v for k, v in zip(keys, vals)}
        entry.update({
            "tp": skor["tp"], "fp": skor["fp"], "fn": skor["fn"],
            "precision": p, "recall": r, "f1": f1,
            "acak_precision": pa, "acak_recall": ra, "acak_f1": f1a,
        })
        hasil_list.append(entry)

        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (len(kombo) - idx - 1)
        params_str = " ".join(f"{k}={v}" for k, v in zip(keys, vals))
        print(f"  [{idx+1:>3}/{len(kombo)}] {params_str:50s} "
              f"P={p:.3f} R={r:.3f} F1={f1:.3f} "
              f"(acak F1={f1a:.3f}) "
              f"[{elapsed:.0f}s, ETA {eta:.0f}s]")

    hasil_list.sort(key=lambda x: x["f1"], reverse=True)
    return hasil_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uji", action="store_true")
    parser.add_argument("--cepat", action="store_true")
    parser.add_argument("--lubang-saja", action="store_true",
                        help="Hanya sweep parameter lubang di atas baseline")
    args = parser.parse_args()

    if args.uji:
        gt = muat_ground_truth()
        daftar = muat_daftar_citra()
        print(f"Ground truth: {len(gt)} citra")
        print(f"Daftar audit: {len(daftar)} citra")
        assert len(gt) == 100
        assert len(daftar) == 100
        # Quick test: satu citra dengan baseline
        cfg = dict(autolabel.CFG)
        item = daftar[0]
        hasil, _ = autolabel.proses_citra(item["path"], item["id_kelas"], item["mode"], cfg)
        print(f"Uji satu citra: {item['nama']} -> {len(hasil['kotak'])} kotak")
        print("Semua uji lulus.")
        return

    print("Memuat data...")
    gt = muat_ground_truth()
    daftar = muat_daftar_citra()
    print(f"Ground truth: {len(gt)} citra, Audit: {len(daftar)} citra")

    # --- Baseline ---
    cfg_base = dict(autolabel.CFG)
    print("\nBaseline (parameter lama)...")
    skor_base = jalankan_satu_konfigurasi(daftar, gt, cfg_base)
    p, r, f1 = metrik(skor_base)
    pa, ra, f1a = metrik_acak(skor_base)
    print(f"  SEMUA   -> P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={skor_base['tp']} FP={skor_base['fp']} FN={skor_base['fn']})")
    print(f"  ACAK    -> P={pa:.3f} R={ra:.3f} F1={f1a:.3f}")

    if args.lubang_saja:
        hasil_lubang = sweep(daftar, gt, SWEEP_LUBANG, cfg_base, label="LUBANG")
        print(f"\n{'TOP 5 LUBANG':=^70}")
        for i, h in enumerate(hasil_lubang[:5]):
            param_keys = [k for k in h if k not in ("tp","fp","fn","precision","recall","f1","acak_precision","acak_recall","acak_f1")]
            params = " ".join(f"{k}={h[k]}" for k in param_keys)
            print(f"  #{i+1} {params}")
            print(f"       semua: P={h['precision']:.3f} R={h['recall']:.3f} F1={h['f1']:.3f}")
            print(f"       acak:  P={h['acak_precision']:.3f} R={h['acak_recall']:.3f} F1={h['acak_f1']:.3f}")
        cfg_final = dict(cfg_base)
        best_lubang = hasil_lubang[0]
        for k in best_lubang:
            if k in cfg_final:
                cfg_final[k] = best_lubang[k]
    else:
        # --- Tahap 1: sweep warna ---
        grid = SWEEP_CEPAT if args.cepat else SWEEP_WARNA
        hasil_warna = sweep(daftar, gt, grid, cfg_base, label="WARNA")

        print(f"\n{'TOP 5 WARNA':=^70}")
        for i, h in enumerate(hasil_warna[:5]):
            param_keys = [k for k in h if k not in ("tp","fp","fn","precision","recall","f1","acak_precision","acak_recall","acak_f1")]
            params = " ".join(f"{k}={h[k]}" for k in param_keys)
            print(f"  #{i+1} {params}")
            print(f"       semua: P={h['precision']:.3f} R={h['recall']:.3f} F1={h['f1']:.3f}")
            print(f"       acak:  P={h['acak_precision']:.3f} R={h['acak_recall']:.3f} F1={h['acak_f1']:.3f}")

        # Ambil terbaik, pakai sebagai basis tahap 2
        best_warna = hasil_warna[0]
        cfg_tahap2 = dict(cfg_base)
        for k in best_warna:
            if k in cfg_tahap2:
                cfg_tahap2[k] = best_warna[k]

        # --- Tahap 2: sweep lubang ---
        if not args.cepat:
            hasil_lubang = sweep(daftar, gt, SWEEP_LUBANG, cfg_tahap2, label="LUBANG")

            print(f"\n{'TOP 3 LUBANG (di atas best warna)':=^70}")
            for i, h in enumerate(hasil_lubang[:3]):
                param_keys = [k for k in h if k not in ("tp","fp","fn","precision","recall","f1","acak_precision","acak_recall","acak_f1")]
                params = " ".join(f"{k}={h[k]}" for k in param_keys)
                print(f"  #{i+1} {params}")
                print(f"       semua: P={h['precision']:.3f} R={h['recall']:.3f} F1={h['f1']:.3f}")
                print(f"       acak:  P={h['acak_precision']:.3f} R={h['acak_recall']:.3f} F1={h['acak_f1']:.3f}")

            best_lubang = hasil_lubang[0]
            cfg_final = dict(cfg_tahap2)
            for k in best_lubang:
                if k in cfg_final:
                    cfg_final[k] = best_lubang[k]
        else:
            cfg_final = cfg_tahap2

    # --- Ringkasan final ---
    print(f"\n{'='*70}")
    print("PARAMETER OPTIMAL")
    print(f"{'='*70}")

    skor_final = jalankan_satu_konfigurasi(daftar, gt, cfg_final)
    pf, rf, f1f = metrik(skor_final)
    pfa, rfa, f1fa = metrik_acak(skor_final)

    print(f"\nBaseline: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    print(f"Optimal:  P={pf:.3f} R={rf:.3f} F1={f1f:.3f}")
    print(f"          acak: P={pfa:.3f} R={rfa:.3f} F1={f1fa:.3f}")
    print(f"\nPerubahan parameter dari baseline:")

    changes = {}
    for key in cfg_final:
        if cfg_final[key] != cfg_base[key]:
            print(f"  {key}: {cfg_base[key]} -> {cfg_final[key]}")
            changes[key] = cfg_final[key]

    if not changes:
        print("  (tidak ada perubahan — baseline sudah optimal)")

    # Simpan semua hasil ke CSV
    HASIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_results = hasil_lubang if args.lubang_saja else hasil_warna
    fieldnames = list(all_results[0].keys())
    with open(HASIL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nHasil lengkap ditulis ke {HASIL_CSV}")


if __name__ == "__main__":
    main()
