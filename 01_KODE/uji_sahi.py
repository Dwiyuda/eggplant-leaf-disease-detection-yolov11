#!/usr/bin/env python3
"""
Tahap 1 SAHI: uji apakah inferensi berbasis ubin (sliced) menaikkan deteksi
lesi kecil, TANPA melatih ulang.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

LATAR BELAKANG. Lesi leaf_spot/insect median ~16 px saat citra dikecilkan ke
imgsz model. SAHI memotong citra jadi ubin lalu tiap ubin di-inferensi pada
resolusi model -> lesi jadi relatif besar, tanpa kehilangan piksel. Ini menyerang
akar objek-kecil, beda dari menaikkan resolusi mentah (yang sudah terbukti buntu:
1280 < 1024 pada label bersih).

Dibandingkan pada model & citra yang SAMA: inferensi standar (citra utuh) vs
SAHI (ubin). Dua bacaan, keduanya relatif tahan penggaris kotak rusak:
  - akurasi tingkat-citra (aturan jumlah_conf) -- "penyakitnya ketebak benar?"
  - recall deteksi per kelas -- "berapa lesi yang ketemu?"

CATATAN penggaris: label test masih kotak autolabel lama (~46% palsu, 56%
terlewat). SAHI yang menemukan lesi asli yang terlewat autolabel akan tampak
sebagai "false positive" pada mAP -- karena itu di sini TIDAK memakai mAP, hanya
akurasi tingkat-citra dan jumlah deteksi.

Pakai:
    python 01_KODE\\uji_sahi.py --uji                 # self-check, tanpa model
    python 01_KODE\\uji_sahi.py                       # test set penuh (208)
    python 01_KODE\\uji_sahi.py --sampel 40           # sampel cepat
    python 01_KODE\\uji_sahi.py --split val
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
DIR_DATASET = AKAR / "02_DATASET" / "dataset_80_10_10"
DIR_KAGGLE = AKAR / "04_KAGGLE" / "FOLDER_HASIL_KAGGLE"

NAMA = {0: "healthy_leaf", 1: "mosaic_virus_disease",
        2: "leaf_spot_disease", 3: "insect_pest_disease"}
LESI = {2, 3}

# Model terbaik yang ada (kotak-lama, label bersih) -- juara sejauh ini.
MODEL_DEFAULT = DIR_KAGGLE / "hasil_yolo11_ablasi_resolusi_dataset_baru" / "img1024" / "best_yolo11s_img1024.pt"
CONF = 0.05
IMGSZ = 1024
SLICE = 1024        # ukuran ubin; ubin di-inferensi pada resolusi model
OVERLAP = 0.2


def kelas_sebenarnya(lbl: Path):
    """Kelas mayoritas dari file label = penyakit dominan citra."""
    ids = [int(b.split()[0]) for b in lbl.read_text().splitlines() if b.strip()]
    return Counter(ids).most_common(1)[0][0] if ids else None


def gabung_jumlah_conf(cats, scores):
    """Aturan jumlah_conf: kelas dengan total kepercayaan terbesar."""
    if not cats:
        return None
    tot = defaultdict(float)
    for c, s in zip(cats, scores):
        tot[c] += s
    return max(tot, key=tot.get)


def ringkas(nama, benar, total, tanpa_deteksi, n_deteksi, rec_num, rec_den):
    akur = benar / total if total else 0.0
    print(f"\n--- {nama} ---")
    print(f"  akurasi citra : {akur:.4f}  ({benar}/{total})")
    print(f"  tanpa deteksi : {tanpa_deteksi}")
    print(f"  total deteksi : {n_deteksi}")
    print("  recall deteksi per kelas (vs kotak label lama):")
    for c in sorted(NAMA):
        d = rec_den.get(c, 0)
        r = rec_num.get(c, 0) / d if d else 0.0
        print(f"    {NAMA[c]:22s} {r:.3f}  ({rec_num.get(c,0)}/{d})")
    return akur


def uji():
    # self-check logika, tanpa model/sahi
    assert kelas_sebenarnya_dari_ids([2, 2, 3]) == 2
    assert kelas_sebenarnya_dari_ids([3, 3, 3, 2]) == 3
    assert gabung_jumlah_conf([], []) is None
    assert gabung_jumlah_conf([2, 3, 3], [0.9, 0.1, 0.1]) == 2   # conf, bukan jumlah
    assert gabung_jumlah_conf([2, 2, 3], [0.1, 0.1, 0.9]) == 3
    assert MODEL_DEFAULT.exists(), f"model tidak ada: {MODEL_DEFAULT}"
    print("SEMUA UJI LOLOS")
    print("model:", MODEL_DEFAULT.name)


def kelas_sebenarnya_dari_ids(ids):
    return Counter(ids).most_common(1)[0][0] if ids else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uji", action="store_true")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--sampel", type=int, default=0, help="hanya N citra (0 = semua)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--slice", type=int, default=SLICE)
    args = ap.parse_args()

    if args.uji:
        uji()
        return

    model_path = Path(args.model) if args.model else MODEL_DEFAULT
    if not model_path.exists():
        sys.exit(f"model tidak ketemu: {model_path}")

    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction

    try:
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(model_path),
            confidence_threshold=CONF, image_size=IMGSZ, device="cpu")
    except Exception:
        model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=str(model_path),
            confidence_threshold=CONF, image_size=IMGSZ, device="cpu")

    split = args.split
    imgs = sorted((DIR_DATASET / "images" / split).glob("*.jpg"))
    if args.sampel:
        imgs = imgs[:args.sampel]
    print(f"model : {model_path.name}")
    print(f"split : {split} ({len(imgs)} citra) | slice {args.slice} overlap {OVERLAP} | conf {CONF}")

    # akumulator untuk dua mode
    akum = {m: dict(benar=0, total=0, tanpa=0, ndet=0,
                    rec_num=defaultdict(int), rec_den=defaultdict(int))
            for m in ("standar", "sahi")}

    for i, img in enumerate(imgs, 1):
        lbl = DIR_DATASET / "labels" / split / f"{img.stem}.txt"
        y = kelas_sebenarnya(lbl) if lbl.exists() else None
        if y is None:
            continue
        n_lbl_kelas = Counter(int(b.split()[0])
                              for b in lbl.read_text().splitlines() if b.strip())

        r_std = get_prediction(str(img), model)
        r_slc = get_sliced_prediction(
            str(img), model, slice_height=args.slice, slice_width=args.slice,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, verbose=0)

        for mode, res in (("standar", r_std), ("sahi", r_slc)):
            cats = [o.category.id for o in res.object_prediction_list]
            scs = [o.score.value for o in res.object_prediction_list]
            a = akum[mode]
            a["total"] += 1
            a["ndet"] += len(cats)
            pred = gabung_jumlah_conf(cats, scs)
            if pred is None:
                a["tanpa"] += 1
            elif pred == y:
                a["benar"] += 1
            # recall deteksi kasar: berapa kelas berlabel yang juga terdeteksi
            for c, jml in n_lbl_kelas.items():
                a["rec_den"][c] += jml
                a["rec_num"][c] += min(jml, cats.count(c))

        if i % 25 == 0:
            print(f"  {i}/{len(imgs)}")

    print("\n" + "=" * 70)
    print(f"  PERBANDINGAN STANDAR vs SAHI  --  {split.upper()}")
    print("=" * 70)
    a_std = ringkas("STANDAR (citra utuh)", akum["standar"]["benar"], akum["standar"]["total"],
                    akum["standar"]["tanpa"], akum["standar"]["ndet"],
                    akum["standar"]["rec_num"], akum["standar"]["rec_den"])
    a_slc = ringkas("SAHI (ubin)", akum["sahi"]["benar"], akum["sahi"]["total"],
                    akum["sahi"]["tanpa"], akum["sahi"]["ndet"],
                    akum["sahi"]["rec_num"], akum["sahi"]["rec_den"])
    print(f"\n>>> selisih akurasi citra (SAHI - standar): {a_slc - a_std:+.4f}")
    print(f">>> total deteksi: standar {akum['standar']['ndet']} -> sahi {akum['sahi']['ndet']}")


if __name__ == "__main__":
    main()
