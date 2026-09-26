"""
analisis_audit_kotak.py — Konversi COCO→YOLO & analisis perbandingan kotak audit

Membaca ekspor Roboflow (COCO) dan label autolabel asli, lalu:
  1. Konversi anotasi COCO ke format YOLO
  2. Bandingkan kotak asli vs koreksi per citra
  3. Hitung statistik: kotak dihapus, ditambah, dipertahankan, digeser
  4. Rangkum pola kesalahan autolabel.py

Jalankan dari akar proyek:
    python 01_KODE\analisis_audit_kotak.py
    python 01_KODE\analisis_audit_kotak.py --uji        # self-check
    python 01_KODE\analisis_audit_kotak.py --simpan      # tulis label YOLO hasil konversi
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent

ROBOFLOW_DIR = AKAR / "audit-kotak-terung-Folder- roboflow_upload.coco"
LABEL_ASLI_DIR = AKAR / "06_AUDIT_KOTAK" / "roboflow_upload"
PRIORITAS_CSV = AKAR / "06_AUDIT_KOTAK" / "prioritas.csv"
HASIL_DIR = AKAR / "06_AUDIT_KOTAK" / "hasil_analisis"

# Pemetaan kelas Roboflow COCO → kelas proyek YOLO
# Roboflow: 0=penyakit-daun(kosong), 1=insect_pest_disease, 2=leaf_spot_disease, 3=mosaic_virus_disease
# Proyek:   0=healthy_leaf, 1=mosaic_virus_disease, 2=leaf_spot_disease, 3=insect_pest_disease
PETA_KELAS_RF_KE_PROYEK = {
    1: 3,  # insect_pest_disease
    2: 2,  # leaf_spot_disease
    3: 1,  # mosaic_virus_disease
}

NAMA_KELAS = {
    0: "healthy_leaf",
    1: "mosaic_virus_disease",
    2: "leaf_spot_disease",
    3: "insect_pest_disease",
}


def _nama_asli(nama_roboflow: str) -> str:
    """Ekstrak nama file asli dari nama Roboflow.
    'Insect Pest Disease Image 343_jpg.rf.1WIG9o1r.jpg' → 'Insect Pest Disease Image 343.jpg'
    """
    bagian = nama_roboflow.split("_jpg.rf.")
    if len(bagian) == 2:
        return bagian[0] + ".jpg"
    return nama_roboflow


def muat_coco(direktori: Path) -> dict:
    """Muat semua anotasi COCO dari ketiga split, kembalikan {nama_asli: [(kelas_proyek, cx, cy, w, h), ...]}."""
    hasil = {}
    for split in ["train", "valid", "test"]:
        fp = direktori / split / "_annotations.coco.json"
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        img_map = {}
        for img in data["images"]:
            img_map[img["id"]] = {
                "nama": _nama_asli(img["file_name"]),
                "w": img["width"],
                "h": img["height"],
            }

        ann_per_img = defaultdict(list)
        for ann in data["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in PETA_KELAS_RF_KE_PROYEK:
                continue
            kelas = PETA_KELAS_RF_KE_PROYEK[cat_id]
            bx, by, bw, bh = ann["bbox"]  # COCO: x_min, y_min, width, height (piksel)
            img_info = img_map[ann["image_id"]]
            iw, ih = img_info["w"], img_info["h"]
            # Konversi ke YOLO: center_x, center_y, width, height (normalisasi 0-1)
            cx = (bx + bw / 2) / iw
            cy = (by + bh / 2) / ih
            nw = bw / iw
            nh = bh / ih
            ann_per_img[ann["image_id"]].append((kelas, cx, cy, nw, nh))

        for img_id, info in img_map.items():
            nama = info["nama"]
            hasil[nama] = ann_per_img.get(img_id, [])

    return hasil


def muat_yolo_asli(direktori: Path) -> dict:
    """Muat label YOLO asli dari folder roboflow_upload, kembalikan {nama.jpg: [(kelas, cx, cy, w, h), ...]}."""
    hasil = {}
    for fp in direktori.glob("*.txt"):
        if fp.name in ("classes.txt", "_darknet.labels"):
            continue
        nama_jpg = fp.stem + ".jpg"
        kotak = []
        with open(fp, encoding="utf-8") as f:
            for baris in f:
                baris = baris.strip()
                if not baris:
                    continue
                bagian = baris.split()
                kelas = int(bagian[0])
                cx, cy, w, h = float(bagian[1]), float(bagian[2]), float(bagian[3]), float(bagian[4])
                kotak.append((kelas, cx, cy, w, h))
        hasil[nama_jpg] = kotak
    return hasil


def hitung_iou(a, b):
    """IoU antara dua kotak YOLO (kelas, cx, cy, w, h). Abaikan kelas."""
    ax1 = a[1] - a[3] / 2
    ay1 = a[2] - a[4] / 2
    ax2 = a[1] + a[3] / 2
    ay2 = a[2] + a[4] / 2
    bx1 = b[1] - b[3] / 2
    by1 = b[2] - b[4] / 2
    bx2 = b[1] + b[3] / 2
    by2 = b[2] + b[4] / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    interseksi = (ix2 - ix1) * (iy2 - iy1)
    luas_a = (ax2 - ax1) * (ay2 - ay1)
    luas_b = (bx2 - bx1) * (by2 - by1)
    gabungan = luas_a + luas_b - interseksi
    if gabungan <= 0:
        return 0.0
    return interseksi / gabungan


def bandingkan_kotak(asli: list, koreksi: list, ambang_iou=0.3):
    """Bandingkan dua set kotak. Kembalikan dict statistik."""
    cocok_asli = set()
    cocok_kor = set()

    # Greedy matching: IoU tertinggi dulu
    pasangan = []
    for i, a in enumerate(asli):
        for j, k in enumerate(koreksi):
            iou = hitung_iou(a, k)
            if iou >= ambang_iou:
                pasangan.append((iou, i, j))
    pasangan.sort(reverse=True)

    dipertahankan = 0  # kotak asli yang cocok dengan koreksi (kelas sama)
    digeser = 0        # kotak asli yang cocok tapi posisi berubah (IoU < 0.7)
    ganti_kelas = 0    # kotak asli yang cocok tapi kelasnya berubah

    for iou, i, j in pasangan:
        if i in cocok_asli or j in cocok_kor:
            continue
        cocok_asli.add(i)
        cocok_kor.add(j)
        if asli[i][0] == koreksi[j][0]:
            if iou >= 0.7:
                dipertahankan += 1
            else:
                digeser += 1
        else:
            ganti_kelas += 1

    dihapus = len(asli) - len(cocok_asli)
    ditambah = len(koreksi) - len(cocok_kor)

    # Analisis kelas kotak yang dihapus
    kelas_dihapus = Counter()
    for i, a in enumerate(asli):
        if i not in cocok_asli:
            kelas_dihapus[a[0]] += 1

    # Analisis kelas kotak yang ditambah
    kelas_ditambah = Counter()
    for j, k in enumerate(koreksi):
        if j not in cocok_kor:
            kelas_ditambah[k[0]] += 1

    # Analisis ukuran kotak yang dihapus (palsu)
    luas_dihapus = []
    for i, a in enumerate(asli):
        if i not in cocok_asli:
            luas_dihapus.append(a[3] * a[4])

    return {
        "n_asli": len(asli),
        "n_koreksi": len(koreksi),
        "dipertahankan": dipertahankan,
        "digeser": digeser,
        "ganti_kelas": ganti_kelas,
        "dihapus": dihapus,
        "ditambah": ditambah,
        "kelas_dihapus": dict(kelas_dihapus),
        "kelas_ditambah": dict(kelas_ditambah),
        "luas_dihapus": luas_dihapus,
    }


def muat_prioritas():
    """Muat prioritas.csv untuk metadata tiap citra."""
    meta = {}
    with open(PRIORITAS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row["citra"]] = row
    return meta


def muat_vonis():
    """Muat vonis dari Audit Kotak.xlsx."""
    try:
        import openpyxl
    except ImportError:
        print("openpyxl belum terpasang: pip install openpyxl")
        sys.exit(1)
    wb = openpyxl.load_workbook(AKAR / "06_AUDIT_KOTAK" / "Audit Kotak.xlsx", data_only=True)
    ws = wb["Audit"]
    vonis = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        gambar = row[3]
        vonis[gambar] = {
            "kelas": row[1],
            "kelompok": row[2],
            "vonis_kotak": row[9],
            "catatan": row[10],
        }
    return vonis


def tulis_label_yolo(koreksi: dict, output_dir: Path):
    """Tulis label YOLO hasil konversi ke folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for nama, kotak_list in koreksi.items():
        fp = output_dir / (Path(nama).stem + ".txt")
        with open(fp, "w", encoding="utf-8") as f:
            for kelas, cx, cy, w, h in kotak_list:
                f.write(f"{kelas} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def jalankan_analisis():
    print("=" * 70)
    print("ANALISIS AUDIT KOTAK — Perbandingan Autolabel vs Koreksi Roboflow")
    print("=" * 70)

    # Muat data
    print("\nMemuat anotasi...")
    koreksi = muat_coco(ROBOFLOW_DIR)
    asli = muat_yolo_asli(LABEL_ASLI_DIR)
    vonis_data = muat_vonis()

    print(f"  Citra di Roboflow: {len(koreksi)}")
    print(f"  Label asli: {len(asli)}")

    # Cocokkan
    citra_dua = set(koreksi.keys()) & set(asli.keys())
    print(f"  Cocok: {len(citra_dua)} citra")

    hanya_rf = set(koreksi.keys()) - set(asli.keys())
    hanya_asli = set(asli.keys()) - set(koreksi.keys())
    if hanya_rf:
        print(f"  Hanya di Roboflow: {hanya_rf}")
    if hanya_asli:
        print(f"  Hanya di asli: {hanya_asli}")

    # Perbandingan per citra
    print("\n" + "-" * 70)
    print("PERBANDINGAN PER CITRA")
    print("-" * 70)

    total_asli = 0
    total_koreksi = 0
    total_dihapus = 0
    total_ditambah = 0
    total_dipertahankan = 0
    total_digeser = 0
    total_ganti_kelas = 0
    semua_luas_dihapus = []
    kelas_dihapus_global = Counter()
    kelas_ditambah_global = Counter()

    per_kelompok = defaultdict(lambda: {
        "n": 0, "asli": 0, "koreksi": 0,
        "dihapus": 0, "ditambah": 0, "dipertahankan": 0,
    })
    per_kelas = defaultdict(lambda: {
        "n": 0, "asli": 0, "koreksi": 0,
        "dihapus": 0, "ditambah": 0, "dipertahankan": 0,
    })
    per_vonis = defaultdict(lambda: {
        "n": 0, "asli": 0, "koreksi": 0,
        "dihapus": 0, "ditambah": 0,
    })

    detail_rows = []

    for nama in sorted(citra_dua):
        stat = bandingkan_kotak(asli[nama], koreksi[nama])
        meta = vonis_data.get(nama, {})
        kelompok = meta.get("kelompok", "?")
        kelas = meta.get("kelas", "?")
        vonis = meta.get("vonis_kotak", "?")

        total_asli += stat["n_asli"]
        total_koreksi += stat["n_koreksi"]
        total_dihapus += stat["dihapus"]
        total_ditambah += stat["ditambah"]
        total_dipertahankan += stat["dipertahankan"]
        total_digeser += stat["digeser"]
        total_ganti_kelas += stat["ganti_kelas"]
        semua_luas_dihapus.extend(stat["luas_dihapus"])

        for k, v in stat["kelas_dihapus"].items():
            kelas_dihapus_global[k] += v
        for k, v in stat["kelas_ditambah"].items():
            kelas_ditambah_global[k] += v

        for grup_key, grup_val in [(kelompok, per_kelompok), (kelas, per_kelas)]:
            d = grup_val[grup_key]
            d["n"] += 1
            d["asli"] += stat["n_asli"]
            d["koreksi"] += stat["n_koreksi"]
            d["dihapus"] += stat["dihapus"]
            d["ditambah"] += stat["ditambah"]
            d["dipertahankan"] += stat["dipertahankan"]

        d = per_vonis[vonis]
        d["n"] += 1
        d["asli"] += stat["n_asli"]
        d["koreksi"] += stat["n_koreksi"]
        d["dihapus"] += stat["dihapus"]
        d["ditambah"] += stat["ditambah"]

        detail_rows.append({
            "citra": nama,
            "kelas": kelas,
            "kelompok": kelompok,
            "vonis": vonis,
            "n_asli": stat["n_asli"],
            "n_koreksi": stat["n_koreksi"],
            "dihapus": stat["dihapus"],
            "ditambah": stat["ditambah"],
            "dipertahankan": stat["dipertahankan"],
            "digeser": stat["digeser"],
            "ganti_kelas": stat["ganti_kelas"],
        })

    # Ringkasan
    print(f"\n{'RINGKASAN KESELURUHAN':=^70}")
    print(f"  Kotak asli (autolabel):     {total_asli:>6}")
    print(f"  Kotak koreksi (Roboflow):   {total_koreksi:>6}")
    print(f"  Perubahan:                  {total_koreksi - total_asli:>+6} ({(total_koreksi/total_asli - 1)*100:+.1f}%)")
    print()
    print(f"  Dipertahankan (IoU≥0.7):    {total_dipertahankan:>6} ({total_dipertahankan/total_asli*100:.1f}% asli)")
    print(f"  Digeser (IoU 0.3-0.7):      {total_digeser:>6}")
    print(f"  Ganti kelas:                {total_ganti_kelas:>6}")
    print(f"  DIHAPUS (palsu):            {total_dihapus:>6} ({total_dihapus/total_asli*100:.1f}% asli)")
    print(f"  DITAMBAH (terlewat):        {total_ditambah:>6}")

    # Persen kotak asli yang palsu
    persen_palsu = total_dihapus / total_asli * 100 if total_asli else 0
    print(f"\n  >>> {persen_palsu:.1f}% kotak autolabel ternyata PALSU <<<")

    # Per kelompok
    print(f"\n{'PER KELOMPOK':=^70}")
    for kel in ["acak", "terburuk"]:
        d = per_kelompok[kel]
        if d["n"] == 0:
            continue
        p = d["dihapus"] / d["asli"] * 100 if d["asli"] else 0
        print(f"  {kel} ({d['n']} citra): asli={d['asli']} koreksi={d['koreksi']} "
              f"dihapus={d['dihapus']}({p:.0f}%) ditambah={d['ditambah']}")

    # Per kelas
    print(f"\n{'PER KELAS':=^70}")
    for kls in sorted(per_kelas):
        d = per_kelas[kls]
        if d["n"] == 0:
            continue
        p = d["dihapus"] / d["asli"] * 100 if d["asli"] else 0
        print(f"  {kls} ({d['n']} citra): asli={d['asli']} koreksi={d['koreksi']} "
              f"dihapus={d['dihapus']}({p:.0f}%) ditambah={d['ditambah']}")

    # Per vonis
    print(f"\n{'PER VONIS':=^70}")
    for von in ["sudah_benar", "kurang_kotak", "sebagian_palsu", "banyak_palsu", "palsu_semua"]:
        d = per_vonis.get(von)
        if not d or d["n"] == 0:
            continue
        p = d["dihapus"] / d["asli"] * 100 if d["asli"] else 0
        print(f"  {von:16s} ({d['n']:>2} citra): asli={d['asli']:>4} koreksi={d['koreksi']:>4} "
              f"dihapus={d['dihapus']:>4}({p:>4.0f}%) ditambah={d['ditambah']:>4}")

    # Kelas kotak yang dihapus
    print(f"\n{'KELAS KOTAK YANG DIHAPUS (PALSU)':=^70}")
    for k in sorted(kelas_dihapus_global):
        nama = NAMA_KELAS.get(k, f"id={k}")
        print(f"  {nama}: {kelas_dihapus_global[k]} kotak dihapus")

    # Kelas kotak yang ditambah
    print(f"\n{'KELAS KOTAK YANG DITAMBAH (TERLEWAT)':=^70}")
    for k in sorted(kelas_ditambah_global):
        nama = NAMA_KELAS.get(k, f"id={k}")
        print(f"  {nama}: {kelas_ditambah_global[k]} kotak ditambah")

    # Analisis ukuran kotak palsu
    if semua_luas_dihapus:
        semua_luas_dihapus.sort()
        n = len(semua_luas_dihapus)
        median_luas = semua_luas_dihapus[n // 2]
        p25 = semua_luas_dihapus[n // 4]
        p75 = semua_luas_dihapus[3 * n // 4]
        print(f"\n{'UKURAN KOTAK PALSU (luas normalisasi)':=^70}")
        print(f"  Median: {median_luas:.6f}  (P25: {p25:.6f}, P75: {p75:.6f})")
        # Konversi ke perkiraan piksel (asumsi 3000x4000)
        piksel_ref = 3000 * 4000
        print(f"  ~{median_luas * piksel_ref:.0f} piksel² pada citra 3000x4000")
        kecil = sum(1 for l in semua_luas_dihapus if l < 0.001)
        sedang = sum(1 for l in semua_luas_dihapus if 0.001 <= l < 0.005)
        besar = sum(1 for l in semua_luas_dihapus if l >= 0.005)
        print(f"  Kecil (<0.1% citra): {kecil} ({kecil/n*100:.0f}%)")
        print(f"  Sedang (0.1-0.5%):   {sedang} ({sedang/n*100:.0f}%)")
        print(f"  Besar (>0.5%):       {besar} ({besar/n*100:.0f}%)")

    # Kelompok acak saja — angka yang boleh dikutip
    print(f"\n{'=' * 70}")
    print("ANGKA YANG BOLEH DIKUTIP (kelompok acak saja)")
    print("=" * 70)
    d = per_kelompok["acak"]
    if d["n"] > 0:
        p_palsu = d["dihapus"] / d["asli"] * 100
        p_terlewat = d["ditambah"] / d["koreksi"] * 100 if d["koreksi"] else 0
        print(f"  Dari {d['asli']} kotak autolabel pada {d['n']} citra acak:")
        print(f"    {d['dihapus']} kotak ({p_palsu:.1f}%) ternyata PALSU")
        print(f"    {d['ditambah']} kotak TERLEWAT (ditambah saat koreksi)")
        print(f"    {d['dipertahankan']} kotak dipertahankan apa adanya")

    return detail_rows, koreksi


def simpan_csv(detail_rows):
    HASIL_DIR.mkdir(parents=True, exist_ok=True)
    fp = HASIL_DIR / "perbandingan_detail.csv"
    with open(fp, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["citra", "kelas", "kelompok", "vonis",
                      "n_asli", "n_koreksi", "dihapus", "ditambah",
                      "dipertahankan", "digeser", "ganti_kelas"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"\nDetail ditulis ke {fp}")


def uji():
    """Self-check."""
    # Uji konversi nama
    assert _nama_asli("Insect Pest Disease Image 343_jpg.rf.1WIG9o1r.jpg") == "Insect Pest Disease Image 343.jpg"
    assert _nama_asli("test.jpg") == "test.jpg"

    # Uji IoU
    a = (0, 0.5, 0.5, 0.2, 0.2)  # kotak di tengah
    b = (0, 0.5, 0.5, 0.2, 0.2)  # identik
    assert abs(hitung_iou(a, b) - 1.0) < 1e-6

    c = (0, 0.9, 0.9, 0.1, 0.1)  # jauh
    assert hitung_iou(a, c) == 0.0

    d = (0, 0.55, 0.5, 0.2, 0.2)  # sebagian tumpang tindih
    iou = hitung_iou(a, d)
    assert 0 < iou < 1

    # Uji bandingkan_kotak
    asli = [(2, 0.5, 0.5, 0.1, 0.1), (2, 0.2, 0.2, 0.05, 0.05)]
    koreksi = [(2, 0.5, 0.5, 0.1, 0.1), (3, 0.8, 0.8, 0.1, 0.1)]  # satu sama, satu baru
    stat = bandingkan_kotak(asli, koreksi)
    assert stat["dipertahankan"] == 1
    assert stat["dihapus"] == 1
    assert stat["ditambah"] == 1

    # Uji peta kelas
    assert PETA_KELAS_RF_KE_PROYEK[1] == 3  # insect
    assert PETA_KELAS_RF_KE_PROYEK[2] == 2  # leaf_spot
    assert PETA_KELAS_RF_KE_PROYEK[3] == 1  # mosaic

    print("Semua uji lulus.")


def main():
    parser = argparse.ArgumentParser(description="Analisis audit kotak")
    parser.add_argument("--uji", action="store_true", help="Jalankan self-check")
    parser.add_argument("--simpan", action="store_true", help="Simpan label YOLO hasil konversi")
    args = parser.parse_args()

    if args.uji:
        uji()
        return

    detail_rows, koreksi = jalankan_analisis()
    simpan_csv(detail_rows)

    if args.simpan:
        output_dir = HASIL_DIR / "label_koreksi_yolo"
        tulis_label_yolo(koreksi, output_dir)
        print(f"Label YOLO koreksi ditulis ke {output_dir} ({len(koreksi)} file)")


if __name__ == "__main__":
    main()
