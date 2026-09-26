#!/usr/bin/env python3
"""
Menerapkan 100 label koreksi kotak (ground-truth Roboflow) ke dataset_80_10_10.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

LATAR BELAKANG. Fase A memperbaiki KELAS tiap citra; kotaknya sendiri tetap
buatan autolabel.py (aturan warna) yang audit membuktikan ~46% palsu dan ~56%
kotak asli terlewat. 100 citra sampel audit sudah dianotasi ulang manual di
Roboflow dan tersimpan sebagai label YOLO di label_koreksi_yolo/. Skrip ini
menimpa label lama 100 citra itu dengan versi koreksinya -- PILOT untuk mengukur
apakah kotak bersih menggerakkan kelas lesi, bukan anotasi ulang penuh.

ATURAN PENTING: citra yang berada di subset TEST DILEWATI. Test set dipertahankan
apa adanya supaya perbandingan sebelum-sesudah tetap sah (penggaris tidak
berubah). Konsekuensinya kenaikan kotak lesi mungkin tidak terlihat penuh di
test mAP karena kunci jawaban test masih autolabel lama -- itu memang batas yang
sudah disepakati untuk pilot ini.

Tidak perlu bangun_split.py sesudahnya: label ditimpa di tempat, citra tidak
pindah split, keanggotaan split tidak berubah.

Pakai:
    python 01_KODE\\terapkan_koreksi_kotak.py --uji        # self-check
    python 01_KODE\\terapkan_koreksi_kotak.py              # KERING: laporan rencana
    python 01_KODE\\terapkan_koreksi_kotak.py --terapkan   # sungguhan (mencadangkan dulu)
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
DIR_KOREKSI = AKAR / "06_AUDIT_KOTAK" / "hasil_analisis" / "label_koreksi_yolo"
DIR_DATASET = AKAR / "02_DATASET" / "dataset_80_10_10" / "labels"
DIR_CADANGAN = AKAR / "99_ARSIP" / "cadangan_koreksi_kotak"
SUBSET = ("train", "val", "test")


def n_kotak(path: Path) -> int:
    """Jumlah baris tak-kosong = jumlah bounding box dalam satu file label."""
    return sum(1 for b in path.read_text().splitlines() if b.strip())


def peta_split() -> dict:
    """stem citra -> subset tempat label-nya berada sekarang."""
    m = {}
    for sub in SUBSET:
        for p in (DIR_DATASET / sub).glob("*.txt"):
            m[p.stem] = sub
    return m


def rencana():
    """Kembalikan (terapkan, lewati_test, hilang) tanpa menyentuh apa pun.

    terapkan   : list (stem, subset, n_lama, n_baru) untuk train/val
    lewati_test: list (stem, n_lama, n_baru) untuk citra di test
    hilang     : list stem koreksi yang tidak ada di dataset
    """
    m = peta_split()
    terapkan, lewati_test, hilang = [], [], []
    for src in sorted(DIR_KOREKSI.glob("*.txt")):
        stem = src.stem
        sub = m.get(stem)
        n_baru = n_kotak(src)
        if sub is None:
            hilang.append(stem)
            continue
        n_lama = n_kotak(DIR_DATASET / sub / f"{stem}.txt")
        if sub == "test":
            lewati_test.append((stem, n_lama, n_baru))
        else:
            terapkan.append((stem, sub, n_lama, n_baru))
    return terapkan, lewati_test, hilang


def cetak_rencana(terapkan, lewati_test, hilang):
    n_tr = sum(1 for _, s, _, _ in terapkan if s == "train")
    n_va = sum(1 for _, s, _, _ in terapkan if s == "val")
    lama = sum(a for _, _, a, _ in terapkan)
    baru = sum(b for _, _, _, b in terapkan)
    print(f"  akan ditimpa : {len(terapkan)} label  (train {n_tr}, val {n_va})")
    print(f"  dilewati     : {len(lewati_test)} label di TEST (dipertahankan)")
    print(f"  tidak ketemu : {len(hilang)}")
    print(f"  kotak train+val : {lama} -> {baru}  ({baru - lama:+d})")
    if hilang:
        print("  PERINGATAN, koreksi tanpa pasangan di dataset:")
        for h in hilang:
            print(f"    - {h}")


def uji():
    assert DIR_KOREKSI.is_dir(), f"folder koreksi tidak ada: {DIR_KOREKSI}"
    assert DIR_DATASET.is_dir(), f"folder dataset tidak ada: {DIR_DATASET}"
    koreksi = list(DIR_KOREKSI.glob("*.txt"))
    assert len(koreksi) == 100, f"harusnya 100 label koreksi, ada {len(koreksi)}"

    terapkan, lewati_test, hilang = rencana()
    assert not hilang, f"ada koreksi tanpa pasangan: {hilang}"
    assert len(terapkan) + len(lewati_test) == 100

    # tiap file koreksi harus valid: class 0-3, 5 kolom, tak kosong
    for src in koreksi:
        baris = [b for b in src.read_text().splitlines() if b.strip()]
        assert baris, f"label koreksi kosong: {src.name}"
        for b in baris:
            k = b.split()
            assert len(k) == 5, f"format salah di {src.name}: {b!r}"
            assert k[0] in {"0", "1", "2", "3"}, f"class id asing di {src.name}: {k[0]}"

    print("SELF-CHECK LOLOS")
    cetak_rencana(terapkan, lewati_test, hilang)


def terapkan_sungguhan(terapkan):
    stempel = datetime.now().strftime("%Y%m%d_%H%M%S")
    tujuan_cadangan = DIR_CADANGAN / stempel
    for _, sub, _, _ in terapkan:
        (tujuan_cadangan / sub).mkdir(parents=True, exist_ok=True)

    for stem, sub, _, _ in terapkan:
        asli = DIR_DATASET / sub / f"{stem}.txt"
        shutil.copy2(asli, tujuan_cadangan / sub / f"{stem}.txt")   # cadangkan dulu
        shutil.copy2(DIR_KOREKSI / f"{stem}.txt", asli)             # baru timpa

    print(f"\nSELESAI. {len(terapkan)} label ditimpa.")
    print(f"Cadangan label lama: {tujuan_cadangan}")

    # verifikasi: nol label kosong di seluruh train+val
    kosong = [p.name for sub in ("train", "val")
              for p in (DIR_DATASET / sub).glob("*.txt") if n_kotak(p) == 0]
    if kosong:
        print(f"PERINGATAN: {len(kosong)} label kosong terdeteksi -> {kosong[:5]}")
    else:
        print("Verifikasi: nol citra tanpa label di train/val. Aman.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uji", action="store_true", help="self-check, tidak menyentuh data")
    ap.add_argument("--terapkan", action="store_true", help="timpa sungguhan (cadangan dulu)")
    args = ap.parse_args()

    if args.uji:
        uji()
        return

    terapkan, lewati_test, hilang = rencana()
    if hilang:
        print("BERHENTI: ada koreksi tanpa pasangan di dataset. Jalankan --uji.")
        cetak_rencana(terapkan, lewati_test, hilang)
        sys.exit(1)

    if args.terapkan:
        print("=== MENERAPKAN KOREKSI KOTAK (dataset_80_10_10) ===")
        cetak_rencana(terapkan, lewati_test, hilang)
        terapkan_sungguhan(terapkan)
    else:
        print("=== MODE KERING (tidak ada yang diubah) ===")
        cetak_rencana(terapkan, lewati_test, hilang)
        print("\nJalankan lagi dengan --terapkan untuk menulis.")


if __name__ == "__main__":
    main()
