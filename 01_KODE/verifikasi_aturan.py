#!/usr/bin/env python3
"""
Verifikasi aturan penggabungan untuk angka akurasi tingkat-citra model final.

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

MASALAH. Catatan 15 Agu menyebut aturan terpilih 'jumlah_kotak' (val 0,8038,
test 0,7981). Sistem web dan sweep SAHI 13 Sep memakai 'jumlah_conf' (juga test
0,7981). ringkasan.json asli 15 Agu sudah tertimpa evaluasi model lain. Sebelum
angka itu dikutip di naskah, aturan yang benar harus dipastikan.

KOMPLIKASI. Pada 11 Sep, 14 label val ditimpa label koreksi Roboflow; 4 di
antaranya berganti kelas mayoritas. Val sekarang != val yang dipakai memilih
aturan pada 15 Agu. Val asli dipulihkan dari cadangan di 99_ARSIP.

Seluruh logika aturan, pemilihan, dan metrik DIIMPOR dari evaluasi_sistem.py,
bukan ditulis ulang, supaya dijamin identik.

Pakai:
    python 01_KODE\\verifikasi_aturan.py --uji
    python 01_KODE\\verifikasi_aturan.py              # pakai cache bila ada
    python 01_KODE\\verifikasi_aturan.py --ulang      # paksa inferensi ulang
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "01_KODE"))
sys.path.insert(0, str(AKAR / "07_SISTEM_WEB"))

import evaluasi_sistem as ev   # ATURAN, AMBANG, terapkan, hitung, matriks, kelas_asli

DS = AKAR / "02_DATASET" / "dataset_80_10_10"
BOBOT = (AKAR / "04_KAGGLE" / "FOLDER_HASIL_KAGGLE"
         / "hasil_yolo11_ablasi_resolusi_dataset_baru" / "img1024" / "best_yolo11s_img1024.pt")
CADANGAN_VAL = AKAR / "99_ARSIP" / "cadangan_koreksi_kotak" / "20260911_114129" / "val"
KELUAR = AKAR / "05_HASIL" / "verifikasi_aturan"
CACHE = KELUAR / "deteksi_cache.json"
IMGSZ = 1024


def pilih(benar, deteksi):
    """Replika persis blok pemilihan evaluasi_sistem.jalankan():
    sapu ATURAN x AMBANG, urutkan stabil menurut -makro_f1, ambil teratas."""
    rekap = []
    for at in ev.ATURAN:
        for am in ev.AMBANG:
            h = ev.hitung(benar, ev.terapkan(deteksi, am, at))
            rekap.append({"aturan": at, "conf": am, "akurasi": h["akurasi"],
                          "makro_f1": h["makro_f1"], "tanpa_deteksi": h["tanpa_deteksi"]})
    rekap.sort(key=lambda r: -r["makro_f1"])
    return rekap


def label_val_asli():
    """Folder label val sebelum 11 Sep: val sekarang + 14 berkas dari cadangan."""
    d = KELUAR / "label_val_asli"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(DS / "labels" / "val", d)
    for f in CADANGAN_VAL.glob("*.txt"):
        shutil.copy2(f, d / f.name)
    return d


def inferensi():
    """Identik dengan evaluasi_sistem.kumpulkan(): conf=0.01, imgsz 1024, tanpa TTA."""
    if CACHE.exists():
        print(f"memakai cache: {CACHE.name}")
        return json.loads(CACHE.read_text(encoding="utf-8"))
    from ultralytics import YOLO
    model = YOLO(str(BOBOT))
    out = {}
    for sp in ("val", "test"):
        berkas = sorted(p for p in (DS / "images" / sp).iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        out[sp] = {}
        for i, p in enumerate(berkas, 1):
            r = model.predict(str(p), conf=0.01, imgsz=IMGSZ, augment=False, verbose=False)[0]
            out[sp][p.stem] = [[int(c), float(cf)] for c, cf in
                               zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            if i % 50 == 0:
                print(f"  {sp} {i}/{len(berkas)}")
    KELUAR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out), encoding="utf-8")
    return out


def susun(det_sp, dir_lbl):
    stem = sorted(det_sp)
    benar, det = [], []
    for s in stem:
        k = ev.kelas_asli(dir_lbl, s)
        if k < 0:
            continue
        benar.append(k)
        det.append([tuple(x) for x in det_sp[s]])
    return stem, benar, det


def cetak_rekap(judul, rekap, n=8):
    print(f"\n--- {judul} ---")
    print(f"  {'aturan':15s} {'conf':>5s} {'akurasi':>8s} {'makro-F1':>9s}")
    for r in rekap[:n]:
        print(f"  {r['aturan']:15s} {r['conf']:5.2f} {r['akurasi']:8.4f} {r['makro_f1']:9.4f}")
    t = rekap[0]
    print(f"  TERPILIH: {t['aturan']} @ {t['conf']:.2f}")


def jalur_web(stem_test):
    """Seluruh citra test lewat jalur sistem web: bytes -> cv2.imdecode -> deteksi.prediksi."""
    import cv2
    import numpy as np
    import deteksi
    ke_id = {v: k for k, v in deteksi.NAMA.items()}
    hasil = {}
    for i, s in enumerate(stem_test, 1):
        data = (DS / "images" / "test" / f"{s}.jpg").read_bytes()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        h = deteksi.prediksi(arr)
        hasil[s] = ke_id.get(h["kelas"], -1)
        if i % 50 == 0:
            print(f"  web {i}/{len(stem_test)}")
    return hasil


def main(ulang):
    if ulang and CACHE.exists():
        CACHE.unlink()
    KELUAR.mkdir(parents=True, exist_ok=True)
    det = inferensi()

    # ---------- 1. pemilihan aturan pada dua versi val ----------
    dir_asli = label_val_asli()
    _, bv_asli, dv = susun(det["val"], dir_asli)
    _, bv_kini, _ = susun(det["val"], DS / "labels" / "val")
    rk_asli, rk_kini = pilih(bv_asli, dv), pilih(bv_kini, dv)

    print("\n" + "=" * 70 + "\n  1. PEMILIHAN ATURAN DARI VAL (kriteria: makro-F1)\n" + "=" * 70)
    cetak_rekap("val ASLI (15 Agu, sebelum koreksi 11 Sep)", rk_asli)
    cetak_rekap("val SEKARANG (setelah 14 label ditimpa)", rk_kini)

    def cari(rk, at, am=0.05):
        return next(r for r in rk if r["aturan"] == at and abs(r["conf"] - am) < 1e-9)
    for at in ("jumlah_kotak", "jumlah_conf"):
        a, k = cari(rk_asli, at), cari(rk_kini, at)
        print(f"  {at:13s} @0.05  val asli: akur {a['akurasi']:.4f} F1 {a['makro_f1']:.4f}"
              f"  |  val kini: akur {k['akurasi']:.4f} F1 {k['makro_f1']:.4f}")

    # ---------- 2. test ----------
    stem_t, bt, dt = susun(det["test"], DS / "labels" / "test")
    print("\n" + "=" * 70 + "\n  2. TEST (208 citra) -- semua aturan pada conf 0.05\n" + "=" * 70)
    tes = {}
    for at in ev.ATURAN:
        h = ev.hitung(bt, ev.terapkan(dt, 0.05, at))
        tes[at] = h
        print(f"  {at:15s} akurasi {h['akurasi']:.4f} ({round(h['akurasi']*h['n'])}/{h['n']})"
              f"  makro-F1 {h['makro_f1']:.4f}")

    tk = ev.terapkan(dt, 0.05, "jumlah_kotak")
    tc = ev.terapkan(dt, 0.05, "jumlah_conf")
    beda = [(stem_t[i], bt[i], tk[i], tc[i]) for i in range(len(bt)) if tk[i] != tc[i]]
    print(f"\n  citra test dengan tebakan BERBEDA (jumlah_kotak vs jumlah_conf): {len(beda)}")
    N = ev.NAMA
    for s, b, k, c in beda:
        print(f"    {s[:34]:34s} benar={N[b]:22s} kotak={N.get(k,'-'):22s} conf={N.get(c,'-')}")
    for nama, teb in (("jumlah_kotak", tk), ("jumlah_conf", tc)):
        print(f"\n  confusion matrix {nama}:")
        ev.cetak_matriks(ev.matriks(bt, teb))

    # ---------- 3. jalur sistem web ----------
    print("\n" + "=" * 70 + "\n  3. JALUR SISTEM WEB pada seluruh 208 citra test\n" + "=" * 70)
    web = jalur_web(stem_t)
    tw = [web[s] for s in stem_t]
    sama = sum(1 for a, b in zip(tw, tc) if a == b)
    hw = ev.hitung(bt, tw)
    print(f"  sama dengan evaluasi jumlah_conf@0.05 : {sama}/{len(tc)}")
    print(f"  akurasi jalur web                     : {hw['akurasi']:.4f} "
          f"({round(hw['akurasi']*hw['n'])}/{hw['n']})")
    selisih = [(s, N[bt[i]], N.get(tc[i], '-'), N.get(tw[i], '-'))
               for i, s in enumerate(stem_t) if tw[i] != tc[i]]
    for x in selisih:
        print(f"    BEDA: {x}")

    (KELUAR / "laporan.json").write_text(json.dumps({
        "val_asli_top10": rk_asli[:10], "val_kini_top10": rk_kini[:10],
        "test_per_aturan_conf005": {k: {"akurasi": v["akurasi"], "makro_f1": v["makro_f1"],
                                        "n": v["n"], "per_kelas": v["per_kelas"]}
                                    for k, v in tes.items()},
        "test_beda_kotak_vs_conf": beda,
        "web": {"sama_dengan_jumlah_conf": sama, "n": len(tc),
                "akurasi": hw["akurasi"], "selisih": selisih},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nlaporan -> {KELUAR / 'laporan.json'}")


def uji():
    assert BOBOT.exists(), BOBOT
    assert CADANGAN_VAL.is_dir() and len(list(CADANGAN_VAL.glob("*.txt"))) == 14
    assert list(ev.ATURAN)[:3] == ["conf_maks", "jumlah_kotak", "jumlah_conf"]
    # pemecah seri: urutan stabil -> aturan yang lebih awal menang
    b = [0, 1]
    d = [[(0, 0.9)], [(1, 0.8)]]
    rk = pilih(b, d)
    # semua aturan seri (tebakan sempurna) -> yang pertama di ATURAN menang
    assert rk[0]["aturan"] == "conf_maks", rk[0]
    assert len({r["makro_f1"] for r in rk if r["conf"] <= 0.75}) == 1
    print("SEMUA UJI LOLOS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uji", action="store_true")
    p.add_argument("--ulang", action="store_true")
    a = p.parse_args()
    uji() if a.uji else main(a.ulang)
