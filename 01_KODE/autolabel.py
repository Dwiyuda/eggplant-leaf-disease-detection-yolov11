#!/usr/bin/env python3
"""
Auto-anotasi bounding box untuk dataset penyakit daun terung (YOLO format).

Deteksi Penyakit Daun Terung Berbasis Object Detection dengan Model YOLOv11 -

Konvensi anotasi (mengikuti contoh anotasi manual di folder contoh_boundingbox):
    healthy_leaf         -> 1 box, seluruh helai daun
    mosaic_virus_disease -> 1 box, seluruh helai daun
    leaf_spot_disease    -> banyak box, satu per bercak/area gejala
    insect_pest_disease  -> banyak box, satu per lubang gigitan

Pembeda leaf_spot vs insect (dari pengamatan patologi ahli domain):
    - lubang TANPA warna di tepinya          -> gigitan serangga (insect)
    - lubang DENGAN halo kuning/hitam/coklat -> bercak daun (leaf_spot)
    - bercak berwarna tanpa lubang           -> bercak daun (leaf_spot)

Kelas label TIDAK ditentukan oleh hasil deteksi, melainkan oleh folder asal
citra (satu kondisi dominan per citra). Hasil analisis warna dipakai sebagai
verifikasi silang: bila isi citra bertentangan dengan folder asalnya, citra
ditandai untuk ditinjau manual -- tidak dianotasi ganda.

Pakai:
    python autolabel.py --sampel 20        # uji coba 20 gambar/kelas + pratinjau
    python autolabel.py                    # seluruh dataset
    python autolabel.py --uji              # self-check logika deteksi
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ======================================================================
# KONFIGURASI
# ======================================================================
AKAR = Path(__file__).resolve().parent.parent   # skrip ada di 01_KODE/
DIR_DATASET = AKAR / "02_DATASET"
DIR_SUMBER  = AKAR / "03_CITRA_SUMBER"
DIR_KAGGLE  = AKAR / "04_KAGGLE" / "FOLDER_HASIL_KAGGLE"
DIR_HASIL   = AKAR / "05_HASIL"
KELUARAN = DIR_HASIL / "autolabel_hasil"

# nama folder -> (id kelas YOLO, mode anotasi)
KELAS = {
    "healthy_leaf":         (0, "daun_utuh"),
    "mosaic_virus_disease": (1, "daun_utuh"),
    "leaf_spot_disease":    (2, "lesi"),
    "insect_pest_disease":  (3, "lesi"),
}

# --- Knob kalibrasi. Semua ambang bersifat relatif terhadap statistik daun
# --- pada citra itu sendiri, sehingga tahan terhadap color cast dan bayangan
# --- yang berbeda-beda antar foto. Setel setelah melihat hasil --sampel.
CFG = {
    "sisi_maks":        1024,   # citra dikecilkan dulu; label YOLO ternormalisasi

    # segmentasi daun
    "v_gelap":            70,   # piksel segelap ini dianggap daun walau saturasi rendah
    "min_frac_daun":    0.02,   # kontur < 2% luas kontur terbesar dibuang
    "frac_daun_kedua":  0.15,   # kontur >= 15% terbesar dihitung sebagai daun terpisah

    # Penyaring kilau spekular. Pantulan cahaya di permukaan daun juga terang
    # dan lolos sebagai "lubang". Ambang diukur empiris atas 233 kandidat di
    # insect (mayoritas lubang asli) vs 32 di healthy (mayoritas kilau):
    #     saturasi p25   : lubang 8   vs kilau 34   -> terpisah 4x
    #     selisih terang : lubang 41  vs kilau 55   -> tumpang tindih
    # Jadi saturasi yang dipakai sebagai penentu; kecerahan hanya penjaga
    # kasar agar bayangan gelap tidak ikut terhitung lubang.
    "maks_sat_lubang":     20,   # isi lubang harus netral seperti latar
    "toleransi_v_lubang": 110,

    # Deteksi lesi warna. Ambang diuji empiris pada 20 citra/kelas (lihat
    # docstring deteksi_lesi_warna): kanal a* memisahkan kelas berlesi dari
    # healthy/mosaic 2-3x, sedangkan b* dan L* tidak memisahkan sama sekali.
    "k_merah":           2.0,   # ambang relatif: a* di atas median + k x MAD
    "delta_merah":      12.0,   # lantai absolut satuan Lab -- WAJIB, lihat catatan
    "erosi_tepi_frac": 0.010,   # tepi daun dikikis sedikit agar rim antialias tak terbaca lesi

    # Mottling kuning -- CADANGAN, hanya untuk leaf_spot yang gejala merahnya
    # tidak muncul. Tidak pernah dipakai di healthy/insect: sapuan empiris
    # membuktikan b* tak bisa memisahkan healthy dari leaf_spot, jadi sinyal
    # ini hanya sah dipakai bila label folder sudah menyatakan citra itu sakit.
    "k_kuning":           1.5,
    "delta_kuning":       8.0,
    "ambang_cadangan":      3,   # < 3 kotak utama -> coba mottling
    "gabung_mottling_frac": 0.05,   # mottling menyebar: dilebur jadi area, bukan bintik
    "min_luas_mottling_frac": 0.004,
    "maks_kotak_mottling":  10,

    # penggabungan & penyaringan kotak
    "gabung_frac":     0.010,   # lesi berjarak < 1% lebar daun dilebur jadi satu kotak
    "min_luas_frac":  0.00060,  # bercak warna: < 0,06% luas daun dibuang (bintik noise)
    # Lubang dinilai lebih longgar: bekas gigitan memang kecil, dan sinyalnya
    # geometris (latar tembus di dalam daun) bukan statistik warna, jadi jauh
    # lebih kecil kemungkinan derainya.
    "min_luas_lubang_frac": 0.00008,
    "maks_kotak":         30,   # ambil N kotak terbesar bila berlebih

    # analisis cincin di sekeliling lubang
    "cincin_frac":     0.006,   # tebal cincin relatif lebar daun
    "cincin_ambang":    0.25,   # >25% cincin berwarna menyimpang -> lubang leaf_spot
    "hijau_h_min":        30,   # rentang hue "hijau sehat" (OpenCV H: 0-179)
    "hijau_h_maks":       90,

    # Cadangan gigitan tepi (convexity defects). Gigitan serangga di tepi daun
    # tidak membentuk rongga tertutup, jadi lolos dari deteksi lubang; ia hanya
    # menakik siluet. Geometri tak bisa memisahkannya dari lobus alami daun
    # terung secara global (lihat sapuan empiris), jadi detektor ini HANYA
    # dipanggil untuk citra insect yang kotaknya < ambang -- menutup citra
    # nyaris kosong tanpa mencemari yang sudah terlabeli baik. Selalu ditandai
    # untuk tinjau manual.
    "cadangan_gigitan_ambang": 4,   # < 4 kotak -> coba gigitan tepi
    "gigitan_depth_min":    0.030,   # kedalaman ceruk ternorm diagonal daun
    "maks_gigitan":            12,

    # ambang penandaan (flag) untuk antrian tinjau manual
    "luas_daun_min":    0.03,   # daun < 3% frame -> mencurigakan
    "luas_daun_maks":   0.95,
}

WARNA_KOTAK = {           # BGR, hanya untuk pratinjau
    "daun_utuh":  (255, 0, 255),
    "lubang":     (0, 0, 255),
    "lubang_bersih":   (0, 0, 255),
    "lubang_berwarna": (0, 140, 255),
    "bercak":     (0, 255, 0),
    "mottling":   (0, 200, 255),
    "gigitan":    (255, 128, 0),
}


# ======================================================================
# SEGMENTASI DAUN
# ======================================================================
def segmentasi_daun(bgr, cfg):
    """
    Pisahkan daun dari latar putih.

    Latar foto berwarna putih tetapi tidak murni: ada bayangan abu-abu lembut
    dan color cast yang berbeda antar foto. Karena itu ambang tidak dipatok
    angka tetap, melainkan dihitung Otsu pada kanal saturasi per gambar --
    daun (hijau/kuning/coklat) punya saturasi jelas, sedangkan latar putih
    maupun bayangan abu sama-sama nyaris tak bersaturasi.

    Mengembalikan:
        mask_daun : siluet daun TERISI PENUH (lubang ikut terisi)
        mask_isi  : piksel yang benar-benar jaringan daun (lubang kosong)
        lubang    : mask area berlatar yang terkurung di dalam daun
        kontur_daun : daftar kontur helai daun (bisa >1 bila ada daun kedua)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    _, m_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Lesi hitam/nekrotik saturasinya rendah tapi gelap -- tanpa baris ini
    # bagian itu ikut terbuang sebagai latar.
    m_gelap = ((val < cfg["v_gelap"]) * 255).astype(np.uint8)
    mask_isi = cv2.bitwise_or(m_sat, m_gelap)

    inti = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_isi = cv2.morphologyEx(mask_isi, cv2.MORPH_OPEN, inti)

    kontur, _ = cv2.findContours(mask_isi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not kontur:
        return None, None, None, []

    luas = [cv2.contourArea(k) for k in kontur]
    luas_maks = max(luas)
    if luas_maks <= 0:
        return None, None, None, []

    # Sebagian foto memuat helai daun kedua di tepi frame. Bila dibiarkan,
    # daun itu tidak berlabel dan justru mengajari model untuk mengabaikan daun.
    kontur_daun = [k for k, a in zip(kontur, luas)
                   if a >= cfg["frac_daun_kedua"] * luas_maks]

    mask_daun = np.zeros(mask_isi.shape, np.uint8)
    cv2.drawContours(mask_daun, kontur_daun, -1, 255, cv2.FILLED)

    # Lubang = area berlatar yang terkurung di dalam siluet daun.
    lubang = cv2.bitwise_and(mask_daun, cv2.bitwise_not(mask_isi))
    lubang = cv2.morphologyEx(lubang, cv2.MORPH_OPEN, inti)

    return mask_daun, mask_isi, lubang, kontur_daun


# ======================================================================
# ANALISIS WARNA
# ======================================================================
def _stat_kokoh(nilai):
    """Median + MAD -- dipakai sebagai pengganti mean/std agar lesi yang
    justru ingin dideteksi tidak ikut menggeser ambangnya sendiri."""
    med = float(np.median(nilai))
    mad = float(np.median(np.abs(nilai - med))) + 1e-6
    return med, mad * 1.4826


def deteksi_lesi_warna(bgr, mask_jaringan, cfg):
    """
    Tandai piksel jaringan yang warnanya menyimpang ke arah merah/coklat.

    Pemilihan kanal a* bukan tebakan, melainkan hasil sapuan empiris atas 20
    citra per kelas (persen piksel daun yang lolos ambang):

        ambang    healthy   mosaic   leaf_spot   insect
        b* +8      6,28%    10,59%     6,47%      4,22%   -> menandai MOSAIC
        L* -25     8,68%     8,80%     8,33%      9,23%   -> tidak memisahkan
        a* +12     0,93%     0,74%     2,00%      2,03%   -> MEMISAHKAN 2-3x

    Jaringan nekrotik bergeser ke merah; mottling mosaic menaikkan kuning
    tanpa menyentuh merah. Karena itu hanya a* yang dipakai -- b* dan L*
    justru menyuntikkan derau bayangan dan tulang daun.

    Ambang memakai DUA syarat sekaligus:
        relatif (median + k x MAD) -- menyesuaikan color cast tiap foto
        absolut (median + delta)   -- tanpa ini mustahil menghasilkan nol,
                                      sebab ambang relatif selalu memotong
                                      persentil teratas walau daunnya mulus
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    piksel = mask_jaringan > 0
    if piksel.sum() < 100:
        return np.zeros(mask_jaringan.shape, np.uint8)

    a = lab[:, :, 1].astype(np.float32)
    med_a, sd_a = _stat_kokoh(a[piksel])
    ambang = med_a + max(cfg["k_merah"] * sd_a, cfg["delta_merah"])

    lesi = (((a > ambang) & piksel) * 255).astype(np.uint8)
    inti = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(lesi, cv2.MORPH_OPEN, inti)


def gigitan_tepi(kontur_daun, cfg):
    """
    Cari takik pada tepi daun akibat gigitan serangga.

    Gigitan tepi tidak menembus daun -- ia mengubah bentuk siluet. Titik
    terjauh setiap convexity defect adalah dasar takik; kotak dibuat meliputi
    segitiga mulut-hull -> dasar takik, yaitu perkiraan jaringan yang hilang.
    Kedalaman disaring relatif diagonal daun agar tak bergantung ukuran foto.

    Dipakai hanya sebagai cadangan (lihat catatan di CFG): daun terung
    berlobus alami, sehingga tanpa prior "citra ini nyaris tak berkotak"
    detektor ini akan salah menandai lekuk daun sebagai gigitan.
    """
    k = max(kontur_daun, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(k)
    diag = (w * w + h * h) ** 0.5
    hull = cv2.convexHull(k, returnPoints=False)
    if hull is None or len(hull) < 4:
        return []
    defects = cv2.convexityDefects(k, hull)
    if defects is None:
        return []

    kotak = []
    for i in range(defects.shape[0]):
        # Bentuk kembalian cv2.convexityDefects berubah antar versi OpenCV:
        #   4.x -> (N, 1, 4)   sehingga defects[i] masih berupa array 2 dimensi
        #   5.x -> (N, 4)      sehingga defects[i] langsung 4 nilai
        # Tanpa penyesuaian ini, OpenCV 5 melempar
        # "cannot unpack non-iterable numpy.int32 object".
        d = defects[i]
        if getattr(d, "ndim", 1) > 1:
            d = d[0]
        s0, e0, f0, depth = (int(v) for v in d[:4])
        if depth / 256.0 / diag < cfg["gigitan_depth_min"]:
            continue
        S, E, F = k[s0][0], k[e0][0], k[f0][0]
        xs = [int(S[0]), int(E[0]), int(F[0])]
        ys = [int(S[1]), int(E[1]), int(F[1])]
        bx, by = min(xs), min(ys)
        kotak.append((bx, by, max(xs) - bx, max(ys) - by))

    kotak.sort(key=lambda t: t[2] * t[3], reverse=True)
    return kotak[:cfg["maks_gigitan"]]


def deteksi_mottling(bgr, mask_jaringan, cfg):
    """
    Tandai area menguning menyebar (mottling / klorosis) lewat kanal b*.

    HANYA sah dipanggil untuk citra yang label folder-nya sudah menyatakan
    sakit. Pada citra sembarang, b* tidak memisahkan sehat dari sakit
    (lihat tabel di deteksi_lesi_warna), jadi tanpa prior folder detektor ini
    akan menandai daun sehat juga.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    piksel = mask_jaringan > 0
    if piksel.sum() < 100:
        return np.zeros(mask_jaringan.shape, np.uint8)

    b = lab[:, :, 2].astype(np.float32)
    med_b, sd_b = _stat_kokoh(b[piksel])
    ambang = med_b + max(cfg["k_kuning"] * sd_b, cfg["delta_kuning"])

    mott = (((b > ambang) & piksel) * 255).astype(np.uint8)
    inti = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mott, cv2.MORPH_CLOSE, inti)


def lubang_asli(hsv, kontur, latar_v, cfg):
    """
    Bedakan lubang tembus dari kilau spekular di permukaan daun.

    Keduanya sama-sama terang dan hampir tak bersaturasi, jadi keduanya lolos
    dari segmentasi. Pembedanya: lubang menampakkan LATAR foto, sehingga
    terang dan netralnya harus setara latar; kilau tetap duduk di atas
    jaringan daun dan biasanya lebih redup serta masih menyimpan rona hijau.
    """
    kanvas = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(kanvas, [kontur], -1, 255, cv2.FILLED)
    isi = kanvas > 0
    if isi.sum() < 8:
        return False
    # Persentil, bukan median: pada lubang kecil separuh piksel adalah tepi
    # antialias (campuran daun-latar) yang redup dan bersaturasi, sehingga
    # median menolak lubang yang sebetulnya sah.
    v = float(np.percentile(hsv[:, :, 2][isi], 75))
    s = float(np.percentile(hsv[:, :, 1][isi], 25))
    return (latar_v - v) <= cfg["toleransi_v_lubang"] and s <= cfg["maks_sat_lubang"]


def cincin_menyimpang(hsv, mask_daun, kontur_lubang, tebal, cfg):
    """
    Periksa cincin jaringan di sekeliling sebuah lubang.

    Inilah pembeda antara kedua kelas berlubang:
        transisi bersih hijau -> latar   = gigitan serangga
        ada halo kuning/coklat/hitam     = bercak daun yang jaringannya jebol

    Mengembalikan proporsi piksel cincin yang warnanya menyimpang dari hijau.
    """
    kanvas = np.zeros(mask_daun.shape, np.uint8)
    cv2.drawContours(kanvas, [kontur_lubang], -1, 255, cv2.FILLED)

    inti = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tebal * 2 + 1,) * 2)
    cincin = cv2.subtract(cv2.dilate(kanvas, inti), kanvas)
    cincin = cv2.bitwise_and(cincin, mask_daun)   # hanya jaringan, bukan latar

    piksel = cincin > 0
    if piksel.sum() < 20:
        return 0.0

    h = hsv[:, :, 0][piksel]
    v = hsv[:, :, 2][piksel]
    hijau = (h >= cfg["hijau_h_min"]) & (h <= cfg["hijau_h_maks"]) & (v >= cfg["v_gelap"])
    return float(1.0 - hijau.mean())


# ======================================================================
# KOTAK
# ======================================================================
def kotak_dari_mask(mask, gabung_px, min_luas, maks_kotak):
    """
    Ubah mask lesi jadi daftar kotak.

    Dilasi sebelum findContours membuat lesi yang berdekatan melebur jadi satu
    kotak -- meniru cara anotator manusia menggabungkan satu area bercak,
    sekaligus menekan jumlah instance agar tidak meledak.
    """
    if gabung_px > 0:
        inti = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gabung_px * 2 + 1,) * 2)
        mask = cv2.dilate(mask, inti)

    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kotak = []
    for k in kontur:
        if cv2.contourArea(k) < min_luas:
            continue
        x, y, w, h = cv2.boundingRect(k)
        kotak.append((x, y, w, h))

    berlebih = len(kotak) > maks_kotak
    kotak.sort(key=lambda t: t[2] * t[3], reverse=True)
    return kotak[:maks_kotak], berlebih


def ke_yolo(kotak, lebar, tinggi, id_kelas):
    """Konversi (x, y, w, h) piksel -> baris label YOLO ternormalisasi."""
    baris = []
    for x, y, w, h in kotak:
        xc = (x + w / 2) / lebar
        yc = (y + h / 2) / tinggi
        baris.append(f"{id_kelas} {xc:.6f} {yc:.6f} {w / lebar:.6f} {h / tinggi:.6f}")
    return baris


# ======================================================================
# PEMROSESAN SATU CITRA
# ======================================================================
def proses_citra(path, id_kelas, mode, cfg):
    """Kembalikan dict hasil: kotak, statistik, dan daftar penanda tinjau."""
    hasil = {
        "file": path.name, "kotak": [], "jenis": [], "tanda": [],
        "n_lubang": 0, "n_lubang_berwarna": 0, "n_bercak": 0, "n_mottling": 0,
        "n_gigitan": 0, "luas_daun_frac": 0.0, "n_daun": 0,
    }

    # np.fromfile dipakai agar path berkarakter non-ASCII tetap terbaca di Windows.
    buf = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        hasil["tanda"].append("gagal_baca")
        return hasil, None

    skala = cfg["sisi_maks"] / max(bgr.shape[:2])
    if skala < 1.0:
        bgr = cv2.resize(bgr, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)
    T, L = bgr.shape[:2]

    mask_daun, mask_isi, lubang, kontur_daun = segmentasi_daun(bgr, cfg)
    if mask_daun is None:
        hasil["tanda"].append("daun_tak_ditemukan")
        return hasil, bgr

    hasil["n_daun"] = len(kontur_daun)
    hasil["luas_daun_frac"] = float((mask_daun > 0).sum()) / (T * L)
    if hasil["n_daun"] > 1:
        hasil["tanda"].append("banyak_daun")
    if hasil["luas_daun_frac"] < cfg["luas_daun_min"]:
        hasil["tanda"].append("daun_kecil")
    if hasil["luas_daun_frac"] > cfg["luas_daun_maks"]:
        hasil["tanda"].append("daun_besar")

    lebar_daun = max(cv2.boundingRect(max(kontur_daun, key=cv2.contourArea))[2], 1)
    luas_daun_px = max(float((mask_daun > 0).sum()), 1.0)

    # --- analisis lubang (dipakai di semua mode, minimal sebagai verifikasi) ---
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    tebal_cincin = max(int(cfg["cincin_frac"] * lebar_daun), 2)
    min_luas_px = cfg["min_luas_frac"] * luas_daun_px

    min_luas_lubang_px = cfg["min_luas_lubang_frac"] * luas_daun_px

    # Terang latar diukur dari piksel di luar daun pada foto itu sendiri --
    # sebagian foto latarnya putih terang, sebagian lagi keabuan.
    luar = mask_daun == 0
    latar_v = float(np.median(hsv[:, :, 2][luar])) if luar.sum() > 100 else 255.0

    kontur_lubang, _ = cv2.findContours(lubang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lubang_bersih, lubang_berwarna = [], []
    for k in kontur_lubang:
        if cv2.contourArea(k) < min_luas_lubang_px:
            continue
        if not lubang_asli(hsv, k, latar_v, cfg):
            continue
        rasio = cincin_menyimpang(hsv, mask_isi, k, tebal_cincin, cfg)
        (lubang_berwarna if rasio >= cfg["cincin_ambang"] else lubang_bersih).append(
            cv2.boundingRect(k))
    hasil["n_lubang"] = len(lubang_bersih) + len(lubang_berwarna)
    hasil["n_lubang_berwarna"] = len(lubang_berwarna)

    # --- bercak berwarna (tanpa lubang) ---
    # tepi daun dikikis dulu supaya rim antialias tidak terbaca sebagai lesi
    kikis = max(int(cfg["erosi_tepi_frac"] * lebar_daun), 1)
    inti_kikis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kikis * 2 + 1,) * 2)
    mask_dalam = cv2.erode(mask_isi, inti_kikis)

    mask_bercak = deteksi_lesi_warna(bgr, mask_dalam, cfg)
    gabung_px = max(int(cfg["gabung_frac"] * lebar_daun), 1)
    kotak_bercak, berlebih = kotak_dari_mask(
        mask_bercak, gabung_px, min_luas_px, cfg["maks_kotak"])
    hasil["n_bercak"] = len(kotak_bercak)
    if berlebih:
        hasil["tanda"].append("kotak_berlebih")

    # --- rakit label sesuai mode ---
    if mode == "daun_utuh":
        for k in kontur_daun:
            hasil["kotak"].append(cv2.boundingRect(k))
            hasil["jenis"].append("daun_utuh")
        # healthy didefinisikan sebagai KETIADAAN gejala: satu lesi saja sudah
        # cukup untuk meragukan labelnya, jadi citra ditandai -- bukan
        # dianotasi ganda. Mosaic dilewati karena mottling-nya justru wajar.
        if id_kelas == KELAS["healthy_leaf"][0] and (hasil["n_bercak"] or hasil["n_lubang"]):
            hasil["tanda"].append("gejala_di_healthy")

    else:  # mode "lesi"
        # Lubang bersih dan lubang berhalo dicatat TERPISAH, bukan sama-sama
        # "lubang". Menurut aturan patologi proyek ini, lubang polos = insect
        # dan lubang berhalo = leaf_spot -- pembedaan itu yang dipakai skrip
        # terapkan_instruksi_label.py untuk citra bergejala campuran.
        for b in lubang_bersih:
            hasil["kotak"].append(b)
            hasil["jenis"].append("lubang_bersih")
        for b in lubang_berwarna:
            hasil["kotak"].append(b)
            hasil["jenis"].append("lubang_berwarna")
        for b in kotak_bercak:
            hasil["kotak"].append(b)
            hasil["jenis"].append("bercak")

        # Cadangan mottling: sebagian citra leaf_spot gejalanya bukan bercak
        # merah melainkan menguning menyebar. Baru dijalankan bila sinyal utama
        # (lubang + bercak merah) nyaris kosong, supaya citra yang bercaknya
        # jelas tidak ikut tertimbun kotak area kuning yang besar-besar.
        if (id_kelas == KELAS["leaf_spot_disease"][0]
                and len(hasil["kotak"]) < cfg["ambang_cadangan"]):
            mask_mott = deteksi_mottling(bgr, mask_dalam, cfg)
            kotak_mott, _ = kotak_dari_mask(
                mask_mott,
                max(int(cfg["gabung_mottling_frac"] * lebar_daun), 1),
                cfg["min_luas_mottling_frac"] * luas_daun_px,
                cfg["maks_kotak_mottling"])
            if kotak_mott:
                hasil["kotak"].extend(kotak_mott)
                hasil["jenis"].extend(["mottling"] * len(kotak_mott))
                hasil["n_mottling"] = len(kotak_mott)
                # Selalu ditinjau: mottling visualnya mirip mosaic, jadi
                # penentuan kelasnya tidak boleh diserahkan ke algoritma.
                hasil["tanda"].append("mottling_cadangan")

        # Cadangan gigitan tepi -- hanya insect, hanya bila kotak masih sedikit.
        # Gigitan tepi tak berongga dan tak berwarna, jadi lolos dari deteksi
        # lubang maupun bercak; inilah jaring penangkap terakhirnya.
        if (id_kelas == KELAS["insect_pest_disease"][0]
                and len(hasil["kotak"]) < cfg["cadangan_gigitan_ambang"]):
            kotak_gigitan = gigitan_tepi(kontur_daun, cfg)
            if kotak_gigitan:
                hasil["kotak"].extend(kotak_gigitan)
                hasil["jenis"].extend(["gigitan"] * len(kotak_gigitan))
                hasil["n_gigitan"] = len(kotak_gigitan)
                # Ditandai: takik geometris kadang lekuk alami, perlu diperiksa.
                hasil["tanda"].append("gigitan_tepi_cadangan")

        if not hasil["kotak"]:
            # File label kosong = "tidak ada penyakit di gambar ini" bagi YOLO.
            # Racun untuk training, jadi wajib masuk antrian tinjau.
            hasil["tanda"].append("tanpa_kotak")

        # Verifikasi silang isi citra terhadap folder asalnya.
        if id_kelas == KELAS["insect_pest_disease"][0]:
            if hasil["n_lubang"] and len(lubang_berwarna) > len(lubang_bersih):
                hasil["tanda"].append("kontradiksi_mirip_leafspot")
        elif id_kelas == KELAS["leaf_spot_disease"][0]:
            if hasil["n_lubang"] and not lubang_berwarna and not kotak_bercak:
                hasil["tanda"].append("kontradiksi_mirip_insect")

        if len(hasil["kotak"]) > cfg["maks_kotak"]:
            urut = sorted(range(len(hasil["kotak"])),
                          key=lambda i: hasil["kotak"][i][2] * hasil["kotak"][i][3],
                          reverse=True)[:cfg["maks_kotak"]]
            hasil["kotak"] = [hasil["kotak"][i] for i in urut]
            hasil["jenis"] = [hasil["jenis"][i] for i in urut]
            if "kotak_berlebih" not in hasil["tanda"]:
                hasil["tanda"].append("kotak_berlebih")

    hasil["ukuran"] = (L, T)
    return hasil, bgr


def gambar_pratinjau(bgr, hasil, judul):
    """Render kotak di atas citra untuk pemeriksaan visual (QC)."""
    kanvas = bgr.copy()
    tebal = max(1, kanvas.shape[1] // 400)
    for (x, y, w, h), jenis in zip(hasil["kotak"], hasil["jenis"]):
        cv2.rectangle(kanvas, (x, y), (x + w, y + h), WARNA_KOTAK[jenis], tebal)
    teks = f"{judul} | {len(hasil['kotak'])} kotak"
    if hasil["tanda"]:
        teks += " | " + ",".join(hasil["tanda"])
    cv2.putText(kanvas, teks, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return kanvas


# ======================================================================
# SELF-CHECK
# ======================================================================
def uji_mandiri():
    """
    Citra sintetis dengan jawaban yang sudah diketahui. Gagal berarti ada
    logika deteksi yang rusak -- dijalankan sebelum melepas ke 2000+ gambar.
    """
    cfg = dict(CFG)
    kanvas = np.full((600, 600, 3), 250, np.uint8)          # latar putih
    cv2.circle(kanvas, (300, 300), 200, (60, 150, 70), -1)  # daun hijau

    cv2.circle(kanvas, (230, 260), 18, (250, 250, 250), -1)  # lubang bersih
    cv2.circle(kanvas, (380, 340), 26, (40, 130, 190), -1)   # halo kuning-coklat
    cv2.circle(kanvas, (380, 340), 14, (250, 250, 250), -1)  # lubang di dalamnya
    cv2.circle(kanvas, (250, 380), 20, (35, 60, 90), -1)     # bercak coklat, tanpa lubang

    tmp = Path(KELUARAN) / "_uji.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp), kanvas)

    hasil, _ = proses_citra(tmp, KELAS["leaf_spot_disease"][0], "lesi", cfg)
    tmp.unlink(missing_ok=True)

    assert hasil["n_daun"] == 1, f"daun harus 1, dapat {hasil['n_daun']}"
    assert hasil["n_lubang"] == 2, f"lubang harus 2, dapat {hasil['n_lubang']}"
    assert hasil["n_lubang_berwarna"] == 1, \
        f"lubang berhalo harus 1, dapat {hasil['n_lubang_berwarna']}"
    assert hasil["n_bercak"] >= 1, "bercak coklat tanpa lubang tidak terdeteksi"
    assert 0.15 < hasil["luas_daun_frac"] < 0.55, \
        f"luas daun tidak wajar: {hasil['luas_daun_frac']:.3f}"

    baris = ke_yolo(hasil["kotak"], *hasil["ukuran"], 2)
    for b in baris:
        nilai = [float(x) for x in b.split()[1:]]
        assert all(0.0 <= n <= 1.0 for n in nilai), f"label di luar rentang: {b}"

    print("uji mandiri LULUS --", hasil["n_lubang"], "lubang,",
          hasil["n_lubang_berwarna"], "berhalo,", hasil["n_bercak"], "bercak")
    return True


# ======================================================================
# MAIN
# ======================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sampel", type=int, default=0,
                   help="proses N gambar per kelas saja (0 = seluruh dataset)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--uji", action="store_true", help="jalankan self-check lalu keluar")
    p.add_argument("--tanpa-pratinjau", action="store_true",
                   help="lewati penulisan citra pratinjau")
    p.add_argument("--bersih", action="store_true",
                   help="hapus label & pratinjau lama sebelum mulai. WAJIB dipakai "
                        "setelah citra berpindah kelas, kalau tidak label lama "
                        "tertinggal di folder kelas asalnya")
    arg = p.parse_args()

    if arg.uji:
        return 0 if uji_mandiri() else 1

    if arg.bersih:
        # Tanpa ini, label citra yang sudah pindah kelas tetap tertinggal di
        # folder kelas lamanya. autolabel hanya MENULIS, tidak pernah menghapus,
        # jadi label basi itu akan ikut terbawa saat split dibangun ulang dan
        # citra bisa punya dua label dengan kelas berbeda.
        import shutil as _sh
        n = 0
        for sub in ("labels", "pratinjau", "manual"):
            d = KELUARAN / sub
            if d.is_dir():
                n += sum(1 for _ in d.rglob("*") if _.is_file())
                _sh.rmtree(d)
        print(f"--bersih: {n} berkas keluaran lama dihapus\n")

    uji_mandiri()   # selalu diverifikasi dulu sebelum menyentuh dataset

    KELUARAN.mkdir(parents=True, exist_ok=True)
    tulis_pratinjau = not arg.tanpa_pratinjau

    laporan = []
    rekap = {}

    for folder, (id_kelas, mode) in KELAS.items():
        dir_citra = DIR_SUMBER / folder
        if not dir_citra.is_dir():
            print(f"LEWATI {folder}: folder tidak ada")
            continue

        berkas = sorted(dir_citra.glob("*.jpg")) + sorted(dir_citra.glob("*.jpeg"))
        if arg.sampel:
            random.Random(arg.seed).shuffle(berkas)
            berkas = berkas[:arg.sampel]

        dir_label = KELUARAN / "labels" / folder
        dir_label.mkdir(parents=True, exist_ok=True)
        if tulis_pratinjau:
            (KELUARAN / "pratinjau" / folder).mkdir(parents=True, exist_ok=True)

        # Citra bertanda tetap dianotasi otomatis seperti yang lain, lalu
        # SALINANNYA dikumpulkan di sini -- label awal ikut serta supaya
        # peninjau tinggal mengoreksi, bukan menganotasi dari nol.
        dir_manual = KELUARAN / "manual" / folder
        for sub in ("images", "labels", "pratinjau"):
            (dir_manual / sub).mkdir(parents=True, exist_ok=True)

        n_kotak, n_manual = 0, 0
        print(f"\n{folder} (id={id_kelas}, mode={mode}) -- {len(berkas)} citra")

        for i, path in enumerate(berkas, 1):
            hasil, bgr = proses_citra(path, id_kelas, mode, CFG)

            baris = ke_yolo(hasil["kotak"], *hasil["ukuran"], id_kelas) if hasil["kotak"] else []
            isi_label = "\n".join(baris) + ("\n" if baris else "")
            (dir_label / f"{path.stem}.txt").write_text(isi_label, encoding="utf-8")
            n_kotak += len(baris)

            gambar_qc = None
            if tulis_pratinjau and bgr is not None:
                gambar_qc = gambar_pratinjau(bgr, hasil, path.stem)
                cv2.imwrite(str(KELUARAN / "pratinjau" / folder / f"{path.stem}.jpg"),
                            gambar_qc, [cv2.IMWRITE_JPEG_QUALITY, 85])

            if hasil["tanda"]:
                n_manual += 1
                shutil.copy2(path, dir_manual / "images" / path.name)
                (dir_manual / "labels" / f"{path.stem}.txt").write_text(
                    isi_label, encoding="utf-8")
                if gambar_qc is not None:
                    cv2.imwrite(str(dir_manual / "pratinjau" / f"{path.stem}.jpg"),
                                gambar_qc, [cv2.IMWRITE_JPEG_QUALITY, 85])

            laporan.append({
                "kelas": folder, "id": id_kelas, "file": path.name,
                "n_kotak": len(hasil["kotak"]), "n_lubang": hasil["n_lubang"],
                "n_lubang_berwarna": hasil["n_lubang_berwarna"],
                "n_bercak": hasil["n_bercak"], "n_mottling": hasil["n_mottling"],
                "n_daun": hasil["n_daun"],
                "luas_daun_frac": round(hasil["luas_daun_frac"], 4),
                "perlu_manual": bool(hasil["tanda"]),
                # Lubang dibobot 2x: buktinya geometris (latar tembus di dalam
                # daun), jauh lebih jarang keliru daripada bercak yang berasal
                # dari ambang statistik warna.
                "skor_gejala": hasil["n_lubang"] * 2 + hasil["n_bercak"],
                "tanda": ";".join(hasil["tanda"]),
            })

            if i % 100 == 0:
                print(f"  {i}/{len(berkas)}")

        rekap[folder] = (len(berkas), n_kotak, n_manual)
        print(f"  selesai: {len(berkas)} citra, {n_kotak} kotak, {n_manual} perlu manual")

    # --- laporan ---
    def tulis_csv(path, baris):
        """CSV yang sedang dibuka di Excel terkunci di Windows. Menggagalkan
        seluruh proses karena itu berarti mengulang 2000+ citra, jadi cukup
        alihkan ke berkas bernama lain."""
        for calon in (path, path.with_name(path.stem + "_baru.csv")):
            try:
                with calon.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
                    w.writeheader()
                    w.writerows(baris)
                return calon
            except PermissionError:
                print(f"  ! {calon.name} terkunci (terbuka di Excel?)")
        raise PermissionError(f"tidak bisa menulis {path}")

    path_csv = tulis_csv(KELUARAN / "laporan.csv", laporan)

    # Diurutkan terparah dulu supaya peninjauan bisa dihentikan begitu
    # sisa antrian sudah bersih, alih-alih menyisir semuanya.
    ditandai = sorted((r for r in laporan if r["tanda"]),
                      key=lambda r: (r["kelas"], -r["skor_gejala"]))
    path_flag = tulis_csv(KELUARAN / "perlu_ditinjau.csv", ditandai)

    (KELUARAN / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(KELAS)}\nnames:\n" +
        "".join(f"  {i}: {n}\n" for n, (i, _) in
                sorted(KELAS.items(), key=lambda kv: kv[1][0])),
        encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"{'kelas':<24}{'citra':>8}{'kotak':>9}{'rata2':>8}{'manual':>9}{'%':>7}")
    for folder, (n_img, n_box, n_man) in rekap.items():
        print(f"{folder:<24}{n_img:>8}{n_box:>9}{n_box / max(n_img, 1):>8.1f}"
              f"{n_man:>9}{100 * n_man / max(n_img, 1):>6.0f}%")

    print(f"\nperlu ditinjau: {len(ditandai)} dari {len(laporan)} citra")
    hitung = {}
    for r in ditandai:
        for t in r["tanda"].split(";"):
            hitung[t] = hitung.get(t, 0) + 1
    for t, n in sorted(hitung.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<28}{n:>5}")

    print(f"\nlabel semua : {KELUARAN / 'labels'}")
    print(f"folder manual: {KELUARAN / 'manual'}")
    print(f"laporan     : {path_csv}")
    print(f"daftar manual: {path_flag}")
    if tulis_pratinjau:
        print(f"pratinjau   : {KELUARAN / 'pratinjau'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
