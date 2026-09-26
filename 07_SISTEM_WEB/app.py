#!/usr/bin/env python3
"""
Server web demo deteksi penyakit daun terung. -

Rute HTTP saja. Seluruh logika model ada di deteksi.py.

Pakai:
    python app.py --uji            # self-check, tidak menyalakan server
    python app.py                  # jalankan di http://localhost:5000
    python app.py --port 5001
"""

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deteksi

from flask import Flask, jsonify, render_template, request

DIR = Path(__file__).resolve().parent
MAKS_BYTE = 15 * 1024 * 1024
TIPE_OK = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = Flask(__name__, template_folder="templat", static_folder="statik")
app.config["MAX_CONTENT_LENGTH"] = MAKS_BYTE


@app.get("/")
def halaman():
    return render_template("index.html", ambang_ragu=deteksi.AMBANG_RAGU)


@app.get("/api/penyakit")
def api_penyakit():
    p = DIR / "konten" / "penyakit.json"
    if not p.exists():
        return jsonify(galat="konten/penyakit.json tidak ada."), 500
    return jsonify(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/contoh")
def api_contoh():
    # Berkasnya sendiri disajikan Flask dari statik/ -- tidak perlu rute sendiri.
    d = DIR / "statik" / "contoh"
    return jsonify(sorted(q.name for q in d.glob("*.jpg")) if d.is_dir() else [])


@app.post("/api/deteksi")
def api_deteksi():
    if "gambar" not in request.files:
        return jsonify(galat="Tidak ada gambar yang dikirim."), 400
    f = request.files["gambar"]
    if f.mimetype not in TIPE_OK:
        return jsonify(galat=f"Berkas harus gambar (JPG/PNG). "
                             f"Yang dikirim: {f.mimetype}"), 400

    data = f.read()
    if not data:
        return jsonify(galat="Berkas kosong."), 400

    try:
        import cv2
        import numpy as np
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("imdecode mengembalikan None")
    except Exception:
        return jsonify(galat="Gambar tidak bisa dibaca -- mungkin rusak. "
                             "Coba berkas lain."), 400

    try:
        hasil = deteksi.prediksi(arr)
    except Exception as e:
        app.logger.exception("prediksi gagal")
        return jsonify(galat=f"Deteksi gagal: {e}"), 500

    hasil["mode"] = request.form.get("mode", "unggah")
    return jsonify(hasil)


@app.errorhandler(413)
def kebesaran(_):
    return jsonify(galat=f"Gambar terlalu besar. "
                         f"Maksimal {MAKS_BYTE // (1024 * 1024)} MB."), 413


def uji():
    c = app.test_client()

    # tanpa berkas
    r = c.post("/api/deteksi")
    assert r.status_code == 400 and "galat" in r.get_json(), r.status_code

    # tipe salah
    r = c.post("/api/deteksi", data={
        "gambar": (io.BytesIO(b"%PDF-1.4"), "x.pdf", "application/pdf")})
    assert r.status_code == 400, r.status_code

    # gambar rusak
    r = c.post("/api/deteksi", data={
        "gambar": (io.BytesIO(b"bukan gambar"), "x.jpg", "image/jpeg")})
    assert r.status_code == 400, r.status_code

    # rute terdaftar
    rute = {str(x.rule) for x in app.url_map.iter_rules()}
    for butuh in ("/", "/api/deteksi", "/api/penyakit"):
        assert butuh in rute, (butuh, rute)

    print("SEMUA UJI LOLOS")


def prasyarat():
    """Validasi di detik pertama dan HENTIKAN -- peringatan yang tidak
    menghentikan sama saja tidak ada (pelajaran openpyxl & GPU proyek ini)."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        sys.exit("BERHENTI: ultralytics belum terpasang.\n"
                 "  .venv/Scripts/python.exe -m pip install "
                 "-r 07_SISTEM_WEB/requirements.txt")

    if not deteksi.BOBOT.exists():
        sys.exit(f"BERHENTI: bobot model tidak ketemu:\n  {deteksi.BOBOT}")

    konten = DIR / "konten" / "penyakit.json"
    if konten.exists():
        try:
            json.loads(konten.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            sys.exit(f"BERHENTI: penyakit.json rusak -- "
                     f"baris {e.lineno}: {e.msg}")

    print("Memuat model...")
    deteksi.muat()
    print(f"Model siap: {deteksi.BOBOT.name}")


def jalankan(port):
    """Nyalakan server; port sibuk dilaporkan jelas, bukan traceback."""
    try:
        app.run(host="127.0.0.1", port=port, debug=False)
    except OSError as e:
        if getattr(e, "errno", None) in (48, 98, 10048):   # mac / linux / windows
            sys.exit(f"BERHENTI: port {port} sudah dipakai.\n"
                     f"  Coba: python app.py --port {port + 1}")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--uji", action="store_true")
    p.add_argument("--port", type=int, default=5000)
    a = p.parse_args()
    if a.uji:
        uji()
    else:
        prasyarat()
        print(f"Buka: http://localhost:{a.port}")
        jalankan(a.port)
