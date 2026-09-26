/* Sistem Deteksi Penyakit Daun Terung -- sisi peramban.
   Tidak tahu apa-apa soal cara model bekerja; hanya bicara ke /api/deteksi. */

const $ = (id) => document.getElementById(id);
const AMBANG_RAGU = parseFloat(document.body.dataset.ambangRagu || "0.25");

/* Warna kelas dibaca dari CSS -- token tetap tinggal di gaya.css (MASTER.md) */
const _cs = getComputedStyle(document.documentElement);
const WARNA = {
  leaf_spot_disease:    _cs.getPropertyValue("--kelas-leafspot").trim(),
  insect_pest_disease:  _cs.getPropertyValue("--kelas-insect").trim(),
  healthy_leaf:         _cs.getPropertyValue("--kelas-healthy").trim(),
  mosaic_virus_disease: _cs.getPropertyValue("--kelas-mosaic").trim(),
};

let PENYAKIT = {};
fetch("/api/penyakit")
  .then((r) => (r.ok ? r.json() : {}))
  .then((j) => (PENYAKIT = j))
  .catch(() => {});

function setStatus(teks, jenis = "") {
  const s = $("status");
  s.textContent = teks;
  s.className = "status " + jenis;
}

function sibuk(on) {
  document.body.classList.toggle("sibuk", on);
}

/* ---------------- komunikasi ---------------- */

async function kirim(blob, mode) {
  const fd = new FormData();
  fd.append("gambar", blob, "kirim.jpg");
  fd.append("mode", mode);
  const ac = new AbortController();
  const jam = setTimeout(() => ac.abort(), 15000);
  try {
    const r = await fetch("/api/deteksi", { method: "POST", body: fd, signal: ac.signal });
    const j = await r.json();
    if (!r.ok) throw new Error(j.galat || "Deteksi gagal.");
    return j;
  } catch (e) {
    if (e.name === "AbortError")
      throw new Error("Deteksi lebih dari 15 detik — server mungkin sibuk.");
    if (e instanceof TypeError)
      throw new Error("Sambungan ke server terputus. Jalankan ulang app.py.");
    throw e;
  } finally {
    clearTimeout(jam);
  }
}

/* ---------------- menggambar ---------------- */

function gambarKotak(kotak, lapis) {
  lapis.innerHTML = "";

  /* Kotak lesi sangat kecil (10-25 px saat ditampilkan), sedangkan labelnya
     lebih lebar dari kotaknya. Tujuh label bertumpuk tidak terbaca sama sekali.
     Karena itu hanya kotak berkepercayaan TERTINGGI per kelas yang diberi label:
     tiap kelas yang hadir tetap punya satu label acuan, sisanya mewarisi warnanya.
     Aturan MASTER.md "warna bukan satu-satunya penanda" tetap terpenuhi. */
  const wakil = new Map();
  kotak.forEach((b, i) => {
    const w = wakil.get(b.kelas);
    if (w === undefined || b.conf > kotak[w].conf) wakil.set(b.kelas, i);
  });
  const berlabel = new Set(wakil.values());

  kotak.forEach((b, i) => {
    const warna = WARNA[b.kelas] || "#000000";
    const d = document.createElement("div");
    d.className = "kotak";
    d.style.left = b.x * 100 + "%";
    d.style.top = b.y * 100 + "%";
    d.style.width = b.w * 100 + "%";
    d.style.height = b.h * 100 + "%";
    d.style.borderColor = warna;
    d.style.background = warna + "17";   // ~9% alpha
    d.style.animationDelay = i * 30 + "ms";

    if (berlabel.has(i)) {
      const u = document.createElement("u");
      u.textContent = `${b.kelas} ${b.conf.toFixed(2)}`;
      u.style.background = warna;
      d.appendChild(u);
    } else {
      d.title = `${b.kelas} ${b.conf.toFixed(2)}`;   // tetap terbaca saat disentuh
    }
    lapis.appendChild(d);
  });
}

function tampilkanInfo(kelas) {
  const box = $("info-penyakit");
  const d = PENYAKIT[kelas];
  if (!d) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML =
    '<p class="eyebrow">Tentang penyakit ini</p>' +
    `<h3>${d.nama_id}</h3>` +
    `<p><b>Gejala.</b> ${d.gejala}</p>` +
    `<p><b>Penyebab.</b> ${d.penyebab}</p>` +
    "<p><b>Penanganan.</b></p><ul>" +
    d.penanganan.map((x) => `<li>${x}</li>`).join("") +
    "</ul>" +
    `<p class="sumber-sitasi">Sumber: ${d.sumber.join("; ")}</p>`;
}

function tampilkanHasil(h) {
  gambarKotak(h.kotak, $("lapis-kotak"));

  $("kartu-hasil").hidden = false;
  if (h.kelas === null) {
    $("nama-kelas").textContent = "Tidak ada gejala terdeteksi";
    $("angka").textContent = `0 kotak · ${h.ms} ms`;
    $("meter-isi").style.width = "0%";
  } else {
    $("nama-kelas").textContent = h.kelas;
    $("angka").textContent =
      `dominasi deteksi ${(h.dominasi * 100).toFixed(1)}% · ` +
      `${h.kotak.length} kotak · ${h.ms} ms`;
    $("meter-isi").style.width = (h.dominasi * 100).toFixed(1) + "%";
  }

  const p = $("peringatan");
  const ragu = h.kelas !== null && h.conf_maks < AMBANG_RAGU;
  p.hidden = !ragu;
  if (ragu) {
    p.textContent =
      "Keyakinan rendah — pastikan gambar adalah satu helai daun terung " +
      "dengan latar polos.";
  }

  $("tabel-teknis").innerHTML =
    h.kotak.map((b) => `<tr><td>${b.kelas}</td><td>${b.conf.toFixed(3)}</td></tr>`).join("") +
    `<tr><td>conf_maks</td><td>${h.conf_maks.toFixed(3)}</td></tr>` +
    "<tr><td>ambang</td><td>0.05</td></tr>" +
    `<tr><td>mode</td><td>${h.mode}</td></tr>`;
  $("panel-teknis").hidden = false;

  if (h.kelas) tampilkanInfo(h.kelas);
  else $("info-penyakit").hidden = true;
}

/* ---------------- alur unggah ---------------- */

/* Satu jalur untuk semua sumber gambar: unggah, seret-lepas, foto contoh,
   dan jepret. Dulu prosesBerkas() dan jepret() menyalin badan yang sama,
   yang berarti dua salinan bisa menyimpang saat salah satu diubah. */
async function deteksiBlob(blob, mode, selesai) {
  const img = $("pratinjau");
  img.src = URL.createObjectURL(blob);
  img.hidden = false;
  $("video").hidden = true;
  $("ajakan").hidden = true;
  $("lapis-kotak").innerHTML = "";
  setStatus("Mendeteksi…");
  sibuk(true);
  try {
    tampilkanHasil(await kirim(blob, mode));
    setStatus("");
  } catch (e) {
    setStatus(e.message, "galat");
  } finally {
    sibuk(false);
    if (selesai) selesai();
  }
}

async function prosesBerkas(file) {
  if (!file.type.startsWith("image/")) {
    setStatus("Berkas harus gambar (JPG/PNG).", "galat");
    return;
  }
  await deteksiBlob(file, "unggah");
}

$("input-berkas").addEventListener("change", (e) => {
  if (e.target.files[0]) prosesBerkas(e.target.files[0]);
});

const zona = $("zona-unggah");
["dragenter", "dragover"].forEach((n) =>
  zona.addEventListener(n, (e) => { e.preventDefault(); zona.classList.add("seret"); }));
["dragleave", "drop"].forEach((n) =>
  zona.addEventListener(n, (e) => { e.preventDefault(); zona.classList.remove("seret"); }));
zona.addEventListener("drop", (e) => {
  if (e.dataTransfer.files[0]) prosesBerkas(e.dataTransfer.files[0]);
});

/* label bertindak sebagai tombol -- harus bisa dipicu keyboard */
document.querySelector('label[for="input-berkas"]').addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("input-berkas").click(); }
});

/* ---------------- kamera ---------------- */

let aliran = null, live = false;

async function mulaiKamera() {
  try {
    aliran = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  } catch (e) {
    const pesan = {
      NotAllowedError:
        "Akses kamera ditolak. Klik ikon kamera di address bar → Izinkan, lalu coba lagi.",
      NotFoundError: "Tidak ada kamera terdeteksi. Gunakan unggah foto.",
      NotReadableError:
        "Kamera sedang dipakai aplikasi lain (Zoom/Teams?). Tutup aplikasi itu lalu coba lagi.",
    };
    setStatus(pesan[e.name] || `Kamera gagal dinyalakan: ${e.name}`, "galat");
    return;   // mode unggah tetap hidup
  }
  const v = $("video");
  v.srcObject = aliran;
  await v.play();
  v.hidden = false;
  $("pratinjau").hidden = true;
  $("ajakan").hidden = true;
  $("tombol-jepret").hidden = false;
  $("tombol-kamera").textContent = "Matikan Kamera";
  live = true;
  loopLive();
}

function hentikanKamera() {
  live = false;
  if (aliran) aliran.getTracks().forEach((t) => t.stop());
  aliran = null;
  $("video").hidden = true;
  $("tombol-jepret").hidden = true;
  $("tombol-kamera").textContent = "Nyalakan Kamera";
}

function frameKe(lebar) {
  const v = $("video");
  const k = document.createElement("canvas");
  const s = lebar ? lebar / v.videoWidth : 1;
  k.width = Math.round(v.videoWidth * s);
  k.height = Math.round(v.videoHeight * s);
  k.getContext("2d").drawImage(v, 0, 0, k.width, k.height);
  return new Promise((res) => k.toBlob(res, "image/jpeg", 0.85));
}

/* Menahan diri tanpa timer: frame berikutnya diambil HANYA setelah
   balasan sebelumnya tiba. Tidak pernah ada dua permintaan sekaligus. */
async function loopLive() {
  while (live) {
    try {
      const h = await kirim(await frameKe(640), "live");
      if (!live) break;
      gambarKotak(h.kotak, $("lapis-kotak"));
      setStatus(`live · ${h.kotak.length} kotak · ${h.ms} ms`);
    } catch (e) {
      if (!live) break;
      setStatus("Sambungan tersendat — mencoba lagi…", "galat");
      await new Promise((r) => setTimeout(r, 1200));   // mundur, jangan berhenti diam-diam
    }
  }
}

async function jepret() {
  live = false;                                              // hentikan loop dulu
  await deteksiBlob(await frameKe(null), "jepret", hentikanKamera);
}

$("tombol-kamera").addEventListener("click", () =>
  (aliran ? hentikanKamera() : mulaiKamera()));
$("tombol-jepret").addEventListener("click", jepret);
window.addEventListener("beforeunload", hentikanKamera);


/* ---------------- foto contoh ---------------- */
/* Melewati alur yang sama persis dengan unggah biasa: ambil berkas,
   bungkus jadi File, lalu serahkan ke prosesBerkas(). */
fetch("/api/contoh")
  .then((r) => (r.ok ? r.json() : []))
  .then((daftar) => {
    if (!daftar.length) return;
    const box = $("daftar-contoh");
    box.hidden = false;
    box.innerHTML = "<span>Foto contoh:</span>";
    daftar.forEach((nama, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = "contoh " + (i + 1);
      b.title = nama;
      b.addEventListener("click", async () => {
        setStatus("Memuat contoh…");
        const blob = await fetch("/statik/contoh/" + encodeURIComponent(nama)).then((r) => r.blob());
        prosesBerkas(new File([blob], nama, { type: "image/jpeg" }));
      });
      box.appendChild(b);
    });
  })
  .catch(() => {});
