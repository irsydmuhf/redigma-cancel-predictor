# Rekap Faktor Penyebab Pembatalan Pesanan REDIGMA

Alat bantu diagnostik berbasis Streamlit dari skripsi **"Klasifikasi Pembatalan Pesanan
E-Commerce Menggunakan Logistic Regression dan XGBoost, dan Analisis Faktor"**
(Irsyad Muhamad Firdaus, Teknik Informatika, Universitas Muhammadiyah Magelang).

Sesuai konsep Sub-bab 3.2.3 skripsi: alat ini **bukan** prediktor real-time yang
menaksir pesanan mana yang akan dibatalkan. Alat ini menganalisis pesanan yang
**statusnya sudah diketahui batal**, mencari faktor SHAP paling dominan di balik
tiap pembatalan tersebut, lalu merekapnya jadi tabel & grafik yang mudah dibaca
tim bisnis -- tanpa menampilkan skor probabilitas atau label prediksi apa pun.

Alurnya:

1. Upload file ekspor **"Order SKU List"** langsung dari platform (xlsx apa
   adanya, tanpa perlu diedit) -- app mengagregasi baris-baris SKU per Order ID,
   lalu otomatis mengambil hanya pesanan yang **sudah berstatus batal**.
2. Filter opsional: rentang tanggal, kategori produk, provinsi, metode
   pembayaran, dan nama toko.
3. Untuk tiap pesanan batal, dihitung nilai kontribusi **SHAP** per fitur
   (model: **XGBoost skenario S3**, Macro F1 tertinggi di antara lima skenario
   yang dibandingkan di Bab 4) -- fitur dengan kontribusi absolut terbesar
   ditetapkan sebagai faktor dominan penyebab pesanan itu batal.
4. Hasilnya direkap: ringkasan naratif otomatis, tabel & grafik faktor dominan,
   plus breakdown rasio pembatalan per toko/provinsi/kategori/produk dan tren
   bulanan -- semua bisa diunduh sebagai CSV.

## Struktur folder

```
.
├── app.py                     # Halaman utama Streamlit
├── src/
│   ├── schema.py               # Konstanta bersama (nama kolom, hyperparameter, threshold)
│   ├── raw_data.py             # Agregasi baris-SKU mentah -> level pesanan (dipakai train.py & app.py)
│   ├── target_encoder.py       # SmoothedTargetEncoder (disalin dari pipeline skripsi)
│   ├── features.py             # Fitur untuk form input manual (dipakai scripts/smoke_test.py)
│   ├── predict.py               # Load model, hitung faktor SHAP dominan (diagnose_batch)
│   ├── product_lookup.py       # Terjemahkan Seller SKU -> nama produk asli
│   └── reference_data.py       # Info model untuk UI
├── reference/
│   └── kode_produk.xlsx        # Katalog statis Seller SKU -> nama produk (ikut di-commit)
├── models/
│   └── model_bundle.joblib     # Dihasilkan oleh scripts/train.py (ikut di-commit ke git)
├── scripts/
│   └── train.py                # Jalankan manual untuk melatih/melatih-ulang model
├── data/                       # Taruh file Excel data pesanan di sini (TIDAK di-commit)
├── requirements.txt
├── .streamlit/config.toml
└── .gitignore
```

## Kenapa modelnya cuma pakai 16 kolom, bukan 29 fitur hasil rekayasa fitur?

Bab 3 skripsi awalnya merekayasa 29 fitur (termasuk fitur marketing dari nama
produk seperti `bundle_size`, `n_hype_words`, dan fitur waktu seperti `hour`,
`is_weekend`), tapi model yang **benar-benar dilaporkan performanya** di Tabel
4.1/4.1b (S3: Macro F1 = 0,5249) hanya memakai 16 fitur yang lolos uji
signifikansi statistik (Sub-bab 4.3.1-4.3.2), mengikuti
`02_pipeline/redigma_pipeline_v5_selected.py` pada repo skripsi. 13 fitur
lainnya terbukti tidak signifikan secara statistik dan dibuang dari model
final. Supaya demo ini konsisten dengan angka yang dipertahankan di sidang,
`scripts/train.py` mereplikasi pipeline v5 tersebut persis -- lihat
`src/schema.py` untuk daftar 16 fitur yang dipakai.

## Format file yang diterima

File yang diupload harus persis format ekspor **"Order SKU List"** platform
(satu baris per SKU/lini pesanan, kolom yang sama seperti `REQUIRED_RAW_COLUMNS`
di `src/raw_data.py`) -- ini format yang sama dengan sumber data training, jadi
tidak perlu diedit/dirapikan dulu sebelum upload. Kolom `Nama Toko` bersifat
opsional: kalau ada, dipakai untuk filter & breakdown per toko; kalau tidak
ada, kolom itu diisi "Tidak diketahui" dan filter toko otomatis tidak berarti.

Catatan teknis: sebagian ekspor "Order SKU List" (mis. dari TikTok Shop Seller
Center) punya metadata dimensi xlsx yang salah/tidak lengkap -- Microsoft Excel
mengabaikannya dan tetap membaca semua kolom, tapi pembaca xlsx yang lebih ketat
bisa salah baca (cuma dapat 1 kolom). `src/raw_data.py` sudah menangani ini
(pakai `engine_kwargs={"read_only": False}`) -- kalau suatu saat upload gagal
dengan pesan "kolom tidak ditemukan" padahal filenya kelihatan normal di Excel,
ini kemungkinan besar penyebabnya.

## Setup

Butuh Python 3.10-3.12 (xgboost/shap belum tentu kompatibel dengan versi Python
paling baru -- kalau `pip install` gagal, coba Python 3.11).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Melatih model

1. Taruh file Excel data pesanan (kolom-kolomnya harus sama persis dengan
   dataset REDIGMA asli -- `Order ID`, `Order Status`, `Created Time`,
   `Seller SKU`, `Province`, `Payment Method`, `Product Category`, dst) di:

   ```
   data/raw_orders.xlsx
   ```

2. Jalankan:

   ```bash
   python scripts/train.py
   ```

3. Cek output di terminal -- bandingkan dengan angka referensi dari skripsi:

   ```
   Referensi dari skripsi (Tabel 4.1, S3: XGBoost + Imbalanced, 16 fitur terpilih):
     accuracy=0.8430 precision=0.2773 recall=0.0902 macro_f1=0.5249 roc_auc=0.5875
   ```

   Kalau datanya sama dengan yang dipakai skripsi, angka yang keluar seharusnya
   identik atau sangat dekat (random_state dikunci = 42). Kalau jauh berbeda,
   kemungkinan besar isi/urutan datanya sudah berubah dari yang asli.

4. Model tersimpan di `models/model_bundle.joblib`. File ini **perlu ikut di-commit**
   ke git supaya Streamlit Community Cloud bisa langsung memakainya (Streamlit
   Cloud tidak menjalankan `train.py` otomatis, hanya menjalankan `app.py`).

Kalau data berubah drastis di masa depan (SKU/kategori baru banyak bermunculan),
sebaiknya jalankan ulang `GridSearchCV` (contoh kode ada di komentar paling
bawah `scripts/train.py`), lalu update `BEST_XGB_PARAMS` di `src/schema.py`
dengan parameter terbaik yang baru -- jangan cuma jalankan `train.py` dengan
parameter lama begitu saja.

## Menjalankan aplikasi secara lokal

```bash
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Deploy ke Streamlit Community Cloud

1. Pastikan `models/model_bundle.joblib` sudah ter-commit & ter-push ke GitHub
   (lihat bagian "Melatih model" di atas -- harus dijalankan dulu sebelum push
   pertama kali, karena file model tidak dibuat otomatis oleh Streamlit Cloud).
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan ke repo ini.
3. Pilih branch `main` dan file utama `app.py`.
4. Deploy. Streamlit Cloud akan otomatis `pip install -r requirements.txt`.

## Keterbatasan (konsisten dengan Sub-bab 5.3 skripsi)

- Hasil analisis sebaiknya jadi salah satu bahan evaluasi internal, bukan
  kesimpulan tunggal -- performa model tergolong moderat (ROC-AUC 0,59-0,65
  pada data uji).
- Fitur risiko (`*_risk`) berbasis riwayat historis per kategori -- kombinasi
  yang sangat jarang/baru di data training akan condong ke rata-rata global.
- Data training mencakup periode sekitar satu tahun (Maret 2025 - April 2026)
  dari satu perusahaan (REDIGMA) -- generalisasi ke bisnis/periode lain terbatas.
- Kategori baru yang tidak pernah muncul saat training (mis. opsi pengiriman
  baru) bisa membuat faktor SHAP-nya tampak dominan secara semu -- lihat
  Sub-bab 4.2.5 skripsi (validasi *out-of-time*) untuk detail temuan ini.
