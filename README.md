# Prediksi Pembatalan Pesanan REDIGMA

Demo interaktif berbasis Streamlit dari skripsi **"Klasifikasi Pembatalan Pesanan
E-Commerce Menggunakan Logistic Regression dan XGBoost, dan Analisis Faktor"**
(Irsyad Muhamad Firdaus, Teknik Informatika, Universitas Muhammadiyah Magelang).

Ada dua mode di aplikasinya:

- **Upload Spreadsheet** (mode utama) -- upload file ekspor **"Order SKU List"**
  langsung dari platform (xlsx apa adanya, tanpa perlu diedit), app mengagregasi
  baris-baris SKU per Order ID lalu memprediksi semua pesanan sekaligus, hasilnya
  bisa di-download sebagai CSV.
- **Input Manual** -- form satu pesanan untuk uji skenario "what-if" cepat.

Model yang dipakai: **XGBoost skenario S3** (skenario dengan Macro F1 tertinggi
di antara 4 skenario yang dibandingkan di Bab 4), lengkap dengan penjelasan
**SHAP** per-prediksi.

## Struktur folder

```
.
├── app.py                  # Halaman utama Streamlit (2 tab: upload spreadsheet & input manual)
├── src/
│   ├── schema.py            # Konstanta bersama (nama kolom, hyperparameter, threshold)
│   ├── raw_data.py          # Agregasi baris-SKU mentah -> level pesanan (dipakai train.py & app.py)
│   ├── target_encoder.py    # SmoothedTargetEncoder (disalin dari pipeline skripsi)
│   ├── features.py          # Ubah input form manual -> fitur model
│   ├── predict.py           # Load model, prediksi (satu & batch), hitung SHAP
│   └── reference_data.py    # Opsi dropdown & info model untuk UI
├── models/
│   └── model_bundle.joblib  # Dihasilkan oleh scripts/train.py (ikut di-commit ke git)
├── scripts/
│   └── train.py             # Jalankan manual untuk melatih/melatih-ulang model
├── data/                    # Taruh file Excel data pesanan di sini (TIDAK di-commit)
├── requirements.txt
├── .streamlit/config.toml
└── .gitignore
```

## Kenapa modelnya butuh 29 kolom input, bukan cuma 15 fitur di Bab 3?

Bab 3 skripsi menarasikan versi sederhana (15 fitur) untuk keterbacaan, tapi
model yang **benar-benar dilaporkan performanya** di Tabel 4.1/4.1b (Macro F1 =
0,5512) memakai fitur yang lebih lengkap dari `02_pipeline/redigma_pipeline_v2.py`
pada repo skripsi -- termasuk fitur marketing yang diekstrak dari nama produk
(`bundle_size`, `n_hype_words`, `has_promo_terms`, `name_length`) dan beberapa
kolom operasional (berat, jumlah baris pesanan, channel, dsb). Supaya demo ini
konsisten dengan angka yang dipertahankan di sidang, `scripts/train.py` mereplikasi
pipeline itu persis, bukan versi sederhana di narasi Bab 3.

Dua kolom (`Fulfillment Type` dan `Normal or Pre-order`) tidak ditanyakan di form
manual karena di seluruh data training nilainya cuma satu macam ("Fulfillment by
seller" dan "Normal") -- lihat `src/schema.FIXED_VALUES`.

## Mode "Upload Spreadsheet" -- format file yang diterima

File yang diupload harus persis format ekspor **"Order SKU List"** platform
(satu baris per SKU/lini pesanan, kolom yang sama seperti `REQUIRED_RAW_COLUMNS`
di `src/raw_data.py`) -- ini format yang sama dengan sumber data training, jadi
tidak perlu diedit/dirapikan dulu sebelum upload. Berbeda dengan `train.py`
(yang cuma memakai pesanan berstatus final -- selesai/dibatalkan, untuk
training), mode upload ini memprediksi **SEMUA** pesanan di file, termasuk yang
masih berjalan (justru itu yang mau diketahui risikonya).

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
   Referensi dari skripsi (Tabel 4.1, S3: XGBoost + Imbalanced):
     accuracy=0.8448 precision=0.3333 recall=0.1311 macro_f1=0.5512 roc_auc=0.5876
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
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan ke repo
   `irsydmuhf/redigma-cancel-predictor`.
3. Pilih branch `main` dan file utama `app.py`.
4. Deploy. Streamlit Cloud akan otomatis `pip install -r requirements.txt`.

## Keterbatasan (konsisten dengan Sub-bab 5.3 skripsi)

- Performa model tergolong moderat (ROC-AUC 0,59-0,65 pada data uji) -- hasil
  prediksi sebaiknya jadi salah satu sinyal pendukung, bukan keputusan tunggal.
- Fitur risiko (`*_risk`) berbasis riwayat historis per kategori -- kombinasi
  yang sangat jarang/baru di data training akan condong ke rata-rata global.
- Data training mencakup periode sekitar satu tahun (Maret 2025 - April 2026)
  dari satu perusahaan (REDIGMA) -- generalisasi ke bisnis/periode lain terbatas.
