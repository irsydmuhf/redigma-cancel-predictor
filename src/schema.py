"""
Konstanta bersama antara scripts/train.py (melatih model) dan src/features.py +
src/predict.py (dipakai app.py saat runtime). Semua nilai di sini HARUS identik
dengan yang dipakai saat training di 02_pipeline/redigma_pipeline_v5_selected.py
pada skripsi aslinya -- jangan diubah tanpa melatih ulang modelnya juga.

Sumber kebenaran: 02_pipeline/redigma_pipeline_v5_selected.py (fitur NUM_SELECTED/
TE_SELECTED/ONEHOT_SELECTED hasil seleksi statistik FDR) -- INI PIPELINE YANG
BENAR-BENAR MENGHASILKAN ANGKA TABEL 4.1/4.1b SKRIPSI (16 fitur, bukan 29).
Pipeline v2 (29 fitur, seluruh hasil rekayasa fitur tanpa seleksi statistik)
adalah versi LAMA yang sudah tidak dipakai lagi sejak revisi bimbingan yang
menambahkan tahap seleksi fitur berbasis uji signifikansi (Sub-bab 4.3.1-4.3.2).
"""

RANDOM_STATE = 42

CANCEL_LABELS = {"dibatalkan", "canceled", "cancelled"}
DONE_LABELS = {"selesai", "completed"}
COD_LABELS = {"bayar di tempat", "cash on delivery", "cash", "cod"}
DATE_FMTS = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"]

HYPE_WORDS = [
    "AMPUH", "ASLI", "ORIGINAL", " ORI ", "TERBUKTI", "TERPERCAYA", "TERLARIS",
    "GARANSI", "DIJAMIN", "TANPA EFEK SAMPING", "BPOM", "HALAL", "BEST SELLER",
    "100%", "UANG KEMBALI", "PALING", "RESMI",
]
PROMO_WORDS = ["COD", "BAYAR DITEMPAT", "BAYAR DI TEMPAT", "GRATIS ONGKIR", "ONGKIR"]

# Fitur numerik terpilih (10 dari 19 fitur numerik awal, lolos uji signifikansi
# FDR -- lihat NUM_SELECTED di redigma_pipeline_v5_selected.py). 9 fitur lain
# (hour, is_weekend, is_rush_hour, n_lines, total_discount, bundle_size,
# n_hype_words, has_promo_terms, name_length) TIDAK signifikan secara statistik
# dan sudah tidak dipakai model final -- jangan dikembalikan tanpa alasan kuat.
NUM_FEATURES = [
    "shipping_ratio", "shipping_fee", "subtotal_after", "is_cod", "discount_ratio",
    "weight", "price_per_item", "subtotal_before", "qty", "day_of_week",
]

# Kolom kategorikal berkardinalitas tinggi -> di-smoothed-target-encode jadi *_risk
TE_COLS = ["sku", "province", "payment", "category"]

# Kolom kategorikal berkardinalitas rendah -> di-label-encode (integer ordinal,
# aman untuk XGBoost karena berbasis pohon keputusan). fulfillment, preorder,
# time_category, dan rush_type SUDAH DIBUANG (tidak lolos seleksi FDR / bernilai
# konstan) -- lihat ONEHOT_SELECTED di redigma_pipeline_v5_selected.py.
ONEHOT_COLS = ["channel", "delivery"]

# Hyperparameter XGBoost hasil GridSearchCV skenario S3 (XGBoost + data imbalanced,
# 16 fitur terpilih) -- Tabel 4.1b skripsi. Macro-F1 = 0,5249 pada nilai ini
# (lihat Tabel 4.1).
BEST_XGB_PARAMS = {
    "learning_rate": 0.3,
    "max_depth": 5,
    "n_estimators": 300,
    "subsample": 0.7,
}

SMOOTHING = 20  # smoothing factor SmoothedTargetEncoder (sama dengan training asli)

# Threshold klasifikasi default utk mode "Input Manual" (scripts/smoke_test.py)
# -- app.py (mode diagnostik) TIDAK memakai nilai ini sama sekali, karena tidak
# menampilkan probabilitas/prediksi. Bukan lagi threshold optimal Tabel 4.2
# skripsi (yang sekarang berbasis S2/Logistic Regression, bukan S3/model ini).
CLASSIFICATION_THRESHOLD = 0.5
