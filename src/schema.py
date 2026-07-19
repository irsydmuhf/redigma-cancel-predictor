"""
Konstanta bersama antara scripts/train.py (melatih model) dan src/features.py +
src/predict.py (dipakai app.py saat runtime). Semua nilai di sini HARUS identik
dengan yang dipakai saat training di 02_pipeline/redigma_pipeline_v2.py pada
skripsi aslinya -- jangan diubah tanpa melatih ulang modelnya juga.

Sumber kebenaran: 02_pipeline/redigma_pipeline_v2.py (fungsi load_order_level,
build_xy_v2, make_pipeline) dan logs/pipeline_v2_full_run.log (hasil
GridSearchCV yang dilaporkan di Tabel 4.1 / 4.1b skripsi).
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

# Fitur numerik langsung (urutan HARUS sama dengan build_xy_v2 di pipeline asli)
NUM_FEATURES = [
    "qty", "subtotal_before", "subtotal_after", "total_discount", "shipping_fee",
    "weight", "n_lines", "discount_ratio", "shipping_ratio", "price_per_item",
    "is_cod", "hour", "day_of_week", "is_weekend", "is_rush_hour",
    "bundle_size", "n_hype_words", "has_promo_terms", "name_length",
]

# Kolom kategorikal berkardinalitas tinggi -> di-smoothed-target-encode jadi *_risk
TE_COLS = ["sku", "province", "payment", "category"]

# Kolom kategorikal berkardinalitas rendah -> di-label-encode (integer ordinal,
# aman untuk XGBoost karena berbasis pohon keputusan)
ONEHOT_COLS = ["channel", "fulfillment", "delivery", "preorder", "time_category", "rush_type"]

# Kolom yang di training HANYA punya 1 nilai unik di seluruh dataset REDIGMA
# (Fulfillment Type = "Fulfillment by seller", Normal or Pre-order = "Normal").
# Tidak ditanyakan di form karena modelnya memang tidak pernah melihat nilai lain.
FIXED_VALUES = {
    "fulfillment": "Fulfillment by seller",
    "preorder": "Normal",
}

# Hyperparameter XGBoost hasil GridSearchCV skenario S3 (XGBoost + data imbalanced)
# -- Tabel 4.1b skripsi. Macro-F1 = 0,5512 pada nilai ini (lihat Tabel 4.1).
BEST_XGB_PARAMS = {
    "learning_rate": 0.3,
    "max_depth": 5,
    "n_estimators": 300,
    "subsample": 0.8,
}

# Threshold klasifikasi -- BUKAN 0,5 default. Sesuai Sub-bab 4.2.3 & Tabel 4.2:
# F1-score kelas "cancel" tertinggi (0,258) dicapai pada threshold 0,2.
CLASSIFICATION_THRESHOLD = 0.2

SMOOTHING = 20  # smoothing factor SmoothedTargetEncoder (sama dengan training asli)
