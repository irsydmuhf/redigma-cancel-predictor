"""Load model_bundle.joblib dan jalankan prediksi + penjelasan SHAP untuk satu
input form dari app.py."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.features import build_feature_row

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_bundle.joblib"

FEATURE_LABELS_ID = {
    "qty": "Kuantitas",
    "subtotal_before": "Subtotal sebelum diskon",
    "subtotal_after": "Subtotal setelah diskon",
    "total_discount": "Total diskon",
    "shipping_fee": "Ongkos kirim awal",
    "weight": "Berat (kg)",
    "n_lines": "Jumlah baris/jenis produk",
    "discount_ratio": "Rasio diskon",
    "shipping_ratio": "Rasio ongkos kirim",
    "price_per_item": "Harga rata-rata per item",
    "is_cod": "Bayar di tempat (COD)",
    "hour": "Jam pemesanan",
    "day_of_week": "Hari dalam minggu",
    "is_weekend": "Akhir pekan",
    "is_rush_hour": "Jam sibuk",
    "bundle_size": "Ukuran paket (nama produk)",
    "n_hype_words": "Jumlah kata klaim promosi (nama produk)",
    "has_promo_terms": "Ada istilah promo di nama produk",
    "name_length": "Panjang nama produk",
    "sku_risk": "Riwayat risiko SKU",
    "province_risk": "Riwayat risiko wilayah",
    "payment_risk": "Riwayat risiko metode bayar",
    "category_risk": "Riwayat risiko kategori produk",
    "channel": "Channel pembelian",
    "fulfillment": "Tipe fulfillment",
    "delivery": "Opsi pengiriman",
    "preorder": "Status pre-order",
    "time_category": "Kategori waktu",
    "rush_type": "Tipe jam sibuk",
}


def load_bundle(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model belum ada di {path}. Jalankan `python scripts/train.py` dulu "
            "(lihat README.md bagian 'Melatih model')."
        )
    return joblib.load(path)


def _safe_label_transform(le, values: pd.Series) -> pd.Series:
    """Transform lewat LabelEncoder yang sudah di-fit saat training. Kalau ada
    nilai yang tidak pernah dilihat model (seharusnya tidak terjadi karena form
    dibatasi dropdown), jatuhkan ke kelas pertama supaya tidak crash."""
    known = set(le.classes_)
    safe_values = values.apply(lambda v: v if v in known else le.classes_[0])
    return le.transform(safe_values)


def predict_one(raw: dict, bundle: dict) -> dict:
    row = build_feature_row(raw)

    row_encoded = row.copy()
    for c in bundle["onehot_cols"]:
        row_encoded[c] = _safe_label_transform(bundle["label_encoders"][c], row_encoded[c])

    pipe = bundle["pipeline"]
    proba_cancel = float(pipe.predict_proba(row_encoded)[0, 1])
    threshold = bundle["threshold"]
    predicted_label = 1 if proba_cancel >= threshold else 0

    te_step = pipe.named_steps["te"]
    clf_step = pipe.named_steps["clf"]
    row_numeric = te_step.transform(row_encoded)  # kolom te_cols diganti *_risk, semua numerik

    explainer = shap.TreeExplainer(clf_step)
    explanation = explainer(row_numeric)
    shap_values = np.asarray(explanation.values).reshape(-1)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])

    contrib = (
        pd.DataFrame({
            "fitur": row_numeric.columns,
            "nilai": row_numeric.iloc[0].values,
            "shap": shap_values,
        })
        .assign(label=lambda d: d["fitur"].map(lambda f: FEATURE_LABELS_ID.get(f, f)))
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "proba_cancel": proba_cancel,
        "threshold": threshold,
        "predicted_label": predicted_label,
        "predicted_text": "Berpotensi Dibatalkan" if predicted_label == 1 else "Kemungkinan Selesai",
        "shap_base_value": base_value,
        "shap_contrib": contrib,
        "row_numeric": row_numeric,
    }
