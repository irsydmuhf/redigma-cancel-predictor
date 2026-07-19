"""Load model_bundle.joblib dan jalankan prediksi + penjelasan SHAP, baik untuk
satu input form (predict_one, mode "Input Manual") maupun banyak pesanan
sekaligus dari upload spreadsheet (predict_batch, mode "Upload Spreadsheet")."""
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


def predict_one(raw: dict, bundle: dict, threshold: float = None) -> dict:
    """threshold: kalau None, pakai default hasil training (bundle['threshold']).
    Boleh di-override (mis. dari slider di app.py) tanpa perlu training ulang --
    cuma menentukan batas Berpotensi Dibatalkan / Kemungkinan Selesai, bukan
    mengubah probabilitas mentahnya."""
    if threshold is None:
        threshold = bundle["threshold"]

    row = build_feature_row(raw)

    row_encoded = row.copy()
    for c in bundle["onehot_cols"]:
        row_encoded[c] = _safe_label_transform(bundle["label_encoders"][c], row_encoded[c])

    pipe = bundle["pipeline"]
    proba_cancel = float(pipe.predict_proba(row_encoded)[0, 1])
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


def predict_batch(order_level_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """order_level_df: hasil src.raw_data.load_order_level(file, require_target=False)
    -- satu baris per Order ID, dari upload spreadsheet "Order SKU List" mentah.
    Mengembalikan DataFrame probabilitas + faktor utama utk SEMUA pesanan sekaligus
    (SHAP dihitung satu kali untuk seluruh batch, bukan per-baris, supaya cepat).
    TIDAK menyertakan kolom "Prediksi" (label Berpotensi Dibatalkan/Selesai) --
    itu bergantung threshold yang dipilih user, lihat label_predictions()."""
    num_features = bundle["num_features"]
    te_cols = bundle["te_cols"]
    onehot_cols = bundle["onehot_cols"]
    ordered_cols = num_features + te_cols + onehot_cols

    X = order_level_df[ordered_cols].copy()
    X[num_features] = X[num_features].replace([np.inf, -np.inf], np.nan)
    X[num_features] = X[num_features].apply(lambda s: s.fillna(s.median()))
    for c in te_cols + onehot_cols:
        X[c] = X[c].astype(str).fillna("UNK").replace("nan", "UNK")

    for c in onehot_cols:
        X[c] = _safe_label_transform(bundle["label_encoders"][c], X[c])

    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(X)[:, 1]

    te_step = pipe.named_steps["te"]
    clf_step = pipe.named_steps["clf"]
    X_numeric = te_step.transform(X)

    explainer = shap.TreeExplainer(clf_step)
    explanation = explainer(X_numeric)
    shap_values = np.asarray(explanation.values)  # shape (n_rows, n_kolom_numerik)

    top_factor = []
    for i in range(len(X_numeric)):
        row_shap = shap_values[i]
        top_idx = int(np.argmax(np.abs(row_shap)))
        feat = X_numeric.columns[top_idx]
        label = FEATURE_LABELS_ID.get(feat, feat)
        arah = "meningkatkan" if row_shap[top_idx] > 0 else "menurunkan"
        top_factor.append(f"{label} ({arah} risiko)")

    result = pd.DataFrame({
        "Order ID": order_level_df["Order ID"].values,
        "Status Saat Ini": order_level_df["order_status"].values,
        "Probabilitas Dibatalkan": proba,
        "Faktor Utama": top_factor,
    })
    return result


def label_predictions(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Tambahkan kolom "Prediksi" ke hasil predict_batch() berdasarkan threshold
    pilihan user, lalu urutkan dari probabilitas tertinggi. Murah secara komputasi
    (tidak menghitung ulang SHAP), jadi aman dipanggil tiap kali slider threshold
    digeser di app.py."""
    out = results.copy()
    out["Prediksi"] = np.where(
        out["Probabilitas Dibatalkan"] >= threshold, "Berpotensi Dibatalkan", "Kemungkinan Selesai"
    )
    cols = ["Order ID", "Status Saat Ini", "Probabilitas Dibatalkan", "Prediksi", "Faktor Utama"]
    return out[cols].sort_values("Probabilitas Dibatalkan", ascending=False).reset_index(drop=True)


def nearest_threshold_stats(bundle: dict, threshold: float) -> dict:
    """Cari baris threshold_sweep (dihitung sekali saat training di data uji)
    paling dekat dengan nilai threshold yang dipilih user di slider."""
    sweep = bundle.get("threshold_sweep")
    if not sweep:
        return None
    return min(sweep, key=lambda row: abs(row["threshold"] - threshold))
