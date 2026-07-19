"""Ubah input mentah dari form Streamlit menjadi satu baris fitur siap-prediksi,
mengikuti logika persis load_order_level() + build_xy_v2() di
02_pipeline/redigma_pipeline_v2.py (skripsi asli). Lihat src/schema.py untuk
daftar kolom & konstanta yang dipakai.
"""
import re

import pandas as pd

from src.schema import COD_LABELS, FIXED_VALUES, HYPE_WORDS, NUM_FEATURES, ONEHOT_COLS, PROMO_WORDS, TE_COLS


def extract_marketing_features(product_name: str) -> dict:
    """Persis extract_marketing_features() di pipeline asli, versi satu string
    (bukan pandas Series) untuk satu input form."""
    name = (product_name or "").strip()
    name_upper = name.upper()

    m = re.search(r"PAKET\s*(\d+)", name_upper)
    bundle_size = float(m.group(1)) if m else 1.0

    n_hype_words = float(sum(1 for w in HYPE_WORDS if w in name_upper))
    has_promo_terms = float(any(w in name_upper for w in PROMO_WORDS))
    name_length = float(len(name))

    return {
        "bundle_size": bundle_size,
        "n_hype_words": n_hype_words,
        "has_promo_terms": has_promo_terms,
        "name_length": name_length,
    }


def _time_category(hour: int) -> str:
    # Sama dengan pd.cut(hour, bins=[-1, 4, 11, 16, 19, 23],
    #                     labels=['malam', 'pagi', 'siang', 'sore', 'malam2'])
    # lalu 'malam2' digabung jadi 'malam'.
    if hour <= 4:
        return "malam"
    if hour <= 11:
        return "pagi"
    if hour <= 16:
        return "siang"
    if hour <= 19:
        return "sore"
    return "malam"  # 20-23 ('malam2' pada pipeline asli, digabung ke 'malam')


def _rush_type(hour: int) -> str:
    if 6 <= hour <= 9:
        return "morning_rush"
    if 17 <= hour <= 21:
        return "evening_rush"
    return "non_rush"


def build_feature_row(raw: dict) -> pd.DataFrame:
    """raw: dict hasil isian form Streamlit (lihat app.py untuk daftar key).
    Mengembalikan DataFrame 1 baris dengan kolom NUM_FEATURES + TE_COLS + ONEHOT_COLS,
    SEBELUM label-encoding ONEHOT_COLS (itu dilakukan terpisah oleh src/predict.py
    memakai LabelEncoder yang sama dengan saat training)."""
    created = raw["created"]

    qty = float(raw["qty"]) if raw["qty"] else 0.0
    subtotal_before = float(raw["subtotal_before"])
    subtotal_after = float(raw["subtotal_after"])
    total_discount = float(raw["total_discount"])
    shipping_fee = float(raw["shipping_fee"])
    shipping_after = float(raw["shipping_after"])
    order_amount = float(raw["order_amount"])
    weight = float(raw["weight"])
    n_lines = float(raw["n_lines"]) if raw["n_lines"] else 1.0

    discount_ratio = (total_discount / subtotal_before) if subtotal_before else 0.0
    shipping_ratio = (shipping_after / order_amount) if order_amount else 0.0
    price_per_item = (subtotal_after / qty) if qty else 0.0
    is_cod = 1.0 if str(raw["payment"]).strip().lower() in COD_LABELS else 0.0

    hour = int(created.hour)
    day_of_week = float(created.weekday())  # Senin=0 ... Minggu=6, sama dengan pandas .dt.dayofweek
    is_weekend = 1.0 if created.weekday() >= 5 else 0.0
    time_category = _time_category(hour)
    rush_type = _rush_type(hour)
    is_rush_hour = 1.0 if rush_type != "non_rush" else 0.0

    marketing = extract_marketing_features(raw.get("product_name", ""))

    row = {
        "qty": qty,
        "subtotal_before": subtotal_before,
        "subtotal_after": subtotal_after,
        "total_discount": total_discount,
        "shipping_fee": shipping_fee,
        "weight": weight,
        "n_lines": n_lines,
        "discount_ratio": discount_ratio,
        "shipping_ratio": shipping_ratio,
        "price_per_item": price_per_item,
        "is_cod": is_cod,
        "hour": float(hour),
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        **marketing,
        # TE_COLS (dibiarkan string mentah, ditangani SmoothedTargetEncoder di pipeline)
        "sku": str(raw["sku"]),
        "province": str(raw["province"]),
        "payment": str(raw["payment"]),
        "category": str(raw["category"]),
        # ONEHOT_COLS (masih string di sini, di-label-encode di src/predict.py)
        "channel": str(raw["channel"]),
        "fulfillment": FIXED_VALUES["fulfillment"],
        "delivery": str(raw["delivery"]),
        "preorder": FIXED_VALUES["preorder"],
        "time_category": time_category,
        "rush_type": rush_type,
    }

    ordered_cols = NUM_FEATURES + TE_COLS + ONEHOT_COLS
    return pd.DataFrame([row], columns=ordered_cols)
