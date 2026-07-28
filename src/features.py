"""Ubah input mentah (dipakai scripts/smoke_test.py utk uji satu pesanan) menjadi
satu baris fitur siap-prediksi, mengikuti logika persis load_order_level() +
build_xy() di 02_pipeline/redigma_pipeline_v5_selected.py (skripsi asli) --
16 fitur hasil seleksi statistik, BUKAN 29 fitur rekayasa penuh. Lihat
src/schema.py untuk daftar kolom & konstanta yang dipakai.
"""
import pandas as pd

from src.schema import COD_LABELS, NUM_FEATURES, ONEHOT_COLS, TE_COLS


def build_feature_row(raw: dict) -> pd.DataFrame:
    """raw: dict berisi input mentah satu pesanan (lihat scripts/smoke_test.py
    untuk daftar key). Mengembalikan DataFrame 1 baris dengan kolom NUM_FEATURES +
    TE_COLS + ONEHOT_COLS, SEBELUM label-encoding ONEHOT_COLS (itu dilakukan
    terpisah oleh src/predict.py memakai LabelEncoder yang sama dengan saat
    training)."""
    created = raw["created"]

    qty = float(raw["qty"]) if raw["qty"] else 0.0
    subtotal_before = float(raw["subtotal_before"])
    subtotal_after = float(raw["subtotal_after"])
    total_discount = float(raw["total_discount"])
    shipping_fee = float(raw["shipping_fee"])
    shipping_after = float(raw["shipping_after"])
    order_amount = float(raw["order_amount"])
    weight = float(raw["weight"])

    discount_ratio = (total_discount / subtotal_before) if subtotal_before else 0.0
    shipping_ratio = (shipping_after / order_amount) if order_amount else 0.0
    price_per_item = (subtotal_after / qty) if qty else 0.0
    is_cod = 1.0 if str(raw["payment"]).strip().lower() in COD_LABELS else 0.0
    day_of_week = float(created.weekday())  # Senin=0 ... Minggu=6, sama dengan pandas .dt.dayofweek

    row = {
        "shipping_ratio": shipping_ratio,
        "shipping_fee": shipping_fee,
        "subtotal_after": subtotal_after,
        "is_cod": is_cod,
        "discount_ratio": discount_ratio,
        "weight": weight,
        "price_per_item": price_per_item,
        "subtotal_before": subtotal_before,
        "qty": qty,
        "day_of_week": day_of_week,
        # TE_COLS (dibiarkan string mentah, ditangani SmoothedTargetEncoder di pipeline)
        "sku": str(raw["sku"]),
        "province": str(raw["province"]),
        "payment": str(raw["payment"]),
        "category": str(raw["category"]),
        # ONEHOT_COLS (masih string di sini, di-label-encode di src/predict.py)
        "channel": str(raw["channel"]),
        "delivery": str(raw["delivery"]),
    }

    ordered_cols = NUM_FEATURES + TE_COLS + ONEHOT_COLS
    return pd.DataFrame([row], columns=ordered_cols)
