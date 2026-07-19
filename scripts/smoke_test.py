"""Smoke test manual: pastikan src.predict.predict_one() jalan end-to-end
(feature engineering -> label encoding -> pipeline -> SHAP) tanpa harus buka
Streamlit UI. Hapus file ini kapan saja, bukan bagian dari aplikasi."""
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import load_bundle, predict_one  # noqa: E402
from src.reference_data import get_dropdown_options  # noqa: E402

bundle = load_bundle()
opts = get_dropdown_options(bundle)
print("dropdown option counts:", {k: len(v) for k, v in opts.items()})

sample = {
    "product_name": "Vitamin C 1000mg Paket 3 Box Original BPOM Garansi",
    "category": opts["category"][0],
    "sku": opts["sku"][0],
    "qty": 2,
    "n_lines": 1,
    "subtotal_before": 150000.0,
    "subtotal_after": 120000.0,
    "total_discount": 30000.0,
    "order_amount": 135000.0,
    "payment": opts["payment"][0],
    "province": opts["province"][0],
    "shipping_fee": 15000.0,
    "shipping_after": 15000.0,
    "weight": 0.5,
    "delivery": opts["delivery"][0],
    "channel": opts["channel"][0],
    "created": datetime.datetime(2026, 4, 18, 20, 30),
}

result = predict_one(sample, bundle)
print("proba_cancel:", result["proba_cancel"])
print("predicted:", result["predicted_text"])
print("top 5 SHAP contrib:")
print(result["shap_contrib"].head(5).to_string(index=False))
print("SMOKE TEST OK")
