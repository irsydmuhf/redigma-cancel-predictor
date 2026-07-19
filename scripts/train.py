#!/usr/bin/env python3
"""Latih ulang model prediksi pembatalan pesanan (skenario S3: XGBoost + data
imbalanced) dan simpan artefaknya untuk dipakai app.py.

Logika feature engineering & training di file ini SENGAJA disalin persis dari
02_pipeline/redigma_pipeline_v2.py pada repo skripsi (fungsi load_order_level,
build_xy_v2, make_pipeline) supaya model yang dihasilkan cocok dengan angka yang
dilaporkan di Tabel 4.1 / 4.1b skripsi (Macro-F1 = 0,5512; ROC-AUC = 0,5876).
Kalau butuh menelusuri asalnya, file itu ada di repo skripsi terpisah.

Cara pakai:
    1. Taruh file Excel data pesanan (kolom-kolom sama seperti dataset asli
       REDIGMA) di data/raw_orders.xlsx (folder ini di-.gitignore, TIDAK ikut
       di-commit ke GitHub).
    2. python scripts/train.py
       (atau: python scripts/train.py --input path/lain.xlsx)
    3. Model tersimpan di models/model_bundle.joblib

Kalau nanti data berubah drastis (jumlah SKU/kategori baru banyak, dsb), best
practice-nya adalah menjalankan ulang GridSearchCV (lihat komentar di bagian
bawah file), bukan cuma menjalankan ulang training dengan parameter lama.
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import (  # noqa: E402
    BEST_XGB_PARAMS,
    CANCEL_LABELS,
    CLASSIFICATION_THRESHOLD,
    COD_LABELS,
    DATE_FMTS,
    DONE_LABELS,
    FIXED_VALUES,
    HYPE_WORDS,
    NUM_FEATURES,
    ONEHOT_COLS,
    PROMO_WORDS,
    RANDOM_STATE,
    SMOOTHING,
    TE_COLS,
)
from src.target_encoder import SmoothedTargetEncoder  # noqa: E402

REQUIRED_RAW_COLUMNS = [
    "Order ID", "Order Status", "Created Time", "Shipped Time", "db_pk_pesanan",
    "SKU ID", "Quantity", "SKU Subtotal Before Discount", "SKU Platform Discount",
    "SKU Seller Discount", "SKU Subtotal After Discount", "Shipping Fee After Discount",
    "Original Shipping Fee", "Order Amount", "Weight(kg)", "Payment Method",
    "Product Category", "Province", "Purchase Channel", "Fulfillment Type",
    "Delivery Option", "Normal or Pre-order", "Seller SKU", "Product Name",
]


def log(*a):
    print(*a, flush=True)


def safe_parse_dates(series, fmts):
    def _one(v):
        if not isinstance(v, str) or not v:
            return None
        v = v.strip()
        for fmt in fmts:
            try:
                return datetime.datetime.strptime(v, fmt)
            except ValueError:
                continue
        return None

    parsed = [_one(v) for v in series.tolist()]
    return pd.Series(parsed, index=series.index, dtype="datetime64[ns]")


def mode_first(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iat[0] if not m.empty else s.iloc[0]


def extract_marketing_features(name_series):
    names_upper = name_series.astype(str).str.upper()

    def bundle_size(s):
        m = re.search(r"PAKET\s*(\d+)", s)
        return float(m.group(1)) if m else 1.0

    bsize = names_upper.apply(bundle_size)
    n_hype = names_upper.apply(lambda s: sum(1 for w in HYPE_WORDS if w in s)).astype("float64")
    has_promo = names_upper.apply(lambda s: any(w in s for w in PROMO_WORDS)).astype("float64")
    name_len = name_series.astype(str).str.len().astype("float64")
    return pd.DataFrame(
        {"bundle_size": bsize, "n_hype_words": n_hype, "has_promo_terms": has_promo, "name_length": name_len},
        index=name_series.index,
    )


def load_order_level(input_path: Path) -> pd.DataFrame:
    log(f"Membaca {input_path} ...")
    raw = pd.read_excel(input_path, dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            "Kolom berikut tidak ditemukan di file input, cek nama kolomnya persis "
            f"sama dengan dataset REDIGMA asli: {missing}"
        )

    raw = raw.mask(raw.map(lambda v: isinstance(v, str) and v.strip() == ""), np.nan)

    st = raw.groupby("Order ID")["Order Status"].apply(lambda s: set(x.strip().lower() for x in s.dropna()))

    def lab(stset):
        if stset & CANCEL_LABELS:
            return 1
        if stset and stset <= DONE_LABELS:
            return 0
        return -1

    order_target = st.map(lab)
    keep_ids = order_target[order_target >= 0].index
    raw_f = raw[raw["Order ID"].isin(keep_ids)].copy()

    num_cols = [
        "Quantity", "SKU Subtotal Before Discount", "SKU Platform Discount",
        "SKU Seller Discount", "SKU Subtotal After Discount",
        "Shipping Fee After Discount", "Original Shipping Fee",
        "Order Amount", "Weight(kg)",
    ]
    for c in num_cols:
        raw_f[c] = pd.to_numeric(raw_f[c], errors="coerce")

    o = raw_f.groupby("Order ID").agg(
        qty=("Quantity", "sum"),
        subtotal_before=("SKU Subtotal Before Discount", "sum"),
        plat_disc=("SKU Platform Discount", "sum"),
        seller_disc=("SKU Seller Discount", "sum"),
        subtotal_after=("SKU Subtotal After Discount", "sum"),
        shipping_fee=("Original Shipping Fee", "max"),
        shipping_after=("Shipping Fee After Discount", "max"),
        order_amount=("Order Amount", "max"),
        weight=("Weight(kg)", "sum"),
        n_lines=("SKU ID", "count"),
        payment=("Payment Method", mode_first),
        category=("Product Category", mode_first),
        province=("Province", mode_first),
        channel=("Purchase Channel", mode_first),
        fulfillment=("Fulfillment Type", mode_first),
        delivery=("Delivery Option", mode_first),
        preorder=("Normal or Pre-order", mode_first),
        sku=("Seller SKU", mode_first),
        product_name=("Product Name", mode_first),
        created=("Created Time", "first"),
        shipped=("Shipped Time", "first"),
        pk=("db_pk_pesanan", "first"),
    ).reset_index()
    o["target"] = o["Order ID"].map(order_target)

    o["total_discount"] = o["plat_disc"] + o["seller_disc"]
    o["discount_ratio"] = o["total_discount"] / o["subtotal_before"].replace(0, np.nan)
    o["shipping_ratio"] = o["shipping_after"] / o["order_amount"].replace(0, np.nan)
    o["price_per_item"] = o["subtotal_after"] / o["qty"].replace(0, np.nan)
    o["is_cod"] = o["payment"].astype(str).str.lower().isin(COD_LABELS).astype(int)

    cd = safe_parse_dates(o["created"], DATE_FMTS)
    ts_raw = o["pk"].astype(str).str.split("#").str[1].str.replace("Z", "", regex=False)
    ts = safe_parse_dates(ts_raw, ["%Y/%m/%dT%H:%M:%S"])
    dt = ts.fillna(cd)

    o["hour"] = dt.dt.hour.astype("float64")
    o["day_of_week"] = dt.dt.dayofweek.astype("float64")
    o["is_weekend"] = (dt.dt.dayofweek >= 5).astype("float64")
    o["time_category"] = pd.cut(
        dt.dt.hour, bins=[-1, 4, 11, 16, 19, 23], labels=["malam", "pagi", "siang", "sore", "malam2"]
    ).astype("object")
    o["time_category"] = o["time_category"].replace("malam2", "malam")

    def rush(h):
        if pd.isna(h):
            return np.nan
        h = int(h)
        if 6 <= h <= 9:
            return "morning_rush"
        if 17 <= h <= 21:
            return "evening_rush"
        return "non_rush"

    o["rush_type"] = o["hour"].apply(rush)
    o["is_rush_hour"] = (o["rush_type"] != "non_rush").astype("float64")

    mkt = extract_marketing_features(o["product_name"])
    o = pd.concat([o, mkt], axis=1)
    return o


def build_xy(o: pd.DataFrame):
    data = o[NUM_FEATURES + TE_COLS + ONEHOT_COLS + ["target"]].copy()
    data[NUM_FEATURES] = data[NUM_FEATURES].replace([np.inf, -np.inf], np.nan)
    data[NUM_FEATURES] = data[NUM_FEATURES].apply(lambda s: s.fillna(s.median()))
    for c in TE_COLS + ONEHOT_COLS:
        data[c] = data[c].astype(str).fillna("UNK").replace("nan", "UNK")

    X = data[NUM_FEATURES + TE_COLS + ONEHOT_COLS].copy()
    y = data["target"].astype(int).values

    label_encoders = {}
    for c in ONEHOT_COLS:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        label_encoders[c] = le

    return X, y, label_encoders


def make_pipeline():
    te = SmoothedTargetEncoder(cols=TE_COLS, smoothing=SMOOTHING, drop_original=True)
    clf = XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, **BEST_XGB_PARAMS
    )
    return ImbPipeline([("te", te), ("clf", clf)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data" / "raw_orders.xlsx"))
    parser.add_argument("--output", default=str(ROOT / "models" / "model_bundle.joblib"))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(
            f"File input tidak ditemukan: {input_path}\n"
            "Taruh file Excel data pesanan (kolom sama seperti dataset REDIGMA asli) "
            "di path ini, atau tunjuk lokasinya lewat --input path/ke/file.xlsx"
        )

    o = load_order_level(input_path)
    log("shape order-level:", o.shape)

    X, y, label_encoders = build_xy(o)
    log("X shape:", X.shape, "| num:", len(NUM_FEATURES), "| target-encode:", TE_COLS, "| label-encode:", ONEHOT_COLS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log("train:", X_train.shape, "test:", X_test.shape)

    log()
    log("=" * 70)
    log("TRAINING XGBoost skenario S3 (imbalanced, parameter dari GridSearchCV Tabel 4.1b)")
    log("=" * 70)
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    yp = pipe.predict(X_test)
    pp = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    metrics_default_threshold = {
        "accuracy": accuracy_score(y_test, yp),
        "precision": precision_score(y_test, yp, zero_division=0),
        "recall": recall_score(y_test, yp, zero_division=0),
        "macro_f1": f1_score(y_test, yp, average="macro"),
        "roc_auc": roc_auc_score(y_test, pp),
        "g_mean": float(np.sqrt(sens * spec)),
    }
    log("--- Metrik pada threshold default (0,5) ---")
    for k, v in metrics_default_threshold.items():
        log(f"  {k:12s}: {v:.4f}")
    log()
    log("Referensi dari skripsi (Tabel 4.1, S3: XGBoost + Imbalanced):")
    log("  accuracy=0.8448 precision=0.3333 recall=0.1311 macro_f1=0.5512 roc_auc=0.5876")
    log("Kalau angka di atas jauh berbeda, kemungkinan data input berbeda dari data asli skripsi.")

    yp_thr = (pp >= CLASSIFICATION_THRESHOLD).astype(int)
    metrics_used_threshold = {
        "threshold": CLASSIFICATION_THRESHOLD,
        "precision": precision_score(y_test, yp_thr, zero_division=0),
        "recall": recall_score(y_test, yp_thr, zero_division=0),
        "f1_cancel": f1_score(y_test, yp_thr, pos_label=1, zero_division=0),
    }
    log()
    log(f"--- Metrik pada threshold={CLASSIFICATION_THRESHOLD} (dipakai app.py, sesuai Sub-bab 4.2.3) ---")
    for k, v in metrics_used_threshold.items():
        log(f"  {k:12s}: {v}")

    # Referensi dropdown untuk form Streamlit: kategori yang benar-benar dilihat model saat training.
    dropdown_options = {c: sorted(o[c].dropna().astype(str).unique().tolist()) for c in TE_COLS}
    dropdown_options.update({c: list(le.classes_) for c, le in label_encoders.items() if c not in FIXED_VALUES})

    bundle = {
        "pipeline": pipe,
        "label_encoders": label_encoders,
        "num_features": NUM_FEATURES,
        "te_cols": TE_COLS,
        "onehot_cols": ONEHOT_COLS,
        "threshold": CLASSIFICATION_THRESHOLD,
        "dropdown_options": dropdown_options,
        "metrics_default_threshold": metrics_default_threshold,
        "metrics_used_threshold": metrics_used_threshold,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_rows_trained": int(len(o)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    log()
    log(f"Model tersimpan: {output_path}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Catatan retrain dengan GridSearchCV ulang (opsional, kalau data berubah banyak):
#
#   from sklearn.model_selection import GridSearchCV
#   param_grid = {
#       "clf__n_estimators": [100, 200, 300],
#       "clf__max_depth": [3, 5, 7],
#       "clf__learning_rate": [0.01, 0.1, 0.3],
#       "clf__subsample": [0.7, 0.8, 1.0],
#   }
#   gs = GridSearchCV(make_pipeline(), param_grid, scoring="f1_macro", cv=5, n_jobs=-1)
#   gs.fit(X_train, y_train)
#   print(gs.best_params_)
#
# lalu update BEST_XGB_PARAMS di src/schema.py dengan hasil gs.best_params_.
# ---------------------------------------------------------------------------
