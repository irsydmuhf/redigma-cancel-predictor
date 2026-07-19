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

from src.raw_data import load_order_level  # noqa: E402
from src.schema import (  # noqa: E402
    BEST_XGB_PARAMS,
    CLASSIFICATION_THRESHOLD,
    FIXED_VALUES,
    NUM_FEATURES,
    ONEHOT_COLS,
    RANDOM_STATE,
    SMOOTHING,
    TE_COLS,
)
from src.target_encoder import SmoothedTargetEncoder  # noqa: E402


def log(*a):
    print(*a, flush=True)


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

    log(f"Membaca {input_path} ...")
    o = load_order_level(input_path, require_target=True)
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

    # Sapuan precision/recall/F1 di berbagai threshold (0,05-0,95) pada data uji --
    # dipakai app.py utk slider threshold interaktif (biar precision/recall trade-off
    # kelihatan, bukan cuma satu angka fixed). Sama semangatnya dengan Tabel 4.2 skripsi.
    threshold_sweep = []
    for t in np.arange(0.05, 0.96, 0.05):
        t = round(float(t), 2)
        yp_t = (pp >= t).astype(int)
        threshold_sweep.append({
            "threshold": t,
            "precision": precision_score(y_test, yp_t, zero_division=0),
            "recall": recall_score(y_test, yp_t, zero_division=0),
            "f1_cancel": f1_score(y_test, yp_t, pos_label=1, zero_division=0),
            "n_flagged": int(yp_t.sum()),
            "n_flagged_pct": float(yp_t.mean()),
        })
    log()
    log("--- Sapuan threshold (dipakai untuk slider di app.py) ---")
    for row in threshold_sweep:
        log(
            f"  thr={row['threshold']:.2f} | precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1_cancel={row['f1_cancel']:.3f} "
            f"flagged={row['n_flagged_pct']:.1%}"
        )

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
        "threshold_sweep": threshold_sweep,
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
