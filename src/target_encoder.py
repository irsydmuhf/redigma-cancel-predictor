"""SmoothedTargetEncoder -- disalin persis dari 02_pipeline/redigma_pipeline_v2.py
(skripsi asli) supaya perilaku encoding provinsi/pembayaran/kategori/SKU identik
dengan model yang dilaporkan di Tabel 4.1.

Ditaruh di modul sendiri (bukan langsung di scripts/train.py) karena joblib perlu
bisa mengimpor kelas ini persis dari path yang sama saat model di-load lagi oleh
src/predict.py.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Target-encoding per kolom kategorikal dengan smoothing ke rata-rata global.
    Fit HANYA pakai data yang diberikan saat .fit() -- aman dari data leakage."""

    def __init__(self, cols, smoothing=20, drop_original=True):
        self.cols = cols
        self.smoothing = smoothing
        self.drop_original = drop_original

    def fit(self, X, y):
        y = np.asarray(y)
        self.global_mean_ = float(np.mean(y))
        self.maps_ = {}
        for c in self.cols:
            tmp = pd.DataFrame({"cat": X[c].values, "y": y})
            stats = tmp.groupby("cat")["y"].agg(["mean", "count"])
            smoothed = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (
                stats["count"] + self.smoothing
            )
            self.maps_[c] = smoothed.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c + "_risk"] = X[c].map(self.maps_[c]).astype("float64").fillna(self.global_mean_)
        if self.drop_original:
            X = X.drop(columns=self.cols)
        return X
