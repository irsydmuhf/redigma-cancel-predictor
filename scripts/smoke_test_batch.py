"""Smoke test manual utk alur upload-spreadsheet (predict_batch). Aman dihapus."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import load_bundle, predict_batch  # noqa: E402
from src.raw_data import load_order_level  # noqa: E402

bundle = load_bundle()

path = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/hp/Downloads/Healthy plus.xlsx"
o = load_order_level(path, require_target=False)
print("order-level shape:", o.shape)
print("kolom Order Status unik:", o["order_status"].value_counts().to_dict())

results = predict_batch(o, bundle)
print("\nresults shape:", results.shape)
print(results.head(10).to_string(index=False))
print("\nJumlah 'Berpotensi Dibatalkan':", (results["Prediksi"] == "Berpotensi Dibatalkan").sum())
print("BATCH SMOKE TEST OK")
