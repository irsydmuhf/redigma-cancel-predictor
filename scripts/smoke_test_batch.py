"""Smoke test manual utk alur upload-spreadsheet (predict_batch + label_predictions
+ nearest_threshold_stats). Aman dihapus."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import label_predictions, load_bundle, nearest_threshold_stats, predict_batch  # noqa: E402
from src.raw_data import load_order_level  # noqa: E402

bundle = load_bundle()

path = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/hp/Downloads/Healthy plus.xlsx"
o = load_order_level(path, require_target=False)
print("order-level shape:", o.shape)
print("kolom Order Status unik:", o["order_status"].value_counts().to_dict())

results_raw = predict_batch(o, bundle)
print("\nresults_raw columns:", list(results_raw.columns))

for thr in [0.2, 0.5, 0.8]:
    labeled = label_predictions(results_raw, thr)
    n_risk = int((labeled["Prediksi"] == "Berpotensi Dibatalkan").sum())
    stats = nearest_threshold_stats(bundle, thr)
    print(f"\nthreshold={thr} -> {n_risk}/{len(labeled)} ter-flag | stats historis: {stats}")

print("\n", label_predictions(results_raw, 0.2).head(5).to_string(index=False))
print("\nBATCH SMOKE TEST OK")
