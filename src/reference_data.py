"""Opsi dropdown untuk form Streamlit, diambil langsung dari data training yang
tersimpan di model_bundle.joblib -- supaya pilihan yang ditampilkan SELALU sama
dengan kategori yang benar-benar pernah dilihat model (tidak ada opsi
"unseen category" di form)."""


def get_dropdown_options(bundle: dict) -> dict:
    """Mengembalikan dict {kolom: [opsi, ...]} untuk sku/province/payment/category
    (target-encoded) dan channel/delivery (label-encoded). fulfillment & preorder
    sengaja tidak disertakan karena nilainya konstan (lihat src/schema.FIXED_VALUES)."""
    opts = dict(bundle["dropdown_options"])
    opts.pop("fulfillment", None)
    opts.pop("preorder", None)
    opts.pop("time_category", None)  # diturunkan otomatis dari waktu pesanan, bukan input form
    opts.pop("rush_type", None)  # idem

    # Buang placeholder data-hilang ("UNK"/"nan") dari daftar pilihan -- kalaupun
    # muncul di data training, bukan pilihan yang masuk akal ditawarkan ke user.
    for c, values in opts.items():
        opts[c] = [v for v in values if v not in ("UNK", "nan", "")]
    return opts


def format_model_info(bundle: dict) -> str:
    m = bundle["metrics_used_threshold"]
    return (
        f"Model dilatih {bundle['trained_at']} dari {bundle['n_rows_trained']} pesanan. "
        f"Pada threshold {bundle['threshold']}: precision={m['precision']:.3f}, "
        f"recall={m['recall']:.3f}, F1 (kelas dibatalkan)={m['f1_cancel']:.3f}."
    )
