"""Agregasi data pesanan mentah (level baris-SKU, format ekspor "Order SKU List"
platform) menjadi level pesanan (satu baris per Order ID) siap dipakai untuk
training (scripts/train.py) maupun prediksi batch (app.py, mode upload
spreadsheet). Logika ini disalin dari load_order_level() di
02_pipeline/redigma_pipeline_v2.py (skripsi asli).

db_pk_pesanan itu kolom internal yang cuma ada di 04_data/test.xlsx - DATABASE.xlsx
(hasil olahan untuk skripsi) -- TIDAK ada di file ekspor "Order SKU List" asli dari
platform. Fungsi di sini memperlakukannya sebagai opsional: kalau ada, dipakai
untuk mempertajam jam pemesanan (fallback historis dari pipeline asli); kalau
tidak ada (kasus umum untuk upload spreadsheet asli), langsung pakai kolom
Created Time saja.
"""
import re

import numpy as np
import pandas as pd

from src.schema import CANCEL_LABELS, COD_LABELS, DATE_FMTS, DONE_LABELS, HYPE_WORDS, PROMO_WORDS

REQUIRED_RAW_COLUMNS = [
    "Order ID", "Order Status", "Created Time", "Shipped Time",
    "SKU ID", "Quantity", "SKU Subtotal Before Discount", "SKU Platform Discount",
    "SKU Seller Discount", "SKU Subtotal After Discount", "Shipping Fee After Discount",
    "Original Shipping Fee", "Order Amount", "Weight(kg)", "Payment Method",
    "Product Category", "Province", "Purchase Channel", "Fulfillment Type",
    "Delivery Option", "Normal or Pre-order", "Seller SKU", "Product Name",
]
OPTIONAL_RAW_COLUMNS = ["db_pk_pesanan", "Nama Toko"]


def log(*a):
    print(*a, flush=True)


def safe_parse_dates(series, fmts):
    def _one(v):
        if not isinstance(v, str) or not v:
            return None
        v = v.strip()
        for fmt in fmts:
            try:
                return __import__("datetime").datetime.strptime(v, fmt)
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


def _time_category_series(hour_series: pd.Series) -> pd.Series:
    cat = pd.cut(
        hour_series, bins=[-1, 4, 11, 16, 19, 23], labels=["malam", "pagi", "siang", "sore", "malam2"]
    ).astype("object")
    return cat.replace("malam2", "malam")


def _rush_type(h):
    if pd.isna(h):
        return np.nan
    h = int(h)
    if 6 <= h <= 9:
        return "morning_rush"
    if 17 <= h <= 21:
        return "evening_rush"
    return "non_rush"


def load_order_level(input_path_or_buffer, require_target: bool = True) -> pd.DataFrame:
    """require_target=True (dipakai scripts/train.py): hanya pesanan dengan status
    yang sudah selesai/dibatalkan (perlu label untuk training).
    require_target=False (dipakai mode upload-spreadsheet app.py): SEMUA pesanan
    ikut diagregasi, termasuk yang masih berjalan (justru ini yang mau diprediksi)."""
    # engine_kwargs read_only=False: sebagian ekspor "Order SKU List" platform (mis. TikTok
    # Shop Seller Center) menyimpan metadata <dimension> yang salah/tidak lengkap di file
    # xlsx-nya. Microsoft Excel mengabaikan metadata itu dan tetap membaca semua kolom, tapi
    # openpyxl dalam mode read_only (default reader pandas) mempercayainya mentah-mentah dan
    # diam-diam memotong sampai kolom A saja. read_only=False memaksa parse penuh yang benar.
    raw = pd.read_excel(
        input_path_or_buffer, dtype=str, engine="openpyxl",
        engine_kwargs={"read_only": False, "data_only": True},
    )
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            "Kolom berikut tidak ditemukan di file input, cek nama kolomnya persis "
            f"sama dengan ekspor 'Order SKU List' platform: {missing}"
        )
    has_pk = "db_pk_pesanan" in raw.columns
    has_store = "Nama Toko" in raw.columns

    # Sebagian ekspor "Order SKU List" menyisipkan satu baris deskripsi kolom tepat di
    # bawah header (mis. "Platform unique order ID.", "Current order status.") -- bukan
    # baris data. Order ID asli selalu berupa angka murni, jadi baris non-angka dibuang.
    raw = raw[raw["Order ID"].astype(str).str.strip().str.fullmatch(r"\d+").fillna(False)].copy()

    raw = raw.mask(raw.map(lambda v: isinstance(v, str) and v.strip() == ""), np.nan)

    st = raw.groupby("Order ID")["Order Status"].apply(lambda s: set(x.strip().lower() for x in s.dropna()))

    def lab(stset):
        if stset & CANCEL_LABELS:
            return 1
        if stset and stset <= DONE_LABELS:
            return 0
        return -1

    order_target = st.map(lab)
    if require_target:
        keep_ids = order_target[order_target >= 0].index
        raw_f = raw[raw["Order ID"].isin(keep_ids)].copy()
    else:
        raw_f = raw.copy()

    num_cols = [
        "Quantity", "SKU Subtotal Before Discount", "SKU Platform Discount",
        "SKU Seller Discount", "SKU Subtotal After Discount",
        "Shipping Fee After Discount", "Original Shipping Fee",
        "Order Amount", "Weight(kg)",
    ]
    for c in num_cols:
        raw_f[c] = pd.to_numeric(raw_f[c], errors="coerce")

    agg_kwargs = dict(
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
        order_status=("Order Status", mode_first),
    )
    if has_pk:
        agg_kwargs["pk"] = ("db_pk_pesanan", "first")
    if has_store:
        agg_kwargs["store"] = ("Nama Toko", mode_first)

    o = raw_f.groupby("Order ID").agg(**agg_kwargs).reset_index()
    if not has_store:
        o["store"] = "Tidak diketahui"
    o["target"] = o["Order ID"].map(order_target)

    o["total_discount"] = o["plat_disc"] + o["seller_disc"]
    o["discount_ratio"] = o["total_discount"] / o["subtotal_before"].replace(0, np.nan)
    o["shipping_ratio"] = o["shipping_after"] / o["order_amount"].replace(0, np.nan)
    o["price_per_item"] = o["subtotal_after"] / o["qty"].replace(0, np.nan)
    o["is_cod"] = o["payment"].astype(str).str.lower().isin(COD_LABELS).astype(int)

    cd = safe_parse_dates(o["created"], DATE_FMTS)
    if has_pk:
        ts_raw = o["pk"].astype(str).str.split("#").str[1].str.replace("Z", "", regex=False)
        ts = safe_parse_dates(ts_raw, ["%Y/%m/%dT%H:%M:%S"])
        dt = ts.fillna(cd)
    else:
        dt = cd

    o["hour"] = dt.dt.hour.astype("float64")
    o["day_of_week"] = dt.dt.dayofweek.astype("float64")
    o["is_weekend"] = (dt.dt.dayofweek >= 5).astype("float64")
    o["time_category"] = _time_category_series(dt.dt.hour)
    o["rush_type"] = o["hour"].apply(_rush_type)
    o["is_rush_hour"] = (o["rush_type"] != "non_rush").astype("float64")

    mkt = extract_marketing_features(o["product_name"])
    o = pd.concat([o, mkt], axis=1)
    return o
