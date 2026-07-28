"""Terjemahkan Seller SKU (kode internal ekspor platform, mis. "VML02") menjadi
nama produk asli (mis. "Vitameal"), dari referensi statis reference/kode_produk.xlsx
(kolom "Kode Finance" -> kode SKU persis, "Kode CRM" -> kode keluarga produk tanpa
angka varian, "Nama" -> nama produk). Referensi ini jarang berubah (katalog
produk), beda dengan data pesanan yang di-upload tiap kali -- jadi disimpan statis
di repo, bukan diupload ulang oleh pengguna.

Sebagian SKU di data pesanan adalah varian dari kode dasar yang tidak persis
terdaftar (mis. VML02/VML03 varian dari VML01/Vitameal) -- kalau exact match
tidak ada, jatuhkan ke prefix 3-huruf awal (Kode CRM) sebagai nama keluarga
produk. Kalau prefix pun tidak dikenali, tampilkan kode SKU aslinya apa adanya
supaya tidak menyembunyikan data yang sebenarnya belum tercatat di katalog."""
from pathlib import Path

import pandas as pd

REF_PATH = Path(__file__).resolve().parent.parent / "reference" / "kode_produk.xlsx"


def load_sku_to_product() -> dict:
    """Kembalikan dict {"exact": {kode_finance: nama}, "prefix": {kode_crm: nama}}.
    Dict kosong kalau file referensi belum ada (fitur nama produk otomatis
    nonaktif, breakdown tetap jalan pakai kode SKU mentah)."""
    if not REF_PATH.exists():
        return {"exact": {}, "prefix": {}}

    kp = pd.read_excel(REF_PATH)
    kp["Kode Finance"] = kp["Kode Finance"].astype(str).str.strip()
    kp["Kode CRM"] = kp["Kode CRM"].astype(str).str.strip()
    kp["Nama"] = kp["Nama"].astype(str).str.strip()

    exact_rows = kp.dropna(subset=["Nama"])[kp["Kode Finance"].notna() & (kp["Kode Finance"] != "nan")]
    exact = dict(zip(exact_rows["Kode Finance"], exact_rows["Nama"]))

    prefix_rows = kp[kp["Kode CRM"].notna() & (kp["Kode CRM"] != "nan")]
    # satu Kode CRM bisa muncul di beberapa baris varian -- ambil nama pertama
    # sebagai representasi nama keluarga produk.
    prefix = prefix_rows.groupby("Kode CRM")["Nama"].first().to_dict()

    return {"exact": exact, "prefix": prefix}


def map_sku_to_product(sku, lookup: dict) -> str:
    if pd.isna(sku):
        return "(SKU kosong)"
    sku = str(sku).strip()
    if sku in lookup.get("exact", {}):
        return lookup["exact"][sku]
    prefix = sku[:3]
    if prefix in lookup.get("prefix", {}):
        return lookup["prefix"][prefix]
    return f"{sku} (belum ada di katalog)"
