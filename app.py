"""Demo interaktif: prediksi potensi pembatalan pesanan REDIGMA (PT Relasi
Digital Marketing) memakai model XGBoost skenario S3 dari skripsi "Klasifikasi
Pembatalan Pesanan E-Commerce ... dan Analisis Faktor" (Irsyad Muhamad Firdaus).

Jalankan: streamlit run app.py
(model harus sudah dilatih lebih dulu lewat scripts/train.py -- lihat README.md)
"""
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.predict import load_bundle, predict_one
from src.reference_data import format_model_info, get_dropdown_options

st.set_page_config(page_title="Prediksi Pembatalan Pesanan REDIGMA", page_icon="📦", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

options = get_dropdown_options(bundle)

st.title("📦 Prediksi Potensi Pembatalan Pesanan")
st.caption(
    "Demo interaktif dari skripsi klasifikasi pembatalan pesanan e-commerce REDIGMA "
    "(model: XGBoost skenario S3, lihat Bab 4)."
)
with st.expander("ℹ️ Tentang model & keterbatasannya", expanded=False):
    st.markdown(
        f"""
- Model: **XGBoost** (skenario S3 pada Bab 4 skripsi -- Macro F1 tertinggi di antara 4 skenario yang dibandingkan).
- Threshold klasifikasi: **{bundle['threshold']}** (bukan 0,5 default -- sesuai temuan Sub-bab 4.2.3 bahwa 0,5 bukan titik operasi optimal untuk kelas minoritas "dibatalkan").
- {format_model_info(bundle)}
- Fitur risiko per wilayah/metode bayar/kategori/SKU dihitung dari riwayat data training (*smoothed target encoding*) -- kombinasi yang sangat jarang muncul di data training akan condong ke rata-rata global, bukan kesalahan sistem.
- Performa model tergolong **moderat** (ROC-AUC 0,59 pada data uji) -- hasil dari tool ini sebaiknya dipakai sebagai salah satu sinyal pendukung, bukan keputusan tunggal.
        """
    )

st.divider()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Produk")
        product_name = st.text_input("Nama produk", placeholder="mis. Vitamin C 1000mg Paket 3 Box Original BPOM")
        category = st.selectbox("Kategori produk", options["category"])
        sku = st.selectbox("Seller SKU", options["sku"])
        qty = st.number_input("Kuantitas", min_value=1, value=1, step=1)
        n_lines = st.number_input("Jumlah baris/jenis produk dalam pesanan", min_value=1, value=1, step=1)

    with col2:
        st.subheader("Transaksi & Diskon")
        subtotal_before = st.number_input("Subtotal sebelum diskon (Rp)", min_value=0.0, value=100000.0, step=1000.0)
        total_discount = st.number_input("Total diskon, platform + penjual (Rp)", min_value=0.0, value=0.0, step=1000.0)
        subtotal_after = st.number_input("Subtotal setelah diskon (Rp)", min_value=0.0, value=100000.0, step=1000.0)
        order_amount = st.number_input("Total nilai pesanan / Order Amount (Rp)", min_value=0.0, value=100000.0, step=1000.0)
        payment = st.selectbox("Metode pembayaran", options["payment"])

    with col3:
        st.subheader("Pengiriman & Waktu")
        province = st.selectbox("Wilayah pengiriman (provinsi)", options["province"])
        shipping_fee = st.number_input("Ongkos kirim awal (Rp)", min_value=0.0, value=15000.0, step=1000.0)
        shipping_after = st.number_input("Ongkos kirim setelah diskon (Rp)", min_value=0.0, value=15000.0, step=1000.0)
        weight = st.number_input("Berat (kg)", min_value=0.0, value=0.5, step=0.1)
        delivery = st.selectbox("Opsi pengiriman / kurir", options["delivery"])
        channel = st.selectbox("Channel pembelian", options["channel"])
        created_date = st.date_input("Tanggal pesanan dibuat", value=datetime.date.today())
        created_time = st.time_input("Jam pesanan dibuat", value=datetime.time(12, 0))

    submitted = st.form_submit_button("Prediksi", use_container_width=True, type="primary")

if submitted:
    raw = {
        "product_name": product_name,
        "category": category,
        "sku": sku,
        "qty": qty,
        "n_lines": n_lines,
        "subtotal_before": subtotal_before,
        "subtotal_after": subtotal_after,
        "total_discount": total_discount,
        "order_amount": order_amount,
        "payment": payment,
        "province": province,
        "shipping_fee": shipping_fee,
        "shipping_after": shipping_after,
        "weight": weight,
        "delivery": delivery,
        "channel": channel,
        "created": datetime.datetime.combine(created_date, created_time),
    }

    result = predict_one(raw, bundle)

    st.divider()
    st.subheader("Hasil Prediksi")

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        if result["predicted_label"] == 1:
            st.error(f"⚠️ {result['predicted_text']}")
        else:
            st.success(f"✅ {result['predicted_text']}")
        st.metric("Probabilitas dibatalkan", f"{result['proba_cancel']:.1%}")
        st.caption(f"Threshold yang dipakai: {result['threshold']:.0%}")

    with res_col2:
        st.markdown("**Kontribusi fitur terhadap prediksi ini (SHAP)**")
        top = result["shap_contrib"].head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["shap"]]
        ax.barh(top["label"], top["shap"], color=colors)
        ax.set_xlabel("Kontribusi SHAP (log-odds, + = mendorong ke arah dibatalkan)")
        ax.axvline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Merah = mendorong prediksi ke arah 'dibatalkan'. Biru = mendorong ke arah 'selesai'. "
            "Panjang batang = seberapa besar pengaruh fitur tersebut pada prediksi input ini secara spesifik."
        )

st.divider()
st.caption(
    "Dibangun dari model & data skripsi \"Klasifikasi Pembatalan Pesanan E-Commerce Menggunakan "
    "Logistic Regression dan XGBoost, dan Analisis Faktor\" -- Irsyad Muhamad Firdaus, "
    "Teknik Informatika, Universitas Muhammadiyah Magelang."
)
