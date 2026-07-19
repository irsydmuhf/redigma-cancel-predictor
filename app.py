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

from src.predict import label_predictions, load_bundle, nearest_threshold_stats, predict_batch, predict_one
from src.raw_data import load_order_level
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

# =============================================================================
# Sidebar: threshold klasifikasi -- dibagi ke kedua tab, supaya trade-off
# precision/recall langsung kelihatan & bisa diubah tanpa mengubah kode.
# =============================================================================
with st.sidebar:
    st.header("⚙️ Threshold Klasifikasi")
    threshold = st.slider(
        "Ambang batas probabilitas",
        min_value=0.05, max_value=0.95, value=float(bundle["threshold"]), step=0.05,
        help="Di atas nilai ini, pesanan diberi label 'Berpotensi Dibatalkan'.",
    )
    stats = nearest_threshold_stats(bundle, threshold)
    if stats:
        st.caption(f"Performa historis pada data uji (threshold≈{stats['threshold']:.2f}):")
        c1, c2 = st.columns(2)
        c1.metric("Precision", f"{stats['precision']:.0%}")
        c2.metric("Recall", f"{stats['recall']:.0%}")
        st.caption(f"F1 (kelas dibatalkan): {stats['f1_cancel']:.3f}")
        st.caption(
            f"≈{stats['n_flagged_pct']:.0%} dari seluruh pesanan di data uji akan ter-flag pada threshold ini."
        )
    st.divider()
    st.caption(
        f"Default skripsi: **{bundle['threshold']}** (F1 kelas 'dibatalkan' tertinggi, Sub-bab 4.2.3). "
        "Naikkan threshold untuk lebih sedikit *false alarm* (presisi naik, recall turun); turunkan "
        "untuk lebih jarang kelewatan pesanan yang benar-benar akan dibatalkan (recall naik, presisi turun)."
    )

st.title("📦 Prediksi Potensi Pembatalan Pesanan")
st.caption(
    "Demo interaktif dari skripsi klasifikasi pembatalan pesanan e-commerce REDIGMA "
    "(model: XGBoost skenario S3, lihat Bab 4)."
)
with st.expander("ℹ️ Tentang model & keterbatasannya", expanded=False):
    st.markdown(
        f"""
- Model: **XGBoost** (skenario S3 pada Bab 4 skripsi -- Macro F1 tertinggi di antara 4 skenario yang dibandingkan).
- Threshold klasifikasi bisa diatur di sidebar kiri (default {bundle['threshold']}, sesuai Sub-bab 4.2.3 -- 0,5 terbukti BUKAN titik operasi optimal untuk kasus ini).
- {format_model_info(bundle)}
- Fitur risiko per wilayah/metode bayar/kategori/SKU dihitung dari riwayat data training (*smoothed target encoding*) -- kombinasi yang sangat jarang muncul di data training akan condong ke rata-rata global, bukan kesalahan sistem.
- Performa model tergolong **moderat** (ROC-AUC 0,59 pada data uji). Pada threshold rendah, WAJAR banyak pesanan yang sebenarnya akan selesai normal ikut ter-flag (precision rendah) -- itu trade-off yang disengaja demi menangkap lebih banyak pesanan yang benar-benar akan dibatalkan (recall tinggi). Lihat angka precision/recall di sidebar untuk threshold yang sedang dipilih. Hasil dari tool ini sebaiknya jadi salah satu sinyal pendukung, bukan keputusan tunggal.
        """
    )

st.divider()

tab_batch, tab_manual = st.tabs(["📂 Upload Spreadsheet (banyak pesanan)", "✍️ Input Manual (satu pesanan)"])

# =============================================================================
# TAB 1: Upload spreadsheet -- alur utama untuk pemakaian sehari-hari
# =============================================================================
with tab_batch:
    st.markdown(
        "Upload file ekspor **\"Order SKU List\"** langsung dari platform (format xlsx apa adanya, "
        "tidak perlu diedit dulu). Cocok untuk pesanan yang **masih berjalan** (belum selesai/dibatalkan) "
        "-- app akan mengelompokkan baris-baris SKU per Order ID lalu memprediksi tiap pesanan."
    )
    uploaded = st.file_uploader("File Order SKU List (.xlsx)", type=["xlsx"])

    if uploaded is not None:
        file_key = (uploaded.name, uploaded.size)
        if st.session_state.get("batch_file_key") != file_key:
            try:
                with st.spinner("Membaca & mengagregasi data pesanan..."):
                    order_level = load_order_level(uploaded, require_target=False)
            except ValueError as e:
                st.error(str(e))
                order_level = None
            except Exception as e:  # format file tidak terbaca sama sekali, dsb.
                st.error(f"Gagal membaca file: {e}")
                order_level = None

            if order_level is not None:
                with st.spinner(f"Memprediksi {len(order_level)} pesanan..."):
                    st.session_state["batch_results_raw"] = predict_batch(order_level, bundle)
                st.session_state["batch_n_orders"] = len(order_level)
                st.session_state["batch_file_key"] = file_key
            else:
                st.session_state.pop("batch_results_raw", None)

        if "batch_results_raw" in st.session_state:
            st.success(f"Berhasil membaca {st.session_state['batch_n_orders']} pesanan unik dari file.")
            results = label_predictions(st.session_state["batch_results_raw"], threshold)

            n_total = len(results)
            n_risk = int((results["Prediksi"] == "Berpotensi Dibatalkan").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Total pesanan", n_total)
            m2.metric("Berpotensi dibatalkan", n_risk)
            m3.metric("Persentase", f"{(n_risk / n_total * 100 if n_total else 0):.1f}%")
            st.caption(
                f"Dihitung dengan threshold {threshold:.2f} (atur di sidebar kiri) -- "
                "geser slider untuk lihat perubahan jumlah yang ter-flag tanpa upload ulang."
            )

            st.dataframe(
                results.style.format({"Probabilitas Dibatalkan": "{:.1%}"}),
                use_container_width=True,
                height=420,
            )

            csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download hasil (CSV)",
                data=csv_bytes,
                file_name=f"prediksi_pembatalan_{datetime.date.today().isoformat()}.csv",
                mime="text/csv",
            )

# =============================================================================
# TAB 2: Input manual -- untuk uji satu skenario "what-if" / demo cepat
# =============================================================================
with tab_manual:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Produk")
            product_name = st.text_input(
                "Nama produk", placeholder="mis. Vitamin C 1000mg Paket 3 Box Original BPOM"
            )
            category = st.selectbox("Kategori produk", options["category"])
            sku = st.selectbox("Seller SKU", options["sku"])
            qty = st.number_input("Kuantitas", min_value=1, value=1, step=1)
            n_lines = st.number_input("Jumlah baris/jenis produk dalam pesanan", min_value=1, value=1, step=1)

        with col2:
            st.subheader("Transaksi & Diskon")
            subtotal_before = st.number_input(
                "Subtotal sebelum diskon (Rp)", min_value=0.0, value=100000.0, step=1000.0
            )
            total_discount = st.number_input(
                "Total diskon, platform + penjual (Rp)", min_value=0.0, value=0.0, step=1000.0,
                help="Dijumlah otomatis ke subtotal setelah diskon di bawah.",
            )
            subtotal_after = max(subtotal_before - total_discount, 0.0)
            st.metric("Subtotal setelah diskon (otomatis)", f"Rp {subtotal_after:,.0f}")
            order_amount = st.number_input(
                "Total nilai pesanan / Order Amount (Rp)", min_value=0.0, value=100000.0, step=1000.0
            )
            payment = st.selectbox("Metode pembayaran", options["payment"])

        with col3:
            st.subheader("Pengiriman & Waktu")
            province = st.selectbox("Wilayah pengiriman (provinsi)", options["province"])
            shipping_fee = st.number_input("Ongkos kirim awal (Rp)", min_value=0.0, value=15000.0, step=1000.0)
            shipping_after = st.number_input(
                "Ongkos kirim setelah diskon (Rp)", min_value=0.0, value=15000.0, step=1000.0
            )
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

        result = predict_one(raw, bundle, threshold=threshold)

        st.divider()
        st.subheader("Hasil Prediksi")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if result["predicted_label"] == 1:
                st.error(f"⚠️ {result['predicted_text']}")
            else:
                st.success(f"✅ {result['predicted_text']}")
            st.metric("Probabilitas dibatalkan", f"{result['proba_cancel']:.1%}")
            st.caption(f"Threshold yang dipakai: {result['threshold']:.0%} (atur di sidebar kiri)")

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
