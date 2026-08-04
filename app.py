"""Implementasi sistem klasifikasi pembatalan pesanan REDIGMA (PT Relasi Digital
Marketing), berdasarkan skripsi "Klasifikasi Pembatalan Pesanan E-Commerce
Menggunakan Machine Learning dan Analisis Faktor" (Irsyad Muhamad Firdaus).
Aplikasi ini menjawab ketiga rumusan masalah skripsi secara langsung:
  RM1 -- model klasifikasi (XGBoost, skenario S3) dipakai untuk mengklasifikasi
         pesanan lewat probabilitas & label prediksi.
  RM2 -- perbandingan performa Logistic Regression vs XGBoost (Tabel 4.1)
         ditampilkan sebagai referensi hasil evaluasi model.
  RM3 -- faktor dominan penyebab pembatalan direkap lewat interpretasi SHAP,
         baik untuk pesanan yang statusnya sudah diketahui batal (rekap faktor)
         maupun sebagai bagian dari penjelasan tiap hasil klasifikasi.

Jalankan: streamlit run app.py
(model harus sudah dilatih lebih dulu lewat scripts/train.py -- lihat README.md)
"""
import pandas as pd
import plotly.express as px
import streamlit as st

REDIGMA_RED = "#d62728"

from src.predict import diagnose_batch, label_predictions, load_bundle, predict_batch
from src.product_lookup import load_sku_to_product, map_sku_to_product
from src.raw_data import load_order_level
from src.reference_data import format_model_info

# Tabel 4.1 skripsi -- Perbandingan Performa Lima Skenario pada Data Uji.
# Statis (bukan dihitung ulang saat runtime) karena LR tidak ikut dideploy;
# tabel ini murni referensi RM2 (perbandingan algoritma), sedangkan model yang
# benar-benar dipakai aplikasi untuk RM1 adalah XGBoost skenario S3.
SCENARIO_COMPARISON = [
    {"Skenario": "S1: LR + Imbalanced", "Accuracy": 0.8651, "Precision": 0.6500, "Recall": 0.0355, "Macro F1": 0.4973, "ROC-AUC": 0.6439},
    {"Skenario": "S2: LR + SMOTE-NC", "Accuracy": 0.6855, "Precision": 0.2091, "Recall": 0.4645, "Macro F1": 0.5433, "ROC-AUC": 0.6344},
    {"Skenario": "S3: XGBoost + Imbalanced (dipakai aplikasi ini)", "Accuracy": 0.8430, "Precision": 0.2773, "Recall": 0.0902, "Macro F1": 0.5249, "ROC-AUC": 0.5875},
    {"Skenario": "S4: XGBoost + SMOTE-NC", "Accuracy": 0.8291, "Precision": 0.2443, "Recall": 0.1175, "Macro F1": 0.5318, "ROC-AUC": 0.5778},
    {"Skenario": "S5: XGBoost + 2 Fitur Teratas", "Accuracy": 0.8576, "Precision": 0.3333, "Recall": 0.0383, "Macro F1": 0.4958, "ROC-AUC": 0.5915},
]


@st.cache_resource
def get_sku_lookup():
    return load_sku_to_product()

st.set_page_config(page_title="Rekap Faktor Pembatalan Pesanan REDIGMA", page_icon="📦", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.title("📦 Klasifikasi & Analisis Faktor Pembatalan Pesanan")
st.caption(
    "Implementasi sistem dari skripsi klasifikasi pembatalan pesanan e-commerce REDIGMA -- "
    "menampilkan hasil klasifikasi model (RM1), perbandingan performa algoritma (RM2), "
    "dan rekap faktor dominan penyebab pembatalan lewat SHAP (RM3)."
)
with st.expander("ℹ️ Tentang model & keterbatasannya", expanded=False):
    st.markdown(
        f"""
- Model yang dipakai aplikasi ini: **XGBoost** (skenario S3 pada Bab 4 skripsi).
- {format_model_info(bundle)}
- Fitur risiko per wilayah/metode bayar/kategori/SKU dihitung dari riwayat data training (*smoothed target encoding*) -- kombinasi yang sangat jarang muncul di data training akan condong ke rata-rata global, bukan kesalahan sistem.
- **Precision model tergolong rendah** (lihat tabel perbandingan di bawah) -- dari seluruh pesanan yang diprediksi "Berpotensi Dibatalkan", tidak semuanya benar-benar akan batal. Hasil klasifikasi sebaiknya dipakai sebagai bahan pertimbangan/prioritas pemantauan, bukan keputusan otomatis tunggal.
        """
    )

with st.expander("📊 Perbandingan Performa Model -- Logistic Regression vs XGBoost (RM2)", expanded=False):
    st.markdown(
        "Ringkasan hasil evaluasi lima skenario eksperimen pada data uji (Tabel 4.1 skripsi). "
        "Baris yang ditandai adalah model yang dipakai aplikasi ini."
    )
    comp_df = pd.DataFrame(SCENARIO_COMPARISON)

    def _highlight_used(row):
        return ["background-color: #fdf3dd" if "dipakai aplikasi ini" in row["Skenario"] else "" for _ in row]

    st.dataframe(
        comp_df.style.apply(_highlight_used, axis=1).format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "Macro F1": "{:.4f}", "ROC-AUC": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Macro F1 tertinggi diperoleh S2 (Logistic Regression + SMOTE-NC), tetapi uji signifikansi "
        "(koreksi Nadeau-Bengio) menunjukkan S2, S3, dan S4 tidak berbeda signifikan secara statistik "
        "(Sub-bab 4.3.3 skripsi) -- ketiganya setara, bukan salah satu terbukti unggul."
    )

st.divider()

st.markdown(
    "Upload file ekspor **\"Order SKU List\"** langsung dari platform (format xlsx apa adanya, "
    "tidak perlu diedit dulu). File boleh berisi pesanan dengan status apa pun -- **seluruh "
    "pesanan** akan diklasifikasikan modelnya, sedangkan rekap faktor dominan (RM3) khusus "
    "dihitung dari baris yang **sudah berstatus batal**."
)
uploaded = st.file_uploader("File Order SKU List (.xlsx)", type=["xlsx"])

if uploaded is not None:
    file_key = (uploaded.name, uploaded.size)
    if st.session_state.get("file_key") != file_key:
        try:
            with st.spinner("Membaca & mengagregasi data pesanan..."):
                order_level = load_order_level(uploaded, require_target=False)
        except ValueError as e:
            st.error(str(e))
            order_level = None
        except Exception as e:  # format file tidak terbaca sama sekali, dsb.
            st.error(f"Gagal membaca file: {e}")
            order_level = None

        st.session_state["order_level"] = order_level
        st.session_state["file_key"] = file_key

    order_level = st.session_state.get("order_level")

    if order_level is not None:
        st.success(f"Berhasil membaca {len(order_level)} pesanan unik dari file.")

        # ---------------------------------------------------------------
        # Filter: rentang waktu + atribut pesanan (opsional), diterapkan ke
        # SELURUH pesanan di file (bukan cuma yang batal) supaya rasio
        # pembatalan per kelompok pada bagian insight tetap benar.
        # ---------------------------------------------------------------
        created = pd.to_datetime(order_level["created"], errors="coerce")
        valid_dates = created.dropna()

        with st.expander("🔎 Filter (opsional)", expanded=False):
            if not valid_dates.empty:
                min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
                date_range = st.date_input("Rentang tanggal pesanan dibuat", value=(min_d, max_d),
                                            min_value=min_d, max_value=max_d)
            else:
                date_range = None
            f1, f2, f3, f4 = st.columns(4)
            cat_opts = sorted(order_level["category"].dropna().unique().tolist())
            prov_opts = sorted(order_level["province"].dropna().unique().tolist())
            pay_opts = sorted(order_level["payment"].dropna().unique().tolist())
            store_opts = sorted(order_level["store"].dropna().unique().tolist())
            sel_cat = f1.multiselect("Kategori produk", cat_opts)
            sel_prov = f2.multiselect("Provinsi", prov_opts)
            sel_pay = f3.multiselect("Metode pembayaran", pay_opts)
            sel_store = f4.multiselect("Nama toko", store_opts)

        mask = pd.Series(True, index=order_level.index)
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            mask &= created.dt.date.between(start, end) | created.isna()
        if sel_cat:
            mask &= order_level["category"].isin(sel_cat)
        if sel_prov:
            mask &= order_level["province"].isin(sel_prov)
        if sel_pay:
            mask &= order_level["payment"].isin(sel_pay)
        if sel_store:
            mask &= order_level["store"].isin(sel_store)

        scoped = order_level[mask].copy()
        cancelled = scoped[scoped["target"] == 1].copy()

        n_scope = len(scoped)
        n_cancel = len(cancelled)
        m1, m2, m3 = st.columns(3)
        m1.metric("Pesanan pada cakupan filter", n_scope)
        m2.metric("Berstatus batal", n_cancel)
        m3.metric("Rasio pembatalan", f"{(n_cancel / n_scope * 100 if n_scope else 0):.1f}%")

        # ------------------------------------------------------- hasil klasifikasi (RM1)
        st.divider()
        st.subheader("Hasil Klasifikasi Model")
        st.caption(
            "Model mengklasifikasi SETIAP pesanan pada cakupan filter (bukan hanya yang "
            "sudah batal) berdasarkan probabilitas pembatalan. Threshold dapat diatur di "
            "bawah ini untuk melihat trade-off precision/recall."
        )
        default_thr = float(bundle["threshold"])
        thr = st.slider(
            "Ambang batas (threshold) klasifikasi", min_value=0.05, max_value=0.95,
            value=default_thr, step=0.05,
            help="Probabilitas >= ambang batas ini diklasifikasikan sebagai 'Berpotensi Dibatalkan'.",
        )
        with st.spinner("Menjalankan klasifikasi model..."):
            pred_raw = predict_batch(scoped, bundle)
            pred_labeled = label_predictions(pred_raw, thr)

        n_flagged = int((pred_labeled["Prediksi"] == "Berpotensi Dibatalkan").sum())
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Diklasifikasikan 'Berpotensi Dibatalkan'", n_flagged)
        pm2.metric("Ambang batas dipakai", f"{thr:.2f}")
        near = None
        try:
            from src.predict import nearest_threshold_stats
            near = nearest_threshold_stats(bundle, thr)
        except Exception:
            near = None
        if near:
            pm3.metric("Perkiraan precision pada ambang ini", f"{near['precision']:.1%}")

        st.dataframe(
            pred_labeled.style.format({"Probabilitas Dibatalkan": "{:.1%}"})
            .background_gradient(subset=["Probabilitas Dibatalkan"], cmap="Reds"),
            use_container_width=True, height=380,
        )
        st.caption(
            "\"Faktor Utama\" pada tabel ini adalah fitur dengan kontribusi SHAP absolut "
            "terbesar terhadap probabilitas klasifikasi pesanan tersebut."
        )
        pred_csv = pred_labeled.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Download hasil klasifikasi (CSV)", data=pred_csv,
                            file_name="hasil_klasifikasi_pesanan.csv", mime="text/csv")

        if n_cancel == 0:
            st.warning("Tidak ada pesanan berstatus batal pada cakupan filter saat ini.")
        else:
            with st.spinner(f"Menganalisis faktor penyebab {n_cancel} pesanan batal..."):
                diag = diagnose_batch(cancelled, bundle)

            recap = (
                diag["Faktor Dominan"].value_counts()
                .rename_axis("Faktor Dominan")
                .reset_index(name="Jumlah Pesanan")
            )
            recap["Persentase"] = recap["Jumlah Pesanan"] / n_cancel * 100

            def rasio_table(df, group_col, out_col, top_n=None):
                t = (
                    df.groupby(group_col)
                    .agg(total=("target", "size"), batal=("target", lambda s: (s == 1).sum()))
                    .assign(rasio=lambda d: d["batal"] / d["total"] * 100)
                    .sort_values("batal", ascending=False)
                )
                if top_n:
                    t = t.head(top_n)
                return t.reset_index().rename(columns={
                    group_col: out_col, "total": "Total Pesanan",
                    "batal": "Jumlah Batal", "rasio": "Rasio Batal (%)",
                })

            def highlight(df):
                return df.style.format({"Rasio Batal (%)": "{:.1f}%"}) \
                    .background_gradient(subset=["Rasio Batal (%)"], cmap="Reds")

            by_prov = rasio_table(scoped, "province", "Provinsi", top_n=10)
            by_store = rasio_table(scoped, "store", "Nama Toko")
            by_cat = rasio_table(scoped, "category", "Kategori Produk", top_n=10)

            sku_lookup = get_sku_lookup()
            scoped_prod = scoped.assign(
                produk=scoped["sku"].apply(lambda s: map_sku_to_product(s, sku_lookup))
            )
            by_prod = rasio_table(scoped_prod, "produk", "Produk", top_n=10)

            monthly = (
                cancelled.assign(bulan=pd.to_datetime(cancelled["created"], errors="coerce").dt.to_period("M").astype(str))
                .groupby("bulan").size().rename("Jumlah Batal").reset_index()
                .sort_values("bulan")
            )

            # ------------------------------------------------------- ringkasan naratif
            st.divider()
            st.subheader("Ringkasan Temuan")
            bullets = [
                f"Dari **{n_scope}** pesanan pada cakupan filter, **{n_cancel}** "
                f"({(n_cancel / n_scope * 100 if n_scope else 0):.1f}%) berstatus batal.",
                f"Faktor paling sering menjadi penyebab dominan: **{recap.iloc[0]['Faktor Dominan']}** "
                f"({recap.iloc[0]['Persentase']:.1f}% dari pesanan batal).",
            ]
            if not by_store.empty:
                r = by_store.iloc[0]
                bullets.append(f"Toko dengan pembatalan terbanyak: **{r['Nama Toko']}** "
                                f"({int(r['Jumlah Batal'])} pesanan, rasio {r['Rasio Batal (%)']:.1f}%).")
            if not by_prov.empty:
                r = by_prov.iloc[0]
                bullets.append(f"Provinsi dengan pembatalan terbanyak: **{r['Provinsi']}** "
                                f"({int(r['Jumlah Batal'])} pesanan, rasio {r['Rasio Batal (%)']:.1f}%).")
            if not by_cat.empty:
                r = by_cat.iloc[0]
                bullets.append(f"Kategori produk dengan pembatalan terbanyak: **{r['Kategori Produk']}** "
                                f"({int(r['Jumlah Batal'])} pesanan, rasio {r['Rasio Batal (%)']:.1f}%).")
            if not by_prod.empty:
                r = by_prod.iloc[0]
                bullets.append(f"Produk dengan pembatalan terbanyak: **{r['Produk']}** "
                                f"({int(r['Jumlah Batal'])} pesanan, rasio {r['Rasio Batal (%)']:.1f}%).")
            st.info("\n".join(f"- {b}" for b in bullets))
            st.caption(
                "Ringkasan berdasarkan jumlah pesanan batal terbanyak (bukan rasio tertinggi), "
                "supaya tidak bias oleh kelompok dengan volume pesanan yang sangat kecil."
            )

            # ------------------------------------------------------- rekap faktor dominan
            st.divider()
            st.subheader("Rekap Faktor Dominan Penyebab Pembatalan")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(
                    recap.style.format({"Persentase": "{:.1f}%"})
                    .background_gradient(subset=["Persentase"], cmap="Reds"),
                    use_container_width=True, height=380,
                )
            with c2:
                top = recap.sort_values("Jumlah Pesanan").tail(12)
                fig = px.bar(
                    top, x="Jumlah Pesanan", y="Faktor Dominan", orientation="h",
                    color="Jumlah Pesanan", color_continuous_scale="Reds",
                    text="Jumlah Pesanan",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    coloraxis_showscale=False, margin=dict(l=0, r=10, t=10, b=0),
                    yaxis_title=None, xaxis_title="Jumlah pesanan batal", height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Faktor dominan dihitung per pesanan lewat SHAP (fitur dengan pengaruh absolut "
                "terbesar terhadap klasifikasi model untuk pesanan tersebut), lalu direkap. "
                "Angka ini menunjukkan pola penyebab, bukan jaminan sebab-akibat tunggal."
            )

            # ------------------------------------------------------- insight tambahan
            st.divider()
            st.subheader("Insight Tambahan")

            i1, i2 = st.columns(2)
            with i1:
                st.markdown("**Tren bulanan pesanan batal**")
                if not monthly.empty:
                    fig2 = px.line(
                        monthly, x="bulan", y="Jumlah Batal", markers=True,
                        color_discrete_sequence=[REDIGMA_RED],
                    )
                    fig2.update_layout(
                        margin=dict(l=0, r=10, t=10, b=0),
                        xaxis_title="Bulan", yaxis_title="Jumlah pesanan batal", height=380,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.caption("Tanggal pesanan tidak terbaca, tren bulanan tidak dapat ditampilkan.")

            with i2:
                st.markdown("**Komposisi faktor dominan per bulan (5 faktor teratas)**")
                diag_dated = cancelled[["Order ID", "created"]].merge(diag, on="Order ID")
                diag_dated["bulan"] = pd.to_datetime(diag_dated["created"], errors="coerce").dt.to_period("M").astype(str)
                diag_dated = diag_dated[diag_dated["bulan"] != "NaT"]
                top5 = recap["Faktor Dominan"].head(5).tolist()
                diag_dated["kelompok"] = diag_dated["Faktor Dominan"].where(
                    diag_dated["Faktor Dominan"].isin(top5), "Lainnya")
                long = diag_dated.groupby(["bulan", "kelompok"]).size().reset_index(name="Jumlah")
                if not long.empty:
                    fig3 = px.bar(
                        long.sort_values("bulan"), x="bulan", y="Jumlah", color="kelompok",
                        color_discrete_sequence=px.colors.sequential.Reds_r[:5] + ["#bbbbbb"],
                    )
                    fig3.update_layout(
                        barmode="stack", margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Bulan", yaxis_title="Jumlah pesanan batal",
                        legend_title=None, legend=dict(font=dict(size=10)), height=380,
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.caption("Tanggal pesanan tidak terbaca, komposisi bulanan tidak dapat ditampilkan.")

            st.markdown("**Rasio pembatalan per toko**")
            st.dataframe(highlight(by_store), use_container_width=True,
                         height=min(380, 60 + 35 * len(by_store)))

            st.markdown("**Rasio pembatalan per provinsi (top 10 volume batal)**")
            st.dataframe(highlight(by_prov), use_container_width=True, height=380)

            st.markdown("**Rasio pembatalan per kategori produk (top 10 volume batal)**")
            st.dataframe(highlight(by_cat), use_container_width=True, height=380)

            st.markdown("**Rasio pembatalan per produk (Seller SKU diterjemahkan ke nama produk, top 10 volume batal)**")
            st.dataframe(highlight(by_prod), use_container_width=True, height=380)
            if not sku_lookup["exact"] and not sku_lookup["prefix"]:
                st.caption(
                    "Katalog kode produk (reference/kode_produk.xlsx) tidak ditemukan -- "
                    "tabel di atas menampilkan kode Seller SKU mentah."
                )

            st.divider()
            detail = cancelled[["Order ID"]].merge(diag, on="Order ID")
            csv_bytes = recap.to_csv(index=False).encode("utf-8-sig")
            detail_bytes = detail.to_csv(index=False).encode("utf-8-sig")
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download rekap faktor (CSV)", data=csv_bytes,
                                file_name="rekap_faktor_pembatalan.csv", mime="text/csv")
            d2.download_button("⬇️ Download detail per pesanan (CSV)", data=detail_bytes,
                                file_name="detail_faktor_per_pesanan.csv", mime="text/csv")

st.divider()
st.caption(
    "Dibangun dari model & data skripsi \"Klasifikasi Pembatalan Pesanan E-Commerce Menggunakan "
    "Logistic Regression dan XGBoost, dan Analisis Faktor\" -- Irsyad Muhamad Firdaus, "
    "Teknik Informatika, Universitas Muhammadiyah Magelang."
)
