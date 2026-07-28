"""Alat bantu diagnostik: rekap faktor penyebab pembatalan pesanan REDIGMA
(PT Relasi Digital Marketing), memakai model XGBoost skenario S3 dari skripsi
"Klasifikasi Pembatalan Pesanan E-Commerce ... dan Analisis Faktor" (Irsyad
Muhamad Firdaus). Sesuai konsep pada Sub-bab 3.2.3 skripsi: alat ini BUKAN
prediktor real-time -- hanya menganalisis pesanan yang STATUSNYA SUDAH batal,
lalu merekap faktor apa yang paling sering menjadi penyebab dominannya. Tidak
ada skor probabilitas atau label prediksi yang ditampilkan ke pengguna.

Jalankan: streamlit run app.py
(model harus sudah dilatih lebih dulu lewat scripts/train.py -- lihat README.md)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import diagnose_batch, load_bundle
from src.product_lookup import load_sku_to_product, map_sku_to_product
from src.raw_data import load_order_level
from src.reference_data import format_model_info


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

st.title("📦 Rekap Faktor Penyebab Pembatalan Pesanan")
st.caption(
    "Alat bantu diagnostik dari skripsi klasifikasi pembatalan pesanan e-commerce REDIGMA "
    "(model: XGBoost skenario S3, lihat Bab 4). Menganalisis pesanan yang **sudah** "
    "berstatus batal -- bukan memprediksi pesanan yang masih berjalan."
)
with st.expander("ℹ️ Tentang model & keterbatasannya", expanded=False):
    st.markdown(
        f"""
- Model: **XGBoost** (skenario S3 pada Bab 4 skripsi -- Macro F1 tertinggi di antara lima skenario yang dibandingkan).
- {format_model_info(bundle)}
- Fitur risiko per wilayah/metode bayar/kategori/SKU dihitung dari riwayat data training (*smoothed target encoding*) -- kombinasi yang sangat jarang muncul di data training akan condong ke rata-rata global, bukan kesalahan sistem.
- Alat ini **tidak** menampilkan skor probabilitas atau label prediksi. Fokusnya adalah menjelaskan pesanan yang statusnya sudah diketahui batal, sebagai bahan evaluasi internal (lihat Sub-bab 3.2.3 & 4.2.4 skripsi), bukan mengambil keputusan otomatis untuk pesanan yang sedang berjalan.
        """
    )

st.divider()

st.markdown(
    "Upload file ekspor **\"Order SKU List\"** langsung dari platform (format xlsx apa adanya, "
    "tidak perlu diedit dulu). File boleh berisi pesanan dengan status apa pun -- app akan "
    "otomatis memilih hanya baris yang **sudah berstatus batal** untuk dianalisis."
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
                fig, ax = plt.subplots(figsize=(6, 4.2))
                ax.barh(top["Faktor Dominan"], top["Jumlah Pesanan"], color="#d62728")
                ax.set_xlabel("Jumlah pesanan batal")
                fig.tight_layout()
                st.pyplot(fig)

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
                    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
                    ax2.plot(monthly["bulan"], monthly["Jumlah Batal"], marker="o", color="#d62728")
                    ax2.set_xlabel("Bulan")
                    ax2.set_ylabel("Jumlah pesanan batal")
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
                    fig2.tight_layout()
                    st.pyplot(fig2)
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
                pivot = diag_dated.groupby(["bulan", "kelompok"]).size().unstack(fill_value=0).sort_index()
                if not pivot.empty:
                    fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
                    bottom = None
                    for col in pivot.columns:
                        ax3.bar(pivot.index, pivot[col], bottom=bottom, label=col)
                        bottom = pivot[col] if bottom is None else bottom + pivot[col]
                    ax3.set_xlabel("Bulan")
                    ax3.set_ylabel("Jumlah pesanan batal")
                    ax3.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
                    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
                    fig3.tight_layout()
                    st.pyplot(fig3)
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
