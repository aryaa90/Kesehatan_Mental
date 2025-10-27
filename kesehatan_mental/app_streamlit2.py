import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_catboost_kesehatan_mental.joblib")

st.title("Klasifikasi Mental")
st.markdown("Prediksi kesehatan mental berdasarkan usia dan kebiasaan harian")

Jurusan = st.selectbox("Jurusan", ["Teknik_Mesin", "Akuntansi", "Multimedia", "Tata_Boga", "Perkantoran", "Teknik_Otomotif"])
Usia = st.slider("Usia", min_value=15, max_value=18, value=15)
Jenis_Kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
Pendapatan_Keluarga = st.selectbox("Pendapatan Keluarga", ["Rendah", "Menengah", "Tinggi"])
Lokasi_Sekolah = st.selectbox("Lokasi Sekolah", ["Rural", "Suburban", "Urban"])
Jumlah_Jam_HP_Harian = st.slider("Jumlah Jam HP Harian", min_value=2, max_value=11, value=8)

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame([[
        Jurusan, Usia, Jenis_Kelamin, Pendapatan_Keluarga,
        Lokasi_Sekolah, Jumlah_Jam_HP_Harian
    ]], columns=['Jurusan', 'Usia', 'Jenis Kelamin', 'Pendapatan Keluarga', 'Lokasi Sekolah', 'Jumlah Jam HP Harian'])

    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])

    st.success(f"Prediksi: {prediksi} (keyakinan: {presentase*100:.2f}%)")
    st.balloons()
    st.snow()

st.divider()
st.caption("Dibuat dengan otak oleh  **Arya Wisanggeni**")