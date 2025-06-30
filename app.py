import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# Judul aplikasi
st.title('🎓 Prediksi Tingkat Overthinking Mahasiswa Semester 8')
st.write('Aplikasi ini memprediksi tingkat overthinking berdasarkan kebiasaan mahasiswa menggunakan Decision Tree')

# 1. Membuat dataset contoh
@st.cache_data
def load_data():
    data = {
        'Jam_Tidur': [6, 5, 7, 6, 5, 8, 7, 6, 5, 7, 6, 5, 4, 6, 7],
        'Beban_Tugas': [8, 9, 6, 7, 9, 5, 6, 8, 9, 7, 8, 9, 10, 7, 6],
        'Aktivitas_Sosial': ['rendah', 'rendah', 'sedang', 'sedang', 'rendah', 'tinggi', 'sedang', 
                            'rendah', 'rendah', 'sedang', 'rendah', 'rendah', 'rendah', 'sedang', 'tinggi'],
        'Penggunaan_MedSos': ['tinggi', 'tinggi', 'sedang', 'sedang', 'tinggi', 'rendah', 'sedang',
                             'tinggi', 'tinggi', 'sedang', 'tinggi', 'tinggi', 'tinggi', 'sedang', 'rendah'],
        'Overthinking': ['tinggi', 'tinggi', 'sedang', 'sedang', 'tinggi', 'rendah', 'sedang',
                        'tinggi', 'tinggi', 'sedang', 'tinggi', 'tinggi', 'tinggi', 'sedang', 'rendah']
    }
    return pd.DataFrame(data)

df = load_data()

# 2. Sidebar untuk input parameter model
st.sidebar.header('Parameter Model')
max_depth = st.sidebar.slider('Kedalaman Maksimal Pohon (max_depth)', 1, 10, 3)
test_size = st.sidebar.slider('Ukuran Data Testing (%)', 10, 40, 30)

# 3. Preprocessing data
le = LabelEncoder()
df_encoded = df.copy()
for col in ['Aktivitas_Sosial', 'Penggunaan_MedSos', 'Overthinking']:
    df_encoded[col] = le.fit_transform(df[col])

X = df_encoded.drop('Overthinking', axis=1)
y = df_encoded['Overthinking']

# 4. Membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# 5. Membangun model
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=max_depth,
    min_samples_split=2,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 7. Tampilkan hasil di main panel
st.header('Hasil Model')
st.write(f'Akurasi Model: **{accuracy:.2%}**')

# Visualisasi Decision Tree
st.subheader('Visualisasi Decision Tree')
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(model, 
              feature_names=X.columns, 
              class_names=['rendah', 'sedang', 'tinggi'], 
              filled=True, 
              rounded=True,
              ax=ax)
st.pyplot(fig)

# 8. Form untuk prediksi baru
st.header('Prediksi Tingkat Overthinking')
with st.form('input_form'):
    st.write("Masukkan data mahasiswa:")
    col1, col2 = st.columns(2)
    
    with col1:
        jam_tidur = st.slider('Jam Tidur per Hari', 3, 10, 6)
        beban_tugas = st.slider('Beban Tugas (Skala 1-10)', 1, 10, 7)
    
    with col2:
        aktivitas_sosial = st.selectbox('Aktivitas Sosial', ['rendah', 'sedang', 'tinggi'])
        penggunaan_medsos = st.selectbox('Penggunaan Media Sosial', ['rendah', 'sedang', 'tinggi'])
    
    submitted = st.form_submit_button('Prediksi')
    
    if submitted:
        # Membuat dataframe dari input
        input_data = pd.DataFrame({
            'Jam_Tidur': [jam_tidur],
            'Beban_Tugas': [beban_tugas],
            'Aktivitas_Sosial': [aktivitas_sosial],
            'Penggunaan_MedSos': [penggunaan_medsos]
        })
        
        # Encoding input
        input_data['Aktivitas_Sosial'] = le.transform(input_data['Aktivitas_Sosial'])
        input_data['Penggunaan_MedSos'] = le.transform(input_data['Penggunaan_MedSos'])
        
        # Prediksi
        prediction = model.predict(input_data)
        prediction_label = le.inverse_transform(prediction)[0]
        
        # Tampilkan hasil
        st.subheader('Hasil Prediksi')
        if prediction_label == 'tinggi':
            st.error(f'Tingkat Overthinking: {prediction_label} 🚨')
            st.write('Rekomendasi: Konsultasi dengan konselor, kurangi beban tugas, tingkatkan aktivitas sosial')
        elif prediction_label == 'sedang':
            st.warning(f'Tingkat Overthinking: {prediction_label} ⚠️')
            st.write('Rekomendasi: Manajemen waktu yang lebih baik, istirahat cukup')
        else:
            st.success(f'Tingkat Overthinking: {prediction_label} ✅')
            st.write('Pertahankan kebiasaan baik Anda!')

# Tampilkan dataset
if st.checkbox('Tampilkan Dataset'):
    st.subheader('Dataset Contoh')
    st.write(df)
    st.caption('Catatan: Dataset ini adalah data simulasi untuk demo aplikasi')

# Cara menjalankan
st.sidebar.header('Petunjuk')
st.sidebar.info("""
1. Atur parameter model di sidebar
2. Isi form prediksi di main panel
3. Klik tombol 'Prediksi'
4. Hasil akan muncul di bagian bawah
""")