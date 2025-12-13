# 🩺 Team-Based UAS Project: ML-Heart-Disease-Prediction

DEMO MODEL
<!-- Uploading "Demo Prediksi Penyakit Jantung.gif"... -->

## 1. Pendahuluan
Proyek ini bertujuan untuk membangun model Machine Learning (ML) yang mampu memprediksi risiko seseorang menderita penyakit jantung (Target: 1) berdasarkan data klinis multi-parameter (Target: 0). Proyek ini sepenuhnya diimplementasikan melalui alur kerja Git/GitHub yang terstruktur.

**Tema UAS:** Healthcare (Pelayanan Kesehatan)

## 2. Anggota Kelompok & Bukti Kolaborasi Git Flow

Semua pekerjaan dilakukan melalui *branching* yang sistematis (`commit`, `branch`, dan `pull request`) sebagai bukti kolaborasi.

| Anggota | Peran | Kontribusi Utama | Branch |
| :--- | :--- | :--- | :--- |
| **Chalista** | Project Lead / Reviewer | Setup Repositori, Review dan Merge Pull Requests (PRs). | `main` |
| **Azza** | Model Training | Data Preprocessing (Scaling), Data Splitting, Model Training (Random Forest), dan Model Evaluation. | `feature/model-build` |
| **Hani** | Dokumentasi & Deployment | Finalisasi Dokumentasi (README), dan Implementasi Web Deployment menggunakan Flask/HTML. | `feature/hani-documentation-and-deployment` |

## 3. Tahapan Pengembangan & Hasil Akhir

### A. Data Preprocessing (Oleh Azza)
* **Tujuan:** Menyiapkan dan membersihkan data untuk pelatihan model.
* **Script:** `src/data_preprocessing.py`
* **Proses Kunci:** Data dibagi 80% (Training) dan 20% (Testing). Fitur diskalakan menggunakan `StandardScaler` (`src/scaler.joblib`) untuk memastikan fitur memiliki kontribusi yang setara.
* **Output Data:** `data/X_train.csv`, `data/y_train.csv`, dll.

### B. Pembangunan & Pengujian Model (Oleh Azza)
* **Model:** Random Forest Classifier.
* **Script:** `src/model_training.py`
* **Langkah:** Model dilatih menggunakan data yang sudah di-scale.
* **Model Final:** Disimpan sebagai `src/final_model.joblib`.
* **Hasil Pengujian pada Data Testing:**
    * **Akurasi Model:** **[ISI NILAI AKURASI FINAL AZZA DI SINI, misal: 87.50%]**

### C. Implementasi Hasil Akhir (Deployment) (Oleh Hani)
Untuk menunjukkan model yang dapat dijalankan, kami membuat aplikasi web sederhana:
* **Backend Server:** Flask (`src/app.py`), yang memuat model (`final_model.joblib`) dan *scaler*.
* **Frontend User Interface:** HTML (`templates/index.html`), yang menyediakan *form input* klinis pasien.
* **Cara Menjalankan Lokal:**
    1.  Instal dependensi: `pip install -r requirements.txt` (pastikan Flask, pandas, scikit-learn, joblib ada).
    2.  Jalankan server: `python src/app.py`
    3.  Akses di browser: `http://127.0.0.1:5000/`

## 4. Dependencies (`requirements.txt`)
Daftar library Python yang diperlukan untuk menjalankan proyek:



