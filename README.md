# 🏥 VeraMed AI: BPJS Claim Fraud Detection & Medical Audit

VeraMed AI adalah sistem deteksi anomali klaim kesehatan yang dirancang khusus untuk ekosistem BPJS Kesehatan Indonesia. Proyek ini menggabungkan kekuatan **Hybrid Machine Learning** untuk analisis data klaim massal dan **Gemini Vision AI** untuk audit kognitif terhadap dokumen rekam medis fisik.

## 📌 Mengapa VeraMed AI?
Berdasarkan data riset, sekitar **59.43%** kendala klaim BPJS bersumber dari dokumentasi medis yang tidak lengkap atau tidak valid. VeraMed AI hadir untuk mengotomatisasi proses verifikasi ini, mendeteksi potensi *inflated cost*, dan memvalidasi keabsahan administrasi secara real-time.

---

## 🚀 Fitur Utama

### 1. Hybrid Risk Scoring
Sistem menggunakan dua pendekatan model sekaligus untuk menghasilkan skor risiko (0-100):
*   **Supervised (XGBoost)**: Mempelajari pola fraud dari data historis berlabel.
*   **Unsupervised (Isolation Forest)**: Mendeteksi anomali statistik pada klaim yang terlihat "aneh" meskipun belum pernah ada presedennya (misal: biaya demam yang melonjak drastis).

### 2. AI Medical Document Extractor (Gemini Vision)
Modul audit kognitif yang mampu membaca foto atau PDF rekam medis untuk:
*   Mengekstraksi data ke format JSON terstruktur.
*   Memverifikasi keberadaan tanda tangan DPJP (Dokter Penanggung Jawab).
*   Mengecek kelengkapan resume medis sesuai standar Permenkes.

### 3. Interactive Analytics Dashboard
Dashboard berbasis Streamlit yang menyajikan:
*   Visualisasi distribusi biaya per kode ICD-10.
*   Audit detail per klaim dengan penjelasan alasan anomali (Explainable AI).
*   Eksport hasil audit ke dataset untuk pelatihan model berkelanjutan.

---

## 🛠️ Teknologi yang Digunakan
*   **Core**: Python 3.10+
*   **Machine Learning**: XGBoost, Scikit-Learn (Isolation Forest)
*   **AI Engine**: Google Gemini 2.0 Flash Lite (Vision API)
*   **Dashboard**: Streamlit, Plotly
*   **Data Handling**: Pandas, PyArrow, Joblib

---

## 📦 Instalasi & Persiapan

1. **Clone Repository**
   ```bash
   git clone https://github.com/username/VeraMed-Ai.git
   cd VeraMed-Ai
   ```

2. **Instal Dependensi**
   ```bash
   pip install -r requirements.txt
   ```

3. **Konfigurasi Environment**
   Salin file `.env.example` menjadi `.env` dan masukkan API Key Anda:
   ```env
   GOOGLE_API_KEY=AIzaSy...
   GEMINI_MODEL=gemini-2.0-flash-lite
   APP_ENV=production
   ```

---

## 💻 Cara Penggunaan

### A. Melatih Model
Jalankan script training untuk menghasilkan model `.pkl` terbaru berdasarkan dataset sintetis:
```bash
python train_model.py
```

### B. Evaluasi Model
Lihat laporan performa model secara mendalam (Accuracy, Precision, Recall, ROC-AUC):
```bash
python evaluate_model.py
```

### C. Menjalankan Dashboard
Jalankan aplikasi web lokal:
```bash
streamlit run app.py
```

---

## 🧠 Arsitektur Logika Audit
VeraMed AI tidak hanya mengandalkan angka, tapi juga aturan medis yang kaku:
1.  **Inflated Cost**: Jika diagnosa adalah `R50.9` (Demam) namun biaya > Rp 5.000.000.
2.  **Admin Validity**: Klaim otomatis ditandai berisiko tinggi jika tanda tangan dokter tidak terdeteksi oleh AI (bobot pengaruh 50.69%).
3.  **Short Stay High Cost**: Durasi inap (LOS) hanya 1 hari namun biaya melampaui ambang batas prosedur ringan.

---

## 📄 Struktur Proyek
```text
├── app.py                   # Entry point aplikasi dashboard
├── train_model.py           # Pipeline pelatihan ML
├── evaluate_model.py        # Metrik evaluasi performa
├── extractor.py             # Engine OCR & Audit Gemini
├── pages/                   # Modul halaman tambahan Streamlit
├── models/                  # Folder penyimpanan model .pkl
└── bpjs_claims_synthetic.csv # Dataset pelatihan
```

---

## ⚖️ Lisensi & Disclaimer
Sistem ini merupakan alat bantu verifikasi (Decision Support System) dan bukan pengganti verifikator manusia sepenuhnya. Seluruh data yang digunakan dalam demo ini adalah data sintetis yang dihasilkan secara acak untuk tujuan pengembangan.

© 2026 **VeraMed AI Team**
