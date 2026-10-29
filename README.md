# Klasifikasi Sentimen Ulasan Aplikasi Pemerintah Indonesia (Dataset IGAR)

Proyek penelitian machine learning untuk klasifikasi sentimen ulasan publik terhadap aplikasi instansi dan BUMN pemerintah Republik Indonesia, dengan fokus khusus pada evaluasi **generalisasi lintas aplikasi (*cross-app generalization*)**.

---

## 📁 Struktur Direktori Repository

```text
government-app-sentiment-analysis/
├── data/
│   ├── Rating_labeled_raw.csv        # Dataset ulasan mentah asli (JANGAN DIUBAH)
│   └── Rating_labeled_processed.csv  # Dataset hasil preprocessing siap pakai untuk Anggota 2 & 3
├── docs/
│   ├── data_audit.md                 # Laporan audit integritas data, label noise, dan risiko leakage
│   └── preprocessing_report.md       # Laporan pipeline preprocessing, kamus slang, dan panduan tim
├── notebooks/
│   └── preprocessing.ipynb           # Notebook pipeline preprocessing (audit konsolidasi, cleaning, metadata)
├── src/
│   └── audit.py                      # Script CLI audit integritas data cepat
├── models/                           # Direktori penyimpanan bobot model dan vektor TF-IDF (.pkl / .joblib)
└── README.md                         # Dokumentasi umum proyek
```

---

## 🚀 Panduan Workflow Preprocessing (Anggota 1)

### 1. Menjalankan Pipeline Preprocessing:
Buka dan jalankan seluruh cell pada notebook:
```bash
jupyter notebook notebooks/preprocessing.ipynb
```
Notebook ini melakukan:
- Validasi skema & konsistensi label (`score` 1–2 = Negative, 3 = Neutral, 4–5 = Positive).
- Audit duplikasi dan konflik rating/label pada teks identik.
- Pembersihan teks secara konservatif (emoji-to-text, lowercase, pembersihan URL/email/karakter non-alfanumerik, reduksi elongasi, normalisasi slang/singkatan, tanpa stemming & tanpa stopword removal).
- Filtering minimal (hanya 980 baris yang kosong/non-usable, raw data 617.722 -> 616.742 baris).
- Ekstraksi metadata grup duplikasi (`duplicate_group`, `group_size`, `has_label_conflict`, `n_apps_in_group`).
- Export otomatis ke `data/Rating_labeled_processed.csv` dan `docs/preprocessing_report.md`.

### 2. Menjalankan Audit Cepat CLI:
```bash
python src/audit.py --data-path data/Rating_labeled_raw.csv
```

---

## 📌 Panduan untuk Anggota 2 (Model Training) & Anggota 3 (Dashboard)

Detail lengkap spesifikasi kolom, rekomendasi penanganan data leakage, class imbalance, dan evaluasi *cross-app* / Leave-One-App-Out (LOAO) dapat dilihat pada dokumen:
- [preprocessing_report.md](docs/preprocessing_report.md)
- [data_audit.md](docs/data_audit.md)
