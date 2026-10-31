# Panduan Dashboard Analisis Sentimen Aplikasi Pemerintah

Proyek ini menyajikan analisis sentimen ulasan pengguna pada enam aplikasi layanan pemerintah Indonesia dalam dataset IGAR. Dashboard membantu pembaca melihat pola sentimen, menelusuri ulasan, membandingkan hasil evaluasi model, dan mencoba prediksi pada teks baru.

Pertanyaan penelitian utamanya adalah: **seberapa baik model mengklasifikasikan sentimen ulasan, dan apakah kemampuannya tetap terjaga ketika diuji pada aplikasi yang tidak digunakan untuk melatih model?**

## Mulai menggunakan dashboard

### 1. Siapkan Python dan dependensi

Pastikan Python tersedia, lalu pasang paket yang digunakan aplikasi:

```bash
python -m pip install streamlit pandas numpy plotly scikit-learn joblib
```

### 2. Pastikan data dan artefak tersedia

Dashboard membaca file yang sudah disiapkan. File data dan model berukuran besar tidak disertakan dalam Git repository ini, jadi pastikan file berikut tersedia di komputer tempat aplikasi dijalankan:

```text
data/Rating_labeled_processed.csv.gz
models/holdout_model_comparison.csv
models/loao_results.csv
models/confusion_matrices.json
models/rating_classifier_bundle.pkl
```

Jika Member 2 memiliki ekspor langsung bundle model dari notebook, dashboard akan memprioritaskan `notebooks/models/rating_classifier_bundle.pkl`, lalu menggunakan `models/rating_classifier_bundle.pkl` sebagai cadangan. File `classification_reports.json` lama tidak digunakan; metrik per kelas dihitung dari jumlah pada confusion matrix.

Tanpa dataset, halaman eksplorasi tidak dapat menampilkan ulasan. Tanpa artefak evaluasi atau model, halaman terkait akan menampilkan informasi bahwa artefaknya belum tersedia.

### 3. Jalankan aplikasi

Dari folder utama proyek, jalankan:

```bash
streamlit run src/dashboard.py
```

Streamlit akan membuka dashboard di browser. Gunakan panel kiri untuk berpindah halaman, memilih mode terang/gelap, dan mengubah bahasa Indonesia/Inggris.

> Membuka dashboard tidak melatih ulang model. Predictor hanya membersihkan teks masukan, mengubahnya dengan TF-IDF yang sudah tersimpan, lalu menjalankan prediksi menggunakan model bundle.

## Panduan membaca halaman

### 1. Ringkasan

Menampilkan jumlah ulasan, aplikasi, dan distribusi sentimen serta rating. Kartu temuan merangkum hal seperti aplikasi dengan ulasan terbanyak atau proporsi sentimen tertinggi. Nilainya dihitung dari dataset yang dimuat dashboard.

### 2. Eksplorasi Dataset

Gunakan filter aplikasi, sentimen, rating, pencarian kata, atau checkbox konflik label untuk menelusuri ulasan. Grafik dan tabel mengikuti filter aktif. Tabel ulasan dapat dicari dan ditampilkan per halaman.

**Konflik label** berarti teks `content_clean` yang identik memiliki label sentimen berbeda. Contohnya, teks yang sama muncul pada dua ulasan dengan rating berbeda. Label mengikuti rating; konflik ini tidak berarti teksnya memiliki makna yang mirip saja.

### 3. Performa Model

Membandingkan hasil evaluasi holdout untuk Multinomial Naive Bayes, Random Forest, dan Linear SVM. Pilih model untuk melihat confusion matrix dan metrik tiap kelas.

- **Accuracy** adalah proporsi seluruh prediksi yang benar.
- **Precision** menunjukkan ketepatan prediksi untuk suatu kelas.
- **Recall** menunjukkan seberapa banyak data aktual dari suatu kelas yang berhasil ditemukan.
- **F1** merangkum precision dan recall.
- **Macro-F1** merata-ratakan F1 kelas secara setara, sehingga kelas yang jumlahnya kecil seperti Netral tetap diperhitungkan.
- **Support** adalah jumlah data aktual pada kelas tersebut.

Confusion matrix memperlihatkan jumlah ulasan untuk setiap kombinasi label aktual dan prediksi. Nilai diagonal adalah prediksi benar; nilai di luar diagonal menunjukkan kelas yang tertukar.

### 4. Analisis Lintas Aplikasi (LOAO)

Halaman ini menguji kemampuan transfer model ke aplikasi yang ulasannya tidak digunakan saat pelatihan. Dalam setiap putaran, satu aplikasi menjadi target pengujian dan aplikasi lain menjadi sumber data latih. Proses diulang untuk setiap aplikasi. Grup teks duplikat yang muncul pada aplikasi target dikeluarkan dari data latih untuk menghindari kebocoran teks identik.

- **LOAO alami** mengevaluasi seluruh ulasan pada aplikasi target.
- **LOAO ketat** mengevaluasi ulasan target yang teksnya hanya muncul di satu aplikasi (`n_apps_in_group == 1`). Ini memeriksa kinerja pada teks yang unik untuk satu aplikasi.
- **Baseline dalam domain** adalah nilai pembanding dari evaluasi holdout dengan `GroupShuffleSplit`. Karena cara pembagian datanya berbeda dari LOAO, gunakan baseline sebagai referensi, bukan sebagai perbandingan yang sepenuhnya setara.

Eksperimen LOAO di notebook menggunakan Linear SVM. Predictor di halaman kelima menggunakan model yang tersimpan di bundle, yaitu MultinomialNB. Kedua halaman menjawab tujuan yang berbeda.

### 5. Prediksi Sentimen

Masukkan teks ulasan untuk mendapatkan prediksi sentimen dari model bundle. Contoh ulasan dapat dipilih sebagai masukan awal. Halaman ini juga menampilkan teks setelah preprocessing dan informasi fitur teks yang dikenali.

Prediksi adalah keluaran model, bukan verifikasi manual atas maksud penulis. Skor probabilitas menunjukkan estimasi model dan tidak menjamin kebenaran label.

## Cara membaca label dan preprocessing

Label dataset diturunkan dari rating pengguna:

| Rating | Label |
| --- | --- |
| 1–2 | Negatif |
| 3 | Netral |
| 4–5 | Positif |

Preprocessing membersihkan teks—misalnya menyeragamkan huruf, menangani URL dan tanda baca, serta menormalkan sejumlah slang. Proses ini membuat teks lebih konsisten untuk analisis; proses tersebut **tidak mengubah rating atau label sentimen** pada dataset.

## Sumber hasil dan dokumentasi

Dashboard memuat hasil evaluasi yang sudah disimpan dari eksperimen di notebook Member 2. Notebook tidak perlu dijalankan untuk membuka dashboard, dan dashboard tidak melatih ulang model saat dijalankan.

- [Notebook preprocessing](notebooks/preprocessing.ipynb)
- [Notebook pelatihan dan evaluasi model](notebooks/Model_Training_Rating_Classification.ipynb)
- [Laporan preprocessing](docs/preprocessing_report.md)
- [Audit data](docs/data_audit.md)
- [Catatan sumber artefak dashboard](docs/dashboard_artifact_provenance.md)

Catatan provenance menjelaskan sumber file evaluasi yang digunakan serta batas verifikasi asal bundle `.pkl` yang saat ini tersedia. Untuk memahami hasil penelitian, gunakan notebook dan laporan terkait sebagai rujukan metodologi; dashboard berfungsi sebagai antarmuka untuk membaca dan menjelajahi hasil tersebut.
