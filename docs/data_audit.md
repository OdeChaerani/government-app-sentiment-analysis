# Laporan Audit Integritas Data (Data Audit Report)
**Proyek:** Klasifikasi Sentimen Ulasan Aplikasi Pemerintah Indonesia (Dataset IGAR)  
**Peran:** Anggota 1 - Data Preprocessing  
**Tahap:** Tahap 1 - Audit, Inspeksi Skema, dan Penilaian Risiko Metodologi  
**Tanggal Audit:** 29 Oktober 2026  
**File Raw yang Diaudit:** `data/Rating_labeled_raw.csv`  

---

## 1. Ringkasan Eksekutif (Executive Summary)

Audit ini dilakukan secara independen terhadap dataset mentah (*raw data*) `data/Rating_labeled_raw.csv` tanpa melakukan modifikasi, mutasi, atau penghapusan data. Tujuan audit adalah untuk:
1. Menjamin integritas skema data dan memverifikasi kesesuaian mapping rating terhadap label sentimen.
2. Memetakan distribusi kelas, ketimpangan volume data antar aplikasi, dan karakteristik teks ulasan.
3. Mengidentifikasi masalah krusial berupa duplikasi teks, kontradiksi label (label noise), dan potensi kebocoran data (*data leakage*) yang berisiko merusak validitas eksperimen, khususnya pengujian generalisasi lintas aplikasi (*cross-app generalization / leave-one-app-out*).

### Ringkasan Angka Kunci:
- **Total Baris:** 617.722 baris ulasan
- **Total Kolom:** 8 kolom
- **Jumlah Aplikasi:** 6 aplikasi instansi/BUMN pemerintah
- **Jumlah Teks Ulasan Unik:** 383.614 ulasan unik (37,90% adalah duplikat teks)
- **Teks Ulasan Lintas Aplikasi:** 8.860 teks unik muncul di lebih dari 1 aplikasi (berdampak pada 234.809 baris / 38,01% dataset)
- **Kontradiksi Label pada Teks Identik:** 3.159 teks unik memiliki label sentimen yang saling bertolak belakang (berdampak pada 202.586 baris / 32,79% dataset)
- **Ulasan Sangat Pendek (≤ 3 Kata):** 299.839 ulasan (48,54% dari seluruh dataset)

---

## 2. Profil Skema dan Integritas Struktur Data

### 2.1. Metadata Kolom dan Tipe Data
Dataset mentah memiliki 8 kolom dengan profil sebagai berikut:

| Nama Kolom | Tipe Data Pandas | Non-Null Count | Null Count | Null Pct (%) | Deskripsi & Catatan Integritas |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Unnamed: 0` | `int64` | 617.722 | 0 | 0.0000% | Berfungsi sebagai ID baris indeks unik (rentang 0 s/d 617.721). Terbukti 100% unik. |
| `app` | `object` (str) | 617.722 | 0 | 0.0000% | Identifier nama aplikasi pemerintah. Tidak ada missing values. |
| `content` | `object` (str) | 617.696 | 26 | 0.0042% | Teks asli ulasan pengguna dalam bahasa Indonesia. Terdapat 26 baris kosong (NaN). |
| `translation` | `object` (str) | 609.947 | 7.775 | 1.2587% | Terjemahan ulasan ke bahasa Inggris. Terdapat 7.775 baris bernilai NaN. |
| `score` | `int64` | 617.722 | 0 | 0.0000% | Skor rating Google Play Store (skala 1 sampai 5). |
| `at` | `object` (str) | 617.722 | 0 | 0.0000% | Timestamp pengiriman ulasan (`YYYY-MM-DD HH:MM:SS`). |
| `appVersion` | `object` (str) | 527.319 | 90.403 | 14.6349% | Versi aplikasi pengguna. Memiliki 204 versi unik, dengan 14,63% nilai kosong. |
| `labelScoreBase` | `object` (str) | 617.722 | 0 | 0.0000% | Label sentimen target bawaan dataset (`Positive`, `Negative`, `Neutral`). |

### 2.2. Temuan Baris Kosong pada Fitur Teks (`content`)
Ditemukan **26 baris** di mana kolom `content` bernilai `NaN`. Ulasan-ulasan ini memiliki rating dan timestamp yang sah (tersebar di BMKG: 8, KAI: 11, pertamina: 3, satusehat: 3, mobileJKN: 1), tetapi tidak memiliki teks sama sekali.
- **Rekomendasi:** Baris dengan `content` kosong wajib di-drop pada pipeline preprocessing karena tidak dapat diekstrak fiturnya oleh representasi TF-IDF.

---

## 3. Analisis Rating dan Pemetaan Label Sentimen

### 3.1. Distribusi Skor Rating (`score`)
Skor ulasan menunjukkan pola distribusi **bipolar (U-shaped)** yang sangat umum pada toko aplikasi: mayoritas pengguna hanya memberikan rating saat sangat puas (bintang 5) atau sangat kecewa (bintang 1).

| Rating (`score`) | Frekuensi (Baris) | Persentase (%) | Visualisasi Proporsi |
| :---: | :---: | :---: | :--- |
| **1** | 216.959 | 35,12% | ███████████████ |
| **2** | 29.939 | 4,85% | ██ |
| **3** | 34.375 | 5,56% | ██ |
| **4** | 45.280 | 7,33% | ███ |
| **5** | 291.169 | 47,14% | ████████████████████ |
| **Total** | **617.722** | **100,00%** | Mean: 3,27 \| Median: 4,0 |

### 3.2. Validasi Pemetaan Label Sentimen (`labelScoreBase`)
Rencana penelitian menetapkan pemetaan rating sebagai berikut:
- **Rating 1–2:** `Negative`
- **Rating 3:** `Neutral`
- **Rating 4–5:** `Positive`

Hasil uji silang (*cross-tabulation*) antara `score` dan `labelScoreBase` pada data mentah:

| Rating (`score`) | Negative | Neutral | Positive | Total |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 216.959 | 0 | 0 | 216.959 |
| **2** | 29.939 | 0 | 0 | 29.939 |
| **3** | 0 | 34.375 | 0 | 34.375 |
| **4** | 0 | 0 | 45.280 | 45.280 |
| **5** | 0 | 0 | 291.169 | 291.169 |
| **Total** | **246.898 (39,97%)** | **34.375 (5,56%)** | **336.449 (54,47%)** | **617.722 (100%)** |

> **Temuan Kunci Integritas:**  
> Ditemukan **0 perbedaan (discrepancy = 0)** antara pemetaan matematis yang direncanakan dengan kolom `labelScoreBase`. Kolom `labelScoreBase` pada dataset raw secara deterministik diturunkan langsung dari nilai kolom `score`.
>
> **Tantangan Modeling (Class Imbalance):**  
> Kelas `Neutral` mengalami ketimpangan kelas yang sangat parah (**hanya 5,56%** dari total data). Jika model dilatih tanpa pembobotan kelas (*class weights*) atau strategi penyeimbangan, model Naive Bayes dan Linear SVM berisiko mengalami degradasi recall tajam pada kelas netral.

---

## 4. Distribusi Lintas Aplikasi dan Ketimpangan Volume

### 4.1. Ketimpangan Volume Data Antar Aplikasi
Terdapat perbedaan volume ulasan yang sangat ekstrem antar keenam aplikasi pemerintah:

| Nama Aplikasi | Jumlah Ulasan | Pangsa Dataset (%) | Rentang Tanggal Ulasan (Timestamp) |
| :--- | :---: | :---: | :--- |
| **JMO** | 200.000 | 32,38% | 2021-09-02 s/d 2023-03-26 |
| **satusehat** | 190.696 | 30,87% | 2020-03-31 s/d 2021-10-14 |
| **mobileJKN** | 122.103 | 19,77% | 2016-06-07 s/d 2021-11-05 |
| **pertamina** | 96.291 | 15,59% | 2017-08-09 s/d 2022-07-02 |
| **KAI** | 7.311 | 1,18% | 2014-07-16 s/d 2017-12-26 |
| **BMKG** | 1.321 | 0,21% | 2012-05-11 s/d 2015-12-13 |
| **Total** | **617.722** | **100,00%** | **2012-05-11 s/d 2023-03-26** |

**Implikasi Penting:**
1. Dua aplikasi (JMO dan satusehat) menguasai **63,25%** dari keseluruhan dataset.
2. BMKG hanya memiliki 1.321 baris (0,21%), sementara JMO memiliki 200.000 baris (rasio 1:151).
3. **Pergeseran Waktu (Temporal Shift):** Ulasan BMKG dan KAI berasal dari rentang tahun 2012–2017, sedangkan satusehat dan JMO terkonsentrasi pada era pandemi dan pasca-pandemi (2020–2023). Gaya bahasa, kosakata istilah, dan slang internet mengalami pergeseran di antara rentang waktu ini.

### 4.2. Distribusi Sentimen per Aplikasi (Pergeseran Distribusi / Prior Shift)
Tabel silang persentase sentimen per aplikasi:

| Nama Aplikasi | Negative (%) | Neutral (%) | Positive (%) | Dominasi Sentimen Lokal |
| :--- | :---: | :---: | :---: | :--- |
| **BMKG** | 8,40% | 7,49% | **84,10%** | Sangat Positif |
| **JMO** | 22,16% | 4,33% | **73,51%** | Sangat Positif |
| **mobileJKN** | 38,64% | 7,04% | **54,32%** | Cenderung Positif |
| **satusehat** | 42,44% | 6,85% | **50,70%** | Berimbang Positif-Negatif |
| **KAI** | 43,85% | 10,67% | **45,48%** | Berimbang |
| **pertamina** | **73,88%** | 3,30% | 22,83% | **Sangat Negatif** |

> **Peringatan Metodologis untuk Evaluasi Cross-App (Leave-One-App-Out):**  
> Perbedaan prior distribution ini sangat krusial. Jika model dilatih pada seluruh aplikasi selain `pertamina` (di mana mayoritas sentimen bernilai positif), dan kemudian diuji pada `pertamina` (yang 73,88% ulasannya negatif akibat kebijakan pembelian BBM bersubsidi), akurasi model baseline akan anjlok drastis bukan karena representasi teks semata, melainkan karena *prior probability shift*. Evaluasi harus memprioritaskan metrik **Macro F1-Score** dan **Balanced Accuracy**, bukan Accuracy biasa.

---

## 5. Audit Duplikasi Teks dan Fenomena Overlap Antar Aplikasi

### 5.1. Analisis Duplikasi
- **Duplikat Baris Penuh (seluruh 8 kolom):** 0 baris (tidak ada baris yang kembar identik termasuk indeks).
- **Duplikat Baris (tanpa kolom `Unnamed: 0`):** 38 baris (19 pasangan review duplikat persis di seluruh teks, timestamp, rating, dan metadata).
- **Duplikat Fitur Teks (`content`):** Dari 617.722 baris ulasan, hanya terdapat **383.614 teks unik**. Artinya terdapat **234.107 baris ulasan duplikat (37,90%)**!

### 5.2. Overlap Teks Lintas Aplikasi (Cross-App Text Overlap)
Dari ulasan unik yang ada, sebanyak **8.860 teks unik** muncul di lebih dari satu aplikasi, mencakup **234.809 baris (38,01% dari total dataset)**.

Rincian keterlibatan ulasan lintas aplikasi per jumlah aplikasi yang memuat teks yang sama:
- Muncul di 2 aplikasi: 5.503 teks unik
- Muncul di 3 aplikasi: 1.931 teks unik
- Muncul di 4 aplikasi: 1.066 teks unik
- Muncul di 5 aplikasi: 272 teks unik
- Muncul di seluruh 6 aplikasi: 88 teks unik

Proporsi baris pada masing-masing aplikasi yang memuat teks lintas aplikasi:
- **JMO:** 97.378 baris (48,69% dari review JMO)
- **satusehat:** 62.519 baris (32,78% dari review satusehat)
- **mobileJKN:** 48.335 baris (39,59% dari review mobileJKN)
- **pertamina:** 24.209 baris (25,14% dari review pertamina)
- **KAI:** 1.733 baris (23,70% dari review KAI)
- **BMKG:** 635 baris (48,07% dari review BMKG)

Karakteristik teks ulasan yang paling sering muncul di keenam aplikasi:
1. Ulasan 1 kata generik: `Ok` (17.466 baris), `Bagus` (13.118 baris), `Mantap` (11.116 baris), `Good` (9.586 baris), `ok` (9.085 baris), `bagus` (8.540 baris), `mantap` (8.047 baris), `good` (5.748 baris).
2. Frasa 2 kata generik: `Sangat membantu` (8.525 baris), `sangat membantu` (5.943 baris).

---

## 6. Inkonsistensi Label dan Anomali Penilaian (Label Noise)

Karena label sentimen diturunkan secara kaku dari rating bintang (`score`), timbul fenomena **label contradiction** pada teks yang sama persis:

### 6.1. Statistik Inkonsistensi
- **Teks unik dengan Rating Berbeda:** **5.579 teks unik** memiliki rating bintang yang berbeda-beda, mencakup **220.369 baris ulasan**.
- **Teks unik dengan Label Sentimen Bertolak Belakang:** **3.159 teks unik** memiliki label sentimen yang bertentangan (misal sekaligus berlabel Positive dan Negative), mencakup **202.586 baris ulasan (32,79% dataset)**.
- **Konflik Label dalam Aplikasi yang Sama:** Terdapat **3.240 pasangan (aplikasi, teks)** yang saling berkonflik label di dalam aplikasi yang sama, mencakup **183.585 baris**.

### 6.2. Studi Kasus Konflik Label pada Teks Populer:
| Teks Ulasan | Total Kemunculan | Sebaran Rating (`score`) | Sebaran Label Sentimen | Analisis Fenomena |
| :--- | :---: | :--- | :--- | :--- |
| `Ok` | 17.466 | Bintang 5: 14.811<br>Bintang 4: 1.756<br>Bintang 3: 564<br>Bintang 1: 221<br>Bintang 2: 114 | **Positive:** 16.567<br>**Neutral:** 564<br>**Negative:** 335 | Teks sama persis memiliki label berlawanan karena pengguna memberi bintang 1 untuk kata "Ok". |
| `Bagus` | 13.118 | Bintang 5: 10.888<br>Bintang 4: 1.504<br>Bintang 3: 444<br>Bintang 1: 185<br>Bintang 2: 97 | **Positive:** 12.392<br>**Neutral:** 444<br>**Negative:** 282 | Kemungkinan salah tekan rating bintang 1 padahal bermaksud memuji, atau sarkasme. |
| `Mantap` | 11.116 | Bintang 5: 10.134<br>Bintang 4: 666<br>Bintang 3: 172<br>Bintang 1: 103<br>Bintang 2: 41 | **Positive:** 10.800<br>**Neutral:** 172<br>**Negative:** 144 | Teks bernada positif kuat berlabel Negative pada 144 baris. |

**Dampak Serius pada Model Pembelajaran Mesin:**
1. **Multinomial Naive Bayes:** Probabilitas kondisional $P(w \mid c)$ akan terdistorsi karena fitur unigram `bagus` atau `ok` mendapatkan count di kelas Negative dan Neutral.
2. **Linear SVM:** Vektor bobot hyperplane akan kesulitan memisahkan ruang fitur karena titik data yang identik memiliki target kelas $y \in \{-1, 0, 1\}$ yang saling bertabrakan, memaksa slack variables ($\xi_i$) membesar dan mereduksi margin optimal.
3. **Random Forest:** Penurunan impuritas Gini / Entropi pada pemisahan node kata-kata pendek ini tidak akan pernah mencapai kondisi murni (*pure leaf*).

---

## 7. Karakteristik Panjang Teks dan Profil Noise NLP

### 7.1. Statistik Panjang Teks (Panjang Kata dan Karakter)
Analisis dilakukan terhadap kolom teks asli (`content`):

| Metrik Statistik | Jumlah Kata per Review | Jumlah Karakter per Review |
| :--- | :---: | :---: |
| **Minimum** | 0 kata | 0 karakter |
| **Kuartil 1 (Q1 - 25%)** | 1 kata | 8 karakter |
| **Median (50%)** | 4 kata | 24 karakter |
| **Rata-rata (Mean)** | 8,42 kata | 52,66 karakter |
| **Kuartil 3 (Q3 - 75%)** | 11 kata | 69 karakter |
| **Maksimum** | 283 kata | 1.812 karakter |

**Fenomena Ulasan Ultra-Pendek:**
- Ulasan tepat **1 kata:** 166.517 baris (**26,96%**)
- Ulasan tepat **2 kata:** 88.133 baris (**14,27%**)
- Ulasan tepat **3 kata:** 45.163 baris (**7,31%**)
- **Akumulasi Ulasan ≤ 3 kata:** **299.839 baris (48,54% dari seluruh dataset)**

### 7.2. Deteksi Pola Noise Teks Spesifik
- **Ulasan Hanya Berisi Tanda Baca (*Punctuation-Only*):** **7.748 baris**. Ulasan seperti `...`, `????`, `---`, `!`, atau `.` yang tidak memiliki nilai semantik sama sekali. Jika tanda baca dihapus saat preprocessing, baris-baris ini akan menjadi string kosong.
- **Ulasan Hanya Berisi Angka (*Digits-Only*):** 2 baris (misal `123`).
- **Ulasan Mengandung Emoji / Karakter Non-ASCII:** **39.866 baris (6,45%)**. Karakter emoji memuat muatan emosional kuat (misal 👍, ⭐, 😡, 🤮) yang perlu diputuskan penanganannya (dihapus vs dikonversi ke token teks/deskripsi).
- **Pengulangan Karakter Berlebih (*Character Elongation* ≥ 4 kali):** **33.421 baris**. Contoh: `baguuuus`, `leleeeet`, `mantappppp`. Perlu normalisasi agar tidak memecah kosakata TF-IDF.
- **Huruf Kapital Penuh (*All-Caps* > 3 karakter):** **4.989 baris**. Sering mencerminkan kemarahan/kekesalan pengguna (misal: `APLIKASI GAGAL TERUS TIDAK BISA LOGIN`). Case folding akan mengubahnya ke lowercase, namun informasi intensitasnya dapat hilang.
- **Email Address:** 69 baris.
- **URL Link / Tag HTML:** 0 baris (bersih dari tautan eksternal).

---

## 8. Analisis Potensi Data Leakage pada Desain Eksperimen

Proyek penelitian merencanakan dua skenario evaluasi:
1. **Random Split (In-Domain)**
2. **Leave-One-App-Out (Cross-App Generalization)**

### 8.1. Bahaya Kebocoran Data (*Data Leakage*) pada Random Split
Jika dataset dibagi secara acak (*random train-test split*, misal 80:20) tanpa memperhatikan duplikasi teks:
- Dari 234.809 baris teks duplikat dan 8.860 teks yang tersebar di banyak aplikasi, sebagian besar teks yang identik (seperti `Bagus`, `Mantap`, `Aplikasi sering keluar sendiri`) akan masuk ke data latih (train) sekaligus data uji (test).
- Model tidak menggeneralisasi bahasa, melainkan menghafal (*memorizing*) kata-kata tersebut.
- Hal ini menyebabkan evaluasi akurasi in-domain menjadi **artifisial tinggi (terlalu optimis/over-optimistic)** dan tidak mencerminkan performa model di dunia nyata.

### 8.2. Bahaya Evaluasi pada Cross-App (Leave-One-App-Out)
Pada evaluasi cross-app di mana 1 aplikasi dijadikan test set dan 5 aplikasi lainnya dijadikan train set:
- **Shortcut Learning:** Model akan mengandalkan kata-kata generik lintas aplikasi (`Ok`, `Mantap`) dan gagal menangkap entitas spesifik aplikasi (misal istilah `pedulilindungi`/`sertifikat vaksin` di satusehat, `klaim jht` di JMO, `rujukan faskes` di mobileJKN, atau `solar subsisi`/`barcode` di pertamina).
- **Prior Probability Shift Ekstrem:** Saat menguji pada aplikasi `pertamina` (73,88% Negatif) menggunakan model yang dilatih pada mayoritas aplikasi positif, model akan menghasilkan prediksi yang bias positif jika tidak diterapkan kompensasi probabilitas prior atau pembobotan kelas.

---

## 9. Rekomendasi Penanganan Preprocessing (Preprocessing Roadmap)

Berdasarkan temuan audit empiris, berikut adalah tahapan rekomendasi pipeline preprocessing untuk Anggota 1:

### Tahap A: Data Cleansing & Filtering (Tahap Kebersihan Dasar)
1. **Hapus Baris Teks Kosong (Null Removal):** Drop 26 baris yang bernilai `NaN` pada kolom `content`.
2. **Hapus Ulasan Non-Semantik (Punctuation-Only & Digits-Only):** Drop 7.748 baris yang hanya berisi tanda baca atau angka murni, karena setelah *text cleaning* baris tersebut menjadi string hampa tanpa fitur.
3. **Standardisasi ID & Kolom:** Buat kolom `review_id` yang bersih, pertahankan metadata `app`, `score`, `at`, dan `labelScoreBase`.

### Tahap B: Text Normalization (Standardisasi Teks Bahasa Indonesia)
1. **Case Folding:** Konversi seluruh teks ke huruf kecil (*lowercase*).
2. **Emoji Handling:** Pilihan antara:
   - *Opsi 1 (Direkomendasikan):* Konversi emoji populer ke kata padanan bahasa Indonesia (misal: 👍 $\rightarrow$ `bagus`, 😡 $\rightarrow$ `marah`) menggunakan kamus representasi sebelum pembersihan tanda baca.
   - *Opsi 2:* Menghapus seluruh karakter non-ASCII.
3. **Normalisasi Elongasi Karakter (*De-elongation*):** Reduksi karakter yang berulang lebih dari 2 kali menjadi maksimal 2 karakter (misal: `baguuuus` $\rightarrow$ `bagus`, `leleeeet` $\rightarrow$ `leet`/`lelet`).
4. **Pembersihan Tanda Baca & Whitespace:** Hapus tanda baca berlebih dan spasi ganda.
5. **Kamus Slang/Singkatan (Slang Normalization):** Mengingat ulasan aplikasi Indonesia sangat kaya bahasa gaul/singkatan (misal: `tdk` $\rightarrow$ `tidak`, `bgt` $\rightarrow$ `banget`, `bisaa` $\rightarrow$ `bisa`, `gak`/`ga` $\rightarrow$ `tidak`), disarankan menerapkan mapping kamus normalisasi informal Indonesia.
6. **Stopwords & Stemming (Pertimbangan Hati-hati):**
   - **Stopwords:** Jangan menghapus kata negasi seperti `tidak`, `bukan`, `jangan`, `kurang`, `belum`, karena penghapusan negasi akan membalikkan polaritas sentimen ulasan.
   - **Stemming:** Stemming Bahasa Indonesia (Sastrawi) sangat lambat pada 600.000+ baris dan sering merusak imbuhan yang membedakan sentimen. Sebaiknya dihindari atau dibatasi pada proses unigram/bigram TF-IDF.

### Tahap C: Feature Extraction (TF-IDF untuk Anggota 2)
1. Gunakan konfigurasi N-gram: unigram + bigram (`ngram_range=(1, 2)`) untuk menangkap frasa negasi seperti "tidak bisa", "kurang bagus", "sangat membantu".
2. Batasi `min_df=5` atau `min_df=10` untuk mengeliminasi typo langka dan membatasi dimensi matriks TF-IDF agar efisien saat ditraining oleh Anggota 2 (Linear SVM, MNB, Random Forest).
3. Batasi `max_features` (misal 10.000 s/d 20.000 fitur) untuk mencegah OOM (*out of memory*) pada RAM lokal.

---

## 10. Keputusan Metodologis yang Memerlukan Konfirmasi Tim

Sebelum Anggota 1 mengeksekusi pipeline preprocessing final dan mengekspor dataset siap pakai, terdapat 5 keputusan metodologi penting yang harus disepakati bersama:

### 1. Strategi Mitigasi Data Leakage pada Random Split
- **Opsi A (Strict Deduplication):** Menghapus seluruh duplikat teks `content` sehingga tiap teks unik hanya muncul 1 kali dalam dataset. (Dataset berkurang dari ~617 ribu menjadi ~383 ribu baris).
- **Opsi B (Group-based Split):** Melakukan train-test split dengan `GroupShuffleSplit` berdasarkan `content`, sehingga ulasan identik dipastikan hanya berada di train set atau test set saja (tidak bocor ke keduanya).
- **Opsi C (Per-App Deduplication):** Hanya menghapus duplikat teks di dalam aplikasi yang sama, tetapi membiarkan duplikat lintas aplikasi untuk keperluan pengujian cross-app.

### 2. Penanganan Konflik Label pada Teks Identik (Label Noise)
Untuk 3.159 teks unik (melibatkan ~202 ribu baris) yang memiliki label bertentangan (misal teks "Bagus" berlabel Negatif pada 282 baris):
- **Opsi A (Majority Voting):** Menetapkan label seragam berdasarkan label mayoritas untuk teks tersebut (misal kata "Bagus" diseragamkan 100% menjadi `Positive`).
- **Opsi B (Filter/Remove Ambiguous Contradictions):** Menghapus baris minoritas yang anomali (misal membuang 282 baris "Bagus" yang berlabel Negatif karena terindikasi salah rating/sarkasme tak terdeteksi).
- **Opsi C (Keep Raw Labels):** Membiarkan label apa adanya sesuai rating pengguna asli.

### 3. Ambang Batas Panjang Teks Minimum
Mengingat 26,96% data hanya berupa 1 kata generik ("Ok", "Bagus", "Mantap"):
- **Opsi A:** Pertahankan seluruh ulasan selama memiliki minimal 1 karakter alfanumerik.
- **Opsi B:** Terapkan filter panjang minimum $\ge 2$ kata atau $\ge 3$ kata untuk menyaring ulasan tidak informatif.

### 4. Strategi Mengatasi Ketimpangan Volume Aplikasi (App Imbalance)
Aplikasi JMO (200.000) dan satusehat (190.696) jauh melampaui BMKG (1.321) dan KAI (7.311):
- **Opsi A (Subsampling Seimbang per Aplikasi):** Mengambil sampel seimbang per aplikasi (misal 1.000 s/d 5.000 baris per aplikasi) khusus untuk evaluasi komparatif cross-app yang adil.
- **Opsi B (Full Dataset with Stratified Weights):** Menggunakan seluruh dataset apa adanya, namun Anggota 2 wajib menyertakan bobot kelas/aplikasi saat training.

### 5. Pilihan Fitur Teks Sumber
Dataset menyediakan `content` (Bahasa Indonesia) dan `translation` (Bahasa Inggris):
- **Opsi A (Direkomendasikan):** Menggunakan `content` asli bahasa Indonesia, karena representasi asli mencerminkan karakteristik riil ulasan masyarakat Indonesia.
- **Opsi B:** Menggunakan `translation` bahasa Inggris (catatan: terdapat 7.775 nilai null pada translation).
