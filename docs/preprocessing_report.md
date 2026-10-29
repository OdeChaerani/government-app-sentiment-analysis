# Preprocessing Report - IGAR Sentiment Classification

**Tanggal Pembaruan:** 2026-10-30  
**File Input:** `data/Rating_labeled_raw.csv`  
**File Output:** `data/Rating_labeled_processed.csv`  
**Notebook:** `notebooks/preprocessing.ipynb`  

---

## 1. Ringkasan Pipeline & Dataset

| Metrik | Nilai | Persentase |
| :--- | :--- | :--- |
| **Baris Raw Asli** | 617,722 | 100.00% |
| **Baris Processed Siap Pakai** | 609,900 | 98.73% |
| **Baris Difilter (Kosong/Non-usable)** | 7,822 | 1.27% |
| **Total Teks Unik (`duplicate_group`)** | 353,798 | - |
| **Grup Singleton (1 baris / teks unik)** | 340,859 | 55.89% baris |
| **Grup Duplikat (>1 baris)** | 12,939 | 44.11% baris (269,041 baris) |
| **Grup dengan Konflik Label** | 3,293 | 37.11% baris (226,323 baris) |
| **Grup Lintas Aplikasi (>1 app)** | 8,727 | 42.04% baris (256,423 baris) |

---

## 2. Alasan Filtering Baris

Sebanyak **7,822 baris (1.27%)** difilter keluar dari dataset processed karena tidak mengandung teks alfabetik yang dapat diekstraksi sebagai fitur NLP:
- **`content_original` NULL:** 26 baris (kosong dari data mentah asli).
- **`content_clean` kosong setelah pembersihan:** 7,796 baris (ulasan yang hanya berisi emoji tanpa kata, tanda baca murni seperti `...` / `???`, atau simbol non-alfanumerik).
- **Penyelamatan Font Stylized:** Berkat penambahan normalisasi Unicode (`NFKD`) di tahap awal cleaning, sebanyak **113 ulasan valid** yang ditulis menggunakan variasi font Unicode (Mathematical Bold, Sans, Italic) berhasil dipertahankan ke dalam teks Latin standar dan tidak ikut terbuang.

---

## 3. Langkah Pembersihan Teks (*Text Cleaning Pipeline*)

Pipeline pembersihan diterapkan secara konservatif dengan urutan sebagai berikut:

1. **Dekomposisi Unicode (NFKD):** Mengubah karakter variasi font khusus (misal `𝐌𝐚𝐧𝐭𝐚𝐩` atau `𝕽𝖎𝖇𝖊𝖙`) menjadi huruf Latin standar (`mantap`, `ribet`).
2. **Case Folding:** Seluruh huruf dikonversi ke huruf kecil (*lowercase*).
3. **Pembersihan URL & Email:** Menghapus tautan web (`http`, `https`, `www`) dan alamat email.
4. **Pembersihan Karakter Non-Alfanumerik & Emoji:**
   - Menggunakan regex `[^a-z0-9\s]`.
   - **Keputusan Terkait Emoji:** Emoji **dihapus langsung (*stripped*) tanpa dipetakan ke kata teks**. Berdasarkan audit data, pemetaan emoji rentan menyebabkan inversi semantik (misal 74.9% emoji 😂/🤣 muncul pada rating 1 bernada sarkasme cemoohan, dan 52.7% emoji 🙏 muncul pada permohonan/keluhan).
5. **Reduksi Elongasi Karakter:** Karakter yang berulang 3 kali atau lebih berturut-turut direduksi menjadi 2 karakter (misal `baguuus` $\rightarrow$ `baguus`).
6. **Normalisasi Slang/Singkatan:** Mengganti kata tidak baku menggunakan kamus terdokumentasi (kata `'dr'` dipertahankan tanpa diubah untuk mencegah konflik makna dengan kata *dokter* pada aplikasi kesehatan).
7. **Normalisasi Whitespace:** Menghapus spasi ganda dan *trimming* spasi di awal/akhir teks.

---

## 4. Dua Varian Fitur Teks untuk Model Training

Dataset menyediakan dua varian teks fitur yang dapat digunakan oleh Anggota 2:

1. **`content_clean` (Fitur Utama):**
   - Teks hasil cleaning tanpa stopword removal.
   - Kata negasi (`tidak`, `bukan`, `jangan`, `belum`, `kurang`) serta preposisi tetap utuh.
   - Sangat direkomendasikan untuk TF-IDF N-gram (Unigram + Bigram) agar frasa negasi seperti `"tidak bagus"` atau `"kurang puas"` tidak terpecah.

2. **`content_clean_stopword` (Fitur Sekunder):**
   - Teks hasil pembersihan lanjutan di mana kata-kata umum (*stopwords*) non-esensial dihapus.
   - **Aturan Sentimen Kritis:** Kata negasi (`tidak`, `bukan`, `jangan`, `belum`, `kurang`) dan kata penguat/keterangan esensial **sengaja DIKECUALIKAN dari daftar stopwords** agar tidak membalikkan makna sentimen.

---

## 5. Distribusi Label & Aplikasi Sesudah Preprocessing

### Distribusi Label:
| Label | Jumlah Baris | Persentase |
| :--- | :--- | :--- |
| **Positive** (score 4–5) | 329,707 | 54.06% |
| **Negative** (score 1–2) | 246,033 | 40.34% |
| **Neutral** (score 3) | 34,160 | 5.60% |
| **Total** | **609,900** | **100.00%** |

### Distribusi Aplikasi:
| Aplikasi | Jumlah Baris | Persentase |
| :--- | :--- | :--- |
| **JMO** | 197,216 | 32.34% |
| **satusehat** | 188,100 | 30.84% |
| **mobileJKN** | 120,497 | 19.76% |
| **pertamina** | 95,494 | 15.66% |
| **KAI** | 7,285 | 1.19% |
| **BMKG** | 1,308 | 0.21% |
| **Total** | **609,900** | **100.00%** |

---

## 6. Spesifikasi Kolom Dataset `data/Rating_labeled_processed.csv`

| No | Kolom | Tipe Data | Deskripsi |
| :---: | :--- | :---: | :--- |
| 1 | `review_id` | int | ID unik ulasan (berasal dari indeks raw asli). |
| 2 | `app` | str | Nama instansi/aplikasi pemerintah. |
| 3 | `content_original` | str | Teks ulasan mentah asli sebelum dibersihkan. |
| 4 | `content_clean` | str | **Fitur teks utama (tanpa stopword removal).** |
| 5 | `content_clean_stopword` | str | **Fitur teks sekunder (stopwords umum dihapus, negasi dipertahankan).** |
| 6 | `score` | int | Rating asli pengguna (1–5). |
| 7 | `label` | str | Target klasifikasi: `Negative`, `Neutral`, `Positive`. |
| 8 | `at` | str | Timestamp review. |
| 9 | `duplicate_group` | int | ID grup unik untuk teks `content_clean` yang identik. |
| 10 | `group_size` | int | Frekuensi kemunculan teks tersebut di seluruh dataset. |
| 11 | `has_label_conflict` | bool | `True` jika teks identik memiliki label yang berbeda. |
| 12 | `n_apps_in_group` | int | Jumlah aplikasi berbeda yang memuat teks identik tersebut. |

---

## 7. Rekomendasi Metodologis untuk Anggota 2 (Model Training)

1. **Pencegahan Data Leakage pada Random Split:**
   - Sebanyak 44.11% baris memiliki teks duplikat.
   - **Wajib menggunakan `GroupShuffleSplit(groups=df['duplicate_group'])`** jika melakukan train-test split acak, agar teks yang sama tidak berada di train dan test secara bersamaan.
2. **Evaluasi Cross-App / Leave-One-App-Out (LOAO):**
   - Sebanyak 42.04% baris berbagi teks antar aplikasi (misal ulasan pendek seperti `"bagus"`, `"update terus"`, `"gagal login"`).
   - Lakukan evaluasi dua tingkat:
     - **Tingkat 1:** Evaluasi LOAO pada seluruh ulasan aplikasi target.
     - **Tingkat 2 (Strict Cross-App):** Evaluasi LOAO khusus pada subset `n_apps_in_group == 1` untuk mengukur daya generalisasi model murni terhadap kosakata unik domain yang belum pernah dilihat di training set.
3. **Penanganan Label Noise:**
   - Terdapat 226,323 baris dengan teks identik namun label bertentangan (misal teks `"ok"` pada rating 5 vs rating 1).
   - Anggota 2 dapat mempertimbangkan eksperimen perbandingan: melatih model pada seluruh data vs memfilter baris dengan `has_label_conflict == True`.
4. **Ketidakseimbangan Kelas (Class Imbalance):**
   - Kelas `Neutral` hanya sebesar 5.60%. Gunakan parameter `class_weight='balanced'` pada Linear SVM, Multinomial Naive Bayes, dan Random Forest, serta prioritaskan metrik **Macro F1-Score**.

---

## 8. Kamus Normalisasi Slang Terdokumentasi

| Slang / Singkatan | Normalisasi | Slang / Singkatan | Normalisasi |
| :--- | :--- | :--- | :--- |
| `aja` | `saja` | `klo` / `kalo` / `klu` | `kalau` |
| `ak` / `aku` / `gw` / `gue` | `saya` | `km` / `lu` | `kamu` |
| `aplk` / `apk` | `aplikasi` | `krg` | `kurang` |
| `baguus` | `bagus` | `krn` / `karna` / `krna` | `karena` |
| `bgs` / `bgus` | `bagus` | `lemott` | `lemot` |
| `bgt` / `bngt` | `banget` | `lg` | `lagi` |
| `blh` | `boleh` | `lmyn` / `lumyan` | `lumayan` |
| `blm` / `blum` | `belum` | `mantab` / `mantep` / `mantapp` | `mantap` |
| `bner` / `bnr` | `benar` | `mksh` / `makasih` / `thx` / `tq` / `trims` | `terima kasih` |
| `bs` / `bsa` | `bisa` | `msh` | `masih` |
| `bnyk` / `byk` | `banyak` | `mslh` | `masalah` |
| `cb` | `coba` | `nggak` / `gak` / `ga` / `gk` / `tdk` | `tidak` |
| `cmn` | `cuma` | `okee` | `oke` |
| `dgn` / `dg` / `dng` | `dengan` | `org` / `orng` | `orang` |
| `dlm` | `dalam` | `pake` / `pke` | `pakai` |
| `doang` | `saja` | `parahh` | `parah` |
| `dpt` | `dapat` | `pdhl` | `padahal` |
| `dri` | `dari` | `sampe` / `smpe` | `sampai` |
| `emang` / `emg` | `memang` | `sdh` / `udh` / `udah` | `sudah` |
| `gimana` / `gmn` / `gmna` | `bagaimana` | `skrg` / `skr` | `sekarang` |
| `hrs` / `hrus` | `harus` | `sm` | `sama` |
| `jgn` / `jng` | `jangan` | `spt` / `sprti` | `seperti` |
| `jlek` / `jlk` / `jelekk` | `jelek` | `tp` / `tpi` | `tapi` |
| `jg` | `juga` | `trs` / `trus` | `terus` |
| `kerenn` | `keren` | `utk` | `untuk` |
| `knp` | `kenapa` | `yg` | `yang` |
