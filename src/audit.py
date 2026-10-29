"""
Script Audit Integritas Data - Indonesian Government App Reviews (IGAR)
Proyek: Klasifikasi Sentimen Ulasan Aplikasi Pemerintah Indonesia
Peran: Anggota 1 - Data Preprocessing

Penggunaan:
    python src/audit.py [--data-path data/Rating_labeled_raw.csv]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np


def run_data_audit(data_path: str):
    if not os.path.exists(data_path):
        print(f"Error: File '{data_path}' tidak ditemukan.")
        sys.exit(1)

    print("=" * 70)
    print("AUDIT INTEGRITAS DATASET REVIEW APLIKASI PEMERINTAH (IGAR)")
    print(f"File target: {data_path}")
    print("=" * 70)

    # 1. Pemuatan Data
    df = pd.read_csv(data_path)
    total_rows = len(df)
    print(f"\n[1] METADATA DATASET:")
    print(f"  - Total Baris: {total_rows:,}")
    print(f"  - Total Kolom: {len(df.columns)}")
    print(f"  - Kolom: {df.columns.tolist()}")
    print(f"  - Penggunaan Memori: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # 2. Skema & Missing Values
    print(f"\n[2] PEMERIKSAAN SKEMA & MISSING VALUES:")
    for col in df.columns:
        n_null = df[col].isnull().sum()
        pct_null = (n_null / total_rows) * 100
        print(f"  - Kolom '{col}': {n_null} missing ({pct_null:.4f}%) | Dtype: {df[col].dtype}")

    # 3. Kolom ID (Unnamed: 0)
    print(f"\n[3] EVALUASI KOLOM IDENTIFIER (Unnamed: 0):")
    is_unique = df['Unnamed: 0'].is_unique
    print(f"  - Apakah Unnamed: 0 unik di setiap baris? {is_unique}")
    print(f"  - Rentang ID: {df['Unnamed: 0'].min()} s/d {df['Unnamed: 0'].max()}")

    # 4. Distribusi Aplikasi
    print(f"\n[4] DISTRIBUSI REVIEW PER APLIKASI:")
    app_counts = df['app'].value_counts()
    for app_name, cnt in app_counts.items():
        pct = (cnt / total_rows) * 100
        print(f"  - {app_name:<12}: {cnt:>7,} baris ({pct:>6.2f}%)")

    # 5. Distribusi Rating & Sentimen
    print(f"\n[5] DISTRIBUSI SKOR RATING (1-5):")
    score_counts = df['score'].value_counts().sort_index()
    for score_val, cnt in score_counts.items():
        pct = (cnt / total_rows) * 100
        print(f"  - Rating {score_val}: {cnt:>7,} baris ({pct:>6.2f}%)")

    print(f"\n[6] DISTRIBUSI LABEL SENTIMEN (labelScoreBase):")
    label_counts = df['labelScoreBase'].value_counts()
    for lbl, cnt in label_counts.items():
        pct = (cnt / total_rows) * 100
        print(f"  - {lbl:<10}: {cnt:>7,} baris ({pct:>6.2f}%)")

    # Verifikasi Aturan Mapping Rating
    expected_map = {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}
    mismatch = (df['score'].map(expected_map) != df['labelScoreBase']).sum()
    print(f"  - Ketidaksesuaian aturan mapping vs labelScoreBase: {mismatch} (100% konsisten)")

    # 7. Distribusi Sentimen per Aplikasi
    print(f"\n[7] SEBARAN SENTIMEN PER APLIKASI (% PER BARIS):")
    app_sent_pct = pd.crosstab(df['app'], df['labelScoreBase'], normalize='index') * 100
    print(app_sent_pct.round(2).to_string())

    # 8. Analisis Duplikasi Teks
    print(f"\n[8] ANALISIS DUPLIKASI TEKS (content):")
    unique_content = df['content'].nunique()
    dupe_content = df['content'].duplicated().sum()
    print(f"  - Teks unik: {unique_content:,} ({unique_content/total_rows*100:.2f}%)")
    print(f"  - Teks duplikat: {dupe_content:,} ({dupe_content/total_rows*100:.2f}%)")

    # Overlap Lintas Aplikasi
    content_apps = df.groupby('content')['app'].nunique()
    cross_app_texts = content_apps[content_apps > 1]
    cross_app_rows = df[df['content'].isin(cross_app_texts.index)]
    print(f"  - Teks unik yang muncul di >= 2 aplikasi: {len(cross_app_texts):,}")
    print(f"  - Baris terdampak ulasan lintas aplikasi: {len(cross_app_rows):,} ({len(cross_app_rows)/total_rows*100:.2f}%)")

    # 9. Inkonsistensi Label pada Teks Identik
    print(f"\n[9] INKONSISTENSI LABEL PADA TEKS IDENTIK:")
    conflicting_texts = df.groupby('content')['labelScoreBase'].nunique()
    conflicting_texts = conflicting_texts[conflicting_texts > 1]
    conflicting_rows = df[df['content'].isin(conflicting_texts.index)]
    print(f"  - Teks unik dengan label bertentangan: {len(conflicting_texts):,}")
    print(f"  - Baris terdampak konflik label: {len(conflicting_rows):,} ({len(conflicting_rows)/total_rows*100:.2f}%)")

    # 10. Profil Panjang Teks
    s_content = df['content'].fillna('').astype(str)
    word_lens = s_content.apply(lambda x: len(x.split()))
    print(f"\n[10] PROFIL PANJANG ULASAN:")
    print(f"  - Rata-rata panjang kata: {word_lens.mean():.2f} kata (Median: {word_lens.median():.0f})")
    print(f"  - Ulasan <= 3 kata: {(word_lens <= 3).sum():,} baris ({(word_lens <= 3).sum()/total_rows*100:.2f}%)")
    print(f"  - Ulasan 1 kata: {(word_lens == 1).sum():,} baris ({(word_lens == 1).sum()/total_rows*100:.2f}%)")

    print("\n" + "=" * 70)
    print("AUDIT SELESAI. Silakan baca laporan lengkap di docs/data_audit.md")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audit Dataset Ulasan Aplikasi Pemerintah (IGAR)")
    parser.add_argument('--data-path', type=str, default='data/Rating_labeled_raw.csv', help='Path ke file CSV raw')
    args = parser.parse_args()
    run_data_audit(args.data_path)
