"""
Data Loader Utility for Government App Sentiment Intelligence Dashboard
Implements memory-optimized loading, caching, and querying for processed review records.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_GZ_PATH = REPO_ROOT / "data" / "Rating_labeled_processed.csv.gz"
DATA_RAW_CSV_PATH = REPO_ROOT / "data" / "Rating_labeled_processed.csv"


def get_data_filepath() -> Path:
    """Returns the existing processed data filepath."""
    if DATA_RAW_CSV_PATH.exists():
        return DATA_RAW_CSV_PATH
    return DATA_GZ_PATH


@st.cache_data(show_spinner="Loading dataset metadata...")
def load_dataset_metadata() -> pd.DataFrame:
    """
    Loads core metadata columns with optimized data types.
    Memory footprint is ~24 MB compared to ~250 MB for full text strings.
    """
    filepath = get_data_filepath()
    if not filepath.exists():
        return pd.DataFrame()

    usecols = [
        'review_id', 'app', 'score', 'label', 'at',
        'duplicate_group', 'group_size', 'has_label_conflict', 'n_apps_in_group'
    ]
    dtypes = {
        'review_id': 'int32',
        'app': 'category',
        'score': 'int8',
        'label': 'category',
        'duplicate_group': 'int32',
        'group_size': 'int16',
        'has_label_conflict': 'bool',
        'n_apps_in_group': 'int8'
    }

    df = pd.read_csv(filepath, usecols=usecols, dtype=dtypes)
    # Parse year-month for time-series trend analysis
    df['year_month'] = df['at'].str.slice(0, 7)
    return df


@st.cache_data
def calculate_overview_kpis(df: pd.DataFrame) -> Dict[str, any]:
    """Calculates KPI summary statistics for the Overview page."""
    if df.empty:
        return {}

    total = len(df)
    labels = df['label'].value_counts()
    pos = int(labels.get('Positive', 0))
    neu = int(labels.get('Neutral', 0))
    neg = int(labels.get('Negative', 0))

    return {
        "total_reviews": total,
        "total_apps": int(df['app'].nunique()),
        "positive_count": pos,
        "positive_pct": (pos / total) * 100 if total else 0.0,
        "neutral_count": neu,
        "neutral_pct": (neu / total) * 100 if total else 0.0,
        "negative_count": neg,
        "negative_pct": (neg / total) * 100 if total else 0.0,
        "duplicate_groups": int(df['duplicate_group'].nunique()),
        "duplicate_rows": int(df[df['group_size'] > 1].shape[0]),
        "duplicate_pct": (df[df['group_size'] > 1].shape[0] / total) * 100 if total else 0.0,
        "conflict_rows": int(df[df['has_label_conflict'] == True].shape[0]),
        "conflict_pct": (df[df['has_label_conflict'] == True].shape[0] / total) * 100 if total else 0.0
    }


@st.cache_data
def calculate_dynamic_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Calculates factual, data-driven domain insights from actual review distributions."""
    if df.empty:
        return []

    insights = []
    total = len(df)

    # 1. Highest review volume app
    app_counts = df['app'].value_counts()
    top_app = app_counts.index[0]
    top_vol = app_counts.iloc[0]
    top_vol_pct = (top_vol / total) * 100
    second_app = app_counts.index[1] if len(app_counts) > 1 else None
    second_vol = app_counts.iloc[1] if len(app_counts) > 1 else 0
    top_two_share = ((top_vol + second_vol) / total) * 100
    combined_apps = f"**{top_app}** and **{second_app}**" if second_app else f"**{top_app}**"
    insights.append({
        "type": "positive",
        "title": "Highest Volume Concentration",
        "content": f"**{top_app}** has the most reviews: **{top_vol:,}** ({top_vol_pct:.2f}% of total). {combined_apps} account for {top_two_share:.2f}% together."
    })

    # 2. Highest negative sentiment proportion
    app_sent = pd.crosstab(df['app'], df['label'], normalize='index') * 100
    if 'Negative' in app_sent.columns:
        neg_series = app_sent['Negative'].sort_values(ascending=False)
        highest_neg_app = neg_series.index[0]
        highest_neg_pct = neg_series.iloc[0]
        insights.append({
            "type": "danger",
            "title": "Most Critical Public Sentiment",
            "content": f"**{highest_neg_app}** has the highest Negative share at **{highest_neg_pct:.2f}%** of its reviews."
        })

    # 3. Class imbalance insight
    neu_pct = (df['label'].eq('Neutral').sum() / total) * 100
    insights.append({
        "type": "warning",
        "title": "Severe Class Imbalance (Neutral Scarcity)",
        "content": f"Neutral reviews represent only **{neu_pct:.2f}%** of the dataset, producing a bipolar (U-shaped) rating curve. Evaluated models must prioritize **Macro-F1** over accuracy to avoid penalizing neutral recognition."
    })

    # 4. Highest positive sentiment app
    if 'Positive' in app_sent.columns:
        pos_series = app_sent['Positive'].sort_values(ascending=False)
        highest_pos_app = pos_series.index[0]
        highest_pos_pct = pos_series.iloc[0]
        second_pos_app = pos_series.index[1] if len(pos_series) > 1 else None
        second_pos_pct = pos_series.iloc[1] if len(pos_series) > 1 else None
        comparison = (
            f" The next highest is {second_pos_app} at {second_pos_pct:.2f}%."
            if second_pos_app is not None else ""
        )
        insights.append({
            "type": "positive",
            "title": "Highest User Satisfaction",
            "content": f"**{highest_pos_app}** has the highest Positive share at **{highest_pos_pct:.2f}%**.{comparison}"
        })

    return insights


@st.cache_data(show_spinner="Querying review samples...")
def query_review_table(
    selected_apps: List[str],
    selected_sentiments: List[str],
    selected_ratings: List[int],
    search_keyword: str = "",
    only_label_conflicts: bool = False,
    page: int = 1,
    page_size: int = 20
) -> Tuple[pd.DataFrame, int, Optional[np.ndarray]]:
    """
    Efficiently queries and paginates review records with text content.
    Loads text columns on-demand with selective filtering to maintain responsive UI.
    """
    filepath = get_data_filepath()
    if not filepath.exists():
        return pd.DataFrame(), 0, None

    usecols = [
        'review_id', 'app', 'content_original', 'score', 'label', 'at',
        'group_size', 'has_label_conflict', 'n_apps_in_group'
    ]
    
    # Read in chunks or full text with dtype optimization
    df = pd.read_csv(filepath, usecols=usecols)
    df['content_original'] = df['content_original'].fillna('')

    # Apply filters
    if selected_apps:
        df = df[df['app'].isin(selected_apps)]
    if selected_sentiments:
        df = df[df['label'].isin(selected_sentiments)]
    if selected_ratings:
        df = df[df['score'].isin(selected_ratings)]
    if only_label_conflicts:
        df = df[df['has_label_conflict'] == True]
    if search_keyword.strip():
        kw = search_keyword.strip().lower()
        df = df[df['content_original'].str.lower().str.contains(kw, regex=False)]

    total_filtered = len(df)
    matching_ids = (
        df['review_id'].to_numpy(dtype=np.int32, copy=True)
        if search_keyword.strip()
        else None
    )
    
    # Pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = df.iloc[start_idx:end_idx].copy()
    
    return page_df, total_filtered, matching_ids
