"""
Page 2: Dataset Explorer — Government App Sentiment Intelligence Dashboard
Interactive filtering, metadata audit exploration, time trends, and paginated review inspector.
"""

import textwrap
from typing import List
import pandas as pd
import streamlit as st

from src.components.kpi_card import render_kpi_card
from src.utils.i18n import translate
from src.components.charts import (
    create_sentiment_donut_chart,
    create_rating_distribution_chart,
    create_app_reviews_bar_chart,
    create_time_series_trend
)
from src.utils.data_loader import query_review_table


def render_explorer_page(df: pd.DataFrame):
    """Renders Page 2: Dataset Explorer."""
    st.markdown(
        textwrap.dedent("""
        <div class="saas-header-container">
            <span class="saas-badge">Data Intelligence</span>
            <h1 class="saas-title">Dataset Explorer & Review Browser</h1>
            <p class="saas-subtitle">
                Explore distribution patterns, inspect duplicate clusters, analyze label contradictions, and search authentic public reviews.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    if df.empty:
        st.warning("Dataset not found. Please ensure `data/Rating_labeled_processed.csv.gz` exists.")
        return

    # 1. Summary Cards Row
    total_records = len(df)
    unique_groups = df['duplicate_group'].nunique()
    dupe_rows = (df['group_size'] > 1).sum()
    conflict_rows = (df['has_label_conflict'] == True).sum()
    cross_app_rows = (df['n_apps_in_group'] > 1).sum()

    m1, m2, m3, m4 = st.columns(4)
    render_kpi_card(
        label="Total Cleaned Records",
        value=f"{total_records:,}",
        subtext=f"{total_records:,} valid reviews",
        icon="",
        col=m1
    )
    render_kpi_card(
        label="Unique Text Groups",
        value=f"{unique_groups:,}",
        subtext=f"{(dupe_rows/total_records)*100:.1f}% duplicate rows",
        icon="",
        badge_text=f"{(dupe_rows/total_records)*100:.1f}% Duplicates",
        badge_variant="neutral",
        col=m2
    )
    render_kpi_card(
        label="Label Conflict Reviews",
        value=f"{conflict_rows:,}",
        subtext=f"{(conflict_rows/total_records)*100:.1f}% of total data",
        icon="",
        badge_text="Label Noise",
        badge_variant="negative",
        col=m3
    )
    render_kpi_card(
        label="Cross-App Review Rows",
        value=f"{cross_app_rows:,}",
        subtext=f"{(cross_app_rows/total_records)*100:.1f}% shared across apps",
        icon="",
        badge_text="Shared Texts",
        badge_variant="positive",
        col=m4
    )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Interactive Filter Bar
    st.markdown(
        textwrap.dedent("""
        <div style="font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem;">
             Interactive Filter Matrix
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    f_col1, f_col2, f_col3 = st.columns(3)
    available_apps = sorted(df['app'].unique().tolist())
    with f_col1:
        selected_apps = st.multiselect(
            "Filter Applications:",
            options=available_apps,
            default=available_apps,
            help="Filter analytics and review browsing by specific government applications."
        )

    with f_col2:
        sentiment_values = ['Positive', 'Neutral', 'Negative']
        sentiment_options = [translate(value) for value in sentiment_values]
        previous_sentiments = st.session_state.get("selected_sentiment_filter", sentiment_values)
        default_sentiment_options = [translate(value) for value in previous_sentiments]
        selected_sentiment_options = st.multiselect(
            "Filter Sentiment Classes:",
            options=sentiment_options,
            default=default_sentiment_options,
            key=f"sentiment_filter_{st.session_state.get('ui_language', 'id')}"
        )
        selected_sentiments = [sentiment_values[sentiment_options.index(value)] for value in selected_sentiment_options]
        st.session_state.selected_sentiment_filter = selected_sentiments

    with f_col3:
        selected_ratings = st.multiselect(
            "Filter Rating Stars (1-5):",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )

    f_check1, f_check2 = st.columns([1, 1.5])
    with f_check1:
        only_conflicts = st.checkbox(
            translate(" Filter ONLY Label-Conflict rows (Identical text, conflicting ratings)"),
            value=False,
        )
    with f_check2:
        search_kw = st.text_input(
            " Search in review text:",
            placeholder="e.g. login, update, mantap, bpjs, antrian, lemot...",
            help="Search filters reviews and charts by keyword. Application, sentiment, rating, and conflict filters also apply."
        )

    page_size = st.session_state.get("explorer_page_size", 15)
    filter_signature = (
        tuple(selected_apps),
        tuple(selected_sentiments),
        tuple(selected_ratings),
        only_conflicts,
        search_kw.strip().casefold(),
        page_size,
    )
    if st.session_state.get("explorer_filter_signature") != filter_signature:
        st.session_state.explorer_filter_signature = filter_signature
        st.session_state.explorer_page = 1
        st.session_state.explorer_page_number = 1

    # Apply filter mask to metadata
    filtered_df = df.copy()
    if selected_apps:
        filtered_df = filtered_df[filtered_df['app'].isin(selected_apps)]
    if selected_sentiments:
        filtered_df = filtered_df[filtered_df['label'].isin(selected_sentiments)]
    if selected_ratings:
        filtered_df = filtered_df[filtered_df['score'].isin(selected_ratings)]
    if only_conflicts:
        filtered_df = filtered_df[filtered_df['has_label_conflict'] == True]

    current_page = st.session_state.get("explorer_page", 1)
    table_df, total_matched, matching_review_ids = query_review_table(
        selected_apps=selected_apps,
        selected_sentiments=selected_sentiments,
        selected_ratings=selected_ratings,
        search_keyword=search_kw,
        only_label_conflicts=only_conflicts,
        page=current_page,
        page_size=page_size,
    )
    max_pages = max(1, (total_matched + page_size - 1) // page_size)
    if current_page > max_pages:
        current_page = 1
        st.session_state.explorer_page = 1
        st.session_state.explorer_page_number = 1
        table_df, total_matched, matching_review_ids = query_review_table(
            selected_apps=selected_apps,
            selected_sentiments=selected_sentiments,
            selected_ratings=selected_ratings,
            search_keyword=search_kw,
            only_label_conflicts=only_conflicts,
            page=current_page,
            page_size=page_size,
        )
    if matching_review_ids is not None:
        filtered_df = filtered_df[filtered_df['review_id'].isin(matching_review_ids)]

    st.markdown(
        textwrap.dedent(f"""
        <div style="margin: 0.75rem 0 1.25rem 0; font-size: 0.85rem; color: var(--text-secondary);">
            Chart Filter Match: <span style="color: #6366F1; font-weight: 700;">{len(filtered_df):,} records</span> 
            ({(len(filtered_df)/total_records)*100:.2f}% of repository)
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    # 3. Filtered Charts
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.plotly_chart(create_sentiment_donut_chart(filtered_df), use_container_width=True)
    with c_chart2:
        st.plotly_chart(create_rating_distribution_chart(filtered_df), use_container_width=True)

    # Time series trend chart
    st.plotly_chart(create_time_series_trend(filtered_df), use_container_width=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 4. Review Explorer (Searchable & Paginated Table)
    st.markdown(
        textwrap.dedent("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="font-size: 1.2rem; font-weight: 700; color: var(--text-primary); margin: 0;">
                 Paginated Review Inspector
            </h3>
            <p style="color: var(--text-secondary); font-size: 0.825rem; margin-top: 2px;">
                Inspect authentic user feedback with full text. Paginated to ensure fast rendering.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    # Pagination controls
    p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
    with p_col1:
        page_size = st.selectbox(
            "Rows per page:",
            options=[15, 30, 50],
            key="explorer_page_size",
        )

    with p_col2:
        selected_page = st.number_input(
            f"Page (of {max_pages:,}):",
            min_value=1,
            max_value=max_pages,
            step=1,
            key="explorer_page_number",
        )
        if selected_page != current_page:
            st.session_state["explorer_page"] = selected_page
            st.rerun()

    with p_col3:
        st.markdown(
            textwrap.dedent(f"""
            <div style="padding-top: 1.85rem; font-size: 0.85rem; color: var(--text-secondary);">
                Showing reviews <b>{(current_page-1)*page_size + 1:,}</b> to <b>{min(current_page*page_size, total_matched):,}</b> 
                of <b>{total_matched:,} matching</b>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )

    # Display Table
    if not table_df.empty:
        display_cols = ['review_id', 'app', 'score', 'label', 'at', 'content_original', 'group_size', 'has_label_conflict']
        styled_df = table_df[display_cols].copy()
        styled_df.rename(columns={
            'review_id': 'ID',
            'app': 'Application',
            'score': 'Rating',
            'label': 'Sentiment',
            'at': 'Timestamp',
            'content_original': 'Review Text',
            'group_size': 'Dupe Frequency',
            'has_label_conflict': translate('Conflicting Label?')
        }, inplace=True)

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rating": st.column_config.NumberColumn(format="%d "),
                "Sentiment": st.column_config.TextColumn(help="Ground truth sentiment based on rating score"),
                "Review Text": st.column_config.TextColumn(width="large"),
                "Timestamp": st.column_config.TextColumn(width="medium")
            }
        )
    else:
        st.info("No reviews matched the current filter combination.")
