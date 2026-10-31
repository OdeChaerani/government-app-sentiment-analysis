"""
Page 1: Overview — Government App Sentiment Intelligence Dashboard
High-level summary of the dataset and sentiment distribution.
"""

import textwrap
import html
import re
import pandas as pd
import streamlit as st

from src.components.kpi_card import render_kpi_card
from src.components.charts import (
    create_sentiment_donut_chart,
    create_app_reviews_bar_chart,
    create_sentiment_stacked_bar,
    create_rating_distribution_chart
)
from src.utils.data_loader import calculate_overview_kpis, calculate_dynamic_insights
from src.utils.i18n import translate


def render_overview_page(df: pd.DataFrame):
    """Renders Page 1: Overview."""
    app_count = int(df["app"].nunique()) if not df.empty else 0
    # Header
    st.markdown(
        textwrap.dedent(f"""
        <div class="saas-header-container">
            <span class="saas-badge">Executive Summary</span>
            <h1 class="saas-title">Government App Sentiment Intelligence</h1>
            <p class="saas-subtitle">
                Comprehensive sentiment analytics across {app_count} Indonesian public services and state-owned enterprise (BUMN) applications based on the IGAR dataset.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    if df.empty:
        st.warning("Dataset not found. Please ensure `data/Rating_labeled_processed.csv.gz` exists.")
        return

    # Calculate KPIs
    kpis = calculate_overview_kpis(df)

    # 1. KPI Cards Row
    c1, c2, c3, c4, c5 = st.columns(5)
    render_kpi_card(
        label="Total Reviews",
        value=f"{kpis['total_reviews']:,}",
        subtext="Curated IGAR records",
        icon="",
        col=c1
    )
    render_kpi_card(
        label="Target Apps",
        value=f"{kpis['total_apps']}",
        subtext="Public agencies & BUMN",
        icon="",
        col=c2
    )
    render_kpi_card(
        label="Positive Reviews",
        value=f"{kpis['positive_count']:,}",
        subtext=f"{kpis['positive_pct']:.1f}% of repository",
        icon="",
        badge_text="Majority",
        badge_variant="positive",
        col=c3
    )
    render_kpi_card(
        label="Neutral Reviews",
        value=f"{kpis['neutral_count']:,}",
        subtext=f"{kpis['neutral_pct']:.1f}% (Class Imbalance)",
        icon="",
        badge_text="Minority",
        badge_variant="neutral",
        col=c4
    )
    render_kpi_card(
        label="Negative Reviews",
        value=f"{kpis['negative_count']:,}",
        subtext=f"{kpis['negative_pct']:.1f}% of repository",
        icon="",
        badge_text="Critical",
        badge_variant="negative",
        col=c5
    )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Charts Row 1: Donut & App Volume
    col_chart1, col_chart2 = st.columns([1, 1.25])
    with col_chart1:
        st.plotly_chart(create_sentiment_donut_chart(df), use_container_width=True)
    with col_chart2:
        st.plotly_chart(create_app_reviews_bar_chart(df), use_container_width=True)

    # 3. Charts Row 2: Sentiment Stacked Bar & Rating Distribution
    col_chart3, col_chart4 = st.columns([1.25, 1])
    with col_chart3:
        st.plotly_chart(create_sentiment_stacked_bar(df), use_container_width=True)
    with col_chart4:
        st.plotly_chart(create_rating_distribution_chart(df), use_container_width=True)

    # 4. Dynamically Calculated Dataset Insights
    st.markdown(
        textwrap.dedent("""
        <div style="margin-top: 1rem; margin-bottom: 0.75rem;">
            <h3 style="font-size: 1.25rem; font-weight: 700; color: var(--text-primary); margin: 0;">
                 Dynamically Calculated Domain Insights
            </h3>
            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 4px;">
                Factual observations generated directly from cross-application sentiment distributions and metadata.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    insights = calculate_dynamic_insights(df)
    c_ins1, c_ins2 = st.columns(2)
    for i, ins in enumerate(insights):
        target_col = c_ins1 if i % 2 == 0 else c_ins2
        insight_content = ins['content']
        if st.session_state.get("ui_language", "id") == "id":
            insight_content = insight_content.replace(" and **", " dan **")
        insight_content = translate(insight_content)
        if st.session_state.get("ui_language", "id") == "id":
            insight_content = insight_content.replace(" at ", " pada ")
        insight_content = re.sub(
            r"\*\*(.+?)\*\*",
            lambda match: f"<strong>{match.group(1)}</strong>",
            html.escape(insight_content),
        )
        with target_col:
            st.markdown(
                textwrap.dedent(f"""
                <div class="insight-box {ins['type']}">
                    <div class="insight-title">{ins['title']}</div>
                    <div class="insight-content">{insight_content}</div>
                </div>
                """).strip(),
                unsafe_allow_html=True
            )
