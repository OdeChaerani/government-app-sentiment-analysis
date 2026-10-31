"""
Standardized Plotly Visualizations for Government App Sentiment Intelligence Dashboard
Implements dark-first SaaS styling matching Linear, Vercel, and Stripe design tokens.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.i18n import translate

# Design Palette
THEME = {
    "bg_card": "#151F33",
    "bg_main": "#0B1020",
    "border": "#26344D",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "positive": "#22C55E",
    "neutral": "#F59E0B",
    "negative": "#F87171",
    "accent_primary": "#6366F1",
    "accent_secondary": "#38BDF8",
    "accent_purple": "#A855F7"
}

CHART_THEMES = {
    "dark": {
        "bg_card": "#151F33", "bg_main": "#0B1020", "border": "#26344D",
        "text_primary": "#F8FAFC", "text_secondary": "#A8B5C8",
        "positive": "#22C55E", "neutral": "#F4B740", "negative": "#F87171",
        "accent_primary": "#716BFF", "accent_secondary": "#38BDF8", "accent_purple": "#A855F7",
    },
    "light": {
        "bg_card": "#FFFFFF", "bg_main": "#F3F6FC", "border": "#D8E1EE",
        "text_primary": "#18243A", "text_secondary": "#52627A",
        "positive": "#16834A", "neutral": "#A96100", "negative": "#D14343",
        "accent_primary": "#5148D8", "accent_secondary": "#087CA4", "accent_purple": "#7542B8",
    },
}

COLOR_MAP = {
    "Positive": THEME["positive"],
    "Neutral": THEME["neutral"],
    "Negative": THEME["negative"]
}


def set_chart_theme(theme_name: str) -> None:
    """Updates Plotly chart colors to match the active dashboard theme."""
    palette = CHART_THEMES.get(theme_name, CHART_THEMES["dark"])
    THEME.update(palette)
    COLOR_MAP.update({
        "Positive": palette["positive"],
        "Neutral": palette["neutral"],
        "Negative": palette["negative"],
    })


def apply_common_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Applies unified dark theme, typography, padding, and tooltip aesthetics."""
    fig.update_layout(
        title={
            "text": translate(title),
            "y": 0.95,
            "x": 0.03,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "color": THEME["text_primary"], "family": "Plus Jakarta Sans"}
        },
        height=height,
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_card"],
        font={"family": "Plus Jakarta Sans, sans-serif", "color": THEME["text_secondary"], "size": 12},
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": THEME["text_secondary"]}
        },
        hoverlabel={
            "bgcolor": THEME["bg_main"],
            "font_size": 12,
            "font_family": "Plus Jakarta Sans",
            "bordercolor": THEME["border"]
        }
    )
    fig.update_xaxes(
        gridcolor=THEME["border"],
        zerolinecolor=THEME["border"],
        tickfont={"color": THEME["text_secondary"]}
    )
    fig.update_yaxes(
        gridcolor=THEME["border"],
        zerolinecolor=THEME["border"],
        tickfont={"color": THEME["text_secondary"]}
    )
    return fig


def create_sentiment_donut_chart(df: pd.DataFrame, height: int = 360) -> go.Figure:
    """Creates a sleek donut chart showing overall sentiment distribution."""
    counts = df['label'].value_counts()
    labels = ['Positive', 'Negative', 'Neutral']
    values = [counts.get(l, 0) for l in labels]
    colors = [COLOR_MAP[l] for l in labels]
    labels = [translate(label) for label in labels]

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.68,
            marker={"colors": colors, "line": {"color": THEME["bg_card"], "width": 3}},
            textinfo="percent+label",
            textposition="outside",
            textfont={"size": 12, "color": THEME["text_primary"]},
            hovertemplate=f"<b>%{{label}}</b><br>{translate('Count:')} %{{value:,}}<br>{translate('Proportion:')} %{{percent}}<extra></extra>"
        )
    ])
    apply_common_layout(fig, title="Overall Sentiment Distribution", height=height)
    fig.update_layout(showlegend=False)
    return fig


def create_app_reviews_bar_chart(df: pd.DataFrame, height: int = 360) -> go.Figure:
    """Creates a horizontal bar chart of review volume per application."""
    counts = df['app'].value_counts().sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index.astype(str),
        orientation='h',
        marker=dict(
            color=counts.values,
            colorscale=[[0, THEME["accent_secondary"]], [1, THEME["accent_primary"]]],
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        ),
        text=[f"{v:,}" for v in counts.values],
        textposition="auto",
        textfont={"color": "#FFFFFF", "size": 11},
        hovertemplate=f"<b>%{{y}}</b><br>{translate('Total Reviews')}: %{{x:,}}<extra></extra>"
    ))
    apply_common_layout(fig, title="Total Reviews by Application", height=height)
    fig.update_xaxes(title_text=translate("Number of Reviews"))
    fig.update_layout(showlegend=False)
    return fig


def create_sentiment_stacked_bar(df: pd.DataFrame, height: int = 360) -> go.Figure:
    """Creates a 100% stacked bar chart showing sentiment ratio per application."""
    crosstab = pd.crosstab(df['app'], df['label'], normalize='index') * 100
    apps = crosstab.index.tolist()

    fig = go.Figure()
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment in crosstab.columns:
            fig.add_trace(go.Bar(
                name=translate(sentiment),
                x=apps,
                y=crosstab[sentiment],
                marker_color=COLOR_MAP[sentiment],
                hovertemplate=f"<b>%{{x}}</b><br>{translate(sentiment)}: %{{y:.2f}}%<extra></extra>"
            ))

    fig.update_layout(barmode='stack')
    apply_common_layout(fig, title="Sentiment Composition by Application (%)", height=height)
    fig.update_yaxes(title_text=translate("Percentage (%)"), range=[0, 100])
    return fig


def create_rating_distribution_chart(df: pd.DataFrame, height: int = 360) -> go.Figure:
    """Visualizes the 1-5 rating score frequency showing bipolar U-shape."""
    rating_counts = df['score'].value_counts().sort_index()
    
    colors = [
        THEME["negative"],  # 1
        THEME["negative"],  # 2
        THEME["neutral"],   # 3
        THEME["positive"],  # 4
        THEME["positive"]   # 5
    ]
    
    fig = go.Figure(go.Bar(
        x=[f" {r}" for r in rating_counts.index],
        y=rating_counts.values,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:,}" for v in rating_counts.values],
        textposition="outside",
        textfont={"color": THEME["text_primary"], "size": 11},
        hovertemplate=f"<b>{translate('Rating')} %{{x}}</b><br>{translate('Reviews')}: %{{y:,}}<extra></extra>"
    ))
    apply_common_layout(fig, title="Rating Score Distribution (1 to 5 Stars)", height=height)
    fig.update_yaxes(title_text=translate("Review Count"))
    fig.update_layout(showlegend=False)
    return fig


def create_time_series_trend(df: pd.DataFrame, height: int = 360) -> go.Figure:
    """Creates a monthly review volume & sentiment trend over time."""
    if 'year_month' not in df.columns:
        return go.Figure()

    monthly = df.groupby(['year_month', 'label'], observed=True).size().unstack(fill_value=0)
    monthly = monthly.sort_index()

    fig = go.Figure()
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment in monthly.columns:
            fig.add_trace(go.Scatter(
                x=monthly.index,
                y=monthly[sentiment],
                name=translate(sentiment),
                mode='lines+markers',
                line=dict(color=COLOR_MAP[sentiment], width=2.5),
                marker=dict(size=4),
                hovertemplate=f"{translate('Year-Month')}: %{{x}}<br>{translate(sentiment)}: %{{y:,}}<extra></extra>"
            ))

    apply_common_layout(fig, title="Review Volume Trend Over Time", height=height)
    fig.update_xaxes(title_text=translate("Year-Month"), tickangle=-45)
    fig.update_yaxes(title_text=translate("Monthly Reviews"))
    return fig


def create_model_ranking_bar(df_models: pd.DataFrame, height: int = 340) -> go.Figure:
    """Horizontal ranking chart based on Macro-F1 score."""
    sorted_df = df_models.sort_values('macro_f1', ascending=True)
    
    colors = [THEME["accent_primary"] if m == sorted_df['macro_f1'].max() else THEME["bg_surface-2"] if False else THEME["border"] for m in sorted_df['macro_f1']]
    colors = [THEME["accent_secondary"] if i < len(sorted_df)-1 else THEME["accent_primary"] for i in range(len(sorted_df))]

    fig = go.Figure(go.Bar(
        x=sorted_df['macro_f1'],
        y=sorted_df['model'],
        orientation='h',
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
        text=[f"{v:.4f}" for v in sorted_df['macro_f1']],
        textposition="inside",
        textfont={"color": "#FFFFFF", "size": 12, "family": "JetBrains Mono"},
        hovertemplate=f"<b>%{{y}}</b><br>{translate('Macro-F1 Score')}: %{{x:.4f}}<extra></extra>"
    ))
    apply_common_layout(fig, title="Model Ranking by Macro-F1 Score", height=height)
    fig.update_xaxes(title_text=translate("Macro-F1 Score"), range=[0.5, 0.75])
    fig.update_layout(showlegend=False)
    return fig


def create_multi_metric_grouped_bar(df_models: pd.DataFrame, height: int = 360) -> go.Figure:
    """Grouped bar chart comparing Accuracy, Macro Precision, Macro Recall, and Macro-F1."""
    metrics = [
        ("accuracy", "Accuracy", THEME["accent_secondary"]),
        ("macro_precision", "Precision", THEME["accent_purple"]),
        ("macro_recall", "Recall", THEME["neutral"]),
        ("macro_f1", "Macro-F1", THEME["accent_primary"])
    ]

    fig = go.Figure()
    for col, name, color in metrics:
        if col in df_models.columns:
            fig.add_trace(go.Bar(
                name=translate(name),
                x=df_models['model'],
                y=df_models[col],
                marker_color=color,
                hovertemplate=f"<b>%{{x}}</b><br>{translate(name)}: %{{y:.4f}}<extra></extra>"
            ))

    apply_common_layout(fig, title="Comprehensive Model Benchmark Comparison", height=height)
    fig.update_yaxes(title_text=translate("Score"), range=[0, 1.0])
    return fig


def create_confusion_matrix_heatmap(matrix: List[List[int]], labels: List[str], model_name: str, height: int = 360) -> go.Figure:
    """Renders confusion matrix heatmap with exact counts and normalized annotations."""
    mat = np.array(matrix)
    display_labels = [translate(label) for label in labels]
    row_sums = mat.sum(axis=1, keepdims=True)
    norm_mat = np.divide(
        mat,
        row_sums,
        out=np.zeros_like(mat, dtype=float),
        where=row_sums != 0,
    ) * 100

    fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=display_labels,
        y=display_labels,
        text=norm_mat,
        texttemplate="%{z:.0f}<br>(%{text:.1f}%)",
        textfont={"color": THEME["text_primary"], "size": 12},
        colorscale=[[0, THEME["bg_main"]], [0.5, THEME["border"]], [1, THEME["accent_primary"]]],
        showscale=False,
        hovertemplate=f"{translate('Actual:')} %{{y}}<br>{translate('Predicted:')} %{{x}}<br>{translate('Count:')} %{{z:.0f}}<extra></extra>"
    ))

    apply_common_layout(fig, title="Confusion Matrix: " + model_name, height=height)
    fig.update_xaxes(title_text=translate("Predicted Label"))
    fig.update_yaxes(title_text=translate("Actual Label"), autorange="reversed")
    return fig


def create_loao_performance_chart(df_loao: pd.DataFrame, height: int = 360) -> go.Figure:
    """Bar chart comparing Natural Cross-App Macro-F1 vs Strict Cross-App Macro-F1."""
    fig = go.Figure()
    
    # Natural Macro-F1
    fig.add_trace(go.Bar(
        name=translate("Natural Cross-App F1"),
        x=df_loao['target_app'],
        y=df_loao['macro_f1'],
        marker_color=THEME["accent_primary"],
        text=[f"{v:.4f}" for v in df_loao['macro_f1']],
        textposition="outside",
        hovertemplate=f"<b>%{{x}}</b><br>{translate('Natural LOAO Macro-F1:')} %{{y:.4f}}<extra></extra>"
    ))

    # Strict Macro-F1
    if 'pure_macro_f1_n_apps_eq_1' in df_loao.columns:
        fig.add_trace(go.Bar(
        name=translate("Strict Cross-App F1 (Pure Unseen)"),
            x=df_loao['target_app'],
            y=df_loao['pure_macro_f1_n_apps_eq_1'],
            marker_color=THEME["accent_secondary"],
            text=[f"{v:.4f}" for v in df_loao['pure_macro_f1_n_apps_eq_1']],
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{translate('Strict LOAO Macro-F1:')} %{{y:.4f}}<extra></extra>"
        ))

    apply_common_layout(fig, title="LOAO Macro-F1 Across Target Applications", height=height)
    fig.update_yaxes(title_text=translate("Macro-F1 Score"), range=[0.4, 0.72])
    return fig


def create_baseline_vs_loao_chart(baseline_f1: float, df_loao: pd.DataFrame, height: int = 360) -> go.Figure:
    """Visualizes performance degradation: In-domain Random Split baseline vs LOAO target apps."""
    fig = go.Figure()

    # Baseline line
    fig.add_hline(
        y=baseline_f1,
        line_dash="dash",
        line_color=THEME["positive"],
        annotation_text=translate(f"Baseline Holdout (Macro-F1: {baseline_f1:.4f})"),
        annotation_position="top left",
        annotation_font_color=THEME["positive"]
    )

    # LOAO bars
    delta = df_loao['macro_f1'] - baseline_f1
    colors = [THEME["positive"] if d >= 0 else THEME["negative"] for d in delta]

    fig.add_trace(go.Bar(
        x=df_loao['target_app'],
        y=df_loao['macro_f1'],
        marker_color=THEME["accent_primary"],
        text=[f"{d:+.4f}" for d in delta],
        textposition="inside",
        textfont={"color": "#FFFFFF", "size": 11, "family": "JetBrains Mono"},
        hovertemplate=f"<b>%{{x}}</b><br>{translate('LOAO Macro-F1: ')}%{{y:.4f}}<br>{translate('Delta vs Baseline:')} %{{text}}<extra></extra>"
    ))

    apply_common_layout(fig, title="Random Split Baseline vs. LOAO Cross-App Generalization", height=height)
    fig.update_yaxes(title_text=translate("Macro-F1 Score"), range=[0.45, 0.72])
    fig.update_layout(showlegend=False)
    return fig


def create_probability_gauge_or_bar(probabilities: Dict[str, float], height: int = 240) -> go.Figure:
    """Horizontal confidence distribution for the predicted sentiment classes."""
    labels = list(probabilities.keys())
    values = [probabilities[l] for l in labels]
    colors = [COLOR_MAP.get(l, THEME["accent_primary"]) for l in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
        text=[f"{v*100:.1f}%" if v <= 1.0 else f"{v:.2f}" for v in values],
        textposition="outside",
        textfont={"color": THEME["text_primary"], "size": 12, "family": "JetBrains Mono"},
        hovertemplate="<b>%{y}</b>: %{text}<extra></extra>"
    ))
    apply_common_layout(fig, title="Prediction Probability Confidence", height=height)
    fig.update_xaxes(range=[0, 1.15], showticklabels=False)
    fig.update_layout(showlegend=False)
    return fig
