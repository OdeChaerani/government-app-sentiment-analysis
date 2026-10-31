"""
Page 3: Model Performance — Government App Sentiment Intelligence Dashboard
Interactive comparison of trained models, Macro-F1 rankings, confusion matrices, and per-class diagnostics.
"""

import textwrap
from typing import Dict, Any
import pandas as pd
import streamlit as st

from src.components.kpi_card import render_kpi_card
from src.components.charts import (
    create_model_ranking_bar,
    create_multi_metric_grouped_bar,
    create_confusion_matrix_heatmap
)
from src.utils.model_loader import (
    load_holdout_comparison,
    load_confusion_matrices,
    load_classification_reports,
    load_model_bundle,
)
from src.utils.i18n import translate


def render_model_performance_page():
    """Renders Page 3: Model Performance."""
    st.markdown(
        textwrap.dedent("""
        <div class="saas-header-container">
            <span class="saas-badge">Model Benchmark</span>
            <h1 class="saas-title">Algorithm Performance & Evaluation</h1>
            <p class="saas-subtitle">
                Holdout evaluation of the candidate models using the saved GroupShuffleSplit experiment results.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    df_models = load_holdout_comparison()
    cms = load_confusion_matrices()
    reports = load_classification_reports()
    bundle = load_model_bundle()

    if df_models.empty:
        st.markdown(
            textwrap.dedent("""
            <div class="empty-state">
                <div class="empty-state-icon"></div>
                <div class="empty-state-title">Benchmark Artifact Not Found</div>
                <div class="empty-state-desc">
                    The model comparison table <code>models/holdout_model_comparison.csv</code> is not yet available.
                    Provide Member 2's saved <code>holdout_model_comparison.csv</code> export. No retraining is needed.
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )
        return

    # 1. Best Model Dynamic Determination
    best_row = df_models.sort_values('macro_f1', ascending=False).iloc[0]
    best_model_name = best_row['model']
    best_macro_f1 = best_row['macro_f1']
    best_acc = best_row['accuracy']
    accuracy_row = df_models.loc[df_models['accuracy'].idxmax()]
    first_matrix = next((v for k, v in cms.items() if k != "labels"), None) if cms else None
    support_total = int(sum(sum(row) for row in first_matrix)) if first_matrix is not None else 0
    metric_explanation = (
        f"{accuracy_row['model']} has the highest accuracy ({accuracy_row['accuracy']:.4f}); "
        f"{best_model_name} has the highest Macro-F1 ({best_macro_f1:.4f}). "
        "Macro-F1 gives each sentiment class equal weight, which is useful when class support is uneven."
    )
    metric_explanation = translate(metric_explanation)

    b1, b2, b3, b4 = st.columns(4)
    render_kpi_card(
        label="Top Model (Macro-F1)",
        value=best_model_name,
        subtext=f"Evaluated on {support_total:,} holdout reviews" if support_total else "Holdout benchmark",
        icon="",
        badge_text="Best Performer",
        badge_variant="positive",
        col=b1
    )
    render_kpi_card(
        label="Best Macro-F1 Score",
        value=f"{best_macro_f1:.4f}",
        subtext="Class-balanced harmonic mean",
        icon="",
        badge_text="Primary Metric",
        badge_variant="accent",
        col=b2
    )
    render_kpi_card(
        label="Highest Accuracy",
        value=f"{accuracy_row['accuracy']:.4f}",
        subtext=str(accuracy_row['model']),
        icon="",
        badge_text="Accuracy Bias",
        badge_variant="neutral",
        col=b3
    )
    render_kpi_card(
        label="Holdout Test Sample",
        value=f"{support_total:,}" if support_total else "Unavailable",
        subtext="20% GroupShuffleSplit",
        icon="",
        badge_text="Zero Leakage",
        badge_variant="positive",
        col=b4
    )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Critical Methodological Note on Metric Selection
    st.markdown(
        textwrap.dedent(f"""
        <div class="insight-box warning">
            <div class="insight-title"> Why Macro-F1 is the Decisive Primary Metric</div>
            <div class="insight-content">
                {metric_explanation}
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    # 2. Model Comparison Table
    st.markdown(
        textwrap.dedent("""
        <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem;">
             Holdout Benchmark Leaderboard
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    display_table = df_models.copy()
    display_table['accuracy'] = display_table['accuracy'].apply(lambda x: f"{x*100:.2f}%")
    display_table['macro_precision'] = display_table['macro_precision'].apply(lambda x: f"{x*100:.2f}%")
    display_table['macro_recall'] = display_table['macro_recall'].apply(lambda x: f"{x*100:.2f}%")
    display_table['macro_f1'] = display_table['macro_f1'].apply(lambda x: f"{x:.4f}")
    display_table['weighted_f1'] = display_table['weighted_f1'].apply(lambda x: f"{x:.4f}")
    if 'training_time_sec' in display_table.columns:
        display_table['training_time_sec'] = display_table['training_time_sec'].apply(lambda x: f"{x:.1f}s")

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "model": st.column_config.TextColumn("Algorithm"),
            "accuracy": st.column_config.TextColumn("Accuracy"),
            "macro_precision": st.column_config.TextColumn("Macro Precision"),
            "macro_recall": st.column_config.TextColumn("Macro Recall"),
            "macro_f1": st.column_config.TextColumn("Macro-F1 "),
            "weighted_f1": st.column_config.TextColumn("Weighted-F1"),
            "training_time_sec": st.column_config.TextColumn("Training Time"),
            "loss_or_alpha": st.column_config.TextColumn("Hyperparameters"),
            "balancing": st.column_config.TextColumn("Class Balancing Strategy")
        }
    )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 3. Visualizations Row: Ranking & Multi-Metric
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.plotly_chart(create_model_ranking_bar(df_models), use_container_width=True)
    with c_chart2:
        st.plotly_chart(create_multi_metric_grouped_bar(df_models), use_container_width=True)

    # 4. Confusion Matrix & Per-Class Diagnostics
    st.markdown(
        textwrap.dedent("""
        <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-top: 1.5rem; margin-bottom: 0.5rem;">
             In-Depth Model Diagnostics
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    col_sel, col_empty = st.columns([1.5, 2.5])
    with col_sel:
        available_models = df_models['model'].tolist()
        selected_model = st.selectbox(
            "Select Model for Detailed Inspection:",
            options=available_models,
            index=0
        )

    c_cm, c_report = st.columns([1.2, 1])

    # Confusion matrix
    with c_cm:
        if cms and selected_model in cms:
            labels = cms.get("labels", ["Negative", "Neutral", "Positive"])
            matrix = cms[selected_model]
            st.plotly_chart(create_confusion_matrix_heatmap(matrix, labels, selected_model), use_container_width=True)
        else:
            st.info("Confusion matrix artifact not loaded.")

    # Per-Class Classification Report Table
    with c_report:
        st.markdown(f"#### Per-Class Breakdown ({selected_model})")

        if reports and selected_model in reports:
            rep = reports[selected_model]
            rep_rows = []
            for cls in ['Negative', 'Neutral', 'Positive']:
                if cls in rep:
                    rep_rows.append({
                        "Class": cls,
                        "Precision": f"{rep[cls]['precision']:.2f}",
                        "Recall": f"{rep[cls]['recall']:.2f}",
                        "F1-Score": f"{rep[cls]['f1-score']:.2f}",
                        "Support": f"{rep[cls]['support']:,}"
                    })
            df_rep = pd.DataFrame(rep_rows)
            st.dataframe(df_rep, use_container_width=True, hide_index=True)

            acc_val = rep.get('accuracy', 0.0)
            macro_f1_val = rep.get('macro avg', {}).get('f1-score', 0.0)
            st.markdown(
                textwrap.dedent(f"""
                <div style="margin-top: 1rem; font-size: 0.8rem; color: var(--text-secondary); line-height: 1.6;">
                    <div>• <b>Overall Accuracy:</b> {acc_val*100:.1f}%</div>
                    <div>• <b>Macro Average F1:</b> {macro_f1_val:.2f}</div>
                    <div>• <b>Total Evaluated:</b> {rep.get('macro avg', {}).get('support', support_total):,} reviews</div>
                </div>
                """).strip(),
                unsafe_allow_html=True
            )

    # Render settings from the persisted inference bundle rather than a copied config.
    if bundle:
        vectorizer = bundle.get("vectorizer")
        model = bundle.get("model")
        config = {
            "Inference model": bundle.get("model_name", type(model).__name__ if model else "Unavailable"),
            "Text column": bundle.get("text_column", "Unavailable"),
            "TF-IDF parameters": str(vectorizer.get_params()) if vectorizer else "Unavailable",
            "Model parameters": str(model.get_params()) if model else "Unavailable",
        }
        st.markdown("#### Serialized inference configuration")
        st.dataframe(pd.DataFrame(config.items(), columns=["Setting", "Value"]), use_container_width=True, hide_index=True)
