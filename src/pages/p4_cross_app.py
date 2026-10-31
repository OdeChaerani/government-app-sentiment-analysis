"""
Page 4: Cross-Application Generalization — Government App Sentiment Intelligence Dashboard
The core scientific research page evaluating Leave-One-App-Out (LOAO) domain transfer and vocabulary shifts.
"""

import textwrap
import pandas as pd
import streamlit as st

from src.components.kpi_card import render_kpi_card
from src.components.charts import (
    create_loao_performance_chart,
    create_baseline_vs_loao_chart
)
from src.utils.model_loader import load_loao_results, load_holdout_comparison
from src.utils.i18n import translate


def render_cross_app_page(df: pd.DataFrame | None = None):
    """Renders Page 4: Cross-Application Generalization."""
    st.markdown(
        textwrap.dedent("""
        <div class="saas-header-container">
            <span class="saas-badge" style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border-color: rgba(56, 189, 248, 0.3);">
                Core Research Objective
            </span>
            <h1 class="saas-title">Cross-Application Generalization (LOAO)</h1>
            <p class="saas-subtitle">
                Evaluating domain transferability across unseen government public services using Leave-One-App-Out (LOAO) cross-validation.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    df_loao = load_loao_results()
    df_holdout = load_holdout_comparison()
    app_count = int(df["app"].nunique()) if df is not None and not df.empty else len(df_loao)
    train_app_count = max(app_count - 1, 0)

    if df_loao.empty:
        st.markdown(
            textwrap.dedent("""
            <div class="empty-state">
                <div class="empty-state-icon"></div>
                <div class="empty-state-title">LOAO Benchmark Artifact Not Found</div>
                <div class="empty-state-desc">
                    The LOAO benchmark table <code>models/loao_results.csv</code> is not yet available.
                    Provide Member 2's saved <code>loao_results.csv</code> export. No retraining is needed.
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )
        return

    # The LOAO experiment uses Linear SVM; source the in-domain baseline from
    # the saved holdout table when it is available.
    baseline_f1 = None
    if not df_holdout.empty:
        svm_rows = df_holdout[df_holdout['model'] == 'LinearSVM_SGD']
        if not svm_rows.empty:
            baseline_f1 = float(svm_rows.iloc[0]['macro_f1'])

    # 1. KPI Summary Cards
    mean_loao_f1 = df_loao['macro_f1'].mean()
    std_loao_f1 = df_loao['macro_f1'].std()
    best_app_row = df_loao.sort_values('macro_f1', ascending=False).iloc[0]
    hardest_app_row = df_loao.sort_values('macro_f1', ascending=True).iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    render_kpi_card(
        label="Mean LOAO Macro-F1",
        value=f"{mean_loao_f1:.4f}",
        subtext=f"Std Dev: ±{std_loao_f1:.4f}",
        icon="",
        badge_text="Transfer Average",
        badge_variant="accent",
        col=k1
    )
    render_kpi_card(
        label="In-Domain Baseline",
        value=f"{baseline_f1:.4f}" if baseline_f1 is not None else "Unavailable",
        subtext="Random Split (GroupShuffleSplit)",
        icon="",
        badge_text="Reference Point",
        badge_variant="neutral",
        col=k2
    )
    render_kpi_card(
        label="Highest Transfer App",
        value=f"{best_app_row['target_app']}",
        subtext=f"Macro-F1: {best_app_row['macro_f1']:.4f}",
        icon="",
        badge_text="Easiest Domain",
        badge_variant="positive",
        col=k3
    )
    render_kpi_card(
        label="Hardest Transfer App",
        value=f"{hardest_app_row['target_app']}",
        subtext=f"Macro-F1: {hardest_app_row['macro_f1']:.4f}",
        icon="",
        badge_text="Highest Shift",
        badge_variant="negative",
        col=k4
    )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2. Methodological Briefing Card
    methodology_markup = textwrap.dedent("""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <div style="font-weight: 700; font-size: 1.05rem; color: var(--text-primary); margin-bottom: 0.5rem;">
                 Methodological Framework: Leave-One-App-Out (LOAO) Protocol
            </div>
            <div style="color: var(--text-secondary); font-size: 0.875rem; line-height: 1.6;">
                In the LOAO protocol, the model is trained exclusively on reviews from <b>five government applications</b> and evaluated on a <b>single unseen target application</b>. 
                Crucially, to prevent data leakage, all duplicate text groups present in the target application are excluded from the training fold.
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 1rem;">
                <div style="padding: 10px; background: var(--bg-main); border-radius: 8px; border-left: 3px solid #6366F1;">
                    <div style="color: var(--text-primary); font-weight: 700; font-size: 0.85rem;">1. Natural Cross-App Evaluation</div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 2px;">
                        Evaluates all reviews in the unseen target app. Reflects realistic transfer where common mobile service phrases (e.g., <i>"bagus"</i>, <i>"update terus"</i>, <i>"gagal login"</i>) naturally overlap.
                    </div>
                </div>
                <div style="padding: 10px; background: var(--bg-main); border-radius: 8px; border-left: 3px solid #38BDF8;">
                    <div style="color: var(--text-primary); font-weight: 700; font-size: 0.85rem;">2. Strict Cross-App Evaluation (Pure Unseen)</div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 2px;">
                        Evaluates <i>only</i> reviews whose text groups appear in exactly one app (<code>n_apps_in_group == 1</code>). Measures pure generalization against domain-specific jargon never seen in training.
                    </div>
                </div>
            </div>
        </div>
        """)
    methodology_markup = methodology_markup.replace("five government applications", f"{train_app_count} government applications")
    st.markdown(methodology_markup.strip(), unsafe_allow_html=True)

    # 3. LOAO Performance Matrix Table
    st.markdown(
        textwrap.dedent("""
        <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem;">
             LOAO Experimental Performance Matrix
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    loao_table = df_loao.copy()
    loao_table['train_rows'] = loao_table['train_rows'].apply(lambda x: f"{x:,}")
    loao_table['test_rows'] = loao_table['test_rows'].apply(lambda x: f"{x:,}")
    loao_table['accuracy'] = loao_table['accuracy'].apply(lambda x: f"{x*100:.2f}%")
    loao_table['macro_precision'] = loao_table['macro_precision'].apply(lambda x: f"{x*100:.2f}%")
    loao_table['macro_recall'] = loao_table['macro_recall'].apply(lambda x: f"{x*100:.2f}%")
    loao_table['macro_f1'] = loao_table['macro_f1'].apply(lambda x: f"{x:.4f}")
    if 'pure_test_rows_n_apps_eq_1' in loao_table.columns:
        loao_table['pure_test_rows_n_apps_eq_1'] = loao_table['pure_test_rows_n_apps_eq_1'].apply(lambda x: f"{x:,}")
    if 'pure_macro_f1_n_apps_eq_1' in loao_table.columns:
        loao_table['pure_macro_f1_n_apps_eq_1'] = loao_table['pure_macro_f1_n_apps_eq_1'].apply(lambda x: f"{x:.4f}")

    st.dataframe(
        loao_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "target_app": st.column_config.TextColumn("Unseen Target App"),
            "train_rows": st.column_config.TextColumn("Train Reviews"),
            "test_rows": st.column_config.TextColumn("Test Reviews"),
            "accuracy": st.column_config.TextColumn("Accuracy"),
            "macro_precision": st.column_config.TextColumn("Macro Precision"),
            "macro_recall": st.column_config.TextColumn("Macro Recall"),
            "macro_f1": st.column_config.TextColumn("Natural LOAO F1 "),
            "pure_test_rows_n_apps_eq_1": st.column_config.TextColumn("Strict Test Rows"),
            "pure_macro_f1_n_apps_eq_1": st.column_config.TextColumn("Strict LOAO F1 ")
        }
    )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # 4. Visualizations Row
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.plotly_chart(create_loao_performance_chart(df_loao), use_container_width=True)
    with c_chart2:
        if baseline_f1 is not None:
            st.plotly_chart(create_baseline_vs_loao_chart(baseline_f1, df_loao), use_container_width=True)
        else:
            st.info("The holdout baseline artifact is not available, so the baseline comparison is hidden.")

    # Data-driven notes: every displayed metric is computed from the saved
    # notebook results or the loaded processed dataset.
    st.markdown("### Findings from the saved experiment")
    top_rows = df_loao.sort_values("macro_f1", ascending=False).head(2)
    low_row = df_loao.sort_values("macro_f1", ascending=True).iloc[0]
    strict_drop = None
    if "pure_macro_f1_n_apps_eq_1" in df_loao.columns:
        strict_drop = (df_loao["macro_f1"] - df_loao["pure_macro_f1_n_apps_eq_1"]).mean()

    finding_cols = st.columns(2)
    with finding_cols[0]:
        top_summary = ", ".join(
            f"{row.target_app} ({row.macro_f1:.4f})" for row in top_rows.itertuples()
        )
        st.info(f"Highest LOAO Macro-F1 results: {top_summary}.")
        delta_text = (
            f" The difference from the holdout baseline is {low_row.macro_f1 - baseline_f1:+.4f}."
            if baseline_f1 is not None else ""
        )
        st.warning(
            f"Lowest transfer score: {low_row.target_app}, Macro-F1 {low_row.macro_f1:.4f}."
            f"{delta_text}"
        )
    with finding_cols[1]:
        if df is not None and not df.empty and "app" in df and "label" in df:
            app_reviews = df[df["app"].astype(str) == str(low_row.target_app)]
            support = len(app_reviews)
            shares = app_reviews["label"].value_counts(normalize=True)
            pos_share = shares.get("Positive", 0.0)
            neg_share = shares.get("Negative", 0.0)
            dataset_summary = (
                f"{low_row.target_app} has {support:,} reviews in the processed dataset; "
                f"{neg_share:.1%} are Negative and {pos_share:.1%} are Positive."
            )
            if st.session_state.get("ui_language", "id") == "id":
                dataset_summary = (
                    f"{low_row.target_app} memiliki {support:,} ulasan pada dataset yang telah diproses; "
                    f"{neg_share:.1%} berlabel Negatif dan {pos_share:.1%} berlabel Positif."
                )
            st.info(translate(dataset_summary))
        if strict_drop is not None:
            st.info(
                f"Across target apps, strict unseen-text Macro-F1 is lower by "
                f"{strict_drop:.4f} on average than natural LOAO Macro-F1."
            )
