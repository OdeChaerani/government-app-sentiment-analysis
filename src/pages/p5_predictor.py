"""
Page 5: AI Sentiment Analyzer — Government App Sentiment Intelligence Dashboard
Interactive NLP inference interface integrating Member 1's exact preprocessing and Member 2's trained model.
"""

import textwrap
from typing import Dict, Any
import streamlit as st

from src.components.charts import create_probability_gauge_or_bar
from src.components.icons import sentiment_icon_svg
from src.utils.model_loader import load_model_bundle, run_sentiment_inference
from src.utils.text_preprocessor import clean_text, analyze_text_features


def render_predictor_page():
    """Renders Page 5: AI Sentiment Analyzer."""
    st.markdown(
        textwrap.dedent("""
        <div class="saas-header-container">
            <span class="saas-badge" style="background: rgba(34, 197, 94, 0.15); color: #22C55E; border-color: rgba(34, 197, 94, 0.3);">
                Live AI Inference
            </span>
            <h1 class="saas-title">AI Sentiment Analyzer</h1>
            <p class="saas-subtitle">
                Test real-time sentiment prediction on custom Indonesian reviews using Member 1's preprocessing pipeline and Member 2's trained classifier.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    # 1. Load Trained Model Bundle
    bundle = load_model_bundle()

    if bundle is None:
        st.markdown(
            textwrap.dedent("""
            <div class="empty-state">
                <div class="empty-state-icon"></div>
                <div class="empty-state-title">Model Bundle Artifact Unavailable</div>
                <div class="empty-state-desc">
                    The serialized inference bundle <code>models/rating_classifier_bundle.pkl</code> is not currently loaded.
                    <br><br>
                    To enable live AI inference, Member 2's trained model and TF-IDF vectorizer must be exported. 
                    The dashboard is fully equipped with Member 1's preprocessing pipeline and will immediately activate once the bundle is detected.
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )
        return

    # 2. Model Selection Banner
    model_name = bundle.get("model_name", "Multinomial Naive Bayes")
    metrics = bundle.get("metrics", {})
    macro_f1_val = metrics.get("macro_f1")
    acc_val = metrics.get("accuracy")
    benchmark_text = (
        f"Macro-F1 {macro_f1_val:.4f} • Acc {acc_val*100:.1f}%"
        if macro_f1_val is not None and acc_val is not None
        else "Benchmark metrics unavailable"
    )

    st.markdown(
        textwrap.dedent(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 12px 18px; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #22C55E; box-shadow: 0 0 10px #22C55E;"></div>
                <span style="font-size: 0.9rem; font-weight: 700; color: var(--text-primary);">Active Inference Engine:</span>
                <span style="font-size: 0.9rem; color: #6366F1; font-weight: 600;">{model_name} (TF-IDF N-Gram)</span>
            </div>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">
                Holdout Benchmark: <span style="color: var(--text-primary); font-weight: 600;">{benchmark_text}</span>
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    # 3. Preset Quick Examples
    st.markdown(
        textwrap.dedent("""
        <div style="font-size: 0.825rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px;">
             Quick Example Presets:
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

    with preset_col1:
        if st.button(" Positive Praise", use_container_width=True):
            st.session_state["review_input"] = "Alhamdulillah aplikasi sangat bagus dan lancar, klaim JHT cepat cair tanpa kendala sama sekali. Mantap BPJS!"

    with preset_col2:
        if st.button(" Critical Failure", use_container_width=True):
            st.session_state["review_input"] = "Aplikasi jelek bgt, login gagal terus tiap mau beli bbm, server lemot parah dan sering force close!"

    with preset_col3:
        if st.button(" Neutral Feedback", use_container_width=True):
            st.session_state["review_input"] = "Aplikasi sudah lumayan bagus tapi tolong update jadwal kereta diperbaiki agar tidak membingungkan pengguna."

    with preset_col4:
        if st.button(" Heavy Slang Review", use_container_width=True):
            st.session_state["review_input"] = "gx bs dipake sm skli, pdhl udh dftr dri kmrn tp msh ttp eror trs knp y, tlg diperbaiki dng bngt"

    # 4. Text Input Area
    default_val = st.session_state.get("review_input", "")
    user_text = st.text_area(
        label="Review Text Input:",
        value=default_val,
        height=140,
        placeholder="Write an Indonesian government application review here...",
        help="Input authentic public review text. Indonesian slang, elongated words, and Unicode stylized fonts are automatically normalized."
    )

    # Real-time counter indicators
    char_len = len(user_text)
    word_len = len(user_text.split())

    c_cnt1, c_cnt2, c_act1, c_act2 = st.columns([1, 1, 1, 1])
    with c_cnt1:
        st.markdown(f"<span style='font-size: 0.78rem; color: var(--text-muted);'>Characters: <b style='color:var(--text-secondary);'>{char_len}</b></span>", unsafe_allow_html=True)
    with c_cnt2:
        st.markdown(f"<span style='font-size: 0.78rem; color: var(--text-muted);'>Words: <b style='color:var(--text-secondary);'>{word_len}</b></span>", unsafe_allow_html=True)
    
    with c_act1:
        analyze_btn = st.button(" Analyze Sentiment", type="primary", use_container_width=True)
    with c_act2:
        if st.button(" Clear Input", use_container_width=True):
            st.session_state["review_input"] = ""
            st.rerun()

    # 5. Prediction Execution
    if analyze_btn or (user_text.strip() and user_text != default_val):
        if not user_text.strip():
            st.warning("Please provide a review text before running sentiment analysis.")
            return

        with st.spinner("Analyzing semantic features..."):
            result = run_sentiment_inference(user_text, bundle)

        if "error" in result:
            st.error(result["error"])
            return

        pred_label = result["predicted_label"]
        has_proba = result["has_probabilities"]
        probs = result["probabilities"]
        conf = result["confidence_score"]
        stats = result["text_stats"]
        interpretation = result["interpretation"]

        status_color = "#22C55E" if pred_label == "Positive" else "#F87171" if pred_label == "Negative" else "#F59E0B"
        status_icon = sentiment_icon_svg(pred_label)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # Result Display Panel
        col_res_left, col_res_right = st.columns([1.1, 1.3])

        with col_res_left:
            st.markdown(
                textwrap.dedent(f"""
                <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-top: 4px solid {status_color}; border-radius: 14px; padding: 1.5rem; height: 100%;">
                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-secondary); font-weight: 700; margin-bottom: 8px;">
                        Predicted Sentiment Class
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                        <span style="font-size: 2.2rem; font-weight: 800; color: {status_color}; letter-spacing: -0.02em;">
                            {pred_label}
                        </span>
                        <span style="color: {status_color}; display: inline-flex; align-items: center;">{status_icon}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 12px; line-height: 1.5;">
                        {interpretation}
                    </div>
                    <div style="padding-top: 10px; border-top: 1px solid var(--border-color); font-size: 0.8rem; color: var(--text-secondary);">
                        Confidence Score: <b style="color: var(--text-primary);">{conf if conf is not None else 'N/A (Decision margin)'}%</b>
                    </div>
                </div>
                """).strip(),
                unsafe_allow_html=True
            )

        with col_res_right:
            if has_proba and probs:
                st.plotly_chart(create_probability_gauge_or_bar(probs, height=220), use_container_width=True)
            else:
                st.markdown(
                    textwrap.dedent("""
                    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 1.5rem;">
                        <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 8px;">Confidence Calibration Note</div>
                        <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">
                            This classifier does not output genuine posterior probabilities. The decision margin indicates hyperplane distance and is not presented as calibrated percentage confidence.
                        </div>
                    </div>
                    """).strip(),
                    unsafe_allow_html=True
                )

        # 6. NLP Preprocessing Breakdown
        st.markdown(
            textwrap.dedent(f"""
            <div style="margin-top: 1.5rem; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 1.25rem;">
                <div style="font-weight: 700; font-size: 0.95rem; color: var(--text-primary); margin-bottom: 8px;">
                     Preprocessing Pipeline Diagnostics (Member 1 Integration)
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.825rem;">
                    <div style="background: var(--bg-main); padding: 10px; border-radius: 8px; border: 1px solid var(--border-color);">
                        <div style="color: var(--text-secondary); font-weight: 600; margin-bottom: 4px;">Original User Input:</div>
                        <div style="color: var(--text-secondary); font-family: inherit;">"{user_text}"</div>
                    </div>
                    <div style="background: var(--bg-main); padding: 10px; border-radius: 8px; border: 1px solid var(--border-color);">
                        <div style="color: #6366F1; font-weight: 600; margin-bottom: 4px;">Normalized Cleaned Text (NFKD + Slang Normalized):</div>
                        <div style="color: var(--text-primary); font-family: 'JetBrains Mono', monospace;">"{stats['cleaned_text']}"</div>
                    </div>
                </div>
            """).strip(),
            unsafe_allow_html=True
        )

        # Display detected slang transformations
        if stats["detected_slang"]:
            slang_chips = " ".join([f"<span class='saas-badge' style='margin-right: 4px; font-size: 0.7rem;'>{orig}  {norm}</span>" for orig, norm in stats["detected_slang"].items()])
            st.markdown(
                textwrap.dedent(f"""
                <div style="margin-top: 10px; font-size: 0.8rem; color: var(--text-secondary);">
                    <b>Detected Slang / Abbreviations Normalized:</b> {slang_chips}
                </div>
                """).strip(),
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # 7. Academic Research Disclaimer
    st.markdown(
        textwrap.dedent("""
        <div style="margin-top: 2rem; padding: 12px 16px; border-radius: 10px; background: color-mix(in srgb, var(--bg-card) 82%, transparent); border: 1px solid var(--border-color); font-size: 0.75rem; color: var(--text-muted); text-align: center;">
             <b>Academic Disclaimer:</b> Predictions are machine learning estimates based on patterns learned from training data and are not manually verified sentiment assessments.
        </div>
        """).strip(),
        unsafe_allow_html=True
    )
