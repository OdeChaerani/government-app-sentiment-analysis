"""
Main Streamlit Application Entry Point
Government App Sentiment Intelligence Dashboard (Member 3)
"""

from pathlib import Path
import sys
import streamlit as st

# Ensure repository root is on sys.path so src.* modules import seamlessly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="GovApp Sentiment Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.components.navbar import render_appearance_controls, render_sidebar_header, render_sidebar_nav
from src.components.charts import set_chart_theme
from src.pages.p1_overview import render_overview_page
from src.pages.p2_explorer import render_explorer_page
from src.pages.p3_model_performance import render_model_performance_page
from src.pages.p4_cross_app import render_cross_app_page
from src.pages.p5_predictor import render_predictor_page
from src.utils.data_loader import load_dataset_metadata
from src.utils.i18n import install_streamlit_translation


def inject_custom_styles(theme: str):
    """Injects the unified dark-first SaaS CSS stylesheet."""
    css_path = Path(__file__).resolve().parent / "styles" / "main.css"
    css_content = ""
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    variables = {
        "dark": {
            "--bg-main": "#0b1020", "--bg-sidebar": "#0e1729", "--bg-card": "#141f33",
            "--bg-card-hover": "#192842", "--bg-surface-2": "#1b2940", "--border-color": "#293951",
            "--text-primary": "#f5f8ff", "--text-secondary": "#a8b5c8", "--text-muted": "#8090a8",
            "--title-start": "#ffffff", "--title-end": "#b8c7e0", "--grid-line": "rgba(149,174,211,.045)",
            "--accent-primary": "#716bff", "--accent-primary-hover": "#5d57e8", "--accent-secondary": "#38bdf8",
        },
        "light": {
            "--bg-main": "#f2f5fb", "--bg-sidebar": "#ffffff", "--bg-card": "#ffffff",
            "--bg-card-hover": "#f4f7fc", "--bg-surface-2": "#edf2f9", "--border-color": "#d7e0ed",
            "--text-primary": "#18243a", "--text-secondary": "#465873", "--text-muted": "#64748b",
            "--title-start": "#18243a", "--title-end": "#465873", "--grid-line": "rgba(47,72,110,.055)",
            "--accent-primary": "#5148d8", "--accent-primary-hover": "#4039bd", "--accent-secondary": "#087ca4",
        },
    }[theme]
    variable_css = ";".join(f"{key}:{value}" for key, value in variables.items())
    st.markdown(f"<style>:root{{--shadow-card:0 12px 30px rgba(24,36,58,.09);{variable_css}}}</style>", unsafe_allow_html=True)


def main():
    """Main dashboard controller."""
    # 1. Apply Design System CSS
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("ui_language", "id")
    install_streamlit_translation()

    # 2. Load dataset metadata and render its current scope in the sidebar
    df = load_dataset_metadata()
    render_sidebar_header()
    render_appearance_controls()
    set_chart_theme(st.session_state.ui_theme)
    inject_custom_styles(st.session_state.ui_theme)
    selected_page = render_sidebar_nav(
        total_reviews=len(df),
        total_apps=int(df["app"].nunique()) if not df.empty else 0,
    )

    # 4. Route to Selected Page
    if selected_page == "Overview":
        render_overview_page(df)
    elif selected_page == "Dataset Explorer":
        render_explorer_page(df)
    elif selected_page == "Model Performance":
        render_model_performance_page()
    elif selected_page == "Cross-App Analysis":
        render_cross_app_page(df)
    elif selected_page == "Sentiment Predictor":
        render_predictor_page()
    else:
        render_overview_page(df)


if __name__ == "__main__":
    main()
