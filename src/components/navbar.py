"""Branding, appearance controls, and primary navigation."""

import textwrap

import streamlit as st
from src.utils.i18n import translate


def render_sidebar_header() -> None:
    st.sidebar.markdown(
        textwrap.dedent("""
        <div class="brand-lockup">
            <div class="brand-mark" aria-hidden="true">
                <svg viewBox="0 0 32 32" role="img">
                    <path d="M6 25h20M9 25V12l7-5 7 5v13M13 25v-7h6v7M7 13h18" />
                </svg>
            </div>
            <div>
                <div class="brand-name">GovApp Intel</div>
                <div class="brand-caption">Sentiment Intelligence</div>
            </div>
        </div>
        <div class="brand-subline"><span class="status-dot"></span> IGAR Dataset <span class="brand-divider"></span> Research dashboard</div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_appearance_controls() -> None:
    st.sidebar.markdown('<div class="control-caption">Appearance</div>', unsafe_allow_html=True)
    theme_columns = st.sidebar.columns(2)
    if theme_columns[0].button(
        "Dark mode",
        key="theme_dark_button",
        type="primary" if st.session_state.ui_theme == "dark" else "secondary",
        use_container_width=True,
    ):
        st.session_state.ui_theme = "dark"
    if theme_columns[1].button(
        "Light mode",
        key="theme_light_button",
        type="primary" if st.session_state.ui_theme == "light" else "secondary",
        use_container_width=True,
    ):
        st.session_state.ui_theme = "light"

    st.sidebar.markdown('<div class="control-caption language-caption">Language</div>', unsafe_allow_html=True)
    language_columns = st.sidebar.columns(2)
    if language_columns[0].button(
        "ID",
        key="language_id_button",
        type="primary" if st.session_state.ui_language == "id" else "secondary",
        use_container_width=True,
    ):
        st.session_state.ui_language = "id"
    if language_columns[1].button(
        "EN",
        key="language_en_button",
        type="primary" if st.session_state.ui_language == "en" else "secondary",
        use_container_width=True,
    ):
        st.session_state.ui_language = "en"


def render_sidebar_nav(total_reviews: int | None = None, total_apps: int | None = None) -> str:
    """Renders navigation; stored values stay English for stable page routing."""
    pages = [
        "Overview",
        "Dataset Explorer",
        "Model Performance",
        "Cross-App Analysis",
        "Sentiment Predictor",
    ]
    st.sidebar.markdown('<div class="control-caption nav-caption">Workspace</div>', unsafe_allow_html=True)
    language = st.session_state.get("ui_language", "id")
    display_pages = [translate(page, language) for page in pages]
    previous_page = st.session_state.get("current_page", "Overview")
    selected_display = st.sidebar.radio(
        label="Navigation Menu",
        options=display_pages,
        index=pages.index(previous_page) if previous_page in pages else 0,
        label_visibility="collapsed",
        key=f"page_navigation_{language}",
    )
    selected = pages[display_pages.index(selected_display)]
    st.session_state.current_page = selected

    app_count = f"{total_apps:,}" if total_apps is not None else "—"
    review_count = f"{total_reviews:,}" if total_reviews is not None else "—"
    scope_markup = f"""
    <div class="scope-card">
        <div class="scope-title">Research scope</div>
        <div class="scope-line"><strong>{app_count}</strong> Indonesian Gov Apps</div>
        <div class="scope-line"><strong>{review_count}</strong> Curated Reviews</div>
        <div class="scope-foot">Cross-App Generalization · LOAO</div>
    </div>
    """
    st.sidebar.markdown(scope_markup, unsafe_allow_html=True)
    return selected
