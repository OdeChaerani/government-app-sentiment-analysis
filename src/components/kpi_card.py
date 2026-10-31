"""
KPI Card Component for Government App Sentiment Intelligence Dashboard
"""

import textwrap
import streamlit as st


def _icon_for_label(label: str) -> str:
    key = label.lower()
    if "positive" in key or "positif" in key:
        shape = '<circle cx="12" cy="12" r="8"/><path d="m8.5 12 2.3 2.3 4.8-5"/>'
    elif "negative" in key or "negatif" in key or "conflict" in key:
        shape = '<circle cx="12" cy="12" r="8"/><path d="m9 9 6 6m0-6-6 6"/>'
    elif "neutral" in key or "netral" in key:
        shape = '<circle cx="12" cy="12" r="8"/><path d="M8.5 12h7"/>'
    elif "app" in key or "aplikasi" in key:
        shape = '<path d="M4 20h16M6 20V9l6-4 6 4v11M9 20v-6h6v6M4 9h16"/>'
    elif "group" in key or "cross" in key or "lintas" in key:
        shape = '<circle cx="6" cy="7" r="2"/><circle cx="18" cy="7" r="2"/><circle cx="12" cy="17" r="2"/><path d="m8 8 3 7m5-7-3 7m-5-8h8"/>'
    elif "model" in key or "macro" in key or "accuracy" in key or "akurasi" in key:
        shape = '<path d="M4 20h16M6 17v-5m6 5V5m6 12V9"/>'
    else:
        shape = '<path d="M6 4h8l4 4v12H6zM14 4v5h4M9 13h6m-6 3h6"/>'
    return f'<svg viewBox="0 0 24 24" aria-hidden="true"><g>{shape}</g></svg>'


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    icon: str = "",
    badge_text: str = "",
    badge_variant: str = "neutral",
    col=None
):
    """
    Renders a modern, rounded SaaS metric card.
    badge_variant can be: 'positive', 'neutral', 'negative', 'accent'
    """
    badge_html = ""
    if badge_text:
        badge_class = f"saas-badge-{badge_variant}" if badge_variant in ['positive', 'neutral', 'negative'] else "saas-badge"
        badge_html = f'<span class="saas-badge {badge_class}" style="margin: 0; padding: 2px 8px; font-size: 0.68rem;">{badge_text}</span>'

    card_html = textwrap.dedent(f"""<div class="kpi-card">
<div class="kpi-top">
<span class="kpi-label">{label}</span>
<div style="display: flex; align-items: center; gap: 6px;">
{badge_html}
<div class="kpi-icon-wrap">{_icon_for_label(label)}</div>
</div>
</div>
<div class="kpi-value">{value}</div>
<div class="kpi-subtext">{subtext}</div>
</div>""")

    if col is not None:
        col.markdown(card_html, unsafe_allow_html=True)
    else:
        st.markdown(card_html, unsafe_allow_html=True)
