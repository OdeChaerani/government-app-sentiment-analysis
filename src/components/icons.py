"""Small inline vector icons used by dashboard UI elements."""


def sentiment_icon_svg(sentiment: str) -> str:
    paths = {
        "Positive": '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/>',
        "Negative": '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6m0-6-6 6"/>',
        "Neutral": '<circle cx="12" cy="12" r="9"/><path d="M8.5 12h7"/>',
    }
    shape = paths.get(sentiment, paths["Neutral"])
    return f'<svg class="sentiment-mark" viewBox="0 0 24 24" aria-hidden="true"><g>{shape}</g></svg>'
