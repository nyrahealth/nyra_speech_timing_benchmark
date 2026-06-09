"""Word normalization for timing-benchmark text matching.

Normalization is intentionally lightweight: lowercase and strip
leading/trailing punctuation (including brackets/parens around non-speech tags
and trailing cut-off hyphens) while preserving internal characters such as
apostrophes and compound hyphens. This standardises fillers and cut-offs so
ASR predictions match conversational ground truth (e.g. Buckeye).
"""

from __future__ import annotations

from typing import Optional


_FILLER_CANON = {
    "uhm": "um",
    "ahm": "um",
    "hmm": "hm",
    "mmm": "hm",
    "mhm": "hm",
    "mm": "hm",
    "mm-hmm": "hm",
    "uh-huh": "uhhuh",
    "umhum": "uhhuh",
}


def normalize_word(word: Optional[str]) -> str:
    """Normalize a word for content matching.

    Lowercases and strips *all* leading/trailing punctuation -- including the
    square brackets and parentheses that ASR models emit around non-speech
    tags. This standardises fillers so a model's ``[UH]``/``(laughter)``
    matches Buckeye's ``uh``/``laughter`` ground truth, and drops cut-off
    markers (a trailing hyphen on a truncated word, e.g. ``reti-`` ->
    ``reti``). Internal characters (apostrophes, compound hyphens like
    ``seventy-seven``) are kept.

    Returns an empty string for ``None`` or whitespace/punctuation-only input,
    which callers use to drop spacer/empty tokens.
    """
    if word is None:
        return ""
    w = word.lower().strip()
    while w and not w[0].isalnum():
        w = w[1:]
    while w and not w[-1].isalnum():
        w = w[:-1]
    return _FILLER_CANON.get(w, w)
