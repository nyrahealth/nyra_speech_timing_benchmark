"""Edit-distance word alignment.

Aligns predicted words to ground-truth words using Levenshtein DP and returns the
``(pred_idx, gt_idx)`` index pairs for positions that match exactly (after
normalization). These matched pairs are what timing metrics are computed on, so
that timing accuracy is decoupled from transcription errors.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .normalize import normalize_word


def align_words(pred_words: Sequence[str], gt_words: Sequence[str]) -> List[Tuple[int, int]]:
    """Align predicted to ground-truth words via edit distance.

    Returns a list of ``(pred_idx, gt_idx)`` pairs for exact (normalized) word
    matches along the minimal edit-distance path.
    """
    p = [normalize_word(w) for w in pred_words]
    g = [normalize_word(w) for w in gt_words]
    n, m = len(p), len(g)

    if n == 0 or m == 0:
        return []

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: List[List[str]] = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if p[i - 1] == g[j - 1] else 1
            choices = [
                (dp[i - 1][j] + 1, "del"),
                (dp[i][j - 1] + 1, "ins"),
                (dp[i - 1][j - 1] + sub_cost, "sub"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])

    i, j = n, m
    matches: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        b = back[i][j]
        if b == "sub":
            if p[i - 1] == g[j - 1]:
                matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif b == "del":
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches
