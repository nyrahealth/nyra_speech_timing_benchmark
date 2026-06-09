"""Evaluation orchestration: align predictions to ground truth and aggregate.

The same scoring code path is used whether predictions come from a file or from a
model adapter. Metrics are aggregated two ways:

* micro: pool raw errors / counts across all samples (headline numbers), and
* macro: average per-sample values (less sensitive to long utterances).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .align import align_words
from .metrics import (
    collar_label,
    mae_on_aligned,
    matched_f1_counts,
    prf,
    standard_segmentation_counts,
)

DEFAULT_COLLARS: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)

Timing = Tuple[float, float]


@dataclass
class SampleResult:
    key: str
    num_pred_words: int
    num_gt_words: int
    num_aligned: int
    start_errors: List[float] = field(default_factory=list)
    end_errors: List[float] = field(default_factory=list)
    start_signed: List[float] = field(default_factory=list)
    end_signed: List[float] = field(default_factory=list)
    ious: List[float] = field(default_factory=list)
    # collar -> (tp, fp, fn) standard ; (tp, count) matched
    standard_counts: Dict[float, Tuple[int, int, int]] = field(default_factory=dict)
    matched_counts: Dict[float, Tuple[int, int]] = field(default_factory=dict)

    @property
    def overall_mae_ms(self) -> Optional[float]:
        errs = self.start_errors + self.end_errors
        return float(np.mean(errs) * 1000.0) if errs else None

    @property
    def miou(self) -> Optional[float]:
        return float(np.mean(self.ious)) if self.ious else None


def evaluate_sample(
    pred_words: Sequence[str],
    pred_times: Sequence[Timing],
    gt_words: Sequence[str],
    gt_times: Sequence[Timing],
    key: str = "",
    collars: Sequence[float] = DEFAULT_COLLARS,
) -> SampleResult:
    """Align one prediction/GT pair and compute its raw metric accumulators."""
    alignment = align_words(pred_words, gt_words)
    errs = mae_on_aligned(pred_times, gt_times, alignment)

    result = SampleResult(
        key=key,
        num_pred_words=len(pred_words),
        num_gt_words=len(gt_words),
        num_aligned=len(alignment),
        start_errors=errs["start_errors"],
        end_errors=errs["end_errors"],
        start_signed=errs["start_signed"],
        end_signed=errs["end_signed"],
        ious=errs["ious"],
    )
    for c in collars:
        result.standard_counts[c] = standard_segmentation_counts(
            pred_words, pred_times, gt_words, gt_times, c
        )
        result.matched_counts[c] = matched_f1_counts(pred_times, gt_times, alignment, c)
    return result


@dataclass
class BenchmarkResult:
    model: str
    dataset: Optional[str]
    collars: List[float]
    overall: Dict = field(default_factory=dict)
    per_sample: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "collars": self.collars,
            "overall": self.overall,
            "per_sample": self.per_sample,
        }


def _mean_ms(values: List[float]) -> Optional[float]:
    return float(np.mean(values) * 1000.0) if values else None


def evaluate_dataset(
    ground_truth: Dict[str, "object"],
    predictions: Dict[str, Tuple[Sequence[str], Sequence[Timing]]],
    model: str = "model",
    dataset: Optional[str] = None,
    collars: Sequence[float] = DEFAULT_COLLARS,
) -> BenchmarkResult:
    """Score all ground-truth samples and aggregate.

    ``ground_truth`` maps sample key -> object with ``.words`` and ``.timings``.
    ``predictions`` maps sample key -> ``(words, timings)``. Ground-truth samples
    without a prediction are scored as empty (counted in coverage).
    """
    collars = list(collars)
    sample_results: List[SampleResult] = []

    for key, gt in ground_truth.items():
        pred_words, pred_times = predictions.get(key, ([], []))
        sr = evaluate_sample(
            pred_words, pred_times, gt.words, gt.timings, key=key, collars=collars
        )
        sample_results.append(sr)

    # --- micro (pooled) accumulators ---
    all_start_err: List[float] = []
    all_end_err: List[float] = []
    all_start_signed: List[float] = []
    all_end_signed: List[float] = []
    all_ious: List[float] = []
    std_counts = {c: [0, 0, 0] for c in collars}
    matched_counts = {c: [0, 0] for c in collars}

    n_pred_words = n_gt_words = n_aligned = 0
    n_scored = 0  # samples with at least one prediction word

    per_sample: List[Dict] = []
    macro_mae: List[float] = []
    macro_miou: List[float] = []

    for sr in sample_results:
        all_start_err += sr.start_errors
        all_end_err += sr.end_errors
        all_start_signed += sr.start_signed
        all_end_signed += sr.end_signed
        all_ious += sr.ious
        n_pred_words += sr.num_pred_words
        n_gt_words += sr.num_gt_words
        n_aligned += sr.num_aligned
        if sr.num_pred_words > 0:
            n_scored += 1
        for c in collars:
            tp, fp, fn = sr.standard_counts[c]
            std_counts[c][0] += tp
            std_counts[c][1] += fp
            std_counts[c][2] += fn
            mtp, mcount = sr.matched_counts[c]
            matched_counts[c][0] += mtp
            matched_counts[c][1] += mcount
        if sr.overall_mae_ms is not None:
            macro_mae.append(sr.overall_mae_ms)
        if sr.miou is not None:
            macro_miou.append(sr.miou)

        per_sample.append(
            {
                "key": sr.key,
                "num_pred_words": sr.num_pred_words,
                "num_gt_words": sr.num_gt_words,
                "num_aligned": sr.num_aligned,
                "overall_mae_ms": sr.overall_mae_ms,
                "miou": sr.miou,
            }
        )

    standard_f1 = {}
    matched_f1 = {}
    for c in collars:
        tp, fp, fn = std_counts[c]
        p, r, f1 = prf(tp, fp, fn)
        standard_f1[collar_label(c)] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        mtp, mcount = matched_counts[c]
        matched_f1[collar_label(c)] = {
            "f1": (mtp / mcount) if mcount > 0 else 0.0,
            "tp": mtp,
            "count": mcount,
        }

    overall = {
        "num_gt_samples": len(ground_truth),
        "num_samples_with_predictions": n_scored,
        "coverage": (n_scored / len(ground_truth)) if ground_truth else 0.0,
        "num_pred_words": n_pred_words,
        "num_gt_words": n_gt_words,
        "num_aligned_words": n_aligned,
        "mae_ms": {
            "overall": _mean_ms(all_start_err + all_end_err),
            "start": _mean_ms(all_start_err),
            "end": _mean_ms(all_end_err),
            "start_bias": _mean_ms(all_start_signed),
            "end_bias": _mean_ms(all_end_signed),
        },
        "mae_ms_macro": float(np.mean(macro_mae)) if macro_mae else None,
        "miou": float(np.mean(all_ious)) if all_ious else None,
        "miou_macro": float(np.mean(macro_miou)) if macro_miou else None,
        "f1_standard": standard_f1,
        "f1_matched": matched_f1,
    }

    return BenchmarkResult(
        model=model,
        dataset=dataset,
        collars=collars,
        overall=overall,
        per_sample=per_sample,
    )
