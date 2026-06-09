"""Prediction file schema, loading and validation.

A predictions file is JSON with the shape::

    {
      "model": "my_model",
      "dataset": "timit",
      "predictions": {
        "<sample_key>": [
          {"word": "the", "start": 0.14, "end": 0.20},
          ...
        ]
      }
    }

``<sample_key>`` must match the ground-truth sample key (the dataset's ``audio``
field). Each word entry needs ``word``, ``start`` and ``end``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PredictionFile:
    """Parsed predictions for a single model/dataset."""

    model: str
    dataset: Optional[str]
    # sample_key -> (words, [(start, end), ...])
    predictions: Dict[str, Tuple[List[str], List[Tuple[float, float]]]] = field(
        default_factory=dict
    )


@dataclass
class ValidationReport:
    """Result of validating a predictions file against ground truth."""

    num_pred_samples: int
    num_gt_samples: int
    matched_keys: List[str]
    unknown_keys: List[str]
    missing_keys: List[str]

    @property
    def coverage(self) -> float:
        if self.num_gt_samples == 0:
            return 0.0
        return len(self.matched_keys) / self.num_gt_samples


def _parse_word_entries(entries) -> Tuple[List[str], List[Tuple[float, float]]]:
    words: List[str] = []
    times: List[Tuple[float, float]] = []
    for e in entries:
        if not isinstance(e, dict):
            raise ValueError(f"Each prediction entry must be an object, got: {e!r}")
        if "word" not in e or "start" not in e or "end" not in e:
            raise ValueError(
                f"Prediction entry missing required keys (word/start/end): {e!r}"
            )
        words.append(str(e["word"]))
        times.append((float(e["start"]), float(e["end"])))
    return words, times


def load_predictions(path: str) -> PredictionFile:
    """Load and parse a predictions JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    if "predictions" not in data or not isinstance(data["predictions"], dict):
        raise ValueError(
            "Predictions file must contain a 'predictions' object mapping "
            "sample keys to lists of word entries."
        )

    parsed: Dict[str, Tuple[List[str], List[Tuple[float, float]]]] = {}
    for key, entries in data["predictions"].items():
        if not isinstance(entries, list):
            raise ValueError(f"Predictions for '{key}' must be a list.")
        parsed[key] = _parse_word_entries(entries)

    return PredictionFile(
        model=str(data.get("model", "model")),
        dataset=data.get("dataset"),
        predictions=parsed,
    )


def validate_predictions(
    pred: PredictionFile, gt_keys: List[str]
) -> ValidationReport:
    """Compare prediction keys against ground-truth keys."""
    gt_set = set(gt_keys)
    pred_set = set(pred.predictions.keys())
    matched = sorted(pred_set & gt_set)
    unknown = sorted(pred_set - gt_set)
    missing = sorted(gt_set - pred_set)
    return ValidationReport(
        num_pred_samples=len(pred_set),
        num_gt_samples=len(gt_set),
        matched_keys=matched,
        unknown_keys=unknown,
        missing_keys=missing,
    )
