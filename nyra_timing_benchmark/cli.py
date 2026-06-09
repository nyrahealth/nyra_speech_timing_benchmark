"""Command-line interface for the timing benchmark.

Subcommands:
    list-datasets   show registered datasets and their test-sample counts
    score           score a predictions file against a dataset
    run             run a built-in model adapter, then score its predictions
    compare         build a leaderboard from several result JSON files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from .audio_io import slice_wav
from .datasets import (
    DEFAULT_DATA_ROOT,
    get_spec,
    labels_path,
    list_datasets,
    load_ground_truth,
)
from .evaluate import DEFAULT_COLLARS, BenchmarkResult, evaluate_dataset
from .metrics import collar_label
from .schema import load_predictions, validate_predictions


def _parse_collars(value: Optional[str]) -> List[float]:
    if not value:
        return list(DEFAULT_COLLARS)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _fmt(v, suffix: str = "", pct: bool = False, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:.{nd}f}%"
    return f"{v:.{nd}f}{suffix}"


def _print_summary(result: BenchmarkResult) -> None:
    o = result.overall
    mae = o["mae_ms"]
    print(f"\n{'=' * 64}")
    print(f"Model:   {result.model}")
    print(f"Dataset: {result.dataset}")
    print(f"{'=' * 64}")
    print(
        f"Samples: {o['num_samples_with_predictions']}/{o['num_gt_samples']} "
        f"with predictions (coverage {_fmt(o['coverage'], pct=True)})"
    )
    print(
        f"Words:   pred={o['num_pred_words']}  gt={o['num_gt_words']}  "
        f"aligned={o['num_aligned_words']}"
    )
    print("\nWord boundary MAE (pooled):")
    print(f"  overall : {_fmt(mae['overall'], 'ms')}")
    print(f"  start   : {_fmt(mae['start'], 'ms')}  (bias {_fmt(mae['start_bias'], 'ms')})")
    print(f"  end     : {_fmt(mae['end'], 'ms')}  (bias {_fmt(mae['end_bias'], 'ms')})")
    print(f"  macro   : {_fmt(o['mae_ms_macro'], 'ms')} (mean per-sample)")
    print(f"mIoU     : {_fmt(o['miou'], pct=True)} (pooled)")

    print("\nF1 @ collar (standard, transcription-coupled):")
    for label, m in o["f1_standard"].items():
        print(
            f"  F1@{label:<6} {_fmt(m['f1'], pct=True):>7}  "
            f"(P {_fmt(m['precision'], pct=True)} / R {_fmt(m['recall'], pct=True)})"
        )

    print("\nF1 @ collar (matched-word, timing-only):")
    for label, m in o["f1_matched"].items():
        print(f"  F1@{label:<6} {_fmt(m['f1'], pct=True):>7}  (tp {m['tp']}/{m['count']})")
    print()


def _save_result(result: BenchmarkResult, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    safe_model = result.model.replace("/", "_").replace(" ", "_")
    fname = f"{safe_model}_{result.dataset}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def cmd_list_datasets(args: argparse.Namespace) -> int:
    print(f"Data root: {args.data_root}\n")
    print(f"{'dataset':<24} {'split':<8} {'test_samples':<14} labels.json")
    print("-" * 72)
    for name in list_datasets():
        spec = get_spec(name)
        path = labels_path(name, args.data_root)
        if os.path.exists(path):
            try:
                gt = load_ground_truth(name, args.data_root)
                count = str(len(gt))
            except Exception as e:  # pragma: no cover
                count = f"error: {e}"
        else:
            count = "missing"
        print(f"{name:<24} {spec.default_split:<8} {count:<14} {path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    collars = _parse_collars(args.collars)
    gt = load_ground_truth(args.dataset, args.data_root, split=args.split)

    pred = load_predictions(args.predictions)
    if pred.dataset and pred.dataset != args.dataset:
        print(
            f"Warning: predictions file dataset '{pred.dataset}' != "
            f"--dataset '{args.dataset}'.",
            file=sys.stderr,
        )

    model_name = args.model_name or pred.model
    report = validate_predictions(pred, list(gt.keys()))
    print(
        f"Validation: {len(report.matched_keys)} matched, "
        f"{len(report.unknown_keys)} unknown keys, "
        f"{len(report.missing_keys)} GT samples missing predictions."
    )
    if report.unknown_keys:
        preview = ", ".join(report.unknown_keys[:3])
        print(f"  e.g. unknown keys: {preview} ...", file=sys.stderr)

    result = evaluate_dataset(
        gt, pred.predictions, model=model_name, dataset=args.dataset, collars=collars
    )
    _print_summary(result)

    if args.out:
        path = _save_result(result, args.out)
        print(f"Saved results to: {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import shutil
    import tempfile

    from .models.base import PredictItem
    from .models.registry import build_model

    collars = _parse_collars(args.collars)
    gt = load_ground_truth(args.dataset, args.data_root, split=args.split)

    keys = list(gt.keys())
    if args.num_samples:
        keys = keys[: args.num_samples]

    # Build work items. Chunked datasets (Buckeye) are sliced from their parent
    # recording into a temp corpus so timings are chunk-relative; the sliced files
    # must persist for the duration of (batched) prediction.
    slice_dir: Optional[str] = None
    items: List[PredictItem] = []
    for key in keys:
        sample = gt[key]
        if not sample.audio_path or not os.path.exists(sample.audio_path):
            print(f"  audio not found, skipping {key}: {sample.audio_path}")
            continue
        audio_path = sample.audio_path
        if sample.chunk_start is not None and sample.chunk_end is not None:
            if slice_dir is None:
                slice_dir = tempfile.mkdtemp(prefix="nyra_slice_")
            out_wav = os.path.join(slice_dir, f"{key.replace('/', '_')}.wav")
            try:
                audio_path = slice_wav(
                    sample.audio_path, sample.chunk_start, sample.chunk_end, out_wav
                )
            except Exception as e:
                print(f"  slice failed for {key}: {e}")
                continue
        items.append(
            PredictItem(
                key=key,
                audio_path=audio_path,
                words=list(sample.words),
                transcript=sample.transcript,
            )
        )

    print(f"Running {args.model} on {len(items)} utterances...")
    model = build_model(args.model)
    predictions: Dict[str, tuple] = {}
    try:
        raw = model.predict_many(items)
        for key, triples in raw.items():
            words = [w for w, _, _ in triples]
            times = [(s, e) for _, s, e in triples]
            predictions[key] = (words, times)
    finally:
        model.close()
        if slice_dir:
            shutil.rmtree(slice_dir, ignore_errors=True)

    # Restrict ground truth to the keys we attempted, for honest coverage.
    gt_subset = {k: gt[k] for k in keys}
    result = evaluate_dataset(
        gt_subset, predictions, model=model.name, dataset=args.dataset, collars=collars
    )
    _print_summary(result)

    if args.out:
        path = _save_result(result, args.out)
        print(f"Saved results to: {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    rows = []
    collar_labels: List[str] = []
    for path in args.results:
        with open(path, "r") as f:
            data = json.load(f)
        o = data["overall"]
        if not collar_labels:
            collar_labels = list(o["f1_standard"].keys())
        rows.append(
            {
                "model": data.get("model", os.path.basename(path)),
                "dataset": data.get("dataset", ""),
                "mae": o["mae_ms"]["overall"],
                "miou": o["miou"],
                "f1": {lbl: o["f1_standard"][lbl]["f1"] for lbl in collar_labels},
            }
        )

    header = f"{'model':<28} {'dataset':<20} {'MAE(ms)':>9} {'mIoU':>7}"
    for lbl in collar_labels:
        header += f" {('F1@' + lbl):>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{r['model'][:27]:<28} {str(r['dataset'])[:19]:<20} "
            f"{_fmt(r['mae'], '') :>9} {_fmt(r['miou'], pct=True):>7}"
        )
        for lbl in collar_labels:
            line += f" {_fmt(r['f1'][lbl], pct=True):>9}"
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nyra-timing",
        description="Speech timing accuracy benchmark (MAE, F1@xms, mIoU).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-datasets", help="List registered datasets.")
    p_list.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_list.set_defaults(func=cmd_list_datasets)

    p_score = sub.add_parser("score", help="Score a predictions file.")
    p_score.add_argument("--dataset", required=True)
    p_score.add_argument("--predictions", required=True, help="Path to predictions JSON.")
    p_score.add_argument("--model-name", default=None)
    p_score.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_score.add_argument("--split", default=None)
    p_score.add_argument("--collars", default=None, help="Comma-separated seconds.")
    p_score.add_argument("--out", default=None, help="Directory to write result JSON.")
    p_score.set_defaults(func=cmd_score)

    from .models.registry import list_models

    p_run = sub.add_parser("run", help="Run a built-in model adapter, then score.")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument(
        "--model",
        required=True,
        help=f"Built-in model name. Available: {', '.join(list_models())}.",
    )
    p_run.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_run.add_argument("--split", default=None)
    p_run.add_argument("--num-samples", type=int, default=None)
    p_run.add_argument("--collars", default=None, help="Comma-separated seconds.")
    p_run.add_argument("--out", default=None, help="Directory to write result JSON.")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="Compare multiple result JSON files.")
    p_cmp.add_argument("results", nargs="+", help="Result JSON files.")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
