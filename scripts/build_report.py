#!/usr/bin/env python3
"""Build the leaderboard tables and the noise-robustness plot from stored results.

    python scripts/build_report.py            # writes docs/RESULTS.md + docs/noise_robustness.png
    python scripts/build_report.py --headline # also prints the README headline tables

Inputs: ``results_full/`` (local systems), ``results_api/`` (hosted APIs),
``results_robustness/`` (TIMIT + Gaussian noise) and ``results_speed/``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = [("buckeye", "Buckeye (conversational)"),
            ("fluency_bank_stutter", "FluencyBank (stuttered speech)"),
            ("timit", "TIMIT (read speech)")]
COLLARS = ["25ms", "50ms", "100ms", "200ms"]
SNRS = [None, 30.0, 20.0, 10.0, 5.0, 0.0]

# Result-file model name prefix -> public display name.
NAMES = {
    "nyra_fa-": "nyra_fa", "nyra_fa_pro-": "nyra_fa_pro", "qwen_fa-": "qwen_fa",
    "mfa-": "mfa", "ctc_segmentation-": "ctc_segmentation", "nemo_fa-": "nemo_fa",
    "crisperwhisper-transformers-large-v2_verbatimize_longform_with_timing_release":
        "crisperwhisper_2.0_large",
    "whisperx-": "whisperx", "canary-": "canary",
}
# Speed comes from the deployment backend where it differs from the scored one.
SPEED_FILE_OVERRIDE = {
    "crisperwhisper_2.0_large":
        "speed_crisperwhisper-ct2-large-v2_verbatimize_longform_with_timing_release_buckeye.json",
}
# Rows left out of the README headline (kept in docs/RESULTS.md).
NOT_IN_HEADLINE = {"cartesia_ink_whisper", "assemblyai_u2_disfl"}
NOISE_PLOT = ["nyra_fa_pro", "nyra_fa", "qwen_fa", "mfa", "elevenlabs_fa",
              "fish_transcribe_1_pro", "crisperwhisper_2.0_large", "xai_grok_stt"]


def display(model: str) -> str:
    for prefix, name in NAMES.items():
        if model == prefix or model.startswith(prefix):
            return name
    return model


def load(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def clean_rows(dataset: str) -> List[Dict]:
    rows = []
    for d in ("results_full", "results_api"):
        for p in glob.glob(os.path.join(ROOT, d, f"*_{dataset}.json")):
            r = load(p)
            if "overall" not in r or "crisperwhisper-ct2" in r["model"]:
                continue
            o, md = r["overall"], r.get("metadata") or {}
            name = display(r["model"])
            rows.append({
                "name": name,
                "aligner": md.get("system_type") == "forced_aligner",
                "hosted": md.get("deployment") == "hosted_api",
                "recall": o["word_recall"],
                "mae": o["mae_ms"]["overall"],
                "medae": o["median_ae_ms"]["overall"],
                "miou": o["miou"],
                "acc": {c: o["f1_matched"][c]["f1"] for c in COLLARS},
                "f1": {c: o["f1_standard"][c]["f1"] for c in COLLARS},
                "rtf": speed(name, r["model"]),
            })
    return rows


def speed(name: str, model: str) -> Optional[float]:
    fname = SPEED_FILE_OVERRIDE.get(name, f"speed_{model}_buckeye.json")
    p = os.path.join(ROOT, "results_speed", fname)
    return load(p)["rtf"] if os.path.exists(p) else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def fmt_rtf(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{x:.4f}" if x < 0.01 else f"{x:.3f}" if x < 1 else f"{x:.2f}"


def kind(r: Dict) -> str:
    return "hosted API" if r["hosted"] else "local"


def acc_table(rows: List[Dict], headline: bool) -> str:
    rows = [r for r in rows if not (headline and r["name"] in NOT_IN_HEADLINE)]
    out = []
    for aligner, title in ((True, "Forced aligners (given the transcript)"),
                           (False, "Transcription systems (no transcript; timed on recognised words)")):
        group = sorted((r for r in rows if r["aligner"] == aligner),
                       key=lambda r: -r["acc"]["50ms"])
        out.append(f"*{title}*\n")
        out.append("| System | Deployment | Word recall | MAE (ms) | mIoU | Acc@25ms | Acc@50ms "
                   "| Acc@100ms | Acc@200ms | RTF |")
        out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in group:
            accs = " | ".join(pct(r["acc"][c]) for c in COLLARS)
            out.append(f"| `{r['name']}` | {kind(r)} | {pct(r['recall'])} | {r['mae']:.1f} "
                       f"| {pct(r['miou'])} | {accs} | {fmt_rtf(r['rtf'])} |")
        out.append("")
    return "\n".join(out)


def f1_table(rows: List[Dict]) -> str:
    out = ["| System | Type | Word recall | MedAE (ms) | F1@25ms | F1@50ms | F1@100ms | F1@200ms |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: -r["f1"]["50ms"]):
        t = "forced aligner" if r["aligner"] else "transcription"
        f1s = " | ".join(pct(r["f1"][c]) for c in COLLARS)
        out.append(f"| `{r['name']}` | {t}, {kind(r)} | {pct(r['recall'])} | {r['medae']:.1f} | {f1s} |")
    return "\n".join(out)


def noise_rows() -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    for p in glob.glob(os.path.join(ROOT, "results_robustness", "*.json")):
        r = load(p)
        if "overall" not in r:
            continue
        snr = (r.get("condition") or {}).get("snr_db")
        o = r["overall"]
        data.setdefault(display(r["model"]), {})[snr] = {
            "f1": o["f1_standard"]["50ms"]["f1"],
            "acc": o["f1_matched"]["50ms"]["f1"],
            "recall": o["word_recall"],
        }
    return data


def noise_table(data: Dict[str, Dict], key: str) -> str:
    head = " | ".join("clean" if s is None else f"{s:.0f} dB" for s in SNRS)
    out = [f"| System | {head} |", "| --- |" + " ---: |" * len(SNRS)]
    for name in sorted(data, key=lambda n: -data[n].get(None, {}).get("f1", 0)):
        vals = [pct(data[name][s][key]) if s in data[name] else "–" for s in SNRS]
        out.append(f"| `{name}` | " + " | ".join(vals) + " |")
    return "\n".join(out)


def noise_plot(data: Dict[str, Dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(SNRS)))
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for name in NOISE_PLOT:
        if name not in data:
            continue
        ys = [100 * data[name][s]["f1"] for s in SNRS]
        style = "-" if name.startswith("nyra") else "--"
        ax.plot(xs, ys, style, marker="o", linewidth=2 if name.startswith("nyra") else 1.4,
                label=name)
    ax.set_xticks(xs, ["clean"] + [f"{s:.0f} dB" for s in SNRS[1:]])
    ax.set_xlabel("Gaussian noise (SNR)")
    ax.set_ylabel("F1@50ms (%)")
    ax.set_title("TIMIT under noise: words recognised and timed within 50 ms")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headline", action="store_true", help="Print the README headline tables.")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)

    noise = noise_rows()
    noise_plot(noise, os.path.join(ROOT, "docs", "noise_robustness.png"))

    parts = ["# Full results\n",
             "Generated by `scripts/build_report.py`. See the README for metric definitions.\n"]
    for ds, title in DATASETS:
        rows = clean_rows(ds)
        parts += [f"## {title}\n", "### Timing accuracy (matched words)\n", acc_table(rows, False),
                  "### F1 (recognition and timing combined)\n", f1_table(rows), ""]
    parts += ["## Noise robustness (TIMIT + Gaussian noise)\n",
              "### F1@50ms\n", noise_table(noise, "f1"), "",
              "### Acc@50ms (matched words only)\n", noise_table(noise, "acc"), "",
              "### Word recall\n", noise_table(noise, "recall"), ""]
    with open(os.path.join(ROOT, "docs", "RESULTS.md"), "w") as f:
        f.write("\n".join(parts))

    if args.headline:
        for ds, title in DATASETS[:2]:
            print(f"### {title}\n")
            print(acc_table(clean_rows(ds), True))
        print(noise_table({k: v for k, v in noise.items() if k in NOISE_PLOT}, "f1"))


if __name__ == "__main__":
    main()
