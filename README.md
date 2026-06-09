# Nyra Speech Timing Benchmark

A small, model-agnostic benchmark for **word-level timing accuracy**. Given the
ground-truth word timings for a dataset and a model's predicted word timings, it
**normalizes**, **aligns** (edit distance) and **scores** the predictions with:

- **MAE** of word start/end boundaries (milliseconds),
- **F1 @ collar** at several tolerances (`F1@50ms` ... `F1@500ms`), in two variants,
- **mIoU** of aligned word intervals.

Ground truth is read at runtime from an existing data directory; no audio is
bundled into this repo.

## Install

```bash
pip install -e .
# optional: built-in WhisperX adapter
pip install -e ".[whisperx]"
```

This installs the `nyra-timing` CLI. `numpy` is the only required dependency.

## Datasets

Ground truth is loaded from `<data-root>/<dataset>/labels.json` (default
`--data-root /home/ubuntu/data/english_speech`). Registered datasets:

| dataset | split | test samples | notes |
| --- | --- | --- | --- |
| `timit` | test | ~1626 | read speech, one wav per sample |
| `fluency_bank_stutter` | test | ~300 | disfluent speech, one wav per sample |
| `buckeye` | test | ~2008 | conversational; chunked sub-segments of parent recordings |

Buckeye is read from `eval_chunks.json` rather than `labels.json`. Each chunk is a
sub-segment of a longer parent recording, with **chunk-relative** word timings, so
the benchmark slices the chunk (`chunk_start_s`..`chunk_end_s`) from the parent
wav before inference. Words come from each timing's `transcript_form`, normalized
to match the research baselines: `[UM]`/`[UH]`/`[laughter]` become `um`/`uh`/
`laughter`, and a trailing `*` (cut-off/mispronunciation marker) is stripped.

```bash
nyra-timing list-datasets
```

## Scoring a predictions file (model-agnostic, primary workflow)

Produce a predictions JSON with your model, then score it:

```bash
nyra-timing score --dataset timit --predictions my_preds.json --out results/
```

### Predictions file schema

```json
{
  "model": "my_model",
  "dataset": "timit",
  "predictions": {
    "<sample_key>": [
      {"word": "the", "start": 0.14, "end": 0.20},
      {"word": "bungalow", "start": 0.20, "end": 0.68}
    ]
  }
}
```

- `<sample_key>` must match the dataset's ground-truth key, which is the
  `audio` field from `labels.json` (e.g. `audio/TEST_DR4_MGMM0_SX139.wav`).
- Times are in **seconds**. Each word entry needs `word`, `start`, `end`.
- Samples missing from `predictions` are scored as empty and lower coverage.
- See [`examples/predictions_example.json`](examples/predictions_example.json).

## Running a built-in model adapter

The benchmark can run a model over the dataset audio and score the result in one
step:

```bash
nyra-timing run --dataset timit --model mfa --num-samples 50 --out results/
```

### Built-in models

| model | kind | model id | optional dependency |
| --- | --- | --- | --- |
| `whisperx` | ASR + alignment | `large-v3-turbo` + wav2vec2 | `whisperx` |
| `canary` | free ASR w/ timestamps | `nvidia/canary-1b-v2` | `nemo_toolkit[asr]` |
| `nemo_fa` | forced aligner | `stt_en_conformer_ctc_large` | `nemo_toolkit[asr]` |
| `ctc_segmentation` | forced aligner | `wav2vec2-base-960h` | `torchaudio`, `ctc-segmentation` |
| `qwen_fa` | forced aligner | `Qwen/Qwen3-ForcedAligner-0.6B` | `qwen-asr` |
| `mfa` | forced aligner | `english_mfa` | MFA binary + `english_mfa` model/dict |
| `crisperwhisper` | free ASR w/ attention timing | `large-v2_verbatimize_longform_with_timing` | local `crisperwhisper` + `torch`, `transformers` |

**Forced aligners** (`nemo_fa`, `ctc_segmentation`, `qwen_fa`, `mfa`) consume the
ground-truth words and only predict their timings; the `run` command passes the
reference text automatically. **Free ASR** models (`whisperx`, `canary`,
`crisperwhisper`) emit their own words, which are then text-aligned to the
ground truth before scoring. `crisperwhisper` reads per-word times off the
decoder cross-attention via a Viterbi over the encoder frames, with the
mel-energy blank reshaped (`blank_gamma=3`, `blank_penalty=3`) so it does not
collapse words on conversational speech.

Adapters import their heavy dependencies lazily, so a missing/incompatible
dependency only fails when that model is requested (with a clear message).

### Verified results (full test sets)

The model adapters were run end-to-end over the **complete** test set of each
corpus on this machine (`crisperwhisper` on TIMIT + Buckeye). Word counts:
TIMIT ≈ 12.6k, Buckeye ≈ 81k, FluencyBank (stutter) ≈ 9k. Lower MAE is better;
higher mIoU / F1 is better.

**TIMIT** (read speech, 1626 utterances):

| model | MAE (ms) | mIoU | F1@50ms | F1@100ms | F1@200ms |
| --- | --- | --- | --- | --- | --- |
| `mfa` | **23.5** | **82.9%** | **77.6%** | 94.5% | 99.1% |
| `qwen_fa` | 30.1 | 76.8% | 70.8% | **95.0%** | **99.8%** |
| `crisperwhisper` | 33.7 | 75.8% | 60.6% | 89.8% | 98.1% |
| `ctc_segmentation` | 49.3 | 69.1% | 38.9% | 82.6% | 96.9% |
| `nemo_fa` | 60.0 | 55.3% | 19.3% | 66.4% | 98.4% |
| `whisperx` | 64.8 | 56.5% | 13.3% | 59.5% | 96.4% |
| `canary` | 85.5 | 44.0% | 12.9% | 41.1% | 86.4% |

**Buckeye** (conversational, 2008 chunks, chunk-relative timings):

| model | MAE (ms) | mIoU | F1@50ms | F1@100ms | F1@200ms |
| --- | --- | --- | --- | --- | --- |
| `qwen_fa` | **43.7** | 70.8% | 74.5% | **93.9%** | **97.4%** |
| `crisperwhisper` | 45.6 | 64.8% | 51.8% | 79.0% | 88.7% |
| `mfa` | 51.8 | **77.9%** | **77.3%** | 90.1% | 94.8% |
| `nemo_fa` | 74.9 | 44.3% | 20.3% | 58.6% | 94.5% |
| `ctc_segmentation` | 83.9 | 63.3% | 48.5% | 74.7% | 85.6% |
| `whisperx` | 93.0 | 41.8% | 7.8% | 48.5% | 84.2% |
| `canary` | 117.7 | 29.8% | 8.9% | 30.1% | 71.2% |

**FluencyBank** (stuttered speech, 300 utterances — hard for all aligners):

| model | MAE (ms) | mIoU | F1@50ms | F1@100ms | F1@200ms |
| --- | --- | --- | --- | --- | --- |
| `qwen_fa` | **109.4** | 60.0% | 45.0% | **74.4%** | **86.4%** |
| `nemo_fa` | 124.2 | 41.7% | 18.1% | 49.2% | 81.8% |
| `canary` | 167.9 | 30.2% | 7.2% | 24.5% | 60.6% |
| `whisperx` | 184.7 | 44.1% | 7.5% | 39.7% | 66.7% |
| `ctc_segmentation` | 208.8 | 50.4% | 25.6% | 48.7% | 67.0% |
| `mfa` | 233.0 | **61.2%** | **47.7%** | 66.1% | 78.0% |

Notes:
- The NeMo-FA Buckeye MAE (74.9 ms) reproduces the independent forced-alignment
  ledger's "fair comparison" figure (74.90 ms), validating the pipeline.
- `crisperwhisper` is the only **free-decode** model competitive with the forced
  aligners (TIMIT 33.7 ms, Buckeye 45.6 ms): it transcribes verbatim and times
  its own words, so it is robust to ground-truth transcripts that do not conform
  to the model's tokenization (e.g. Buckeye's filler/cut-off forms). Teacher
  forcing those non-conforming transcripts is markedly worse (Buckeye ≈ 121 ms).
  The mel-blank shaping (`blank_gamma`/`blank_penalty`) is what keeps it from
  collapsing words on conversational speech (Buckeye 87 → 46 ms).
- On **FluencyBank**, pooled MAE is dominated by a tail of disfluency outliers, so
  it ranks differently from the boundary-centric F1/mIoU metrics. MFA has the best
  mIoU and F1@50ms there despite the worst MAE — i.e. its boundaries are tightest
  on the words it does place, but disfluencies blow up the mean. Read MAE and
  F1/mIoU together on conversational/stuttered data.

#### Environment notes for `ctc_segmentation` and `qwen_fa`

Both are now runnable; they needed environment fixes:

- **`ctc_segmentation`**: the wheel must match the active NumPy ABI. Rebuild it in
  place against NumPy 2:
  ```bash
  pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir ctc-segmentation
  ```
- **`qwen_fa`**: `qwen_asr` (0.0.6) needs an older `transformers` (~4.57) than the
  one NeMo/WhisperX require (5.x). Use a dedicated venv that reuses the system
  torch:
  ```bash
  python -m venv --system-site-packages /path/to/qwen_venv
  /path/to/qwen_venv/bin/pip install "transformers==4.57.1"
  /path/to/qwen_venv/bin/python -m nyra_timing_benchmark.cli run --dataset timit --model qwen_fa
  ```
  The adapter also installs a small compatibility shim so `qwen_asr`'s
  `@check_model_inputs()` decorator works on either transformers signature.
- **`whisperx`**: PyTorch ≥2.6 defaults `torch.load(weights_only=True)`, which
  rejects pyannote's VAD checkpoint. The adapter allowlists the needed
  `omegaconf` globals and forces `weights_only=False` for the (trusted, local)
  VAD checkpoint at load time, so no manual intervention is required.

### Large model downloads / full disk

NeMo/HF models are large. If the root disk is small, redirect caches and temp to
a bigger volume:

```bash
HF_HOME=/ephemeral/hf_cache TMPDIR=/ephemeral/tmp \
  nyra-timing run --dataset timit --model canary --num-samples 8 --out results/
```

### Adding your own model

Subclass `nyra_timing_benchmark.models.base.TimingModel`, implement
`predict(audio_path, transcript=None, words=None) -> [(word, start, end)]` (set
`requires_transcript = True` for forced aligners), and register it in
`nyra_timing_benchmark/models/registry.py`.

For models that are much faster in bulk, override
`predict_many(items) -> {key: [(word, start, end)]}` instead of (or in addition
to) `predict`. `MfaModel` does this: it prepares one corpus and runs a single
`mfa align` over the whole dataset (one model load) rather than one subprocess per
utterance, and `QwenForcedAligner` batches alignment calls. The default
`predict_many` simply loops `predict`.

## Hosted API providers

Local model adapters cover open models; hosted STT APIs are run separately under
`generation/` (mirroring the verbatim speech benchmark). Each
`generate_<provider>.py` calls a provider, parses its word timestamps, and writes
both the canonical scoring file (`predictions/<provider>_<dataset>.json`) and the
raw provider responses (`generation/outputs/raw/<provider>_<dataset>.json`).

```bash
pip install -r generation/requirements.txt
python generation/generate_deepgram_nova3.py --dataset all   # TIMIT + Buckeye
nyra-timing score --dataset timit --predictions predictions/deepgram_nova3_timit.json
```

Providers with word timings: **xAI Grok STT, ElevenLabs Scribe, Cartesia
Ink-Whisper, Deepgram Nova-3**. Providers that do **not** expose word-level
timestamps (kept for reference / raw transcripts only): **Inworld STT-1** (API
returns empty `wordTimestamps`) and **Azure MAI-Transcribe-1.5** (segment-level
only). Keys are read from the research repo `.env` (or `NYRA_ENV_FILE`). See
[`generation/README.md`](generation/README.md) for the full list and how to add a
provider.

### Verified provider results (full test sets)

**TIMIT** (1626 utterances):

| provider | MAE (ms) | mIoU | F1@50ms | F1@100ms | F1@200ms |
| --- | --- | --- | --- | --- | --- |
| `xai_grok_stt` | **37.1** | **71.5%** | **50.2%** | **91.7%** | 98.6% |
| `elevenlabs_scribe` | 51.3 | 64.3% | 29.1% | 78.2% | 98.3% |
| `deepgram_nova3` | 63.3 | 62.3% | 20.1% | 59.7% | 96.6% |
| `cartesia_ink_whisper` | 69.4 | 63.5% | 21.4% | 60.6% | 89.6% |

**Buckeye** (2008 chunks):

| provider | MAE (ms) | mIoU | F1@50ms | F1@100ms | F1@200ms |
| --- | --- | --- | --- | --- | --- |
| `xai_grok_stt` | **47.1** | **62.8%** | **46.2%** | **80.4%** | 89.3% |
| `elevenlabs_scribe` | 59.6 | 53.9% | 21.5% | 67.1% | 87.0% |
| `deepgram_nova3` | 88.4 | 49.5% | 13.7% | 47.0% | 82.3% |
| `cartesia_ink_whisper` | 125.7 | 59.7% | 32.8% | 59.8% | 74.7% |

xAI Grok STT is the strongest hosted API on both corpora — close to the best
local forced aligners (`mfa`, `qwen_fa`) despite being a free-transcription ASR.

## Comparing models

```bash
nyra-timing compare results/*.json
```

Prints a leaderboard of MAE, mIoU and F1 across collars.

## Metric definitions

Words are normalized (lowercase, strip leading/trailing punctuation) before
matching. Predictions are aligned to ground truth by minimal edit distance; only
exact (normalized) text matches contribute to timing metrics, so timing accuracy
is **decoupled from transcription errors**.

- **MAE**: mean absolute error of start and end boundaries over aligned word
  pairs. `overall = (start + end) / 2`. Reported pooled over all words (micro)
  and averaged per sample (macro). Signed biases are also reported (positive =
  prediction is late).
- **mIoU**: mean intersection-over-union of aligned word intervals.
- **F1 @ collar (standard)**: transcription-coupled. A predicted word is a true
  positive if an unmatched GT word has the same text and both boundaries are
  within the collar. Yields precision/recall/F1.
- **F1 @ collar (matched-word)**: timing-only. Among text-aligned pairs, the
  fraction whose boundaries are both within the collar.

Default collars: `0.05, 0.1, 0.2, 0.3, 0.4, 0.5` seconds. Override with
`--collars 0.05,0.1,0.2`.

## Library use

```python
from nyra_timing_benchmark import load_ground_truth, load_predictions, evaluate_dataset

gt = load_ground_truth("timit", data_root="/home/ubuntu/data/english_speech")
pred = load_predictions("my_preds.json")
result = evaluate_dataset(gt, pred.predictions, model="my_model", dataset="timit")
print(result.overall["mae_ms"]["overall"])
```
