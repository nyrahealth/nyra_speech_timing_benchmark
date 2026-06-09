#!/usr/bin/env python3
"""Generate word timings with ElevenLabs Scribe (speech-to-text).

POST https://api.elevenlabs.io/v1/speech-to-text with ``model_id=scribe_v1`` and
``timestamps_granularity=word`` returns ``words: [{text, start, end, type}]``
(seconds); we keep entries of ``type == "word"``. Reads ``ELEVENLABS_API_KEY``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import requests

from common import add_common_args, load_env_file, require_key, run_provider

PROVIDER_ID = "elevenlabs_scribe"
API_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def build_transcriber(api_key: str, model_id: str, language: str):
    def transcribe(audio_path: Path) -> Tuple[List[Tuple[str, float, float]], dict]:
        with audio_path.open("rb") as f:
            response = requests.post(
                API_URL,
                headers={"xi-api-key": api_key},
                data={
                    "model_id": model_id,
                    "timestamps_granularity": "word",
                    "language_code": language,
                },
                files={"file": (audio_path.name, f, "audio/wav")},
                timeout=180,
            )
        response.raise_for_status()
        data = response.json()
        triples: List[Tuple[str, float, float]] = []
        for w in data.get("words") or []:
            if w.get("type") not in (None, "word"):
                continue
            text = (w.get("text") or "").strip()
            start, end = w.get("start"), w.get("end")
            if text and start is not None and end is not None:
                triples.append((text, float(start), float(end)))
        return triples, data

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-id", default="scribe_v1")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    load_env_file()
    api_key = require_key("ELEVENLABS_API_KEY", args.api_key)
    run_provider(
        PROVIDER_ID,
        build_transcriber(api_key, args.model_id, args.language),
        args,
    )


if __name__ == "__main__":
    main()
