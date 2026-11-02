"""Cartesia ``ink-whisper`` over the streaming STT WebSocket.

``wss://api.cartesia.ai/stt/websocket`` with ``model=ink-whisper`` returns
final ``transcript`` events carrying ``words: [{word, start, end}]``. Unlike the
batch ``/stt`` endpoint (``cartesia_ink_whisper``), it does not start the first
word at 0 s. Audio is sent as 16-bit mono PCM at its native rate in 100 ms
chunks without real-time pacing, then ``finalize`` + ``close``. ``ink-2`` is not
usable here: it returns text without word timings.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import PredictItem, TimingModel

WS_URL = "wss://api.cartesia.ai/stt/websocket"
CARTESIA_VERSION = "2026-03-01"
_GEN_DIR = Path(__file__).resolve().parents[2] / "generation"


class CartesiaStreamModel(TimingModel):
    requires_transcript = False

    def __init__(
        self,
        model: str = "ink-whisper",
        language: str = "en",
        max_concurrency: int = 8,
        max_retries: int = 4,
    ):
        if str(_GEN_DIR) not in sys.path:
            sys.path.insert(0, str(_GEN_DIR))
        from common import load_env_file, require_key

        load_env_file()
        self._key = require_key("CARTESIA_API_KEY")
        self.model = model
        self.language = language
        self.name = "cartesia_ink_whisper_stream"
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.raw_responses: Dict[str, object] = {}

    async def _transcribe(self, audio_path: str):
        import websockets

        from ..noise import load_audio

        y, sr = load_audio(audio_path)
        pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        url = (
            f"{WS_URL}?model={self.model}&language={self.language}"
            f"&encoding=pcm_s16le&sample_rate={sr}"
            f"&api_key={self._key}&cartesia_version={CARTESIA_VERSION}"
        )
        events = []
        async with websockets.connect(url, max_size=None) as ws:
            step = sr // 10 * 2
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i : i + step])
            await ws.send("finalize")
            await ws.send("close")
            async for msg in ws:
                d = json.loads(msg)
                events.append(d)
                if d.get("type") == "error":
                    raise RuntimeError(d.get("message") or str(d))
                if d.get("type") == "done":
                    break
        words: List[Tuple[str, float, float]] = []
        for d in events:
            if d.get("type") != "transcript" or not d.get("is_final"):
                continue
            for w in d.get("words") or []:
                t = (w.get("word") or "").strip()
                s, e = w.get("start"), w.get("end")
                if t and s is not None and e is not None:
                    words.append((t, float(s), float(e)))
        return words, events

    async def _with_retries(self, audio_path: str):
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                return await self._transcribe(audio_path)
            except Exception:
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        return asyncio.run(self._with_retries(audio_path))[0]

    def predict_many(
        self, items: Sequence[PredictItem], progress_every: int = 100
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        out: Dict[str, List[Tuple[str, float, float]]] = {}
        self.raw_responses = {}
        total = len(items)

        async def run_all():
            sem = asyncio.Semaphore(self.max_concurrency)
            done = 0

            async def one(it: PredictItem):
                nonlocal done
                async with sem:
                    try:
                        words, raw = await self._with_retries(it.audio_path)
                        out[it.key] = words
                        self.raw_responses[it.key] = raw
                    except Exception as e:
                        print(f"  error on {it.key}: {type(e).__name__}: {e}")
                        self.raw_responses[it.key] = {"error": f"{type(e).__name__}: {e}"}
                done += 1
                if progress_every and (done % progress_every == 0 or done == 1 or done == total):
                    print(f"  [{done}/{total}] done ({len(out)} ok)", flush=True)

            await asyncio.gather(*(one(it) for it in items))

        asyncio.run(run_all())
        return out
