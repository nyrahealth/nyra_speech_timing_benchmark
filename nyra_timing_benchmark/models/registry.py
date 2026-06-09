"""Registry of built-in model adapters.

Adapters are stored as factory callables so their (potentially heavy) imports
only happen when a model is actually instantiated.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .base import TimingModel


def _make_whisperx(**kwargs) -> TimingModel:
    from .whisperx_adapter import WhisperXModel

    return WhisperXModel(**kwargs)


def _make_nemo_fa(**kwargs) -> TimingModel:
    from .nemo_fa_adapter import NemoForcedAligner

    return NemoForcedAligner(**kwargs)


def _make_ctc_segmentation(**kwargs) -> TimingModel:
    from .ctc_segmentation_adapter import CtcSegmentationModel

    return CtcSegmentationModel(**kwargs)


def _make_canary(**kwargs) -> TimingModel:
    from .canary_adapter import CanaryModel

    return CanaryModel(**kwargs)


def _make_qwen_fa(**kwargs) -> TimingModel:
    from .qwen_fa_adapter import QwenForcedAligner

    return QwenForcedAligner(**kwargs)


def _make_mfa(**kwargs) -> TimingModel:
    from .mfa_adapter import MfaModel

    return MfaModel(**kwargs)


def _make_crisperwhisper(**kwargs) -> TimingModel:
    from .crisperwhisper_adapter import CrisperWhisperModel

    return CrisperWhisperModel(**kwargs)


MODEL_FACTORIES: Dict[str, Callable[..., TimingModel]] = {
    "whisperx": _make_whisperx,
    "nemo_fa": _make_nemo_fa,
    "ctc_segmentation": _make_ctc_segmentation,
    "canary": _make_canary,
    "qwen_fa": _make_qwen_fa,
    "mfa": _make_mfa,
    "crisperwhisper": _make_crisperwhisper,
}

#: Whether each model is a forced aligner (needs reference words).
MODEL_IS_FORCED_ALIGNER: Dict[str, bool] = {
    "whisperx": False,
    "nemo_fa": True,
    "ctc_segmentation": True,
    "canary": False,
    "qwen_fa": True,
    "mfa": True,
    "crisperwhisper": False,
}


def list_models() -> List[str]:
    return sorted(MODEL_FACTORIES.keys())


def build_model(name: str, **kwargs) -> TimingModel:
    if name not in MODEL_FACTORIES:
        raise KeyError(
            f"Unknown model '{name}'. Available: {', '.join(list_models())}"
        )
    return MODEL_FACTORIES[name](**kwargs)
