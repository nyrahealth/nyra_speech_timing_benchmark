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


def _make_nyra_fa(**kwargs) -> TimingModel:
    from .nyra_fa_adapter import NyraForcedAligner

    return NyraForcedAligner(**kwargs)


def _make_nyra_fa_pro(**kwargs) -> TimingModel:
    from .nyra_fa_adapter import NyraForcedAligner

    return NyraForcedAligner(pro=True, **kwargs)


def _make_elevenlabs_fa(**kwargs) -> TimingModel:
    from .elevenlabs_fa_adapter import ElevenLabsForcedAligner

    return ElevenLabsForcedAligner(**kwargs)


def _hosted_stt(provider: str) -> Callable[..., TimingModel]:
    def make(**kwargs) -> TimingModel:
        from .hosted_stt_adapter import HostedSttModel

        return HostedSttModel(provider, **kwargs)

    return make


def _make_cartesia_stream(**kwargs) -> TimingModel:
    from .cartesia_stream_adapter import CartesiaStreamModel

    return CartesiaStreamModel(**kwargs)


def _make_mai_transcribe_2(**kwargs) -> TimingModel:
    from .mai_transcribe_adapter import MaiTranscribeModel

    return MaiTranscribeModel(**kwargs)


def _assemblyai(variant: str) -> Callable[..., TimingModel]:
    def make(**kwargs) -> TimingModel:
        from .assemblyai_adapter import AssemblyAIModel

        return AssemblyAIModel(variant, **kwargs)

    return make


def _make_fish_transcribe_1_pro(**kwargs) -> TimingModel:
    from .fish_asr_adapter import FishAsrModel

    return FishAsrModel(model="transcribe-1-pro", **kwargs)


MODEL_FACTORIES: Dict[str, Callable[..., TimingModel]] = {
    "whisperx": _make_whisperx,
    "nemo_fa": _make_nemo_fa,
    "ctc_segmentation": _make_ctc_segmentation,
    "canary": _make_canary,
    "qwen_fa": _make_qwen_fa,
    "mfa": _make_mfa,
    "crisperwhisper": _make_crisperwhisper,
    "nyra_fa": _make_nyra_fa,
    "nyra_fa_pro": _make_nyra_fa_pro,
    "elevenlabs_fa": _make_elevenlabs_fa,
    "xai_grok_stt": _hosted_stt("xai_grok_stt"),
    "deepgram_nova3": _hosted_stt("deepgram_nova3"),
    "cartesia_ink_whisper": _hosted_stt("cartesia_ink_whisper"),
    "cartesia_ink_whisper_stream": _make_cartesia_stream,
    "mai_transcribe_2": _make_mai_transcribe_2,
    "assemblyai_u35_pro": _assemblyai("assemblyai_u35_pro"),
    "assemblyai_u2_disfl": _assemblyai("assemblyai_u2_disfl"),
    "fish_transcribe_1_pro": _make_fish_transcribe_1_pro,
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
    "nyra_fa": True,
    "nyra_fa_pro": True,
    "elevenlabs_fa": True,
    "xai_grok_stt": False,
    "deepgram_nova3": False,
    "cartesia_ink_whisper": False,
    "cartesia_ink_whisper_stream": False,
    "mai_transcribe_2": False,
    "assemblyai_u35_pro": False,
    "assemblyai_u2_disfl": False,
    "fish_transcribe_1_pro": False,
}

#: Adapters that call a hosted API (all others run locally).
HOSTED_MODELS = frozenset(
    {
        "elevenlabs_fa",
        "xai_grok_stt",
        "deepgram_nova3",
        "cartesia_ink_whisper",
        "cartesia_ink_whisper_stream",
        "mai_transcribe_2",
        "assemblyai_u35_pro",
        "assemblyai_u2_disfl",
        "fish_transcribe_1_pro",
    }
)


def list_models() -> List[str]:
    return sorted(MODEL_FACTORIES.keys())


def build_model(name: str, **kwargs) -> TimingModel:
    if name not in MODEL_FACTORIES:
        raise KeyError(
            f"Unknown model '{name}'. Available: {', '.join(list_models())}"
        )
    return MODEL_FACTORIES[name](**kwargs)
