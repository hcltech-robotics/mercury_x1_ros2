"""Compatibility shim: openWakeWord's train.py does `from generate_samples import generate_samples`
(old script layout) and never passes `model=`. Forward to the packaged implementation with the
LibriTTS-R checkpoint as the default model."""
from pathlib import Path
from piper_sample_generator.__main__ import generate_samples as _generate_samples

_DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "en_US-libritts_r-medium.pt"


def generate_samples(*args, model=None, **kwargs):
    kwargs.pop("auto_reduce_batch_size", None)  # not supported by the packaged version
    return _generate_samples(*args, model=str(model or _DEFAULT_MODEL), **kwargs)
