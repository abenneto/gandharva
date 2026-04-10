"""gandharva —— 歌声合成（SVS）与歌声转换（SVC）框架。

公开接口围绕两条链：

- **SVS**：:func:`~gandharva.svs.synthesize_score` 把乐谱唱成歌声；
- **SVC**：:func:`~gandharva.svc.convert_pitch` / :func:`~gandharva.svc.convert_voice`
  对已有歌声变调、变声。

底层的音高（:mod:`gandharva.pitch`）、发音（:mod:`gandharva.voice`）与
声码器（:mod:`gandharva.vocoder`）也可单独使用。
"""

from gandharva._version import __version__
from gandharva.audio import read_wav, write_wav
from gandharva.core import F0Contour, Note, Score, Voice
from gandharva.pitch.estimate import estimate_f0
from gandharva.svc import convert_pitch, convert_to_key, convert_voice
from gandharva.svs import SynthConfig, synthesize_score

__all__ = [
    "__version__",
    "Note",
    "Score",
    "Voice",
    "F0Contour",
    "SynthConfig",
    "synthesize_score",
    "convert_pitch",
    "convert_voice",
    "convert_to_key",
    "estimate_f0",
    "read_wav",
    "write_wav",
]
