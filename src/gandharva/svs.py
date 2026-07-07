"""歌声合成（SVS）引擎：乐谱 → 歌声波形。

流程：
1. 乐谱 → 基频轨迹（音高 / 旋律控制，含滑音、颤音、起音过冲）；
2. 每个音符的音节 → 元音 → 共振峰谱包络（发音 / 音色控制）；
3. 基频驱动激励，经谱包络整形，重叠相加成波形（声码器）。

这条链把“语音的发音建模”与“音乐的音高控制”合到一处，
是本框架“交叉集大成”的落点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SAMPLE_RATE,
)
from gandharva.core import Score
from gandharva.voice.phonemes import vowel_of

FloatArray = NDArray[np.float64]


@dataclass
class SynthConfig:
    """SVS 合成参数。"""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_length: int = DEFAULT_FRAME_LENGTH
    hop_length: int = DEFAULT_HOP_LENGTH
    vibrato_depth_cents: float = 40.0
    vibrato_rate_hz: float = 5.5
    portamento_ms: float = 80.0
    overshoot_cents: float = 20.0


def synthesize_score(score: Score, config: SynthConfig | None = None) -> None:
    """把乐谱合成为歌声波形（后续提交补全实现）。"""
    raise NotImplementedError


def _frame_vowels(score: Score, n_frames: int, hop_length: int, sample_rate: int) -> list[str]:
    """给每一帧分配它所属音符的元音（休止符 / 空档记为空串）。"""
    frame_times = np.arange(n_frames) * hop_length / sample_rate
    vowels = [""] * n_frames
    for note in score:
        if note.is_rest:
            continue
        v = vowel_of(note.lyric)
        for i, t in enumerate(frame_times):
            if note.start <= t < note.end:
                vowels[i] = v
    return vowels
