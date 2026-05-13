"""音符时间轴工具：把乐谱事件映射到分析帧。

从 :mod:`gandharva.svs` 抽出的通用时间映射逻辑，方便 SVS 与将来的
对齐 / 编辑功能复用。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gandharva.core import Score
from gandharva.voice.phonemes import vowel_of

FloatArray = NDArray[np.float64]


def frame_times(n_frames: int, hop_length: int, sample_rate: int) -> FloatArray:
    """返回每帧中心对应的时刻（秒）。"""
    return np.arange(n_frames) * hop_length / sample_rate


def frame_vowels(score: Score, n_frames: int, hop_length: int, sample_rate: int) -> list[str]:
    """给每一帧分配它所属音符的元音（休止符 / 空档记为空串）。"""
    times = frame_times(n_frames, hop_length, sample_rate)
    vowels = [""] * n_frames
    for note in score:
        if note.is_rest:
            continue
        v = vowel_of(note.lyric)
        for i, t in enumerate(times):
            if note.start <= t < note.end:
                vowels[i] = v
    return vowels
