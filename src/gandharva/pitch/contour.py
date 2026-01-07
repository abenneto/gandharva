"""由乐谱生成基频轨迹（F0 contour）。

把离散的音符音高转成逐帧、连续的基频曲线，是“旋律控制”的核心：
静态音高只是起点，真正像人声还需要滑音、颤音与起音过冲。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import DEFAULT_HOP_LENGTH, DEFAULT_SAMPLE_RATE
from gandharva.convert_units import midi_to_hz
from gandharva.core import F0Contour, Score

FloatArray = NDArray[np.float64]


def notes_to_f0(
    score: Score,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> F0Contour:
    """把乐谱渲染成阶梯状（尚无过渡）的基频轨迹。

    休止符对应的帧记为 0（清音）。后续函数在此基础上叠加滑音 / 颤音。
    """
    n_frames = int(np.ceil(score.duration * sample_rate / hop_length)) + 1
    f0 = np.zeros(n_frames, dtype=np.float64)
    frame_times = np.arange(n_frames) * hop_length / sample_rate

    for note in score:
        if note.is_rest:
            continue
        mask = (frame_times >= note.start) & (frame_times < note.end)
        f0[mask] = float(midi_to_hz(note.pitch))

    return F0Contour(values=f0, hop_length=hop_length, sample_rate=sample_rate)
