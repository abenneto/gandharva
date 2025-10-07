"""音高单位换算：MIDI 音高、频率（Hz）、音分（cents）之间的转换。

以 A4 = 440 Hz、MIDI 69 为参照。所有函数都支持标量与 numpy 数组。
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import A4_HZ, A4_MIDI, SEMITONES_PER_OCTAVE

FloatArray = NDArray[np.float64]


@overload
def midi_to_hz(midi: float) -> float: ...
@overload
def midi_to_hz(midi: FloatArray) -> FloatArray: ...


def midi_to_hz(midi: float | FloatArray) -> float | FloatArray:
    """MIDI 音高编号 → 频率（Hz）。

    f = 440 * 2 ** ((m - 69) / 12)
    """
    return A4_HZ * 2.0 ** ((np.asarray(midi, dtype=np.float64) - A4_MIDI) / SEMITONES_PER_OCTAVE)


@overload
def hz_to_midi(hz: float) -> float: ...
@overload
def hz_to_midi(hz: FloatArray) -> FloatArray: ...


def hz_to_midi(hz: float | FloatArray) -> float | FloatArray:
    """频率（Hz） → MIDI 音高编号。midi_to_hz 的逆运算。"""
    hz_arr = np.asarray(hz, dtype=np.float64)
    return A4_MIDI + SEMITONES_PER_OCTAVE * np.log2(hz_arr / A4_HZ)
