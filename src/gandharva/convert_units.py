"""音高单位换算：MIDI 音高、频率（Hz）、音分（cents）之间的转换。

以 A4 = 440 Hz、MIDI 69 为参照。所有函数都支持标量与 numpy 数组。
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import (
    A4_HZ,
    A4_MIDI,
    CENTS_PER_OCTAVE,
    EPS,
    SEMITONES_PER_OCTAVE,
)

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


def cents_between(f_from: float | FloatArray, f_to: float | FloatArray) -> float | FloatArray:
    """两个频率之间的音分差：1200 * log2(f_to / f_from)。

    正值表示 ``f_to`` 比 ``f_from`` 高。清音（0 Hz）会先被替换成一个极小值，
    以避免 log(0)。
    """
    a = np.asarray(f_from, dtype=np.float64)
    b = np.asarray(f_to, dtype=np.float64)
    a = np.where(a <= 0.0, EPS, a)
    b = np.where(b <= 0.0, EPS, b)
    return CENTS_PER_OCTAVE * np.log2(b / a)


def transpose_hz(hz: float | FloatArray, semitones: float) -> float | FloatArray:
    """将频率整体升降 ``semitones`` 个半音（可为小数、可为负）。"""
    factor = 2.0 ** (semitones / SEMITONES_PER_OCTAVE)
    return np.asarray(hz, dtype=np.float64) * factor
