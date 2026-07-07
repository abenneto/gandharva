"""短时傅里叶变换（STFT）及其逆变换。

建立在 :mod:`gandharva.dsp.windows` 的分帧 / OLA 之上，
用于声码器的谱域分析与合成。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gandharva.dsp.windows import frame_signal, get_window, overlap_add

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def stft(
    signal: FloatArray,
    frame_length: int,
    hop_length: int,
    *,
    window: str = "hann",
) -> ComplexArray:
    """前向 STFT。

    返回形状为 ``(n_frames, frame_length // 2 + 1)`` 的复数谱，
    使用 :func:`numpy.fft.rfft`（实信号）。
    """
    win = get_window(window, frame_length)
    frames = frame_signal(signal, frame_length, hop_length, pad=True)
    return np.fft.rfft(frames * win, axis=1)


def istft(
    spectrum: ComplexArray,
    frame_length: int,
    hop_length: int,
    *,
    window: str = "hann",
) -> FloatArray:
    """逆 STFT。

    对每帧做 :func:`numpy.fft.irfft`，再用加窗重叠相加拼回时域信号。
    与 :func:`stft` 使用同一分析窗，OLA 内部做能量归一化。
    """
    win = get_window(window, frame_length)
    frames = np.fft.irfft(spectrum, n=frame_length, axis=1)
    return overlap_add(frames, hop_length, window=win)
