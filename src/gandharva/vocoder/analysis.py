"""声码器分析：把波形拆成可修改的参数。

对每帧提取 LPC 谱包络（音色）、增益（响度）与基频（音高），
得到一份可编辑的“歌声参数”，供转换后再合成。
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
from gandharva.dsp.lpc import lpc
from gandharva.dsp.windows import frame_signal, get_window
from gandharva.pitch.estimate import estimate_f0

FloatArray = NDArray[np.float64]


@dataclass
class VocoderFrames:
    """逐帧分析结果。"""

    lpc_coeffs: list[FloatArray]  # 每帧的 LPC 系数（首项为 1）
    gains: FloatArray  # 每帧激励增益
    f0: FloatArray  # 每帧基频（0 = 清音）
    hop_length: int
    sample_rate: int
    frame_length: int

    @property
    def n_frames(self) -> int:
        return len(self.gains)


def analyze(
    signal: FloatArray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    lpc_order: int = 24,
) -> VocoderFrames:
    """对信号做逐帧 LPC + 基频分析。"""
    win = get_window("hann", frame_length)
    frames = frame_signal(signal, frame_length, hop_length, pad=True)
    contour = estimate_f0(
        signal,
        sample_rate=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    coeffs: list[FloatArray] = []
    gains = np.zeros(len(frames), dtype=np.float64)
    for i, frame in enumerate(frames):
        a, gain = lpc(frame * win, lpc_order)
        coeffs.append(a)
        gains[i] = gain

    # f0 帧数可能与 frames 略有出入，对齐到相同长度
    n = len(frames)
    f0 = np.zeros(n, dtype=np.float64)
    m = min(n, len(contour.values))
    f0[:m] = contour.values[:m]

    return VocoderFrames(
        lpc_coeffs=coeffs,
        gains=gains,
        f0=f0,
        hop_length=hop_length,
        sample_rate=sample_rate,
        frame_length=frame_length,
    )


def estimate_aperiodicity(
    signal: FloatArray,
    f0: FloatArray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> FloatArray:
    """逐帧估计非周期性 [0, 1]。

    以过零率作为清 / 浊与气声程度的廉价代理：过零率越高，
    非周期成分越多。清音帧（f0<=0）直接记为 1。
    """
    frames = frame_signal(signal, frame_length, hop_length, pad=True)
    n = len(frames)
    ap = np.ones(n, dtype=np.float64)

    for i, frame in enumerate(frames):
        if i < len(f0) and f0[i] > 0.0:
            # 归一化过零率
            zc = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
            # 浊音的气声程度大致落在 [0.05, 0.5]
            ap[i] = float(np.clip(zc * 1.5, 0.05, 0.6))
    return ap
