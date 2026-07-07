"""基频（F0）估计：YIN 算法。

YIN（de Cheveigné & Kawahara, 2002）通过差分函数与累积均值归一化
在时域鲁棒地估计基频，适合歌声这种谐波丰富、音高范围宽的信号。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SAMPLE_RATE,
    F0_CEIL_HZ,
    F0_FLOOR_HZ,
)
from gandharva.core import F0Contour
from gandharva.dsp.windows import frame_signal

FloatArray = NDArray[np.float64]


def difference_function(frame: FloatArray, max_lag: int) -> FloatArray:
    """YIN 差分函数 d(tau)，tau = 0..max_lag-1。

    d(tau) = sum_j (x_j - x_{j+tau})^2
           = r(0) + (energy of shifted window) - 2 * acf(tau)

    自相关部分用 FFT（Wiener-Khinchin）加速。
    """
    x = frame.astype(np.float64)
    n = len(x)
    # 平方项的累积和，便于取任意窗口能量
    power = np.concatenate([[0.0], np.cumsum(x * x)])
    # FFT 自相关：acf[tau] = sum_j x_j x_{j+tau}
    nfft = 1 << (2 * n).bit_length()
    spec = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(spec * np.conj(spec), nfft)[:max_lag]

    taus = np.arange(max_lag)
    # 前 (n - tau) 个样本的能量，随 tau 增大而减小
    energy_head = power[n] - power[taus]
    d = power[n] + energy_head - 2.0 * acf
    d[0] = 0.0
    return np.maximum(d, 0.0)


def cumulative_mean_normalized(d: FloatArray) -> FloatArray:
    """累积均值归一化差分函数 d'(tau)（CMNDF）。

    d'(0) = 1；d'(tau) = d(tau) / ((1/tau) * sum_{j=1..tau} d(j))。
    归一化让阈值判定与信号能量无关。
    """
    cmndf = np.ones_like(d)
    running = np.cumsum(d[1:])
    taus = np.arange(1, len(d))
    cmndf[1:] = d[1:] * taus / np.maximum(running, 1e-12)
    return cmndf


def absolute_threshold(cmndf: FloatArray, threshold: float, min_lag: int) -> int:
    """在 CMNDF 上做绝对阈值搜索，返回选中的整数 tau；未找到返回 -1。

    取第一个低于 ``threshold`` 的局部谷；若没有则退回全局最小值。
    """
    tau = min_lag
    n = len(cmndf)
    while tau < n:
        if cmndf[tau] < threshold:
            # 走到该谷的局部最低点
            while tau + 1 < n and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            return tau
        tau += 1
    # 无低于阈值者：若整体较好则用全局最小，否则判为清音
    best = int(np.argmin(cmndf[min_lag:])) + min_lag if n > min_lag else -1
    return best if best >= min_lag and cmndf[best] < 1.0 else -1


def parabolic_interpolation(cmndf: FloatArray, tau: int) -> float:
    """对选中的整数 tau 做抛物线插值，得到亚采样精度的周期。

    用 tau 及其左右邻点拟合抛物线，返回顶点横坐标（浮点 lag）。
    """
    if tau <= 0 or tau >= len(cmndf) - 1:
        return float(tau)
    a = cmndf[tau - 1]
    b = cmndf[tau]
    c = cmndf[tau + 1]
    denom = a + c - 2.0 * b
    if abs(denom) < 1e-12:
        return float(tau)
    shift = 0.5 * (a - c) / denom
    return float(tau) + shift


def estimate_f0(
    signal: FloatArray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = F0_FLOOR_HZ,
    fmax: float = F0_CEIL_HZ,
    threshold: float = 0.15,
) -> F0Contour:
    """对整段信号逐帧运行 YIN，返回 :class:`~gandharva.core.F0Contour`。

    清音帧（未过阈值或能量过低）的基频记为 0。
    """
    min_lag = max(2, int(sample_rate / fmax))
    max_lag = min(frame_length - 1, int(sample_rate / fmin) + 1)
    frames = frame_signal(signal, frame_length, hop_length, pad=True)

    f0 = np.zeros(len(frames), dtype=np.float64)
    for i, frame in enumerate(frames):
        # 能量太低直接判为清音，省去计算
        if float(np.dot(frame, frame)) < 1e-6:
            continue
        d = difference_function(frame, max_lag)
        cmndf = cumulative_mean_normalized(d)
        tau = absolute_threshold(cmndf, threshold, min_lag)
        if tau < 0:
            continue
        period = parabolic_interpolation(cmndf, tau)
        if period > 0:
            f0[i] = sample_rate / period

    return F0Contour(values=f0, hop_length=hop_length, sample_rate=sample_rate)
