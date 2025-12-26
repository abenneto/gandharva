"""基频（F0）估计：YIN 算法。

YIN（de Cheveigné & Kawahara, 2002）通过差分函数与累积均值归一化
在时域鲁棒地估计基频，适合歌声这种谐波丰富、音高范围宽的信号。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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
