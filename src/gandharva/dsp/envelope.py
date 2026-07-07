"""基于倒谱（cepstrum）的谱包络平滑。

与 LPC 的全极点包络互补：倒谱法通过在倒谱域低通（保留低quefrency）
得到平滑包络，不假设全极点结构，对宽带噪声更鲁棒。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import EPS

FloatArray = NDArray[np.float64]


def cepstral_envelope(magnitude: FloatArray, n_coeffs: int) -> FloatArray:
    """对幅度谱做倒谱平滑，返回同长度的平滑幅度谱包络。

    参数
    ----
    magnitude:
        单帧的幅度谱（rfft 长度，非负）。
    n_coeffs:
        保留的低倒谱系数个数（“提升器”截止），越小越平滑。
    """
    log_mag = np.log(np.maximum(magnitude, EPS))
    # 构造整谱（共轭对称）后做实倒谱
    full = np.concatenate([log_mag, log_mag[-2:0:-1]])
    cepstrum = np.fft.irfft(np.fft.rfft(full))
    # 低通提升：保留前 n_coeffs 与其镜像
    lifter = np.zeros_like(cepstrum)
    lifter[:n_coeffs] = 1.0
    lifter[-n_coeffs + 1 :] = 1.0
    lifter[0] = 1.0
    smoothed_full = np.fft.irfft(np.fft.rfft(cepstrum * lifter))
    smoothed = smoothed_full[: len(magnitude)]
    return np.exp(smoothed).astype(np.float64)


def _liftered_cepstrum(magnitude: FloatArray, n_coeffs: int) -> FloatArray:
    """返回低通提升后的对数域包络（供内部复用）。"""
    return np.log(np.maximum(cepstral_envelope(magnitude, n_coeffs), EPS))
