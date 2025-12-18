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
    # 把对数幅度谱（半谱）视作实偶谱，逆变换到倒谱（quefrency）域
    cepstrum = np.fft.irfft(log_mag)
    # 低通提升：只保留低 quefrency（含其镜像），抹去精细谐波结构
    lifter = np.zeros_like(cepstrum)
    lifter[:n_coeffs] = 1.0
    lifter[-(n_coeffs - 1) :] = 1.0
    lifter[0] = 1.0
    env_log = np.fft.rfft(cepstrum * lifter).real
    return np.exp(env_log).astype(np.float64)


def _liftered_cepstrum(magnitude: FloatArray, n_coeffs: int) -> FloatArray:
    """返回低通提升后的对数域包络（供内部复用）。"""
    return np.log(np.maximum(cepstral_envelope(magnitude, n_coeffs), EPS))
