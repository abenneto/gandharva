"""线性预测编码（LPC）：自相关、Levinson-Durbin 递推与谱包络。

LPC 把一帧信号建模为全极点滤波器：源-滤波器模型里，这个滤波器
近似声道的共振（共振峰）特性，是歌声转换中保持音色的关键。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import EPS
from gandharva.exceptions import ParameterError

FloatArray = NDArray[np.float64]


def autocorrelate(frame: FloatArray, order: int) -> FloatArray:
    """计算一帧信号 0..order 阶的自相关序列。

    使用 FFT 加速（Wiener-Khinchin），返回长度为 ``order + 1`` 的数组。
    """
    if order < 0:
        raise ParameterError("order 必须为非负")
    n = len(frame)
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(frame, nfft)
    acf = np.fft.irfft(spec * np.conj(spec), nfft)[: order + 1]
    return acf.astype(np.float64)


def levinson_durbin(acf: FloatArray, order: int) -> tuple[FloatArray, float]:
    """Levinson-Durbin 递推。

    从自相关序列 ``acf`` 解全极点模型的系数，返回 ``(a, err)``：

    - ``a`` 形如 ``[1, a1, ..., ap]``，即 A(z) = 1 + a1 z^-1 + ... 的系数；
    - ``err`` 为最终的预测残差能量。

    比直接求 Toeplitz 逆更稳定，且天然给出反射系数（此处未返回）。
    """
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    err = float(acf[0])
    if err <= 0.0:
        # 全零 / 直流帧，返回平凡解
        return a, max(err, EPS)

    for i in range(1, order + 1):
        acc = acf[i]
        for j in range(1, i):
            acc += a[j] * acf[i - j]
        k = -acc / err
        # 就地更新系数（对称回代）
        a_prev = a[1:i].copy()
        a[1:i] += k * a_prev[::-1]
        a[i] = k
        err *= 1.0 - k * k
        if err <= 0.0:
            err = EPS
            break
    return a, err


def lpc(frame: FloatArray, order: int) -> tuple[FloatArray, float]:
    """对单帧信号做 ``order`` 阶 LPC 分析。

    返回 ``(a, gain)``：``a`` 是全极点滤波器分母系数（首项为 1），
    ``gain`` 是激励增益 sqrt(err)，使合成幅度与原帧匹配。
    """
    if order < 1:
        raise ParameterError("LPC order 必须 >= 1")
    acf = autocorrelate(frame, order)
    a, err = levinson_durbin(acf, order)
    gain = float(np.sqrt(max(err, EPS)))
    return a, gain
