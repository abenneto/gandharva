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
