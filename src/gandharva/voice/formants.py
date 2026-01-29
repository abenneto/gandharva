"""元音共振峰建模。

每个元音由若干共振峰（formant）刻画——声道的共鸣频率。
用一组带通谐振器叠加，得到该元音的谱包络；这正是听感上
区分 “a / e / i / o / u” 的关键，也是发音（音色）控制的落点。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# 男声中性嗓音的前三共振峰中心频率（Hz），来自经典元音声学测量的近似值。
# 参考量级：Peterson & Barney (1952) 一类数据的粗略折中。
FORMANT_TABLE: dict[str, tuple[float, float, float]] = {
    "a": (730.0, 1090.0, 2440.0),
    "e": (530.0, 1840.0, 2480.0),
    "i": (270.0, 2290.0, 3010.0),
    "o": (570.0, 840.0, 2410.0),
    "u": (300.0, 870.0, 2240.0),
}

# 各共振峰的带宽（Hz），越大越“钝”。
FORMANT_BANDWIDTHS: tuple[float, float, float] = (60.0, 90.0, 120.0)


def formants_for(vowel: str) -> tuple[float, float, float]:
    """返回元音的三共振峰中心频率；未知元音回退到 ``a``。"""
    return FORMANT_TABLE.get(vowel.lower(), FORMANT_TABLE["a"])


def formant_envelope(
    vowel: str,
    n_fft: int,
    sample_rate: int,
    *,
    shift: float = 1.0,
) -> FloatArray:
    """生成某元音的幅度谱包络，长度 ``n_fft // 2 + 1``。

    每个共振峰建模为一个洛伦兹型峰（二阶谐振器的幅度响应），
    三峰叠加即得包络。``shift`` 为共振峰频率整体缩放系数：
    >1 使共振峰上移（听感更“小”的声道 / 更亮），用于 SVC 音色变换。
    """
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    envelope = np.zeros_like(freqs)
    centers = formants_for(vowel)
    # 高频共振峰能量略低，符合自然嗓音的谱倾斜
    amplitudes = (1.0, 0.7, 0.45)

    for fc, bw, amp in zip(centers, FORMANT_BANDWIDTHS, amplitudes, strict=True):
        center = fc * shift
        # 洛伦兹峰：amp / (1 + ((f - fc)/(bw/2))^2)
        envelope += amp / (1.0 + ((freqs - center) / (bw / 2.0)) ** 2)

    # 加一个轻微的谱倾斜（-6 dB/oct 附近），更接近真实声源
    tilt = 1.0 / (1.0 + (freqs / 200.0))
    return (envelope * tilt + 1e-3).astype(np.float64)
