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
