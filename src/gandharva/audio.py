"""WAV 读写与基础波形处理。

只依赖标准库 ``wave`` 与 numpy，不引入 soundfile / librosa，
以保持离线、零系统依赖。真正的实现会在后续补上。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def read_wav(path: str) -> tuple[FloatArray, int]:
    """读取单声道 WAV，返回 (归一化到 [-1, 1] 的 float64 波形, 采样率)。"""
    raise NotImplementedError


def write_wav(path: str, samples: FloatArray, sample_rate: int) -> None:
    """将 [-1, 1] 的 float 波形写为 16-bit PCM WAV。"""
    raise NotImplementedError
