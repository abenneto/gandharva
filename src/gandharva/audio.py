"""WAV 读写与基础波形处理。

只依赖标准库 ``wave`` 与 numpy，不引入 soundfile / librosa，
以保持离线、零系统依赖。
"""

from __future__ import annotations

import wave

import numpy as np
from numpy.typing import NDArray

from gandharva.exceptions import AudioError

FloatArray = NDArray[np.float64]

_INT16_MAX = 32767


def read_wav(path: str) -> tuple[FloatArray, int]:
    """读取 16-bit PCM WAV，返回 (归一化到 [-1, 1] 的 float64 波形, 采样率)。

    多通道会被下混为单声道（取各通道平均）。
    """
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise AudioError("只支持 16-bit PCM WAV")
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    data = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data / _INT16_MAX, sample_rate


def write_wav(path: str, samples: FloatArray, sample_rate: int) -> None:
    """将 [-1, 1] 的 float 波形写为 16-bit PCM 单声道 WAV。"""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * _INT16_MAX).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
