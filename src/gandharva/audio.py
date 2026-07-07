"""WAV 读写与基础波形处理。

只依赖标准库 ``wave`` 与 numpy，不引入 soundfile / librosa，
以保持离线、零系统依赖。
"""

from __future__ import annotations

import wave

import numpy as np
import scipy.signal as sps
from numpy.typing import NDArray

from gandharva.constants import EPS
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
    """将 [-1, 1] 的 float 波形写为 16-bit PCM 单声道 WAV。

    非有限值（NaN / inf）会先被置零，再做限幅，避免写出损坏的 PCM。
    """
    finite = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = np.clip(finite, -1.0, 1.0)
    pcm = np.round(clipped * _INT16_MAX).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def normalize_peak(samples: FloatArray, target: float = 0.99) -> FloatArray:
    """峰值归一化：把最大绝对值缩放到 ``target``。全零输入原样返回。"""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < EPS:
        return samples.astype(np.float64)
    return (samples * (target / peak)).astype(np.float64)


def resample(samples: FloatArray, orig_sr: int, target_sr: int) -> FloatArray:
    """把波形从 ``orig_sr`` 重采样到 ``target_sr``（多相 FIR）。"""
    if orig_sr <= 0 or target_sr <= 0:
        raise AudioError("采样率必须为正")
    if orig_sr == target_sr:
        return samples.astype(np.float64)
    g = np.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    out: FloatArray = sps.resample_poly(samples, up, down).astype(np.float64)
    return out
