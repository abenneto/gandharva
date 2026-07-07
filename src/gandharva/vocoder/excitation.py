"""源-滤波器模型中的“源”：激励信号生成。

浊音由准周期脉冲串驱动（对应声带振动），清音由白噪声驱动。
真实歌声介于两者之间，用非周期性（aperiodicity）在两者间混合。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import EPS

FloatArray = NDArray[np.float64]


def pulse_train(f0: FloatArray, hop_length: int, sample_rate: int) -> FloatArray:
    """由逐帧基频生成时域脉冲串。

    通过累积瞬时相位，在相位每越过一个整周期时放置一个单位脉冲，
    脉冲间隔严格跟随 f0，因此能承载滑音与颤音。清音帧（f0<=0）不产生脉冲。
    """
    n_frames = len(f0)
    total = n_frames * hop_length
    # 把逐帧 f0 上采样到逐样本
    frame_idx = np.arange(total) // hop_length
    f0_samples = f0[np.clip(frame_idx, 0, n_frames - 1)]

    excitation = np.zeros(total, dtype=np.float64)
    phase = 0.0
    for n in range(total):
        f = f0_samples[n]
        if f <= 0.0:
            phase = 0.0
            continue
        phase += f / sample_rate
        if phase >= 1.0:
            phase -= 1.0
            excitation[n] = 1.0
    return excitation


def mixed_excitation(
    f0: FloatArray,
    hop_length: int,
    sample_rate: int,
    *,
    aperiodicity: FloatArray | None = None,
    seed: int = 0,
) -> FloatArray:
    """浊音脉冲串与噪声按非周期性混合的激励。

    ``aperiodicity`` 为逐帧 [0, 1]：0 表示纯周期（脉冲），1 表示纯噪声。
    默认清音帧噪声、浊音帧少量气声，符合歌声的自然嗓音质感。
    """
    pulses = pulse_train(f0, hop_length, sample_rate)
    total = len(pulses)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(total)

    n_frames = len(f0)
    if aperiodicity is None:
        # 浊音默认 0.1 气声，清音全噪声
        ap = np.where(f0 > 0.0, 0.1, 1.0)
    else:
        ap = aperiodicity
    frame_idx = np.clip(np.arange(total) // hop_length, 0, n_frames - 1)
    ap_samples = ap[frame_idx]

    # 能量守恒式混合：sqrt 权重
    voiced_gain = np.sqrt(np.clip(1.0 - ap_samples, 0.0, 1.0))
    noise_gain = np.sqrt(np.clip(ap_samples, 0.0, 1.0))
    mixed = voiced_gain * pulses + noise_gain * noise * 0.3
    peak = float(np.max(np.abs(mixed)))
    return (mixed / (peak + EPS)).astype(np.float64)
