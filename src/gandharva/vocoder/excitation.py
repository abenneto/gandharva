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
