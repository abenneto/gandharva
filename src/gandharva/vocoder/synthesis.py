"""声码器合成：把参数（激励 + LPC 谱包络）还原成波形。

对每帧，用该帧的 LPC 全极点滤波器塑形对应段的激励，再加窗重叠相加。
这是源-滤波器模型的“滤波”一侧，与 :mod:`gandharva.vocoder.excitation`
生成的“源”配合，构成完整声码器。
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps
from numpy.typing import NDArray

from gandharva.dsp.windows import get_window, overlap_add
from gandharva.vocoder.analysis import VocoderFrames

FloatArray = NDArray[np.float64]


def synthesize(frames: VocoderFrames, excitation: FloatArray) -> FloatArray:
    """由分析参数与激励信号合成波形（骨架版：逐帧全极点滤波）。"""
    frame_length = frames.frame_length
    hop = frames.hop_length
    n_frames = frames.n_frames

    out_frames = np.zeros((n_frames, frame_length), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        seg = excitation[start : start + frame_length]
        if len(seg) < frame_length:
            seg = np.pad(seg, (0, frame_length - len(seg)))
        a = frames.lpc_coeffs[i]
        # 全极点滤波：H(z) = gain / A(z)
        shaped = sps.lfilter([frames.gains[i]], a, seg)
        out_frames[i] = shaped

    win = get_window("hann", frame_length)
    return overlap_add(out_frames, hop, window=win)


def synthesize_from_envelopes(
    excitation: FloatArray,
    envelopes: FloatArray,
    frame_length: int,
    hop_length: int,
) -> FloatArray:
    """用逐帧幅度谱包络对激励做谱域整形后 OLA 合成。

    参数
    ----
    excitation:
        时域激励（脉冲串 / 混合激励）。
    envelopes:
        形状 ``(n_frames, frame_length // 2 + 1)`` 的幅度谱包络序列，
        通常来自元音共振峰或 LPC 包络。

    每帧取激励对应段，做 rfft、乘以该帧包络、irfft，再加窗重叠相加。
    保留激励的相位（承载音高与谐波结构），只把幅度塑成目标音色。
    """
    n_frames = envelopes.shape[0]
    win = get_window("hann", frame_length)
    out_frames = np.zeros((n_frames, frame_length), dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_length
        seg = excitation[start : start + frame_length]
        if len(seg) < frame_length:
            seg = np.pad(seg, (0, frame_length - len(seg)))
        spectrum = np.fft.rfft(seg * win)
        shaped = spectrum * envelopes[i]
        out_frames[i] = np.fft.irfft(shaped, n=frame_length)

    return overlap_add(out_frames, hop_length, window=win)
