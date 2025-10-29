"""分帧、加窗与重叠相加（OLA）重建。

分帧使用 numpy 的 stride trick，零拷贝地得到重叠窗口视图。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gandharva.exceptions import ParameterError

FloatArray = NDArray[np.float64]

_WINDOWS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "bartlett": np.bartlett,
    "rect": np.ones,
}


def get_window(name: str, length: int) -> FloatArray:
    """按名称返回长度为 ``length`` 的窗函数。

    支持 ``hann``、``hamming``、``blackman``、``bartlett``、``rect``。
    """
    if length <= 0:
        raise ParameterError("窗长必须为正")
    try:
        fn = _WINDOWS[name]
    except KeyError as exc:
        raise ParameterError(f"未知窗函数 {name!r}，可选：{sorted(_WINDOWS)}") from exc
    return fn(length).astype(np.float64)



def frame_signal(
    signal: FloatArray,
    frame_length: int,
    hop_length: int,
    *,
    pad: bool = True,
) -> FloatArray:
    """把一维信号切成形状为 ``(n_frames, frame_length)`` 的二维数组。

    参数
    ----
    signal:
        一维波形。
    frame_length:
        每帧长度（采样点），必须为正。
    hop_length:
        相邻帧的间隔（采样点），必须为正。
    pad:
        为 True 时在末尾补零，使最后一帧完整；为 False 时丢弃不足一帧的尾部。
    """
    if frame_length <= 0 or hop_length <= 0:
        raise ParameterError("frame_length 与 hop_length 必须为正")
    if signal.ndim != 1:
        raise ParameterError("frame_signal 只接受一维信号")

    signal = np.ascontiguousarray(signal, dtype=np.float64)
    if pad:
        remainder = (len(signal) - frame_length) % hop_length
        if len(signal) < frame_length:
            pad_len = frame_length - len(signal)
        elif remainder != 0:
            pad_len = hop_length - remainder
        else:
            pad_len = 0
        if pad_len:
            signal = np.concatenate([signal, np.zeros(pad_len, dtype=np.float64)])

    if len(signal) < frame_length:
        return np.empty((0, frame_length), dtype=np.float64)

    n_frames = 1 + (len(signal) - frame_length) // hop_length
    stride = signal.strides[0]
    return np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_frames, frame_length),
        strides=(hop_length * stride, stride),
        writeable=False,
    )


def overlap_add(
    frames: FloatArray,
    hop_length: int,
    *,
    window: FloatArray | None = None,
) -> FloatArray:
    """重叠相加：把 ``(n_frames, frame_length)`` 帧序列拼回一维信号。

    若给出 ``window``，则做加权归一化（WOLA），逐样本除以窗能量之和，
    从而抵消分析窗与合成窗叠加带来的幅度起伏。
    """
    if frames.ndim != 2:
        raise ParameterError("overlap_add 需要二维帧数组")
    n_frames, frame_length = frames.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)

    out_len = (n_frames - 1) * hop_length + frame_length
    out = np.zeros(out_len, dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)
    win = np.ones(frame_length) if window is None else window

    for i in range(n_frames):
        start = i * hop_length
        out[start : start + frame_length] += frames[i] * win
        norm[start : start + frame_length] += win**2

    nonzero = norm > 1e-12
    out[nonzero] /= norm[nonzero]
    return out
