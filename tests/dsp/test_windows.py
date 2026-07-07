"""分帧、窗函数与重叠相加的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from gandharva.dsp.windows import frame_signal, get_window, overlap_add
from gandharva.exceptions import ParameterError


def test_frame_shape_no_pad() -> None:
    sig = np.arange(10, dtype=np.float64)
    frames = frame_signal(sig, frame_length=4, hop_length=2, pad=False)
    # (10 - 4) // 2 + 1 = 4
    assert frames.shape == (4, 4)
    np.testing.assert_array_equal(frames[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(frames[1], [2, 3, 4, 5])


def test_frame_pads_last_frame() -> None:
    sig = np.arange(9, dtype=np.float64)
    frames = frame_signal(sig, frame_length=4, hop_length=4, pad=True)
    # 9 -> 补到 12，得到 3 帧
    assert frames.shape == (3, 4)
    np.testing.assert_array_equal(frames[-1], [8, 0, 0, 0])


def test_frame_rejects_bad_params() -> None:
    with pytest.raises(ParameterError):
        frame_signal(np.zeros(8), frame_length=0, hop_length=2)


@pytest.mark.parametrize("name", ["hann", "hamming", "blackman", "bartlett", "rect"])
def test_window_length_and_range(name: str) -> None:
    win = get_window(name, 32)
    assert win.shape == (32,)
    # blackman 在端点会有量级 1e-17 的负值，属浮点误差
    assert np.all(win >= -1e-9)
    assert win.max() <= 1.0 + 1e-9


def test_unknown_window_raises() -> None:
    with pytest.raises(ParameterError):
        get_window("nope", 16)


def test_overlap_add_reconstructs_constant() -> None:
    # 恒定信号经分帧→加窗→OLA 应基本还原
    n = 200
    sig = np.ones(n)
    frame_length, hop = 32, 8
    win = get_window("hann", frame_length)
    frames = frame_signal(sig, frame_length, hop, pad=True) * win
    rec = overlap_add(frames, hop, window=win)
    # 比较中间稳定段（避开边界过渡）
    core = slice(frame_length, n - frame_length)
    np.testing.assert_allclose(rec[core], sig[core], atol=1e-6)
