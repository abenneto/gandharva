"""YIN 基频估计的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from gandharva.pitch.estimate import (
    cumulative_mean_normalized,
    difference_function,
    estimate_f0,
    parabolic_interpolation,
)


def _tone(freq: float, sr: int, dur: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)


def test_difference_function_zero_at_lag0() -> None:
    x = _tone(200.0, 16000, 0.05)
    d = difference_function(x, max_lag=256)
    assert d[0] == 0.0
    assert np.all(d >= 0.0)


def test_cmndf_starts_at_one() -> None:
    x = _tone(200.0, 16000, 0.05)
    d = difference_function(x, max_lag=256)
    cmndf = cumulative_mean_normalized(d)
    assert cmndf[0] == 1.0


@pytest.mark.parametrize("freq", [110.0, 220.0, 440.0])
def test_estimate_f0_accurate(freq: float) -> None:
    sr = 24000
    sig = _tone(freq, sr)
    contour = estimate_f0(sig, sample_rate=sr)
    voiced = contour.values[contour.voiced]
    assert voiced.size > 0
    median = float(np.median(voiced))
    # 允许 2% 误差
    assert abs(median - freq) / freq < 0.02


def test_silence_is_unvoiced() -> None:
    sr = 24000
    contour = estimate_f0(np.zeros(sr // 2), sample_rate=sr)
    assert not np.any(contour.voiced)


def test_parabolic_interpolation_refines() -> None:
    # 顶点在 tau=10 与 11 之间的抛物线谷
    taus = np.arange(20, dtype=float)
    cmndf = (taus - 10.4) ** 2
    refined = parabolic_interpolation(cmndf, 10)
    assert 10.0 < refined < 11.0
