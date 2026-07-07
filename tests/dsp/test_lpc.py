"""LPC 在已知 AR 过程上的测试。"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps

from gandharva.dsp.lpc import autocorrelate, levinson_durbin, lpc, lpc_envelope


def test_autocorrelation_symmetry_and_peak() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(512)
    acf = autocorrelate(x, order=10)
    # 零延迟自相关最大，且等于能量
    assert acf[0] == acf.max()
    assert np.isclose(acf[0], np.dot(x, x), rtol=1e-6)


def test_levinson_recovers_ar_coeffs() -> None:
    # 由已知 AR(2) 滤波器生成信号，LPC 应恢复其分母系数
    true_a = np.array([1.0, -1.2, 0.5])
    rng = np.random.default_rng(2)
    excitation = rng.standard_normal(20000)
    signal = sps.lfilter([1.0], true_a, excitation)

    acf = autocorrelate(signal, order=2)
    a, err = levinson_durbin(acf, order=2)
    np.testing.assert_allclose(a, true_a, atol=0.05)
    assert err > 0.0


def test_lpc_gain_positive() -> None:
    rng = np.random.default_rng(3)
    frame = rng.standard_normal(400)
    a, gain = lpc(frame, order=12)
    assert a[0] == 1.0
    assert gain > 0.0


def test_lpc_envelope_peaks_near_formant() -> None:
    # 合成一个带单一共振峰的信号，包络峰值应落在该频率附近
    sr = 16000
    f_formant = 800.0
    # 二阶谐振器
    r = 0.95
    w = 2 * np.pi * f_formant / sr
    a = np.array([1.0, -2 * r * np.cos(w), r * r])
    rng = np.random.default_rng(4)
    sig = sps.lfilter([1.0], a, rng.standard_normal(4000))

    coeffs, gain = lpc(sig, order=8)
    n_fft = 1024
    env = lpc_envelope(coeffs, gain, n_fft)
    peak_bin = int(np.argmax(env))
    peak_hz = peak_bin * sr / n_fft
    assert abs(peak_hz - f_formant) < 120.0
