"""倒谱谱包络的测试。"""

from __future__ import annotations

import numpy as np

from gandharva.dsp.envelope import cepstral_envelope


def test_envelope_shape_and_positive() -> None:
    rng = np.random.default_rng(0)
    mag = np.abs(np.fft.rfft(rng.standard_normal(1024)))
    env = cepstral_envelope(mag, n_coeffs=40)
    assert env.shape == mag.shape
    assert np.all(env > 0.0)


def test_envelope_is_smoother_than_input() -> None:
    rng = np.random.default_rng(1)
    mag = np.abs(np.fft.rfft(rng.standard_normal(2048)))
    env = cepstral_envelope(mag, n_coeffs=30)
    # 平滑包络的对数谱逐点差分方差应更小
    rough = np.std(np.diff(np.log(mag + 1e-9)))
    smooth = np.std(np.diff(np.log(env)))
    assert smooth < rough


def test_envelope_tracks_broad_peak() -> None:
    # 一个宽的高斯峰，包络峰值 bin 应与之接近
    n = 513
    x = np.arange(n)
    mag = 1.0 + 5.0 * np.exp(-0.5 * ((x - 200) / 30.0) ** 2)
    env = cepstral_envelope(mag, n_coeffs=25)
    assert abs(int(np.argmax(env)) - 200) < 20


def test_envelope_handles_zeros() -> None:
    # 含 0 的幅度谱不应产生 nan / inf（内部有 EPS 保护）
    mag = np.zeros(257)
    mag[50] = 1.0
    env = cepstral_envelope(mag, n_coeffs=20)
    assert np.all(np.isfinite(env))
    assert np.all(env > 0.0)


def test_more_coeffs_less_smoothing() -> None:
    rng = np.random.default_rng(7)
    mag = np.abs(np.fft.rfft(rng.standard_normal(2048)))
    coarse = cepstral_envelope(mag, n_coeffs=10)
    fine = cepstral_envelope(mag, n_coeffs=60)
    # 系数越多，越贴近原始 -> 逐点差分方差更大
    assert np.std(np.diff(np.log(fine))) > np.std(np.diff(np.log(coarse)))


def test_flat_spectrum_stays_flat() -> None:
    mag = np.full(257, 2.0)
    env = cepstral_envelope(mag, n_coeffs=15)
    np.testing.assert_allclose(env, 2.0, rtol=1e-3)
