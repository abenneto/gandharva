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
