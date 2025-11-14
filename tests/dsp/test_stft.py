"""STFT / iSTFT 往返测试。"""

from __future__ import annotations

import numpy as np

from gandharva.dsp.stft import istft, stft


def test_stft_shape() -> None:
    sig = np.random.default_rng(0).standard_normal(4096)
    spec = stft(sig, frame_length=512, hop_length=128)
    assert spec.shape[1] == 512 // 2 + 1
    assert np.iscomplexobj(spec)


def test_stft_istft_round_trip() -> None:
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(8000)
    frame_length, hop = 1024, 256
    spec = stft(sig, frame_length, hop)
    rec = istft(spec, frame_length, hop)

    # 比较中间稳定段，长度取二者较小值
    n = min(len(sig), len(rec))
    core = slice(frame_length, n - frame_length)
    np.testing.assert_allclose(rec[core], sig[core], atol=1e-6)


def test_stft_of_sine_has_peak_at_bin() -> None:
    sr = 16000
    f = 500.0
    t = np.arange(sr) / sr
    sig = np.sin(2 * np.pi * f * t)
    frame_length, hop = 1024, 256
    spec = stft(sig, frame_length, hop)
    mag = np.abs(spec).mean(axis=0)
    peak_bin = int(np.argmax(mag))
    expected_bin = round(f * frame_length / sr)
    assert abs(peak_bin - expected_bin) <= 1
