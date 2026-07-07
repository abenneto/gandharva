"""激励生成的测试。"""

from __future__ import annotations

import numpy as np

from gandharva.vocoder.excitation import mixed_excitation, pulse_train


def test_pulse_train_period_matches_f0() -> None:
    sr = 24000
    hop = 240
    n_frames = 100
    f0 = np.full(n_frames, 200.0)  # 恒定 200 Hz
    exc = pulse_train(f0, hop, sr)
    pulse_idx = np.flatnonzero(exc)
    # 相邻脉冲间隔应接近 sr / f0 = 120 采样
    gaps = np.diff(pulse_idx)
    assert abs(np.median(gaps) - sr / 200.0) <= 1.5


def test_unvoiced_frames_have_no_pulses() -> None:
    sr = 24000
    hop = 240
    f0 = np.zeros(50)
    exc = pulse_train(f0, hop, sr)
    assert not np.any(exc)


def test_mixed_excitation_normalized() -> None:
    sr = 24000
    hop = 240
    f0 = np.concatenate([np.full(30, 150.0), np.zeros(20)])
    exc = mixed_excitation(f0, hop, sr, seed=1)
    assert np.max(np.abs(exc)) <= 1.0 + 1e-9
    assert len(exc) == 50 * hop
