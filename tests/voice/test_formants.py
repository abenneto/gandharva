"""元音共振峰包络的测试。"""

from __future__ import annotations

import numpy as np

from gandharva.voice.formants import (
    FORMANT_TABLE,
    formant_envelope,
    formants_for,
)


def test_known_vowels_present() -> None:
    for v in "aeiou":
        assert v in FORMANT_TABLE
        f1, f2, f3 = formants_for(v)
        assert 0 < f1 < f2 < f3


def test_unknown_vowel_falls_back() -> None:
    assert formants_for("x") == FORMANT_TABLE["a"]


def _nearest_bin(freqs_hz: float, n_fft: int, sr: int) -> int:
    return round(freqs_hz * n_fft / sr)


def test_envelope_peaks_near_formants() -> None:
    sr = 24000
    n_fft = 2048
    env = formant_envelope("i", n_fft, sr)
    assert env.shape == (n_fft // 2 + 1,)
    # 前两个共振峰位置附近应出现局部能量高点
    f1, f2, _ = formants_for("i")
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # 在 F1 邻域内的最大值应显著高于整体中位数
    near_f1 = np.abs(freqs - f1) < 150
    assert env[near_f1].max() > np.median(env) * 2


def test_shift_moves_formants_up() -> None:
    sr = 24000
    n_fft = 2048
    base = formant_envelope("a", n_fft, sr, shift=1.0)
    shifted = formant_envelope("a", n_fft, sr, shift=1.2)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    peak_base = freqs[int(np.argmax(base))]
    peak_shifted = freqs[int(np.argmax(shifted))]
    assert peak_shifted > peak_base
