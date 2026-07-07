"""音高单位换算的测试。"""

from __future__ import annotations

import numpy as np

from gandharva.convert_units import (
    cents_between,
    hz_to_midi,
    midi_to_hz,
    transpose_hz,
)


def test_a4_anchor() -> None:
    assert np.isclose(midi_to_hz(69), 440.0)
    assert np.isclose(hz_to_midi(440.0), 69.0)


def test_middle_c() -> None:
    # MIDI 60 = 中央 C ≈ 261.63 Hz
    assert np.isclose(midi_to_hz(60), 261.6255653, atol=1e-4)


def test_octave_doubles_frequency() -> None:
    assert np.isclose(midi_to_hz(72) / midi_to_hz(60), 2.0)


def test_midi_hz_round_trip_array() -> None:
    midi = np.array([48.0, 55.5, 69.0, 81.0])
    np.testing.assert_allclose(hz_to_midi(midi_to_hz(midi)), midi, atol=1e-9)


def test_cents_of_semitone() -> None:
    f0 = 440.0
    f1 = midi_to_hz(70)  # 高一个半音
    assert np.isclose(cents_between(f0, f1), 100.0, atol=1e-6)


def test_transpose_up_octave() -> None:
    assert np.isclose(transpose_hz(220.0, 12), 440.0)
