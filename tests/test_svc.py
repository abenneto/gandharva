"""歌声转换（SVC）的测试。"""

from __future__ import annotations

import numpy as np

from gandharva.convert_units import hz_to_midi
from gandharva.svc import convert_pitch, convert_to_key, convert_voice, snap_to_key


def _harmonic(freq: float, sr: int, dur: float = 0.7) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    sig = np.zeros_like(t)
    for k, amp in enumerate([1.0, 0.5, 0.3, 0.15], start=1):
        sig += amp * np.sin(2 * np.pi * freq * k * t)
    return 0.5 * sig / np.max(np.abs(sig))


def _spectral_fundamental(signal: np.ndarray, sr: int, fmax: float = 600.0) -> float:
    windowed = signal * np.hanning(len(signal))
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    band = (freqs > 50.0) & (freqs < fmax)
    return float(freqs[band][np.argmax(spec[band])])


def test_convert_pitch_up_a_fourth() -> None:
    sr = 24000
    freq = 150.0
    sig = _harmonic(freq, sr)
    voice = convert_pitch(sig, semitones=5)
    expected = freq * 2 ** (5 / 12)
    assert abs(_spectral_fundamental(voice.samples, sr) - expected) / expected < 0.05


def test_convert_pitch_preserves_length() -> None:
    sr = 24000
    sig = _harmonic(180.0, sr)
    voice = convert_pitch(sig, semitones=-3)
    assert abs(len(voice.samples) - len(sig)) / len(sig) < 0.02


def test_convert_voice_keeps_pitch_when_zero_semitones() -> None:
    sr = 24000
    freq = 160.0
    sig = _harmonic(freq, sr)
    voice = convert_voice(sig, semitones=0, formant_shift=1.15)
    assert abs(_spectral_fundamental(voice.samples, sr) - freq) / freq < 0.05


def test_snap_to_key_rounds_to_scale() -> None:
    # 略微跑调的音高应被吸附到 C 大调音级
    f0 = np.array([262.0, 300.0, 350.0])  # ~C4, ~D4+, ~F4
    snapped = snap_to_key(f0, tonic_midi=60)
    midi = np.round(hz_to_midi(snapped)).astype(int)
    # C 大调音级：60,62,64,65,67,69,71...
    for m in midi:
        assert (m - 60) % 12 in (0, 2, 4, 5, 7, 9, 11)


def test_snap_preserves_unvoiced() -> None:
    f0 = np.array([0.0, 262.0, 0.0])
    snapped = snap_to_key(f0, tonic_midi=60)
    assert snapped[0] == 0.0
    assert snapped[2] == 0.0


def test_convert_to_key_runs() -> None:
    sr = 24000
    sig = _harmonic(261.0, sr)
    voice = convert_to_key(sig, tonic_midi=60)
    assert np.all(np.isfinite(voice.samples))
