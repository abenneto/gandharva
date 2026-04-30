"""WAV 往返与波形工具的测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gandharva.audio import normalize_peak, read_wav, resample, write_wav


def test_wav_round_trip(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr) / sr
    sig = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    path = str(tmp_path / "tone.wav")
    write_wav(path, sig, sr)

    read_back, read_sr = read_wav(path)
    assert read_sr == sr
    assert len(read_back) == len(sig)
    # 16-bit 量化误差上界约 1/32767
    np.testing.assert_allclose(read_back, sig, atol=2e-4)


def test_normalize_peak() -> None:
    sig = np.array([0.1, -0.2, 0.05])
    out = normalize_peak(sig, target=1.0)
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_normalize_all_zero_is_safe() -> None:
    sig = np.zeros(10)
    out = normalize_peak(sig)
    np.testing.assert_array_equal(out, sig)


def test_resample_changes_length() -> None:
    sr = 8000
    sig = np.sin(2 * np.pi * 300 * np.arange(sr) / sr)
    out = resample(sig, sr, 16000)
    assert abs(len(out) - 2 * sr) <= 2


def test_resample_identity() -> None:
    sig = np.arange(100, dtype=np.float64)
    np.testing.assert_array_equal(resample(sig, 16000, 16000), sig)


def test_resample_downsample_halves_length() -> None:
    sr = 24000
    sig = np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
    out = resample(sig, sr, sr // 2)
    assert abs(len(out) - sr // 2) <= 2


def test_resample_preserves_tone_frequency() -> None:
    # 220 Hz 正弦重采样后谱峰频率不应改变
    src_sr = 16000
    dst_sr = 24000
    freq = 220.0
    t = np.arange(src_sr) / src_sr
    sig = np.sin(2 * np.pi * freq * t)
    out = resample(sig, src_sr, dst_sr)
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1.0 / dst_sr)
    peak = freqs[int(np.argmax(spec))]
    assert abs(peak - freq) < 5.0


def test_clip_on_write(tmp_path: Path) -> None:
    # 超出 [-1,1] 的样本应在写入时被截断，读回不超界
    sr = 8000
    loud = np.full(sr // 10, 2.5)
    path = str(tmp_path / "loud.wav")
    write_wav(path, loud, sr)
    back, _ = read_wav(path)
    assert np.max(np.abs(back)) <= 1.0 + 1e-6
