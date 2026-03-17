"""声码器分析 → 合成往返测试。"""

from __future__ import annotations

import numpy as np

from gandharva.vocoder.analysis import analyze, estimate_aperiodicity
from gandharva.vocoder.excitation import mixed_excitation
from gandharva.vocoder.synthesis import synthesize


def _voiced_signal(freq: float, sr: int, dur: float = 0.6) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    # 一个带几个谐波的准周期信号，粗略模拟浊音
    sig = np.zeros_like(t)
    for k, amp in enumerate([1.0, 0.5, 0.3, 0.15], start=1):
        sig += amp * np.sin(2 * np.pi * freq * k * t)
    return 0.5 * sig / np.max(np.abs(sig))


def test_analyze_returns_consistent_lengths() -> None:
    sr = 24000
    sig = _voiced_signal(160.0, sr)
    frames = analyze(sig, sample_rate=sr)
    assert frames.n_frames == len(frames.gains) == len(frames.f0)
    assert len(frames.lpc_coeffs) == frames.n_frames


def _spectral_fundamental(signal: np.ndarray, sr: int, fmax: float = 400.0) -> float:
    """用低频段的谱峰估计基频，对脉冲驱动信号比 YIN 更稳（不受倍频程误判影响）。"""
    windowed = signal * np.hanning(len(signal))
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    band = (freqs > 50.0) & (freqs < fmax)
    return float(freqs[band][np.argmax(spec[band])])


def test_resynthesis_preserves_pitch() -> None:
    sr = 24000
    freq = 180.0
    sig = _voiced_signal(freq, sr)
    frames = analyze(sig, sample_rate=sr)
    ap = estimate_aperiodicity(sig, frames.f0, sample_rate=sr)
    exc = mixed_excitation(frames.f0, frames.hop_length, sr, aperiodicity=ap)
    out = synthesize(frames, exc)

    # 合成信号谱峰应落在原始基频附近（YIN 对脉冲信号易倍频程误判，故用谱峰）
    assert abs(_spectral_fundamental(out, sr) - freq) / freq < 0.05


def test_synthesis_output_finite() -> None:
    sr = 24000
    sig = _voiced_signal(200.0, sr)
    frames = analyze(sig, sample_rate=sr)
    exc = mixed_excitation(frames.f0, frames.hop_length, sr)
    out = synthesize(frames, exc)
    assert np.all(np.isfinite(out))
