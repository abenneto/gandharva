"""提取一段歌声的基频轨迹并打印统计（分析）。

为了让 YIN 能稳定跟踪，这里用一个谐波丰富、带轻微颤音的合成元音当输入
（比纯脉冲激励的谱更接近真实嗓音），再估计它的基频。

运行：
    python examples/analyze_f0/run.py
"""

from __future__ import annotations

import numpy as np

from gandharva import estimate_f0
from gandharva.convert_units import hz_to_midi, midi_to_hz


def _sung_vowel(midi_pitch: float, sr: int, dur: float = 2.0) -> np.ndarray:
    """合成一个带颤音的谐波元音：基频 + 若干衰减谐波。"""
    t = np.arange(int(dur * sr)) / sr
    base = float(midi_to_hz(midi_pitch))
    # 5.5 Hz、±40 音分的颤音
    vibrato = base * 2.0 ** ((40.0 / 1200.0) * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(vibrato) / sr
    sig = np.zeros_like(t)
    for k, amp in enumerate([1.0, 0.5, 0.3, 0.2, 0.1], start=1):
        sig += amp * np.sin(k * phase)
    return 0.5 * sig / np.max(np.abs(sig))


def main() -> None:
    sr = 24000
    signal = _sung_vowel(69, sr)  # A4 = MIDI 69 ≈ 440 Hz

    contour = estimate_f0(signal, sample_rate=sr)
    voiced = contour.values[contour.voiced]

    print(f"帧数：{len(contour)}")
    print(f"浊音比例：{contour.voiced.mean():.2f}")
    if voiced.size:
        median_hz = float(np.median(voiced))
        print(f"中位基频：{median_hz:.1f} Hz（≈ MIDI {hz_to_midi(median_hz):.1f}）")
        print(f"基频抖动范围：{voiced.min():.1f} – {voiced.max():.1f} Hz（颤音所致）")


if __name__ == "__main__":
    main()
