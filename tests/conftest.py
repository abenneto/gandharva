"""供多个测试复用的 pytest fixtures。"""

from __future__ import annotations

import numpy as np
import pytest

from gandharva.core import Note, Score


@pytest.fixture
def sample_rate() -> int:
    return 24000


@pytest.fixture
def harmonic_tone(sample_rate: int):  # type: ignore[no-untyped-def]
    """返回一个生成谐波音（准浊音）的工厂函数。"""

    def _make(freq: float, dur: float = 0.6) -> np.ndarray:
        t = np.arange(int(dur * sample_rate)) / sample_rate
        sig = np.zeros_like(t)
        for k, amp in enumerate([1.0, 0.5, 0.3, 0.15], start=1):
            sig += amp * np.sin(2 * np.pi * freq * k * t)
        return 0.5 * sig / np.max(np.abs(sig))

    return _make


@pytest.fixture
def simple_score() -> Score:
    """一段简单的五音上行。"""
    pitches = [60, 62, 64, 65, 67]
    lyrics = ["la", "le", "li", "lo", "lu"]
    notes = [
        Note(i * 0.4, 0.4, p, ly) for i, (p, ly) in enumerate(zip(pitches, lyrics, strict=True))
    ]
    return Score(notes)
