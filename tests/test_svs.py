"""歌声合成（SVS）端到端冒烟测试。"""

from __future__ import annotations

import numpy as np

from gandharva.core import Note, Score, Voice
from gandharva.svs import SynthConfig, synthesize_score


def _scale_score() -> Score:
    pitches = [60, 62, 64, 65, 67]
    lyrics = ["la", "le", "li", "lo", "lu"]
    notes = [
        Note(i * 0.4, 0.4, p, ly) for i, (p, ly) in enumerate(zip(pitches, lyrics, strict=True))
    ]
    return Score(notes)


def test_synthesize_returns_voice() -> None:
    voice = synthesize_score(_scale_score())
    assert isinstance(voice, Voice)
    assert voice.sample_rate == SynthConfig().sample_rate


def test_output_is_finite_and_bounded() -> None:
    voice = synthesize_score(_scale_score())
    assert np.all(np.isfinite(voice.samples))
    assert np.max(np.abs(voice.samples)) <= 1.0 + 1e-6


def test_duration_matches_score() -> None:
    score = _scale_score()
    voice = synthesize_score(score)
    # 允许尾帧补零带来的少量误差
    assert abs(voice.duration - score.duration) < 0.1


def test_carries_f0_contour() -> None:
    voice = synthesize_score(_scale_score())
    assert voice.f0 is not None
    assert voice.f0.voiced.mean() > 0.8


def test_samples_length_matches_f0_frames() -> None:
    voice = synthesize_score(_scale_score())
    assert voice.f0 is not None
    hop = SynthConfig().hop_length
    # 波形长度应约等于帧数 * hop（允许一帧的尾差）
    expected = len(voice.f0) * hop
    assert abs(len(voice.samples) - expected) <= SynthConfig().frame_length


def test_longer_score_longer_audio() -> None:
    short = synthesize_score(Score([Note(0.0, 0.5, 60, "la")]))
    long = synthesize_score(Score([Note(0.0, 2.0, 60, "la")]))
    assert len(long.samples) > len(short.samples)


def test_custom_config_changes_sample_rate() -> None:
    cfg = SynthConfig(sample_rate=16000)
    voice = synthesize_score(Score([Note(0.0, 0.5, 60, "la")]), cfg)
    assert voice.sample_rate == 16000


def test_rest_produces_silence_region() -> None:
    score = Score([Note(0.0, 0.4, 60, "la"), Note(0.8, 0.4, 64, "le")])
    voice = synthesize_score(score)
    assert voice.f0 is not None
    # 0.4~0.8s 的空档应基本清音
    gap = (voice.f0.times >= 0.45) & (voice.f0.times < 0.75)
    assert np.mean(voice.f0.voiced[gap]) < 0.2
