"""乐谱 → 基频轨迹（滑音 / 颤音 / 起音过冲）的测试。"""

from __future__ import annotations

import numpy as np

from gandharva.convert_units import hz_to_midi
from gandharva.core import Note, Score
from gandharva.pitch.contour import (
    apply_attack_overshoot,
    apply_portamento,
    apply_vibrato,
    notes_to_f0,
)


def _three_note_score() -> Score:
    return Score(
        [
            Note(0.0, 0.5, 60, "do"),
            Note(0.5, 0.5, 64, "mi"),
            Note(1.0, 0.5, 67, "so"),
        ]
    )


def test_notes_to_f0_hits_target_pitches() -> None:
    contour = notes_to_f0(_three_note_score())
    voiced = contour.values[contour.voiced]
    midi = hz_to_midi(voiced)
    # 台阶应集中在 60 / 64 / 67 附近
    rounded = np.round(midi)
    assert set(np.unique(rounded)).issubset({60, 64, 67})


def test_rest_is_unvoiced() -> None:
    score = Score([Note(0.0, 0.3, 60, "a"), Note(0.5, 0.3, 62, "a")])
    contour = notes_to_f0(score)
    # 0.3~0.5s 之间应存在清音帧
    gap = (contour.times >= 0.32) & (contour.times < 0.48)
    assert np.any(~contour.voiced[gap])


def test_portamento_smooths_jump() -> None:
    contour = notes_to_f0(_three_note_score())
    smooth = apply_portamento(contour, transition_ms=100.0)

    # 平滑后相邻帧的最大对数跳变应变小
    def max_log_jump(c: np.ndarray) -> float:
        v = c[c > 0]
        return float(np.max(np.abs(np.diff(np.log2(v)))))

    assert max_log_jump(smooth.values) < max_log_jump(contour.values)


def test_vibrato_adds_oscillation() -> None:
    contour = notes_to_f0(Score([Note(0.0, 1.0, 62, "a")]))
    vib = apply_vibrato(contour, depth_cents=50.0, onset_ms=100.0)
    # 稳定段末尾应出现调制，标准差明显大于原始（原始为常数）
    tail = vib.values[-40:]
    assert np.std(tail) > 0.5


def test_attack_overshoot_peaks_above_target() -> None:
    contour = notes_to_f0(Score([Note(0.0, 0.5, 60, "a")]))
    target = contour.values[contour.voiced][20]
    over = apply_attack_overshoot(contour, overshoot_cents=40.0, attack_ms=80.0)
    # 段首应短暂高于目标音高
    assert np.max(over.values[:10]) > target


def test_single_note_contour_golden_hz() -> None:
    # A4 (MIDI 69) 应精确渲染成 440 Hz 的稳定台阶
    contour = notes_to_f0(Score([Note(0.0, 0.5, 69, "a")]))
    voiced = contour.values[contour.voiced]
    assert np.allclose(voiced, 440.0, atol=1e-6)


def test_portamento_is_monotonic_between_notes() -> None:
    # 上行两音之间，滑音段的对数音高应单调不减
    score = Score([Note(0.0, 0.5, 60, "a"), Note(0.5, 0.5, 67, "a")])
    contour = notes_to_f0(score)
    smooth = apply_portamento(contour, transition_ms=120.0)
    # 取过渡窗附近
    idx = np.searchsorted(smooth.times, 0.5)
    window = smooth.values[idx - 6 : idx + 6]
    window = window[window > 0]
    diffs = np.diff(np.log2(window))
    assert np.all(diffs >= -1e-6)


def test_vibrato_rate_matches_setting() -> None:
    # 颤音周期数应与 rate * duration 大致相符
    dur = 2.0
    rate = 6.0
    contour = notes_to_f0(Score([Note(0.0, dur, 62, "a")]))
    vib = apply_vibrato(contour, rate_hz=rate, depth_cents=60.0, onset_ms=50.0)
    # 用去趋势后的过零数估计周期数
    v = vib.values[vib.voiced]
    centered = v - np.mean(v)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(centered)))) / 2
    # 每个周期两次过零，允许 ±30% 误差
    expected = rate * dur
    assert 0.7 * expected < zero_crossings / 2 < 1.3 * expected
