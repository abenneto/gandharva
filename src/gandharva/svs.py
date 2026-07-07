"""歌声合成（SVS）引擎：乐谱 → 歌声波形。

流程：
1. 乐谱 → 基频轨迹（音高 / 旋律控制，含滑音、颤音、起音过冲）；
2. 每个音符的音节 → 元音 → 共振峰谱包络（发音 / 音色控制）；
3. 基频驱动激励，经谱包络整形，重叠相加成波形（声码器）。

这条链把“语音的发音建模”与“音乐的音高控制”合到一处，
是本框架“交叉集大成”的落点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SAMPLE_RATE,
)
from gandharva.core import Score, Voice
from gandharva.pitch.contour import (
    apply_attack_overshoot,
    apply_portamento,
    apply_vibrato,
    notes_to_f0,
)
from gandharva.vocoder.excitation import mixed_excitation
from gandharva.vocoder.synthesis import synthesize_from_envelopes
from gandharva.voice.formants import formant_envelope
from gandharva.voice.phonemes import vowel_of

FloatArray = NDArray[np.float64]


@dataclass
class SynthConfig:
    """SVS 合成参数。"""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_length: int = DEFAULT_FRAME_LENGTH
    hop_length: int = DEFAULT_HOP_LENGTH
    vibrato_depth_cents: float = 40.0
    vibrato_rate_hz: float = 5.5
    portamento_ms: float = 80.0
    overshoot_cents: float = 20.0


def synthesize_score(score: Score, config: SynthConfig | None = None) -> Voice:
    """把乐谱合成为歌声波形。

    依次完成：基频轨迹（含旋律表情）→ 逐帧元音共振峰包络 →
    基频驱动的混合激励 → 谱域整形 + 重叠相加。返回 :class:`~gandharva.core.Voice`。
    """
    cfg = config or SynthConfig()

    # 1. 基频轨迹：静态音高 → 加滑音 / 颤音 / 起音过冲
    contour = notes_to_f0(score, sample_rate=cfg.sample_rate, hop_length=cfg.hop_length)
    contour = apply_portamento(contour, transition_ms=cfg.portamento_ms)
    contour = apply_attack_overshoot(contour, overshoot_cents=cfg.overshoot_cents)
    contour = apply_vibrato(
        contour, rate_hz=cfg.vibrato_rate_hz, depth_cents=cfg.vibrato_depth_cents
    )

    n_frames = len(contour)

    # 2. 逐帧元音 → 共振峰谱包络
    vowels = _frame_vowels(score, n_frames, cfg.hop_length, cfg.sample_rate)
    envelopes = _build_envelopes(vowels, cfg.frame_length, cfg.sample_rate)

    # 3. 基频驱动激励
    excitation = mixed_excitation(contour.values, cfg.hop_length, cfg.sample_rate)

    # 4. 谱域整形 + OLA 合成
    samples = synthesize_from_envelopes(excitation, envelopes, cfg.frame_length, cfg.hop_length)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0:
        samples = samples * (0.95 / peak)

    return Voice(samples=samples, sample_rate=cfg.sample_rate, f0=contour)


def _frame_vowels(score: Score, n_frames: int, hop_length: int, sample_rate: int) -> list[str]:
    """给每一帧分配它所属音符的元音（休止符 / 空档记为空串）。"""
    frame_times = np.arange(n_frames) * hop_length / sample_rate
    vowels = [""] * n_frames
    for note in score:
        if note.is_rest:
            continue
        v = vowel_of(note.lyric)
        for i, t in enumerate(frame_times):
            if note.start <= t < note.end:
                vowels[i] = v
    return vowels


def _build_envelopes(
    vowels: list[str],
    frame_length: int,
    sample_rate: int,
    *,
    smooth_frames: int = 3,
) -> FloatArray:
    """按逐帧元音生成共振峰谱包络矩阵，并在元音切换处做时间平滑。

    形状 ``(n_frames, frame_length // 2 + 1)``。清音 / 空档帧包络记为极小值，
    避免噪声被整形出可闻能量。
    """
    n_frames = len(vowels)
    n_bins = frame_length // 2 + 1
    envelopes = np.full((n_frames, n_bins), 1e-3, dtype=np.float64)

    # 缓存每种元音的包络，避免重复计算
    cache: dict[str, FloatArray] = {}
    for i, v in enumerate(vowels):
        if not v:
            continue
        if v not in cache:
            cache[v] = formant_envelope(v, frame_length, sample_rate)
        envelopes[i] = cache[v]

    # 沿时间轴做滑动平均，弱化相邻元音间的突变
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames) / smooth_frames
        envelopes = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=envelopes
        )
    return envelopes
