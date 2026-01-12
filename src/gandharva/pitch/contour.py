"""由乐谱生成基频轨迹（F0 contour）。

把离散的音符音高转成逐帧、连续的基频曲线，是“旋律控制”的核心：
静态音高只是起点，真正像人声还需要滑音、颤音与起音过冲。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gandharva.constants import DEFAULT_HOP_LENGTH, DEFAULT_SAMPLE_RATE
from gandharva.convert_units import midi_to_hz
from gandharva.core import F0Contour, Score

FloatArray = NDArray[np.float64]


def notes_to_f0(
    score: Score,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> F0Contour:
    """把乐谱渲染成阶梯状（尚无过渡）的基频轨迹。

    休止符对应的帧记为 0（清音）。后续函数在此基础上叠加滑音 / 颤音。
    """
    n_frames = int(np.ceil(score.duration * sample_rate / hop_length)) + 1
    f0 = np.zeros(n_frames, dtype=np.float64)
    frame_times = np.arange(n_frames) * hop_length / sample_rate

    for note in score:
        if note.is_rest:
            continue
        mask = (frame_times >= note.start) & (frame_times < note.end)
        f0[mask] = float(midi_to_hz(note.pitch))

    return F0Contour(values=f0, hop_length=hop_length, sample_rate=sample_rate)


def apply_portamento(contour: F0Contour, transition_ms: float = 80.0) -> F0Contour:
    """在浊音音高突变处加入滑音（portamento）。

    在对数频率域，对每个音高台阶的边界用 S 形（余弦）曲线平滑过渡，
    过渡时长为 ``transition_ms`` 毫秒。清音段（0）保持不变。
    """
    values = contour.values.copy()
    voiced = values > 0
    log_f = np.zeros_like(values)
    log_f[voiced] = np.log2(values[voiced])

    trans_frames = max(1, int(transition_ms / 1000.0 * contour.sample_rate / contour.hop_length))
    # 找到浊音内部的音高跳变位置
    boundaries = np.where(
        voiced[:-1] & voiced[1:] & (np.abs(np.diff(log_f)) > 1e-4)
    )[0]

    for b in boundaries:
        start = max(0, b - trans_frames // 2)
        end = min(len(values) - 1, b + trans_frames // 2 + 1)
        if end <= start or not (voiced[start] and voiced[end]):
            continue
        # 余弦渐变从 start 值滑到 end 值
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, end - start)))
        log_f[start:end] = log_f[start] + (log_f[end] - log_f[start]) * ramp

    out = values.copy()
    out[voiced] = 2.0 ** log_f[voiced]
    return F0Contour(values=out, hop_length=contour.hop_length, sample_rate=contour.sample_rate)


def apply_vibrato(
    contour: F0Contour,
    *,
    rate_hz: float = 5.5,
    depth_cents: float = 40.0,
    onset_ms: float = 300.0,
) -> F0Contour:
    """在持续浊音段上叠加颤音（vibrato）。

    颤音为一个正弦音高调制，深度以音分计。每个浊音段在 ``onset_ms``
    毫秒后才逐渐进入满深度，模拟歌手先稳住音高再加颤音的习惯。
    """
    values = contour.values.copy()
    voiced = values > 0
    times = contour.times
    depth_ratio = depth_cents / 1200.0  # 转到对数域倍率

    # 逐个连续浊音段处理
    idx = 0
    n = len(values)
    while idx < n:
        if not voiced[idx]:
            idx += 1
            continue
        seg_start = idx
        while idx < n and voiced[idx]:
            idx += 1
        seg = slice(seg_start, idx)
        local_t = times[seg] - times[seg_start]
        # 渐入包络
        onset = np.clip(local_t / (onset_ms / 1000.0), 0.0, 1.0)
        mod = depth_ratio * onset * np.sin(2 * np.pi * rate_hz * local_t)
        values[seg] = values[seg] * (2.0**mod)

    return F0Contour(values=values, hop_length=contour.hop_length, sample_rate=contour.sample_rate)


def apply_attack_overshoot(
    contour: F0Contour,
    *,
    overshoot_cents: float = 25.0,
    attack_ms: float = 60.0,
) -> F0Contour:
    """在每个浊音段起始处加入音高过冲（attack overshoot）。

    歌手起音时常先略微冲过目标音高再回落。这里在段首 ``attack_ms``
    毫秒内叠加一个先扬后落的半周期，峰值为 ``overshoot_cents`` 音分。
    """
    values = contour.values.copy()
    voiced = values > 0
    times = contour.times
    peak_ratio = overshoot_cents / 1200.0
    attack_s = attack_ms / 1000.0

    idx = 0
    n = len(values)
    while idx < n:
        if not voiced[idx]:
            idx += 1
            continue
        seg_start = idx
        while idx < n and voiced[idx]:
            idx += 1
        local_t = times[seg_start:idx] - times[seg_start]
        env = np.where(
            local_t < attack_s,
            np.sin(np.pi * local_t / attack_s),  # 半周期：0→1→0
            0.0,
        )
        values[seg_start:idx] = values[seg_start:idx] * (2.0 ** (peak_ratio * env))

    return F0Contour(values=values, hop_length=contour.hop_length, sample_rate=contour.sample_rate)
