"""歌声转换（SVC）：改变已录歌声的音高，保持音色。

与 SVS 相反，SVC 从真实歌声出发：分析出基频 + 谱包络，
只移动音高（激励），保留声道包络（音色），再重新合成。
formant-preserving 变调是它与朴素变速变调的关键区别。
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
from gandharva.convert_units import hz_to_midi, midi_to_hz
from gandharva.core import F0Contour, Voice
from gandharva.vocoder.analysis import analyze, estimate_aperiodicity
from gandharva.vocoder.excitation import mixed_excitation
from gandharva.vocoder.synthesis import synthesize

FloatArray = NDArray[np.float64]


@dataclass
class ConvertConfig:
    """SVC 转换参数。"""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_length: int = DEFAULT_FRAME_LENGTH
    hop_length: int = DEFAULT_HOP_LENGTH
    lpc_order: int = 24


def convert_pitch(
    signal: FloatArray, semitones: float, config: ConvertConfig | None = None
) -> Voice:
    """把歌声整体升降 ``semitones`` 个半音，保持时长与音色。

    做法：分析 → 基频乘以变调系数（激励音高变）→ LPC 包络不动（音色不变）
    → 重新合成。
    """
    cfg = config or ConvertConfig()
    frames = analyze(
        signal,
        sample_rate=cfg.sample_rate,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        lpc_order=cfg.lpc_order,
    )
    factor = 2.0 ** (semitones / 12.0)
    shifted_f0 = frames.f0 * factor

    ap = estimate_aperiodicity(
        signal,
        frames.f0,
        sample_rate=cfg.sample_rate,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )
    excitation = mixed_excitation(shifted_f0, cfg.hop_length, cfg.sample_rate, aperiodicity=ap)
    samples = synthesize(frames, excitation)

    contour = F0Contour(shifted_f0, cfg.hop_length, cfg.sample_rate)
    return Voice(samples=samples, sample_rate=cfg.sample_rate, f0=contour)


def _warp_lpc_poles(a: FloatArray, warp: float) -> FloatArray:
    """把 LPC 全极点滤波器的极点半径向外 / 内缩放，等效平移共振峰频率。

    对每个极点 ``p``，用 ``p * warp`` 替换后重建多项式：``warp>1`` 让
    共振峰整体上移（声道听感更“小 / 亮”），``warp<1`` 反之。这样即可在
    变调之外，独立地做“变声”。
    """
    if abs(warp - 1.0) < 1e-6:
        return a
    roots = np.roots(a)
    # 用角度不变、仅改半径的方式，把共振峰频率按 warp 缩放
    angles = np.angle(roots) * warp
    radii = np.abs(roots)
    # 保证仍在单位圆内
    radii = np.clip(radii, 0.0, 0.999)
    warped = radii * np.exp(1j * angles)
    new_a = np.poly(warped)
    return np.real(new_a).astype(np.float64)


def convert_voice(
    signal: FloatArray,
    semitones: float,
    *,
    formant_shift: float = 1.0,
    config: ConvertConfig | None = None,
) -> Voice:
    """在变调的同时独立平移共振峰（音色变换）。

    ``semitones`` 控制音高，``formant_shift`` 控制共振峰频率缩放：
    两者解耦，即可做出“升调但音色不变”或“音高不变但更明亮”的效果。
    """
    cfg = config or ConvertConfig()
    frames = analyze(
        signal,
        sample_rate=cfg.sample_rate,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        lpc_order=cfg.lpc_order,
    )
    if abs(formant_shift - 1.0) > 1e-6:
        frames.lpc_coeffs = [_warp_lpc_poles(a, formant_shift) for a in frames.lpc_coeffs]

    factor = 2.0 ** (semitones / 12.0)
    shifted_f0 = frames.f0 * factor
    ap = estimate_aperiodicity(
        signal,
        frames.f0,
        sample_rate=cfg.sample_rate,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )
    excitation = mixed_excitation(shifted_f0, cfg.hop_length, cfg.sample_rate, aperiodicity=ap)
    samples = synthesize(frames, excitation)
    contour = F0Contour(shifted_f0, cfg.hop_length, cfg.sample_rate)
    return Voice(samples=samples, sample_rate=cfg.sample_rate, f0=contour)


# 大调音阶的半音偏移（相对主音）
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)


def snap_to_key(
    f0: FloatArray, tonic_midi: int, scale: tuple[int, ...] = _MAJOR_SCALE
) -> FloatArray:
    """把逐帧基频吸附到指定调式的最近音级（自动修音 / autotune）。

    对每个浊音帧，把其 MIDI 音高映射到 ``tonic`` 所定调式里最近的音级。
    清音帧保持为 0。
    """
    out = np.zeros_like(f0)
    voiced = f0 > 0
    if not np.any(voiced):
        return out

    midi = hz_to_midi(f0[voiced])
    # 相对主音的音级（对八度取模）
    rel = midi - tonic_midi
    octave = np.floor(rel / 12.0)
    degree = rel - octave * 12.0
    scale_arr = np.asarray(scale, dtype=np.float64)
    # 找每个音对最近的音级
    nearest = scale_arr[np.argmin(np.abs(degree[:, None] - scale_arr[None, :]), axis=1)]
    snapped_midi = tonic_midi + octave * 12.0 + nearest
    out[voiced] = midi_to_hz(snapped_midi)
    return out


def convert_to_key(
    signal: FloatArray,
    tonic_midi: int,
    *,
    scale: tuple[int, ...] = _MAJOR_SCALE,
    config: ConvertConfig | None = None,
) -> Voice:
    """把歌声修到指定调式上（保持音色），返回修音后的歌声。"""
    cfg = config or ConvertConfig()
    frames = analyze(
        signal,
        sample_rate=cfg.sample_rate,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        lpc_order=cfg.lpc_order,
    )
    snapped = snap_to_key(frames.f0, tonic_midi, scale)
    ap = estimate_aperiodicity(
        signal,
        frames.f0,
        sample_rate=cfg.sample_rate,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )
    excitation = mixed_excitation(snapped, cfg.hop_length, cfg.sample_rate, aperiodicity=ap)
    samples = synthesize(frames, excitation)
    contour = F0Contour(snapped, cfg.hop_length, cfg.sample_rate)
    return Voice(samples=samples, sample_rate=cfg.sample_rate, f0=contour)
