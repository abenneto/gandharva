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

from gandharva.constants import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SAMPLE_RATE,
)
from gandharva.core import Score


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


def synthesize_score(score: Score, config: SynthConfig | None = None) -> None:
    """把乐谱合成为歌声波形（后续提交补全实现）。"""
    raise NotImplementedError
