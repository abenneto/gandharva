"""全局常量与默认参数。

集中放置采样率、帧长、音高参照等在多个模块间共享的数值，
避免魔法数字散落各处。
"""

from __future__ import annotations

# ---- 音频 ----
DEFAULT_SAMPLE_RATE: int = 24000
"""默认采样率（Hz）。24 kHz 对歌声足够，且比 44.1 kHz 更省算力。"""

# ---- 分析帧 ----
DEFAULT_FRAME_LENGTH: int = 1024
"""默认分析窗长（采样点）。"""

DEFAULT_HOP_LENGTH: int = 240
"""默认帧移（采样点）。240 @ 24 kHz ≈ 10 ms。"""

# ---- 音高参照 ----
A4_HZ: float = 440.0
"""A4 的频率（Hz），MIDI 音高 69。"""

A4_MIDI: int = 69
"""A4 对应的 MIDI 音高编号。"""

SEMITONES_PER_OCTAVE: int = 12
CENTS_PER_SEMITONE: int = 100
CENTS_PER_OCTAVE: int = SEMITONES_PER_OCTAVE * CENTS_PER_SEMITONE

# ---- 音高搜索范围（歌声）----
F0_FLOOR_HZ: float = 70.0
"""基频搜索下限，覆盖男低音。"""

F0_CEIL_HZ: float = 1000.0
"""基频搜索上限，覆盖女高音的高把位。"""

# ---- 数值 ----
EPS: float = 1e-10
"""避免除零 / log(0) 的极小值。"""

# TODO: F0_CEIL 对花腔女高音（可达 ~1500 Hz）偏低，之后按声部可配。
