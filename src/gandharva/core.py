"""核心数据容器：音符与乐谱。

一份乐谱由若干带歌词的音符组成，是 SVS 引擎的输入。
时间一律以秒为单位（float），音高以 MIDI 编号表示（int，允许微分音时可传 float）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from gandharva.exceptions import ScoreError


@dataclass
class Note:
    """一个带歌词音节的音符。

    参数
    ----
    start:
        起始时刻（秒）。
    duration:
        时长（秒），必须为正。
    pitch:
        MIDI 音高编号，60 = 中央 C。允许 float 以表达微分音。
    lyric:
        该音符演唱的音节（如 "a"、"la"、"ma"）。休止符用空串。
    """

    start: float
    duration: float
    pitch: float
    lyric: str = "a"

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ScoreError(f"音符时长必须为正，得到 {self.duration}")
        if self.start < 0:
            raise ScoreError(f"音符起始时刻不能为负，得到 {self.start}")

    @property
    def end(self) -> float:
        """结束时刻（秒）。"""
        return self.start + self.duration

    @property
    def is_rest(self) -> bool:
        """歌词为空视为休止符。"""
        return self.lyric.strip() == ""


@dataclass
class Score:
    """一段单声部乐谱：按时间排序的音符序列。"""

    notes: list[Note] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.notes.sort(key=lambda n: n.start)
        self._check_no_overlap()

    def _check_no_overlap(self) -> None:
        for a, b in zip(self.notes, self.notes[1:], strict=False):
            # 允许极小的浮点误差
            if b.start + 1e-6 < a.end:
                raise ScoreError(
                    f"音符重叠：[{a.start:.3f}, {a.end:.3f}] 与起于 {b.start:.3f} 的音符"
                )

    @property
    def duration(self) -> float:
        """整段乐谱时长（秒）。"""
        return self.notes[-1].end if self.notes else 0.0

    def __len__(self) -> int:
        return len(self.notes)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.notes)


@dataclass
class F0Contour:
    """逐帧基频轨迹。

    ``values`` 中 0 表示清音 / 无声段；``hop_length`` 与 ``sample_rate``
    共同决定每帧对应的时刻。
    """

    values: NDArray[np.float64]
    hop_length: int
    sample_rate: int

    @property
    def times(self) -> NDArray[np.float64]:
        """每帧中心对应的时刻（秒）。"""
        n = len(self.values)
        return np.arange(n, dtype=np.float64) * self.hop_length / self.sample_rate

    @property
    def voiced(self) -> NDArray[np.bool_]:
        """布尔掩码：该帧是否为浊音（基频 > 0）。"""
        return self.values > 0.0

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class Voice:
    """已渲染的歌声：波形加上采样率，可选携带生成它的基频轨迹。"""

    samples: NDArray[np.float64]
    sample_rate: int
    f0: F0Contour | None = None

    @property
    def duration(self) -> float:
        """波形时长（秒）。"""
        return len(self.samples) / self.sample_rate
