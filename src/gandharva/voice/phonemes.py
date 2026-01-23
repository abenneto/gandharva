"""音节 → 音素切分。

针对歌声合成的简化发音建模：把一个演唱音节拆成
（可选的）辅音起始 + 元音核心。元音承载音高与共振峰，
辅音只占很短的起始时间。
"""

from __future__ import annotations

from dataclasses import dataclass

VOWELS = frozenset("aeiou")

# 常见辅音到近似发音时长比例（占音节起始的相对权重）
_CONSONANT_LEAD = 0.12


@dataclass
class Phoneme:
    """一个音素及其在音节内的相对时间区间 [0, 1]。"""

    symbol: str
    is_vowel: bool
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start
