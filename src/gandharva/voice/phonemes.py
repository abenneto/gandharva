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


def split_syllable(syllable: str) -> list[Phoneme]:
    """把一个音节拆成音素列表。

    规则（面向 SVS 的简化模型）：

    - 取音节里最后一个元音字母作为元音核心；
    - 它之前的字母合并成一个辅音起始（占开头一小段）；
    - 找不到元音时，退化为默认元音 ``a``。
    """
    text = syllable.strip().lower()
    if not text:
        return []

    vowel_positions = [i for i, ch in enumerate(text) if ch in VOWELS]
    if not vowel_positions:
        return [Phoneme("a", is_vowel=True, start=0.0, end=1.0)]

    v_idx = vowel_positions[-1]
    vowel = text[v_idx]
    lead = text[:v_idx]

    phonemes: list[Phoneme] = []
    if lead:
        phonemes.append(Phoneme(lead, is_vowel=False, start=0.0, end=_CONSONANT_LEAD))
        phonemes.append(Phoneme(vowel, is_vowel=True, start=_CONSONANT_LEAD, end=1.0))
    else:
        phonemes.append(Phoneme(vowel, is_vowel=True, start=0.0, end=1.0))
    return phonemes


def vowel_of(syllable: str) -> str:
    """返回音节的元音核心（找不到则默认 ``a``）。"""
    for ph in reversed(split_syllable(syllable)):
        if ph.is_vowel:
            return ph.symbol
    return "a"
