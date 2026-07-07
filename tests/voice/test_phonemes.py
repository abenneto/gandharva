"""音节切分与元音识别的测试。"""

from __future__ import annotations

from gandharva.voice.phonemes import Phoneme, split_syllable, vowel_of


def test_pure_vowel() -> None:
    phs = split_syllable("a")
    assert len(phs) == 1
    assert phs[0].is_vowel
    assert phs[0].symbol == "a"
    assert phs[0].start == 0.0
    assert phs[0].end == 1.0


def test_consonant_vowel() -> None:
    phs = split_syllable("la")
    assert len(phs) == 2
    assert not phs[0].is_vowel
    assert phs[0].symbol == "l"
    assert phs[1].is_vowel
    assert phs[1].symbol == "a"
    # 辅音应只占开头一小段
    assert phs[0].end < 0.3


def test_multi_consonant_lead() -> None:
    phs = split_syllable("tra")
    assert phs[0].symbol == "tr"
    assert phs[-1].symbol == "a"


def test_no_vowel_defaults_to_a() -> None:
    phs = split_syllable("mmm")
    assert phs == [Phoneme("a", is_vowel=True, start=0.0, end=1.0)]


def test_vowel_of_picks_last_vowel() -> None:
    assert vowel_of("sol") == "o"
    assert vowel_of("mi") == "i"
    assert vowel_of("") == "a"


def test_empty_syllable() -> None:
    assert split_syllable("   ") == []


def test_uppercase_normalized() -> None:
    phs = split_syllable("LA")
    assert phs[0].symbol == "l"
    assert phs[1].symbol == "a"


def test_all_vowels_recognized() -> None:
    for v in "aeiou":
        assert vowel_of(v) == v


def test_diphthong_takes_last_vowel() -> None:
    # "hai" 里最后一个元音是 i
    assert vowel_of("hai") == "i"


def test_consonant_lead_duration_sums_to_one() -> None:
    phs = split_syllable("tra")
    total = sum(p.duration for p in phs)
    assert abs(total - 1.0) < 1e-9
