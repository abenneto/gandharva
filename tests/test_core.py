"""核心容器（Note / Score）的测试。"""

from __future__ import annotations

import pytest

from gandharva.core import Note, Score
from gandharva.exceptions import ScoreError


def test_note_end_and_rest() -> None:
    note = Note(1.0, 0.5, 60, "la")
    assert note.end == 1.5
    assert not note.is_rest
    assert Note(0.0, 0.2, 60, "  ").is_rest


def test_note_rejects_nonpositive_duration() -> None:
    with pytest.raises(ScoreError):
        Note(0.0, 0.0, 60)
    with pytest.raises(ScoreError):
        Note(0.0, -1.0, 60)


def test_note_rejects_negative_start() -> None:
    with pytest.raises(ScoreError):
        Note(-0.1, 0.5, 60)


def test_score_sorts_notes() -> None:
    score = Score([Note(1.0, 0.5, 62), Note(0.0, 0.5, 60)])
    assert [n.start for n in score] == [0.0, 1.0]


def test_score_detects_overlap() -> None:
    with pytest.raises(ScoreError):
        Score([Note(0.0, 1.0, 60), Note(0.5, 1.0, 62)])


def test_score_duration() -> None:
    score = Score([Note(0.0, 0.5, 60), Note(0.5, 0.5, 62)])
    assert score.duration == 1.0
    assert Score([]).duration == 0.0


def test_from_dict_round_trip() -> None:
    data = {
        "notes": [
            {"start": 0.0, "duration": 0.5, "pitch": 60, "lyric": "do"},
            {"start": 0.5, "duration": 0.5, "pitch": 64},
        ]
    }
    score = Score.from_dict(data)
    assert len(score) == 2
    assert score.notes[1].lyric == "a"  # 默认元音


def test_from_dict_missing_field() -> None:
    with pytest.raises(ScoreError):
        Score.from_dict({"notes": [{"start": 0.0, "pitch": 60}]})


def test_from_dict_missing_notes() -> None:
    with pytest.raises(ScoreError):
        Score.from_dict({})
