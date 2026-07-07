"""CLI 子命令的测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gandharva.audio import write_wav
from gandharva.cli import main


def test_synth_from_json(tmp_path: Path) -> None:
    score = {
        "notes": [
            {"start": 0.0, "duration": 0.4, "pitch": 60, "lyric": "la"},
            {"start": 0.4, "duration": 0.4, "pitch": 64, "lyric": "le"},
        ]
    }
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(score), encoding="utf-8")
    out = tmp_path / "out.wav"
    rc = main(["synth", str(score_path), "-o", str(out)])
    assert rc == 0
    assert out.exists()


def test_f0_command(tmp_path: Path) -> None:
    sr = 24000
    t = np.arange(sr // 2) / sr
    sig = np.sin(2 * np.pi * 220.0 * t)
    wav = tmp_path / "tone.wav"
    write_wav(str(wav), sig, sr)
    assert main(["f0", str(wav)]) == 0


def test_convert_command(tmp_path: Path) -> None:
    sr = 24000
    t = np.arange(sr // 2) / sr
    sig = 0.5 * np.sin(2 * np.pi * 180.0 * t)
    wav = tmp_path / "in.wav"
    write_wav(str(wav), sig, sr)
    out = tmp_path / "shifted.wav"
    rc = main(["convert", str(wav), "-s", "2", "-o", str(out)])
    assert rc == 0
    assert out.exists()


def test_no_command_prints_help() -> None:
    assert main([]) == 1


def test_bad_score_returns_error_code(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"notes": [{"start": 0.0, "pitch": 60}]}', encoding="utf-8")
    out = tmp_path / "out.wav"
    assert main(["synth", str(bad), "-o", str(out)]) == 2
    assert not out.exists()
