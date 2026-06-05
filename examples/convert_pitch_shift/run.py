"""对一段合成歌声做保持音色的变调（SVC）。

为了自足，这个例子先合成一小段歌声当作「输入」，
再把它整体升 4 个半音，写出对比 WAV。

运行：
    python examples/convert_pitch_shift/run.py
"""

from __future__ import annotations

from pathlib import Path

from gandharva import Note, Score, convert_pitch, synthesize_score, write_wav


def main() -> None:
    here = Path(__file__).parent

    # 先造一段输入歌声
    score = Score(
        [
            Note(0.0, 0.5, 62, "la"),
            Note(0.5, 0.5, 64, "la"),
            Note(1.0, 1.0, 65, "la"),
        ]
    )
    source = synthesize_score(score)
    write_wav(str(here / "source.wav"), source.samples, source.sample_rate)

    # 升四度（+5 半音），保持时长与音色
    shifted = convert_pitch(source.samples, semitones=5)
    write_wav(str(here / "shifted_up.wav"), shifted.samples, shifted.sample_rate)

    print("已写出 source.wav 与 shifted_up.wav（+5 半音）")


if __name__ == "__main__":
    main()
