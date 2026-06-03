"""合成一段 C 大调音阶（do-re-mi-…-do）并写成 WAV。

运行：
    python examples/synth_scale/run.py
"""

from __future__ import annotations

import json
from pathlib import Path

from gandharva import Score, synthesize_score, write_wav


def main() -> None:
    here = Path(__file__).parent
    with open(here / "score.json", encoding="utf-8") as f:
        score = Score.from_dict(json.load(f))

    voice = synthesize_score(score)
    out = here / "scale.wav"
    write_wav(str(out), voice.samples, voice.sample_rate)
    print(f"已写出 {out}（{voice.duration:.2f}s，{voice.sample_rate} Hz）")


if __name__ == "__main__":
    main()
