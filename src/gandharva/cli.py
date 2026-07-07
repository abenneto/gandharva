"""命令行入口。

三个子命令：
- ``synth``   由 JSON 乐谱合成歌声（SVS）；
- ``convert`` 对已有歌声变调 / 修音（SVC）；
- ``f0``      提取歌声基频轨迹并打印统计。
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from gandharva.audio import read_wav, write_wav
from gandharva.core import Score
from gandharva.pitch.estimate import estimate_f0
from gandharva.svc import convert_pitch, convert_to_key
from gandharva.svs import synthesize_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gandharva",
        description="歌声合成（SVS）与转换（SVC）命令行工具",
    )
    sub = parser.add_subparsers(dest="command")

    p_synth = sub.add_parser("synth", help="由 JSON 乐谱合成歌声")
    p_synth.add_argument("score", help="乐谱 JSON 文件路径")
    p_synth.add_argument("-o", "--output", default="out.wav", help="输出 WAV 路径")

    p_convert = sub.add_parser("convert", help="对歌声变调")
    p_convert.add_argument("input", help="输入 WAV 路径")
    p_convert.add_argument("-s", "--semitones", type=float, required=True, help="变调半音数")
    p_convert.add_argument("-o", "--output", default="converted.wav", help="输出 WAV 路径")

    p_f0 = sub.add_parser("f0", help="提取基频轨迹")
    p_f0.add_argument("input", help="输入 WAV 路径")

    p_key = sub.add_parser("key", help="把歌声修到指定调式（autotune）")
    p_key.add_argument("input", help="输入 WAV 路径")
    p_key.add_argument("-t", "--tonic", type=int, default=60, help="主音 MIDI 音高（默认 60 = C4）")
    p_key.add_argument("-o", "--output", default="tuned.wav", help="输出 WAV 路径")

    return parser


def _load_score(path: str) -> Score:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Score.from_dict(data)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    if command is None:
        parser.print_help()
        return 1

    if command == "synth":
        score = _load_score(args.score)
        voice = synthesize_score(score)
        write_wav(args.output, voice.samples, voice.sample_rate)
        print(f"合成完成：{args.output}（{voice.duration:.2f}s）")
        return 0

    if command == "convert":
        signal, sr = read_wav(args.input)
        voice = convert_pitch(signal, args.semitones)
        write_wav(args.output, voice.samples, voice.sample_rate)
        print(f"变调完成：{args.output}（{args.semitones:+g} 半音）")
        return 0

    if command == "f0":
        signal, sr = read_wav(args.input)
        contour = estimate_f0(signal, sample_rate=sr)
        voiced = contour.values[contour.voiced]
        if voiced.size:
            print(
                f"帧数={len(contour)} 浊音比例={contour.voiced.mean():.2f} "
                f"中位基频={np.median(voiced):.1f}Hz"
            )
        else:
            print("未检测到浊音帧")
        return 0

    if command == "key":
        signal, sr = read_wav(args.input)
        voice = convert_to_key(signal, args.tonic)
        write_wav(args.output, voice.samples, voice.sample_rate)
        print(f"修音完成：{args.output}（主音 MIDI {args.tonic}）")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
