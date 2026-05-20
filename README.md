# gandharva

[![CI](https://github.com/abenneto/gandharva/actions/workflows/ci.yml/badge.svg)](https://github.com/abenneto/gandharva/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-orange)](https://github.com/astral-sh/ruff)

> 语音 × 音乐的交叉实验：**歌声合成（SVS）** 与 **歌声转换（SVC）**，纯 NumPy / SciPy，离线、可复现。

**gandharva**（乾闼婆，佛教中的天界歌者）把两件事拼在一起：

- **语音**那一侧的*发音建模*——元音、共振峰、源-滤波器声道模型；
- **音乐**那一侧的*音高 / 旋律控制*——音符、基频轨迹、滑音、颤音、起音过冲。

于是同一套代码既能把一份乐谱「唱」出来（SVS），也能把一段真实歌声变调 / 变声 / 修音（SVC）。
没有神经网络、没有预训练权重、不联网——一切都是可读的数字信号处理，方便学习与复现。

## 为什么是「集大成」

歌声既是语言又是音乐。传统语音合成只关心「说清楚」，忽略音高旋律；
而音乐里的音高工具又不管发音。gandharva 把二者接到同一条流水线上：

```
乐谱 ──► 基频轨迹（音高/旋律）─┐
音节 ──► 元音 ──► 共振峰包络（发音/音色）─┼─► 源-滤波器声码器 ─► 歌声波形
                          基频驱动激励 ─┘
```

## 安装

```bash
pip install gandharva          # 需要 Python 3.10+
# 或从源码
pip install -e ".[dev]"
```

依赖只有 `numpy` 与 `scipy`。

## 快速上手

### 歌声合成（SVS）

```python
from gandharva import Note, Score, synthesize_score, write_wav

score = Score([
    Note(start=0.0, duration=0.5, pitch=60, lyric="do"),
    Note(start=0.5, duration=0.5, pitch=62, lyric="re"),
    Note(start=1.0, duration=0.5, pitch=64, lyric="mi"),
    Note(start=1.5, duration=1.0, pitch=67, lyric="so"),
])
voice = synthesize_score(score)
write_wav("scale.wav", voice.samples, voice.sample_rate)
```

### 歌声转换（SVC）

```python
from gandharva import read_wav, convert_pitch, write_wav

signal, sr = read_wav("input.wav")
higher = convert_pitch(signal, semitones=+3)   # 升三个半音，保持音色与时长
write_wav("higher.wav", higher.samples, higher.sample_rate)
```

### 命令行

```bash
gandharva synth score.json -o out.wav      # 合成
gandharva convert in.wav -s 5 -o up.wav    # 变调 +5 半音
gandharva key in.wav -t 60 -o tuned.wav    # 修到 C 大调（autotune）
gandharva f0 in.wav                        # 打印基频统计
```

## 功能一览

- **音高单位换算**：MIDI ↔ Hz ↔ 音分（cents）
- **YIN 基频估计**：差分函数 + CMNDF + 抛物线插值
- **旋律表情**：滑音（portamento）、颤音（vibrato）、起音过冲（overshoot）
- **发音建模**：音节切分、元音共振峰表、共振峰谐振器包络
- **LPC 分析 / 合成**：自相关 + Levinson-Durbin，含数值正则化保证稳定
- **源-滤波器声码器**：脉冲串 / 噪声混合激励、谱域整形、重叠相加
- **SVC 变声**：变调、共振峰独立平移、调式吸附（autotune）

## 文档

- [架构总览](docs/architecture.md)
- [使用指南](docs/usage.md)
- [设计笔记：音高模型](docs/design-notes.md)
- [API 参考](docs/api-reference.md)

## 开发

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy
pytest --cov=gandharva
```

## 许可证

[MIT](LICENSE) © Rong Shengxue
