# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)，
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

## [0.1.0] - 2026-07-08

首个公开版本。语音 × 音乐交叉的歌声合成与转换框架，纯 NumPy / SciPy。

### 新增

- **核心容器**：`Note` / `Score` / `F0Contour` / `Voice`，含乐谱校验与 `Score.from_dict`。
- **单位换算**：MIDI / Hz / 音分互转，`transpose_hz`。
- **DSP 基础**：分帧（stride trick）、窗函数（带缓存）、重叠相加、STFT/iSTFT、
  LPC（自相关 + Levinson-Durbin + 正则化）、倒谱谱包络。
- **基频估计**：YIN（差分函数 + CMNDF + 抛物线插值）。
- **旋律表情**：滑音、颤音、起音过冲。
- **发音建模**：音节切分、元音共振峰表与谐振器包络。
- **声码器**：脉冲 / 噪声混合激励、声门源谱倾斜、LPC 全极点合成、谱域整形合成。
- **SVS**：`synthesize_score`，乐谱 → 歌声。
- **SVC**：`convert_pitch`（变调）、`convert_voice`（独立平移共振峰）、
  `convert_to_key`（调式吸附 / autotune）。
- **命令行**：`synth` / `convert` / `key` / `f0` 四个子命令。
- **音频 I/O**：仅用标准库 `wave` 的 16-bit WAV 读写、峰值归一化、重采样。

[Unreleased]: https://github.com/abenneto/gandharva/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/abenneto/gandharva/releases/tag/v0.1.0
