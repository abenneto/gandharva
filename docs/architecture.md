# 架构总览

gandharva 按「信号处理的层次」组织，从底层 DSP 一路搭到 SVS / SVC 两条应用链。
每一层只依赖它下面的层，方便单独测试与替换。

```
                    ┌───────────────────────────────┐
   应用层           │   svs.py (SVS)   svc.py (SVC)  │
                    └───────────────┬───────────────┘
                                    │
      ┌──────────────┬─────────────┼──────────────┬───────────────┐
   领域层        pitch/         voice/         vocoder/        core.py
             （音高/旋律）    （发音/音色）   （源-滤波器）    （数据容器）
      └──────────────┴─────────────┼──────────────┴───────────────┘
                                    │
   基础层                    dsp/  ·  convert_units.py  ·  audio.py
                     （分帧/窗/STFT/LPC/包络 · 单位换算 · WAV I/O）
```

## 分层说明

### 基础层

- **`dsp/windows.py`** — 分帧（stride trick 零拷贝）、窗函数（带缓存）、重叠相加。
- **`dsp/stft.py`** — 前向 / 逆 STFT，建立在分帧 + OLA 之上。
- **`dsp/lpc.py`** — 自相关、Levinson-Durbin 递推、LPC 谱包络。含自相关正则化
  （白噪声地板 + 滞后窗），保证强周期信号也能得到稳定的全极点滤波器。
- **`dsp/envelope.py`** — 倒谱谱包络平滑，作为 LPC 之外的另一种包络来源。
- **`convert_units.py`** — MIDI / Hz / 音分互转，标量与数组通吃。
- **`audio.py`** — 只用标准库 `wave` 的 16-bit WAV 读写、峰值归一化、重采样。

### 领域层

- **`core.py`** — `Note` / `Score`（乐谱输入）、`F0Contour`（逐帧基频）、`Voice`（渲染输出）。
- **`pitch/`** — `estimate.py`（YIN 基频估计）与 `contour.py`（乐谱 → 基频轨迹 + 旋律表情）。
- **`voice/`** — `phonemes.py`（音节 → 元音）、`formants.py`（元音共振峰包络）、
  `timing.py`（音符 → 帧的时间映射）。
- **`vocoder/`** — `excitation.py`（激励）、`analysis.py`（分析）、`synthesis.py`（合成）。

### 应用层

- **`svs.py`** — 把领域层拼成「乐谱 → 歌声」。
- **`svc.py`** — 把领域层拼成「歌声 → 变调 / 变声 / 修音」。
- **`cli.py`** — 三条链的命令行入口。

## 数据流

### SVS（合成）

```
Score
  │ notes_to_f0            → 阶梯基频
  │ apply_portamento       → 加滑音
  │ apply_attack_overshoot → 加起音过冲
  │ apply_vibrato          → 加颤音
  ▼
F0Contour ──┐
            │ mixed_excitation      → 基频驱动的脉冲/噪声激励
frame_vowels│ formant_envelope      → 逐帧元音共振峰包络
            ▼
      synthesize_from_envelopes（谱域整形 + OLA）
            ▼
          Voice
```

### SVC（转换）

```
波形
  │ analyze              → 逐帧 LPC 包络 + 增益 + 基频
  │ f0 × 变调系数         → 只动音高（激励）
  │ LPC 包络保持不变       → 音色不变（或按 formant_shift 独立平移）
  │ mixed_excitation + synthesize
  ▼
Voice
```

## 设计取舍

- **纯 NumPy/SciPy、离线优先**：便于在 CI 里跑、便于教学复现，不追求实时或 SOTA 音质。
- **源-滤波器模型**：激励（音高）与谱包络（音色）解耦，这正是 SVC 能「变调不变声」的基础。
- **数值稳健优先于极致精度**：LPC 正则化、清浊判定阈值都偏保守，宁可稍钝也不要爆音 / 误判。
