# 使用指南

## 乐谱格式

SVS 的输入是 `Score`，由一串 `Note` 组成。既可以在代码里构造，也可以从 JSON 读入。

### 代码构造

```python
from gandharva import Note, Score

score = Score([
    Note(start=0.0, duration=0.5, pitch=60, lyric="la"),
    Note(start=0.5, duration=0.5, pitch=64, lyric="le"),
])
```

- `start` / `duration`：秒（float）。
- `pitch`：MIDI 音高，60 = 中央 C；可传 float 表达微分音。
- `lyric`：演唱的音节；空串表示休止符。

`Score` 会自动按起始时间排序，并检查音符是否重叠（重叠会抛 `ScoreError`）。

### JSON 格式

```json
{
  "notes": [
    {"start": 0.0, "duration": 0.5, "pitch": 60, "lyric": "do"},
    {"start": 0.5, "duration": 0.5, "pitch": 62, "lyric": "re"}
  ]
}
```

```python
import json
from gandharva import Score

with open("score.json", encoding="utf-8") as f:
    score = Score.from_dict(json.load(f))
```

## 合成（SVS）

```python
from gandharva import synthesize_score, SynthConfig, write_wav

# 默认参数
voice = synthesize_score(score)

# 自定义表情
cfg = SynthConfig(
    sample_rate=24000,
    vibrato_depth_cents=60.0,   # 更深的颤音
    vibrato_rate_hz=6.0,
    portamento_ms=120.0,        # 更慢的滑音
    overshoot_cents=30.0,       # 更明显的起音过冲
)
voice = synthesize_score(score, cfg)
write_wav("out.wav", voice.samples, voice.sample_rate)
```

返回的 `Voice` 带有 `samples`、`sample_rate`，以及生成它用的 `f0`（`F0Contour`）。

## 转换（SVC）

### 变调（保持音色）

```python
from gandharva import read_wav, convert_pitch, write_wav

signal, sr = read_wav("vocal.wav")
out = convert_pitch(signal, semitones=-2)   # 降两个半音
write_wav("down.wav", out.samples, out.sample_rate)
```

### 变声（独立平移共振峰）

```python
from gandharva.svc import convert_voice

# 音高不变（0 半音），但共振峰上移 → 听感更「亮 / 小」
out = convert_voice(signal, semitones=0, formant_shift=1.15)
```

### 修音 / 吸附到调式（autotune）

```python
from gandharva.svc import convert_to_key

# 把跑调的演唱吸附到 C 大调（主音 MIDI 60）
out = convert_to_key(signal, tonic_midi=60)
```

## 基频分析

```python
from gandharva import read_wav, estimate_f0

signal, sr = read_wav("vocal.wav")
contour = estimate_f0(signal, sample_rate=sr)

print(contour.values)   # 逐帧 Hz，0 表示清音
print(contour.times)    # 每帧对应的时刻（秒）
print(contour.voiced)   # 布尔掩码
```

## 命令行

```bash
gandharva synth score.json -o out.wav
gandharva convert in.wav --semitones 5 -o up.wav
gandharva key in.wav --tonic 62 -o dmajor.wav
gandharva f0 in.wav
```

## 常见问题

- **音量偏小 / 偏「电子」**：这是简化源-滤波器模型的固有音色，可用 `normalize_peak`
  后处理提升响度。
- **变调后有轻微「颗粒感」**：脉冲激励在极端变调时会暴露，属预期；小幅变调（±5 半音内）效果最好。
- **只支持单声道 16-bit WAV**：多声道会自动下混；其他格式请先转码。
