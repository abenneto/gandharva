# API 参考

只列公开接口。完整签名以源码 docstring 为准。

## 顶层（`gandharva`）

```python
from gandharva import (
    Note, Score, Voice, F0Contour,   # 数据容器
    SynthConfig, synthesize_score,   # SVS
    convert_pitch, convert_voice, convert_to_key,  # SVC
    estimate_f0,                     # 分析
    read_wav, write_wav,             # 音频 I/O
)
```

## 数据容器（`gandharva.core`）

### `Note(start, duration, pitch, lyric="a")`

一个带歌词音节的音符。

- `end` → `float`：结束时刻。
- `is_rest` → `bool`：歌词为空则为休止符。

### `Score(notes)`

按时间排序的音符序列，构造时校验重叠。

- `duration` → `float`
- `Score.from_dict(data)` → `Score`：从 `{"notes": [...]}` 构造。
- 支持 `len()` 与迭代。

### `F0Contour(values, hop_length, sample_rate)`

逐帧基频轨迹。

- `times` → `ndarray`：每帧时刻（秒）。
- `voiced` → `ndarray[bool]`：基频 > 0 的掩码。

### `Voice(samples, sample_rate, f0=None)`

渲染后的歌声。

- `duration` → `float`

## 单位换算（`gandharva.convert_units`）

| 函数 | 说明 |
|------|------|
| `midi_to_hz(midi)` | MIDI 音高 → Hz |
| `hz_to_midi(hz)` | Hz → MIDI 音高 |
| `cents_between(f_from, f_to)` | 两频率的音分差 |
| `transpose_hz(hz, semitones)` | 频率升降若干半音 |

## 基频估计（`gandharva.pitch`）

### `estimate_f0(signal, *, sample_rate, frame_length, hop_length, fmin, fmax, threshold)` → `F0Contour`

逐帧 YIN 基频估计。

### 轨迹生成（`gandharva.pitch.contour`）

| 函数 | 说明 |
|------|------|
| `notes_to_f0(score, *, sample_rate, hop_length)` | 乐谱 → 阶梯基频 |
| `apply_portamento(contour, transition_ms)` | 加滑音 |
| `apply_vibrato(contour, *, rate_hz, depth_cents, onset_ms)` | 加颤音 |
| `apply_attack_overshoot(contour, *, overshoot_cents, attack_ms)` | 加起音过冲 |

## 发音建模（`gandharva.voice`）

| 函数 | 说明 |
|------|------|
| `split_syllable(syllable)` | 音节 → 音素列表 |
| `vowel_of(syllable)` | 取音节元音核心 |
| `formants_for(vowel)` | 元音三共振峰中心频率 |
| `formant_envelope(vowel, n_fft, sample_rate, *, shift=1.0)` | 共振峰谱包络 |

## 声码器（`gandharva.vocoder`）

| 函数 | 说明 |
|------|------|
| `pulse_train(f0, hop_length, sample_rate)` | 基频驱动脉冲串 |
| `mixed_excitation(f0, hop_length, sample_rate, *, aperiodicity, seed)` | 脉冲 / 噪声混合激励 |
| `analyze(signal, *, sample_rate, frame_length, hop_length, lpc_order)` | 逐帧 LPC + 基频分析 |
| `estimate_aperiodicity(signal, f0, ...)` | 逐帧非周期性 |
| `synthesize(frames, excitation)` | 全极点滤波合成 |
| `synthesize_from_envelopes(excitation, envelopes, frame_length, hop_length)` | 谱域整形合成 |

## DSP 基础（`gandharva.dsp`）

| 函数 | 说明 |
|------|------|
| `frame_signal(signal, frame_length, hop_length, *, pad=True)` | 分帧 |
| `get_window(name, length)` | 窗函数（带缓存） |
| `overlap_add(frames, hop_length, *, window=None)` | 重叠相加 |
| `stft / istft(...)` | 短时傅里叶变换及其逆 |
| `lpc(frame, order, *, sample_rate)` | LPC 分析 |
| `lpc_envelope(a, gain, n_fft)` | LPC 谱包络 |
| `cepstral_envelope(magnitude, n_coeffs)` | 倒谱谱包络 |

## 音频 I/O（`gandharva.audio`）

| 函数 | 说明 |
|------|------|
| `read_wav(path)` → `(samples, sample_rate)` | 读 16-bit WAV，归一化到 [-1,1] |
| `write_wav(path, samples, sample_rate)` | 写 16-bit 单声道 WAV |
| `normalize_peak(samples, target=0.99)` | 峰值归一化 |
| `resample(samples, orig_sr, target_sr)` | 多相重采样 |

## 异常（`gandharva.exceptions`）

- `GandharvaError` — 基类
- `ScoreError` — 乐谱非法
- `AudioError` — 音频 I/O 错误
- `ParameterError` — 参数非法
