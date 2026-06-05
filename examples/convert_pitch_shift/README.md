# 示例：保持音色的变调（SVC）

先合成一小段歌声当输入，再把它升四度（+5 半音），时长和音色都不变。

```bash
python examples/convert_pitch_shift/run.py
```

生成 `source.wav`（原始）与 `shifted_up.wav`（变调后）。对比听感：音高变了，音色没变。
