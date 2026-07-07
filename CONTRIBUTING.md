# 贡献指南

欢迎参与 gandharva。这是一个偏教学 / 研究性质的项目，代码可读性优先于极致性能。

## 环境准备

```bash
git clone https://github.com/abenneto/gandharva.git
cd gandharva
pip install -e ".[dev]"
pre-commit install   # 可选，但推荐
```

需要 Python 3.10 及以上，依赖只有 numpy 与 scipy。

## 开发流程

1. 从 `main` 切出特性分支：`git checkout -b feat/my-change`
2. 写代码 + 对应测试
3. 本地跑通所有检查（见下）
4. 提交、推送、开 PR，填写 PR 模板

## 提交前必过

```bash
ruff check src tests        # 代码规范
ruff format --check src tests  # 格式
mypy                        # 类型（strict）
pytest --cov=gandharva      # 测试 + 覆盖率
```

CI 会在 Python 3.10–3.13 上重跑这些；本地全绿基本就能过 CI。

## 代码风格

- 行宽 100，用 ruff 统一格式。
- 公开函数写 docstring；DSP 公式里保留单字母变量（更贴近论文记号）。
- 类型注解齐全，`mypy --strict` 必须干净。numpy 相关的 `no-any-return`
  用「先赋值给带注解的局部变量再返回」的方式处理。
- 提交信息不强制某种格式，但请言之有物。

## 加新功能的一点建议

gandharva 分层清晰（见 [docs/architecture.md](docs/architecture.md)）：

- 通用信号处理放 `dsp/`；
- 音高相关放 `pitch/`，发音相关放 `voice/`，声码器放 `vocoder/`；
- 每个模块都应有对应的 `tests/` 测试文件。

涉及听感的改动，PR 里最好附上数值对比（谱峰、基频、往返误差之类），
纯主观「更好听」不容易评审。

## 报告问题

用 issue 模板，尽量给出最小复现（乐谱 / 音频 + 调用代码）。

## 行为准则

参与本项目即表示你同意遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
