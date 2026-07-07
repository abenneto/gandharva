"""gandharva 的异常层级。"""

from __future__ import annotations


class GandharvaError(Exception):
    """所有 gandharva 异常的基类。"""


class ScoreError(GandharvaError):
    """乐谱结构非法：时间重叠、音高越界、字段缺失等。"""


class AudioError(GandharvaError):
    """音频 I/O 或格式错误：不支持的位深、通道数不符等。"""


class ParameterError(GandharvaError):
    """传入参数非法：窗长非正、帧移大于窗长等。"""
