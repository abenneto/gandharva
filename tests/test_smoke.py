"""最基本的冒烟测试：包能被导入、版本号存在。"""

import gandharva


def test_version_is_string() -> None:
    assert isinstance(gandharva.__version__, str)
    assert gandharva.__version__


def test_package_importable() -> None:
    assert hasattr(gandharva, "__all__")
