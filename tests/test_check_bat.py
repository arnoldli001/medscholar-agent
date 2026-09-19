"""``scripts/check_bat.py``：`.bat` 约定守卫自身的测试。

这个脚本是**守卫**，所以它自己坏了最危险：它会在"校验通过"的打印上崩掉，
看起来像 `.bat` 有问题，实际是守卫自己编码失败。

真实事故（本文件最后一组测试就是为它写的回归测试）：

    在中文 Windows（ANSI 代码页 cp936）下把输出重定向/管道给别人看时，
    ``print("  ✓ ...")`` 直接抛 ``UnicodeEncodeError: 'gbk' codec can't encode
    character '\\u2713'``。真实控制台不会触发，因为 Python 对控制台走
    ``WriteConsoleW`` 绕开了编码 —— 所以这个 bug 只在"重定向"这条路径上出现，
    而 CI 跑在 UTF-8 的 ubuntu 上，永远不会发现它。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_bat  # noqa: E402  （需要先插入 scripts 目录）

SCRIPT = ROOT / "scripts" / "check_bat.py"


# --------------------------------------------------------------------- 单元行为


def test_ascii_crlf_passes(tmp_path: Path) -> None:
    bat = tmp_path / "run.bat"
    bat.write_bytes(b"@echo off\r\necho hi\r\n")
    assert check_bat.check_one(bat, tmp_path) == []


def test_lf_only_is_rejected(tmp_path: Path) -> None:
    """LF 会让 goto 与标签失效，报出完全指错方向的 `"dp0" 不是内部或外部命令`。"""
    bat = tmp_path / "run.bat"
    bat.write_bytes(b"@echo off\ngoto :eof\n")
    problems = check_bat.check_one(bat, tmp_path)
    assert any("LF-only" in p for p in problems)


def test_non_ascii_is_rejected(tmp_path: Path) -> None:
    """UTF-8 中文在 936 代码页下会乱码，用户看到的是一堆问号。"""
    bat = tmp_path / "run.bat"
    bat.write_bytes("@echo off\r\necho 完成\r\n".encode())
    problems = check_bat.check_one(bat, tmp_path)
    assert any("非 ASCII" in p for p in problems)
    # 报出行号，方便直接定位
    assert any("第 2 行" in p for p in problems)


def test_bom_is_rejected(tmp_path: Path) -> None:
    bat = tmp_path / "run.bat"
    bat.write_bytes(b"\xef\xbb\xbf@echo off\r\n")
    problems = check_bat.check_one(bat, tmp_path)
    assert any("BOM" in p for p in problems)


def test_lone_cr_is_rejected(tmp_path: Path) -> None:
    bat = tmp_path / "run.bat"
    bat.write_bytes(b"@echo off\recho hi\r\n")
    problems = check_bat.check_one(bat, tmp_path)
    assert any("孤立的 CR" in p for p in problems)


def test_skip_dirs_are_not_scanned(tmp_path: Path) -> None:
    """便携运行时与构建产物里的 .bat 不该被当成项目约定的一部分。"""
    for skip in (".python", "dist", ".venv"):
        d = tmp_path / skip
        d.mkdir(parents=True)
        (d / "bad.bat").write_bytes(b"@echo off\necho \xe4\xb8\xad\n")
    (tmp_path / "run.bat").write_bytes(b"@echo off\r\n")
    found = [p.relative_to(tmp_path).as_posix() for p in check_bat.find_bat_files(tmp_path)]
    assert found == ["run.bat"]


def test_main_reports_failure_with_exit_code_1(tmp_path: Path, capsys) -> None:
    (tmp_path / "bad.bat").write_bytes(b"@echo off\ngoto :eof\n")
    assert check_bat.main(["--root", str(tmp_path)]) == 1
    assert "未通过" in capsys.readouterr().out


def test_main_quiet_hides_success_detail(tmp_path: Path, capsys) -> None:
    (tmp_path / "run.bat").write_bytes(b"@echo off\r\n")
    assert check_bat.main(["--root", str(tmp_path), "--quiet"]) == 0
    assert capsys.readouterr().out == ""


# ------------------------------------------------- 真实子进程 + cp936 管道（回归）


def _run_script(env_extra: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    import os

    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(cwd)],
        capture_output=True,
        env=env,
        cwd=str(ROOT),
        timeout=120,
    )


def test_survives_cp936_pipe(tmp_path: Path) -> None:
    """回归：cp936 管道下打印 ✓ 曾直接崩掉守卫脚本本身。

    用 ``PYTHONIOENCODING=gbk`` 强制非 UTF-8 管道编码 —— 这与真实控制台不同
    （真实控制台走 WriteConsoleW 不会触发），正是当初漏掉的那条路径。
    """
    (tmp_path / "run.bat").write_bytes(b"@echo off\r\n")
    proc = _run_script({"PYTHONIOENCODING": "gbk"}, tmp_path)
    assert b"UnicodeEncodeError" not in proc.stderr, proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    # 输出是 UTF-8，且 ✓ 完整保留（不是被 replace 成 ?）
    assert "\u2713" in proc.stdout.decode("utf-8")


@pytest.mark.parametrize("enc", ["gbk", "cp1252", "ascii"])
def test_survives_narrow_pipe_encodings(tmp_path: Path, enc: str) -> None:
    """更窄的编码（连中文都没有）也只能降级，不能崩。"""
    (tmp_path / "bad.bat").write_bytes(b"@echo off\ngoto :eof\n")
    proc = _run_script({"PYTHONIOENCODING": enc}, tmp_path)
    assert b"UnicodeEncodeError" not in proc.stderr, proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 1  # 校验失败，但不是崩溃
