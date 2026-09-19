"""命令行入口必须真的能跑起来。

**这条测试是为一个"安静地什么都不做"的缺陷写的**：迁移 CLI 定义了 ``run()`` 与 ``main()``，
却漏了 `if __name__ == "__main__":`，于是文档里写的

    .python\\python.exe -X utf8 -m medscholar.db.migrations.cli

会**退出码 0、零输出**。这比报错危险得多 —— 用户以为迁移跑过了，
而实际上什么都没发生；等到某天发现库结构落后，已经很难回溯是哪一步没做。

用真实子进程断言（不是 import 进来调函数）：文档里的用法就是 `-m`，
只有子进程才能验证"这条命令真的可用"。同类问题在 `.bat` 守卫脚本上踩过一次
（打印 `✓` 触发 cp936 编码崩溃），所以这里的断言也包含"stderr 里没有 traceback"。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str], *, home: Path) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env["MEDSCHOLAR_HOME"] = str(home)
    env["MEDSCHOLAR_OFFLINE"] = "true"
    env["MEDSCHOLAR_EMBED_PROVIDER"] = "hashing"
    env.pop("PYTHONIOENCODING", None)
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        timeout=180,
    )


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


class TestMigrationCli:
    def test_module_entry_available(self, tmp_path: Path):
        """没有数据库时：必须**失败**（退出码 1 + 人话），而不是安静地退出 0。"""
        proc = _run(
            ["-X", "utf8", "-m", "medscholar.db.migrations.cli"],
            home=tmp_path / "empty",
        )
        assert proc.returncode == 1, f"应当报错退出；实际 rc={proc.returncode}"
        assert "数据库不存在" in _decode(proc.stderr)
        assert "Traceback" not in _decode(proc.stderr)

    def test_status_prints_version_after_bootstrap(self, tmp_path: Path):
        """建库之后，无参数调用必须打印状态（当前版本 / 已应用 / 待应用）。"""
        home = tmp_path / "db"
        home.mkdir(parents=True, exist_ok=True)
        boot = _run(
            ["-X", "utf8", "-c", "from medscholar.db.connect import get_db; get_db()"],
            home=home,
        )
        assert boot.returncode == 0, _decode(boot.stderr)

        proc = _run(["-X", "utf8", "-m", "medscholar.db.migrations.cli"], home=home)
        out = _decode(proc.stdout)
        assert proc.returncode == 0, _decode(proc.stderr)
        assert "当前版本" in out, f"状态输出缺少版本信息：{out!r}"
        assert "已应用" in out and "待应用" in out

    def test_plan_is_read_only_and_reports_nothing_pending(self, tmp_path: Path):
        home = tmp_path / "db2"
        home.mkdir(parents=True, exist_ok=True)
        _run(["-X", "utf8", "-c", "from medscholar.db.connect import get_db; get_db()"], home=home)
        proc = _run(
            ["-X", "utf8", "-m", "medscholar.db.migrations.cli", "--plan"], home=home
        )
        assert proc.returncode == 0, _decode(proc.stderr)
        assert "迁移" in _decode(proc.stdout)


class TestMainCli:
    @pytest.mark.parametrize("argv", [["--help"], ["--version"], ["doctor", "--help"]])
    def test_main_cli_help_paths(self, argv: list[str], tmp_path: Path):
        """主 CLI 的 --help / --version 必须可用（打包后朋友第一个会敲的就是它们）。"""
        proc = _run(["-X", "utf8", "-m", "medscholar.cli", *argv], home=tmp_path / "cli")
        assert proc.returncode == 0, _decode(proc.stderr)
        assert "Traceback" not in _decode(proc.stderr)
