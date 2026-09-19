"""命令行入口必须真的能跑起来，而且**必须是文档里写的那个**。

**这条测试是为一次真实的乌龙写的。** 我在 `scripts/README.md` 里按想当然的模块名写了
`python -m medscholar.db.migrations.cli`，跑起来"零输出、退出码 0"，于是判断
"这个 CLI 漏了 `__main__` 入口"并去补了一个 —— 结果立刻被架构校验器抓住：
`cli` 一旦 import `migrate`（哪怕写在函数体里），就会形成
`db.migrate ↔ db.migrations.cli` 循环依赖。

真相是：入口**一直都在** `medscholar.db.migrate`，而且那个环是被人用**工厂注入**
刻意打断的（migrate 把 `MigrationRunner` 传给 cli，cli 不反向 import migrate）。
我"修"的是一个不存在的缺陷，同时打破了一个有意的设计。

所以这里的测试断言两件事：
1. 文档里写的命令**真的能跑**（用真实子进程，因为 `-m` 的可用性只有在子进程里才成立）；
2. 入口必须是 `medscholar.db.migrate` —— 顺带把"cli 不是入口"这件事固定下来，
   免得以后有人又去给它加 `__main__` 把环接回去。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: 文档里承诺的命令行入口（改这里之前先看 scripts/README.md 与 check_arch.py）
MIGRATE_ENTRY = "medscholar.db.migrate"


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
    def test_missing_database_fails_loudly(self, tmp_path: Path):
        """没有数据库时：必须**失败**（退出码 1 + 人话），而不是安静地退出 0。"""
        proc = _run(["-X", "utf8", "-m", MIGRATE_ENTRY], home=tmp_path / "empty")
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

        proc = _run(["-X", "utf8", "-m", MIGRATE_ENTRY], home=home)
        out = _decode(proc.stdout)
        assert proc.returncode == 0, _decode(proc.stderr)
        assert "当前版本" in out, f"状态输出缺少版本信息：{out!r}"
        assert "已应用" in out and "待应用" in out

    def test_plan_is_read_only(self, tmp_path: Path):
        home = tmp_path / "db2"
        home.mkdir(parents=True, exist_ok=True)
        _run(["-X", "utf8", "-c", "from medscholar.db.connect import get_db; get_db()"], home=home)
        proc = _run(["-X", "utf8", "-m", MIGRATE_ENTRY, "--plan"], home=home)
        assert proc.returncode == 0, _decode(proc.stderr)
        assert "迁移" in _decode(proc.stdout)

    def test_cli_module_is_not_a_run_entry(self):
        """`migrations.cli` 刻意**不是** `-m` 入口。

        它需要 `MigrationRunner`，而 `migrate` 需要它 —— 若 cli 反过来 import migrate，
        就会出现 `db.migrate ↔ db.migrations.cli` 循环依赖（架构校验器会红）。
        这个环靠工厂注入打断，所以这里断言 cli 里没有 `__main__` 块：
        如果哪天有人"顺手"加上，这条测试会先失败，提醒他去看 cli 模块的 docstring。

        用 AST 而不是字符串查找：该模块的 docstring 里**就会提到** `if __name__ == "__main__"`
        （解释为什么故意不写），字符串匹配会把说明文字当成代码 ——
        第一版就是这么误报的。
        """
        import ast

        source = (ROOT / "medscholar" / "db" / "migrations" / "cli.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        guards = [
            node
            for node in tree.body
            if isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
        ]
        assert not guards, (
            "migrations/cli.py 不应注册为运行入口（见该模块 docstring 与 check_arch.py 的循环依赖检查）"
        )


class TestMainCli:
    @pytest.mark.parametrize("argv", [["--help"], ["--version"], ["doctor", "--help"]])
    def test_main_cli_help_paths(self, argv: list[str], tmp_path: Path):
        """主 CLI 的 --help / --version 必须可用（打包后朋友第一个会敲的就是它们）。"""
        proc = _run(["-X", "utf8", "-m", "medscholar.cli", *argv], home=tmp_path / "cli")
        assert proc.returncode == 0, _decode(proc.stderr)
        assert "Traceback" not in _decode(proc.stderr)

