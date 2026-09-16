"""MedScholar Agent 启动引导（run.bat / run-doctor.bat / run-cli.bat 的真正入口）。

为什么要有这一层
----------------

``.bat`` 文件里**不能放中文**。cmd.exe 用控制台 OEM 代码页（中文 Windows 是
936/GBK）解析批处理文件，UTF-8 的中文字节在被误当成 GBK 解码时，尾字节会
"吃掉" 后面的换行，把下一行命令截断执行，于是出现满屏的
``"dp0" 不是内部或外部命令`` 这类报错，而且 ``chcp 65001`` 也救不了
（cmd 的批处理解析器在 65001 下有已知缺陷）。

因此三个 .bat 只保留纯 ASCII 的引导逻辑（找 Python、转发参数、出错时 pause），
**所有中文提示都由本脚本用 Python 打印** —— Python 在 Windows 上通过
``WriteConsoleW`` 直接写 Unicode，与控制台代码页无关，任何 locale 都不会乱码。

本脚本只依赖标准库：依赖包缺失时它还要负责把依赖装上。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
CONFIG_EXAMPLE = ROOT / "config.example.yaml"
CONFIG_FILE = ROOT / "config.yaml"

#: (import 名, pip 包名) —— 用 find_spec 检查，不触发真正的 import
REQUIRED = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn[standard]"),
    ("httpx", "httpx"),
    ("pydantic", "pydantic"),
    ("yaml", "pyyaml"),
    ("sqlite_vec", "sqlite-vec"),
)

#: 国内镜像（按顺序回退）
PIP_MIRRORS = (
    None,
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
)

BAR = "=" * 74


# --------------------------------------------------------------------- 输出
def setup_console() -> None:
    """把标准输出切到 UTF-8。

    在真实 Windows 控制台上，``sys.stdout`` 底层是 ``_WindowsConsoleIO``，
    它**要求** UTF-8 输入并负责转成 UTF-16 交给 ``WriteConsoleW``，
    所以这里设成 utf-8 是正确的（且与控制台代码页无关）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def say(message: str = "") -> None:
    print(message, flush=True)


def title(text: str) -> None:
    say(BAR)
    say(f"  {text}")
    say(BAR)


def pause(message: str = "按任意键继续 . . .") -> None:
    """仅在交互式控制台里等待按键，避免自动化环境被卡住。"""
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return
    except (AttributeError, ValueError):
        return
    try:
        input(f"\n{message}")
    except (EOFError, KeyboardInterrupt):
        pass


# ----------------------------------------------------------------- 依赖检查
def missing_packages() -> list[str]:
    missing: list[str] = []
    for module, package in REQUIRED:
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(package)
        except (ImportError, ValueError):
            missing.append(package)
    return missing


def ensure_pip() -> bool:
    if subprocess.call(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0:
        return True
    say("  · 未检测到 pip，正在通过 ensurepip 安装…")
    subprocess.call(
        [sys.executable, "-m", "ensurepip", "--upgrade"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return subprocess.call(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0


def pip_install(missing: list[str]) -> bool:
    """安装缺失依赖；直连失败时自动切换国内镜像重试。"""
    if not ensure_pip():
        say("  ✗ 无法使用 pip（既没有 pip 也无法通过 ensurepip 安装）。")
        say("    请手动安装 Python 3.10+（勾选 Add to PATH）后重试，")
        say("    或参考 README.md 的「环境要求」一节。")
        return False

    if REQUIREMENTS.exists():
        base = ["-r", str(REQUIREMENTS)]
    else:  # pragma: no cover - 分享包里一定有 requirements.txt
        base = list(missing)

    for index, mirror in enumerate(PIP_MIRRORS):
        label = "官方源" if mirror is None else f"镜像 {mirror}"
        say(f"  · 正在安装依赖（{label}）…")
        command = [sys.executable, "-m", "pip", "install", *base]
        if mirror:
            command += ["-i", mirror, "--trusted-host", mirror.split("/")[2]]
        if subprocess.call(command) == 0:
            return True
        if index + 1 < len(PIP_MIRRORS):
            say("  · 该源安装失败，尝试下一个源…")

    say("  ✗ 依赖安装失败（所有源都试过了）。")
    say("")
    say("    如果你在国内网络环境，可以手动执行：")
    say(f'      "{sys.executable}" -m pip install -r requirements.txt \\')
    say("          -i https://pypi.tuna.tsinghua.edu.cn/simple")
    return False


def check_environment(*, auto_install: bool = True) -> bool:
    """确认运行环境可用；缺依赖时按需安装。"""
    if sys.version_info < (3, 10):
        say(f"  ✗ Python 版本过低：{sys.version.split()[0]}，需要 3.10 及以上。")
        say("    请安装新版 Python，或使用随包分发的 .python\\python.exe。")
        return False

    missing = missing_packages()
    if not missing:
        return True

    say(f"  ! 缺少依赖：{', '.join(missing)}")
    if not auto_install:
        return False
    if not pip_install(missing):
        return False

    still = missing_packages()
    if still:
        say(f"  ✗ 安装后仍缺少：{', '.join(still)}")
        return False
    say("  ✓ 依赖安装完成")
    return True


# ----------------------------------------------------------------- 配置准备
def ensure_config() -> None:
    """首次运行时从模板生成 config.yaml。"""
    if CONFIG_FILE.exists() or not CONFIG_EXAMPLE.exists():
        return
    try:
        shutil.copyfile(CONFIG_EXAMPLE, CONFIG_FILE)
    except OSError as exc:  # pragma: no cover
        say(f"  ! 配置文件生成失败（不影响启动，将使用默认值）：{exc}")
        return
    say("  ✓ 已生成配置文件 config.yaml")
    say("    想填写 API Key 或切换模型，用记事本打开它即可（文件内有详细注释）。")


# ------------------------------------------------------------------- 执行器
def run_medscholar(args: list[str]) -> int:
    """用当前解释器以模块方式运行 medscholar，保证使用同一套依赖。"""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    command = [sys.executable, "-X", "utf8", "-m", "medscholar", *args]
    try:
        return subprocess.call(command, cwd=str(ROOT), env=env)
    except KeyboardInterrupt:
        return 130


def explain_failure(code: int) -> None:
    say("")
    say(BAR)
    say(f"  服务异常退出（退出码 {code}）")
    say(BAR)
    say("  排查建议：")
    say("    1. 双击 run-doctor.bat 做环境自检 —— 它会逐项说明哪里有问题")
    say("    2. 常见原因之一是端口被占用，可换一个端口启动：")
    say("         run.bat --port 8888")
    say("    3. 若提示缺少依赖，可手动执行：")
    say(f'         "{sys.executable}" -m pip install -r requirements.txt')
    say("            -i https://pypi.tuna.tsinghua.edu.cn/simple")
    say("")


# ------------------------------------------------------------------ CLI 菜单
MENU = (
    ("1", "联网检索文献并入库", lambda: _ask_run(["search"], "请输入检索词：", ["--limit", "20"])),
    ("2", "本地知识库混合检索", lambda: _ask_run(["kb"], "请输入检索词：", [])),
    ("3", "跑完整 Agent 工作流（生成综述）", lambda: _ask_run(["run"], "请输入研究课题：", [])),
    ("4", "单篇文献速读", lambda: _ask_run(["summarize"], "请输入 paper_id：", [])),
    ("5", "生成参考文献", lambda: _ask_run(["cite"], "请输入 paper_id（空格分隔）：", [])),
    ("6", "导出文献（BibTeX）", lambda: run_medscholar(["export", "--format", "bibtex"])),
    ("7", "查看知识库统计", lambda: run_medscholar(["stats"])),
    ("8", "数据库维护（统计 / 优化索引）", lambda: run_medscholar(["db", "--stats", "--optimize"])),
    ("9", "启动 MCP Server", lambda: run_medscholar(["mcp"])),
)


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _ask_run(base: list[str], prompt: str, extra: list[str]) -> int:
    answer = _ask(prompt)
    if not answer:
        say("  （已取消）")
        return 0
    # summarize / cite 接受多个位置参数（paper_id 列表），其余命令整体作为一个参数
    multi = base[0] in {"summarize", "cite"}
    args = [*base, *(answer.split() if multi else [answer]), *extra]
    return run_medscholar(args)


def cli_menu() -> int:
    title("MedScholar Agent — 命令行工具")
    say("")
    for key, label, _ in MENU:
        say(f"   {key}. {label}")
    say("   0. 查看完整帮助")
    say("   q. 退出")
    say("")

    choice = _ask("请输入序号：").lower()
    if choice in {"q", ""}:
        return 0
    if choice == "0":
        return run_medscholar(["--help"])
    for key, _label, action in MENU:
        if key == choice:
            return int(action() or 0)
    say("  （无效的选择）")
    return 1


# ------------------------------------------------------------------ 主入口
def main(argv: list[str] | None = None) -> int:
    setup_console()

    parser = argparse.ArgumentParser(
        prog="medscholar-bootstrap",
        description="MedScholar Agent 启动引导",
        add_help=False,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--serve", action="store_true", help="启动 Web 工作台（默认）")
    mode.add_argument("--doctor", action="store_true", help="环境自检")
    mode.add_argument("--cli", action="store_true", help="命令行菜单")
    parser.add_argument("--no-install", action="store_true", help="缺少依赖时不自动安装")
    parser.add_argument("--no-pause", action="store_true", help="出错时不等待按键")
    known, extra = parser.parse_known_args(argv)

    def _pause(message: str = "按任意键继续 . . .") -> None:
        if not known.no_pause:
            pause(message)

    # 环境自检模式：把 banner 交给 medscholar doctor 打印，这里只做依赖兜底
    if known.doctor:
        if not check_environment(auto_install=not known.no_install):
            _pause()
            return 1
        ensure_config()
        code = run_medscholar(["doctor", *extra])
        return 0 if code == 0 else code

    if known.cli:
        if not check_environment(auto_install=not known.no_install):
            _pause()
            return 1
        ensure_config()
        code = 0
        while True:
            try:
                code = cli_menu()
            except KeyboardInterrupt:
                say("\n（已退出）")
                return 0
            if _ask("\n按回车返回菜单，输入 q 退出：").lower() == "q":
                return code

    # 默认：启动 Web 工作台
    title("MedScholar Agent — 正在启动")
    say("")
    if not check_environment(auto_install=not known.no_install):
        _pause()
        return 1
    ensure_config()

    code = run_medscholar(["serve", *extra])
    if code != 0:
        explain_failure(code)
        _pause()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
