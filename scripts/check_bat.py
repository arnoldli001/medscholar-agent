"""守住 `.bat` 文件的两条硬约束：**纯 ASCII** 与 **CRLF 换行**。

为什么值得单独写一个检查脚本：这两条是踩过真实故障才定下来的，而且**违反了不会立刻报错**，
往往只在别人的机器上、或某个分支路径上才炸。

故障回顾：

* `.bat` 里写 UTF-8 中文，cmd.exe 用 936 代码页读会乱码 —— 用户看到一堆问号，
  完全不知道哪一步失败了。
* `.bat` 用 LF 换行时，`goto` 与标签会失效，报出 `"dp0" 不是内部或外部命令`
  这种完全指错方向的错误（真实排查花了很久）。

所以约定是：**`.bat` 只放纯 ASCII 与 CRLF，所有中文提示都交给 Python 侧输出**
（Python 用 `WriteConsoleW` 能正确写 Unicode 控制台）。

用法::

    .python\\python.exe scripts\\check_bat.py          # 检查，有问题退出码 1
    .python\\python.exe scripts\\check_bat.py --quiet  # 只在出错时输出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: 不检查的目录（第三方运行时、构建产物、缓存）
SKIP_DIRS = {".python", ".venv", "venv", "dist", "build", ".cache", ".git",
             "node_modules", "__pycache__", ".pytest_cache"}

BAT_SUFFIXES = {".bat", ".cmd"}

UTF8_BOM = b"\xef\xbb\xbf"


def find_bat_files(root: Path) -> list[Path]:
    """收集仓库内的 .bat/.cmd（跳过第三方运行时与构建产物）。"""
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in BAT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        found.append(path)
    return found


def check_one(path: Path, root: Path) -> list[str]:
    """检查单个文件，返回问题列表（空表示通过）。"""
    problems: list[str] = []
    data = path.read_bytes()

    if data.startswith(UTF8_BOM):
        problems.append("含 UTF-8 BOM —— cmd.exe 会把 BOM 当命令的一部分，删掉它")

    non_ascii = [b for b in data if b > 127]
    if non_ascii:
        # 找出第一处，方便直接定位
        offset = next(i for i, b in enumerate(data) if b > 127)
        line = data[:offset].count(b"\n") + 1
        problems.append(
            f"含 {len(non_ascii)} 个非 ASCII 字节（首次出现在第 {line} 行）"
            " —— UTF-8 中文在 936 代码页下会乱码；请改为纯 ASCII，"
            "中文提示交给 Python 侧（scripts/bootstrap.py）输出"
        )

    # 孤立 LF：LF 前面不是 CR
    bare_lf = 0
    first_bare_line = 0
    line_no = 1
    for i, byte in enumerate(data):
        if byte == 0x0A:
            if i == 0 or data[i - 1] != 0x0D:
                bare_lf += 1
                if not first_bare_line:
                    first_bare_line = line_no
            line_no += 1
        elif byte == 0x0D and (i + 1 >= len(data) or data[i + 1] != 0x0A):
            problems.append("存在孤立的 CR（没有跟随 LF）—— 请统一为 CRLF")
            break

    if bare_lf:
        problems.append(
            f"含 {bare_lf} 个 LF-only 换行（首次在第 {first_bare_line} 行）"
            " —— LF 会让 goto 与标签失效，报出 `\"dp0\" 不是内部或外部命令` 这类误导性错误；"
            "请改为 CRLF"
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check-bat", description="校验 .bat 文件为纯 ASCII + CRLF"
    )
    parser.add_argument("--quiet", action="store_true", help="只在出错时输出")
    parser.add_argument("--root", default=None, help="仓库根目录（默认脚本上一级）")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    files = find_bat_files(root)
    if not files:
        print(f"没有找到 .bat/.cmd 文件（根目录：{root}）")
        return 0

    failures: list[tuple[Path, list[str]]] = []
    for path in files:
        problems = check_one(path, root)
        if problems:
            failures.append((path, problems))

    if failures:
        print("=" * 72)
        print("  .bat 文件校验未通过")
        print("=" * 72)
        for path, problems in failures:
            print(f"\n  {path.relative_to(root)}")
            for problem in problems:
                print(f"    ✗ {problem}")
        print(
            "\n  修复方式：用支持 CRLF 的编辑器把文件另存为「ASCII 编码 + CRLF 换行」。\n"
            "  Python 侧可以用：\n"
            "      p = pathlib.Path('run.bat')\n"
            "      p.write_bytes(p.read_text().replace('\\r\\n','\\n')"
            ".replace('\\n','\\r\\n').encode('ascii'))\n"
        )
        return 1

    if not args.quiet:
        print(f".bat 校验通过：{len(files)} 个文件均为纯 ASCII + CRLF")
        for path in files:
            print(f"  ✓ {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
