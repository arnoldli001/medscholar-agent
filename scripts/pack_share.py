"""打包分享包。

生成 ``dist/MedScholar-Agent-v<版本>-win64.zip``，解压即可运行：

* 包含自包含的 Python 便携运行时（**朋友无需安装 Python**）
* 包含全部依赖、程序源码、前端、配置模板与启动器
* **不包含**你的数据库、导出文件、API Key、日志与缓存

    .python\\python.exe scripts\\pack_share.py
    .python\\python.exe scripts\\pack_share.py --no-runtime   # 不含运行时（体积小，但对方需自备 Python）
    .python\\python.exe scripts\\pack_share.py --with-config  # 把你当前的 config.yaml 一并打包
                                                              # （⚠️ 内含 API Key，仅打包给信任的人）
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

from medscholar import __version__  # noqa: E402

DIST = ROOT / "dist"

#: 需要打进分享包的顶层条目
INCLUDE: tuple[str, ...] = (
    "medscholar",
    "scripts",
    "docs",
    "run.bat",
    "run-doctor.bat",
    "run-cli.bat",
    "config.example.yaml",
    ".env.example",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "LICENSE",
)

#: 一律排除的目录名与文件名
EXCLUDE_DIRS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "dist", "build", "data", ".cache", ".git",
    ".idea", ".vscode", "node_modules", ".ipynb_checkpoints",
}
EXCLUDE_FILES = {
    "config.yaml", ".env", "medscholar.db", "medscholar.db-wal", "medscholar.db-shm",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd.log", ".log", ".db", ".db-wal", ".db-shm"}

#: 仅供开发/测试使用的依赖，不应进入分享包（否则体积白白翻倍）。
#: 这些包在运行期完全用不到：pytest/ruff 是测试与 lint 工具，
#: pygments/iniconfig/pluggy 只是它们的依赖。
DEV_ONLY_PACKAGES = frozenset(
    {
        "pytest", "_pytest", "pytest_asyncio", "ruff", "pygments",
        "iniconfig", "pluggy", "coverage", "mypy", "black",
    }
)

#: 开发工具的可执行文件（ruff.exe 单个就有 25 MB，绝不能进分享包）
DEV_ONLY_SCRIPTS = frozenset(
    {"ruff", "pytest", "py.test", "pygmentize", "coverage", "mypy", "black", "isort", "flake8"}
)


def should_skip(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    if path.name in EXCLUDE_FILES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    # 开发工具的可执行文件（位于 Scripts/ 下）
    if path.parent.name.lower() == "scripts" and path.stem.lower() in DEV_ONLY_SCRIPTS:
        return True
    # 剔除纯开发依赖（含 dist-info 与 __pycache__）
    parts = [p.lower() for p in path.parts]
    for marker in ("site-packages", "lib"):
        if marker in parts:
            index = len(parts) - 1 - parts[::-1].index(marker)
            for segment in parts[index + 1 :]:
                base = segment.split("-")[0].removesuffix(".dist")
                if base in DEV_ONLY_PACKAGES:
                    return True
            break
    return False


def iter_files(base: Path):
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if should_skip(path.relative_to(base)):
            continue
        yield path


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024.0
    return f"{size:.1f} GB"


def build_readme_first(root: str) -> str:
    return f"""# MedScholar Agent v{__version__} — 解压即用

## 三步开始

1. **解压**（不要直接在压缩包里双击运行，请先解压到任意目录，**路径中尽量不要有中文和空格**）
2. **安装 Ollama**（可选，但不装就无法生成综述）：
   - 下载 https://ollama.com/download
   - 安装后打开命令行执行：
     ```
     ollama pull qwen3:8b
     ollama pull nomic-embed-text
     ```
3. **双击 `run.bat`** —— 会自动启动服务并打开浏览器（地址 http://127.0.0.1:8760）

## 出问题了？

双击 **`run-doctor.bat`**。它会逐项检查并告诉你**具体**哪里有问题、该执行什么命令。

## 这是什么

面向医学研究者的本地化学术智能体：输入研究课题，自动检索 PubMed / Europe PMC / OpenAlex /
Crossref 等免费学术库，建立本地可语义检索的知识库，并生成带引用的综述草稿。

- 文献数据全部来自官方免费开放 API
- 数据只存在你自己电脑上的 `data\\medscholar.db` 一个文件里
- 默认使用本地 Ollama 模型，不产生任何 API 费用
- 非开放获取文献只保存元数据与出版商链接，不下载全文

## 更多说明

完整文档见 `README.md`；HTTP 接口契约见 `docs/API.md`。

## 想调优？

用记事本打开 `config.yaml`：

- 填 API Key 可以把检索速率提升 3 倍以上（每个 Key 都是免费申请的，见文件内注释）
- 电脑配置一般可以换更小的模型：`llm.model: llama3.2:3b`
- 中文文献多，建议换多语言嵌入模型：`embedding.model: bge-m3` 且 `embedding.dim: 1024`

## 命令行

不用网页也可以用命令行：

```
.python\\python.exe -X utf8 -m medscholar --help
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="打包 MedScholar 分享包")
    parser.add_argument("--no-runtime", action="store_true",
                        help="不打包 .python 便携运行时（对方需自备 Python）")
    parser.add_argument("--with-config", action="store_true",
                        help="把你当前的 config.yaml 也打进去（注意其中可能含 API Key）")
    parser.add_argument("--output", default=None, help="输出 zip 路径")
    args = parser.parse_args()

    DIST.mkdir(parents=True, exist_ok=True)
    out = (
        Path(args.output)
        if args.output
        else DIST / f"MedScholar-Agent-v{__version__}-win64.zip"
    )
    if out.exists():
        out.unlink()

    # 先检查便携运行时
    runtime = ROOT / ".python"
    runtime_ok = (runtime / "python.exe").exists()
    if not args.no_runtime and not runtime_ok:
        print("⚠️  未找到 .python\\python.exe（便携运行时）。")
        print("    请先执行：python scripts\\setup_portable_python.py")
        print("    或使用 --no-runtime 打包一个需要自备 Python 的版本。\n")

    include_runtime = runtime_ok and not args.no_runtime

    print("=" * 74)
    print(f"打包 MedScholar Agent v{__version__}")
    print("=" * 74)
    print(f"  输出    : {out}")
    print(f"  运行时  : {'包含 .python（对方无需装 Python）' if include_runtime else '不包含（对方需自备 Python）'}")
    print(f"  配置    : {'包含你当前的 config.yaml' if args.with_config else '仅含模板 config.example.yaml'}")
    print()

    count = 0
    total = 0
    errors: list[str] = []

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        prefix = f"MedScholar-Agent-v{__version__}"

        # 1) 使用说明（放最外层，解压就能看到）
        zf.writestr(f"{prefix}/【先读我】快速开始.txt", build_readme_first(prefix))
        count += 1

        # 2) 便携运行时
        if include_runtime:
            print("  [1/3] 打包便携运行时…", flush=True)
            n = 0
            for path in iter_files(runtime):
                arc = f"{prefix}/.python/{path.relative_to(runtime).as_posix()}"
                try:
                    zf.write(path, arc)
                except OSError as exc:
                    errors.append(f"{arc}: {exc}")
                    continue
                total += path.stat().st_size
                n += 1
                count += 1
            print(f"        {n} 个文件")

        # 3) 源码与资源
        print("  [2/3] 打包程序文件…", flush=True)
        for entry in INCLUDE:
            source = ROOT / entry
            if not source.exists():
                continue
            if source.is_file():
                zf.write(source, f"{prefix}/{entry}")
                total += source.stat().st_size
                count += 1
                continue
            for path in iter_files(source):
                arc = f"{prefix}/{path.relative_to(ROOT).as_posix()}"
                try:
                    zf.write(path, arc)
                except OSError as exc:
                    errors.append(f"{arc}: {exc}")
                    continue
                total += path.stat().st_size
                count += 1

        # 4) 可选：用户配置
        if args.with_config and (ROOT / "config.yaml").exists():
            zf.write(ROOT / "config.yaml", f"{prefix}/config.yaml")
            count += 1

        # 5) 空的 data 目录占位
        zf.writestr(
            f"{prefix}/data/说明.txt",
            "本目录存放你的知识库。medscholar.db 就是全部文献数据，"
            "备份/迁移只需拷贝这一个文件。\n",
        )
        count += 1

    print("  [3/3] 完成")
    print()
    size = out.stat().st_size
    digest = hashlib.sha256(out.read_bytes()).hexdigest()

    print("=" * 74)
    print(f"  文件数 : {count}")
    print(f"  原始   : {human(total)}")
    print(f"  压缩后 : {human(size)}")
    print(f"  SHA256 : {digest}")
    print("=" * 74)
    print()
    print("分发建议：")
    print("  · 通过网盘/微信把 zip 发给朋友，让他**先解压再运行**，")
    print("    路径尽量不要含中文或空格（部分 Python 依赖对长路径敏感）")
    print("  · 朋友机器上唯一需要额外装的是 Ollama（不装也能检索和建库）")
    if args.with_config:
        print("  ⚠️  你勾选了 --with-config：这个包里含你的 API Key，只发给信任的人")
    if errors:
        print()
        print(f"⚠️  有 {len(errors)} 个文件写入失败（通常是文件被占用）：")
        for item in errors[:5]:
            print(f"     {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
