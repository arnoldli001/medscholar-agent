"""在工作区内构建**自包含 Python 运行时**。

目的有两个：

1. 开发期：DSH 沙箱只允许执行工作区内的程序，系统 Python 位于工作区外会被拒绝；
2. 交付期：把该目录随包分发，朋友解压后**无需安装 Python** 即可运行 MedScholar。

做法是把 python.org 的 embeddable 发行版解压到 ``.python/``，再复用 ``.venv`` 里
已经装好的依赖，最后用 ``python313._pth`` 把两者串起来。

用法::

    python scripts/setup_portable_python.py
"""

from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_DIR = ROOT / ".python"
CACHE = ROOT / ".cache"
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"

VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

#: 国内可直连的镜像优先，python.org 兜底
MIRRORS = (
    f"https://registry.npmmirror.com/-/binary/python/{VERSION}/python-{VERSION}-embed-amd64.zip",
    f"https://mirrors.huaweicloud.com/python/{VERSION}/python-{VERSION}-embed-amd64.zip",
    f"https://mirrors.aliyun.com/python-release/windows/python-{VERSION}-embed-amd64.zip",
    f"https://www.python.org/ftp/python/{VERSION}/python-{VERSION}-embed-amd64.zip",
)


def log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def download(url: str, dest: Path, timeout: float = 120.0) -> bool:
    log(f"尝试下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "MedScholar-Bootstrap/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            dest.parent.mkdir(parents=True, exist_ok=True)
            read = 0
            with dest.open("wb") as fh:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    read += len(chunk)
            if read < 1_000_000:
                log(f"  体积异常（{read} 字节），放弃该镜像")
                dest.unlink(missing_ok=True)
                return False
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log(f"  失败：{exc}")
        dest.unlink(missing_ok=True)
        return False
    log(f"  完成：{read / 1_048_576:.1f} MB" + (f" / {total / 1_048_576:.1f} MB" if total else ""))
    return True


def fetch_embed_zip() -> Path:
    dest = CACHE / f"python-{VERSION}-embed-amd64.zip"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log(f"使用缓存 {dest}")
        return dest
    for url in MIRRORS:
        if download(url, dest):
            return dest
    raise SystemExit(
        "所有镜像均下载失败。请手动下载 python-"
        f"{VERSION}-embed-amd64.zip 放到 {CACHE} 后重试。"
    )


def extract(zip_path: Path) -> None:
    PY_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(PY_DIR)
    log(f"已解压到 {PY_DIR}")


def copy_site_packages() -> None:
    """把 .venv 的依赖复制进便携运行时（真正自包含，便于整包分发）。"""
    if not VENV_SITE.exists():
        log("未找到 .venv 依赖目录，跳过依赖复制")
        return
    target = PY_DIR / "Lib" / "site-packages"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(VENV_SITE, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    log(f"已复制依赖到 {target}（{size / 1_048_576:.1f} MB）")


def write_pth() -> None:
    """配置 sys.path。

    embeddable 发行版靠 ``python3xx._pth`` 取代常规的 site 机制：
    列出 ``import site`` 才会处理 ``.pth`` 文件（部分包依赖它注册钩子）。
    """
    pth = PY_DIR / f"python{sys.version_info.major}{sys.version_info.minor}._pth"
    lines = [
        f"python{sys.version_info.major}{sys.version_info.minor}.zip",
        ".",
        "Lib\\site-packages",
        "..",
        "import site",
        "",
    ]
    pth.write_text("\n".join(lines), encoding="utf-8")
    log(f"已写入 {pth.name}:\n    " + "\n    ".join(lines[:4]))


def write_sitecustomize() -> None:
    """让便携解释器默认把项目根加入 sys.path（可直接 ``python -m medscholar``）。"""
    target = PY_DIR / "Lib" / "site-packages" / "sitecustomize.py"
    if not target.parent.exists():
        return
    target.write_text(
        "import sys, pathlib\n"
        "_root = pathlib.Path(__file__).resolve().parents[3]\n"
        "if _root.is_dir() and str(_root) not in sys.path:\n"
        "    sys.path.insert(0, str(_root))\n",
        encoding="utf-8",
    )
    log("已写入 sitecustomize.py（自动挂载项目根目录）")


def main() -> int:
    log(f"目标 Python 版本：{VERSION}")
    zip_path = fetch_embed_zip()
    extract(zip_path)
    copy_site_packages()
    write_pth()
    write_sitecustomize()

    exe = PY_DIR / "python.exe"
    log(f"便携解释器：{exe}")
    if not exe.exists():
        raise SystemExit("python.exe 缺失，解压可能失败")
    log("完成。后续请使用 .python\\python.exe 运行本项目。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
