"""最小化定位脚本：找出哪个端点会挂住。

每个请求都用 ``asyncio.wait_for`` 包住，一旦超时就能拿到确切的等待位置。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SANDBOX = Path(tempfile.gettempdir()) / "medscholar_hang"
if SANDBOX.exists():
    shutil.rmtree(SANDBOX, ignore_errors=True)
SANDBOX.mkdir(parents=True, exist_ok=True)
os.environ["MEDSCHOLAR_HOME"] = str(SANDBOX)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import httpx  # noqa: E402

from medscholar.db import repo  # noqa: E402
from medscholar.models import Paper  # noqa: E402
from medscholar.server.app import app  # noqa: E402

PATHS = [
    ("GET", "/api/health"),
    ("GET", "/api/sessions"),
    ("GET", "/api/artifacts"),
    ("GET", "/api/projects"),
    ("GET", "/api/cite/styles"),
    ("GET", "/api/papers"),
]


async def main() -> int:
    db = repo.get_db()
    repo.insert_papers(
        [Paper(title="hang probe paper", source="manual", abstract="probe")], db=db, embed=False
    )
    print("seeded", flush=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=20.0) as client:
            for method, path in PATHS:
                print(f"--> {method} {path}", flush=True)
                try:
                    response = await asyncio.wait_for(client.request(method, path), timeout=20)
                except asyncio.TimeoutError:
                    print("    !! TIMEOUT 20s", flush=True)
                    return 1
                except Exception as exc:
                    print(f"    !! {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc(limit=4)
                    return 1
                print(f"    ok {response.status_code}", flush=True)
    print("ALL OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
