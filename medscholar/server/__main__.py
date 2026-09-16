"""``python -m medscholar.server`` 入口。"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    from ..config import get_config

    cfg = get_config()
    parser = argparse.ArgumentParser(prog="medscholar-server", description="MedScholar Web 工作台")
    parser.add_argument("--host", default=cfg.server.host)
    parser.add_argument("--port", type=int, default=cfg.server.port)
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重载")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    import uvicorn

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.reload:
        uvicorn.run(
            "medscholar.server.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
    else:
        from .app import app

        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
