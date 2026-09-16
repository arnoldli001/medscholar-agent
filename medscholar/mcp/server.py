"""MCP Server：把 MedScholar 的检索/格式化能力以 Model Context Protocol 暴露出去。

用途：让 Claude Desktop、Cursor、Cherry Studio 等任意支持 MCP 的客户端
直接调用本地知识库与学术检索能力（需求 3.3）。

启动::

    .python\\python.exe -m medscholar.mcp.server          # stdio 传输
    .python\\python.exe -m medscholar.mcp.server --http   # 流式 HTTP 传输

依赖可选的 ``mcp`` 包：``.python\\python.exe -m pip install mcp``
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from ..tools import TOOLS, call_tool

logger = logging.getLogger(__name__)

__all__ = ["build_server", "main", "MCP_AVAILABLE"]

try:  # 可选依赖
    from mcp.server.fastmcp import FastMCP  # type: ignore

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于是否安装
    FastMCP = None  # type: ignore
    MCP_AVAILABLE = False


def build_server(name: str = "medscholar"):
    """构建 FastMCP 实例，把 :data:`medscholar.tools.TOOLS` 注册成 MCP 工具。"""
    if not MCP_AVAILABLE:
        raise RuntimeError(
            "未安装 mcp 包，无法启动 MCP Server。安装命令：\n"
            "    .python\\python.exe -m pip install mcp"
        )

    server = FastMCP(name)

    for spec in TOOLS.values():
        # 用闭包绑定工具名，避免循环变量捕获问题
        def _make(tool_name: str, description: str):
            async def _handler(**kwargs: Any) -> dict[str, Any]:
                result = await call_tool(tool_name, kwargs)
                if not result.get("ok", True):
                    # MCP 约定：失败信息直接放进返回内容，由客户端决定如何呈现
                    return {"error": result.get("error", "未知错误")}
                return {k: v for k, v in result.items() if k != "ok"}

            _handler.__name__ = tool_name
            _handler.__doc__ = description
            return _handler

        server.tool(name=spec.name, description=spec.description)(_make(spec.name, spec.description))

    # 额外提供一个总览资源，方便客户端做能力发现
    @server.tool(
        name="list_capabilities",
        description="返回 MedScholar 的可用工具清单与本地知识库概况。",
    )
    async def list_capabilities() -> dict[str, Any]:
        from ..db.connect import get_db

        return {
            "tools": [
                {"name": spec.name, "description": spec.description} for spec in TOOLS.values()
            ],
            "library": get_db().stats(),
        }

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="medscholar-mcp", description="MedScholar MCP Server"
    )
    parser.add_argument("--http", action="store_true", help="使用流式 HTTP 传输（默认 stdio）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8761)
    parser.add_argument("--name", default="medscholar", help="MCP server 名称")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # stdio 传输时 stdout 必须留给协议本身
    )

    if not MCP_AVAILABLE:
        print(
            "未安装 mcp 包。请先执行：\n"
            "    .python\\python.exe -m pip install mcp\n"
            "安装后重试。",
            file=sys.stderr,
        )
        return 1

    server = build_server(args.name)
    if args.http:
        server.settings.host = args.host
        server.settings.port = args.port
        logger.info("MedScholar MCP Server 启动于 http://%s:%d", args.host, args.port)
        server.run(transport="streamable-http")
    else:
        logger.info("MedScholar MCP Server 启动（stdio）")
        server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
