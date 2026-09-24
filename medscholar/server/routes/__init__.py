"""HTTP 路由层：54 个接口按中文业务标签拆成六个 APIRouter（system/papers/search/agent/library/writing）。

纪律：router 之间不得互相 import（共享上提到 deps.py）；只注册路由、调业务模块，装配归 app.py。
"""

from __future__ import annotations
