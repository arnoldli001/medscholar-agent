"""横切平台层：配置、可观测性、韧性、安全、提示词。

**依赖规则（由 ``scripts/check_arch.py`` 强制）**：这一层是最底层 ——
所有层都可以依赖它，它**只依赖标准库**（和极少数无业务含义的第三方库），
绝不导入 ``medscholar`` 的其他层。原因是它被所有层使用，
一旦它反向依赖业务层，整个依赖图立刻变成环。

* :mod:`medscholar.platform.config` —— 配置加载（YAML + .env + 环境变量覆盖）；
* :mod:`medscholar.platform.observability` —— trace/span、token 与成本账本、失败分类；
* :mod:`medscholar.platform.resilience` —— 重试、超时、熔断、令牌桶、并发舱壁；
* :mod:`medscholar.platform.security` —— 提示注入防御、密钥脱敏、输出护栏；
* :mod:`medscholar.platform.cache` —— 带 TTL 与失效策略的进程内缓存；
* :mod:`medscholar.platform.prompts` —— 带版本号的提示词注册表。
"""

from __future__ import annotations

__all__: list[str] = []
