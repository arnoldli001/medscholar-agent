"""可观测性：trace/span、LLM 用量与成本账本、失败分类。

本层是最底层基础设施，只依赖标准库：它被 api / agent / llm / server 所有层使用，
一旦反过来依赖业务层，依赖图立刻成环（由 ``scripts/check_arch.py`` 强制）。

本地学术智能体跑一次综述会发出几十次 LLM 调用和上百次网络请求。出问题时用户只会说
"卡住了/没结果"，而日志里是几千行散乱输出。本模块把三件事变成可聚合的结构化数据：

1. :class:`LLMUsage` + :class:`UsageLedger` —— 每次调用的 token、耗时、成本、阶段，
   用来回答"这次综述花了多少钱、慢在哪一步"；
2. :func:`classify_failure` —— 把异常归类成有限几个取值，让"翻日志"变成"看面板"；
3. :class:`TraceRecorder` —— 一棵 span 树，还原每次运行的调用链路与耗时归属。

设计取舍（都是踩过坑之后定的）：

* 并发隔离用 contextvars，绝不用模块级全局变量/threading.local。项目里已经踩过
  "``httpx.AsyncClient`` 连接池绑定事件循环"和"``asyncio.Event`` 跨事件循环复用"导致的
  静默挂起：单事件循环下多个 asyncio 任务共享同一线程，threading.local 根本区分不开它们，
  于是 A 任务的 span 会被 B 任务当成自己的父节点，trace 树串台且极难定位；
  跨事件循环时全局变量更是直接泄漏。contextvars 的语义是"每个 asyncio 任务/线程拿到
  自己的副本"，这正是要的隔离粒度。
* 账本用 ``threading.Lock`` 而不是 ``asyncio.Lock``。记账是纯内存的微秒级操作，用锁足够；
  用 asyncio.Lock 会强迫所有调用点变成 ``await``，而且一旦有人忘了 await 就是静默失效。
  锁内绝不做 IO、绝不 await，所以不会长时间占用事件循环线程。
* 账本明细有上限（默认 1000 条）。长期驻留的 Web 服务不能无界增长，而统计所需的
  数字在 ``record()`` 时就累加好了，明细只服务于"看最近发生了什么"，因此可以丢老数据。
* 未知模型的成本算 0，而不是抛异常。统计模块绝不能因为"模型没登记单价"把主流程搞挂；
  宁可少算也不能让一次综述因为成本核算失败而中断。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "FAILURE_KINDS",
    "LEDGER",
    "PRICES",
    "LLMUsage",
    "Span",
    "TraceRecorder",
    "UsageLedger",
    "classify_failure",
    "create_trace",
    "current_span",
    "current_trace",
    "estimate_cost",
    "run_in_trace",
    "set_current_trace",
]


# ---------------------------------------------------------------------------
# 1) 单价表与成本估算
# ---------------------------------------------------------------------------

#: 单价表：``{模型前缀: (输入单价, 输出单价)}``，单位 元 / 100 万 token。
#:
#: 单价会变，这里的数字仅用于量级估算，以各家官网当期价格为准。
#: 硬编码而非配置项的原因：
#:   - 这是"估算"不是"账单"，差 30% 不影响"这次综述花了多少"的判断；
#:   - 放进配置会让每个用户都要自己去查价格才能看懂成本面板，得不偿失；
#:   - 真正的账以云厂商账单为准，这里只回答"是一分钱还是一块钱"。
#:
#: 本地模型一律 0 价——README 里明确承诺"本地推理成本为 0，云端一次综述 2~4 万 token
#: 不到一毛钱"。本地跑在用户自己的显卡上，唯一成本是电费，把它算成钱只会让成本面板
#: 出现无意义的数字，所以 ``ollama`` 全系和任何带 ``:tag`` 的本地模型都是 0。
PRICES: dict[str, tuple[float, float]] = {
    # 本地推理：显式写 0，让成本面板上的"0 元"是可解释的，而不是"查不到单价所以是 0"
    "ollama": (0.0, 0.0),
    "qwen": (0.0, 0.0),
    "llama": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "gemma": (0.0, 0.0),
    "phi": (0.0, 0.0),
    "deepseek-r1": (0.0, 0.0),
    # 云端（元 / 百万 token，量级估算；缓存命中价更低，这里按未命中保守估）
    "deepseek-chat": (2.0, 8.0),
    "deepseek-reasoner": (4.0, 16.0),
    "deepseek": (2.0, 8.0),
    "gpt-4o-mini": (1.0, 4.0),
    "gpt-4o": (18.0, 72.0),
    "gpt-4": (180.0, 360.0),
    "qwen-max": (20.0, 60.0),
    "qwen-plus": (0.8, 2.0),
    "claude": (22.0, 110.0),
    "moonshot": (12.0, 12.0),
    "glm": (1.0, 1.0),
}


def _is_local_model(model: str) -> bool:
    """判断是不是跑在用户机器上的本地模型。

    判据：``provider:tag`` 形式的带 tag 模型名（``qwen3:8b``、``llama3.1:70b``）
    一定是本地推理——云端模型从来不会在 API 里带 ``:tag``。这一个信号就能覆盖
    用户自己 pull 下来的任意模型，不需要维护一份永远追不上的本地模型清单。
    """
    return ":" in model


def _match_price(model: str) -> tuple[float, float] | None:
    """按前缀匹配单价，最长前缀优先。

    最长优先是必须的：``deepseek-chat-v3`` 同时匹配 ``deepseek`` 和 ``deepseek-chat``，
    只按字典顺序取第一个会拿到更粗的档位；反过来（先短后长）会把
    ``deepseek-reasoner`` 误判成便宜的 ``deepseek``，成本直接少算一半。
    另外模型名大小写不统一是常态，统一小写后匹配。
    """
    key = (model or "").strip().lower()
    if not key:
        return None
    best_key = ""
    best: tuple[float, float] | None = None
    for prefix, price in PRICES.items():
        if key.startswith(prefix) and len(prefix) > len(best_key):
            best_key = prefix
            best = price
    return best


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """估算一次调用的人民币成本（元）。

    * 本地模型 / 带 tag 的本地模型 —— 0.0；
    * 前缀匹配（最长优先），所以 ``deepseek-chat-v3`` 也能命中 ``deepseek-chat``；
    * 未登记的模型返回 0.0 而不是抛异常：统计代码在任何情况下都不该把主流程搞挂，
      少算一次成本只是面板偏低，抛异常却会让用户的综述直接失败。
    """
    if _is_local_model(model):
        return 0.0
    price = _match_price(model)
    if price is None:
        return 0.0
    prompt_price, completion_price = price
    cost = (max(0, int(prompt_tokens)) * prompt_price) / 1_000_000.0
    cost += (max(0, int(completion_tokens)) * completion_price) / 1_000_000.0
    # 保留 6 位小数：一次本地调用是 0，一次云端小调用是 0.00012 这种量级，
    # 不舍入的话浮点尾巴会出现在 JSON 和面板上，看起来像 bug。
    return round(cost, 6)


# ---------------------------------------------------------------------------
# 2) 失败分类
# ---------------------------------------------------------------------------

#: 失败类别全集（顺序即匹配优先级，见 :func:`classify_failure`）。
#:
#: 值域固定成这几个字符串：无法分类的失败等于没有告警。异常原文里既有
#: HTTP 状态码又有服务端返回的 JSON，直接展示只能靠人一条条读；收敛成有限取值之后
#: 才能做"rate_limited 突然涨了 10 倍"这种真正的告警，也才能把"翻日志"变成"看面板"。
FAILURE_KINDS: tuple[str, ...] = (
    "timeout",
    "rate_limited",
    "auth",
    "bad_request",
    "not_found",
    "server_error",
    "connection",
    "parse",
    "context_overflow",
    "cancelled",
    "unknown",
)

#: 关键词规则表：顺序敏感，先匹配更具体的。
#:
#: * ``context_overflow`` 必须排在 ``bad_request`` 之前——上下文超长本身就是 400，
#:   但它有明确的处置办法（截断/换模型），混进 bad_request 就再也分不出来了；
#:   微软/Azure 的报错文案是 "maximum context length"，所以两条都收。
#: * ``parse`` 排在 ``connection`` 之前——截断的流式响应经常同时出现
#:   "unexpected end of JSON" 和超时字样，这里按"更可行动"的一侧归类：
#:   解析失败要查 prompt 与响应格式，连接失败要查网络。
#: * 5xx 放在最后——"500 internal server error" 里不含上面的词，但
#:   "502 bad gateway" 含 "bad gateway"，所以不能把 5xx 排到 bad_request 前面。
_STRING_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("context_overflow", ("context length", "context_length", "maximum context", "context window",
                          "too many tokens", "token limit", "上下文超长", "超出上下文")),
    ("rate_limited", ("429", "rate limit", "ratelimit", "too many requests", "quota",
                      "throttl", "限流", "请求过多", "配额")),
    ("auth", ("401", "403", "unauthorized", "forbidden", "invalid api key", "invalid_api_key",
              "api key", "apikey", "authentication", "permission denied", "密钥", "鉴权")),
    ("bad_request", ("400", "422", "invalid request", "validation error", "unprocessable",
                     "bad request", "参数错误")),
    ("not_found", ("404", "not found", "no such", "unknown model", "model_not_found",
                   "does not exist", "未找到")),
    ("parse", ("json", "decode", "expecting value", "unterminated", "parse error", "parseerror",
               "invalid json", "解析失败")),
    ("connection", ("connection", "connect", "network", "unreachable", "reset by peer",
                    "broken pipe", "eof", "dns", "socket", "ssl", "10061", "连接", "网络")),
    ("timeout", ("timeout", "timed out", "deadline exceeded", "超时")),
    ("cancelled", ("cancel", "取消")),
    ("server_error", ("500", "502", "503", "504", "internal server error", "bad gateway",
                      "service unavailable", "gateway timeout", "server error", "服务不可用")),
)


def classify_failure(exc: BaseException | str | None) -> str:
    """把异常/错误文本归类成 :data:`FAILURE_KINDS` 中的一个取值。

    匹配顺序：异常类型优先，其次关键词，最后 unknown。

    几个刻意的选择：

    * ``CancelledError`` 必须按类型先判。它是任务被主动取消（用户点了停止、请求断开），
      不是故障；在 Python 3.8+ 它继承自 ``BaseException`` 而非 ``Exception``，
      ``except Exception`` 根本抓不到它，用字符串判又会因为消息为空而落到 unknown。
    * ``TimeoutError`` 也按类型先判，而且必须排在 OSError 之前——``TimeoutError``
      是 ``OSError`` 的子类，先判 OSError 会把超时误报成连接失败。Python 3.11 起
      ``asyncio.TimeoutError`` 就是内建 ``TimeoutError`` 的别名，这里一并覆盖。
    * 传 ``None`` 返回 ``""``（空串）而不是 ``"unknown"``：调用方用"没有错误信息"表示
      成功或未采集，返回 unknown 会让 ``by_error_kind`` 里混进一堆假故障。
    * 未知类型的异常仍归入 ``"unknown"`` 并保留在计数里——宁可承认"我不知道这是什么"
      也不要塞进某个已知类别制造假信号。
    """
    if exc is None:
        return ""

    # 类型判定：这几类从类型上就是确定的，读字符串只会引入误判
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if isinstance(exc, TimeoutError):
        return "timeout"

    if isinstance(exc, BaseException):
        if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError, ValueError)):
            # ValueError 本身不一定是解析问题，所以只在"错误文本像解析失败"时才算 parse，
            # 否则继续按文本走下面更具体的规则（例如 pydantic 的 validation error）。
            if isinstance(exc, json.JSONDecodeError) or "json" in str(exc).lower():
                return "parse"
        text = f"{type(exc).__name__}: {exc}"
    else:
        text = str(exc)

    lowered = text.lower()
    if not lowered.strip():
        return "unknown"
    for kind, keywords in _STRING_RULES:
        if any(kw in lowered for kw in keywords):
            return kind
    return "unknown"


# ---------------------------------------------------------------------------
# 3) 用量账本
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMUsage:
    """一次 LLM 调用的记录（不可变：记账之后就不该被任何一层改写）。

    字段刻意扁平、全部是可 JSON 序列化的简单值：它要能直接进日志、进 SSE 事件、
    进 SQLite，不需要任何自定义编解码。
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ok: bool = True
    phase: str = ""
    run_id: str = ""
    error_kind: str = ""
    cache_hit: bool = False

    @property
    def total_tokens(self) -> int:
        """输入 + 输出 token 总数。"""
        return int(self.prompt_tokens) + int(self.completion_tokens)

    @property
    def cost_yuan(self) -> float:
        """这一次调用的人民币成本（本地模型为 0）。"""
        return estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)

    def to_dict(self) -> dict[str, Any]:
        """转成可直接 ``json.dumps`` 的 dict（含派生出来的总 token 与成本）。"""
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(float(self.latency_ms), 3),
            "ok": self.ok,
            "phase": self.phase,
            "run_id": self.run_id,
            "error_kind": self.error_kind,
            "cache_hit": self.cache_hit,
            "cost_yuan": self.cost_yuan,
        }


#: 明细保留上限。统计数字在 record() 时就已经累加，明细只服务于"最近发生了什么"，
#: 因此可以丢老数据 —— 长期驻留的 Web 服务绝不能无界增长（一次批量综述就是几千条）。
_MAX_RECENT = 1000


class UsageLedger:
    """进程内账本：记录每次 LLM 调用，按模型/阶段/运行聚合，并算出人民币成本。

    线程安全：所有读写都在 ``threading.Lock`` 之内，且锁内不做任何 IO、不 await，
    因此既能被工作线程（批处理）调用，也能被事件循环里的协程直接调用而不会阻塞别人。
    """

    def __init__(self, max_recent: int = _MAX_RECENT) -> None:
        self._lock = threading.Lock()
        self.max_recent = max(1, int(max_recent))
        self._items: deque[LLMUsage] = deque(maxlen=self.max_recent)
        self._total_recorded = 0

    # -- 写入 -----------------------------------------------------------------

    def record(self, usage: LLMUsage) -> None:
        """记一次调用。这是唯一的热路径，必须便宜（纯内存追加 + 计数）。"""
        with self._lock:
            self._items.append(usage)
            self._total_recorded += 1

    def reset(self) -> None:
        """清空账本（测试/新一轮会话用）。"""
        with self._lock:
            self._items.clear()
            self._total_recorded = 0

    # -- 读取 -----------------------------------------------------------------

    def all(self) -> list[LLMUsage]:
        """返回当前的明细快照（副本，改它不会影响账本）。"""
        with self._lock:
            return list(self._items)

    def recent(self, n: int = 20) -> list[LLMUsage]:
        """返回最近 ``n`` 条（新的在后）。

        用 ``deque`` 而不是 ``list``：高频记账时 ``list.pop(0)`` 是 O(n)，
        几千条之后每次记账都在搬内存，会成为事件循环里的隐形卡顿。
        """
        if n <= 0:
            return []
        with self._lock:
            items = list(self._items)
        return items[-n:]

    def summary(self) -> dict[str, Any]:
        """聚合统计。返回值全部是可 JSON 序列化的简单值，可直接作为 API 响应。

        返回的键（沿用这些名字，前端与测试都依赖）：

        * ``calls`` / ``ok_calls`` / ``failed_calls`` / ``cache_hits``
        * ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``
        * ``cost_yuan``（本地模型为 0）
        * ``latency_ms_p50`` / ``latency_ms_p95``
        * ``by_model`` / ``by_phase`` / ``by_error_kind``（每个都是 dict）

        注：这里的统计只覆盖当前保留的明细（见 ``max_recent``）；被丢弃的老数据
        不再参与统计。这是有意的取舍——精确的长期累计应该落到 SQLite，进程内账本
        负责的是"这次运行/今天这批"的量级。
        """
        with self._lock:
            items = list(self._items)

        calls = len(items)
        ok_calls = sum(1 for it in items if it.ok)
        failed_calls = calls - ok_calls
        cache_hits = sum(1 for it in items if it.cache_hit)
        prompt_tokens = sum(int(it.prompt_tokens) for it in items)
        completion_tokens = sum(int(it.completion_tokens) for it in items)
        cost = round(sum(it.cost_yuan for it in items), 6)
        # 排序一次，p50/p95 与各分组的分位数复用同一个有序列表
        latencies = sorted(float(it.latency_ms) for it in items)

        by_model: dict[str, dict[str, Any]] = {}
        by_phase: dict[str, dict[str, Any]] = {}
        by_error_kind: dict[str, dict[str, Any]] = {}
        model_latencies: dict[str, list[float]] = {}
        phase_latencies: dict[str, list[float]] = {}

        for it in items:
            model_key = it.model or "(未知模型)"
            model_bucket = by_model.setdefault(
                model_key,
                {
                    "calls": 0,
                    "ok_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_yuan": 0.0,
                },
            )
            model_bucket["calls"] += 1
            model_bucket["ok_calls"] += 1 if it.ok else 0
            model_bucket["prompt_tokens"] += int(it.prompt_tokens)
            model_bucket["completion_tokens"] += int(it.completion_tokens)
            model_bucket["total_tokens"] += it.total_tokens
            model_bucket["cost_yuan"] = round(model_bucket["cost_yuan"] + it.cost_yuan, 6)
            model_latencies.setdefault(model_key, []).append(float(it.latency_ms))

            # phase 可能是空串（不是每次调用都在 agent 阶段里），归到 "(未标注)"
            phase_key = it.phase or "(未标注)"
            phase_bucket = by_phase.setdefault(
                phase_key, {"calls": 0, "failed_calls": 0, "total_tokens": 0, "cost_yuan": 0.0}
            )
            phase_bucket["calls"] += 1
            phase_bucket["failed_calls"] += 0 if it.ok else 1
            phase_bucket["total_tokens"] += it.total_tokens
            phase_bucket["cost_yuan"] = round(phase_bucket["cost_yuan"] + it.cost_yuan, 6)
            phase_latencies.setdefault(phase_key, []).append(float(it.latency_ms))

            # 只有失败才记 error_kind：成功记录没有类别，塞进去只会稀释计数
            if it.error_kind:
                err_bucket = by_error_kind.setdefault(
                    it.error_kind, {"calls": 0, "models": []}
                )
                err_bucket["calls"] += 1
                if model_key not in err_bucket["models"]:
                    err_bucket["models"].append(model_key)

        # 分组内也带上 p95/最慢一次：只看总 p95 无法判断"是慢在云端还是慢在本地"
        for key, values in model_latencies.items():
            by_model[key]["latency_ms_p95"] = _percentile(sorted(values), 0.95)
            by_model[key]["latency_ms_max"] = round(max(values), 3)
        for key, values in phase_latencies.items():
            by_phase[key]["latency_ms_p95"] = _percentile(sorted(values), 0.95)
            by_phase[key]["latency_ms_max"] = round(max(values), 3)

        return {
            "calls": calls,
            "ok_calls": ok_calls,
            "failed_calls": failed_calls,
            "cache_hits": cache_hits,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_yuan": cost,
            "latency_ms_p50": _percentile(latencies, 0.5),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "by_model": by_model,
            "by_phase": by_phase,
            "by_error_kind": by_error_kind,
            # 明细上限：面板上要能看出"统计只覆盖最近 N 次"，否则会被当成全局累计
            "window_size": self.max_recent,
            "recorded_total": self._total_recorded,
        }


def _percentile(sorted_values: list[float], q: float) -> float:
    """手写分位数（线性插值），``sorted_values`` 必须已升序。

    不用 numpy：这是最底层模块，为了一个分位数引入几十 MB 的科学计算栈不划算，
    而且它会随平台/版本漂移（打包给用户的便携运行时里多一个二进制依赖就是多一个坑）。
    用线性插值而不是"取第 k 个"：在只有 3~20 个样本的场景（一次综述的调用次数
    就这么点）里，最近邻会把 p95 变成"最大值"，夸大尾部延迟；线性插值至少是连续的。

    空输入返回 0.0（而不是抛异常或 NaN）：空账本是正常状态（进程刚起来、还没调用
    过模型），成本面板上应该显示 0 而不是崩掉或显示 NaN。
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return round(float(sorted_values[0]), 3)
    q = min(1.0, max(0.0, float(q)))
    pos = q * (n - 1)
    lower = int(pos)
    upper = min(lower + 1, n - 1)
    frac = pos - lower
    value = float(sorted_values[lower]) * (1.0 - frac) + float(sorted_values[upper]) * frac
    return round(value, 3)


#: 模块级单例，供没有依赖注入的地方（脚本、CLI、诊断接口）直接用。
LEDGER = UsageLedger()


# ---------------------------------------------------------------------------
# 4) trace / span
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """一个带耗时的操作节点。

    ``attrs`` 只允许放可 JSON 序列化的简单值（str/int/float/bool/None 及它们的
    浅层容器）。不要塞 Paper 对象、ORM 行、httpx 响应：span 最终要落 JSONL 和发给前端，
    放活对象会在序列化时炸，而且会把整棵对象图钉在内存里导致泄漏。
    """

    name: str
    start: float  # time.monotonic()，不能用 time.time()（会被 NTP 校时回跳）
    end: float | None = None
    status: str = "running"  # running / ok / error
    error: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """耗时（毫秒）。未结束的 span 按"到现在为止"计算，便于运行中查看进度。"""
        end = time.monotonic() if self.end is None else self.end
        return max(0.0, (end - self.start) * 1000.0)

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的 dict（子节点递归）。"""
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
            "attrs": dict(self.attrs),
            "children": [child.to_dict() for child in self.children],
        }


# 当前 trace / 当前 span 栈。
#
# 用 ContextVar 而不是全局变量：单事件循环下多个 asyncio 任务共享同一个线程，
# 全局变量/threading.local 无法区分它们，于是并发任务（本项目里常见：多源检索、
# 批量摘要）会互相把对方的 span 当成父节点，trace 树串台；跨事件循环时全局变量
# 还会直接泄漏到别的请求。contextvars 保证"每个任务/线程一份副本"，这正是要的粒度。
_CURRENT_TRACE: ContextVar[TraceRecorder | None] = ContextVar("medscholar_trace", default=None)
_SPAN_STACK: ContextVar[tuple[Span, ...]] = ContextVar("medscholar_span_stack", default=())


class TraceRecorder:
    """一棵 span 树：还原一次运行的调用链路，并把耗时归属到具体的 span。

    用 ``contextvars`` 保证 asyncio 并发任务之间互不串台（详见模块 docstring）。
    """

    def __init__(self, trace_id: str | None = None, name: str = "run") -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.started_at = time.time()  # 墙上时间只用于展示，耗时一律用 monotonic
        self.root = Span(name=name, start=time.monotonic())
        self.root.end = self.root.start
        self.root.status = "ok"
        self._lock = threading.Lock()  # 工作线程里也可能打点，children 追加要防并发
        self._enter_token: Any = None  # ``with`` 进出时用于精确还原上下文

    # -- 打点 -----------------------------------------------------------------

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """开一个 span，支持嵌套；异常时自动标 ``error`` 并记录文本，仍然向外抛。

        不吞异常：trace 的职责是"记录发生了什么"，不是"决定要不要继续"。
        一旦在这里吞掉异常，上层就再也看不到真实失败，会变成更难查的静默错误。

        span 的父节点取当前 contextvars 栈顶：同一个协程里嵌套调用天然成树；
        若两个协程各自持有一份上下文，它们的栈互不可见，因此不会串台。
        """
        stack = _SPAN_STACK.get()
        parent = stack[-1] if stack else self.root
        node = Span(name=name, start=time.monotonic(), attrs=dict(attrs))
        with self._lock:
            parent.children.append(node)

        token = _SPAN_STACK.set(stack + (node,))
        try:
            yield node
        except BaseException as exc:  # noqa: BLE001 - 记完必须原样抛出，这里只是过路
            node.end = time.monotonic()
            node.status = "error"
            # 只留一行短的错误文本：完整堆栈进日志，span 树里留摘要足够定位
            node.error = f"{type(exc).__name__}: {exc}"[:500]
            node.attrs.setdefault("error_kind", classify_failure(exc))
            raise
        else:
            node.end = time.monotonic()
            node.status = "ok"
        finally:
            _SPAN_STACK.reset(token)

    # -- 输出 -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """整棵树的 dict 形式，``root.duration_ms`` 即整次运行的耗时。"""
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "root": self.root.to_dict(),
        }

    def to_jsonl(self, path: str | Path) -> None:
        """把整棵树作为一行 JSON 追加到 ``path``（JSONL：一行一次运行）。

        刻意用追加而不是覆盖：排障时往往要对比"正常那次"和"失败那次"，
        追加模式下每次运行一行，``select`` 出来直接可比。

        编码固定 ``utf-8`` 且 ``ensure_ascii=False``：在 Windows 中文环境下默认编码
        可能是 cp936，中文 span 名和错误文本会直接乱码或抛 UnicodeEncodeError——
        项目里已经因为 .bat 的代码页问题踩过一次，这里不留同样的坑。
        父目录自动创建，调用方不需要先 mkdir。
        """
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self.to_dict(), ensure_ascii=False, default=str)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def summary(self) -> dict[str, Any]:
        """每个 span 名称的调用次数 / 总耗时 / 最慢一次。

        用来回答"这次运行时间花在哪一类操作上"——比看完整棵树快得多。
        同名 span 会出现多次（例如每篇文献一个 ``fetch``），所以按名字聚合而非按节点。
        """
        stats: dict[str, dict[str, Any]] = {}

        def walk(node: Span) -> None:
            bucket = stats.setdefault(
                node.name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "errors": 0}
            )
            duration = round(node.duration_ms, 3)
            bucket["count"] += 1
            bucket["total_ms"] = round(bucket["total_ms"] + duration, 3)
            bucket["max_ms"] = max(bucket["max_ms"], duration)
            bucket["errors"] += 1 if node.status == "error" else 0
            for child in node.children:
                walk(child)

        walk(self.root)
        return {"trace_id": self.trace_id, "spans": stats}

    # -- 绑定到当前上下文 -------------------------------------------------------

    def __enter__(self) -> TraceRecorder:
        """作为上下文管理器绑定为当前 trace，退出时精确还原。"""
        self._enter_token = _CURRENT_TRACE.set(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        token, self._enter_token = self._enter_token, None
        if token is not None:
            _CURRENT_TRACE.reset(token)

    def finish(self) -> dict[str, Any]:
        """结束这棵 trace：解绑当前上下文并返回 ``to_dict()``。

        ``trace = create_trace()`` 之后记得在 ``finally`` 里调一次：trace 是绑定在
        contextvars 上的，而一个长期存活的任务（Web 请求、CLI 命令）的上下文是复用的，
        不解绑就会让下一次操作把 span 挂到上一棵树上——这种"串台"表现为 trace 越来越
        长、耗时归属错乱，而且不会报错。
        """
        set_current_trace(None)
        if self.root.end is None:
            self.root.end = time.monotonic()
        return self.to_dict()


def create_trace(name: str = "run") -> TraceRecorder:
    """新建一棵 trace 并绑定为当前 trace，返回它。

    这是最常用的入口：``trace = create_trace("review")``，之后同协程/同线程内的
    ``trace.span(...)`` 自动成树，``current_trace()`` 也能拿到它。用完调 ``finish()``。

    并发隔离从哪来：asyncio 里每个 ``Task`` 在创建时就拷贝了一份 context，
    所以 ``asyncio.gather(a(), b())`` 中的两个任务各自绑定自己的 trace，互不可见——
    这正是模块 docstring 里说的"不能靠全局变量"的原因（全局变量下两个任务会互相
    覆盖对方的 trace）。反过来，先建 trace 再派生任务时子任务会继承同一棵 trace，
    这是有意的：那属于同一条调用链，挂在同一棵树上才对。
    """
    recorder = TraceRecorder(name=name)
    _CURRENT_TRACE.set(recorder)
    return recorder


def current_trace() -> TraceRecorder | None:
    """取当前上下文绑定的 trace；没有则返回 None（方便调用方按需打点）。"""
    return _CURRENT_TRACE.get()


def set_current_trace(recorder: TraceRecorder | None) -> Any:
    """绑定/解绑当前 trace，返回 ``contextvars.Token``。

    返回 token 是为了能精确还原（``_CURRENT_TRACE.reset(token)``），而不是把"清空"
    实现成 set(None) —— 后者在嵌套场景（请求里再起子任务）会抹掉外层 trace。
    """
    return _CURRENT_TRACE.set(recorder)


def run_in_trace(func: Any, *args: Any, name: str = "run", **kwargs: Any) -> Any:
    """在隔离的上下文里跑 ``func``，返回 ``(结果, trace)``；``func`` 抛异常则原样抛出。

    这是批处理/线程池场景的正确姿势：``concurrent.futures`` 的工作线程和
    ``asyncio.to_thread`` 都各自持有一份独立的 contextvars 上下文，所以在里面
    ``set`` 既不会串到调用方，也不会串到别的工作线程——不需要加锁，也不会串台。

    一个必须说清的边界：如果直接在当前线程调用它，绑定会持续到本上下文结束
    （和 :func:`create_trace` 一样），因此适合"一个工作线程跑一批、跑完线程就复用下一批"
    的用法；如果要在同一上下文里连续跑多段且互不干扰，调用方自己包一层
    ``contextvars.copy_context().run(...)`` 即可获得完全隔离（``copy_context`` 的
    ``set`` 不会回写外层，这一点已在本仓库用最小用例验证过）。
    """
    recorder = TraceRecorder(name=name)
    _CURRENT_TRACE.set(recorder)
    result = func(*args, **kwargs)
    if recorder.root.end is None:
        recorder.root.end = time.monotonic()
    return result, recorder


def current_span() -> Span | None:
    """取当前上下文栈顶的 span；没有则返回 None。

    给中间层打补充信息用（例如在 HTTP 层拿到状态码后回填 ``status_code``），
    避免为了写一个属性而把 span 对象层层往下传参。
    """
    stack = _SPAN_STACK.get()
    return stack[-1] if stack else None
