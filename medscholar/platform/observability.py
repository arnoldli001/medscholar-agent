"""可观测性：trace/span 树、LLM 用量与成本账本、失败分类。

最底层基础设施，只依赖标准库：被 api/agent/llm/server 各层共用，反向依赖业务层
即成环（``scripts/check_arch.py`` 强制）。本模块把一次综述的几十次 LLM 调用、
上百次网络请求聚合成结构化数据：UsageLedger 记 token/耗时/成本/阶段，
classify_failure 把异常收敛成有限取值，TraceRecorder 用 span 树还原调用链。

关键取舍：
- 并发隔离用 contextvars，不用全局变量/threading.local：同一事件循环里多个 asyncio
  任务共享线程，threading.local 区分不开，span 会互相挂错父节点；跨循环全局变量会泄漏。
- 账本用 threading.Lock（纯内存微秒级操作，锁内不 IO 不 await），避免逼所有调用点 await。
- 明细有上限（默认 1000），统计在 record() 时已累加，长期驻留的 Web 服务不能无界增长。
- 未登记模型成本算 0 不抛异常：统计模块绝不能因为单价缺失把主流程搞挂。
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

#: 单价表：``{模型前缀: (输入单价, 输出单价)}``，单位 元/百万 token，仅作量级估算，
#: 以各家官网当期价格为准。硬编码而非配置项：这是估算不是账单，差 30% 不影响
#: "花了一分钱还是一块钱"的判断，真账以云厂商账单为准。
#: 本地模型一律 0 价：跑在用户自己显卡上，唯一成本是电费（README 承诺本地推理成本为 0）。
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
    """是否跑在用户机器上的本地模型：带 ``:tag`` 的模型名（``qwen3:8b``）一定是
    本地推理——云端 API 的模型名从不带 tag。这一个信号即可覆盖用户自 pull 的任意模型，
    无需维护永远追不上的本地模型清单。
    """
    return ":" in model


def _match_price(model: str) -> tuple[float, float] | None:
    """按前缀匹配单价，最长前缀优先、统一小写。

    最长优先是必须的：``deepseek-reasoner`` 同时匹配 ``deepseek``，先短后长会
    把它误判成便宜档位，成本直接少算一半。
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
    """估算一次调用的人民币成本（元）。本地模型为 0；前缀最长优先匹配；
    未登记模型返回 0.0 而不是抛异常——少算只是面板偏低，抛异常会让综述直接失败。
    """
    if _is_local_model(model):
        return 0.0
    price = _match_price(model)
    if price is None:
        return 0.0
    prompt_price, completion_price = price
    cost = (max(0, int(prompt_tokens)) * prompt_price) / 1_000_000.0
    cost += (max(0, int(completion_tokens)) * completion_price) / 1_000_000.0
    # 保留 6 位小数：云端小调用是 0.00012 量级，不 round 浮点尾巴会出现在 JSON/面板上像 bug。
    return round(cost, 6)


# ---------------------------------------------------------------------------
# 2) 失败分类
# ---------------------------------------------------------------------------

#: 失败类别全集（顺序即匹配优先级，见 :func:`classify_failure`）。
#: 收敛成固定的有限取值，才能做"rate_limited 突然涨 10 倍"这类聚合告警；
#: 无法分类的失败等于没有告警。
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

#: 关键词规则表，顺序敏感，先匹配更具体的：
#: - context_overflow 必须在 bad_request 之前：超长本身就是 400，但它有明确处置办法
#:   （截断/换模型），且 Azure 文案是 "maximum context length"，两条都要收；
#: - parse 在 connection 之前：截断的流式响应常同时像 JSON 错误和超时，按"更可行动"
#:   一侧归类（解析失败查 prompt/格式，连接失败查网络）；
#: - 5xx 放最后："502 bad gateway" 含 "bad"，排到 bad_request 前会被抢走。
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
    """把异常/错误文本归类成 :data:`FAILURE_KINDS` 之一：异常类型优先，其次关键词，最后 unknown。

    * ``CancelledError`` 按类型先判：主动取消（用户停止、请求断开）不是故障；
      Py3.8+ 它继承 ``BaseException``，``except Exception`` 抓不到，字符串判又会因消息为空落到 unknown。
    * ``TimeoutError`` 按类型且排在 OSError 之前：它是 OSError 子类，先判 OSError 会把
      超时误报成连接失败；Py3.11 起 ``asyncio.TimeoutError`` 是内建 TimeoutError 别名。
    * 传 ``None`` 返回 ``""`` 而非 ``"unknown"``：无错误信息表示成功或未采集，
      返回 unknown 会让 by_error_kind 混进假故障。
    * 真不认识的异常归 ``"unknown"`` 并保留计数，不硬塞进已知类别制造假信号。
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
            # ValueError 不一定是解析问题，仅当错误文本像 JSON 失败时才算 parse，
            # 否则继续走下面更具体的规则（如 pydantic validation error）。
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
    """一次 LLM 调用记录（不可变：记账后不应被任何层改写）。

    字段刻意扁平、全部可 JSON 序列化，可直接进日志、SSE 事件与 SQLite，无需自定义编解码。
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


#: 明细保留上限：统计在 record() 时已累加，明细只服务"最近发生了什么"，
#: 长期驻留的 Web 服务不能无界增长（一次批量综述就是几千条）。
_MAX_RECENT = 1000


class UsageLedger:
    """进程内账本：记录每次 LLM 调用，按模型/阶段/运行聚合并算人民币成本。

    所有读写在 threading.Lock 内，锁内不做 IO、不 await，工作线程与事件循环协程
    都可直接调用而不阻塞别人。
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
        """最近 ``n`` 条（新的在后）。用 deque 而非 list：高频记账下
        ``list.pop(0)`` 是 O(n)，几千条后每次记账都在搬内存，是事件循环里的隐形卡顿。
        """
        if n <= 0:
            return []
        with self._lock:
            items = list(self._items)
        return items[-n:]

    def summary(self) -> dict[str, Any]:
        """聚合统计，值全部可 JSON 序列化，可直接作 API 响应。

        键名被前端与测试依赖，勿改名：``calls`` / ``ok_calls`` / ``failed_calls`` /
        ``cache_hits`` / ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` /
        ``cost_yuan`` / ``latency_ms_p50`` / ``latency_ms_p95`` /
        ``by_model`` / ``by_phase`` / ``by_error_kind`` / ``window_size`` / ``recorded_total``。
        统计只覆盖当前保留明细（``max_recent``），丢弃的老数据不参与：精确的长期累计
        应落 SQLite，进程内账本只负责"这次运行/今天这批"的量级。
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
    """手写分位数（线性插值），``sorted_values`` 必须已升序；空输入返回 0.0
    （空账本是正常状态，面板显示 0，不崩也不返 NaN）。

    不用 numpy：最底层模块为一个分位数拖入几十 MB 二进制栈不划算。样本只有 3~20 个
    （一次综述的调用量）时最近邻会把 p95 变成最大值、夸大尾延迟，线性插值至少连续。
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

    ``attrs`` 只放可 JSON 序列化的简单值（str/int/float/bool/None 及其浅层容器）：
    span 最终要落 JSONL、发前端，塞 Paper/ORM 行/httpx 响应会在序列化时炸，
    还会把整棵对象图钉在内存里造成泄漏。
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


#: span 错误文本只留一行摘要：完整堆栈进日志，trace 树要序列化发前端，不能被长异常撑爆
_SPAN_ERROR_MAX = 500

# 当前 trace / span 栈用 ContextVar 而非全局变量/threading.local：同一事件循环的
# 多个 asyncio 任务共享线程，全局/线程局部变量区分不开，并发任务（多源检索、批量摘要）
# 会互相把对方的 span 当成父节点；跨事件循环时全局变量还会直接泄漏。
_CURRENT_TRACE: ContextVar[TraceRecorder | None] = ContextVar("medscholar_trace", default=None)
_SPAN_STACK: ContextVar[tuple[Span, ...]] = ContextVar("medscholar_span_stack", default=())


class TraceRecorder:
    """一棵 span 树：还原一次运行的调用链路，把耗时归属到具体 span。

    用 contextvars 保证 asyncio 并发任务互不串台（隔离原理见模块 docstring）。
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
        """开一个支持嵌套的 span；异常时标 error、记一行短文本后原样抛出。

        不吞异常：trace 只负责记录，吞掉会让上层看不到真实失败。父节点取 contextvars
        栈顶：同协程嵌套天然成树，不同协程各持一份上下文，栈互不可见故不串台。
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
            node.error = f"{type(exc).__name__}: {exc}"[:_SPAN_ERROR_MAX]
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
        """整棵树作为一行 JSON 追加到 ``path``（JSONL：一行一次运行）。

        用追加而非覆盖：排障时要对比"正常那次"和"失败那次"。固定 utf-8 且
        ensure_ascii=False：Windows 中文环境默认 cp936，中文 span 名会乱码或抛
        UnicodeEncodeError（.bat 代码页已踩过）。父目录自动创建。
        """
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self.to_dict(), ensure_ascii=False, default=str)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def summary(self) -> dict[str, Any]:
        """按 span 名聚合调用次数/总耗时/最慢一次/错误数，回答"时间花在哪类操作"。

        同名 span 有多个（每篇文献一个 fetch），故按名字聚合而非按节点。
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
        """结束 trace：解绑当前上下文并返回 ``to_dict()``。

        ``create_trace()`` 后必须在 finally 里调一次：长寿命任务（Web 请求、CLI 命令）
        复用上下文，不解绑会让下次操作把 span 挂到上一棵树上，表现为 trace 越来越长、
        耗时归属错乱，且不会报错。
        """
        set_current_trace(None)
        if self.root.end is None:
            self.root.end = time.monotonic()
        return self.to_dict()


def create_trace(name: str = "run") -> TraceRecorder:
    """新建 trace 并绑定为当前 trace，返回它；用完调 ``finish()``。

    之后同协程/同线程内的 ``span(...)`` 自动成树。asyncio Task 创建时拷贝 context，
    故 ``gather(a(), b())`` 两个任务各绑各的 trace、互不可见；先建 trace 再派生的
    子任务继承同一棵树（同一条调用链，有意为之）。
    """
    recorder = TraceRecorder(name=name)
    _CURRENT_TRACE.set(recorder)
    return recorder


def current_trace() -> TraceRecorder | None:
    """取当前上下文绑定的 trace；没有则返回 None（方便调用方按需打点）。"""
    return _CURRENT_TRACE.get()


def set_current_trace(recorder: TraceRecorder | None) -> Any:
    """绑定/解绑当前 trace，返回 ``contextvars.Token``。

    精确还原必须用 ``reset(token)``：把"清空"实现成 set(None) 在嵌套场景
    （请求里再起子任务）会抹掉外层 trace。
    """
    return _CURRENT_TRACE.set(recorder)


def run_in_trace(func: Any, *args: Any, name: str = "run", **kwargs: Any) -> Any:
    """在隔离上下文里跑 ``func``，返回 ``(结果, trace)``；异常原样抛出。

    批处理/线程池场景的正确姿势：工作线程与 ``asyncio.to_thread`` 各持独立的
    contextvars 副本，里面 set 不串调用方也不串别的线程，无需加锁。
    边界：直接在当前线程调用时绑定持续到本上下文结束（适合一个线程跑一批后复用）；
    要在同一上下文连跑多段且完全隔离，调用方用 ``contextvars.copy_context().run(...)``
    包一层（copy_context 内的 set 不回写外层）。
    """
    recorder = TraceRecorder(name=name)
    _CURRENT_TRACE.set(recorder)
    result = func(*args, **kwargs)
    if recorder.root.end is None:
        recorder.root.end = time.monotonic()
    return result, recorder


def current_span() -> Span | None:
    """当前栈顶 span，无则 None；供中间层回填属性（如 HTTP 层拿到状态码后补
    ``status_code``），避免为一个属性把 span 对象层层传参。
    """
    stack = _SPAN_STACK.get()
    return stack[-1] if stack else None
