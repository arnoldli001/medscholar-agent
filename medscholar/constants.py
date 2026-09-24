"""集中管理项目中的语义化常量。

项目里散布着大量"魔鬼数字"——同样的 200、600、800 在不同位置出现，
改一个地方容易漏改另一个；而且数字本身不带语义，阅读时要靠上下文猜
"这个 256 是干嘛的"。把有明确语义、可能被调整、或跨模块复用的数字
集中到这里，做到「一处定义，处处引用」。

分组按关注点（LLM 生成 / 文本裁剪 / 评分阈值 / 预算 / 运行时 / 数据库 / HTTP）；
常量名要自解释（``LLM_TEMPERATURE_OUTLINE`` 而不是 ``T1``）；
已经在 :mod:`medscholar.platform.config` 里可配置的（如 ``review_min_chars``）
不重复放在这里——配置项走 config，硬编码调优参数走这里。
"""

from __future__ import annotations

__all__ = [
    # LLM 生成参数
    "LLM_TEMPERATURE_OUTLINE",
    "LLM_TEMPERATURE_SECTION",
    "LLM_TEMPERATURE_ABSTRACT",
    "LLM_TEMPERATURE_SUMMARY",
    "LLM_TEMPERATURE_PLAN",
    "LLM_TEMPERATURE_CRITIQUE",
    "LLM_TEMPERATURE_REVIEW",
    "LLM_TEMPERATURE_REVISE",
    "LLM_TEMPERATURE_MANUSCRIPT",
    "LLM_MAX_TOKENS_OUTLINE",
    "LLM_MAX_TOKENS_ABSTRACT",
    "LLM_MAX_TOKENS_SUMMARY",
    "LLM_MAX_TOKENS_REVIEW",
    "LLM_MAX_TOKENS_CRITIQUE_BASE",
    "LLM_MAX_TOKENS_CRITIQUE_PER_PAPER",
    "LLM_MAX_TOKENS_CRITIQUE_CAP",
    "LLM_MAX_TOKENS_PLAN",
    "LLM_MAX_TOKENS_MANUSCRIPT",
    "LLM_MAX_TOKENS_CAP",
    # 文本裁剪上限
    "ABSTRACT_TRUNCATE_DETECT",
    "ABSTRACT_TRUNCATE_SAMPLE",
    "DIGEST_MAX_ABSTRACT_OUTLINE",
    "DIGEST_MAX_ABSTRACT_CRITIQUE",
    "DIGEST_MAX_ABSTRACT_REVIEW",
    "DIGEST_MAX_ABSTRACT_REVISE",
    "DRAFT_TRUNCATE_REVIEW",
    "DRAFT_TRUNCATE_REVISE",
    "BODY_TRUNCATE_SUMMARY",
    "TITLE_TRUNCATE_SESSION",
    "TITLE_TRUNCATE_TOPIC_LOG",
    "TITLE_TRUNCATE_FULLTEXT",
    "TITLE_TRUNCATE_SUGGESTION",
    "FEEDBACK_TRUNCATE",
    "ERROR_TRUNCATE_LLM",
    # 评分阈值（Critic 启发式）
    "SCORE_MAX",
    "SAMPLE_EXTRACT_MIN",
    "SAMPLE_SIZE_LARGE",
    "SAMPLE_SIZE_MEDIUM",
    "SAMPLE_SIZE_SMALL",
    "SAMPLE_SIZE_MIN",
    "SAMPLE_MAX",
    "CITED_LARGE",
    "CITED_MEDIUM",
    "CITED_SMALL",
    "AGE_RECENT",
    "AGE_OLD",
    "QUALITY_ADJUST_LARGE_SAMPLE",
    "QUALITY_ADJUST_MEDIUM_SAMPLE",
    "QUALITY_ADJUST_SMALL_SAMPLE",
    "QUALITY_PENALTY_TINY_SAMPLE",
    "QUALITY_ADJUST_LARGE_CITED",
    "QUALITY_ADJUST_MEDIUM_CITED",
    "QUALITY_ADJUST_SMALL_CITED",
    "QUALITY_PENALTY_NO_CITATION",
    "QUALITY_ADJUST_RECENT",
    "QUALITY_PENALTY_OLD",
    "QUALITY_PENALTY_NO_ABSTRACT",
    "RELEVANCE_BASE",
    "RELEVANCE_RANGE",
    "RELEVANCE_BONUS",
    "USE_IN_REVIEW_THRESHOLD",
    "EVIDENCE_HIGH_QUALITY",
    "EVIDENCE_HIGH_RCT_COUNT",
    "EVIDENCE_MEDIUM_QUALITY",
    "EVIDENCE_LOW_QUALITY",
    # Writer 预算
    "TOKEN_RESERVE",
    "MIN_BUDGET",
    "MIN_HEADROOM",
    "MIN_TOKEN_CAP",
    "MAX_TOKEN_CAP",
    "SECTION_MIN_CHARS",
    "SECTION_CHARS_GAP",
    "CHARS_PER_TOKEN",
    "TOKEN_SAFETY_MARGIN",
    "CITATION_RANGE_MAX_SPAN",
    "REVISE_MIN_LENGTH_RATIO",
    "DEFAULT_SECTION_COUNT",
    # 运行时
    "RUN_RETENTION_SECONDS",
    "EVENT_POLL_SECONDS",
    "HEARTBEAT_SECONDS",
    "STREAM_TIMEOUT_SECONDS",
    "REVIEW_CHARS_MIN",
    "REVIEW_CHARS_MAX",
    "REVIEW_CHARS_DEFAULT_MIN",
    "REVIEW_CHARS_DEFAULT_MAX",
    "REVIEW_CHARS_HIGH_FLOOR",
    "REVIEW_CHARS_GAP",
    "REVIEW_MIN_TO_MAX_RATIO",
    "REVIEW_MAX_TO_MIN_RATIO",
    "RUNS_LIST_LIMIT",
    "MAX_ERRORS_SNAPSHOT",
    "MAX_ERRORS_PERSIST",
    "MAX_ISSUES_REVISE",
    # 数据库
    "DB_BUSY_TIMEOUT_MS",
    "DB_CACHE_SIZE_KB",
    "DB_CONNECTION_TIMEOUT",
    # HTTP / PDF
    "HTTP_CONNECT_TIMEOUT",
    "PDF_DOWNLOAD_TIMEOUT",
    "PDF_CONNECT_TIMEOUT",
    "PDF_MAGIC_BYTES",
    "MAX_UPLOAD_BYTES",
    # 检索默认值
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_TOP_K",
    "DEFAULT_REFERENCE_LIMIT",
]

# ============================================================ LLM 生成参数
#: 大纲细化：温度低一点保证结构稳定
LLM_TEMPERATURE_OUTLINE: float = 0.2
#: 综述正文：温度适中，兼顾流畅与稳定
LLM_TEMPERATURE_SECTION: float = 0.35
#: 摘要生成
LLM_TEMPERATURE_ABSTRACT: float = 0.3
#: 单篇速读 / 全文速读
LLM_TEMPERATURE_SUMMARY: float = 0.2
#: 检索规划
LLM_TEMPERATURE_PLAN: float = 0.25
#: 文献评估（要客观，温度要低）
LLM_TEMPERATURE_CRITIQUE: float = 0.1
#: 自我审查
LLM_TEMPERATURE_REVIEW: float = 0.1
#: 自动修订
LLM_TEMPERATURE_REVISE: float = 0.2
#: 论文撰写
LLM_TEMPERATURE_MANUSCRIPT: float = 0.3

#: 大纲生成的 token 上限
LLM_MAX_TOKENS_OUTLINE: int = 1200
#: 摘要生成的 token 上限
LLM_MAX_TOKENS_ABSTRACT: int = 600
#: 单篇速读的 token 上限
LLM_MAX_TOKENS_SUMMARY: int = 900
#: 自我审查的 token 上限
LLM_MAX_TOKENS_REVIEW: int = 1200
#: LLM 评估的基础 token 数（不含逐篇增量）
LLM_MAX_TOKENS_CRITIQUE_BASE: int = 400
#: LLM 评估每篇文献的增量 token
LLM_MAX_TOKENS_CRITIQUE_PER_PAPER: int = 320
#: LLM 评估的 token 上限
LLM_MAX_TOKENS_CRITIQUE_CAP: int = 3000
#: 规划的 token 上限
LLM_MAX_TOKENS_PLAN: int = 1800
#: 论文各节的 token 上限
LLM_MAX_TOKENS_MANUSCRIPT: int = 2000
#: 单节生成的硬上限（防止极端配置把上下文吃光）
LLM_MAX_TOKENS_CAP: int = 6144

# ============================================================ 文本裁剪上限
#: 证据等级检测时摘要裁剪长度
ABSTRACT_TRUNCATE_DETECT: int = 1500
#: 样本量提取时标题+摘要裁剪长度
ABSTRACT_TRUNCATE_SAMPLE: int = 4000
#: 大纲细化时材料摘要长度
DIGEST_MAX_ABSTRACT_OUTLINE: int = 280
#: LLM 评估时材料摘要长度
DIGEST_MAX_ABSTRACT_CRITIQUE: int = 600
#: 自我审查时材料摘要长度
DIGEST_MAX_ABSTRACT_REVIEW: int = 400
#: 自动修订时材料摘要长度
DIGEST_MAX_ABSTRACT_REVISE: int = 350
#: 自我审查时草稿裁剪长度
DRAFT_TRUNCATE_REVIEW: int = 6000
#: 自动修订时草稿裁剪长度
DRAFT_TRUNCATE_REVISE: int = 8000
#: 速读时正文裁剪长度
BODY_TRUNCATE_SUMMARY: int = 12000
#: 会话标题裁剪长度
TITLE_TRUNCATE_SESSION: int = 40
#: 日志中课题裁剪长度
TITLE_TRUNCATE_TOPIC_LOG: int = 60
#: 全文获取消息中标题裁剪长度
TITLE_TRUNCATE_FULLTEXT: int = 48
#: 建议列表中标题裁剪长度
TITLE_TRUNCATE_SUGGESTION: int = 36
#: 审批反馈裁剪长度
FEEDBACK_TRUNCATE: int = 80
#: LLM 错误信息裁剪长度
ERROR_TRUNCATE_LLM: int = 60

# ============================================================ 评分阈值
#: 质量/相关性分数上限（0~10 分制）
SCORE_MAX: float = 10.0

# --- 样本量分档
SAMPLE_EXTRACT_MIN: int = 5     # 提取时的合理下限（过滤页码等小数字）
SAMPLE_SIZE_LARGE: int = 1000   # >= 1000 大样本
SAMPLE_SIZE_MEDIUM: int = 300   # >= 300 中样本
SAMPLE_SIZE_SMALL: int = 100    # >= 100 小样本
SAMPLE_SIZE_MIN: int = 30       # < 30 样本量偏小
SAMPLE_MAX: int = 500_000       # 样本量合理上限（过滤年份等误提取）

# --- 被引次数分档
CITED_LARGE: int = 500
CITED_MEDIUM: int = 100
CITED_SMALL: int = 30

# --- 时效性（年）
AGE_RECENT: int = 3    # <= 3 年算近期
AGE_OLD: int = 12      # >= 12 年算老旧

# --- 质量加减分
QUALITY_ADJUST_LARGE_SAMPLE: float = 1.2
QUALITY_ADJUST_MEDIUM_SAMPLE: float = 0.8
QUALITY_ADJUST_SMALL_SAMPLE: float = 0.4
QUALITY_PENALTY_TINY_SAMPLE: float = -0.8
QUALITY_ADJUST_LARGE_CITED: float = 1.2
QUALITY_ADJUST_MEDIUM_CITED: float = 0.8
QUALITY_ADJUST_SMALL_CITED: float = 0.4
QUALITY_PENALTY_NO_CITATION: float = -0.3
QUALITY_ADJUST_RECENT: float = 0.5
QUALITY_PENALTY_OLD: float = -0.6
QUALITY_PENALTY_NO_ABSTRACT: float = -1.2

# --- 相关性
RELEVANCE_BASE: float = 3.0
RELEVANCE_RANGE: float = 7.0
RELEVANCE_BONUS: float = 0.8
USE_IN_REVIEW_THRESHOLD: float = 4.0

# --- 整体证据等级
EVIDENCE_HIGH_QUALITY: float = 7.5
EVIDENCE_HIGH_RCT_COUNT: int = 3
EVIDENCE_MEDIUM_QUALITY: float = 6.0
EVIDENCE_LOW_QUALITY: float = 4.0

# ============================================================ Writer 预算
#: 给系统提示词和不确定开销预留的 token
TOKEN_RESERVE: int = 256
#: 材料预算下限（低于这个数材料就没意义了）
MIN_BUDGET: int = 900
#: 输出空间下限
MIN_HEADROOM: int = 600
#: 单节 token 下限
MIN_TOKEN_CAP: int = 600
#: 单节 token 硬上限
MAX_TOKEN_CAP: int = LLM_MAX_TOKENS_CAP
#: 单节最少字数（低于这个写不出有效内容）
SECTION_MIN_CHARS: int = 200
#: 单节字数上下限之差
SECTION_CHARS_GAP: int = 150
#: 实测 qwen3 中文约 1.7 字/token，这里用偏保守的 1.5
CHARS_PER_TOKEN: float = 1.5
#: token 估算的安全余量
TOKEN_SAFETY_MARGIN: int = 200
#: 引用编号范围最大跨度（[1-51] 这种超长范围视为异常）
CITATION_RANGE_MAX_SPAN: int = 50
#: 自动修订结果至少要达到原草稿的比例（低于则视为不完整）
REVISE_MIN_LENGTH_RATIO: float = 0.5
#: 兜底大纲的章节数
DEFAULT_SECTION_COUNT: int = 5

# ============================================================ 运行时
#: 已完成运行在内存中的保留时长（秒）
RUN_RETENTION_SECONDS: int = 3600
#: 事件轮询间隔（秒）
EVENT_POLL_SECONDS: float = 0.2
#: SSE 心跳间隔（秒）
HEARTBEAT_SECONDS: float = 15.0
#: 事件流超时（秒）
STREAM_TIMEOUT_SECONDS: float = 3600.0

#: 综述字数下限（比这个更短写不出综述）
REVIEW_CHARS_MIN: int = 800
#: 综述字数上限（本地模型一次生成不现实）
REVIEW_CHARS_MAX: int = 40000
#: 默认综述字数下限
REVIEW_CHARS_DEFAULT_MIN: int = 4000
#: 默认综述字数上限
REVIEW_CHARS_DEFAULT_MAX: int = 8000
#: high 下限夹取时的地板
REVIEW_CHARS_HIGH_FLOOR: int = 1000
#: min < max 时保证的最小间隔
REVIEW_CHARS_GAP: int = 200
#: 只有上限时，下限 = 上限 * 该比例
REVIEW_MIN_TO_MAX_RATIO: float = 0.6
#: 只有下限时，上限 = 下限 * 该比例
REVIEW_MAX_TO_MIN_RATIO: float = 1.6

#: 运行列表默认返回条数
RUNS_LIST_LIMIT: int = 30
#: 快照中保留的错误条数
MAX_ERRORS_SNAPSHOT: int = 5
#: 落库时拼接的错误条数
MAX_ERRORS_PERSIST: int = 3
#: 自动修订时取的问题条数
MAX_ISSUES_REVISE: int = 8

# ============================================================ 数据库
#: SQLite busy_timeout（毫秒）—— WAL 模式下读写等待
DB_BUSY_TIMEOUT_MS: int = 15000
#: SQLite page cache 大小（KB，负数表示 KB）
DB_CACHE_SIZE_KB: int = -32000
#: SQLite 连接超时（秒）
DB_CONNECTION_TIMEOUT: float = 30.0

# ============================================================ HTTP / PDF
#: HTTP 连接超时（秒）—— 建立 TCP 连接的最长等待时间
HTTP_CONNECT_TIMEOUT: float = 15.0
#: PDF 下载总超时（秒）
PDF_DOWNLOAD_TIMEOUT: float = 90.0
#: PDF 连接超时（秒）—— 复用 HTTP 连接超时
PDF_CONNECT_TIMEOUT: float = HTTP_CONNECT_TIMEOUT
#: PDF 魔数前缀长度
PDF_MAGIC_BYTES: int = 5
#: 上传文件大小上限（64 MB）
MAX_UPLOAD_BYTES: int = 64 * 1024 * 1024

# ============================================================ 检索默认值
#: 检索默认返回条数
DEFAULT_SEARCH_LIMIT: int = 20
#: 检索默认 top_k
DEFAULT_TOP_K: int = 20
#: 引用获取默认条数
DEFAULT_REFERENCE_LIMIT: int = 50
