"""配置加载。

优先级（后者覆盖前者）：
    内置默认值  <  ``config.yaml``  <  环境变量 / ``.env``

数据目录（数据库、全文、导出物）默认落在**项目目录下的 data/**，
这样整包解压后拷给朋友即可使用；若本包是被 ``pip install`` 到 site-packages 的，
则自动退回到 ``~/.medscholar``。可用 ``MEDSCHOLAR_HOME`` 显式指定。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "AppConfig",
    "SourceSettings",
    "get_config",
    "reload_config",
    "project_root",
    "data_home",
    "config_path",
    "DEFAULT_CONFIG_FILENAME",
]

DEFAULT_CONFIG_FILENAME = "config.yaml"


# --------------------------------------------------------------------- 路径
#: 源码/解压包根目录的"标记文件"：有这个文件就说明这一层是项目根。
#: 用标记而不是写死 `Path(__file__).parent.parent`，是因为层级会随重构变化：
#: 这个模块从 `medscholar/config.py` 搬到 `medscholar/platform/config.py` 之后，
#: 写死的 `.parent.parent` 就少了一层——源码树判定失败 → `data_home()` 悄悄退回到
#: `~/.medscholar` → 用户打开应用看到"知识库中还没有文献"（真实事故：仓库里的 587 篇
#: 文献读不到了，而且没有任何报错）。
ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", "run.bat", "config.example.yaml")


def project_root() -> Path:
    """返回项目根目录（源码/解压包目录）。

    实现是**从本文件向上找标记文件**，而不是数固定的层数：
    这样以后再挪动模块位置也不会静默失效。找不到标记时退回"上两级"的历史行为。
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
        if candidate == candidate.parent:  # 到达盘符根，停止
            break
    return here.parent.parent


def _is_source_tree(root: Path) -> bool:
    return any((root / marker).exists() for marker in ROOT_MARKERS)


def data_home() -> Path:
    """数据目录：数据库、全文缓存、导出文件都放这里。"""
    env = os.environ.get("MEDSCHOLAR_HOME")
    if env:
        return Path(env).expanduser().resolve()
    root = project_root()
    if _is_source_tree(root):
        return root / "data"
    return Path.home() / ".medscholar"


def config_path() -> Path:
    """配置文件位置。"""
    env = os.environ.get("MEDSCHOLAR_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    home_cfg = data_home() / DEFAULT_CONFIG_FILENAME
    if home_cfg.exists():
        return home_cfg
    return project_root() / DEFAULT_CONFIG_FILENAME


# ----------------------------------------------------------------- 配置模型
class SourceSettings(BaseModel):
    """单个学术数据源的设置。"""

    enabled: bool = True
    api_key: str = ""
    email: str = ""          # NCBI / OpenAlex 的 polite pool 联系邮箱
    rps: float = 3.0         # 每秒请求数上限
    timeout: float = 30.0
    retries: int = 3
    backoff: float = 0.8     # 指数退避基数（秒）
    page_size: int = 50
    max_results: int = 100

    @field_validator("rps")
    @classmethod
    def _positive_rps(cls, v: float) -> float:
        return max(0.05, float(v))


class SourcesSettings(BaseModel):
    """全部数据源设置。默认速率严格贴合各 API 的公开限制。"""

    pubmed: SourceSettings = Field(
        default_factory=lambda: SourceSettings(rps=3.0, email="", page_size=100, max_results=100)
    )
    europepmc: SourceSettings = Field(
        default_factory=lambda: SourceSettings(rps=8.0, page_size=25, max_results=100)
    )
    semantic_scholar: SourceSettings = Field(
        default_factory=lambda: SourceSettings(rps=0.3, page_size=20, max_results=100, retries=2, backoff=1.5)
    )
    openalex: SourceSettings = Field(
        default_factory=lambda: SourceSettings(rps=8.0, email="", page_size=50, max_results=100)
    )
    crossref: SourceSettings = Field(
        default_factory=lambda: SourceSettings(rps=5.0, email="", page_size=50, max_results=100)
    )
    arxiv: SourceSettings = Field(
        default_factory=lambda: SourceSettings(rps=0.33, page_size=50, max_results=50)
    )
    #: CNKI 公开检索页已改为 JS 渲染，服务端 HTML 不含文献条目（见 api/cnki_client.py）
    cnki: SourceSettings = Field(
        default_factory=lambda: SourceSettings(enabled=False, rps=0.2, page_size=20, max_results=40)
    )
    #: Unpaywall：不参与关键词检索，只在取全文时按 DOI 查合法 OA 副本。
    #: 必须填 email（免费，Unpaywall 用它识别调用方）。
    unpaywall: SourceSettings = Field(
        default_factory=lambda: SourceSettings(enabled=True, rps=2.0, timeout=20.0, retries=2)
    )
    #: DOAJ：开放获取期刊论文检索，免费无需 key。
    doaj: SourceSettings = Field(
        default_factory=lambda: SourceSettings(enabled=True, rps=2.0, page_size=50, max_results=100)
    )
    #: CORE：聚合全球机构库，需要免费 API key（https://core.ac.uk/services/api）。
    core: SourceSettings = Field(
        default_factory=lambda: SourceSettings(enabled=False, rps=1.0, page_size=50, max_results=100)
    )

    def as_dict(self) -> dict[str, SourceSettings]:
        return {
            "pubmed": self.pubmed,
            "europepmc": self.europepmc,
            "semantic_scholar": self.semantic_scholar,
            "openalex": self.openalex,
            "crossref": self.crossref,
            "arxiv": self.arxiv,
            "cnki": self.cnki,
            "unpaywall": self.unpaywall,
            "doaj": self.doaj,
            "core": self.core,
        }

    def get(self, name: str) -> SourceSettings:
        """按数据源短名取设置；``s2`` 是 ``semantic_scholar`` 的别名。"""
        key = {"s2": "semantic_scholar", "semanticscholar": "semantic_scholar"}.get(
            name, name
        )
        return self.as_dict().get(key, SourceSettings())


class EmbeddingSettings(BaseModel):
    """向量嵌入设置。

    provider:
        ``ollama``                 本地 Ollama 嵌入模型（默认，零成本、离线可用）
        ``sentence-transformers``  本地 PubMedBERT（需装 ``[local-embed]`` 额外依赖）
        ``hashing``                纯 Python 哈希嵌入，仅供无模型环境下的冒烟测试
    """

    provider: Literal["ollama", "sentence-transformers", "hashing"] = "ollama"
    model: str = "nomic-embed-text:latest"
    dim: int = 768
    base_url: str = "http://127.0.0.1:11434"
    batch_size: int = 16
    timeout: float = 120.0
    max_chars: int = 6000      # 单篇送嵌入的最大字符数
    auto_embed: bool = True    # 新文献入库后自动补嵌入
    idle_seconds: float = 0.0  # 每批之间的间隔，避免打满 CPU


class LLMSettings(BaseModel):
    """推理模型设置。

    provider:
        ``ollama``            本地 Ollama（默认）
        ``deepseek``          DeepSeek 云端 API（OpenAI 兼容）
        ``openai-compatible`` 任意 OpenAI 兼容端点（vLLM / One-API / 硅基流动等）
    """

    provider: Literal["ollama", "deepseek", "openai-compatible"] = "ollama"
    model: str = "qwen3:8b"
    base_url: str = "http://127.0.0.1:11434"
    api_key: str = ""
    temperature: float = 0.3
    top_p: float = 0.9
    #: 单次输出上限。推理模型的思维链与正文共用这个配额（DeepSeek 的
    #: deepseek-flash / deepseek-v4-pro 都属于推理模型），设得太小会导致
    #: 模型把预算全花在思考上、正文返回空字符串。用云端推理模型时建议 >= 4000。
    max_tokens: int = 3000
    timeout: float = 600.0
    think: bool = False          # Qwen3 等推理模型的思考开关
    num_ctx: int = 8192
    #: Ollama 模型在显存里的保留时长。Ollama 默认只有 5 分钟，一旦超时卸载，
    #: 下次调用要重新加载——实测 8B 模型重新载入要 60~120 秒，比生成还慢。
    #: 设成 "-1" 表示常驻不卸载（显存够用时最省时间）。
    keep_alive: str = "30m"
    failure_hint: str = ""       # 模型不可用时展示给用户的提示

    def consistency_error(self) -> str:
        """检查 provider 与 model 是否自洽，不自洽时返回可照做的中文说明。

        这是为了根治一类真实故障：``provider: deepseek`` 却配着
        ``model: qwen3:8b``（Ollama 的模型名），请求发到云端必然 400，
        而报错信息是英文的、且不会告诉你该怎么改。
        """
        name = (self.model or "").strip()
        if not name:
            return "llm.model 未配置。"

        if self.provider in {"deepseek", "openai-compatible"} and _looks_like_ollama_model(name):
            return (
                f"配置不一致：llm.provider 是「{self.provider}」（云端 API），"
                f"但 llm.model 是「{name}」，看起来是 Ollama 的本地模型名。\n"
                f"    云端 API 不认识本地模型名，会直接返回 HTTP 400。\n"
                f"    两种改法（二选一）：\n"
                f"      · 想用本地模型 → 把 llm.provider 改回 ollama，"
                f"base_url 保持 http://127.0.0.1:11434\n"
                f"      · 想用云端     → 把 llm.model 改成该服务商支持的模型名"
                f"（例如 deepseek-chat / deepseek-flash），"
                f"并把 llm.base_url 指向对应网关"
            )

        if self.provider == "ollama" and not _looks_like_ollama_model(name):
            return (
                f"配置可能不一致：llm.provider 是 ollama，但 llm.model 是「{name}」，"
                f"不像 Ollama 的模型名（Ollama 通常形如 qwen3:8b）。\n"
                f"    请确认已执行 `ollama pull {name}`，或把 llm.model 改成已安装的模型。"
            )
        return ""


def _looks_like_ollama_model(name: str) -> bool:
    """判断模型名是否像 Ollama 的本地模型标签（``name:tag`` 或 ``xxx-7b``）。

    >>> _looks_like_ollama_model("qwen3:8b")
    True
    >>> _looks_like_ollama_model("deepseek-chat")
    False
    """
    if ":" in name:
        return True
    return bool(re.search(r"[-_]\d+(\.\d+)?b$", name, re.IGNORECASE))


class RetrievalSettings(BaseModel):
    """混合检索设置。"""

    rrf_k: int = 60              # RRF 融合常数（论文推荐 60）
    fts_candidates: int = 100    # BM25 召回条数
    vector_candidates: int = 100  # 向量召回条数
    top_k: int = 20              # 融合后返回条数
    bm25_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "title": 8.0,
            "mesh_terms": 3.0,
            "keywords": 3.0,
            "abstract": 1.0,
            "authors": 0.5,
            "journal": 0.5,
        }
    )
    min_score: float = 0.0


class AgentSettings(BaseModel):
    """Agent 工作流设置。"""

    max_search_rounds: int = 2
    max_papers_per_source: int = 30
    reflect_enabled: bool = True
    require_approval: bool = True   # Plan 后是否等待用户审批
    auto_embed_after_search: bool = True
    writer_max_papers: int = 25     # 送入写作上下文的文献上限
    #: 交给 LLM 逐篇点评的文献数上限。本地 8B 模型在 CPU 上约 7 tokens/s，
    #: 评 25 篇要生成上千 token（数分钟），因此默认只让 LLM 评最相关的若干篇，
    #: 其余用启发式评分兜底（两者等权融合，见 CriticAgent.assess）。
    critique_max_papers: int = 12
    context_char_budget: int = 24000
    warm_fulltext: bool = True      # 是否为开放获取文献预取全文
    fulltext_top_n: int = 4         # 预取全文的文献数（抓取较慢，不宜过多）
    auto_revise: bool = True        # 自我审查发现问题后是否自动修订一轮
    max_revise_rounds: int = 1
    #: 综述正文的目标字数范围（中文字符计，不含参考文献）。
    #: 这是写作提示词里的硬指令：以前每节固定"约 900 字"，5 节只有 4500 字左右，
    #: 用户普遍反馈太短。改为按总字数范围反推每节目标。
    #: 注意：字数越大，单节生成时间越长（本地 8B 约 45 tok/s，约 1.7 字/token），
    #: 而且为了给正文腾出上下文，塞进提示词的材料会被自动裁剪。
    review_min_chars: int = 4000
    review_max_chars: int = 8000


class ServerSettings(BaseModel):
    """Web 服务设置。"""

    host: str = "127.0.0.1"
    port: int = 8760
    open_browser: bool = True
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:8760", "http://localhost:8760"]
    )


class AppConfig(BaseModel):
    """应用总配置。"""

    app_name: str = "MedScholar Agent"
    language: str = "zh-CN"
    offline: bool = False        # True 时只用本地知识库，完全不联网
    data_dir: str = ""
    db_filename: str = "medscholar.db"

    sources: SourcesSettings = Field(default_factory=SourcesSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    # ------------------------------------------------------------- 计算属性
    @property
    def home(self) -> Path:
        return Path(self.data_dir).expanduser().resolve() if self.data_dir else data_home()

    @property
    def db_path(self) -> Path:
        return self.home / self.db_filename

    @property
    def fulltext_dir(self) -> Path:
        return self.home / "fulltext"

    @property
    def export_dir(self) -> Path:
        return self.home / "exports"

    @property
    def upload_dir(self) -> Path:
        return self.home / "uploads"

    def ensure_dirs(self) -> None:
        for path in (
            self.home,
            self.fulltext_dir,
            self.export_dir,
            self.upload_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, Any]:
        """给前端用的配置摘要（**不包含任何 API Key 明文**）。"""
        return {
            "app_name": self.app_name,
            "language": self.language,
            "offline": self.offline,
            "db_path": str(self.db_path),
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "has_api_key": bool(self.llm.api_key),
            },
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dim": self.embedding.dim,
            },
            "retrieval": {
                "rrf_k": self.retrieval.rrf_k,
                "top_k": self.retrieval.top_k,
            },
            "sources": {
                name: {
                    "enabled": s.enabled,
                    "has_api_key": bool(s.api_key),
                    "rps": s.rps,
                }
                for name, s in self.sources.as_dict().items()
            },
        }


# ----------------------------------------------------------------- 加载逻辑
def _load_dotenv(path: Path) -> None:
    """极简 .env 解析（不引入额外依赖）。已有环境变量优先，不被覆盖。"""
    if not path.exists():
        return
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """把环境变量合并进配置字典。"""

    def setdefault_path(path: tuple[str, ...], value: str | None) -> None:
        if not value:
            return
        node = cfg
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    def to_bool(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    # 数据源 Key
    setdefault_path(("sources", "pubmed", "api_key"), os.environ.get("NCBI_API_KEY"))
    setdefault_path(("sources", "pubmed", "email"), os.environ.get("NCBI_EMAIL"))
    setdefault_path(
        ("sources", "semantic_scholar", "api_key"), os.environ.get("S2_API_KEY")
    )
    setdefault_path(("sources", "openalex", "api_key"), os.environ.get("OPENALEX_API_KEY"))
    setdefault_path(("sources", "openalex", "email"), os.environ.get("OPENALEX_EMAIL"))

    # LLM
    #
    # 这里刻意不根据 DEEPSEEK_API_KEY 自动切换 provider。
    # 曾经这么做过，结果是灾难性的：用户机器上另一个项目把 DEEPSEEK_API_KEY
    # 设成了用户级环境变量，于是 MedScholar 静默地把后端从 ollama 切到了 deepseek，
    # 却沿用了 config.yaml 里的 model: qwen3:8b，最终把 Ollama 的模型名发给了
    # DeepSeek 网关，得到 HTTP 400。
    # 结论：环境里存在某个 Key，不等于用户想让本程序用它。
    # Key 只在用户显式把 provider 设为 deepseek/openai-compatible 后才生效。
    # 想切云端，请改 config.yaml 的 llm.provider（本文件下方会做一致性校验）。
    provider = os.environ.get("MEDSCHOLAR_LLM_PROVIDER")
    if provider:
        cfg.setdefault("llm", {})["provider"] = provider
    if os.environ.get("DEEPSEEK_API_KEY"):
        cfg.setdefault("llm", {})["api_key"] = os.environ["DEEPSEEK_API_KEY"]
    if os.environ.get("DEEPSEEK_BASE_URL"):
        cfg.setdefault("llm", {})["base_url"] = os.environ["DEEPSEEK_BASE_URL"]
    setdefault_path(("llm", "model"), os.environ.get("MEDSCHOLAR_LLM_MODEL"))
    setdefault_path(("llm", "base_url"), os.environ.get("OLLAMA_HOST"))
    setdefault_path(("embedding", "base_url"), os.environ.get("OLLAMA_HOST"))
    setdefault_path(("embedding", "model"), os.environ.get("MEDSCHOLAR_EMBED_MODEL"))
    setdefault_path(("embedding", "provider"), os.environ.get("MEDSCHOLAR_EMBED_PROVIDER"))

    # 运行模式
    if os.environ.get("MEDSCHOLAR_OFFLINE"):
        cfg["offline"] = to_bool(os.environ["MEDSCHOLAR_OFFLINE"])
    if os.environ.get("MEDSCHOLAR_DATA_DIR"):
        cfg["data_dir"] = os.environ["MEDSCHOLAR_DATA_DIR"]
    if os.environ.get("MEDSCHOLAR_PORT"):
        try:
            cfg.setdefault("server", {})["port"] = int(os.environ["MEDSCHOLAR_PORT"])
        except ValueError:
            pass
    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，override 优先。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - 用户配置损坏
        raise RuntimeError(f"配置文件解析失败：{path}\n{exc}") from exc
    return raw if isinstance(raw, dict) else {}


_CONFIG: AppConfig | None = None


def get_config(*, refresh: bool = False) -> AppConfig:
    """获取全局配置单例（首次调用时加载）。"""
    global _CONFIG
    if _CONFIG is None or refresh:
        _CONFIG = _build_config()
    return _CONFIG


def reload_config() -> AppConfig:
    """强制重新读取配置文件与环境变量。"""
    return get_config(refresh=True)


def _build_config() -> AppConfig:
    root = project_root()
    _load_dotenv(root / ".env")
    _load_dotenv(data_home() / ".env")

    merged: dict[str, Any] = {}
    for candidate in (root / DEFAULT_CONFIG_FILENAME, config_path()):
        merged = _deep_merge(merged, _load_file(candidate))
    merged = _env_overrides(merged)

    cfg = AppConfig.model_validate(merged)
    cfg.ensure_dirs()
    return cfg
