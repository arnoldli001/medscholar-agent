-- MedScholar Agent 数据库结构（SQLite 3.35+，需 FTS5）
--
-- 设计说明：
--   * papers 为唯一事实来源；paper_embeddings 可以是 vec0 虚拟表（sqlite-vec 可用时）
--     也可以是普通表（纯 Python 暴力检索回退），两张表同名不同实现，由连接层决定。
--   * papers_fts / fulltext_fts 为**独立** FTS5 表（非 external-content），
--     因为中文需要在写入前做逐字切分，SQL 触发器无法完成该转换。
--     其内容由 medscholar.db.repo 中的 _fts_sync() 统一维护。

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------------ 元信息
CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- -------------------------------------------------------------------- 文献
CREATE TABLE IF NOT EXISTS papers (
    paper_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid             TEXT UNIQUE,
    pmcid            TEXT,
    doi              TEXT UNIQUE,
    title            TEXT NOT NULL,
    abstract         TEXT,
    authors          TEXT NOT NULL DEFAULT '[]',   -- JSON 数组
    journal          TEXT,
    pub_year         INTEGER,
    source           TEXT NOT NULL DEFAULT 'manual',
    source_id        TEXT,                          -- 数据源内部 ID（S2 paperId / OpenAlex id）
    mesh_terms       TEXT NOT NULL DEFAULT '[]',   -- JSON 数组
    keywords         TEXT NOT NULL DEFAULT '[]',   -- JSON 数组
    cited_by_count   INTEGER NOT NULL DEFAULT 0,
    is_open_access   INTEGER NOT NULL DEFAULT 0,
    full_text_path   TEXT,                          -- 本地 PDF/XML 路径
    full_text_url    TEXT,                          -- 开放获取全文链接
    url              TEXT,
    volume           TEXT,
    issue            TEXT,
    pages            TEXT,
    publication_type TEXT,
    language         TEXT,
    title_key        TEXT,                          -- 标题指纹，用于无 DOI/PMID 时去重
    note             TEXT,                          -- 用户批注
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_papers_year    ON papers(pub_year DESC);
CREATE INDEX IF NOT EXISTS idx_papers_source  ON papers(source);
CREATE INDEX IF NOT EXISTS idx_papers_cited   ON papers(cited_by_count DESC);
CREATE INDEX IF NOT EXISTS idx_papers_titlekey ON papers(title_key);
CREATE INDEX IF NOT EXISTS idx_papers_oa      ON papers(is_open_access);
CREATE INDEX IF NOT EXISTS idx_papers_pmcid   ON papers(pmcid);

-- ------------------------------------------------- 标题/摘要全文索引（FTS5）
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    paper_id   UNINDEXED,
    title,
    abstract,
    authors,
    journal,
    mesh_terms,
    keywords,
    tokenize = "unicode61 remove_diacritics 2"
);

-- ------------------------------------------------------- 开放获取全文（FTS5）
CREATE TABLE IF NOT EXISTS paper_fulltext (
    paper_id   INTEGER PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    content    TEXT NOT NULL,
    char_count INTEGER NOT NULL DEFAULT 0,
    origin     TEXT,                                -- europepmc / pdf / upload
    source_url TEXT,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ------------------------------------------------ 全文抓取尝试记录（避免白费功夫）
--
-- 为什么需要它：开放获取全文的失败绝大多数是**永久性**的 ——
-- 会议摘要/勘误/社论在 PMC 里本来就没有正文，出版商也长期拒绝自动下载。
-- 不记录的话，用户每点一次「补齐全文」都会把这些死条目重新试一遍，
-- 白等好几分钟。实测某次「失败 100 条」中 48% 属于"文献本身没有正文"。
CREATE TABLE IF NOT EXISTS fulltext_attempts (
    paper_id   INTEGER PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    attempts   INTEGER NOT NULL DEFAULT 1,
    last_error TEXT NOT NULL DEFAULT '',
    permanent  INTEGER NOT NULL DEFAULT 0,   -- 1 = 确定取不到，不再重试
    checked_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS fulltext_fts USING fts5(
    paper_id UNINDEXED,
    content,
    tokenize = "unicode61 remove_diacritics 2"
);

-- ------------------------------------------------------------------ 引用关系
CREATE TABLE IF NOT EXISTS citations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    citing_paper_id   INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    cited_paper_id    INTEGER REFERENCES papers(paper_id) ON DELETE CASCADE,
    cited_external_id TEXT,        -- 被引文献不在本地库时，记录其 DOI / PMID / S2 ID
    cited_title       TEXT,
    source            TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (citing_paper_id, cited_paper_id, cited_external_id)
);

CREATE INDEX IF NOT EXISTS idx_citations_cited  ON citations(cited_paper_id);
CREATE INDEX IF NOT EXISTS idx_citations_ext    ON citations(cited_external_id);

-- ------------------------------------------------------------------ 检索历史
CREATE TABLE IF NOT EXISTS search_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query        TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT '',
    result_count INTEGER NOT NULL DEFAULT 0,
    new_count    INTEGER NOT NULL DEFAULT 0,
    duration_ms  INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_search_logs_time ON search_logs(created_at DESC);

-- --------------------------------------------------------------- 课题 / 分类
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    keywords    TEXT NOT NULL DEFAULT '[]',   -- 订阅关键词（JSON 数组）
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS project_papers (
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    paper_id   INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    note       TEXT NOT NULL DEFAULT '',
    added_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, paper_id)
);

-- --------------------------------------------------------------- 会话与消息
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL DEFAULT '新会话',
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    topic      TEXT NOT NULL DEFAULT '',       -- 原始课题描述
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,                  -- user / assistant / system / tool
    content    TEXT NOT NULL DEFAULT '',
    meta       TEXT NOT NULL DEFAULT '{}',     -- JSON：工具调用、引用、阶段等
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id);

-- ------------------------------------------------- Agent 运行记录（跨重启可追溯）
--
-- 为什么需要它：运行状态原本只存在内存里，服务一重启就全丢，
-- 而前端仍在等一个永远不会到来的「artifact」事件，用户看到的是
-- "阶段都走完了但草稿是空的"，完全无从判断。
-- 实测就踩到过：用户 17:00 发起研究，17:10 服务重启，运行被杀死，
-- 界面既不报错也不产出内容。
--
-- 有了这张表，服务重启后可以把上次未完成的运行标记为 interrupted，
-- 前端就能明确告诉用户"这次运行在综合阶段被中断了"。
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id      TEXT PRIMARY KEY,
    session_id  INTEGER REFERENCES chat_sessions(id) ON DELETE SET NULL,
    topic       TEXT NOT NULL DEFAULT '',
    phase       TEXT NOT NULL DEFAULT 'pending',
    status      TEXT NOT NULL DEFAULT 'running',  -- running/awaiting_approval/done/cancelled/error/interrupted
    papers      INTEGER NOT NULL DEFAULT 0,
    citations   INTEGER NOT NULL DEFAULT 0,
    artifact_id INTEGER,
    error       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at DESC);

-- ------------------------------------------------- 运行的阶段快照（断点续跑）
-- 每完成一个阶段就存一份当时的完整状态，这样：
--   1. 运行中途失败/被中断时，前面几个阶段的成果不会白跑，下次可以接着往下跑；
--   2. 刷新页面后能直接把已完成的方案、文献、草稿重新显示出来。
-- 一次运行每个阶段只保留一行（PRIMARY KEY 覆盖写）。
CREATE TABLE IF NOT EXISTS run_steps (
    run_id     TEXT NOT NULL,
    phase      TEXT NOT NULL,                 -- plan/execute/reflect/synthesize/review
    payload    TEXT NOT NULL DEFAULT '{}',    -- 该阶段结束时的状态快照（JSON）
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, phase)
);

CREATE INDEX IF NOT EXISTS idx_run_steps_run ON run_steps(run_id, created_at);

-- ------------------------------------------------- 用户反馈与质疑（学习闭环）
--
-- 为什么要落库而不是只记在日志里：这是"记忆与学习"的唯一事实来源。
--   * 质疑/纠错 → 成为下次生成时的 few-shot 记忆（即时生效，不需要训练）；
--   * 赞/踩 → 构成偏好对，可导出成 DPO/RLHF 训练数据（离线生效）；
--   * 被反复指出问题的文献/数据源 → 参与检索重加权。
-- verdict: up / down / challenge（质疑）
-- category: citation / fact / omission / structure / style / data / other
CREATE TABLE IF NOT EXISTS feedback (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type    TEXT NOT NULL DEFAULT 'message',  -- message / artifact / paper / plan / manuscript
    target_id      TEXT NOT NULL DEFAULT '',
    run_id         TEXT NOT NULL DEFAULT '',
    session_id     INTEGER,
    verdict        TEXT NOT NULL DEFAULT 'up',
    category       TEXT NOT NULL DEFAULT '',
    comment        TEXT NOT NULL DEFAULT '',
    -- 用户给出的正确版本；用于"纠错记忆"与偏好对的 chosen 侧
    corrected_text TEXT NOT NULL DEFAULT '',
    -- 质疑的是哪段原文（便于精确定位与复核）
    quoted_text    TEXT NOT NULL DEFAULT '',
    topic          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_feedback_target ON feedback(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_feedback_run    ON feedback(run_id);
CREATE INDEX IF NOT EXISTS idx_feedback_time   ON feedback(created_at DESC);

-- ------------------------------------------------------ 实验数据与论文稿件
--
-- 用户把自己的目标/方法/实验数据填进来，系统在**综述 + 本地知识库**的基础上
-- 生成 IMRaD 初稿。关键约束：正文里出现的每个数字都必须能在用户提供的数据里
-- 找到出处（见 manuscript.py 的溯源校验），杜绝"编数据"。
CREATE TABLE IF NOT EXISTS manuscripts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES chat_sessions(id) ON DELETE SET NULL,
    title        TEXT NOT NULL DEFAULT '',
    -- 结构化输入：目标 / 方法 / 数据 / 统计结果 / 目标期刊
    brief        TEXT NOT NULL DEFAULT '{}',
    draft        TEXT NOT NULL DEFAULT '',
    -- 数字溯源与规范检查的结果
    checks       TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'draft',  -- draft / checked / exported
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_manuscripts_session ON manuscripts(session_id, id DESC);

-- ------------------------------------------------------------------ 产物文件
CREATE TABLE IF NOT EXISTS artifacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES chat_sessions(id) ON DELETE SET NULL,
    kind       TEXT NOT NULL DEFAULT 'draft',  -- draft / review / outline / table / export
    title      TEXT NOT NULL DEFAULT '',
    content    TEXT NOT NULL DEFAULT '',
    fmt        TEXT NOT NULL DEFAULT 'markdown',
    meta       TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id, id DESC);

-- ------------------------------------------------------- 定时追踪订阅（P2）
CREATE TABLE IF NOT EXISTS subscriptions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    query        TEXT NOT NULL,
    sources      TEXT NOT NULL DEFAULT '[]',   -- JSON 数组
    since_days   INTEGER NOT NULL DEFAULT 30,
    last_run_at  TEXT,
    last_pmids   TEXT NOT NULL DEFAULT '[]',
    enabled      INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
