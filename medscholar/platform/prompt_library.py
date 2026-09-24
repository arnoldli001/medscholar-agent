"""提示词正文库（带版本号）：项目里所有提示词的单一事实来源。

:mod:`medscholar.platform.prompts` 提供机制（登记、选版、渲染、统计），本模块提供内容；
装配方向是 ``本模块 → prompts``，反向 import 会成环。
迁移自 ``llm/prompts.py`` 的正文逐字不改（``tests/test_prompts.py`` 逐条断言与源常量
``==``）；函数型提示词只把每次调用变化的值留成 ``{占位符}``，其余原样搬迁。
键名规则 ``<类别>.<用途>``（类别见 prompts.PROMPT_KINDS），分支片段也各占一个 key。
文本进注册表还是留调用方，判据是"这句话会不会出现在给模型的提示词里"：会 → 注册表
（含兜底文案）；不会（取值计算、拼接顺序、JSON 序列化）→ 留调用方，故 _PLAN_SCHEMA
等数据结构仍在外壳以保证 json.dumps 输出逐字不变。模板只有 ``{名字}`` 一种语法，
只替换声明过的占位符，未声明的花括号（如示范 JSON）原样保留。
``writer.system`` / ``writer.section`` 另登记 ``citation-strict`` 变体，
默认版一字不动，运行期用 set_variant 切换。
"""

from __future__ import annotations

from medscholar.platform.prompts import (
    REGISTRY,
    PromptRegistry,
    PromptVersion,
    register_library_loader,
)

__all__ = [
    "ROLE",
    "HARD_RULES",
    "PLAN_SYSTEM",
    "TRANSLATE_SYSTEM",
    "OUTLINE_SYSTEM",
    "SECTION_SYSTEM",
    "CRITIQUE_SYSTEM",
    "REFLECT_SYSTEM",
    "SUMMARY_SYSTEM",
    "CHAT_SYSTEM",
    "ASK_SYSTEM",
    "MANUSCRIPT_SYSTEM",
    "FAITHFULNESS_SYSTEM",
    "MIGRATED_DESCRIPTION",
    "VERSIONS",
    "VARIANT_VERSIONS",
    "install",
]

#: 迁移条目的统一说明：迁移是纯搬迁，需要解释"为什么改"的是后续版本。
MIGRATED_DESCRIPTION = "从 llm/prompts.py 原样迁移，行为与迁移前完全一致"

# ===========================================================================
# 一、迁移自 llm/prompts.py 的正文（逐字不变）
# ===========================================================================

#: 统一角色设定，被 9 个系统提示词复用，故单独占 key，避免改人设要动 9 处文本。
ROLE = (
    "你是 MedScholar，一位严谨的医学研究助理，服务对象是康复科医生与医学研究生。"
    "你熟悉循证医学方法学（RCT、Meta 分析、GRADE 证据分级）与康复医学常用术语。"
)

#: 医学场景五条底线约束（不编造、不猜、不自造编号等），集中一处保证各阶段纪律一致。
HARD_RULES = (
    "硬性规则：\n"
    "1. 只使用我提供的文献材料，绝不编造文献、作者、期刊、年份、数据或结论；\n"
    "2. 材料不足以支撑结论时，直接说明「现有材料不足以判断」，不要猜测；\n"
    "3. 引用文献时使用我给的编号（如 [3]），不要自造编号；\n"
    "4. 涉及疗效时区分「统计学显著」与「临床意义」，并注意样本量与随访时长；\n"
    "5. 严格按要求的结构与格式输出，不要添加额外寒暄。"
)

# ------------------------------------------------------------------ Plan
PLAN_SYSTEM = (
    f"{ROLE}\n\n"
    "当前处于**检索规划**阶段。你的任务是：把用户的研究课题拆解成可执行的检索策略，"
    "并给出综述大纲。\n\n"
    f"{HARD_RULES}\n\n"
    "只输出一个 JSON 对象，不要任何解释文字或 Markdown 围栏。"
)

#: schema 由调用方 json.dumps 后填入，示范 JSON 的缩进与转义仍由标准库负责，迁移不动一个空格。
PLAN_USER = (
    "研究课题：{topic}\n"
    "{extra_note}{note}\n\n"
    "请按以下 JSON 结构输出（字段名保持一致，不要增删顶层字段）：\n"
    "{schema}\n\n"
    "要求：\n"
    "- queries 给 2~4 条，覆盖不同侧重（如干预疗效、机制、指南/综述）；\n"
    "- 检索式要包含同义词与常用缩写（例如 rTMS / repetitive transcranial magnetic stimulation）；\n"
    "- 检索式内部**不要**使用英文双引号（会破坏 JSON）；需要短语时用单引号或直接写词组；\n"
    "- outline 给 4~6 个章节，符合医学综述的常规结构（引言、方法、结果、讨论、结论）；\n"
    "- year_from 用整数年份，综述类课题可设得早一些。"
)

#: "用户补充说明"片段；无补充时调用方传空串——分支留调用方，文本留注册表。
PLAN_USER_EXTRA_NOTE = "\n用户补充说明：{extra}"

#: 离线时必须显式让模型把 queries 留空，否则小模型会凭记忆编出检索式。
PLAN_USER_NOTE_OFFLINE = (
    "\n注意：当前为**离线模式**，不联网检索，请把 queries 留空数组，"
    "重点给出大纲与关键问题。"
)

#: 数据源清单直接写进正文（本就是提示词的一部分），不由调用方拼字符串，避免漏源无人发现。
PLAN_USER_NOTE_ONLINE = (
    "\n可用数据源（sources 字段只能从中选择）："
    "pubmed, europepmc, openalex, crossref, semantic_scholar, arxiv\n"
    "中文课题请**同时**给出中文检索式（供 OpenAlex 中文过滤）与英文检索式（供 PubMed）。"
)

# --------------------------------------------------------------- Translate
TRANSLATE_SYSTEM = (
    f"{ROLE}\n\n"
    "当前处于**术语翻译**阶段。把中文医学检索词翻译成最适合 PubMed 检索的英文表达。\n"
    "只输出 JSON，不要解释。"
)

#: 正文里的 ``{"en": ...}`` 是示范 JSON（未声明的花括号），受限替换不会碰它。
TRANSLATE_USER = (
    "中文内容：{text}\n\n"
    "输出 JSON：{\"en\": \"英文检索式\", \"mesh\": [\"候选 MeSH 词\"]}\n"
    "要求：英文使用医学文献中的标准术语与常见缩写并列（如 "
    "\"repetitive transcranial magnetic stimulation\" OR rTMS）。"
)

# ---------------------------------------------------------------- Outline
OUTLINE_SYSTEM = (
    f"{ROLE}\n\n"
    "当前处于**大纲规划**阶段。基于已检索到的文献材料，为综述确定最终大纲。\n\n"
    f"{HARD_RULES}\n\n"
    "只输出 JSON，不要解释。"
)

OUTLINE_USER = (
    "研究课题：{topic}\n\n"
    "已检索到的文献材料：\n{digest}\n\n"
    "请输出 JSON：{\"outline\": [{\"title\": \"章节标题\", \"points\": [\"要点1\", \"要点2\"]}]}\n"
    "要求：共 {section_count} 个章节；章节顺序符合医学综述逻辑；"
    "每个章节 2~4 个要点，且要点必须能从上述材料中找到支撑。"
)

# ---------------------------------------------------------------- Section
#: 综述撰写（写作）阶段的系统提示词。
SECTION_SYSTEM = (
    f"{ROLE}\n\n"
    "当前处于**综述撰写**阶段。你要基于给定材料撰写综述的某一个章节。\n\n"
    f"{HARD_RULES}\n\n"
    "写作要求：\n"
    "- 使用规范的学术中文，客观陈述，避免「我认为」「众所周知」等主观表达；\n"
    "- 每处关键论断后附引用编号，如「加速 rTMS 可缩短起效时间 [2][5]」；\n"
    "- 有具体数据（样本量、效应量、P 值、置信区间）时优先引用；\n"
    "- 段落之间用空行分隔，不要输出 Markdown 标题（标题由系统添加）。"
)

WRITER_SECTION = (
    "研究课题：{topic}\n"
    "本章节标题：{section_title}\n"
    "本章节需覆盖的要点：\n{bullet}\n\n"
    "可引用的文献材料：\n{digest}\n\n"
    "{length_clause}\n"
    "写作时请注意：\n"
    "- 不要用空话、套话凑字数；篇幅靠**具体证据**支撑——研究设计、样本量、"
    "干预参数（频率/强度/疗程）、效应量与置信区间、不良反应发生率、随访时长等；\n"
    "- 每一条要点都展开成独立段落，并在段末给出引用编号 [n]；\n"
    "- 若材料中确有信息，就写足；确实没有的内容不要编造，也不要用重复表述填充。"
)

#: 有下限时给区间并明确"写满下限"：只给"约 900 字"模型真会只写 900（见 tests/test_review_length.py）。
WRITER_SECTION_LENGTH_WITH_MIN = (
    "请撰写「{style}」，篇幅要求：**不少于 {min_chars} 字**，"
    "以 {max_chars} 字为目标（尽量写到接近 {max_chars} 字，但不要超过）。"
    "直接输出正文。"
)

#: 只有上限（或只有目标字数）时的篇幅要求。
WRITER_SECTION_LENGTH_PLAIN = "请撰写「{style}」，约 {max_chars} 字，直接输出正文。"

#: 章节无要点时的兜底文案；是给模型看的话，所以进注册表而不是留在调用方分支里。
WRITER_SECTION_NO_POINTS = "（无额外要点）"

# ---------------------------------------------------------------- Critique
CRITIQUE_SYSTEM = (
    f"{ROLE}\n\n"
    "当前处于**文献批判**阶段。你要按循证医学标准评估文献质量与相关性。\n\n"
    "评分标准（0~10 分）：\n"
    "- 9~10：大样本 RCT / 高质量 Meta 分析 / 权威指南\n"
    "- 7~8：小样本 RCT、队列研究、系统评价\n"
    "- 5~6：病例对照、横断面、有对照的临床观察\n"
    "- 3~4：病例系列、个案报告、纯机制/动物实验\n"
    "- 0~2：述评、编者按、无方法学信息的短文\n\n"
    "只输出 JSON，不要解释。"
)

CRITIQUE_USER = (
    "研究课题：{topic}\n\n"
    "待评估文献（共 {count} 篇）：\n{digest}\n\n"
    "请逐篇评估，并按以下 JSON 结构输出：\n"
    "{schema}\n\n"
    "要求：assessments 必须覆盖上面每一篇文献的编号；评分要有区分度，"
    "不要所有文献都给同一个分数。"
)

# ----------------------------------------------------------------- Reflect
REFLECT_SYSTEM = (
    f"{ROLE}\n\n"
    "当前处于**自我审查**阶段。你要以严格的审稿人视角检查刚生成的综述草稿。\n\n"
    "重点检查：\n"
    "1. 引号编号是否越界（引用了材料中不存在的编号）；\n"
    "2. 是否出现材料中查无实据的数据、结论或文献；\n"
    "3. 论证是否有跳跃、前后矛盾；\n"
    "4. 是否遗漏了重要文献；\n"
    "5. 是否符合医学综述的表述规范。\n\n"
    "只输出 JSON，不要解释。"
)

REFLECT_USER = (
    "研究课题：{topic}\n\n"
    "草稿中**允许出现**的文献编号只有：{ids}\n\n"
    "综述草稿：\n{draft}\n\n"
    "可依据的文献材料：\n{digest}\n\n"
    "请按以下 JSON 结构输出审查结果：\n"
    "{schema}\n\n"
    "若草稿没有实质问题，verdict 填 pass 并把 issues 设为空数组。"
)

#: 没有任何合法编号时的兜底文案（如材料为空），同样是给模型看的文本。
REFLECT_USER_NO_IDS = "（无）"

# ----------------------------------------------------------------- Summary
SUMMARY_SYSTEM = (
    f"{ROLE}\n\n"
    "当前处于**单篇文献速读**阶段。用中文为研究者提炼这篇文献的关键信息。\n\n"
    f"{HARD_RULES}\n\n"
    "输出格式（纯文本，不要 JSON、不要 Markdown 表格）：\n"
    "【研究设计】\n【研究对象】\n【干预与对照】\n【主要结局】\n【关键结果】\n"
    "【结论】\n【局限】\n【对本课题的用处】"
)

#: 速读提示词按块拼装（课题块/关注点块可选），拆开后"有/无课题"两条路径可独立审阅改版。
SUMMARY_USER_TOPIC = "我的研究课题是：{topic}"
SUMMARY_USER_FOCUS = "我特别关心：{focus}"
SUMMARY_USER_BODY = "文献内容：\n{paper_text}"
SUMMARY_USER_TAIL = "请按系统提示的格式输出速读笔记，每节 1~3 句，务必简洁。"

# -------------------------------------------------------------------- Chat
CHAT_SYSTEM = (
    f"{ROLE}\n\n"
    "当前处于**对话**阶段。你可以调用工具检索文献、检索本地知识库、生成综述。\n\n"
    f"{HARD_RULES}\n\n"
    "回答风格：先给结论，再给依据；用中文；适当使用短列表；不要长篇大论。"
)

CHAT_USER_QUESTION = "研究者的问题：{question}"
CHAT_USER_CONTEXT = "当前课题上下文：\n{context}"
CHAT_USER_LOCAL_HITS = (
    "本地知识库检索到的文献（可直接引用，编号为方括号内的数字）：\n{local_hits}"
)
CHAT_USER_TAIL = "请回答问题。若上述材料不足以回答，请明确说明，并建议下一步检索方向。"

# ----------------------------------------------------------------- 知识库问答
ASK_SYSTEM = (
    f"{ROLE}\n\n"
    "当前处于**知识库问答**阶段。研究者问的是一个具体问题，你要基于给定的本地文献材料回答。\n\n"
    f"{HARD_RULES}\n\n"
    "回答要求：\n"
    "- **先给结论，再给依据**，不要绕圈子；\n"
    "- 用中文，篇幅控制在 150~400 字（除非问题明显需要更多）；\n"
    "- 每处涉及具体结论的地方用 [n] 标注来源编号；\n"
    "- 如果材料不足以回答，直接说明「本地知识库中没有足够材料回答这个问题」，"
    "并建议需要补充检索的方向 —— **不要用常识硬答**；\n"
    "- 如果问题与文献无关（例如在问软件怎么用），就直接回答，不必强行引用文献。"
)

ASK_USER = "研究者的问题：{question}\n\n{materials}"

ASK_MATERIALS_DIGEST = (
    "本地知识库里检索到 {paper_count} 篇相关文献：\n\n{digest}\n\n"
    "请基于以上材料回答问题。"
)

#: 库为空时必须显式告诉模型"没有材料"，否则小模型会用常识硬答——本项目最不能接受的行为。
ASK_MATERIALS_EMPTY = (
    "本地知识库中没有检索到相关文献（库可能还是空的，或这个问题不涉及文献）。\n"
    "如果问题与学术文献相关但库中没有材料，请明确说明；"
    "如果是关于本软件如何使用的操作性问题，直接给出回答。"
)

# ===========================================================================
# 二、其它模块里"散落"的系统提示词
# ===========================================================================
# 这两条原写在 manuscript.py 与 eval/faithfulness.py，是"提示词散落"的实例：
# 先登记（文本逐字复制，由 tests/test_prompts.py 断言与源常量 ==），模块本身暂不改。

#: 论文初稿撰写（medscholar.manuscript._SYSTEM）。
MANUSCRIPT_SYSTEM = (
    "你是一位资深医学论文写作指导专家，帮助研究者把**他们自己的**实验数据"
    "整理成规范的学术论文。\n\n"
    "不可违背的规则：\n"
    "1. 绝对不得编造、推测或「补全」任何数据。正文里出现的每个数字都必须"
    "来自用户提供的数据或统计结果；没有的数据就写「本研究未测量」或留待补充。\n"
    "2. 不得把相关性表述为因果；不得使用「证实」「治愈」「突破性」等超出证据的措辞。\n"
    "3. 引用一律使用方括号编号 [n]，且编号必须来自给定文献材料；没有依据就不写引用。\n"
    "4. 输出规范的学术中文（除非指明英文），不使用口语与主观表达。\n"
)

#: 引用忠实度 LLM 裁判（medscholar.eval.faithfulness._JUDGE_SYSTEM）。
FAITHFULNESS_SYSTEM = (
    "你是医学文献核查专家。给定一句带引用的论断，以及**该引用所指文献**的摘要/正文片段，"
    "判断该文献是否支持这句话。\n\n"
    "判定必须从下列四选一：\n"
    "- supported：文献明确支持该论断；\n"
    "- partial：部分支持，或证据弱于论断的表述强度；\n"
    "- unsupported：文献中找不到支持该论断的内容；\n"
    "- contradicted：文献结论与该论断**方向相反**。\n\n"
    "严格要求：\n"
    "1. 只依据给出的文献片段判断，不要用你自己的先验知识；\n"
    "2. 不要因为论断写得流畅就判为 supported；\n"
    "3. 必须引用文献片段中的**原文词句**作为证据；找不到就填空并在理由里说明；\n"
    "4. 只输出 JSON：{\"verdict\": \"...\", \"evidence\": \"原文片段\", \"reason\": \"一句话理由\"}"
)

#: 裁判的用户提示词：论断 + 引用文献片段（片段之间空行分隔，由调用方拼好）。
FAITHFULNESS_JUDGE = (
    "待核查的论断（来自一篇综述草稿）：\n"
    "「{claim}」\n\n"
    "该论断引用的文献内容：\n{blocks}"
)

#: 每条引用对应的文献块；查不到内容时用下面的兜底文案。
FAITHFULNESS_SOURCE_BLOCK = "【文献 [{cid}]】\n{content}"
FAITHFULNESS_SOURCE_EMPTY = "（无可用内容）"

# ===========================================================================
# 三、citation-strict 变体（新增，用于 A/B）
# ===========================================================================
# 只加在写作/引用类提示词上：引用幻觉是本地小模型最典型、也最像正常文章的失效模式。
# 默认变体正文一字未动，切换是纯运行期行为（set_variant）。

#: 追加在默认版之后的严格引用纪律。
_CITATION_STRICT_ADDENDUM = (
    "\n\n引用纪律（本版为 citation-strict 变体，要求严于默认版）：\n"
    "1. 只能引用材料中**已经给出**的编号（如 [3]），超出该范围的编号一律不得出现；\n"
    "2. 每处引用都必须能在材料里找到对应的原文依据；找不到依据的论断宁可不写，也不要配一个编号；\n"
    "3. 正文里出现的每一个数字（样本量、效应量、P 值、置信区间、随访时长）都必须能在材料中"
    "逐字找到；材料没有的数字一律不写，也不要用其他数字推算或换算；\n"
    "4. 材料只支持一半的表述（例如只有短期结局），就只写这一半，不要顺手把结论说满。"
)

_VARIANT_DESCRIPTION = (
    "citation-strict 变体 v1（新增，用于 A/B）：正文 = 默认版 + 更严格的引用纪律"
    "（只允许给定编号、不得编造、数字必须能在材料中逐字找到）；默认变体不受影响。"
)

#: writer.system 的严格引用变体。
CITATION_STRICT_SECTION_SYSTEM = SECTION_SYSTEM + _CITATION_STRICT_ADDENDUM
#: writer.section 的严格引用变体。
CITATION_STRICT_WRITER_SECTION = WRITER_SECTION + _CITATION_STRICT_ADDENDUM

# ===========================================================================
# 四、版本登记表
# ===========================================================================
# 正文与登记表分开：正文要能逐字读，版本/说明/标签/占位符是元数据，混在一起会淹没正文。
# 登记顺序只影响阅读，不影响取用——选版永远按版本号走。

#: 默认变体的全部版本。迁移条目一律 v1，说明统一用 MIGRATED_DESCRIPTION。
VERSIONS: tuple[PromptVersion, ...] = (
    # ---- 共用片段
    PromptVersion(
        key="core.role", version=1, text=ROLE, description=MIGRATED_DESCRIPTION,
        tags=("core", "role"),
    ),
    PromptVersion(
        key="core.hard_rules", version=1, text=HARD_RULES, description=MIGRATED_DESCRIPTION,
        tags=("core", "rules"),
    ),
    # ---- 检索规划
    PromptVersion(
        key="plan.system", version=1, text=PLAN_SYSTEM, description=MIGRATED_DESCRIPTION,
        tags=("system", "plan"),
    ),
    PromptVersion(
        key="plan.user", version=1, text=PLAN_USER, description=MIGRATED_DESCRIPTION,
        tags=("user", "plan"), placeholders=("topic", "extra_note", "note", "schema"),
    ),
    PromptVersion(
        key="plan.user.extra_note", version=1, text=PLAN_USER_EXTRA_NOTE,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "plan"), placeholders=("extra",),
    ),
    PromptVersion(
        key="plan.user.note_offline", version=1, text=PLAN_USER_NOTE_OFFLINE,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "plan", "offline"),
    ),
    PromptVersion(
        key="plan.user.note_online", version=1, text=PLAN_USER_NOTE_ONLINE,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "plan", "online"),
    ),
    # ---- 术语翻译
    PromptVersion(
        key="translate.system", version=1, text=TRANSLATE_SYSTEM, description=MIGRATED_DESCRIPTION,
        tags=("system", "translate"),
    ),
    PromptVersion(
        key="translate.user", version=1, text=TRANSLATE_USER, description=MIGRATED_DESCRIPTION,
        tags=("user", "translate"), placeholders=("text",),
    ),
    # ---- 大纲
    PromptVersion(
        key="outline.system", version=1, text=OUTLINE_SYSTEM, description=MIGRATED_DESCRIPTION,
        tags=("system", "outline"),
    ),
    PromptVersion(
        key="outline.user", version=1, text=OUTLINE_USER, description=MIGRATED_DESCRIPTION,
        tags=("user", "outline"), placeholders=("topic", "digest", "section_count"),
    ),
    # ---- 综述撰写
    PromptVersion(
        key="writer.system", version=1, text=SECTION_SYSTEM, description=MIGRATED_DESCRIPTION,
        tags=("system", "writer", "citation"),
    ),
    PromptVersion(
        key="writer.section", version=1, text=WRITER_SECTION, description=MIGRATED_DESCRIPTION,
        tags=("user", "writer", "citation"),
        placeholders=("topic", "section_title", "bullet", "digest", "length_clause"),
    ),
    PromptVersion(
        key="writer.section.length_with_min", version=1, text=WRITER_SECTION_LENGTH_WITH_MIN,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "writer", "length"),
        placeholders=("style", "min_chars", "max_chars"),
    ),
    PromptVersion(
        key="writer.section.length_plain", version=1, text=WRITER_SECTION_LENGTH_PLAIN,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "writer", "length"),
        placeholders=("style", "max_chars"),
    ),
    PromptVersion(
        key="writer.section.no_points", version=1, text=WRITER_SECTION_NO_POINTS,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "writer"),
    ),
    # ---- 文献批判
    PromptVersion(
        key="critique.system", version=1, text=CRITIQUE_SYSTEM, description=MIGRATED_DESCRIPTION,
        tags=("system", "critique"),
    ),
    PromptVersion(
        key="critique.user", version=1, text=CRITIQUE_USER, description=MIGRATED_DESCRIPTION,
        tags=("user", "critique"), placeholders=("topic", "count", "digest", "schema"),
    ),
    # ---- 自我审查
    PromptVersion(
        key="reflect.system", version=1, text=REFLECT_SYSTEM, description=MIGRATED_DESCRIPTION,
        tags=("system", "reflect"),
    ),
    PromptVersion(
        key="reflect.user", version=1, text=REFLECT_USER, description=MIGRATED_DESCRIPTION,
        tags=("user", "reflect"), placeholders=("topic", "ids", "draft", "digest", "schema"),
    ),
    PromptVersion(
        key="reflect.user.no_ids", version=1, text=REFLECT_USER_NO_IDS,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "reflect"),
    ),
    # ---- 单篇速读
    PromptVersion(
        key="summary.system", version=1, text=SUMMARY_SYSTEM, description=MIGRATED_DESCRIPTION,
        tags=("system", "summary"),
    ),
    PromptVersion(
        key="summary.user.topic", version=1, text=SUMMARY_USER_TOPIC,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "summary"), placeholders=("topic",),
    ),
    PromptVersion(
        key="summary.user.focus", version=1, text=SUMMARY_USER_FOCUS,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "summary"), placeholders=("focus",),
    ),
    PromptVersion(
        key="summary.user.body", version=1, text=SUMMARY_USER_BODY,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "summary"),
        placeholders=("paper_text",),
    ),
    PromptVersion(
        key="summary.user.tail", version=1, text=SUMMARY_USER_TAIL,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "summary"),
    ),
    # ---- 对话
    PromptVersion(
        key="chat.system", version=1, text=CHAT_SYSTEM, description=MIGRATED_DESCRIPTION,
        tags=("system", "chat"),
    ),
    PromptVersion(
        key="chat.user.question", version=1, text=CHAT_USER_QUESTION,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "chat"), placeholders=("question",),
    ),
    PromptVersion(
        key="chat.user.context", version=1, text=CHAT_USER_CONTEXT,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "chat"), placeholders=("context",),
    ),
    PromptVersion(
        key="chat.user.local_hits", version=1, text=CHAT_USER_LOCAL_HITS,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "chat"), placeholders=("local_hits",),
    ),
    PromptVersion(
        key="chat.user.tail", version=1, text=CHAT_USER_TAIL,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "chat"),
    ),
    # ---- 知识库问答
    PromptVersion(
        key="ask.system", version=1, text=ASK_SYSTEM, description=MIGRATED_DESCRIPTION,
        tags=("system", "ask"),
    ),
    PromptVersion(
        key="ask.user", version=1, text=ASK_USER, description=MIGRATED_DESCRIPTION,
        tags=("user", "ask"), placeholders=("question", "materials"),
    ),
    PromptVersion(
        key="ask.materials.digest", version=1, text=ASK_MATERIALS_DIGEST,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "ask", "materials"),
        placeholders=("paper_count", "digest"),
    ),
    PromptVersion(
        key="ask.materials.empty", version=1, text=ASK_MATERIALS_EMPTY,
        description=MIGRATED_DESCRIPTION, tags=("fragment", "ask", "materials"),
    ),
    # ---- 其它模块里散落的提示词（文本逐字复制，见本节前的说明）
    PromptVersion(
        key="manuscript.system", version=1, text=MANUSCRIPT_SYSTEM,
        description="从 medscholar/manuscript.py 的 _SYSTEM 原样迁移，文本逐字不变",
        tags=("system", "manuscript"),
    ),
    PromptVersion(
        key="faithfulness.system", version=1, text=FAITHFULNESS_SYSTEM,
        description="从 medscholar/eval/faithfulness.py 的 _JUDGE_SYSTEM 原样迁移，文本逐字不变",
        tags=("system", "faithfulness"),
    ),
    PromptVersion(
        key="faithfulness.judge", version=1, text=FAITHFULNESS_JUDGE,
        description="从 medscholar/eval/faithfulness.py 的 _build_judge_prompt 原样迁移，文本逐字不变",
        tags=("user", "faithfulness"), placeholders=("claim", "blocks"),
    ),
    PromptVersion(
        key="faithfulness.source_block", version=1, text=FAITHFULNESS_SOURCE_BLOCK,
        description="从 medscholar/eval/faithfulness.py 的 _build_judge_prompt 原样迁移，文本逐字不变",
        tags=("fragment", "faithfulness"), placeholders=("cid", "content"),
    ),
    PromptVersion(
        key="faithfulness.source_empty", version=1, text=FAITHFULNESS_SOURCE_EMPTY,
        description="从 medscholar/eval/faithfulness.py 的 _build_judge_prompt 原样迁移，文本逐字不变",
        tags=("fragment", "faithfulness"),
    ),
)

#: 非 default 变体的版本。变体与版本是两个正交的轴，这里的 v1 与默认变体 v1
#: 互不影响——A/B 切的是变体轴，不是版本号。
VARIANT_VERSIONS: tuple[tuple[str, PromptVersion], ...] = (
    (
        "citation-strict",
        PromptVersion(
            key="writer.system", version=1, text=CITATION_STRICT_SECTION_SYSTEM,
            description=_VARIANT_DESCRIPTION,
            tags=("system", "writer", "citation", "citation-strict"),
        ),
    ),
    (
        "citation-strict",
        PromptVersion(
            key="writer.section", version=1, text=CITATION_STRICT_WRITER_SECTION,
            description=_VARIANT_DESCRIPTION,
            tags=("user", "writer", "citation", "citation-strict"),
            placeholders=("topic", "section_title", "bullet", "digest", "length_clause"),
        ),
    ),
)


def install(registry: PromptRegistry) -> None:
    """把 :data:`VERSIONS` 与 :data:`VARIANT_VERSIONS` 装进给定注册表。

    写成函数而非 import 副作用，是为了让 reset_registry 能重建回内建状态
    （见 prompts 模块 docstring 关于"不能反向 import"的说明）。
    """
    for item in VERSIONS:
        registry.register(item)
    for variant, item in VARIANT_VERSIONS:
        registry.register(item, variant=variant)


register_library_loader(install)
install(REGISTRY)
