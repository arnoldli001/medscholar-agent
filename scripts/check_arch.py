"""架构约束校验器：把"分层"从文档里的口头约定变成**会红的检查**。

    .python\\python.exe scripts\\check_arch.py            # 检查，违规退出码 1
    .python\\python.exe scripts\\check_arch.py --verbose  # 打印层次总览与全部边
    .python\\python.exe scripts\\check_arch.py --json     # 机器可读输出

## 为什么需要它

分层架构的死法从来不是"不知道该怎么分"，而是**知道但守不住**：
某人为了赶需求在 `db` 里 import 了一下 `server`，测试全绿、评审通过，
三个月后依赖图变成一团，再也没人敢重构。
人眼 review 抓不住这种漂移（它不体现在 diff 的业务逻辑里），
所以必须交给机器 —— 这就是 ArchUnit（Java）/ import-linter（Python）存在的理由。

本项目不引入第三方依赖，用 ~300 行 AST 分析自己实现，规则写在下面的常量里：

1. **层次依赖方向**：高层可以依赖低层，低层**绝不能**反向依赖；
2. **无循环依赖**：模块级 SCC 必须为空（`db.repo ↔ embedding.pipeline` 就是这么被抓出来的）；
3. **规模上限**：单文件 / 单函数 / 单类的行数上限，防止 god module 重新长出来。

## 白名单策略（重要）

已存在的违规可以进白名单，但**白名单只能变小**：每条都必须写明原因与退出条件，
且校验器每次运行都会把白名单内容打印出来 —— 让"技术债"保持在视线内，
而不是藏在一个没人看的 TODO 里。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover - 非标准流
    pass

ROOT = Path(__file__).resolve().parent.parent
PKG = "medscholar"

# ---------------------------------------------------------------------------
# 1) 层次定义
# ---------------------------------------------------------------------------
# 依赖方向：下层 ← 上层。左侧的层可以 import 右侧任意层；反向则违规。
LAYERS: tuple[str, ...] = (
    "platform",       # 横切：配置、观测、韧性、安全、缓存、提示词（只依赖标准库）
    "domain",         # 领域：模型、文本算法、引用格式、去重、证据等级（无 IO）
    "infrastructure",  # 适配器：SQLite、外部 API、LLM、嵌入、导入导出
    "application",    # 用例编排：Agent 工作流、检索、工具、反馈、写作
    "interface",      # 入口：HTTP、CLI、MCP
    "eval",           # 离线评测（可依赖全部；但运行时不得依赖它）
)

#: 允许的依赖矩阵：key 可以 import value 里的层（含自身）
ALLOWED: dict[str, set[str]] = {
    "platform": {"platform"},
    "domain": {"domain"},
    "infrastructure": {"platform", "domain", "infrastructure"},
    "application": {"platform", "domain", "infrastructure", "application"},
    "interface": set(LAYERS),
    "eval": set(LAYERS),
    # 包根 __init__ 只放版本号与包级文档
    "root": set(LAYERS) | {"root"},
}

#: 模块路径 → 层。**最长前缀优先**，所以顺序无关，写清楚即可。
#:
#: 说明：`models` / `textutil` / `dedupe` / `cite` / `config` 现在是**兼容外壳**
#: （真实现在 `domain/` 与 `platform/` 下），仍然按目标层归属，
#: 这样外壳本身也被同一套规则约束（外壳只能 import 它对应的那一层）。
MODULE_LAYERS: dict[str, str] = {
    # platform
    "medscholar.platform": "platform",
    "medscholar.config": "platform",
    # domain
    "medscholar.domain": "domain",
    "medscholar.models": "domain",
    "medscholar.textutil": "domain",
    "medscholar.dedupe": "domain",
    "medscholar.cite": "domain",
    "medscholar.query": "domain",
    # infrastructure
    "medscholar.db": "infrastructure",
    "medscholar.api": "infrastructure",
    "medscholar.llm": "infrastructure",
    "medscholar.embedding": "infrastructure",
    "medscholar.importers": "infrastructure",
    "medscholar.export": "infrastructure",
    "medscholar.bulk": "infrastructure",
    "medscholar.zotero": "infrastructure",
    # application
    "medscholar.application": "application",
    "medscholar.agent": "application",
    "medscholar.retrieval": "application",
    "medscholar.tools": "application",
    "medscholar.feedback": "application",
    "medscholar.manuscript": "application",
    "medscholar.prisma": "application",
    # interface
    "medscholar.server": "interface",
    "medscholar.cli": "interface",
    "medscholar.mcp": "interface",
    # eval
    "medscholar.eval": "eval",
}

#: 层次违规白名单：{(源模块, 目标模块): 原因}。**只允许删，不允许加**。
EDGE_ALLOWLIST: dict[tuple[str, str], str] = {}

#: 循环依赖白名单：{frozenset({a, b}): 原因}
CYCLE_ALLOWLIST: dict[frozenset[str], str] = {}

# ---------------------------------------------------------------------------
# 2) 规模上限
# ---------------------------------------------------------------------------
MAX_MODULE_LOC = 800
MAX_FUNCTION_LOC = 300
MAX_CLASS_LOC = 450

#: 规模白名单：{模块: 原因}。**只允许删，不允许加**。
SIZE_ALLOWLIST: dict[str, str] = {
    "medscholar.agent.graph": "ResearchGraph 的阶段逻辑待拆为 phases/*（见 REFACTORING 待办）",
    "medscholar.agent.runtime": "AgentRuntime 的运行生命周期待抽为 use case 服务",
    "medscholar.cli": "命令行子命令较多；已按子命令分段，进一步拆分收益低",
    "medscholar.eval.faithfulness": "规则集与判定聚合同属一个内聚单元，拆开反而增加跳转成本",
}


@dataclass
class Module:
    name: str
    path: Path
    layer: str
    loc: int
    imports: set[str] = field(default_factory=set)
    sizes: list[tuple[str, int, int]] = field(default_factory=list)  # (kind, name, loc)


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def layer_of(name: str) -> str:
    """最长前缀匹配；找不到就归 root（仅包根会命中）。"""
    best, best_layer = -1, "root"
    for prefix, layer in MODULE_LAYERS.items():
        if (name == prefix or name.startswith(prefix + ".")) and len(prefix) > best:
            best, best_layer = len(prefix), layer
    return best_layer


def _raw_import_targets(path: Path, name: str) -> set[str]:
    """把 AST 里的 import 语句还原成**候选绝对模块名**。

    这里有两个容易写错的地方，都踩过：

    1. **相对导入的层级**。``from .x import y`` 在普通模块里表示"同包下的 x"，
       但在 ``__init__.py`` 里 ``.`` 指**这个包自己**（Python 的语义是
       "当前模块所属包"，而包的 ``__init__`` 就属于它自己）。第一版把两者
       当成一样的，结果 ``medscholar/api/__init__.py`` 里的
       ``from .pubmed_client import PubMedClient`` 被解析成
       ``medscholar.pubmed_client``（根本不存在的模块），
       于是校验器报出 27 条"infrastructure 依赖 root"的假违规 —— 假阳性会把
       真问题淹没，所以解析必须先对。
    2. **``from X import a`` 的真实依赖是 ``X.a`` 而不是 ``X``**。
       ``from medscholar.db import repo`` 依赖的是 ``medscholar.db.repo``；
       只记 ``X`` 会漏掉依赖（这里是漏报，比假报更危险）。
       所以这里返回候选集合，再由调用方用"最长已知模块前缀"归一化。
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    is_package = path.name == "__init__.py"
    pkg_parts = name.split(".") if is_package else name.split(".")[:-1]

    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidates.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(pkg_parts) - (node.level - 1)
                base_parts = pkg_parts[:keep] if keep >= 0 else []
            else:
                base_parts = (node.module or "").split(".") if node.module else []
            if node.module:
                base_parts = base_parts + node.module.split(".")
            base = ".".join(p for p in base_parts if p)
            if base:
                candidates.add(base)
            for alias in node.names:  # `from X import a` → 也可能是 X.a
                if alias.name != "*" and base:
                    candidates.add(f"{base}.{alias.name}")
    return candidates


def parse_module(
    path: Path, known: set[str] | None = None
) -> tuple[set[str], list[tuple[str, int, int]]]:
    """返回（归一化后的包内导入集合，[(kind, name, loc)]）。"""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:  # pragma: no cover - check.py 会先抓到语法错误
        print(f"  [语法错误] {path}: {exc}")
        return set(), []

    me = module_name(path)
    candidates = _raw_import_targets(path, me)

    imports: set[str] = set()
    for cand in candidates:
        if not (cand == PKG or cand.startswith(PKG + ".")):
            continue
        if known is None:
            imports.add(cand)
            continue
        # 归一化：取"已知模块"里最长的前缀（cand 本身优先）
        best = ""
        parts = cand.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in known:
                best = prefix
                break
        imports.add(best or cand)

    internal = {i for i in imports if i == PKG or i.startswith(PKG + ".")}

    sizes: list[tuple[str, int, int]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            sizes.append(("class", node.name, node.end_lineno - node.lineno + 1))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sizes.append(("def", node.name, node.end_lineno - node.lineno + 1))
    return internal, sizes


def collect() -> list[Module]:
    paths = [p for p in sorted((ROOT / PKG).rglob("*.py")) if "__pycache__" not in p.parts]
    known = {module_name(p) for p in paths}
    modules: list[Module] = []
    for path in paths:
        name = module_name(path)
        internal, sizes = parse_module(path, known)
        modules.append(
            Module(
                name=name,
                path=path,
                layer=layer_of(name),
                loc=len(path.read_text(encoding="utf-8", errors="replace").splitlines()),
                imports=internal,
                sizes=sizes,
            )
        )
    return modules


def is_facade(path: Path) -> bool:
    """门面模块：只由 docstring / import / `__all__` 赋值组成（允许 TYPE_CHECKING 块）。

    重导出型模块（例如拆包后保留的 `db/repo.py`）天然很短，
    识别出来是为了让规模检查聚焦在"有实现"的模块上，而不是逼人把它塞大。

    语法错误时返回 False 而不是抛异常：校验器的职责是**报告问题**。
    实测踩过：往 `papers.py` 里写进一个语法错误后，本函数直接抛 SyntaxError
    把整个校验器打断，看到的是校验器的 traceback 而不是"哪一行语法错了"。
    校验器因为被测代码有问题而崩掉，人会开始怀疑校验器并绕开它 ——
    那正是架构约束失效的起点。
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if targets and all(t.startswith("__") for t in targets):
                continue
            return False
        if isinstance(node, ast.If):
            test = ast.dump(node.test)
            if "TYPE_CHECKING" in test:
                continue
            return False
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.startswith("__"):
                continue
        return False
    return True


# ---------------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------------


def check_edges(modules: list[Module]) -> list[str]:
    problems: list[str] = []
    for mod in modules:
        allowed = ALLOWED[mod.layer]
        for dep in sorted(mod.imports):
            if dep == PKG:
                continue
            dep_layer = layer_of(dep)
            if dep_layer in allowed:
                continue
            if (mod.name, dep) in EDGE_ALLOWLIST:
                continue
            problems.append(
                f"{mod.name} ({mod.layer}) → {dep} ({dep_layer})："
                f"{mod.layer} 不允许依赖 {dep_layer}"
            )
    return problems


def check_cycles(modules: list[Module]) -> list[list[str]]:
    graph = {m.name: sorted(d for d in m.imports if d != PKG) for m in modules}
    known = set(graph)
    graph = {k: [d for d in v if d in known] for k, v in graph.items()}

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = [0]
    cycles: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.get(v, []):
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                cycles.append(sorted(comp))

    sys.setrecursionlimit(10000)
    for node in graph:
        if node not in index:
            strongconnect(node)
    return [c for c in cycles if frozenset(c) not in CYCLE_ALLOWLIST]


def check_sizes(modules: list[Module]) -> list[str]:
    problems: list[str] = []
    for mod in modules:
        facade = is_facade(mod.path)
        if mod.loc > MAX_MODULE_LOC and mod.name not in SIZE_ALLOWLIST:
            problems.append(
                f"{mod.name}：文件 {mod.loc} 行 > {MAX_MODULE_LOC}"
                + ("（疑似门面模块，请确认是否只是重导出）" if facade else "")
            )
        if facade:
            continue
        for kind, name, size in mod.sizes:
            limit = MAX_CLASS_LOC if kind == "class" else MAX_FUNCTION_LOC
            if size > limit and mod.name not in SIZE_ALLOWLIST:
                problems.append(f"{mod.name}::{name}（{kind}）{size} 行 > {limit}")
    return problems


def render(modules: list[Module], verbose: bool) -> None:
    by_layer: dict[str, list[Module]] = {layer: [] for layer in (*LAYERS, "root")}
    for mod in modules:
        by_layer[mod.layer].append(mod)

    print("=" * 78)
    print("架构分层总览（按依赖方向自上而下：下层的层号小，被上层依赖）")
    print("=" * 78)
    for i, layer in enumerate(LAYERS):
        mods = by_layer[layer]
        total = sum(m.loc for m in mods)
        edges = Counter()
        for m in mods:
            for d in m.imports:
                if d != PKG:
                    edges[layer_of(d)] += 1
        deps = ", ".join(f"{k}×{v}" for k, v in sorted(edges.items()) if k != layer) or "—"
        print(f"  L{i} {layer:<15} {len(mods):>2} 个模块 {total:>6} 行   依赖: {deps}")
    if verbose:
        for layer in (*LAYERS, "root"):
            print(f"\n  [{layer}]")
            for mod in sorted(by_layer[layer], key=lambda m: -m.loc):
                print(f"      {mod.loc:>5}  {mod.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-arch", description="校验分层架构约束")
    parser.add_argument("--verbose", action="store_true", help="打印层次总览与模块清单")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    modules = collect()
    edge_problems = check_edges(modules)
    cycles = check_cycles(modules)
    size_problems = check_sizes(modules)
    ok = not (edge_problems or cycles or size_problems)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "modules": len(modules),
                    "loc": sum(m.loc for m in modules),
                    "layer_edges": [
                        {"from": m.name, "layer": m.layer, "to": d, "to_layer": layer_of(d)}
                        for m in modules
                        for d in sorted(m.imports)
                        if d != PKG and layer_of(d) != m.layer
                    ],
                    "violations": {
                        "edges": edge_problems,
                        "cycles": cycles,
                        "sizes": size_problems,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if ok else 1

    if args.verbose:
        render(modules, verbose=True)

    print("=" * 78)
    print(f"架构约束校验：{len(modules)} 个模块 / {sum(m.loc for m in modules)} 行")
    print("=" * 78)

    for label, items, hint in (
        ("层次依赖违规", edge_problems, "低层不得依赖高层；确需共享请下沉到 platform/domain"),
        ("循环依赖", [" ↔ ".join(c) for c in cycles], "用依赖倒置（注入回调/协议）打断环"),
        ("规模超限", size_problems, "按业务边界拆分，并保留稳定门面对外"),
    ):
        if not items:
            print(f"  ✓ {label}：无")
            continue
        print(f"  ✗ {label}（{len(items)}）")
        for item in items:
            print(f"      - {item}")
        print(f"      处理建议：{hint}")

    if CYCLE_ALLOWLIST or EDGE_ALLOWLIST or SIZE_ALLOWLIST:
        print("\n  白名单（只能变小，删掉一条就是一次还债）：")
        for (src, dst), why in sorted(EDGE_ALLOWLIST.items()):
            print(f"      [边] {src} → {dst}：{why}")
        for cyc, why in CYCLE_ALLOWLIST.items():
            print(f"      [环] {' ↔ '.join(sorted(cyc))}：{why}")
        for mod, why in sorted(SIZE_ALLOWLIST.items()):
            print(f"      [规模] {mod}：{why}")

    print()
    print("  ARCH:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
