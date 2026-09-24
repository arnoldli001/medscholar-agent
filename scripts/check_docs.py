"""最终一致性核对：所有文档对"冻结值"的口径必须统一。

写七份文档最容易出的问题不是错字，而是**数字互相打架** ——
翻两份材料看到两个测试数，会立刻怀疑整批材料的可靠性。
所以这里把关键口径做成可执行检查，而不是靠人肉比对。
"""

from __future__ import annotations

import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

#: 冻结后的口径：(检查串, 含义)。
#: 只保留**真正需要跨文档一致**的四个：测试数、包行数、接口口径、架构结论。
#: 曾经的第五项"README 规模表口径"是冗余的 —— "1374" 已经在检查里，
#: 再加一条同义串只会制造假失败（第一版就是这么误报的）。
FROZEN = {
    "1374": "单元测试数",
    "28344": "包内代码行数",
    "49 个 `/api`": "接口路径口径",
    "ARCH: PASS": "架构门禁结论",
    "35 个编号": "PRISMA 清单编号数",
}

#: 这些文件必须带完整口径（材料类）；其余只需内部自洽
REQUIRED = {
    "docs/HIGHLIGHTS.md",
    "docs/INTERVIEW-PROJECT.md",
    "docs/PROBLEMS-AND-STRATEGY.md",
    "docs/RESUME.md",
    "docs/_FACT-PACK.md",
}

STALE = ("28310", "28330", "28337", "54 个接口", "56 个接口", "757 个测试", "1180+ 个测试")


def main() -> int:
    # docs/ 已按"含个人信息"的理由移出版本控制（见 .gitignore）。
    # 所以在**别人的克隆**里它根本不存在 —— 这时必须明确地说"跳过"并返回 0，
    # 而不是报五条"缺少口径"的失败：那会让人以为仓库坏了，
    # 或者更糟 —— 逼着下一个人把个人材料提交上去"修好 CI"。
    docs_dir = pathlib.Path("docs")
    if not docs_dir.is_dir():
        print("docs/ 不存在（个人材料不入版本控制），跳过文档口径检查。")
        return 0

    files = sorted(
        list(pathlib.Path(".").glob("*.md"))
        + list(docs_dir.glob("*.md"))
        + [pathlib.Path("scripts/README.md")]
    )

    print("=" * 70)
    print("冻结值覆盖情况")
    print("=" * 70)
    problems: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        missing = [label for token, label in FROZEN.items() if token not in text]
        stale = [token for token in STALE if token in text]
        name = path.as_posix()
        if name in REQUIRED and missing:
            problems.append(f"{name} 缺少口径：{missing}")
        if stale:
            # 时间线里的历史值可以保留（它们是在解释"数字为什么会动"），但要显式标注。
            # 标注词收得比较宽：文档作者的措辞不会统一，检查器不该逼人用特定词。
            history_markers = (
                "写作期间",
                "历史",
                "快照",
                "当时",
                "旧口径",
                "已废弃",
                "过程中",
                "我测到过",
                "曾经",
                "一开始",
            )
            for token in stale:
                for line_no, line in enumerate(text.splitlines(), 1):
                    if token in line and not any(marker in line for marker in history_markers):
                        problems.append(f"{name}:{line_no} 出现过期数字 {token} 且未标注为历史值")
        status = "缺" + str(len(missing)) if missing else "完整"
        print(f"  {name:34} {status:>6}  过期数字 {len(stale)}")

    print()
    if problems:
        print("不一致：")
        for item in problems:
            print("  ✗", item)
        return 1
    print("口径统一：PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
