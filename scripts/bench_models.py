"""对比 Ollama 本地模型在**中文医学写作**任务上的速度与质量。

同一个提示词分别跑各模型，输出 token 速率与实际文本，供选型参考。

    .python\\python.exe -X utf8 scripts\\bench_models.py
    .python\\python.exe -X utf8 scripts\\bench_models.py --models qwen2.5:7b qwen3.5:4b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

OLLAMA = "http://127.0.0.1:11434"

#: 与综述写作阶段同类的任务：给出材料，要求写一段带引用的学术中文
PROMPT = """你是医学综述写作助手。请基于以下材料，写一段「加速 rTMS 治疗卒中后抑郁的疗效」的综述正文，约 220 字。

[1] 一项随机对照试验纳入 60 例卒中后抑郁患者，加速 rTMS 组 HAMD 评分较对照组显著降低（P<0.01）。
[2] 一项网络 Meta 分析纳入 12 项研究，提示不同 rTMS 模式疗效存在差异，加速方案可缩短起效时间。

要求：客观陈述，每处论断后用 [n] 标注引用，不要编造材料以外的数据。"""


def list_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=8) as response:
            return [m["name"] for m in json.load(response)["models"]]
    except Exception:
        return []


def generate(model: str, prompt: str, *, max_tokens: int = 420) -> dict:
    """单次生成并测量耗时与 token 速率（关闭 thinking 以对齐 MedScholar 默认配置）。"""
    return _call(model, prompt, max_tokens=max_tokens)


def warmup(model: str) -> float:
    """先把模型载入内存，避免把"冷加载耗时"算进生成速度。

    Ollama 默认空闲 5 分钟就卸载模型；8 GB 内存的机器上同时只挂得住一个模型，
    因此**换模型后第一次请求必然要重新加载**（实测 20~30 秒）。
    不做预热的话，测出来的 tok/s 会被加载时间严重拉低（实测差 2 倍以上）。
    """
    result = _call(model, "回复：好", max_tokens=4)
    return float(result.get("load_s") or 0.0)


def _call(model: str, prompt: str, *, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_ctx": 8192, "num_predict": max_tokens},
    }
    request = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.load(response)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed": time.perf_counter() - started,
        }

    elapsed = time.perf_counter() - started
    text = (data.get("message") or {}).get("content", "") or ""
    eval_count = int(data.get("eval_count") or 0)
    load_ms = int(data.get("load_duration") or 0) / 1e6
    eval_ns = int(data.get("eval_duration") or 0) / 1e9
    # eval_duration 是 Ollama 自报的**纯生成**耗时，不含加载与提示词处理，
    # 用它算 tok/s 才是真正的稳态速度
    tps = eval_count / eval_ns if eval_ns > 0 else (eval_count / elapsed if elapsed > 0 else 0.0)
    return {
        "ok": bool(text.strip()),
        "text": text.strip(),
        "elapsed": elapsed,
        "eval_count": eval_count,
        "tps": tps,
        "load_s": load_ms / 1000,
        "prompt_ms": int(data.get("prompt_eval_duration") or 0) / 1e6,
    }


def check_citation_discipline(text: str, valid: set[str]) -> tuple[int, list[str], list[str]]:
    """统计引用标记，并找出**越界编号**与**占位符**。

    后者容易被忽略但很致命：小模型有时会输出字面量 ``[n]`` 而不是 ``[2]``，
    正文看起来"有引用"，实际上读者根本对不上参考文献表。
    只检查数字编号是抓不到这类问题的。
    """
    import re

    found = re.findall(r"[\[【]\s*(\d{1,3})", text)
    bad = sorted({n for n in found if n not in valid})
    placeholders = re.findall(r"[\[【]\s*(?:n|x|N|X|编号|引用|citation|ref)\s*[\]】]", text)
    return len(found), bad, sorted(set(placeholders))


async def main() -> int:
    parser = argparse.ArgumentParser(description="本地模型中文写作能力对比")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-tokens", type=int, default=420)
    args = parser.parse_args()

    available = list_models()
    if not available:
        print("无法连接 Ollama，请先运行 `ollama serve`。")
        return 1

    if args.models:
        models = [m for m in args.models if m in available] or args.models
    else:
        preferred = ["qwen3:8b", "qwen2.5:7b", "qwen3.5:4b", "llama3.2:3b", "glm4:9b"]
        models = [m for m in preferred if m in available]
    if not models:
        print("没有可测试的模型。已安装：", ", ".join(available))
        return 1

    print("=" * 78)
    print("中文医学写作对比（同一提示词，关闭 thinking，temperature=0.3）")
    print("=" * 78)
    print(f"已安装模型：{', '.join(available)}")
    print(f"待测模型  ：{', '.join(models)}\n")

    rows = []
    for model in models:
        print("=" * 78)
        print(f"模型：{model}")
        print("=" * 78)
        load_s = warmup(model)
        if load_s:
            print(f"  （冷加载 {load_s:.1f}s，已预热，下面只统计稳态生成速度）")
        result = generate(model, PROMPT, max_tokens=args.max_tokens)
        if not result["ok"]:
            print(f"  失败：{result.get('error', '无输出')}\n")
            rows.append((model, None, None, None, None, load_s))
            continue

        citations, bad, placeholders = check_citation_discipline(result["text"], {"1", "2"})
        print(result["text"])
        print("-" * 78)
        verdict = "引用全部合法"
        if bad:
            verdict = f"⚠️ 越界引用 {bad}"
        if placeholders:
            verdict += f" | ⚠️ 出现占位符 {placeholders}（未被替换成真实编号）"
        print(
            f"  稳态生成 {result['tps']:.1f} tok/s | 输出 {result['eval_count']} tokens | "
            f"纯生成 {result['elapsed'] - result['load_s']:.1f}s | 冷加载 {result['load_s']:.1f}s | "
            f"引用标记 {citations} 处 | {verdict}"
        )
        print()
        rows.append(
            (model, result["elapsed"], result["tps"], result["eval_count"],
             len(bad) + len(placeholders), result["load_s"])
        )

    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"  {'模型':<18}{'稳态tok/s':>11}{'纯生成':>9}{'冷加载':>9}{'输出tokens':>12}{'引用问题':>10}")
    for model, elapsed, tps, count, bad, load_s in rows:
        if elapsed is None:
            print(f"  {model:<18}{'失败':>11}")
            continue
        print(
            f"  {model:<18}{tps:>10.1f}{elapsed - load_s:>8.1f}s{load_s:>8.1f}s{count:>12}{bad:>10}"
        )
    print()
    print("说明：")
    print("  · 「稳态 tok/s」是模型已在内存时的真实生成速度，这才是长文写作的瓶颈")
    print("  · 「冷加载」是模型不在内存时需要先付出的固定成本（换模型后必然发生）")
    print("  · 「引用问题」= 越界编号 + 未替换的占位符（如直接把 [n] 写进正文）")
    print("  · 换模型只需改 config.yaml 的 llm.model 一行，重启生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
