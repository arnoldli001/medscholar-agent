"""云端 vs 本地：中文医学综述写作能力与速度对比。

回答一个具体问题：**DeepSeek 适不适合做学术文章的分析推理与综述写作？**

用同一段任务分别跑本地 Ollama 与 DeepSeek 云端，比较：

* 速度（端到端延迟、输出速率）
* 中文医学书面语质量
* **引用纪律** —— 这是综述写作最硬的要求：
  必须只用给定的 [1][2] 编号，不能编造 [3]，也不能把占位符 [n] 原样写出来
* 事实忠实度 —— 是否会添加材料里没有的数据

    .python\\python.exe -X utf8 scripts\\bench_cloud.py
    .python\\python.exe -X utf8 scripts\\bench_cloud.py --cloud deepseek-flash
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
DEEPSEEK = "https://api.deepseek.com/v1"

SYSTEM = (
    "你是医学综述写作助手。只使用我提供的材料，绝不编造文献、数据或结论。"
    "引用只能用我给出的编号。"
)

#: 与 scripts/bench_models.py 使用同一任务，便于横向比较
PROMPT = """请基于以下材料，写一段「加速 rTMS 治疗卒中后抑郁的疗效」的综述正文，约 220 字。

[1] 一项随机对照试验纳入 60 例卒中后抑郁患者，加速 rTMS 组 HAMD 评分较对照组显著降低（P<0.01）。
[2] 一项网络 Meta 分析纳入 12 项研究，提示不同 rTMS 模式疗效存在差异，加速方案可缩短起效时间。

要求：客观陈述，每处论断后用 [n] 标注引用，不要编造材料以外的数据。"""

#: 材料里明确没有的信息 —— 用来检测幻觉
FORBIDDEN_FACTS = ["80 例", "80例", "120 例", "120例", "随访 12 个月", "治愈率", "有效率 95%"]


def deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    import subprocess

    try:
        return subprocess.run(
            ["powershell", "-c",
             '[Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY","User")'],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def call_ollama(model: str, prompt: str, *, max_tokens: int = 500) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
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
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "elapsed": time.perf_counter() - started}
    elapsed = time.perf_counter() - started
    eval_ns = int(data.get("eval_duration") or 0) / 1e9
    count = int(data.get("eval_count") or 0)
    return {
        "ok": True,
        "text": (data.get("message") or {}).get("content", "").strip(),
        "elapsed": elapsed,
        "tokens": count,
        "tps": count / eval_ns if eval_ns else 0.0,
        "load_s": int(data.get("load_duration") or 0) / 1e6 / 1000,
    }


def call_deepseek(model: str, prompt: str, *, key: str, max_tokens: int = 500) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{DEEPSEEK}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.load(response)
    except Exception as exc:
        detail = ""
        if hasattr(exc, "read"):
            detail = f" {exc.read().decode('utf-8', 'replace')[:200]}"
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}{detail}", "elapsed": time.perf_counter() - started}
    elapsed = time.perf_counter() - started
    usage = data.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "ok": True,
        "text": ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip(),
        "elapsed": elapsed,
        "tokens": completion,
        "tps": completion / elapsed if elapsed else 0.0,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cache_hit": int((usage.get("prompt_cache_hit_tokens") or 0)),
    }


def audit(text: str, valid: set[str]) -> dict:
    """审计引用纪律与幻觉。"""
    numbers = re.findall(r"[\[【]\s*(\d{1,3})\s*[\]】]", text)
    out_of_range = sorted({n for n in numbers if n not in valid})
    placeholders = re.findall(r"[\[【]\s*(?:n|x|N|编号|引用|citation)\s*[\]】]", text)
    hallucinated = [f for f in FORBIDDEN_FACTS if f in text]
    return {
        "citations": len(numbers),
        "out_of_range": out_of_range,
        "placeholders": sorted(set(placeholders)),
        "hallucinated": hallucinated,
        "chars": len(text),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="云端 vs 本地 医学写作对比")
    parser.add_argument("--cloud", nargs="*", default=["deepseek-flash", "deepseek-v4-pro"])
    parser.add_argument("--local", nargs="*", default=["qwen3:8b"])
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    key = deepseek_key()
    if not key:
        print("未找到 DEEPSEEK_API_KEY（环境变量或用户级变量）。")
        print("本脚本需要它来测试云端模型；跳过云端，仅测本地。")
        args.cloud = []

    print("=" * 80)
    print("中文医学综述写作：云端 vs 本地")
    print("=" * 80)

    results: list[tuple[str, dict | None, dict | None]] = []

    for model in args.local:
        print(f"\n{'=' * 80}\n本地 · {model}\n{'=' * 80}")
        # 预热，避免把冷加载算进生成速度
        call_ollama(model, "回复：好", max_tokens=4)
        result = call_ollama(model, PROMPT, max_tokens=args.max_tokens)
        if not result["ok"]:
            print("  失败：", result.get("error"))
            results.append((f"本地 {model}", None, None))
            continue
        report = audit(result["text"], {"1", "2"})
        print(result["text"])
        print("-" * 80)
        print(
            f"  延迟 {result['elapsed']:.1f}s | 输出 {result['tokens']} tokens | "
            f"{result['tps']:.1f} tok/s | 引用 {report['citations']} 处"
        )
        print(f"  审计：越界 {report['out_of_range'] or '无'} | "
              f"占位符 {report['placeholders'] or '无'} | 幻觉 {report['hallucinated'] or '无'}")
        results.append((f"本地 {model}", result, report))

    for model in args.cloud:
        print(f"\n{'=' * 80}\n云端 · {model}\n{'=' * 80}")
        result = call_deepseek(model, PROMPT, key=key, max_tokens=args.max_tokens)
        if not result["ok"]:
            print("  失败：", result.get("error"))
            results.append((f"云端 {model}", None, None))
            continue
        report = audit(result["text"], {"1", "2"})
        print(result["text"])
        print("-" * 80)
        print(
            f"  延迟 {result['elapsed']:.1f}s | 输入 {result.get('prompt_tokens', 0)} / "
            f"输出 {result['tokens']} tokens | {result['tps']:.1f} tok/s | 引用 {report['citations']} 处"
        )
        print(f"  审计：越界 {report['out_of_range'] or '无'} | "
              f"占位符 {report['placeholders'] or '无'} | 幻觉 {report['hallucinated'] or '无'}")
        results.append((f"云端 {model}", result, report))

    print(f"\n{'=' * 80}\n汇总\n{'=' * 80}")
    print(f"  {'模型':<22}{'延迟':>9}{'tok/s':>9}{'引用':>6}{'越界':>6}{'占位符':>8}{'幻觉':>6}")
    for name, result, report in results:
        if result is None:
            print(f"  {name:<22}{'失败':>9}")
            continue
        print(
            f"  {name:<22}{result['elapsed']:>8.1f}s{result['tps']:>9.1f}"
            f"{report['citations']:>6}{len(report['out_of_range']):>6}"
            f"{len(report['placeholders']):>8}{len(report['hallucinated']):>6}"
        )
    print()
    print("判读要点：")
    print("  · 「越界」= 引用了材料里不存在的编号（如凭空写出 [3]）")
    print("  · 「占位符」= 把 [n] 原样写进正文而没有替换成真实编号")
    print("  · 「幻觉」= 写出了材料中不存在的样本量等数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
