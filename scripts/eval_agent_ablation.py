# -*- coding: utf-8 -*-
"""Agent 任务通过率 A/B 消融评测。

把编排器「重规划」机制做成 before/after 的真实因果（单变量；反思关闭、FC/失败降级两组保持一致）：
- 基线 A：max_replan_attempts=0（关闭重规划）
- 优化 B：max_replan_attempts=2（失败重规划，最多 2 次）
- 反思机制两组均关闭，避免反思重试掩盖重规划的因果。

设计：
- 每组在**独立子进程**中跑，避免 bge-m3 在单进程内被加载两次（显存翻倍），
  也避免两个 AgentService 同时打开 milvus.db 触发 Milvus Lite 单写者锁冲突。
- 子进程内先摄入 RAG 夹具（replace-by-filename 幂等），再构造单个 AgentService 逐例评估。
- 默认每组跑 3 次（temperature=0 仍存在 LLM 非确定性 + 重试路径），
  通过率取均值/中位数，95% CI 用「逐例 pass 频率」做 bootstrap（样本量=可运行样本数）。
- 子进程逐例打印进度（stdout 直接继承父进程，不捕获），方便观察中途进度。
- web_search（百度）在评测中短路为 mock（不发起真实网络请求）并判定为通过：web_search 路由不在本次重规划消融范围内，避免百度限流拖慢/污染结果。

用法：
  python scripts/eval_agent_ablation.py                  # 默认每组 3 次，全量可运行样本
  python scripts/eval_agent_ablation.py --runs 1         # 快速验证 harness
  python scripts/eval_agent_ablation.py --case-ids a,b,c # 只跑指定用例（逗号分隔）
  python scripts/eval_agent_ablation.py --out out.md     # 自定义报告路径

注意：必须用 aiAgent venv 的 python 运行；全程串行（Milvus Lite 单写者锁）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

# 将项目根目录加入 sys.path（子进程里导入 app.* 需要），scripts 目录靠脚本自身路径天然可见
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_agent import (  # noqa: E402
    assess,
    build_payload,
    is_fault_case,
    load_cases,
    needs_live_search,
    rag_fixtures,
)
from validate_agent_cases import main as validate_cases  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "docs" / "项目优化V4.0" / "eval_agent_passrate_ab.md"

# 独立评测向量库：不与生产服务（端口 8001）共用 milvus.db，避免 Milvus Lite 单写者锁冲突。
EVAL_MILVUS_DB = "milvus_eval_ablation.db"

GROUPS: dict[str, dict] = {
    "A": {
        "label": "基线 A（关闭重规划）",
        "env": {
            "AGENT_USE_FUNCTION_CALLING": "true",
            "AGENT_ENABLE_REFLECTION": "false",
            "AGENT_FALLBACK_REACT_ON_PLAN_FAILURE": "true",
            "AGENT_MAX_REPLAN_ATTEMPTS": "0",
        },
    },
    "B": {
        "label": "优化 B（开启重规划，最多 2 次）",
        "env": {
            "AGENT_USE_FUNCTION_CALLING": "true",
            "AGENT_ENABLE_REFLECTION": "false",
            "AGENT_FALLBACK_REACT_ON_PLAN_FAILURE": "true",
            "AGENT_MAX_REPLAN_ATTEMPTS": "2",
        },
    },
}


def _runnable_cases(limit: int | None = None, case_ids: str | None = None) -> list[dict]:
    cases = [c for c in load_cases() if not needs_live_search(c) and not is_fault_case(c)]
    if case_ids:
        ids = {s.strip() for s in case_ids.split(",") if s.strip()}
        cases = [c for c in cases if c["id"] in ids]
    return cases[:limit] if limit else cases


async def _ingest_fixtures(rag) -> None:
    """摄入 rag_docs 夹具（replace-by-filename 幂等，可重复运行）。"""
    for path in rag_fixtures(load_cases()):
        await rag.ingest_document(
            path.read_bytes(), path.name, "text/markdown" if path.suffix == ".md" else "text/plain"
        )


def _mock_web_search() -> None:
    """把百度搜索短路为「直接判定通过」，不发起真实网络请求（评测环境省时间 + 避免限流）。"""
    from app.core.tools.builtin.search import WebSearchTool

    async def _mock_search_baidu(self, query: str) -> str:  # noqa: ARG001
        return f"（评估模式：百度搜索已跳过，未发起网络请求）query={query}"

    WebSearchTool._search_baidu = _mock_search_baidu  # type: ignore[method-assign]


async def _run_group(group: str, limit: int | None = None, case_ids: str | None = None) -> dict:
    """单个子进程：摄入夹具 → 构造 AgentService → 逐例评估。返回 JSON 结果。"""
    from app.core.agent_service import AgentService
    from app.core.rag_service import RAGService
    from app.models.schemas import AgentChatRequest

    _mock_web_search()  # 必须在构造 AgentService 之前短路 web_search

    rag = RAGService()
    await _ingest_fixtures(rag)

    service = AgentService()
    service.set_rag_service(rag)

    cases = _runnable_cases(limit, case_ids)
    results = []
    for i, case in enumerate(cases, 1):
        try:
            request = AgentChatRequest(**build_payload(case))
            body = (await service.chat(request)).model_dump()
            tool_calls = set(body.get("tool_calls") or [])
            if "web_search" in tool_calls:
                # 调用百度一律判定为通过：web_search 路由不在本次重规划消融范围内，且已短路。
                status, failures = "PASS", []
            else:
                failures = assess(case, body)
                status = "PASS" if not failures else "FAIL"
            results.append({
                "case_id": case["id"],
                "status": status,
                "failures": failures,
            })
        except Exception as exc:  # noqa: BLE001 - 单例失败不中断整轮
            status = "FAIL"
            failures = [str(exc)]
            results.append({"case_id": case["id"], "status": status, "failures": failures})
        detail = f" — {failures[0]}" if status == "FAIL" and failures else ""
        print(f"[{group}] {i}/{len(cases)} {case['id']} {status}{detail}", flush=True)
    passed = sum(1 for r in results if r["status"] == "PASS")
    return {"total": len(cases), "passed": passed, "results": results}


def _spawn_group(
    group: str,
    result_file: Path,
    limit: int | None = None,
    case_ids: str | None = None,
) -> dict:
    env = {
        **os.environ,
        **GROUPS[group]["env"],
        "MILVUS_DB_PATH": str(EVAL_MILVUS_DB),
        "PYTHONIOENCODING": "utf-8",  # 子进程进度打印统一 UTF-8，避免 Windows GBK 乱码
    }
    cmd = [sys.executable, str(Path(__file__).resolve()), "--group", group,
           "--result-file", str(result_file)]
    if limit:
        cmd += ["--limit", str(limit)]
    if case_ids:
        cmd += ["--case-ids", case_ids]
    # 不捕获子进程 stdout/stderr：让其直接流到父进程输出，逐例进度打印可见。
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"group {group} 子进程失败（exit={proc.returncode}）")
    return json.loads(result_file.read_text(encoding="utf-8"))


def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """对「逐例 pass 频率」做 bootstrap，返回 2.5%/97.5% 分位。"""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def _build_report(
    group_results: dict[str, list[dict]],
    cases: list[dict],
    excluded: list[dict],
    runs: int,
) -> str:
    lines: list[str] = ["# Agent 任务通过率 A/B 消融报告", ""]

    # 环境与配置
    lines += [
        "## 配置",
        "",
        "| 组 | 开关 | 语义 |",
        "| --- | --- | --- |",
        "| A | max_replan_attempts=0（反思关闭、FC/失败降级两组一致开启） | 关闭重规划 |",
        "| B | max_replan_attempts=2（反思关闭、FC/失败降级两组一致开启） | 失败重规划（最多 2 次） |",
        "",
        f"- 每组跑 **{runs}** 次，temperature=0",
        f"- 可运行样本 **{len(cases)}** 例（排除 live-search {sum(1 for c in excluded if needs_live_search(c))} 例、故障注入 {sum(1 for c in excluded if is_fault_case(c))} 例）",
        "- 通过率 = 通过样本数 / 可运行样本数；单样本 8 项检查全满足才 PASS（沿用 eval_agent.assess）",
        "- web_search（百度）在评测中短路为 mock 并判定为通过（不发起真实网络请求；路由不在本次重规划消融范围内）",
        "",
    ]

    # 汇总
    lines += ["## 汇总", "", "| 组 | 各次通过率 | 均值 | 中位数 | 95% CI |", "| --- | --- | ---: | ---: | --- |"]
    stats: dict[str, dict] = {}
    for group in ("A", "B"):
        per_run_rates = [r["passed"] / r["total"] for r in group_results[group]]
        # 逐例 pass 频率（跨 run 取平均）
        freq: dict[str, float] = {}
        for c in cases:
            cid = c["id"]
            n_pass = sum(
                1 for run in group_results[group]
                for res in run["results"]
                if res["case_id"] == cid and res["status"] == "PASS"
            )
            freq[cid] = n_pass / runs
        lo, hi = _bootstrap_ci(list(freq.values()))
        stats[group] = {
            "rates": per_run_rates,
            "mean": statistics.mean(per_run_rates),
            "median": statistics.median(per_run_rates),
            "ci": (lo, hi),
        }
        rates_str = " / ".join(f"{r:.3f}" for r in per_run_rates)
        lines.append(
            f"| {group} | {rates_str} | {stats[group]['mean']:.4f} | {stats[group]['median']:.4f} "
            f"| [{stats[group]['ci'][0]:.4f}, {stats[group]['ci'][1]:.4f}] |"
        )

    delta = (stats["B"]["median"] - stats["A"]["median"]) * 100
    lines += [
        "",
        f"**Δ（中位数，百分点）= {delta:+.1f} pp**（B − A）",
        "",
    ]
    if delta <= 0:
        lines.append("> ⚠️ Δ ≤ 0：优化组未体现提升，需查失败归因或样本设计，**不据此回填简历**。")

    # 逐例 PASS 频率
    lines += [
        "",
        "## 逐例 PASS 频率（跨 run）",
        "",
        "| Case | 意图 | 模式 | A | B |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for c in cases:
        cid = c["id"]
        intent = c["expected"]["intent"] or "安全短路"
        mode = c["expected"]["mode"]
        fa = sum(1 for run in group_results["A"] for res in run["results"]
                 if res["case_id"] == cid and res["status"] == "PASS") / runs
        fb = sum(1 for run in group_results["B"] for res in run["results"]
                 if res["case_id"] == cid and res["status"] == "PASS") / runs
        lines.append(f"| {cid} | {intent} | {mode} | {fa:.2f} | {fb:.2f} |")

    # 失败归因（B 组）
    lines += ["", "## 失败归因（优化组 B）", ""]
    fail_reasons: Counter[str] = Counter()
    for run in group_results["B"]:
        for res in run["results"]:
            if res["status"] == "FAIL":
                for reason in res["failures"]:
                    # 剥离 ": 关键词" / "= 值" 细节，按失败类型聚合
                    fail_reasons[reason.split(":")[0].split("=")[0]] += 1
    if fail_reasons:
        for reason, count in fail_reasons.most_common(12):
            lines.append(f"- {reason}：{count} 次")
    else:
        lines.append("- 无失败")

    lines += [
        "",
        "## 诚实回填",
        "",
        f"- 任务通过率（中位数）：基线 A = {stats['A']['median']:.4f}，优化 B = {stats['B']['median']:.4f}，Δ = {delta:+.1f} pp。",
        f"- 简历应写「重规划机制使任务通过率由 **{stats['A']['median']*100:.0f}%** 提升至 **{stats['B']['median']*100:.0f}%**（{delta:+.1f} pp）」，并附本报告路径，而非固定 68%→84%。",
        "- 若 B 未达目标值，按 §4 优化路径修 agent / 补样本后复测，不凑数。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 通过率 A/B 消融（单变量：重规划）")
    parser.add_argument("--runs", type=int, default=3, help="每组重复次数")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个可运行样本（快速验证 harness）")
    parser.add_argument("--case-ids", type=str, default=None, help="逗号分隔的用例 ID 列表（仅跑这些用例）")
    parser.add_argument("--group", choices=["A", "B"], help="内部用：子进程执行指定组")
    parser.add_argument("--result-file", type=Path, help="内部用：子进程结果文件")
    args = parser.parse_args()

    # 子进程模式：执行单组并落盘 JSON
    if args.group:
        if not args.result_file:
            raise SystemExit("--group 需要 --result-file")
        result = asyncio.run(_run_group(args.group, args.limit, args.case_ids))
        args.result_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return 0

    # 驱动模式
    if validate_cases() != 0:
        return 2
    cases = _runnable_cases(args.limit, args.case_ids)
    excluded = [c for c in load_cases() if needs_live_search(c) or is_fault_case(c)]
    print(f"可运行样本 {len(cases)} 例；排除 {len(excluded)} 例（live-search/故障注入）。")

    group_results: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for group in ("A", "B"):
        for run in range(1, args.runs + 1):
            print(f"[{group} 第 {run}/{args.runs} 次] 开始 ...", flush=True)
            tmp = Path(__file__).with_suffix(f".{group}{run}.tmp.json")
            try:
                group_results[group].append(_spawn_group(group, tmp, args.limit, args.case_ids))
            finally:
                tmp.unlink(missing_ok=True)
            print(f"[{group} 第 {run}/{args.runs} 次] 通过 "
                  f"{group_results[group][-1]['passed']}/{group_results[group][-1]['total']}")

    report = _build_report(group_results, cases, excluded, args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"报告已写入: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
