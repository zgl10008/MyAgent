"""通过 Agent HTTP API 执行评测集并输出 Markdown 报告。"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from validate_agent_cases import CASES_FILE, PROJECT_ROOT, main as validate_cases


DEFAULT_API_URL = "http://127.0.0.1:8001/api/v1"
DEFAULT_REPORT = PROJECT_ROOT / "scripts" / "eval_agent_report.md"


@dataclass
class CaseResult:
    case_id: str
    status: str
    elapsed_ms: int = 0
    failures: list[str] | None = None
    trace_id: str = ""


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_FILE.read_text(encoding="utf-8-sig"))


def needs_live_search(case: dict[str, Any]) -> bool:
    return "web_search" in case["expected"]["required_tools"]


def is_fault_case(case: dict[str, Any]) -> bool:
    return any("fault_configs/" in fixture for fixture in case["fixture"])


def rag_fixtures(cases: list[dict[str, Any]]) -> list[Path]:
    fixtures = {
        PROJECT_ROOT / fixture
        for case in cases
        for fixture in case["fixture"]
        if fixture.endswith((".md", ".txt"))
    }
    return sorted(fixtures)


async def ingest_rag_fixtures(client: httpx.AsyncClient, api_url: str, cases: list[dict[str, Any]]) -> None:
    for path in rag_fixtures(cases):
        content_type = "text/markdown" if path.suffix == ".md" else "text/plain"
        response = await client.post(
            f"{api_url}/rag/ingest",
            files={"file": (path.name, path.read_bytes(), content_type)},
        )
        response.raise_for_status()


def build_payload(case: dict[str, Any]) -> dict[str, Any]:
    messages = list(case["history"])
    messages.append({"role": "user", "content": case["query"]})
    return {
        "messages": messages,
        "conversation_id": f"agent-eval-{case['id']}",
        "domain": case["domain"],
        "agent_mode": case["request"]["agent_mode"],
        "temperature": 0,
    }


def assess(case: dict[str, Any], response: dict[str, Any]) -> list[str]:
    expected = case["expected"]
    failures: list[str] = []
    content = str(response.get("content") or "")
    tool_calls = set(response.get("tool_calls") or [])

    if not content:
        failures.append("content 为空")
    if not response.get("trace_id"):
        failures.append("缺少 trace_id")
    if response.get("agent_mode") != expected["mode"]:
        failures.append(f"mode={response.get('agent_mode')!r}，期望 {expected['mode']!r}")
    if expected["intent"] is not None:
        actual_intent = (response.get("intent") or {}).get("intent")
        if actual_intent != expected["intent"]:
            failures.append(f"intent={actual_intent!r}，期望 {expected['intent']!r}")
    elif response.get("intent") is not None:
        failures.append("安全短路响应不应返回 intent")
    missing_tools = set(expected["required_tools"]) - tool_calls
    if missing_tools:
        failures.append(f"缺少工具调用: {', '.join(sorted(missing_tools))}")
    forbidden = set(expected["forbidden_tools"]) & tool_calls
    if forbidden:
        failures.append(f"调用了禁用工具: {', '.join(sorted(forbidden))}")
    if bool(response.get("degraded")) != expected["degraded"]:
        failures.append(f"degraded={response.get('degraded')!r}，期望 {expected['degraded']!r}")
    for keyword in expected["answer_keywords"]:
        if keyword not in content:
            failures.append(f"答案缺少关键词: {keyword}")
    return failures


def write_report(results: list[CaseResult], report_path: Path) -> None:
    passed = sum(result.status == "PASS" for result in results)
    failed = sum(result.status == "FAIL" for result in results)
    skipped = sum(result.status == "SKIP" for result in results)
    lines = [
        "# Agent 评测报告",
        "",
        f"- 通过：{passed}",
        f"- 失败：{failed}",
        f"- 跳过：{skipped}",
        "",
        "| Case | 状态 | 耗时(ms) | Trace / 原因 |",
        "| --- | --- | ---: | --- |",
    ]
    for result in results:
        detail = result.trace_id or "；".join(result.failures or [])
        lines.append(f"| {result.case_id} | {result.status} | {result.elapsed_ms} | {detail} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def evaluate(args: argparse.Namespace) -> int:
    if validate_cases() != 0:
        return 2
    cases = load_cases()
    timeout = httpx.Timeout(args.timeout)
    results: list[CaseResult] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.ingest_rag_fixtures:
            await ingest_rag_fixtures(client, args.api_url, cases)
        for case in cases:
            if needs_live_search(case) and not args.include_live_search:
                results.append(CaseResult(case["id"], "SKIP", failures=["需要 --include-live-search"]))
                continue
            if is_fault_case(case):
                results.append(CaseResult(case["id"], "SKIP", failures=["故障注入需专用集成测试"] ))
                continue
            started = time.perf_counter()
            try:
                response = await client.post(f"{args.api_url}/chat/agent", json=build_payload(case))
                response.raise_for_status()
                body = response.json()
                failures = assess(case, body)
                results.append(CaseResult(
                    case["id"], "PASS" if not failures else "FAIL",
                    round((time.perf_counter() - started) * 1000), failures, body.get("trace_id", ""),
                ))
            except (httpx.HTTPError, ValueError) as exc:
                results.append(CaseResult(
                    case["id"], "FAIL", round((time.perf_counter() - started) * 1000), [str(exc)],
                ))
    write_report(results, args.report)
    print(f"报告已写入: {args.report}")
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    return 1 if any(result.status == "FAIL" for result in results) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 Agent API 评测")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ingest-rag-fixtures", action="store_true")
    parser.add_argument("--include-live-search", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(evaluate(parse_args())))
