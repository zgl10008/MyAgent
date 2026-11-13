"""静态校验 Agent 评测集的标签契约与离线夹具。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = PROJECT_ROOT / "scripts" / "eval_agent_cases.json"

VALID_RISKS = {"safe", "clarify", "emergency"}
VALID_INTENTS = {"知识问答", "实时搜索", "数学计算", "文档管理", "闲聊对话"}
VALID_MODES = {"react", "plan_execute", "safety", "clarify"}
VALID_TOOLS = {"calculator", "web_search", "rag_search"}


def add_error(errors: list[str], case_id: str, message: str) -> None:
    errors.append(f"{case_id}: {message}")


def validate_case(case: dict[str, Any], seen_ids: set[str], errors: list[str]) -> None:
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        add_error(errors, "<unknown>", "缺少 id")
        return
    if case_id in seen_ids:
        add_error(errors, case_id, "id 重复")
    seen_ids.add(case_id)

    if not isinstance(case.get("query"), str) or not case["query"].strip():
        add_error(errors, case_id, "query 不能为空")
    if case.get("domain") not in {"general", "cardiovascular"}:
        add_error(errors, case_id, "domain 必须为 general 或 cardiovascular")

    request = case.get("request")
    if not isinstance(request, dict) or request.get("agent_mode") not in {
        "auto",
        "react",
        "plan_execute",
    }:
        add_error(errors, case_id, "request.agent_mode 非法")

    history = case.get("history")
    if not isinstance(history, list):
        add_error(errors, case_id, "history 必须是数组")
    else:
        for message in history:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                add_error(errors, case_id, "history 包含非法 role")
                break
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                add_error(errors, case_id, "history content 不能为空")
                break

    expected = case.get("expected")
    if not isinstance(expected, dict):
        add_error(errors, case_id, "缺少 expected 对象")
        return

    risk = expected.get("risk")
    intent = expected.get("intent")
    mode = expected.get("mode")
    required_tools = expected.get("required_tools")
    forbidden_tools = expected.get("forbidden_tools")

    if risk not in VALID_RISKS:
        add_error(errors, case_id, f"非法 risk: {risk}")
    if mode not in VALID_MODES:
        add_error(errors, case_id, f"非法 mode: {mode}")
    if risk == "safe" and intent not in VALID_INTENTS:
        add_error(errors, case_id, "safe 样本 intent 必须为五类意图之一")
    if risk in {"emergency", "clarify"} and intent is not None:
        add_error(errors, case_id, "安全短路样本 intent 必须为 null")
    if risk == "emergency" and mode != "safety":
        add_error(errors, case_id, "emergency 样本 mode 必须为 safety")
    if risk == "clarify" and mode != "clarify":
        add_error(errors, case_id, "clarify 样本 mode 必须为 clarify")

    for field, tools in (("required_tools", required_tools), ("forbidden_tools", forbidden_tools)):
        if not isinstance(tools, list) or any(tool not in VALID_TOOLS for tool in tools):
            add_error(errors, case_id, f"{field} 只能使用已注册工具")
    if risk in {"emergency", "clarify"} and required_tools:
        add_error(errors, case_id, "安全短路样本 required_tools 必须为空")
    if isinstance(request, dict) and request.get("agent_mode") in {"react", "plan_execute"}:
        if risk == "safe" and mode != request["agent_mode"]:
            add_error(errors, case_id, "显式 agent_mode 必须与 expected.mode 一致")

    if not isinstance(expected.get("answer_keywords"), list) or not expected["answer_keywords"]:
        add_error(errors, case_id, "answer_keywords 必须为非空数组")
    if not isinstance(expected.get("degraded"), bool):
        add_error(errors, case_id, "degraded 必须为布尔值")

    fixtures = case.get("fixture")
    if not isinstance(fixtures, list):
        add_error(errors, case_id, "fixture 必须为数组")
    else:
        for fixture in fixtures:
            if not isinstance(fixture, str) or not (PROJECT_ROOT / fixture).is_file():
                add_error(errors, case_id, f"夹具不存在: {fixture}")
        has_fault_config = any("fault_configs/" in str(fixture) for fixture in fixtures)
        if expected.get("degraded") != has_fault_config:
            add_error(errors, case_id, "degraded 必须与 fault_configs 夹具一致")


def main() -> int:
    try:
        cases = json.loads(CASES_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"无法读取评测集: {exc}")
        return 2
    if not isinstance(cases, list):
        print("评测集根节点必须是数组")
        return 2

    errors: list[str] = []
    seen_ids: set[str] = set()
    for case in cases:
        if isinstance(case, dict):
            validate_case(case, seen_ids, errors)
        else:
            errors.append("存在非对象样本")

    if errors:
        print("Agent 评测集校验失败：")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"PASS: {len(cases)} 条 Agent 评测样本与夹具均有效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
