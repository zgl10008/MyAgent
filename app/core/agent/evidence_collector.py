# -*- coding: utf-8 -*-
"""Evidence 收集器：从 Agent 执行步骤中提取外部证据片段用于反思幻觉检测。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvidenceSnippet:
    """单条外部证据。"""
    source: str       # "rag_search" | "web_search" | "calculator"
    query: str        # 检索时用的 query
    content: str      # 返回的内容（截断前 2000 字符）


class EvidenceCollector:
    """从 Agent 执行步骤中提取工具调用结果作为反思证据。

    目前反思的 evidence_snippets 参数从未被传入，反思 LLM 只能凭空判断幻觉。
    本收集器从 step 的 observation 中提取 RAG 和搜索的返回内容，
    使反思能对照实际证据判断回答是否虚构。
    """

    MAX_CONTENT_LENGTH = 2000
    SOURCE_TOOLS = frozenset({"rag_search", "web_search"})

    def collect(self, steps: list[dict[str, Any]]) -> list[str]:
        """提取所有工具调用结果，格式化为证据文本列表。"""
        snippets: list[str] = []
        seen: set[str] = set()  # 去重

        for step in steps:
            action = (
                step.get("action")
                or step.get("parsed", {}).get("action")
                or ""
            )
            if action not in self.SOURCE_TOOLS:
                continue

            obs = str(step.get("observation") or "")
            if not obs.strip():
                continue

            # 去重：相同 action + 相同内容的 observation 只保留一条
            dedup_key = f"{action}:{obs[:200]}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            content = obs[:self.MAX_CONTENT_LENGTH]
            snippets.append(f"[来源: {action}]\n{content}")

        return snippets

    def collect_structured(self, steps: list[dict[str, Any]]) -> list[EvidenceSnippet]:
        """提取结构化证据（保留来源和查询信息）。"""
        snippets: list[EvidenceSnippet] = []
        seen: set[str] = set()

        for step in steps:
            action = (
                step.get("action")
                or step.get("parsed", {}).get("action")
                or ""
            )
            if action not in self.SOURCE_TOOLS:
                continue

            obs = str(step.get("observation") or "")
            if not obs.strip():
                continue

            action_input = str(
                step.get("action_input")
                or step.get("parsed", {}).get("action_input")
                or ""
            )

            dedup_key = f"{action}:{obs[:200]}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            snippets.append(EvidenceSnippet(
                source=action,
                query=action_input[:500],
                content=obs[:self.MAX_CONTENT_LENGTH],
            ))

        return snippets
