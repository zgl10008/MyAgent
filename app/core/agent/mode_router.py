# -*- coding: utf-8 -*-
"""
Agent 模式路由器：根据意图识别结果自动选择最优执行模式。

决策路径：
1. 规则快路径 — 高置信度 + 意图→模式明确映射，零延迟
2. LLM Router — 低置信度 / 模糊场景，用 LLM 二次判断
3. 最终兜底 — react（最安全默认）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.intent.recognizer import IntentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class ModeDecision:
    """模式路由决策结果。"""

    mode: str  # "react" | "plan_execute" | "function_call"
    source: str  # "rule" | "llm_router" | "fallback"
    rationale: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# 规则映射表
# ---------------------------------------------------------------------------

_INTENT_MODE_RULES: dict[str, tuple[str, float]] = {
    "数学计算": ("react", 0.70),
    "实时搜索": ("plan_execute", 0.70),
    "闲聊对话": ("react", 0.0),
    "知识问答": ("react", 0.0),
    "文档管理": ("react", 0.0),
}

# 意图置信度过低 → 交给 LLM Router
_RULE_CONFIDENCE_FLOOR = 0.50

# 时间敏感关键词（知识问答升级为 plan_execute）
_TIME_SENSITIVE_KW = {"今天", "最近", "最新", "目前", "当前", "实时", "新闻", "2025", "2026"}


# ---------------------------------------------------------------------------
# LLM Router Prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """你是 Agent 模式路由器。根据意图分析结果，选择最合适的 Agent 执行模式。

## 可选模式
- react: 思考→行动→观察循环。适合简单问答、闲聊、单步工具调用。
- plan_execute: 先规划再逐步执行（每个子任务走 ReAct 循环），失败时重规划。适合多步复杂任务。

## 决策原则
1. 需要多步协调（搜索+分析+整合）→ plan_execute
2. 简单问答、闲聊、单步计算 → react
3. 意图不明确时保守选择 react
4. 若当前输入是追问/指代（如"那它呢？""还有吗？"），结合对话历史判断真实任务后再选模式；
   没有对话历史或输入完整时按单轮处理。

## 输出格式（仅 JSON，不要其他文字）
{"mode": "react", "rationale": "一句话说明选择理由", "confidence": 0.0~1.0}

## 示例
用户：帮我算 (123+456)*789
信息：意图=数学计算 置信度=0.95
输出：{"mode": "react", "rationale": "单步数学计算，ReAct 可直接调用计算器工具完成", "confidence": 0.90}

用户：最近 AI Agent 领域有什么新进展
信息：意图=实时搜索 置信度=0.72
输出：{"mode": "plan_execute", "rationale": "需要搜索→筛选→总结，多步任务适合先规划再执行", "confidence": 0.80}

用户：你好呀
信息：意图=闲聊对话 置信度=0.98
输出：{"mode": "react", "rationale": "简单问候无需工具，直接回答即可", "confidence": 0.95}

用户：什么是高血压
信息：意图=知识问答 置信度=0.90
输出：{"mode": "react", "rationale": "概念解释类问题，LLM 可直接回答", "confidence": 0.90}"""


def _build_router_user_prompt(
    query: str,
    intent: str,
    confidence: float,
    sub_intent: str | None,
    slots: dict[str, Any],
    tool_names: list[str],
    history: list[str] | None = None,
) -> str:
    hist_block = ""
    if history:
        hist_lines = "\n".join(f"  {item}" for item in history)
        hist_block = f"\n## 对话历史\n{hist_lines}"
    return f"""## 用户查询
{query}
{hist_block}

## 意图分析结果
- 主意图: {intent}
- 置信度: {confidence:.2f}
- 子意图: {sub_intent or "无"}
- 槽位: {json.dumps(slots, ensure_ascii=False) if slots else "无"}

## 上下文
- 可用工具: {", ".join(tool_names) if tool_names else "无"}

请根据以上信息，选择最合适的执行模式并输出 JSON。"""


def _extract_json(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取 JSON 对象。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("无法从 LLM Router 输出中解析 JSON")
    return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# ModeRouter
# ---------------------------------------------------------------------------


class ModeRouter:
    """Agent 模式路由器：规则快路径 + LLM Router 兜底。"""

    VALID_MODES = frozenset({"react", "plan_execute"})

    def __init__(
        self,
        llm: Any = None,
        tool_names: list[str] | None = None,
    ) -> None:
        """
        :param llm: 含 ``acomplete`` 方法的 LLM 封装（复用 IntentRecognizer 的 LLM）
        :param tool_names: 可用工具名列表（含 rag_search 表示知识库可用）
        """
        self._llm = llm
        self._tool_names = list(tool_names or [])

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    async def resolve(
        self,
        query: str,
        intent_result: IntentResult,
        history: list[str] | None = None,
    ) -> ModeDecision:
        """
        根据意图结果自动选择 Agent 模式。

        决策优先级：规则快路径 → LLM Router → react 兜底

        history 为可选会话历史（方案 A），会注入 LLM Router 提示词，
        用于多轮追问/指代场景的模式决策。
        """
        # ── 1. 规则快路径 ──
        decision = self._rule_resolve(query, intent_result)
        if decision is not None:
            logger.info(
                "ModeRouter [规则] query=%.50s intent=%s conf=%.2f → mode=%s",
                query, intent_result.intent, intent_result.confidence, decision.mode,
            )
            return decision

        # ── 2. LLM Router ──
        if self._llm is not None:
            try:
                decision = await self._llm_resolve(
                    query, intent_result, history=history
                )
                logger.info(
                    "ModeRouter [LLM] query=%.50s intent=%s conf=%.2f → mode=%s reason=%s",
                    query, intent_result.intent, intent_result.confidence,
                    decision.mode, decision.rationale,
                )
                return decision
            except Exception as exc:
                logger.warning("ModeRouter LLM 决策失败，降级 react: %s", exc)

        # ── 3. 最终兜底 ──
        logger.info(
            "ModeRouter [兜底] query=%.50s → mode=react", query,
        )
        return ModeDecision(
            mode="react",
            source="fallback",
            rationale="LLM Router 不可用或决策失败，安全默认 react",
            confidence=0.0,
        )

    # ------------------------------------------------------------------
    # 规则快路径
    # ------------------------------------------------------------------

    def _rule_resolve(
        self,
        query: str,
        intent_result: IntentResult,
    ) -> ModeDecision | None:
        """规则匹配成功返回 ModeDecision，否则返回 None 交由 LLM Router。"""
        pair = _INTENT_MODE_RULES.get(intent_result.intent)
        if pair is None:
            return None

        mode, threshold = pair

        # 置信度低于底线 → 交给 LLM Router 二次判断
        if intent_result.confidence < _RULE_CONFIDENCE_FLOOR:
            return None

        # 置信度低于该意图的门槛 → 交给 LLM Router
        if threshold > 0 and intent_result.confidence < threshold:
            return None

        # 知识问答的特殊升级检查
        if intent_result.intent == "知识问答":
            # 时间敏感词 → 升级 plan_execute（多步搜索+整合）
            if any(kw in query for kw in _TIME_SENSITIVE_KW):
                return ModeDecision(
                    mode="plan_execute",
                    source="rule",
                    rationale="知识问答含时间敏感词，升级为 plan_execute 多步搜索",
                    confidence=intent_result.confidence,
                )
            # 数字槽位 → 保持 react（ReAct 循环会通过 FC 后端或正则调用工具）
            # FC 是 ReAct 内部的工具调用优化，不改变范式选择

        return ModeDecision(
            mode=mode,
            source="rule",
            rationale=f"意图 [{intent_result.intent}] conf={intent_result.confidence:.2f} → {mode}",
            confidence=intent_result.confidence,
        )

    # ------------------------------------------------------------------
    # LLM Router
    # ------------------------------------------------------------------

    async def _llm_resolve(
        self,
        query: str,
        intent_result: IntentResult,
        history: list[str] | None = None,
    ) -> ModeDecision:
        """调用 LLM 进行模式决策。"""
        user_prompt = _build_router_user_prompt(
            query=query,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            sub_intent=intent_result.sub_intent,
            slots=intent_result.slots,
            tool_names=self._tool_names,
            history=history,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw = await self._llm.acomplete(messages, temperature=0.1)
        data = _extract_json(raw)

        mode = str(data.get("mode", "react")).strip().lower()
        if mode not in self.VALID_MODES:
            logger.warning("LLM Router 返回非法 mode=%r，降级 react", mode)
            mode = "react"

        rationale = str(data.get("rationale", ""))
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))

        return ModeDecision(
            mode=mode,
            source="llm_router",
            rationale=rationale,
            confidence=conf,
        )
