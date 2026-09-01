# -*- coding: utf-8 -*-
"""医疗风险分级（关键词粗筛 + LLM 确认）的单元测试。"""

from __future__ import annotations

import json

from app.core.domain.policy import CARDIOVASCULAR_POLICY
from app.core.domain.risk_classifier import (
    RiskDisposition,
    classify_risk,
)


class _FakeRiskLLM:
    """按预设标签返回 JSON 的假 LLM。"""

    def __init__(self, level: str) -> None:
        self._level = level

    async def acomplete(self, messages, **kwargs):
        return json.dumps({"level": self._level, "reason": "测试依据"})


class _BrokenLLM:
    async def acomplete(self, messages, **kwargs):
        raise RuntimeError("LLM 不可用")


async def test_no_keyword_is_safe_without_llm():
    decision = await classify_risk(CARDIOVASCULAR_POLICY, "如何记录血压")
    assert decision.disposition is RiskDisposition.SAFE


async def test_hypothetical_question_is_clarify():
    # 「如果我胸痛怎么办」是条件句，不直接拒答，而是追问确认
    llm = _FakeRiskLLM("clarify")
    decision = await classify_risk(CARDIOVASCULAR_POLICY, "如果我胸痛怎么办", llm)
    assert decision.disposition is RiskDisposition.CLARIFY
    assert decision.answer


async def test_past_history_is_safe():
    llm = _FakeRiskLLM("safe")
    decision = await classify_risk(CARDIOVASCULAR_POLICY, "我以前胸痛、呼吸困难", llm)
    assert decision.disposition is RiskDisposition.SAFE


async def test_ongoing_symptom_is_emergency():
    llm = _FakeRiskLLM("emergency")
    decision = await classify_risk(CARDIOVASCULAR_POLICY, "我现在突然胸痛、呼吸困难", llm)
    assert decision.disposition is RiskDisposition.EMERGENCY
    assert "120" in decision.answer


async def test_concept_question_is_safe():
    llm = _FakeRiskLLM("safe")
    decision = await classify_risk(CARDIOVASCULAR_POLICY, "胸痛是冠心病的症状吗", llm)
    assert decision.disposition is RiskDisposition.SAFE


async def test_no_llm_falls_back_to_emergency():
    decision = await classify_risk(CARDIOVASCULAR_POLICY, "我现在突然胸痛")
    assert decision.disposition is RiskDisposition.EMERGENCY


async def test_llm_failure_falls_back_to_emergency():
    decision = await classify_risk(CARDIOVASCULAR_POLICY, "我现在突然胸痛", _BrokenLLM())
    assert decision.disposition is RiskDisposition.EMERGENCY
