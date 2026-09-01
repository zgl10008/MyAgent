# -*- coding: utf-8 -*-
"""反思风险分级（ReflectionGating）单元测试。

覆盖：
  - score_risk / classify_risk 纯函数的分档与边界
  - ReflectionLoop.allow_retry=False 时低质量降级为自我修正（不重跑 Agent）
"""

from __future__ import annotations

import json

from app.core.agent.reflection_gating import (
    ReflectionGating,
    RiskLevel,
    classify_risk,
    score_risk,
)


def _steps(actions: list[str]) -> list[dict]:
    """构造带 action 的步骤列表（供证据识别使用）。"""
    return [{"action": a, "observation": "x"} for a in actions]


# ==============================================================================
# score_risk / classify_risk 纯函数测试
# ==============================================================================


def test_high_risk_qa_with_evidence_is_r3() -> None:
    score = score_risk(
        intent="知识问答", confidence=0.88,
        steps=_steps(["rag_search"]), degraded=False, domain="cardiovascular",
    )
    assert score >= 65
    assert classify_risk(score) == RiskLevel.R3_HIGH


def test_low_risk_chitchat_is_r1() -> None:
    score = score_risk(intent="闲聊对话", confidence=0.98, steps=[], degraded=False)
    assert score < 30
    assert classify_risk(score) == RiskLevel.R1_LOW


def test_calculation_is_r1() -> None:
    score = score_risk(
        intent="数学计算", confidence=0.90, steps=[], degraded=False, domain="cardiovascular",
    )
    assert classify_risk(score) == RiskLevel.R1_LOW


def test_document_management_is_r1() -> None:
    score = score_risk(
        intent="文档管理", confidence=0.88, steps=[], degraded=False, domain="cardiovascular",
    )
    assert classify_risk(score) == RiskLevel.R1_LOW


def test_unknown_intent_never_r1_when_uncertain() -> None:
    """未知意图 + 降级 + 低置信 → 保守不跳过。"""
    score = score_risk(intent="未知意图", confidence=0.4, steps=[], degraded=True)
    assert classify_risk(score) != RiskLevel.R1_LOW


def test_degraded_raises_risk() -> None:
    base = score_risk(intent="知识问答", confidence=0.6, steps=[], degraded=False)
    degraded = score_risk(intent="知识问答", confidence=0.6, steps=[], degraded=True)
    assert degraded > base


def test_low_confidence_raises_risk() -> None:
    high_conf = score_risk(intent="知识问答", confidence=0.9, steps=[], degraded=False)
    low_conf = score_risk(intent="知识问答", confidence=0.4, steps=[], degraded=False)
    assert low_conf > high_conf


def test_domain_upweights() -> None:
    general = score_risk(
        intent="知识问答", confidence=0.7, steps=[], degraded=False, domain="general",
    )
    medical = score_risk(
        intent="知识问答", confidence=0.7, steps=[], degraded=False, domain="cardiovascular",
    )
    assert medical > general


def test_qa_without_evidence_falls_to_r2() -> None:
    """知识问答 + 无证据 + 高置信 + 通用 → R2（不跳过也不完整重试）。"""
    score = score_risk(intent="知识问答", confidence=0.9, steps=[], degraded=False, domain="general")
    assert 30 <= score < 65
    assert classify_risk(score) == RiskLevel.R2_MEDIUM


def test_gating_decide_returns_risk_level() -> None:
    gating = ReflectionGating()
    low = gating.decide(intent="闲聊对话", confidence=0.98, steps=[], degraded=False)
    assert low == RiskLevel.R1_LOW
    high = gating.decide(
        intent="知识问答", confidence=0.88,
        steps=_steps(["rag_search"]), degraded=False, domain="cardiovascular",
    )
    assert high == RiskLevel.R3_HIGH


# ==============================================================================
# ReflectionLoop.allow_retry=False 集成测试
# ==============================================================================


def _reflection_json(quality_score: int, **overrides) -> str:
    """生成反思 LLM 返回的 JSON 字符串。"""
    data = {
        "quality_score": quality_score,
        "is_complete": True,
        "likely_hallucination": False,
        "hallucination_reasons": [],
        "completeness_notes": "",
        "suggestions": [],
        "summary": "自动审查完成",
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


async def test_allow_retry_false_downgrades_to_self_correction() -> None:
    """allow_retry=False 时，<60 分应降级为自我修正，而非返回 retry_needed。"""
    from app.core.agent.reflection import ReflectionAgent, ReflectionLoop, SelfCorrector
    from tests.conftest import FakeLLM

    llm = FakeLLM([
        _reflection_json(35, likely_hallucination=True, hallucination_reasons=["编造"]),
        "修正后的回答",
        _reflection_json(70),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="test",
        answer="有问题的回答",
        steps=[],
        allow_retry=False,
    )

    assert result.action_taken == "self_correction"
    assert result.retry_context is None
    assert result.corrections_made == 1


async def test_allow_retry_true_still_retries() -> None:
    """allow_retry=True（默认）时，<60 分仍返回 retry_needed。"""
    from app.core.agent.reflection import ReflectionAgent, ReflectionLoop, SelfCorrector
    from tests.conftest import FakeLLM

    llm = FakeLLM([
        _reflection_json(35, likely_hallucination=True, hallucination_reasons=["编造"]),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="test",
        answer="有问题的回答",
        steps=[],
        allow_retry=True,
    )

    assert result.action_taken == "retry_needed"
    assert result.retry_context is not None
