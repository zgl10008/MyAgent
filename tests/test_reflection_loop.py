# -*- coding: utf-8 -*-
"""反思闭环（ReflectionLoop）三级响应单元测试。

覆盖三种场景：
  - 高质量（≥80）：直接通过
  - 中等质量（60-79）：自我修正
  - 低质量（<60）：触发重试

使用 FakeLLM 控制反思和修正的返回内容，无需真实 LLM。
"""

from __future__ import annotations

import json

from app.core.agent.reflection import (
    ReflectionAgent,
    ReflectionLoop,
    ReflectionLoopResult,
    SelfCorrector,
)


# ==============================================================================
# 工具函数
# ==============================================================================


def _make_llm(responses: list[str]):
    """创建一个按顺序返回字符串的 LLM 替身。"""
    from tests.conftest import FakeLLM
    return FakeLLM(responses)


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


def _empty_steps() -> list[dict]:
    """空步骤列表（无工具调用 = 无 evidence）。"""
    return []


def _steps_with_evidence() -> list[dict]:
    """带 RAG 证据的步骤列表。"""
    return [
        {
            "step": 1,
            "phase": "react",
            "action": "rag_search",
            "action_input": {"query": "高血压用药指南"},
            "observation": "高血压常用药物包括 ACEI、ARB、CCB 等五大类。",
        },
        {
            "step": 2,
            "phase": "react",
            "thought": "已获取足够信息",
        },
    ]


# ==============================================================================
# 场景 1：高质量回答（≥80）— 直接通过
# ==============================================================================


async def test_reflection_loop_high_score_passes_directly() -> None:
    """反思评分 ≥80 → action_taken='none'，不修正不重试。"""
    llm = _make_llm([
        _reflection_json(quality_score=85, summary="回答质量良好"),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="高血压怎么治疗？",
        answer="高血压治疗包括生活方式干预和药物治疗。常用药物有ACEI、ARB、CCB等。",
        steps=_steps_with_evidence(),
    )

    assert result.action_taken == "none", f"应为 none，实际: {result.action_taken}"
    assert result.corrections_made == 0
    assert result.final_score >= 80
    assert result.warnings == []


async def test_reflection_loop_score_exactly_80_passes() -> None:
    """边界值：score==80 应直接通过。"""
    llm = _make_llm([
        _reflection_json(quality_score=80),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="test",
        answer="answer",
        steps=_empty_steps(),
    )
    assert result.action_taken == "none"


async def test_reflection_loop_perfect_score() -> None:
    """满分 100 也应直接通过。"""
    llm = _make_llm([
        _reflection_json(quality_score=100, summary="完美回答"),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="1+1=?",
        answer="2",
        steps=_empty_steps(),
    )
    assert result.action_taken == "none"
    assert result.final_score == 100


# ==============================================================================
# 场景 2：中等质量（60-79）— 自我修正
# ==============================================================================


async def test_reflection_loop_medium_score_triggers_correction() -> None:
    """60-79 分 → action_taken='self_correction'，调用修正器。"""
    # 注意：同一个 FakeLLM 被 ReflectionAgent 和 SelfCorrector 共享
    # 调用顺序：1) 第一次反思  2) SelfCorrector 修正  3) 第二次反思
    llm = _make_llm([
        # 1) 第一次反思 JSON
        _reflection_json(
            quality_score=65,
            likely_hallucination=True,
            hallucination_reasons=["声称某药物治愈高血压，缺乏依据"],
            suggestions=["应注明药物需在医生指导下使用"],
            completeness_notes="未提及生活方式干预",
        ),
        # 2) SelfCorrector 返回修正文本
        "修正后的回答：高血压治疗应在医生指导下进行，常用药物包括ACEI、ARB、CCB等五大类，同时需要生活方式干预如低盐饮食和规律运动。",
        # 3) 第二次反思 JSON（修正后评估）
        _reflection_json(
            quality_score=78,
            likely_hallucination=False,
            hallucination_reasons=[],
            summary="修正后回答更准确",
        ),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="高血压怎么治疗？",
        answer="某药物可以治愈高血压，不需要看医生。",
        steps=_steps_with_evidence(),
    )

    assert result.action_taken == "self_correction"
    assert result.corrections_made == 1
    assert result.final_score > 65  # 修正后应该更高
    assert "不需" not in result.final_answer  # 修正后不应再有错误内容


async def test_reflection_loop_correction_then_worse_keeps_original() -> None:
    """修正后分数反而更低 → 保留原始回答。"""
    llm = _make_llm([
        _reflection_json(quality_score=65, suggestions=["改一下表述"]),
        _reflection_json(quality_score=50, summary="修正后更差"),  # 修正后反而低了
        "糟糕的修正回答...",
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    original = "原始回答，正确但表述不够好。"
    result = await loop.evaluate_and_improve(
        user_query="test",
        answer=original,
        steps=_empty_steps(),
    )

    assert result.final_answer == original  # 保留原始
    assert result.action_taken == "self_correction"
    assert result.corrections_made == 1


async def test_reflection_loop_correction_adds_warning() -> None:
    """原始回答有幻觉 → 警告中加入修正提示。"""
    llm = _make_llm([
        _reflection_json(
            quality_score=62,
            likely_hallucination=True,
            hallucination_reasons=["虚构数据"],
        ),
        _reflection_json(quality_score=75),
        "修正后的回答",
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="test",
        answer="有虚构数据的回答",
        steps=_empty_steps(),
    )

    assert len(result.warnings) > 0
    assert any("修正" in w for w in result.warnings)


async def test_reflection_loop_score_exactly_60_triggers_correction() -> None:
    """边界值：score==60 应触发修正（不是重试）。"""
    llm = _make_llm([
        _reflection_json(quality_score=60),
        _reflection_json(quality_score=72),
        "修正后回答",
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="test",
        answer="answer",
        steps=_empty_steps(),
    )
    assert result.action_taken == "self_correction"


# ==============================================================================
# 场景 3：低质量（<60）— 触发重试
# ==============================================================================


async def test_reflection_loop_low_score_triggers_retry() -> None:
    """<60 分 → action_taken='retry_needed'，返回 RetryContext。"""
    llm = _make_llm([
        _reflection_json(
            quality_score=35,
            likely_hallucination=True,
            hallucination_reasons=["编造了不存在的研究"],
            completeness_notes="完全没有回答用户问题",
            suggestions=["需要重新检索资料后回答"],
            summary="回答严重不合格",
        ),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="糖尿病最新治疗方法有哪些？",
        answer="糖尿病不用治疗，多喝水就好。",
        steps=_empty_steps(),
    )

    assert result.action_taken == "retry_needed"
    assert result.retry_context is not None
    assert result.retry_context.previous_score == 35
    assert len(result.retry_context.hallucination_reasons) > 0
    assert len(result.retry_context.suggestions) > 0
    assert result.final_score == 35  # 重试前的分数


async def test_retry_context_to_hint_format() -> None:
    """RetryContext.to_hint() 生成可注入 Agent 的提示文本。"""
    from app.core.agent.reflection import RetryContext

    ctx = RetryContext(
        hallucination_reasons=["虚构了研究数据", "错误的药物推荐"],
        missing_points="未提及饮食控制",
        suggestions=["请基于权威指南回答"],
        previous_score=30,
    )
    hint = ctx.to_hint()

    assert "虚构了研究数据" in hint
    assert "错误的药物推荐" in hint
    assert "未提及饮食控制" in hint
    assert "30" in hint  # 包含评分


async def test_retry_context_from_reflection() -> None:
    """RetryContext.from_reflection() 从 ReflectionReport 正确构建。"""
    from app.core.agent.reflection import ReflectionReport, RetryContext

    report = ReflectionReport(
        quality_score=25,
        is_complete=False,
        likely_hallucination=True,
        hallucination_reasons=["伪造的临床试验数据"],
        completeness_notes="遗漏了药物治疗和监测方案",
        suggestions=["请重新检索", "对照指南核实"],
        summary="不合格",
    )
    ctx = RetryContext.from_reflection(report)

    assert ctx.previous_score == 25
    assert "伪造的临床试验数据" in ctx.hallucination_reasons
    assert "药物治疗和监测方案" in ctx.missing_points
    assert len(ctx.suggestions) == 2


# ==============================================================================
# 场景 4：异常处理 & 边界
# ==============================================================================


async def test_reflection_loop_with_evidence_collection() -> None:
    """验证 evidence 被正确收集并传递。"""
    llm = _make_llm([
        _reflection_json(quality_score=88),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="高血压用药？",
        answer="ACEI类药物是常用降压药。",
        steps=_steps_with_evidence(),
    )
    assert result.action_taken == "none"
    # 验证 LLM 调用中包含了 evidence（从 steps 提取）
    # FakeLLM.calls[0] = messages 列表: [system_msg, user_msg]
    # user_msg 包含 evidence（外部证据/观察片段）
    assert len(llm.calls) > 0
    user_msg = llm.calls[0][1]["content"] if len(llm.calls[0]) > 1 else ""
    # evidence 中的 observation 内容应该出现在 user prompt 中
    assert "ACEI" in user_msg or "高血压常用药物" in user_msg


async def test_reflection_loop_empty_steps_ok() -> None:
    """无工具调用步骤时也能正常反思（无 evidence）。"""
    llm = _make_llm([
        _reflection_json(quality_score=90),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="hello",
        answer="hi there",
        steps=_empty_steps(),
    )
    assert result.action_taken == "none"


async def test_reflection_loop_score_very_low_with_hallucination() -> None:
    """极低分 + 幻觉 → 触发重试 + 用户警告。"""
    llm = _make_llm([
        _reflection_json(
            quality_score=10,
            likely_hallucination=True,
            hallucination_reasons=["整个回答都是编造的"],
            summary="完全失败",
        ),
    ])
    agent = ReflectionAgent(llm=llm, min_quality_to_pass=60)
    corrector = SelfCorrector(llm=llm)
    loop = ReflectionLoop(agent, corrector)

    result = await loop.evaluate_and_improve(
        user_query="问个问题",
        answer="编造的回答",
        steps=_empty_steps(),
    )

    assert result.action_taken == "retry_needed"
    assert len(result.warnings) > 0
    assert any("不准确" in w for w in result.warnings)
    assert result.retry_context is not None


async def test_self_corrector_returns_original_when_no_issues() -> None:
    """反思没有发现问题 → 修正器返回原回答（不做修改）。"""
    from app.core.agent.reflection import ReflectionReport

    llm = _make_llm([])  # 不会被调用
    corrector = SelfCorrector(llm=llm)

    report = ReflectionReport(
        quality_score=70,
        is_complete=True,
        likely_hallucination=False,
        hallucination_reasons=[],
        completeness_notes="",
        suggestions=[],
        summary="无问题",
    )
    original = "原始回答"
    result = await corrector.correct(original, report, evidence=[])
    assert result == original


async def test_self_corrector_failure_returns_original() -> None:
    """修正 LLM 调用失败 → 返回原始回答。"""
    from app.core.agent.reflection import ReflectionReport

    class _FailingLLM:
        async def acomplete(self, messages, **kwargs):
            raise RuntimeError("LLM 挂了")

    corrector = SelfCorrector(llm=_FailingLLM())

    report = ReflectionReport(
        quality_score=65,
        is_complete=False,
        likely_hallucination=True,
        hallucination_reasons=["有问题"],
        completeness_notes="不完整",
        suggestions=["需要改进"],
        summary="需修正",
    )
    original = "有问题的原始回答"
    result = await corrector.correct(original, report, evidence=[])
    assert result == original  # 失败退回原始


async def test_evidence_collector_deduplication() -> None:
    """重复的 observation 只保留一条。"""
    from app.core.agent.evidence_collector import EvidenceCollector

    collector = EvidenceCollector()
    steps = [
        {"action": "rag_search", "action_input": {"query": "q"}, "observation": "相同内容"},
        {"action": "rag_search", "action_input": {"query": "q2"}, "observation": "相同内容"},
    ]
    snippets = collector.collect(steps)
    assert len(snippets) == 1  # 去重


async def test_evidence_collector_ignores_other_tools() -> None:
    """calculator 工具的 observation 不被收集为 evidence。"""
    from app.core.agent.evidence_collector import EvidenceCollector

    collector = EvidenceCollector()
    steps = [
        {"action": "calculator", "action_input": {"expression": "1+1"}, "observation": "2"},
        {"action": "rag_search", "action_input": {"query": "q"}, "observation": "证据"},
    ]
    snippets = collector.collect(steps)
    assert len(snippets) == 1
    assert "rag_search" in snippets[0]
