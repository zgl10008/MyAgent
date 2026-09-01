# -*- coding: utf-8 -*-
"""
反思 Agent：对输出做质量检查、幻觉与完整性评估，并给出改进建议。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


REFLECTION_SYSTEM_PROMPT = """你是严格的输出质量审查员。请对用户问题与助手回答进行审查。

## 你必须只输出一个 JSON 对象，格式如下：
{
  "quality_score": 0-100 的整数,
  "is_complete": true/false,
  "likely_hallucination": true/false,
  "hallucination_reasons": ["若怀疑幻觉，列出具体疑点；否则为空数组"],
  "completeness_notes": "是否遗漏关键要点",
  "suggestions": ["可执行的改进建议，面向助手"],
  "summary": "一句中文总结审查结论"
}

审查标准：
- 幻觉：回答是否包含无依据的具体事实、虚构来源或与用户问题无关的断言。
- 完整性：是否覆盖用户问题的核心子问题。
- 质量分：综合考虑正确性、清晰度与有用性。
"""


@dataclass
class ReflectionReport:
    """反思审查报告。"""

    quality_score: int
    is_complete: bool
    likely_hallucination: bool
    hallucination_reasons: List[str]
    completeness_notes: str
    suggestions: List[str]
    summary: str
    raw_model_output: Optional[str] = None
    parse_error: Optional[str] = None


class ReflectionLLM(Protocol):
    """反思阶段使用的 LLM。"""

    async def acomplete(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        ...


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("无法解析 JSON")
    return json.loads(m.group(0))


class ReflectionAgent:
    """对 Agent 最终输出进行反思与质量把关。"""

    def __init__(self, llm: ReflectionLLM, min_quality_to_pass: int = 60) -> None:
        self._llm = llm
        self.min_quality_to_pass = max(0, min(100, min_quality_to_pass))

    async def reflect(
        self,
        user_query: str,
        agent_answer: str,
        evidence_snippets: Optional[List[str]] = None,
        trace_summary: Optional[str] = None,
    ) -> ReflectionReport:
        """
        对助手答案进行质量检查。

        evidence_snippets：可选的检索或工具观察片段，用于对照幻觉。
        trace_summary：可选的执行轨迹摘要，用于判断推理是否支撑结论。
        """
        ev_block = "\n".join(f"- {s}" for s in (evidence_snippets or [])) or "（无外部证据）"
        trace_block = trace_summary or "（无轨迹）"
        user_content = f"""## 用户问题
{user_query}

## 助手回答
{agent_answer}

## 外部证据/观察片段
{ev_block}

## 执行轨迹摘要
{trace_block}

请输出 JSON 审查结果。"""

        messages: Sequence[Dict[str, str]] = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = await self._llm.acomplete(messages, temperature=0.1)
            data = _extract_json(raw)
        except Exception as e:  # noqa: BLE001
            logger.exception("反思模型调用或解析失败")
            return ReflectionReport(
                quality_score=50,
                is_complete=False,
                likely_hallucination=False,
                hallucination_reasons=[],
                completeness_notes="反思阶段解析失败，无法完成自动审查",
                suggestions=["请人工复核该回答"],
                summary="反思流程异常，已降级为保守评分",
                raw_model_output=None,
                parse_error=str(e),
            )

        try:
            score = int(data.get("quality_score", 0))
        except (TypeError, ValueError):
            score = 50

        report = ReflectionReport(
            quality_score=max(0, min(100, score)),
            is_complete=bool(data.get("is_complete", False)),
            likely_hallucination=bool(data.get("likely_hallucination", False)),
            hallucination_reasons=list(data.get("hallucination_reasons") or []),
            completeness_notes=str(data.get("completeness_notes", "")),
            suggestions=list(data.get("suggestions") or []),
            summary=str(data.get("summary", "")),
            raw_model_output=raw[:8000],
        )
        return report

    def should_retry_or_warn(self, report: ReflectionReport) -> Dict[str, Any]:
        """
        根据报告给出是否建议重试/告警的企业级决策结构。
        """
        warn = report.quality_score < self.min_quality_to_pass
        retry = report.likely_hallucination or not report.is_complete
        return {
            "warn_low_quality": warn,
            "suggest_retry": retry,
            "min_quality_threshold": self.min_quality_to_pass,
        }


# ==============================================================================
# 自我修正器：对中等质量回答进行单次 LLM 修复，比完整重试更便宜
# ==============================================================================

SELF_CORRECTION_PROMPT = """你是答案修正专家。请根据审查意见修正助手回答。

## 审查意见
{reflection_feedback}

## 原始回答
{original_answer}

## 外部证据（用于核实和修正）
{evidence}

请直接输出修正后的完整回答，不要输出 JSON 或其他格式。
修正规则：
- 移除审查指出的虚构事实和无依据断言
- 补充遗漏的关键要点
- 保留正确的部分不做改动
- 修正后不得新增虚构事实
- 如果原始回答基本正确，只需微调"""


class SelfCorrector:
    """单次 LLM 调用的答案修正器。

    用于质量分在 60-79 区间的回答：比重新跑 Agent 便宜得多（1 次 LLM vs 完整 ReAct 循环），
    但只能修复局部问题（措辞、幻觉片段、遗漏点），无法补充需要工具调用的信息。
    """

    def __init__(self, llm: ReflectionLLM) -> None:
        self._llm = llm

    async def correct(
        self,
        original_answer: str,
        reflection: "ReflectionReport",
        evidence: list[str],
    ) -> str:
        """根据反思报告修正回答，返回修正后的文本。"""
        feedback_parts: list[str] = []

        if reflection.hallucination_reasons:
            feedback_parts.append(
                "疑似幻觉（需核实或移除）：" + "；".join(reflection.hallucination_reasons)
            )
        if reflection.suggestions:
            feedback_parts.append(
                "改进建议：" + "；".join(reflection.suggestions)
            )
        if not reflection.is_complete:
            feedback_parts.append(
                "完整性不足（需补充）：" + reflection.completeness_notes
            )

        if not feedback_parts:
            return original_answer

        ev_block = "\n".join(f"- {e}" for e in evidence) if evidence else "（无外部证据）"

        prompt = SELF_CORRECTION_PROMPT.format(
            reflection_feedback="\n".join(feedback_parts),
            original_answer=original_answer,
            evidence=ev_block,
        )

        try:
            raw = await self._llm.acomplete(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            corrected = raw if isinstance(raw, str) else str(raw)
            return corrected.strip() or original_answer
        except Exception:
            logger.exception("自我修正失败，退回原始回答")
            return original_answer


# ==============================================================================
# 重试上下文：携带前一轮反思的反馈，注入 Agent 重试提示
# ==============================================================================

@dataclass
class RetryContext:
    """携带反思反馈的 Agent 重试上下文。"""

    hallucination_reasons: list[str]
    missing_points: str
    suggestions: list[str]
    previous_score: int

    def to_hint(self) -> str:
        """生成注入 Agent 的反思反馈提示。"""
        parts: list[str] = []
        if self.hallucination_reasons:
            parts.append(
                "【重要】上一轮回答存在以下疑似虚构内容，请避免重复："
                + "；".join(self.hallucination_reasons)
            )
        if self.missing_points:
            parts.append(
                "【重要】上一轮遗漏了以下要点，请补充："
                + self.missing_points
            )
        if self.suggestions:
            parts.append(
                "改进方向：" + "；".join(self.suggestions)
            )
        if self.previous_score < 40:
            parts.append(
                "上一轮质量评分仅 {}/100，请重新组织思路，确保回答准确完整。".format(
                    self.previous_score
                )
            )
        return "\n".join(parts)

    @classmethod
    def from_reflection(cls, report: "ReflectionReport") -> "RetryContext":
        """从反思报告构建重试上下文。"""
        return cls(
            hallucination_reasons=list(report.hallucination_reasons),
            missing_points=report.completeness_notes,
            suggestions=list(report.suggestions),
            previous_score=report.quality_score,
        )


# ==============================================================================
# 反思闭环：评估 → 决策 → 修正/重试 → 再评估
# ==============================================================================

@dataclass
class ReflectionLoopResult:
    """反思闭环的最终结果。"""

    final_answer: str
    final_score: int
    reflection_report: "ReflectionReport"
    corrections_made: int
    action_taken: str  # "none" | "self_correction" | "retry_needed"
    warnings: list[str]

    # 仅 action_taken == "retry_needed" 时有效
    retry_context: "RetryContext | None" = None


class ReflectionLoop:
    """反思驱动的质量控制循环。

    流程：反思 → 分级响应
      score >= 80  → 直接通过
      score 60-79  → 自我修正（单次 LLM 修复）
      score < 60   → 返回 RetryContext，由编排器重新执行 Agent
    """

    PASS_THRESHOLD = 80
    CORRECTION_THRESHOLD = 60

    def __init__(
        self,
        reflection_agent: "ReflectionAgent",
        self_corrector: "SelfCorrector",
    ) -> None:
        self._reflect = reflection_agent
        self._correct = self_corrector

    async def evaluate_and_improve(
        self,
        user_query: str,
        answer: str,
        steps: list[dict],
        allow_retry: bool = True,
    ) -> ReflectionLoopResult:
        """评估回答质量，必要时触发修正或标记需要重试。

        allow_retry=False 时，低质量回答降级为单次自我修正，
        而不是返回 retry_needed（用于中风险任务，避免重跑整个 Agent）。
        """
        from app.core.agent.evidence_collector import EvidenceCollector

        collector = EvidenceCollector()
        evidence = collector.collect(steps)

        trace_summary = str([
            s.get("phase") or s.get("subtask_id", "")
            for s in steps[-20:]
        ])

        # 第一次反思
        report = await self._reflect.reflect(
            user_query,
            answer,
            evidence_snippets=evidence,
            trace_summary=trace_summary,
        )
        best_answer = answer
        best_score = report.quality_score
        best_report = report
        corrections = 0
        warnings: list[str] = []

        # ── 高质量：直接通过 ──
        if best_score >= self.PASS_THRESHOLD:
            return ReflectionLoopResult(
                final_answer=best_answer,
                final_score=best_score,
                reflection_report=best_report,
                corrections_made=0,
                action_taken="none",
                warnings=[],
            )

        # ── 中等质量：自我修正 ──
        if best_score >= self.CORRECTION_THRESHOLD:
            corrected = await self._correct.correct(answer, report, evidence)
            if corrected != answer:
                corrections = 1

            # 修正后再反思
            report2 = await self._reflect.reflect(
                user_query,
                corrected,
                evidence_snippets=evidence,
            )
            if report2.quality_score > best_score:
                best_answer = corrected
                best_score = report2.quality_score
                best_report = report2

            if report.likely_hallucination:
                warnings.append("回答经自动修正，可能存在不准确内容，建议核实")

            return ReflectionLoopResult(
                final_answer=best_answer,
                final_score=best_score,
                reflection_report=best_report,
                corrections_made=corrections,
                action_taken="self_correction",
                warnings=warnings,
            )

        # ── 低质量 ──
        # allow_retry=False（中风险 R2）：降级为单次自我修正，不重跑 Agent。
        if not allow_retry:
            corrected = await self._correct.correct(answer, report, evidence)
            if corrected != answer:
                corrections = 1

            # 修正后再反思
            report2 = await self._reflect.reflect(
                user_query,
                corrected,
                evidence_snippets=evidence,
            )
            if report2.quality_score > best_score:
                best_answer = corrected
                best_score = report2.quality_score
                best_report = report2

            if report.likely_hallucination:
                warnings.append("回答经自动修正，可能存在不准确内容，建议核实")

            return ReflectionLoopResult(
                final_answer=best_answer,
                final_score=best_score,
                reflection_report=best_report,
                corrections_made=corrections,
                action_taken="self_correction",
                warnings=warnings,
            )

        # ── 低质量：需要完整重试（高风险 R3）──
        retry_ctx = RetryContext.from_reflection(report)
        if report.likely_hallucination and report.quality_score < 40:
            warnings.append("回答可能包含不准确信息，系统已尝试重新生成")

        return ReflectionLoopResult(
            final_answer=best_answer,
            final_score=best_score,
            reflection_report=best_report,
            corrections_made=0,
            action_taken="retry_needed",
            warnings=warnings,
            retry_context=retry_ctx,
        )
