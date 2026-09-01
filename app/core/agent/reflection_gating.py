# -*- coding: utf-8 -*-
"""反思风险分级：按任务风险决定反思策略，避免低风险任务浪费 LLM 调用。

风险分 = 意图分 + 证据分 + 置信度分 + 降级分 + 领域分
  >= 65  → R3 高风险（完整反思：评估 + 修正 + 可能重试）
  30~64 → R2 中风险（轻量反思：评估 + 修正，不重试）
  < 30  → R1 低风险（跳过反思）

所有因子都来自请求链路里已有的信息，无需新增 LLM 调用，纯函数打分。
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """反思风险等级。"""

    R3_HIGH = "r3_high"      # 完整反思（评估 + 修正 + 可能重试）
    R2_MEDIUM = "r2_medium"  # 轻量反思（评估 + 修正，不重试）
    R1_LOW = "r1_low"        # 跳过反思


# 意图 → 风险基础分
_INTENT_SCORE: dict[str, int] = {
    "知识问答": 50,
    "实时搜索": 50,
    "数学计算": -40,
    "文档管理": -20,
    "闲聊对话": -50,
}
_UNKNOWN_INTENT_SCORE = 30  # 未知意图保守按偏高处理

_EVIDENCE_ACTIONS = {"rag_search", "web_search"}

# 分档阈值
R3_THRESHOLD = 65
R2_THRESHOLD = 30


def _has_retrieval_evidence(steps: list[dict[str, Any]]) -> bool:
    """steps 中是否包含检索类工具调用（有证据反思才有意义）。"""
    return any(s.get("action") in _EVIDENCE_ACTIONS for s in steps)


def score_risk(
    *,
    intent: str,
    confidence: float,
    steps: list[dict[str, Any]],
    degraded: bool,
    domain: str = "general",
) -> int:
    """计算风险分。纯函数，无副作用，便于单元测试。"""
    score = _INTENT_SCORE.get(intent, _UNKNOWN_INTENT_SCORE)

    # 证据分
    if _has_retrieval_evidence(steps):
        score += 20

    # 置信度分：置信度低 = 意图可能判错 = 风险更高
    if confidence < 0.55:
        score += 20
    elif confidence > 0.80:
        score -= 10

    # 降级分：FC→正则降级后解析失败率高，风险高
    if degraded:
        score += 20

    # 领域分：医疗场景整体上调
    if domain == "cardiovascular":
        score += 10

    return score


def classify_risk(score: int) -> RiskLevel:
    """风险分 → 风险等级。"""
    if score >= R3_THRESHOLD:
        return RiskLevel.R3_HIGH
    if score >= R2_THRESHOLD:
        return RiskLevel.R2_MEDIUM
    return RiskLevel.R1_LOW


class ReflectionGating:
    """反思分级器：对外唯一入口，返回该任务的风险等级。"""

    def decide(
        self,
        *,
        intent: str,
        confidence: float,
        steps: list[dict[str, Any]],
        degraded: bool,
        domain: str = "general",
    ) -> RiskLevel:
        score = score_risk(
            intent=intent,
            confidence=confidence,
            steps=steps,
            degraded=degraded,
            domain=domain,
        )
        return classify_risk(score)
