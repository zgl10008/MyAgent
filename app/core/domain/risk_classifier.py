# -*- coding: utf-8 -*-
"""医疗风险分级：关键词粗筛 + LLM 确认，区分「正在发作 / 询问知识 / 需要澄清」。

关键词匹配是「高召回、低精确」的粗筛——医疗场景里漏拦（假阴性）的代价远高于
误拦（假阳性），所以先用关键词把所有含风险信号的输入都抓出来；再由 LLM 做
「高精确」确认，把假设性提问、既往史、概念咨询这类无害场景放行，避免一刀切
误伤。LLM 不可用或确认失败时，保守降级为 EMERGENCY（与纯关键词行为一致）。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

from app.core.domain.policy import DomainPolicy

logger = logging.getLogger(__name__)


class RiskDisposition(str, Enum):
    """风险分级处置结果。"""

    SAFE = "safe"          # 放行：正常流程，末尾追加 disclaimer
    CLARIFY = "clarify"    # 追问：确认是否正在发作
    EMERGENCY = "emergency"  # 急诊：短路返回急诊话术


@dataclass
class RiskDecision:
    """一次风险分级的结论。"""

    disposition: RiskDisposition
    reason: str = ""
    answer: str = ""  # EMERGENCY / CLARIFY 时返回给用户的话术


class RiskLLM(Protocol):
    """风险确认阶段使用的 LLM，复用 acomplete 接口。"""

    async def acomplete(self, messages: Sequence[dict], **kwargs: Any) -> str:
        ...


RISK_CLASSIFY_SYSTEM_PROMPT = """你是医疗分诊助手。用户输入可能包含与急重症相关的症状词。
请判断用户是「正在自述症状发作」，还是「询问知识 / 假设 / 回忆既往史」。

只输出一个 JSON 对象，格式：
{
  "level": "emergency" | "clarify" | "safe",
  "reason": "一句话判断依据"
}

判定标准：
- emergency：用户描述自己或他人「现在 / 正在 / 刚刚 / 突然」出现胸痛、呼吸困难、
  意识不清等急性症状，即正在发作，需立即就医。
- clarify：出现症状词，但无法确定是否正在发作——如仅提症状名、或「如果 / 假设 / 万一」
  等条件句、或时态不明。
- safe：明确是询问概念或科普（如「胸痛是什么病」）、既往史（以前 / 曾经 / 过去 / 上次 / 去年），
  或否定（没有 / 未出现），不含正在发作的急性症状。
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("无法解析 JSON")
    return json.loads(m.group(0))


async def classify_risk(
    policy: DomainPolicy,
    text: str,
    llm: RiskLLM | None = None,
) -> RiskDecision:
    """两阶段风险分级：关键词粗筛 → LLM 确认。

    第一阶段（粗筛，高召回）：未命中关键词直接放行，零 LLM 成本。
    第二阶段（确认，高精确）：命中关键词后交给 LLM 判断是否真的在发作。
    无 LLM 或 LLM 失败时，保守降级为 EMERGENCY，优先保证不漏拦正在发作的信号。
    """
    # 第一阶段：粗筛
    if not policy.is_high_risk(text):
        return RiskDecision(RiskDisposition.SAFE, reason="未命中高风险关键词")

    # 第二阶段：LLM 确认
    if llm is None:
        return RiskDecision(
            RiskDisposition.EMERGENCY,
            reason="命中高风险关键词，但未提供 LLM，保守按急诊处理",
            answer=policy.safety_prefix(text),
        )

    messages: Sequence[dict] = [
        {"role": "system", "content": RISK_CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f"用户输入：{text}"},
    ]
    try:
        raw = await llm.acomplete(messages, temperature=0.0)
        data = _extract_json(raw)
        level = str(data.get("level", "clarify")).strip().lower()
    except Exception as e:  # noqa: BLE001
        logger.warning("风险分级 LLM 确认失败，保守按急诊处理: %s", e)
        return RiskDecision(
            RiskDisposition.EMERGENCY,
            reason="LLM 确认失败，保守按急诊处理",
            answer=policy.safety_prefix(text),
        )

    reason = str(data.get("reason", ""))
    if level == "emergency":
        return RiskDecision(
            RiskDisposition.EMERGENCY,
            reason=reason,
            answer=policy.safety_prefix(text),
        )
    if level == "safe":
        return RiskDecision(RiskDisposition.SAFE, reason=reason)
    # 默认 clarify（含 "clarify" 及任何无法识别的标签）
    return RiskDecision(
        RiskDisposition.CLARIFY,
        reason=reason,
        answer=policy.clarify_notice,
    )
