# -*- coding: utf-8 -*-
"""意图识别：LLM 分类 + 关键词降级 + 置信度与澄清引导。"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """意图识别结果。"""

    intent: str = Field(description="主意图标签")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0~1")
    slots: dict[str, Any] = Field(default_factory=dict, description="槽位信息")
    sub_intent: str | None = Field(default=None, description="子意图（树形第二层）")
    rationale: str = Field(default="", description="简要理由（可给模型或日志用）")


# ---------------------------------------------------------------------------
# LLM 分类 Prompt
# ---------------------------------------------------------------------------
INTENT_CLASSIFY_PROMPT = """你是意图分类专家。分析用户输入，输出 JSON。

## 意图类别
- 知识问答：询问概念、原理、对比、方法（什么/为什么/如何/区别）
- 实时搜索：需要最新资讯、新闻、事实核查（今天/最近/最新/2026）
- 数学计算：涉及数值运算、表达式求值
- 文档管理：上传、下载、删除文档
- 闲聊对话：问候、感谢、日常寒暄

## 多轮规则
- 若提供了对话历史且当前输入是追问/指代（如"那它呢？""还有吗？""这个药呢？"），
  必须结合历史推断真实意图，不要按字面单轮分类。
- 若没有对话历史或输入本身完整，按单轮分类。

## 槽位提取
- 如涉及数字，放入 numbers 数组
- 如涉及文档，放入 doc_action（upload/delete/list）
- 如涉及对比，放入 compare_entities 数组

## 输出格式（仅 JSON）
{
  "intent": "类别名称",
  "confidence": 0.0~1.0,
  "rationale": "一句中文理由",
  "slots": {}
}

## 示例
用户：今天天气怎么样
输出：{"intent":"实时搜索","confidence":0.92,"rationale":"询问天气需要实时信息","slots":{}}

用户：React和Vue有什么区别
输出：{"intent":"知识问答","confidence":0.95,"rationale":"概念对比属于知识问答","slots":{"compare_entities":["React","Vue"]}}

用户：帮我算一下 123*456
输出：{"intent":"数学计算","confidence":0.90,"rationale":"明确要求数值计算","slots":{"numbers":["123","456"]}}

用户：你好呀
输出：{"intent":"闲聊对话","confidence":0.98,"rationale":"日常问候语","slots":{}}

用户：上传这个 PDF 文档
输出：{"intent":"文档管理","confidence":0.88,"rationale":"明确要求上传文档","slots":{"doc_action":"upload"}}
"""


# ---------------------------------------------------------------------------
# 关键词规则（LLM 不可用时的降级兜底）
# ---------------------------------------------------------------------------
_INTENT_TREE: dict[str, dict[str, list[str]]] = {
    "知识问答": {
        "概念原理": ["什么", "为什么", "如何", "解释", "含义", "区别", "对比", "哪个好", "原理"],
        "教程指导": ["怎么做", "步骤", "教程", "方法", "技巧"],
    },
    "实时搜索": {
        "新闻资讯": ["今天", "最近", "最新", "新闻", "2026", "2025", "目前", "当前", "实时"],
        "事实核查": ["真的吗", "是不是", "有没有"],
    },
    "数学计算": {
        "表达式计算": ["计算", "等于", "算一下", "+", "-", "*", "/"],
        "单位换算": ["换算", "多少", "等于多少"],
    },
    "文档管理": {
        "上传文档": ["上传", "导入", "文档", "pdf", "word"],
        "文档查询": ["文档列表", "有哪些文件", "已上传"],
    },
    "闲聊对话": {
        "日常寒暄": ["你好", "谢谢", "再见", "嗨", "哈喽", "hello"],
        "身份询问": ["你是谁", "你能做什么", "你叫什么"],
    },
}


# ---------------------------------------------------------------------------
# LLM 最小协议（不依赖具体类型）
# ---------------------------------------------------------------------------
class IntentLLM(Protocol):
    """意图分类用 LLM：只需 acomplete 方法。"""

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...


def _extract_json(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取 JSON 对象。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("无法从模型输出中解析 JSON")
    return json.loads(m.group(0))


def _keyword_classify(query: str) -> tuple[str, str | None, float, str]:
    """关键词规则分类（LLM 不可用时的降级方案）。"""
    q = query.strip().lower()
    best_intent = "知识问答"
    best_child: str | None = None
    best_hits = 0

    for intent, children in _INTENT_TREE.items():
        for child, kws in children.items():
            hits = sum(1 for kw in kws if kw.lower() in q)
            if hits > best_hits:
                best_hits = hits
                best_intent = intent
                best_child = child

    if best_hits == 0:
        conf = 0.35 if ("?" in q or "？" in q) else 0.25
        return "知识问答", None, conf, "关键词未命中，弱规则推断"

    conf = min(0.95, 0.45 + 0.12 * best_hits)
    return best_intent, best_child, conf, f"关键词命中 {best_hits} 次"


class IntentRecognizer:
    """意图识别器：优先 LLM 分类，LLM 不可用时降级关键词规则。"""

    def __init__(
        self,
        llm: IntentLLM | None = None,
        confidence_threshold: float = 0.55,
    ) -> None:
        """
        :param llm: 用于意图分类的 LLM 接口（有 acomplete 方法即可）
        :param confidence_threshold: 低于此值触发澄清
        """
        self._llm = llm
        self._threshold = confidence_threshold

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    @property
    def uses_llm(self) -> bool:
        """是否启用 LLM 分类模式。"""
        return self._llm is not None

    async def _llm_classify(
        self,
        query: str,
        history: list[str] | None = None,
    ) -> IntentResult:
        """用 LLM 做意图分类；history 用于多轮追问/指代消解。"""
        user_content = f"请对以下输入做意图分类：\n{query}"
        if history:
            hist_block = "\n".join(f"  {item}" for item in history)
            user_content = (
                "请结合对话历史对当前输入做意图分类。\n"
                f"[对话历史]\n{hist_block}\n\n"
                f"[当前输入]\n{query}"
            )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": INTENT_CLASSIFY_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            raw = await self._llm.acomplete(messages, temperature=0.1)
            data = _extract_json(raw)
        except Exception as exc:
            logger.warning("LLM 意图分类失败，降级关键词: {}", exc)
            raise  # 抛出让上层 catch 走关键词降级

        intent = str(data.get("intent", "知识问答"))
        conf = float(data.get("confidence", 0.7))
        conf = max(0.0, min(1.0, conf))
        rationale = str(data.get("rationale", ""))
        slots: dict[str, Any] = data.get("slots") or {}

        # 补充数字槽位
        if not slots.get("numbers"):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", query)
            if nums:
                slots["numbers"] = nums[:5]

        return IntentResult(
            intent=intent,
            confidence=conf,
            slots=slots,
            sub_intent=None,
            rationale=rationale,
        )

    async def recognize(self, query: str, context: dict | None = None) -> IntentResult:
        """识别用户意图。优先 LLM 分类，失败降级关键词规则。

        context 可携带历史上下文（方案 A）：
            {"history": ["user: ...", "assistant: ...", ...]}
        历史会注入 LLM 分类提示词，用于多轮追问/指代消解。
        """
        history: list[str] | None = None
        if context:
            raw = context.get("history")
            if isinstance(raw, list):
                history = [str(item) for item in raw if str(item).strip()]
            elif isinstance(raw, str) and raw.strip():
                history = [raw.strip()]

        # ── 尝试 LLM 分类 ──
        if self._llm is not None:
            try:
                result = await self._llm_classify(query, history=history)
                logger.info(
                    "LLM 意图 query_preview={} -> {} conf={:.2f} reason={}",
                    query[:80],
                    result.intent,
                    result.confidence,
                    result.rationale,
                )
                return result
            except Exception:
                pass  # 走下方关键词降级

        # ── 关键词降级 ──
        intent, child, conf, why = _keyword_classify(query)
        slots: dict[str, Any] = {}
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", query)
        if nums:
            slots["numbers"] = nums[:5]

        result = IntentResult(
            intent=intent,
            confidence=conf,
            slots=slots,
            sub_intent=child,
            rationale=why,
        )
        logger.info(
            "关键词意图 query_preview={} -> {} conf={:.2f}",
            query[:80],
            intent,
            conf,
        )
        return result

    async def clarify(self, query: str, intent_result: IntentResult) -> str:
        """置信度不足时生成澄清提示。"""
        if intent_result.confidence >= self._threshold:
            return ""

        return (
            f"我不太确定您的具体需求（当前置信度 {intent_result.confidence:.2f}）。"
            "请问您是想要：\n"
            "  • 知识问答（概念、原理、对比）\n"
            "  • 实时搜索（最新资讯、新闻）\n"
            "  • 数学计算\n"
            "  • 文档管理\n"
            "请补充更多信息。"
        )
