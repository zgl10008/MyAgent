# -*- coding: utf-8 -*-
"""领域配置与安全策略。

通用 Agent/RAG 能力不依赖具体行业；领域策略只负责提示词、风险分流和回答边界。
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DomainPolicy:
    """一个可插拔领域的回答策略。"""

    name: str
    display_name: str
    system_prompt: str
    risk_keywords: tuple[str, ...] = ()
    emergency_notice: str = ""
    disclaimer: str = ""
    clarify_notice: str = ""

    def is_high_risk(self, text: str) -> bool:
        """判断输入是否命中需要人工/急诊分流的高风险信号。"""
        normalized = re.sub(r"\s+", "", text or "").lower()
        return any(keyword.lower() in normalized for keyword in self.risk_keywords)

    def safety_prefix(self, text: str) -> str:
        """对高风险问题生成优先级最高的安全提示。"""
        if self.emergency_notice and self.is_high_risk(text):
            return self.emergency_notice
        return ""

    def with_boundary(self, answer: str, query: str = "") -> str:
        """为普通领域回答追加边界说明；高风险问题优先返回急诊提示。"""
        emergency = self.safety_prefix(query)
        if emergency:
            return emergency
        if not answer.strip() or not self.disclaimer:
            return answer
        if self.disclaimer in answer:
            return answer
        return f"{answer.rstrip()}\n\n{self.disclaimer}"


GENERAL_POLICY = DomainPolicy(
    name="general",
    display_name="通用文档",
    system_prompt=(
        "你是严谨的通用知识助手。仅根据提供的检索上下文作答；"
        "如果上下文不足，请明确说明，不要编造事实。回答尽量引用上下文编号。"
    ),
)

CARDIOVASCULAR_POLICY = DomainPolicy(
    name="cardiovascular",
    display_name="心血管慢病管理",
    system_prompt=(
        "你是心血管慢病知识助手。仅根据提供的检索上下文作答；"
        "不要进行疾病确诊，不要擅自建议停药、换药或调整剂量；"
        "如果信息不足，请建议用户咨询专业医务人员，并在必要时引用上下文编号。"
    ),
    risk_keywords=(
        "胸痛", "胸闷伴冷汗", "呼吸困难", "喘不上气", "意识不清",
        "昏厥", "晕厥", "口角歪斜", "言语不清", "一侧肢体无力",
        "突发剧烈头痛", "心跳骤停",
    ),
    emergency_notice=(
        "你描述的症状可能提示需要紧急医疗评估（例如急性心脑血管事件）。"
        "请立即拨打 120 或前往最近的急诊，不要自行驾车；如身边有人，请让其陪同。"
        "本助手不能进行急诊分诊或替代医生。"
    ),
    disclaimer=(
        "以上内容仅供健康信息参考，不能替代医生诊断、处方或面对面医疗评估；"
        "请勿据此自行停药、换药或调整药物剂量。"
    ),
    clarify_notice=(
        "为了准确判断情况，请确认：你现在是否正出现这些症状？"
        "如果症状正在发作，请立即拨打 120 或前往最近的急诊，不要自行驾车；"
        "如果只是询问知识或回忆既往情况，请继续说明，我会为你解答。"
    ),
)

_POLICIES = {
    GENERAL_POLICY.name: GENERAL_POLICY,
    CARDIOVASCULAR_POLICY.name: CARDIOVASCULAR_POLICY,
}


def get_domain_policy(domain: str | None) -> DomainPolicy:
    """按名称选择领域策略，未知领域安全地回退到通用策略。"""
    key = (domain or CARDIOVASCULAR_POLICY.name).strip().lower()
    return _POLICIES.get(key, GENERAL_POLICY)
