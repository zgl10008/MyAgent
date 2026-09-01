"""可插拔领域策略。"""

from app.core.domain.policy import (
    CARDIOVASCULAR_POLICY,
    GENERAL_POLICY,
    DomainPolicy,
    get_domain_policy,
)
from app.core.domain.risk_classifier import (
    RiskDecision,
    RiskDisposition,
    classify_risk,
)

__all__ = [
    "CARDIOVASCULAR_POLICY",
    "GENERAL_POLICY",
    "DomainPolicy",
    "get_domain_policy",
    "RiskDecision",
    "RiskDisposition",
    "classify_risk",
]
