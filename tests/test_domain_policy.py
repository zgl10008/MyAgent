from app.core.domain.policy import CARDIOVASCULAR_POLICY, GENERAL_POLICY, get_domain_policy


def test_policy_defaults_to_cardiovascular():
    assert get_domain_policy(None) is CARDIOVASCULAR_POLICY


def test_cardiovascular_high_risk_gets_emergency_notice():
    answer = CARDIOVASCULAR_POLICY.with_boundary("请先观察。", "突然胸痛并呼吸困难")
    assert "120" in answer
    assert "自行停药" not in answer


def test_cardiovascular_normal_answer_gets_boundary():
    answer = CARDIOVASCULAR_POLICY.with_boundary("建议记录血压。", "如何记录血压")
    assert "不能替代医生诊断" in answer


def test_general_policy_does_not_add_medical_disclaimer():
    answer = GENERAL_POLICY.with_boundary("文档内容如下。", "如何配置产品")
    assert "医生诊断" not in answer
