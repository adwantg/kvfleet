"""Tests for policy engine, PII detection, and tenant management."""

import pytest

from kvfleet.config.schema import (
    BudgetConfig,
    ModelConfig,
    PolicyConfig,
    PolicyRule,
    ProviderType,
    TenantConfig,
)
from kvfleet.policy.engine import PolicyContext, PolicyEngine
from kvfleet.policy.pii import PIIDetector
from kvfleet.policy.residency import ResidencyEngine, ResidencyRule
from kvfleet.policy.tenant import BudgetTracker, TenantManager

# ───────────────────────── Policy Engine ─────────────────────────


@pytest.fixture
def models():
    return [
        ModelConfig(
            name="local-llama",
            endpoint="http://a:8000",
            provider=ProviderType.VLLM,
            allowed_data_classes=["public", "internal", "confidential"],
        ),
        ModelConfig(
            name="openai-gpt",
            endpoint="https://api.openai.com",
            provider=ProviderType.OPENAI_COMPAT,
            allowed_data_classes=["public"],
        ),
    ]


class TestPolicyEngine:
    def test_disabled_passes_all(self, models):
        engine = PolicyEngine(PolicyConfig(enabled=False))
        filtered, decisions = engine.evaluate(models, PolicyContext())
        assert len(filtered) == 2
        assert len(decisions) == 0

    def test_data_class_filter(self, models):
        engine = PolicyEngine(PolicyConfig(enabled=True))
        ctx = PolicyContext(data_class="confidential")
        filtered, _decisions = engine.evaluate(models, ctx)
        assert len(filtered) == 1
        assert filtered[0].name == "local-llama"

    def test_pii_filter(self, models):
        engine = PolicyEngine(PolicyConfig(enabled=True, pii_detection=True))
        ctx = PolicyContext(has_pii=True)
        filtered, _decisions = engine.evaluate(models, ctx)
        # Only private models should remain
        assert all(
            m.provider.value in ("vllm", "triton", "tgi", "ollama", "custom_http") for m in filtered
        )

    def test_custom_rule_block(self, models):
        rule = PolicyRule(
            name="block-openai",
            condition="always",
            action="block",
            target_models=["openai-gpt"],
        )
        engine = PolicyEngine(PolicyConfig(enabled=True, rules=[rule]))
        filtered, _decisions = engine.evaluate(models, PolicyContext())
        assert len(filtered) == 1
        assert filtered[0].name == "local-llama"

    def test_custom_rule_require(self, models):
        rule = PolicyRule(
            name="require-local",
            condition="data_class == confidential",
            action="require_model",
            target_models=["local-llama"],
        )
        engine = PolicyEngine(PolicyConfig(enabled=True, rules=[rule]))
        ctx = PolicyContext(data_class="confidential")
        filtered, _decisions = engine.evaluate(models, ctx)
        assert len(filtered) == 1
        assert filtered[0].name == "local-llama"


# ───────────────────────── PII Detector ─────────────────────────


class TestPIIDetector:
    def test_detect_email(self):
        detector = PIIDetector()
        result = detector.detect("Contact me at user@example.com for details.")
        assert result.has_pii
        assert "email" in result.pii_types

    def test_detect_phone(self):
        detector = PIIDetector()
        result = detector.detect("Call (555) 123-4567 now.")
        assert result.has_pii
        assert "phone_us" in result.pii_types

    def test_detect_ssn(self):
        detector = PIIDetector()
        result = detector.detect("SSN: 123-45-6789")
        assert result.has_pii
        assert "ssn" in result.pii_types

    def test_detect_credit_card(self):
        detector = PIIDetector()
        result = detector.detect("Card: 4111-1111-1111-1111")
        assert result.has_pii
        assert "credit_card" in result.pii_types

    def test_no_pii(self):
        detector = PIIDetector()
        result = detector.detect("The weather is sunny today.")
        assert not result.has_pii

    def test_redact(self):
        detector = PIIDetector()
        result = detector.redact("Email me at user@example.com")
        assert "[REDACTED](email)" in result.redacted_text

    def test_has_pii_quick(self):
        detector = PIIDetector()
        assert detector.has_pii("user@example.com")
        assert not detector.has_pii("no pii here")


# ───────────────────────── Residency ─────────────────────────


class TestResidencyEngine:
    def test_no_rules(self):
        engine = ResidencyEngine()
        assert engine.is_compliant("us-east-1", "us-west-2", "vllm")

    def test_region_restriction(self):
        rules = [
            ResidencyRule(
                name="eu-data",
                source_regions=["eu-west-1", "eu-central-1"],
                allowed_model_regions=["eu-west-1", "eu-central-1"],
            )
        ]
        engine = ResidencyEngine(rules=rules)
        assert engine.is_compliant("eu-west-1", "eu-west-1", "vllm")
        assert not engine.is_compliant("eu-west-1", "us-east-1", "vllm")

    def test_blocked_provider(self):
        rules = [
            ResidencyRule(
                name="no-cloud",
                source_regions=["gov-us-1"],
                blocked_providers=["openai_compat", "bedrock"],
            )
        ]
        engine = ResidencyEngine(rules=rules)
        assert not engine.is_compliant("gov-us-1", "us-east-1", "openai_compat")
        assert engine.is_compliant("gov-us-1", "us-east-1", "vllm")


# ───────────────────────── Budget & Tenant ─────────────────────────


class TestBudgetTracker:
    def test_record_spend(self):
        tracker = BudgetTracker()
        tracker.record_spend("t1", 10.0)
        tracker.record_spend("t1", 5.0)
        record = tracker.get_spend("t1")
        assert record.total_usd == 15.0
        assert record.requests == 2

    def test_remaining_budget(self):
        tracker = BudgetTracker()
        tracker.record_spend("t1", 800.0)
        budget = BudgetConfig(monthly_budget_usd=1000.0)
        assert tracker.get_remaining_budget("t1", budget) == 200.0

    def test_over_budget(self):
        tracker = BudgetTracker()
        tracker.record_spend("t1", 1001.0)
        budget = BudgetConfig(monthly_budget_usd=1000.0)
        assert tracker.is_over_budget("t1", budget)


class TestTenantManager:
    def test_filter_preferred(self):
        tenant = TenantConfig(name="team-a", preferred_models=["model-a", "model-b"])
        manager = TenantManager(tenants={"team-a": tenant})
        filtered = manager.filter_models_for_tenant("team-a", ["model-a", "model-b", "model-c"])
        assert filtered == ["model-a", "model-b"]

    def test_filter_blocked(self):
        tenant = TenantConfig(name="team-a", blocked_models=["model-c"])
        manager = TenantManager(tenants={"team-a": tenant})
        filtered = manager.filter_models_for_tenant("team-a", ["model-a", "model-b", "model-c"])
        assert "model-c" not in filtered

    def test_unknown_tenant(self):
        manager = TenantManager()
        filtered = manager.filter_models_for_tenant("unknown", ["a", "b"])
        assert filtered == ["a", "b"]

    def test_budget_check(self):
        tenant = TenantConfig(
            name="t1",
            budget=BudgetConfig(enabled=True, monthly_budget_usd=100.0),
        )
        manager = TenantManager(tenants={"t1": tenant})
        assert manager.check_budget("t1", 5.0)
        manager.record_request("t1", 100.0)
        assert not manager.check_budget("t1", 5.0)
