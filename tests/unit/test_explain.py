"""Tests for route explanation."""

from kvfleet.router.explain import RouteExplanation, CandidateScore, PolicyDecision


class TestRouteExplanation:
    def test_summary(self):
        explanation = RouteExplanation(
            strategy="hybrid_score",
            selected_model="model-a",
            candidates=[
                CandidateScore(model_name="model-a", total_score=0.8, selected=True),
                CandidateScore(model_name="model-b", total_score=0.5, rejected_reason="Lower score"),
            ],
        )
        summary = explanation.summary()
        assert "model-a" in summary
        assert "hybrid_score" in summary

    def test_to_dict(self):
        explanation = RouteExplanation(
            strategy="static",
            selected_model="m1",
            candidates=[CandidateScore(model_name="m1", total_score=1.0, selected=True)],
            policy_decisions=[PolicyDecision(rule_name="pii", passed=True, reason="No PII")],
        )
        d = explanation.to_dict()
        assert d["strategy"] == "static"
        assert len(d["candidates"]) == 1
        assert len(d["policy_decisions"]) == 1

    def test_fallback_in_summary(self):
        explanation = RouteExplanation(
            strategy="cost_first",
            selected_model="m2",
            fallback_triggered=True,
            fallback_chain=["m1", "m2"],
        )
        summary = explanation.summary()
        assert "Fallback" in summary
        assert "m1" in summary
