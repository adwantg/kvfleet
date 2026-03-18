"""Tests for KV-cache affinity, fingerprinting, and semantic cache."""


from kvfleet.adapters.base import ChatMessage
from kvfleet.cache.fingerprints import PromptFingerprint, PromptFingerprinter, _normalize_text
from kvfleet.cache.kv_affinity import ConsistentHashRing, KVAffinityScorer, SessionAffinityStore
from kvfleet.cache.semantic_cache import SemanticCache

# ───────────────────────── Fingerprinting ─────────────────────────


class TestPromptFingerprinter:
    def test_fingerprint_basic(self):
        fp = PromptFingerprinter()
        messages = [ChatMessage(role="user", content="Hello, world!")]
        result = fp.fingerprint(messages)
        assert result.full_hash
        assert result.prefix_hash
        assert result.token_estimate > 0

    def test_fingerprint_with_system(self):
        fp = PromptFingerprinter()
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is Python?"),
        ]
        result = fp.fingerprint(messages)
        assert result.system_hash != ""
        assert result.session_key

    def test_same_input_same_hash(self):
        fp = PromptFingerprinter()
        msgs = [ChatMessage(role="user", content="Test message")]
        h1 = fp.fingerprint(msgs)
        h2 = fp.fingerprint(msgs)
        assert h1.full_hash == h2.full_hash

    def test_different_input_different_hash(self):
        fp = PromptFingerprinter()
        h1 = fp.fingerprint([ChatMessage(role="user", content="Hello")])
        h2 = fp.fingerprint([ChatMessage(role="user", content="Goodbye")])
        assert h1.full_hash != h2.full_hash

    def test_similarity_identical(self):
        fp = PromptFingerprinter()
        msgs = [ChatMessage(role="user", content="Test")]
        h1 = fp.fingerprint(msgs)
        assert fp.similarity(h1, h1) == 1.0

    def test_similarity_same_system(self):
        fp = PromptFingerprinter()
        h1 = fp.fingerprint(
            [
                ChatMessage(role="system", content="Be helpful"),
                ChatMessage(role="user", content="Q1"),
            ]
        )
        h2 = fp.fingerprint(
            [
                ChatMessage(role="system", content="Be helpful"),
                ChatMessage(role="user", content="Q2"),
            ]
        )
        assert fp.similarity(h1, h2) >= 0.5

    def test_normalize_text(self):
        assert _normalize_text("  Hello   World  ") == "hello world"


# ───────────────────────── Consistent Hash Ring ─────────────────────────


class TestConsistentHashRing:
    def test_add_and_get(self):
        ring = ConsistentHashRing(virtual_nodes=10)
        ring.add_node("server-1")
        ring.add_node("server-2")
        result = ring.get_node("test-key")
        assert result in ("server-1", "server-2")

    def test_consistent_results(self):
        ring = ConsistentHashRing(virtual_nodes=10)
        ring.add_node("a")
        ring.add_node("b")
        r1 = ring.get_node("key")
        r2 = ring.get_node("key")
        assert r1 == r2

    def test_empty_ring(self):
        ring = ConsistentHashRing()
        assert ring.get_node("key") is None

    def test_remove_node(self):
        ring = ConsistentHashRing(virtual_nodes=5)
        ring.add_node("a")
        ring.add_node("b")
        ring.remove_node("a")
        result = ring.get_node("any-key")
        assert result == "b"


# ───────────────────────── Session Affinity ─────────────────────────


class TestSessionAffinityStore:
    def test_set_get(self):
        store = SessionAffinityStore(ttl_seconds=3600)
        store.set("session-1", "endpoint-a")
        assert store.get("session-1") == "endpoint-a"

    def test_missing_session(self):
        store = SessionAffinityStore()
        assert store.get("nonexistent") is None

    def test_clear_expired(self):
        store = SessionAffinityStore(ttl_seconds=0)  # Immediate expiry
        store.set("s1", "ep1")
        import time

        time.sleep(0.01)
        assert store.get("s1") is None


# ───────────────────────── KV Affinity Scorer ─────────────────────────


class TestKVAffinityScorer:
    def test_score_affinity(self):
        scorer = KVAffinityScorer()
        scorer.register_endpoints("model-a", ["http://a:8000", "http://a2:8000"])
        fp = PromptFingerprint(
            full_hash="abc", system_hash="sys", prefix_hash="pfx", conversation_hash="conv"
        )
        scores = scorer.score_affinity(fp, "model-a", ["http://a:8000", "http://a2:8000"])
        assert len(scores) == 2
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_record_and_affinity(self):
        scorer = KVAffinityScorer()
        fp = PromptFingerprint(
            full_hash="h1", system_hash="s1", prefix_hash="p1", conversation_hash="c1"
        )
        scorer.record_routing(fp, "http://a:8000")
        # Same fingerprint should now have session affinity
        scores = scorer.score_affinity(fp, "model", ["http://a:8000", "http://b:8000"])
        assert scores["http://a:8000"] >= scores["http://b:8000"]

    def test_best_endpoint(self):
        scorer = KVAffinityScorer()
        fp = PromptFingerprint(
            full_hash="h", system_hash="s", prefix_hash="p", conversation_hash="c"
        )
        scorer.record_routing(fp, "http://a:8000")
        best, score = scorer.best_endpoint(fp, "m", ["http://a:8000", "http://b:8000"])
        assert best == "http://a:8000"
        assert score >= 0.0

    def test_cache_stats(self):
        scorer = KVAffinityScorer()
        stats = scorer.get_cache_stats()
        assert stats["active_sessions"] == 0


# ───────────────────────── Semantic Cache ─────────────────────────


class TestSemanticCache:
    def test_put_get(self):
        cache = SemanticCache()
        fp = PromptFingerprint(
            full_hash="h1", system_hash="s", prefix_hash="p", conversation_hash="c"
        )
        cache.put(fp, "Hello response", "model-a")
        result = cache.get(fp)
        assert result is not None
        assert result.content == "Hello response"

    def test_miss(self):
        cache = SemanticCache()
        fp = PromptFingerprint(
            full_hash="miss", system_hash="s", prefix_hash="p", conversation_hash="c"
        )
        assert cache.get(fp) is None

    def test_invalidate(self):
        cache = SemanticCache()
        fp = PromptFingerprint(
            full_hash="h", system_hash="s", prefix_hash="p", conversation_hash="c"
        )
        cache.put(fp, "content", "model")
        cache.invalidate(fp)
        assert cache.get(fp) is None

    def test_clear(self):
        cache = SemanticCache()
        fp = PromptFingerprint(
            full_hash="h", system_hash="s", prefix_hash="p", conversation_hash="c"
        )
        cache.put(fp, "content", "model")
        cache.clear()
        assert cache.size == 0

    def test_stats(self):
        cache = SemanticCache(max_size=100, ttl_seconds=60)
        stats = cache.stats()
        assert stats["max_size"] == 100
        assert stats["total_entries"] == 0

    def test_eviction(self):
        cache = SemanticCache(max_size=3)
        for i in range(5):
            fp = PromptFingerprint(
                full_hash=f"h{i}",
                system_hash=f"s{i}",
                prefix_hash=f"p{i}",
                conversation_hash=f"c{i}",
            )
            cache.put(fp, f"content-{i}", "model")
        assert cache.size <= 6  # 3 entries * 2 (full_hash + prefix_key)
