import os
import json
import hashlib
import numpy as np
import redis

SIMILARITY_THRESHOLD = 0.82

EMBEDDING_PREFIX = "qa_emb:"
ANSWER_PREFIX    = "qa_ans:"
INDEX_KEY        = "qa_index"

_redis_client = None
_embedder     = None

POLICY_KEYWORDS = [
    "jeevan anand",
    "jeevan labh",
    "jeevan umang",
    "jeevan amar",
    "tech term",
    "endowment",
    "money back",
    "children",
]

def _detect_policy_scope(question: str) -> str:
    q = question.lower()
    for keyword in POLICY_KEYWORDS:
        if keyword in q:
            return keyword.replace(" ", "_")
    return "general"

def init_semantic_cache(embedder):
    global _redis_client, _embedder
    redis_url     = os.getenv("REDIS_URL", "redis://localhost:6379")
    _redis_client = redis.from_url(redis_url)
    _embedder     = embedder
    _redis_client.ping()
    count = len(_redis_client.lrange(INDEX_KEY, 0, -1))
    print(f"✅ Redis question-level cache active at {redis_url} ({count} entries)")
    return _redis_client


def _cosine_similarity(a: list, b: list) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_cached_answer(question: str) -> str | None:
    if _redis_client is None or _embedder is None:
        return None

    try:
        scope       = _detect_policy_scope(question)
        index_key   = f"{INDEX_KEY}:{scope}"
        q_embedding = _embedder.embed_query(question)
        hashes      = _redis_client.lrange(index_key, 0, -1)

        best_score = -1.0
        best_hash  = None

        for h in hashes:
            h_str    = h.decode("utf-8")
            emb_json = _redis_client.get(f"{EMBEDDING_PREFIX}{scope}:{h_str}")
            if emb_json is None:
                continue
            stored_emb = json.loads(emb_json)
            score      = _cosine_similarity(q_embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_hash  = h_str

        if best_score >= SIMILARITY_THRESHOLD and best_hash:
            answer = _redis_client.get(f"{ANSWER_PREFIX}{scope}:{best_hash}")
            if answer:
                print(f"✅ Cache HIT  scope={scope} score={best_score:.4f}")
                return answer.decode("utf-8")

        print(f"❌ Cache MISS scope={scope} best={best_score:.4f}")
        return None

    except Exception as e:
        print(f"⚠️ Cache lookup error: {e}")
        return None


def store_cached_answer(question: str, answer: str) -> None:
    if _redis_client is None or _embedder is None:
        return

    try:
        scope     = _detect_policy_scope(question)
        index_key = f"{INDEX_KEY}:{scope}"

        q_embedding = _embedder.embed_query(question)
        q_hash      = hashlib.md5(question.encode()).hexdigest()

        _redis_client.set(f"{EMBEDDING_PREFIX}{scope}:{q_hash}", json.dumps(q_embedding))
        _redis_client.set(f"{ANSWER_PREFIX}{scope}:{q_hash}", answer)
        _redis_client.rpush(index_key, q_hash)

        print(f"✅ Cache STORE scope={scope} — '{question[:60]}...'")

    except Exception as e:
        print(f"⚠️ Cache store error: {e}")