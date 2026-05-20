import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
from langchain_postgres import PGVector
from langchain_community.retrievers import BM25Retriever
from embedder import load_embedder

CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql+psycopg://postgres:Bohra%40123@localhost:5432/insurance_qa"
)

COLLECTION_NAME = "insurance_policies"
TOP_K = 10               # single policy queries — high enough to capture broad benefit sections
TOP_K_COMPARISON = 8    # per-policy chunk count for comparison queries
TOP_K_BM25 = 8          # keyword search candidate count
SIMILARITY_THRESHOLD = 0.25

all_chunks_cache = None

POLICY_KEYWORDS = {
    "jeevan labh": "LIC_Jeevan_Labh.pdf",
    "jeevan anand": "LIC_Jeevan_Anand.pdf",
    "jeevan umang": "LIC_Jeevan_Umang.pdf",
    "tech term": "LIC_Tech-Term.pdf",
    "endowment": "LIC_Endowment.pdf",
    "money back": "LIC_ChidrensMoney_BackPlan.pdf",
    "jeevan amar": "LIC_New_Jeevan_Amar.pdf",
}

def detect_policy(query: str):
    query_lower = query.lower()
    detected = []
    for keyword, filename in POLICY_KEYWORDS.items():
        if keyword in query_lower:
            detected.append(filename)
    return detected if detected else None

def get_all_chunks(vectorstore):
    global all_chunks_cache
    if all_chunks_cache is None:
        results = vectorstore.similarity_search("insurance policy", k=1000)
        all_chunks_cache = results
        print(f"Loaded {len(all_chunks_cache)} chunks for BM25")
    return all_chunks_cache

def get_retriever():
    embedder = load_embedder()
    vectorstore = PGVector(
        embeddings=embedder,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True
    )
    return vectorstore

def retrieve_chunks(query: str, vectorstore: PGVector):
    detected_policy = detect_policy(query)
    is_comparison = detected_policy and len(detected_policy) > 1

    if detected_policy:
        if not is_comparison:
            # Single policy — filter to that policy and retrieve TOP_K chunks
            vector_results = vectorstore.similarity_search_with_score(
                query,
                k=TOP_K,
                filter={"source": {"$like": f"%{detected_policy[0]}%"}}
            )
        else:
            # Comparison query — pull TOP_K_COMPARISON from EACH detected policy
            # separately so both policies are equally represented before reranking
            vector_results = []
            for policy_file in detected_policy:
                policy_results = vectorstore.similarity_search_with_score(
                    query,
                    k=TOP_K_COMPARISON,
                    filter={"source": {"$like": f"%{policy_file}%"}}
                )
                vector_results.extend(policy_results)

        # Fallback to unfiltered if nothing returned
        if not vector_results:
            vector_results = vectorstore.similarity_search_with_score(query, k=TOP_K)
    else:
        vector_results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    filtered = [
        (doc, score) for doc, score in vector_results
        if score >= SIMILARITY_THRESHOLD
    ]

    # BM25 keyword search — higher k improves recall for keyword-heavy queries
    # like "proposer dies", "suicide revival", "foreclosure" where semantic
    # similarity alone may not surface the right chunk
    all_chunks = get_all_chunks(vectorstore)
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = TOP_K_BM25
    bm25_results = bm25_retriever.invoke(query)

    # Merge — vector results first (higher confidence), then BM25 additions
    seen_contents = set()
    merged = []

    for doc, score in filtered:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            merged.append((doc, score))

    for doc in bm25_results:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            merged.append((doc, 0.30))

    # Force include — single policy queries only, capped at 20 chunks.
    # 20 is generous enough to guarantee cover page and benefits section
    # chunks always reach the reranker pool regardless of vector/BM25 scores,
    # without flooding the reranker the way uncapped force-include did.
    # Comparison queries excluded — per-policy vector search handles them.
    if detected_policy and not is_comparison:
        policy_file = detected_policy[0]
        added = 0
        for chunk in all_chunks:
            if added >= 20:
                break
            source = chunk.metadata.get("source", "")
            if policy_file in source and chunk.page_content not in seen_contents:
                seen_contents.add(chunk.page_content)
                merged.append((chunk, 0.35))
                added += 1

    if not merged:
        return None

    print(f"Retrieved {len(merged)} chunks after hybrid merge")
    for i, (doc, score) in enumerate(merged[:8]):
        source = doc.metadata.get("source", "unknown")
        print(f"Chunk {i+1} | Score: {score:.4f} | Source: {source}")

    return merged