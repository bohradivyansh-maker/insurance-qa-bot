# insurance_agent/agent/plan_analyst.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.retriever import retrieve_chunks, get_all_chunks
from app.reranker import CrossEncoder
from insurance_agent.agent.prompts import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_USER_TEMPLATE

from groq import Groq
from insurance_agent.agent.langfuse_tracker import get_langfuse
langfuse = get_langfuse()
AGENT_TOP_N          = 10
SIMILARITY_THRESHOLD = 0.25

def _get_plan_display_name(filename: str) -> str:
    return {
        "LIC_Tech-Term.pdf":              "LIC Tech Term",
        "LIC_Jeevan_Anand.pdf":           "LIC Jeevan Anand",
        "LIC_Endowment.pdf":              "LIC New Endowment Plan",
        "LIC_Jeevan_Labh.pdf":            "LIC Jeevan Labh",
        "LIC_Jeevan_Umang.pdf":           "LIC Jeevan Umang",
        "LIC_New_Jeevan_Amar.pdf":        "LIC New Jeevan Amar",
        "LIC_ChidrensMoney_BackPlan.pdf": "LIC Children's Money Back Plan",
    }.get(filename, filename)


def _source_locked_retrieve(query: str, plan_filename: str, vectorstore) -> list:
    """
    Retrieves chunks locked to a single plan using two strategies:

    Strategy 1 — pgVector similarity search with metadata filter.
                  Uses $eq on source field with full path matching via Python-side filter.
    Strategy 2 — Fallback: load all chunks from BM25 cache and filter by filename in Python.
                  Guarantees we always get chunks even if pgVector filter fails.

    Returns list of (doc, score) tuples, source-locked to plan_filename only.
    """
    print(f"[Plan Analyst] Source-locked retrieval for: {plan_filename}")

    # Strategy 1 — pgVector similarity with source filter
    # Try both $like and direct similarity, filter in Python afterward
    try:
        raw_results = vectorstore.similarity_search_with_score(query, k=150)

        # Filter in Python by checking if plan_filename appears in source path
        filtered = [
            (doc, score)
            for doc, score in raw_results
            if plan_filename in doc.metadata.get("source", "")
            and score >= SIMILARITY_THRESHOLD
        ]

        print(f"[Plan Analyst] Strategy 1: {len(filtered)} chunks from pgVector (filtered in Python)")

        if filtered:
            return filtered

    except Exception as e:
        print(f"[Plan Analyst] Strategy 1 failed: {e}")

    # Strategy 2 — BM25 cache fallback, filter by filename in Python
    try:
        all_chunks = get_all_chunks(vectorstore)
        filtered = [
            (doc, 0.30)
            for doc in all_chunks
            if plan_filename in doc.metadata.get("source", "")
        ]
        print(f"[Plan Analyst] Strategy 2 fallback: {len(filtered)} chunks from BM25 cache")
        return filtered if filtered else None

    except Exception as e:
        print(f"[Plan Analyst] Strategy 2 failed: {e}")
        return None


def _rerank_agent(query: str, retrieved_chunks: list, reranker: CrossEncoder) -> list:
    """
    Reranks source-locked chunks using cross-encoder.
    Keeps top AGENT_TOP_N=10 (vs base RAG's 5).
    """
    if not retrieved_chunks:
        return None

    docs  = [doc for doc, score in retrieved_chunks]
    pairs = [(query, str(doc.page_content)) for doc in docs if doc.page_content is not None]
    docs  = [doc for doc in docs if doc.page_content is not None]
    scores = reranker.predict(pairs)

    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top    = scored[:AGENT_TOP_N]

    print(f"[Agent Reranker] Kept top {len(top)} from {len(docs)} chunks")
    for i, (doc, score) in enumerate(top):
        print(f"  Rank {i+1} | Score: {score:.4f} | Page: {doc.metadata.get('page','?')}")

    return top if top else None


def _format_chunks_for_prompt(reranked_chunks: list) -> str:
    if not reranked_chunks:
        return "No relevant chunks retrieved."
    parts = []
    for i, (doc, score) in enumerate(reranked_chunks):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[Chunk {i+1} | Source: {source} | Page: {page} | Score: {score:.3f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def analyse_plan(
    plan_filename: str,
    profile: dict,
    vectorstore,
    reranker,
    groq_client: Groq
) -> str:
    """
    Full analysis pipeline for one plan:
    1. Source-locked retrieval  — only chunks from this plan's PDF
    2. Cross-encoder reranking  — top 10 most relevant chunks
    3. LLM analysis             — extract facts strictly from chunks
    """
    plan_display = _get_plan_display_name(plan_filename)

    goal_query_map = {
        "pension":        "survival benefit regular income pension annuity payout",
        "lump_sum":       "maturity benefit sum assured bonus lump sum",
        "protection":     "death benefit sum assured term cover life protection",
        "savings":        "maturity benefit savings bonus endowment",
        "child_planning": "child education money back survival benefit payout ages",
    }
    goal_keywords = goal_query_map.get(profile.get("goal", ""), "benefits maturity death")

    queries = [
        f"{plan_display} death benefit sum assured percentage formula",
        f"{plan_display} maturity benefit survival benefit payout percentage",
        f"{plan_display} {goal_keywords} eligibility age premium",
        f"{plan_display} policy term premium paying term conditions",
    ]

    # Step 1 — Source-locked retrieval with multiple queries
    retrieved_all = []
    seen_contents = set()
    for q in queries:
        results = _source_locked_retrieve(q, plan_filename, vectorstore)
        if results:
            for doc, score in results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    retrieved_all.append((doc, score))

    retrieved = retrieved_all if retrieved_all else None
    retrieval_query = f"{plan_display} {goal_keywords} death benefit maturity benefit"

    if not retrieved:
        return (
            f"### {plan_display}\n\n"
            f"Could not retrieve any chunks from {plan_filename}. "
            f"Please ensure the document was ingested correctly."
        )

    # Step 2 — Rerank with AGENT_TOP_N=10
    reranked = _rerank_agent(retrieval_query, retrieved, reranker)

    if not reranked:
        return (
            f"### {plan_display}\n\n"
            f"Retrieved chunks did not pass relevance threshold after reranking."
        )

    # Step 3 — LLM analysis strictly from chunks
    chunks_text  = _format_chunks_for_prompt(reranked)
    user_message = ANALYSIS_USER_TEMPLATE.format(
        age=profile["age"],
        annual_income=profile["annual_income"],
        dependents=profile["dependents"],
        goal=profile["goal"],
        plan_name=plan_display,
        chunks=chunks_text
    )
    # replace your existing groq call with this
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="agent_analysis",
        model="llama-3.3-70b-versatile",
        input=str({"plan": plan_filename, "profile": profile})
    ) as obs:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        analysis = response.choices[0].message.content
        obs.update(output=analysis)
    return analysis