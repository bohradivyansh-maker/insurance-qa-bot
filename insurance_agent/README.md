# LIC Insurance Advisor Agent

An agentic upgrade to the LIC Insurance Policy RAG system.
Built as a capstone extension demonstrating multi-step reasoning, plan shortlisting, and personalized recommendation.

## What this does differently from the base RAG

| Base RAG | This Agent |
|---|---|
| Answers "what does this policy say" | Answers "which policy is best for ME" |
| Single retrieval call | 3-step pipeline: profile → shortlist → analyse → recommend |
| No memory | Conversational memory within session |
| No personalization | Tailored to user's age, income, dependents, goals |

## Architecture

```
User input
    ↓
[Profile Collector] — conversational chain, collects age/income/dependents/spending/goal
    ↓ profile dict
[Plan Shortlister] — LLM reasons which 2-3 of 7 plans are worth analysing
    ↓ shortlisted plan filenames
[Plan Analyst] — for each plan: RAG retrieval → reranking → LLM analysis (reuses existing retriever + reranker)
    ↓ per-plan analysis texts
[Recommender] — final LLM call: personalized recommendation with justification
    ↓
Streamed output to Chainlit UI
```

## Groq API calls per session
- Profile collection: 1 call per user message until profile complete (typically 3-4 exchanges)
- Shortlisting: 1 call
- Plan analysis: 1 call per shortlisted plan (2-3 calls)
- Recommendation: 1 call
- **Total: ~8-10 Groq calls per full session**

## How to run

```bash
# From INSUARANCE_QA root directory
cd insurance_agent
chainlit run main.py
```

Make sure:
1. Your PostgreSQL + pgVector container is running (same as base RAG)
2. `.env` in root INSUARANCE_QA directory has GROQ_API_KEY set
3. Base RAG ingestion has already been run (embeddings exist in pgVector)

## Folder structure

```
insurance_agent/
  agent/
    prompts.py          — all LLM prompts
    memory.py           — ConversationBufferMemory setup
    profile_collector.py — conversational profile extraction
    plan_shortlister.py  — LLM-based plan elimination
    plan_analyst.py      — RAG analysis per plan (hooks into ../app/retriever.py)
    recommender.py       — final recommendation generation + streaming
  main.py               — Chainlit UI + orchestration
  requirements.txt
  README.md
```

## Known limitations (future work)
- Session memory only — resets on new chat
- Max 3 plans analysed simultaneously (k=3 after reranking)
- PDF table extraction limitation inherited from base RAG
- Query rewriting not yet implemented for plan analysis queries
