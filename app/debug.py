# app/debug.py
# Run this to see exactly what chunks plan_analyst retrieves,
# what analysis it generates, and how the recommender builds the comparison table.
# Usage: python debug.py

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from retriever import get_retriever
from reranker import load_reranker
from groq import Groq
from insurance_agent.agent.plan_analyst import (
    analyse_plan,
    _source_locked_retrieve,
    _rerank_agent,
    _format_chunks_for_prompt,
    _get_plan_display_name
)
from insurance_agent.agent.recommender import generate_recommendation

# ── Config ────────────────────────────────────────────────────────────────
PLAN_TO_DEBUG  = "LIC_Tech-Term.pdf"   # change this to any plan
# To test recommender with two plans, add a second plan here
SECOND_PLAN    = "LIC_New_Jeevan_Amar.pdf"   # set to None to skip recommender test
TEST_PROFILE   = {
    "age": 41,
    "annual_income": 1800000,
    "dependents": 4,
    "monthly_spending": 90000,
    "goal": "protection"
}
RUNS = 3  # how many times to run to check consistency

def main():
    print("Loading components...")
    vectorstore = get_retriever()
    reranker    = load_reranker()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    plan_name   = _get_plan_display_name(PLAN_TO_DEBUG)

    print(f"\n{'='*60}")
    print(f"Debugging plan: {plan_name}")
    print(f"Profile: {TEST_PROFILE}")
    print(f"Running {RUNS} times to check consistency")
    print(f"{'='*60}")

    # Store analyses across runs to compare recommender table consistency
    all_analyses = []

    for run in range(1, RUNS + 1):
        print(f"\n{'─'*40}")
        print(f"RUN {run}")
        print(f"{'─'*40}")

        # ── Step 1 — show retrieved chunks ────────────────────────────────
        goal_query_map = {
            "pension":        "survival benefit regular income pension annuity payout",
            "lump_sum":       "maturity benefit sum assured bonus lump sum",
            "protection":     "death benefit sum assured term cover life protection",
            "savings":        "maturity benefit savings bonus endowment",
            "child_planning": "child education money back survival benefit payout ages",
        }
        goal_keywords = goal_query_map.get(TEST_PROFILE.get("goal", ""), "benefits")
        retrieval_query = (
            f"{plan_name} {goal_keywords} "
            f"premium payment term policy term eligibility age {TEST_PROFILE.get('age', '')} "
            f"death benefit maturity benefit survival benefit conditions exclusions"
        )

        print(f"\nRetrieval query:\n{retrieval_query}")

        retrieved = _source_locked_retrieve(retrieval_query, PLAN_TO_DEBUG, vectorstore)
        if not retrieved:
            print("ERROR: No chunks retrieved")
            continue

        reranked = _rerank_agent(retrieval_query, retrieved, reranker)
        if not reranked:
            print("ERROR: No chunks after reranking")
            continue

        print(f"\nTop chunks retrieved ({len(reranked)}):")
        for i, (doc, score) in enumerate(reranked):
            page = doc.metadata.get('page', '?')
            preview = doc.page_content[:200].replace('\n', ' ')
            print(f"  [{i+1}] Page {page} | Score: {score:.4f}")
            print(f"       {preview}...")

        # ── Step 2 — show analysis LLM generates from these chunks ────────
        print(f"\nGenerating analysis...")
        analysis = analyse_plan(
            plan_filename=PLAN_TO_DEBUG,
            profile=TEST_PROFILE,
            vectorstore=vectorstore,
            reranker=reranker,
            groq_client=groq_client
        )

        print(f"\nAnalysis output:")
        print(analysis)

        not_mentioned_count = analysis.lower().count("not available in retrieved chunks")
        print(f"\nNot-mentioned phrases found: {not_mentioned_count}")

        all_analyses.append({PLAN_TO_DEBUG: analysis})

    # ── Step 3 — test recommender table consistency across runs ───────────
    # Uses the analyses collected above to check if the comparison table
    # is consistent even when analyst output has minor variance.
    if SECOND_PLAN:
        print(f"\n{'='*60}")
        print(f"RECOMMENDER TABLE CONSISTENCY TEST")
        print(f"Running recommender {RUNS} times with collected analyses")
        print(f"{'='*60}")

        # Get a second plan analysis once (to pair with primary plan)
        print(f"\nGenerating analysis for second plan: {SECOND_PLAN}...")
        second_analysis = analyse_plan(
            plan_filename=SECOND_PLAN,
            profile=TEST_PROFILE,
            vectorstore=vectorstore,
            reranker=reranker,
            groq_client=groq_client
        )

        for run in range(1, RUNS + 1):
            print(f"\n{'─'*40}")
            print(f"RECOMMENDER RUN {run}")
            print(f"{'─'*40}")

            # Use the analysis from the matching analyst run
            primary_analysis = all_analyses[run - 1][PLAN_TO_DEBUG]

            plan_analyses = {
                PLAN_TO_DEBUG: primary_analysis,
                SECOND_PLAN: second_analysis
            }

            rec = generate_recommendation(
                profile=TEST_PROFILE,
                plan_analyses=plan_analyses,
                shortlist_reasoning="Plans shortlisted for pure protection goal.",
                groq_client=groq_client
            )

            # Print only the comparison table section
            table_start = rec.find("Head-to-Head")
            recommendation_start = rec.find("**Recommendation")
            if recommendation_start == -1:
                recommendation_start = rec.find("## Recommendation")

            if table_start != -1 and recommendation_start != -1:
                print("\nComparison table section:")
                print(rec[table_start:recommendation_start])
            elif table_start != -1:
                print("\nComparison table section (recommendation marker not found):")
                print(rec[table_start:table_start + 1200])
            else:
                print("\nWARNING: 'Head-to-Head' section not found in recommender output.")
                print("Full recommender output:")
                print(rec)

            # Count not-mentioned in table section only
            table_text = rec[table_start:recommendation_start] if table_start != -1 else rec
            not_mentioned_in_table = (
                table_text.lower().count("not mentioned") +
                table_text.lower().count("not available") +
                table_text.lower().count("not specified")
            )
            print(f"\nNot-mentioned/available phrases in table: {not_mentioned_in_table}")

    print(f"\n{'='*60}")
    print("Debug complete.")
    print("Analyst: If chunks same but analysis differs → LLM extraction variance")
    print("Recommender: If table rows differ across runs → prompt or temperature issue")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()