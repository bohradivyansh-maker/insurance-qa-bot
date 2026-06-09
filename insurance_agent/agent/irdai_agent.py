# app/agent/irdai_agent.py

import os
from tavily import AsyncTavilyClient
from groq import AsyncGroq

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.retriever import retrieve_chunks

async def handle_irdai_query(
    query: str, 
    groq_client: AsyncGroq, 
    vectorstore, 
    session_id: str
) -> str:
    """
    Fetches live IRDAI data via Tavily, supplements with local pgVector policies,
    and synthesizes a banking-grade analytical response.
    """
    tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # 1. Fetch live web data (Non-blocking)
    try:
        search_result = await tavily_client.search(
            query=f"IRDAI official data India {query}", 
            search_depth="advanced", 
            max_results=4
        )
        web_context = "\n\n".join([
            f"Source: {res.get('url', 'Web')}\n{res.get('content', '')}" 
            for res in search_result.get('results', [])
        ])
    except Exception as e:
        print(f"[IRDAI Agent] Tavily search failed: {e}")
        web_context = "Live web data currently unavailable."

    # 2. Fetch local policy data (Non-blocking fallback)
    try:
        # Take top 3 chunks to avoid context explosion
        retrieved = retrieve_chunks(query, vectorstore)[:3]
        local_context = "\n---\n".join([doc.page_content for doc, _ in retrieved]) if retrieved else "No specific references found in local LIC policies."
    except Exception as e:
        print(f"[IRDAI Agent] Vector retrieval failed: {e}")
        local_context = "Local policy data unavailable."

    # 3. Synthesize Answer
    system_prompt = """You are a senior insurance analyst and actuary. 
You are answering a query regarding IRDAI regulations, claim settlement ratios, or market data.
Base your answer on the Live Web Context provided. If the Local Policy Context is relevant, integrate it smoothly.
Maintain a highly professional, banking-grade tone. Use bullet points for readability."""

    user_prompt = f"""
Query: {query}

[LIVE WEB CONTEXT (Tavily)]
{web_context}

[LOCAL POLICY CONTEXT (pgVector)]
{local_context}
"""

    response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=800,
    )
    
    return response.choices[0].message.content