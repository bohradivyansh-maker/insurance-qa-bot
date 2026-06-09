# app/main.py — Day 3: Mentor-Proofing, UX Safety & True Contextual Memory

import os
import sys
import asyncio
import random
import chainlit as cl
from dotenv import load_dotenv
from typing import TypedDict, Optional, List

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))

# ── Core imports ──────────────────────────────────────────────────────────────
from retriever import get_retriever, retrieve_chunks
from reranker import load_reranker, rerank_chunks
from chain import load_llm, run_chain, format_context, expand_query
from langfuse_client import get_langfuse_handler, get_langfuse_client
from trace_helpers import _trace_guardrail, _trace_intent
from embedder import load_embedder
from cache import init_semantic_cache, get_cached_answer, store_cached_answer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from groq import AsyncGroq

from insurance_agent.agent.profile_collector import ProfileCollector, detect_specific_plans
from insurance_agent.agent.plan_shortlister import shortlist_plans
from insurance_agent.agent.plan_analyst import analyse_plan
from insurance_agent.agent.recommender import generate_recommendation
from insurance_agent.agent.irdai_agent import handle_irdai_query
from insurance_agent.agent.calculator_tool import run_calculator_tool

# ── LangGraph imports ─────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

GROQ_MODEL = "llama-3.3-70b-versatile"

class InsuranceState(TypedDict):
    query:            str
    query_lower:      str
    is_insurance:     Optional[bool]
    intent:           Optional[str]
    route:            Optional[str]
    response:         Optional[str]
    blocked_reason:   Optional[str]
    conversation_history: List[dict]

PLAN_NAME_MAP = {
    "LIC_Tech-Term.pdf":              "LIC Tech Term",
    "LIC_Jeevan_Anand.pdf":           "LIC Jeevan Anand",
    "LIC_Endowment.pdf":              "LIC New Endowment Plan",
    "LIC_Jeevan_Labh.pdf":            "LIC Jeevan Labh",
    "LIC_Jeevan_Umang.pdf":           "LIC Jeevan Umang",
    "LIC_New_Jeevan_Amar.pdf":        "LIC New Jeevan Amar",
    "LIC_ChidrensMoney_BackPlan.pdf": "LIC Children's Money Back Plan",
}

DISPLAY_PLANS = [
    "Jeevan Anand", "Jeevan Labh", "Tech Term", 
    "Jeevan Umang", "Endowment", "Money Back", "Jeevan Amar"
]

AGENT_TRIGGERS = [
    "best plan for me", "recommend a plan", "suggest a plan",
    "suggest the most suitable", "suggest the best",
    "which plan should i buy", "which plan should i take",
    "which plan should i choose", "help me choose a plan",
    "advise me on a plan", "what plan should i get",
    "suitable plan for me", "most suitable plan", "most suitable lic",
    "any policy i can take", "i am a", "year old", "years old",
    "looking for a plan", "want to buy a policy", "which are for", "which provide",""
    "compare","comparision between","comparison between","comparison"
]

PLAN_TYPE_MAP = {
    "jeevan anand": "LIC's New Jeevan Anand Plan is a Par, Non-Linked, Life, Individual, Savings Plan.",
    "jeevan labh":  "LIC's Jeevan Labh is a Par, Non-Linked, Life, Individual, Savings Plan.",
    "jeevan umang": "LIC's Jeevan Umang is a Par, Non-Linked, Individual, Savings, Whole Life Insurance Plan.",
    "endowment":    "LIC's New Endowment Plan is a Par, Non-Linked, Life, Individual, Savings Plan.",
    "money back":   "LIC's New Children's Money Back Plan is a Par, Non-Linked, Life, Individual, Savings Plan.",
    "jeevan amar":  "LIC's New Jeevan Amar is a Non-Par, Non-Linked, Life, Individual, Pure Risk Plan.",
    "tech term":    "LIC's New Tech-Term is a Non-Par, Non-Linked, Life, Individual, Pure Risk Plan.",
}

GUARDRAIL_RESPONSES = [
    "🤔 I specialize exclusively in LIC Life Insurance policies. While I can't advise on mutual funds, health insurance, or general topics, I can help you find a great LIC savings or term plan!",
    "😄 I appreciate the curiosity! But I am strictly an LIC insurance bot. Try asking me about term plans or claim ratios.",
    "📋 My knowledge is 100% LIC policies and 0% everything else. Whoever built me was very focused.",
    "🏦 Great question for a general AI, but I only speak fluent LIC insurance.",
    "😅 That falls outside my expertise. I only handle LIC claims, premiums, and life cover recommendations."
]

GREETING_RESPONSES = {
    "hello":    "👋 Hello! I am your Insurance Policy Assistant. Ask me anything about LIC policies or get a personalized plan recommendation!",
    "hi":       "👋 Hi there! How can I help you with your insurance queries today?",
    "hey":      "👋 Hey! Ask me anything about LIC policies or say 'suggest a plan for me' to get a personalized recommendation!",
    "thanks":   "😊 No problem! Glad I could help. Feel free to ask anything else.",
    "thank you":"😊 Happy to help! Let me know if you have more questions.",
    "bye":      "👋 Goodbye! Stay insured and stay protected!",
    "goodbye":  "👋 Take care! Come back anytime you have insurance questions.",
    "how are you": "😊 I am doing great and ready to help! Ask about policies or say 'recommend a plan' to get started!",
}

# ── EXPANDED: Catches Meta-Questions & Offerings ──
DOCUMENT_QUERIES = [
    "what documents", "which documents", "what pdfs",
    "what are you indexed on", "what do you know about",
    "what policies do you have", "which policies",
    "list plans", "what options do i have", "list all policies", 
    "what documents are you aware of", "list all the policies",
    "what are your offerings", "what can you do", "how can you help me",
    "what are your capabilities", "offerings", 
]

IRDAI_TRIGGERS = [
    "irdai", "claim settlement ratio", "settlement ratio",
    "insurance regulator", "market share", "irdai report",
    "grievance ratio", "solvency ratio", "policyholder protection",
    "insurance penetration", "insurance density", "annual report irdai",
    "best settlement", "which company settles", "claim ratio"
]
CALCULATOR_TRIGGERS = [
        "premium", "how much", "cost of", "price of",
        "monthly payment", "yearly payment", "annual payment",
        "what will i pay", "what would i pay", "how much do i pay",
        "calculate", "calculator", "premium amount", "premium for",
        "premium of", "afford", "instalment amount",
    ]

MAX_HISTORY = 3

def add_to_memory(query: str, response: str):
    history = cl.user_session.get("conversation_history", [])
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response[:500]})
    if len(history) > MAX_HISTORY * 2:
        history = history[-(MAX_HISTORY * 2):]
    cl.user_session.set("conversation_history", history)

def format_memory() -> str:
    history = cl.user_session.get("conversation_history", [])
    if not history: return ""
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

def get_memory_context_prefix() -> str:
    parts = []

    # Inject collected agent profile if available
    agent_profile = cl.user_session.get("agent_profile")
    if agent_profile:
        goal_display = {
            "pension":        "Regular pension / retirement income",
            "lump_sum":       "Lump sum savings at maturity",
            "protection":     "Pure life protection",
            "savings":        "Long-term savings with life cover",
            "child_planning": "Child education planning",
        }.get(agent_profile.get("goal", ""), agent_profile.get("goal", ""))

        parts.append(
            f"[USER PROFILE — collected earlier in this conversation. "
            f"Use this to personalize answers. Do not ask for details already known.]\n"
            f"Age: {agent_profile.get('age')} | "
            f"Annual Income: ₹{agent_profile.get('annual_income', 0):,} | "
            f"Dependents: {agent_profile.get('dependents')} | "
            f"Monthly Spending: ₹{agent_profile.get('monthly_spending', 0):,} | "
            f"Goal: {goal_display}\n"
        )

    # Inject conversation memory
    memory = format_memory()
    if memory:
        parts.append(
            f"[Previous conversation — use for context, do not repeat verbatim]\n{memory}"
        )

    if not parts:
        return ""

    return "\n\n".join(parts) + "\n\n[Current question below]\n"

# ── Contextual Query Rewriter ────────────────────────────────────────────────
async def _rewrite_query_with_context(query: str) -> str:
    """
    Uses the short-term conversation memory to rewrite vague follow-ups.
    """
    query_lower = query.lower()
    
    # SAFETY BYPASS: Do not rewrite if the user is asking for a recommendation or giving age
    if any(t in query_lower for t in AGENT_TRIGGERS):
        return query
        
    history = cl.user_session.get("conversation_history", [])
    if not history:
        return query

    recent_history = history[-4:]
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])
    
    groq_client = cl.user_session.get("groq_client")
    
    prompt = f"""
    Given the following conversation history, rewrite the user's latest query to be a fully standalone question.
    
    RULES:
    1. If the query references something from history (e.g., "what about for X?", "same for Y?", 
       "and for Jeevan Labh?"), rewrite it as a full standalone question using that context.
       Example: history has "death benefit in Jeevan Anand", user says "What about for Jeevan Labh?" 
       → Rewrite as "What is the death benefit in LIC Jeevan Labh?"
    2. If the query is already fully standalone (mentions plan + topic), return it exactly.
    3. If it is a greeting or unrelated new topic, return it exactly.
    DO NOT answer the question. ONLY return the rewritten standalone query.
    
    Conversation History:
    {history_str}
    
    Latest User Query: "{query}"
    
    Rewritten Query:"""
    
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        ))
        rewritten = response.choices[0].message.content.strip().strip('"')
        return rewritten
    except Exception:
        return query

# ── TIGHTENED GUARDRAIL: LIC Strict ──
GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for an LIC Life Insurance Assistant.
Reply YES if the query is about:
- LIC life insurance policies, premiums, claims, benefits, or plan suggestions
- IRDAI regulations, claim settlement ratios, insurance industry data, insurer rankings, market share, solvency ratios
- Indian insurance market statistics, top insurers, regulatory reports
- Standard greetings
Reply NO if it is about mutual funds, stock markets, real estate, health insurance (non-LIC), or topics entirely unrelated to insurance.
Reply EXACTLY ONE WORD: YES or NO."""),
    ("human", "{query}")
])

# ── EXPANDED INTENT PROMPT: Added Clarify ──
INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system","""You are a routing classifier for an insurance assistant.
Classify the user query into exactly one of these categories:
- RECOMMEND  — user wants a personalized plan suggestion, mentions their personal details, OR asks to filter plans by goal.
- IRDAI      — query is about IRDAI reports, claim settlement ratios, market data, regulatory info.
- CALCULATE  — user asks about premium amount, monthly/yearly cost, how much they would pay, or affordability of any plan.
- FACTUAL    — specific factual question about a policy (benefits, eligibility, terms, riders).
- ANALYTICAL — general comparison between named policies WITHOUT asking for personalized advice.
- CLARIFY    — very short, vague queries with no context.

Reply with exactly one word: RECOMMEND, IRDAI, CALCULATE, FACTUAL, ANALYTICAL, or CLARIFY."""),
    ("human", "{query}")
])

async def guardrail_node(state: InsuranceState) -> InsuranceState:
    llm        = cl.user_session.get("llm")
    lf_client  = cl.user_session.get("lf_client")
    session_id = cl.user_session.get("session_id")
    query      = state["query"]

    chain = GUARDRAIL_PROMPT | llm
    try:
        result = await chain.ainvoke({"query": query})
        answer = result.content.strip().upper()
    except Exception:
        answer = "YES"

    is_insurance = (answer == "YES")

    try:
        _trace_guardrail(lf_client, query, answer, session_id)
    except Exception:
        pass

    updated = dict(state)
    updated["is_insurance"] = is_insurance
    if not is_insurance:
        updated["route"] = "blocked"
    return updated

async def intent_node(state: InsuranceState) -> InsuranceState:
    llm        = cl.user_session.get("llm")
    lf_client  = cl.user_session.get("lf_client")
    session_id = cl.user_session.get("session_id")
    query      = state["query"]

    query_lower = query.lower()

    if any(t in query_lower for t in IRDAI_TRIGGERS):
        intent = "IRDAI"
    elif any(t in query_lower for t in CALCULATOR_TRIGGERS):
        intent = "CALCULATE"
    elif any(t in query_lower for t in AGENT_TRIGGERS):
        intent = "RECOMMEND"
    elif len(query.split()) < 3 and query_lower in ["tell me more", "details", "benefits", "why", "how", "what"]:
        intent = "CLARIFY"
    else:
        chain = INTENT_PROMPT | llm
        try:
            result = await chain.ainvoke({"query": query})
            intent = result.content.strip().upper()
        except Exception:
            intent = "FACTUAL"

        if intent not in ("RECOMMEND", "IRDAI", "FACTUAL", "ANALYTICAL", "CLARIFY", "CALCULATE"):
            intent = "FACTUAL"


    try:
        _trace_intent(lf_client, query, intent, session_id)
    except Exception:
        pass

    updated = dict(state)
    updated["intent"] = intent
    updated["route"]  = intent.lower()
    return updated

def route_after_guardrail(state: InsuranceState) -> str:
    if state.get("route") == "blocked": return "blocked"
    return "intent"

def route_after_intent(state: InsuranceState) -> str:
    return state.get("route", "factual")

def build_routing_graph() -> StateGraph:
    graph = StateGraph(InsuranceState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("intent",    intent_node)
    graph.set_entry_point("guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"blocked": END, "intent": "intent"}
    )
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "recommend":  END,
            "irdai":      END,
            "factual":    END,
            "analytical": END,
            "clarify":    END,
            "calculate":  END,
        }
    )

    return graph.compile()

_routing_graph = build_routing_graph()

def check_plan_type_query(query: str):
    query_lower = query.lower()
    # Never intercept document/eligibility queries even if plan name is present
    if any(w in query_lower for w in ["document", "eligib", "require", "kyc", "proof"]):
        return None
    plan_type_triggers = [
        "type of plan", "what type", "kind of plan", "what kind of plan",
        "is it a", "is this a", "par or non", "linked or non", "plan type",
        "category of plan", "classified as", "what plan type",
    ]
    if not any(trigger in query_lower for trigger in plan_type_triggers):
        return None
    for keyword, answer in PLAN_TYPE_MAP.items():
        if keyword in query_lower: return answer
    return None

def _format_profile_summary(profile: dict) -> str:
    goal_map = {
        "pure life protection": "Protection", "life protection": "Protection",
        "pure protection": "Protection", "pure risk": "Protection",
        "term insurance": "Protection",
        "1": "Pension", "pension": "Pension", "regular pension": "Pension",
        "retirement": "Pension", "regular income": "Pension",
        "2": "Lump Sum", "lump sum": "Lump Sum", "maturity": "Lump Sum",
        "3": "Protection", "term": "Protection", "protect": "Protection", "pure": "Protection",
        "4": "Savings", "long-term": "Savings", "endowment": "Savings", "long term": "Savings",
        "5": "Child Planning", "child": "Child Planning", "children": "Child Planning",
        "education": "Child Planning", "kid": "Child Planning",
    }
    goal_display  = goal_map.get(profile.get("goal", ""), str(profile.get("goal", "")).title())
    specific      = profile.get("specific_plans", [])
    mode          = profile.get("mode", "suggestion")
    
    specific_disp = ", ".join([PLAN_NAME_MAP.get(p, p) for p in specific]) if specific else "Seeking best matches"
    mode_display  = "⚖️ Comparison Analysis" if mode == "comparison" else "💡 Smart Suggestion"

    return (
        f"### 📋 Client Profile Snapshot\n"
        f"---\n"
        f"**👤 Age:** {profile.get('age')} yrs  |  **👨‍👩‍👧 Dependents:** {profile.get('dependents')}\n\n"
        f"**💼 Ann. Income:** ₹{profile.get('annual_income'):,}  |  **💳 Mo. Expenses:** ₹{profile.get('monthly_spending'):,}\n\n"
        f"**🎯 Primary Goal:** {goal_display}\n\n"
        f"**🔎 Mode:** {mode_display} ({specific_disp})\n"
        f"---\n"
    )

def _build_source_elements(reranked) -> list:
    source_elements = []
    seen_sources    = set()
    source_num      = 1
    for doc, score in reranked:
        source_file = os.path.basename(
            doc.metadata.get('source', 'unknown')
        ).replace('.pdf', '').replace('_', ' ')
        page       = doc.metadata.get('page', '?')
        source_key = f"{source_file}_p{page}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            content = (
                f"📄 **{source_file}**\n"
                f"📑 Page: {page}\n\n"
                f"**Relevant excerpt:**\n\n"
                f"{doc.page_content[:400].strip()}"
            )
            source_elements.append(
                cl.Text(
                    name=f"[{source_num}] {source_file} · Page {page}",
                    content=content,
                    display="side"
                )
            )
            source_num += 1
    return source_elements

async def _generate_dynamic_suggestions(user_query: str, bot_response: str) -> list[str]:
    groq_client = cl.user_session.get("groq_client")
    
    past_suggestions = cl.user_session.get("past_suggestions", [])
    past_str = ", ".join(past_suggestions[-15:]) if past_suggestions else "None"
    
    prompt = f"""
    You are a proactive insurance assistant. 
    User asked: "{user_query}"
    Bot replied: "{bot_response[:400]}..."
    
    Generate exactly 3 short, highly relevant follow-up questions.
    CRITICAL RULES:
    1. Keep them under 8 words each. 
    2. Format strictly as a comma-separated list.
    3. DO NOT suggest vague things like "Compare policies". If you suggest comparison, name a specific policy.
    4. Focus on specific details (e.g., "What is the policy term?", "How to pay premiums?").
    5. DO NOT repeat the user's question or ANY of these past suggestions: [{past_str}].
    """
    
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=60
        ))
        
        suggestions = response.choices[0].message.content.strip().split(',')
        valid_suggestions = [s.strip() for s in suggestions if s.strip()][:3]
        
        past_suggestions.extend(valid_suggestions)
        cl.user_session.set("past_suggestions", past_suggestions)
        return valid_suggestions
        
    except Exception:
        return ["Suggest a plan", "What are the tax benefits?"]

async def _needs_policy_disambiguation(query: str) -> bool:
    if cl.user_session.get("active_plan") or detect_specific_plans(query):
        return False
        
    groq_client = cl.user_session.get("groq_client")
    prompt = f"""
    User query: "{query}"
    
    Classify this query. 
    Return 'SPECIFIC' if the user is asking about ANY policy feature, condition, period, limit, rule, or benefit (e.g., "free look period", "grace period", "maturity age", "surrender value", "death benefit"). We MUST ask them which policy they mean.
    Return 'GENERIC' ONLY if the user is asking for a pure dictionary definition of a general insurance concept (e.g., "what is a nominee?", "what is premium?", "how to pay generally") or a casual greeting.
    
    Reply with EXACTLY ONE WORD: SPECIFIC or GENERIC.
    """
    
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        ))
        answer = response.choices[0].message.content.strip().upper()
        return answer == "SPECIFIC"
    except Exception:
        return False


async def _run_rag(query: str, memory_prefix: str = ""):
    llm              = cl.user_session.get("llm")
    langfuse_handler = cl.user_session.get("langfuse_handler")
    vectorstore      = cl.user_session.get("vectorstore")
    reranker         = cl.user_session.get("reranker")

    # 1. Disambiguation Check with Dynamic UI String
    if await _needs_policy_disambiguation(query):
        actions = [
            cl.Action(name="select_policy", payload={"query": query, "plan": plan}, label=plan) 
            for plan in DISPLAY_PLANS
        ]
        await cl.Message(
            content=f"🎯 **Select a Policy**\n*Which policy's **{query}** are you looking for? Details vary by plan.*", 
            actions=actions
        ).send()
        return

    active_plan = cl.user_session.get("active_plan")
    search_query = f"{active_plan} {query}" if active_plan and active_plan.lower() not in query.lower() else query

    # 2. Semantic Cache Bypass (Ignore cache if query is short, to prevent false collisions)
    word_count = len(query.strip().split())
    cached = None
    if word_count >= 4 and not active_plan:
        cached = get_cached_answer(search_query)

    if cached:
        suggestions = await _generate_dynamic_suggestions(query, cached)
        actions = [cl.Action(name="follow_up", payload={"value": sug}, label=sug) for sug in suggestions]
        prefix = f"*(Viewing details for **{active_plan}**)*\n\n" if active_plan else ""
        await cl.Message(content=prefix + cached, actions=actions).send()
        add_to_memory(query, cached)
        return

    retrieval_query = expand_query(search_query)
    retrieved = retrieve_chunks(retrieval_query, vectorstore)
    reranked  = rerank_chunks(search_query, retrieved, reranker)

    msg = cl.Message(content="")
    
    if active_plan:
        msg.content = f"*(Viewing details for **{active_plan}**)*\n\n"
        
    await msg.send()

    if reranked is None:
        msg.content += "I cannot find specific information about this in the available policy documents."
        await msg.update()
        return

    # 3. STRICT RAG INSTRUCTION: Synthesize if conversational, strictly refuse if missing
    if active_plan:
        memory_prefix += (
            f"\n[INSTRUCTION: The user is asking about {active_plan}. "
            f"The retrieved chunks below are already filtered to this plan. "
            f"Extract and present the answer directly from the chunks even if the wording uses pipe characters | or table formatting — read through the formatting to find the actual content. "
            f"If the chunks genuinely contain no relevant information at all, say so. "
            f"Do not refuse to answer if the information is present, even partially.]\n\n"
        )
    else:
        memory_prefix += (
            f"\n[INSTRUCTION: Answer from the retrieved chunks below. "
            f"Read through any pipe | characters or table formatting artifacts to find the actual content. "
            f"If the chunks genuinely contain no relevant information, say so clearly.]\n\n"
        )
    def _clean_chunk_artifacts(text: str) -> str:
        import re
        # Remove leading pipe characters from table-extracted PDF lines
        text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)
        # Collapse multiple spaces
        text = re.sub(r'  +', ' ', text)
        return text.strip()

    cleaned_reranked = [
        (type('Doc', (), {
            'page_content': _clean_chunk_artifacts(doc.page_content),
            'metadata': doc.metadata
        })(), score)
        for doc, score in reranked
    ]
    context = memory_prefix + format_context(cleaned_reranked)  

    from prompts import get_rag_prompt, is_regulatory_query, get_regulatory_prompt
    if is_regulatory_query(query):
        prompt = get_regulatory_prompt()
    else:
        prompt = get_rag_prompt()

    chain = prompt | llm

    full_response = ""
    try:
        async for chunk in chain.astream(
            {"context": context, "question": search_query},
            config={"callbacks": [langfuse_handler]}
        ):
            full_response += chunk.content
            await msg.stream_token(chunk.content)
            
        if word_count >= 4:
            store_cached_answer(search_query, full_response)
            
        add_to_memory(query, full_response)
        
        suggestions = await _generate_dynamic_suggestions(query, full_response)
        actions = [cl.Action(name="follow_up", payload={"value": sug}, label=sug) for sug in suggestions]
        
        if active_plan:
            actions.append(cl.Action(name="clear_context", payload={"value": "clear"}, label="🔄 Change Policy"))

        msg.elements = _build_source_elements(reranked)
        msg.actions = actions
        await msg.update()
        
    except Exception as e:
        msg.content = "I'm sorry, I am having trouble connecting to my servers right now. Please try again."
        await msg.update()

async def _run_agent_pipeline(profile, vectorstore, reranker, groq_client):
    loop = asyncio.get_event_loop()
    mode           = profile.get("mode", "suggestion")
    specific_plans = profile.get("specific_plans", [])

    if mode == "suggestion":
        await cl.Message(
            content="🔍 **Processing Smart Suggestion** — scanning documents for optimal matches..."
        ).send()

        async with cl.Step(name="⚙️ AI Actuary: Shortlisting Plans") as step:
            shortlisted, reasoning = await loop.run_in_executor(
                None, lambda: shortlist_plans(profile, groq_client)
            )
            display_names = [PLAN_NAME_MAP.get(p, p) for p in shortlisted]
            step.output = f"Selected: {', '.join(display_names)}"

        cards = "\n".join([f"> 🛡️ **{name}**" for name in display_names])
        await cl.Message(
            content=f"### 🎯 Top Matches Found\n\n{cards}\n\n*Rationale: {reasoning}*"
        ).send()

    else:
        shortlisted   = specific_plans
        display_names = [PLAN_NAME_MAP.get(p, p) for p in shortlisted]
        reasoning     = "User specified these plans directly for comparison."
        
        cards = "\n".join([f"> ⚖️ **{name}**" for name in display_names])
        await cl.Message(
            content=f"### 📊 Comparison Analysis Initiated\n\n{cards}"
        ).send()

    plan_analyses = {}
    for plan_file in shortlisted:
        plan_name = PLAN_NAME_MAP.get(plan_file, plan_file)
        async with cl.Step(name=f"📄 Deep Scan: {plan_name}") as step:
            analysis = await loop.run_in_executor(
                None,
                lambda pf=plan_file: analyse_plan(
                    plan_filename=pf,
                    profile=profile,
                    vectorstore=vectorstore,
                    reranker=reranker,
                    groq_client=groq_client
                )
            )
            plan_analyses[plan_file] = analysis
            step.output = f"Analysis complete for {plan_name}."

    async with cl.Step(name="✅ Finalizing Recommendation") as step:
        recommendation = await loop.run_in_executor(
            None,
            lambda: generate_recommendation(
                profile=profile,
                plan_analyses=plan_analyses,
                shortlist_reasoning=reasoning,
                groq_client=groq_client
            )
        )
        step.output = "Report generated."

    await asyncio.sleep(0.3)

    if len(shortlisted) == 1:
        cl.user_session.set("active_plan", PLAN_NAME_MAP.get(shortlisted[0], shortlisted[0]))
    else:
        cl.user_session.set("active_plan", None)

    await cl.Message(content=recommendation).send()

    # ── Extract winning plan from recommendation text ─────────────────────
    # The recommender names the winner in the 🏆 section — detect it
    recommended_plan_file = _extract_recommended_plan(recommendation, shortlisted)
    await _show_recommendation_premium(profile, [recommended_plan_file] + [
        p for p in shortlisted if p != recommended_plan_file
    ], groq_client)

    suggestions = await _generate_dynamic_suggestions("Provide plan recommendations", recommendation)
    actions = [
        cl.Action(name="follow_up", payload={"value": sug}, label=sug)
        for sug in suggestions
    ]
    actions.append(
        cl.Action(
            name="follow_up",
            payload={"value": "Calculate premium for different sum assured"},
            label="💰 Try Different Sum Assured"
        )
    )

    cl.user_session.set("agent_state", "done")
    await cl.Message(
        content="---\n💬 *What would you like to explore next?*",
        actions=actions
    ).send()

def _extract_recommended_plan(recommendation_text: str, shortlisted: list) -> str:
    """
    Scans the recommendation text for the 🏆 Recommendation section
    and returns the plan file that the LLM named as the winner.
    Falls back to shortlisted[0] if detection fails.
    """
    # Map display keywords to plan files
    keyword_to_file = {
        "jeevan labh":  "LIC_Jeevan_Labh.pdf",
        "jeevan anand": "LIC_Jeevan_Anand.pdf",
        "jeevan umang": "LIC_Jeevan_Umang.pdf",
        "tech term":    "LIC_Tech-Term.pdf",
        "endowment":    "LIC_Endowment.pdf",
        "money back":   "LIC_ChidrensMoney_BackPlan.pdf",
        "jeevan amar":  "LIC_New_Jeevan_Amar.pdf",
    }

    # Only scan the recommendation section — find 🏆 block
    text_lower = recommendation_text.lower()
    recommendation_section = text_lower

    trophy_idx = text_lower.find("🏆")
    if trophy_idx != -1:
        recommendation_section = text_lower[trophy_idx:]

    # Find first plan keyword mentioned in that section
    for keyword, plan_file in keyword_to_file.items():
        if keyword in recommendation_section and plan_file in shortlisted:
            return plan_file

    # Fallback
    return shortlisted[0] if shortlisted else ""

async def _show_recommendation_premium(profile: dict, shortlisted: list, groq_client):
    """
    Automatically calculates and displays premium for the top recommended plan
    using the user's profile data collected during the agentic flow.

    Sum assured derivation:
      - Term plans (tech_term, jeevan_amar): 15× annual income
      - Savings/endowment plans: 10× annual income
      - Capped at plan minimums if derived figure is too low
    Shown as a recommended figure — user can ask to recalculate with different SA.
    """
    from insurance_agent.agent.premium_calculator import (
        calculate_premium, format_premium_result, PLAN_META
    )

    if not shortlisted:
        return

    # Use the top recommended plan (first in shortlist)
    top_plan_file = shortlisted[0]
    top_plan_name = PLAN_NAME_MAP.get(top_plan_file, top_plan_file)

    # Map plan file to calculator plan_name key
    plan_key_map = {
        "LIC_Tech-Term.pdf":              "tech_term",
        "LIC_Jeevan_Anand.pdf":           "jeevan_anand",
        "LIC_Endowment.pdf":              "endowment",
        "LIC_Jeevan_Labh.pdf":            "jeevan_labh",
        "LIC_Jeevan_Umang.pdf":           "jeevan_umang",
        "LIC_New_Jeevan_Amar.pdf":        "jeevan_amar",
        "LIC_ChidrensMoney_BackPlan.pdf": "money_back",
    }
    plan_key = plan_key_map.get(top_plan_file)
    if not plan_key:
        return

    age           = profile.get("age", 30)
    annual_income = profile.get("annual_income", 500_000)
    goal          = profile.get("goal", "savings")

    # Derive affordable sum assured
    # Cap at 25% of annual income as yearly premium — work backwards from affordability
    # Standard LIC thumb rule: SA = 10x income, but cap so premium <= 20% of income
    term_plans = ["tech_term", "jeevan_amar"]
    if plan_key in term_plans:
        recommended_sa = annual_income * 15
    else:
        recommended_sa = annual_income * 10

    # Affordability cap — premium should not exceed 20% of annual income
    # Rough check: base premium ≈ SA × 0.055 for endowment plans
    # So max SA = (annual_income × 0.20) / 0.055
    max_affordable_sa = int((annual_income * 0.20) / 0.055)
    recommended_sa    = min(recommended_sa, max_affordable_sa)

    # Enforce plan minimums
    meta       = PLAN_META.get(plan_key, {})
    min_sa     = meta.get("min_sa", 100_000)
    sum_assured = max(int(recommended_sa), min_sa)
    # Round to nearest lakh for clean display
    sum_assured = round(sum_assured / 100_000) * 100_000

    # Derive sensible policy term from goal and age
    goal_term_map = {
        "pension":        30,
        "lump_sum":       20,
        "protection":     25,
        "savings":        20,
        "child_planning": 25,
    }
    policy_term = goal_term_map.get(goal, 20)

    # For Jeevan Umang use PPT not policy term
    ppt = None
    if plan_key == "jeevan_umang":
        ppt         = 20
        policy_term = 100

    # Derive mode from income — lower income → monthly, higher → yearly
    if annual_income < 300_000:
        mode = "monthly"
    elif annual_income < 600_000:
        mode = "quarterly"
    else:
        mode = "yearly"

    result = calculate_premium(
        plan_name=           plan_key,
        age=                 age,
        sum_assured=         sum_assured,
        policy_term=         policy_term,
        premium_paying_term= ppt,
        mode=                mode,
    )

    if not result["success"]:
        # Silent fail — don't show error in recommendation flow
        return

    formatted = format_premium_result(result)

    # Add a header explaining this is auto-calculated from their profile
    sa_in_lakhs = sum_assured // 100_000
    header = (
        f"---\n"
        f"### 💰 Indicative Premium for Your Profile\n\n"
        f"*Based on your age ({age} yrs) and income (₹{annual_income:,}), "
        f"a recommended Sum Assured of **₹{sum_assured:,} ({sa_in_lakhs}L)** "
        f"gives you the following premium estimate for {top_plan_name}:*\n\n"
    )

    # Add recalculate action
    actions = [
        cl.Action(
            name="follow_up",
            payload={"value": f"Calculate premium for {top_plan_name} with 25 lakhs sum assured"},
            label="🔁 Recalculate with 25L"
        ),
        cl.Action(
            name="follow_up",
            payload={"value": f"Calculate premium for {top_plan_name} with 50 lakhs sum assured"},
            label="🔁 Recalculate with 50L"
        ),
        cl.Action(
            name="follow_up",
            payload={"value": f"What is monthly premium for {top_plan_name}"},
            label="📅 Show Monthly Premium"
        ),
    ]

    await cl.Message(
        content=header + formatted,
        actions=actions
    ).send()

async def show_welcome_dashboard():
    """Helper to render the beautiful welcome dashboard."""
    welcome_msg = """
### 🏦 Welcome to LIC SecureAdvisor™
*Your Personal Digital Insurance Expert*
***

I can assist you with:
🔍 **Instant Policy Queries** — Ask about premiums, benefits, or claim rules.
📊 **Smart Recommendations** — Type *"Suggest a plan"* for tailored advice.
⚖️ **Direct Comparisons** — E.g., *"Compare Jeevan Anand and Tech Term"*.

*How can I help secure your future today?*
"""
    actions = [
        cl.Action(name="follow_up", payload={"value": "Suggest a plan for me"}, label="💡 Suggest a Plan"),
        cl.Action(name="follow_up", payload={"value": "What is the claim settlement ratio?"}, label="📈 View IRDAI Stats"),
        cl.Action(name="follow_up", payload={"value": "I want to calculate my LIC policy premium"}, label="💰 Calculate Premium"),
        cl.Action(name="follow_up", payload={"value": "Which ones are pure term?"}, label="🛡️ Filter Pure Term"),
    ]
    await cl.Message(content=welcome_msg, actions=actions).send()


@cl.on_chat_start
async def on_chat_start():
    loading = cl.Message(content="⚙️ Initializing SecureAdvisor Agent...")
    await loading.send()

    vectorstore      = get_retriever()
    reranker         = load_reranker()
    llm              = load_llm()
    langfuse_handler = get_langfuse_handler()
    lf_client        = get_langfuse_client()

    embedder = load_embedder()
    try:
        init_semantic_cache(embedder)
    except Exception:
        pass

    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    import uuid
    session_id = str(uuid.uuid4())

    cl.user_session.set("vectorstore",          vectorstore)
    cl.user_session.set("reranker",             reranker)
    cl.user_session.set("llm",                  llm)
    cl.user_session.set("langfuse_handler",     langfuse_handler)
    cl.user_session.set("lf_client",            lf_client)
    cl.user_session.set("session_id",           session_id)
    cl.user_session.set("groq_client",          groq_client)
    cl.user_session.set("active_mode",          "rag")
    cl.user_session.set("collector",            None)
    cl.user_session.set("agent_state",          None)
    cl.user_session.set("agent_profile",        None)
    
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("past_suggestions",     [])
    cl.user_session.set("is_processing",        False)
    cl.user_session.set("active_plan",          None) 

    await loading.remove()
    await show_welcome_dashboard()


# ── Dynamic Filtering Helper ──
def _get_plans_for_goal(goal: str) -> list[str]:
    mapping = {
        "protection": ["LIC Tech Term", "LIC New Jeevan Amar"],
        "pension": ["LIC Jeevan Umang"],
        "lump_sum": ["LIC Jeevan Anand", "LIC New Endowment Plan", "LIC Jeevan Labh"],
        "savings": ["LIC Jeevan Anand", "LIC New Endowment Plan", "LIC Jeevan Labh"],
        "child_planning": ["LIC Children's Money Back Plan"]
    }
    return mapping.get(goal, [])

async def _get_progressive_hint(partial_profile: dict) -> str:
    groq_client = cl.user_session.get("groq_client")
    active_fields = {k: v for k, v in partial_profile.items() if v and k not in ["specific_plans", "mode"]}
    if not active_fields: return ""

    prompt = f"""
    You are an AI insurance recommender. The user is partially through a profile questionnaire.
    Current known details: {active_fields}.
    Based ONLY on this, name 1 or 2 LIC plans that broadly align.
    Keep it to exactly one short, friendly sentence.
    """
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=60
        ))
        return response.choices[0].message.content.strip()
    except Exception:
        return "Scanning database for optimal matches..."

@cl.on_message
async def on_message(message: cl.Message):
    if cl.user_session.get("is_processing"):
        await cl.Message(content="⏳ *I am currently processing your previous request. Please wait a moment...*").send()
        return

    cl.user_session.set("is_processing", True)
    try:
        await _handle_message_logic(message)
    except Exception as e:
        import traceback
        print(f"[CRITICAL ERROR]\n{traceback.format_exc()}")
        await cl.Message(content=f"⚠️ Error: {str(e)[:200]}").send()
    finally:
        cl.user_session.set("is_processing", False)

async def _handle_message_logic(message: cl.Message):
    original_query = message.content.strip()
    active_mode = cl.user_session.get("active_mode")
    agent_state = cl.user_session.get("agent_state")

    if active_mode == "rag" or agent_state == "lobby" or agent_state == "done":
        query = await _rewrite_query_with_context(original_query)
    else:
        query = original_query
        
    query_lower = query.lower()
    # --- BUG 5 FIX: Early age validation for recommendation queries ---
    if any(t in query_lower for t in AGENT_TRIGGERS):
        from insurance_agent.agent.profile_collector import extract_fields_from_text
        extracted_early = extract_fields_from_text(query)
        early_age = extracted_early.get("age")
        if early_age is not None and (early_age < 18 or early_age > 75):
            await cl.Message(
                content=(
                    f"⚠️ **Age Not Eligible**\n\n"
                    f"LIC plans are generally available for individuals between **18 and 75 years** of age. "
                    f"At age **{early_age}**, you may not be eligible for most plans in our catalogue.\n\n"
                    f"Please consult a licensed LIC advisor for options specific to your situation."
                )
            ).send()
            return

    detected_plans = detect_specific_plans(query)
    if len(detected_plans) == 1:
        cl.user_session.set("active_plan", PLAN_NAME_MAP.get(detected_plans[0], detected_plans[0]))
    elif len(detected_plans) > 1:
        cl.user_session.set("active_plan", None)

    # --- BUG 1 FIX: Clear active_plan for any multi-plan or generic query ---
    GENERIC_QUERY_SIGNALS = [
        "which plans", "which policies", "what plans", "list",
        "pure term", "pension", "provide", "offer", "available",
        "all plans", "compare", "vs", "versus",
    ]
    should_clear_plan = (
        any(phrase in query_lower for phrase in DOCUMENT_QUERIES)
        or any(trigger in query_lower for trigger in AGENT_TRIGGERS)
        or any(trigger in query_lower for trigger in IRDAI_TRIGGERS)
        or any(sig in query_lower for sig in GENERIC_QUERY_SIGNALS)
        or len(detect_specific_plans(query)) > 1  # multiple plans mentioned
    )
    if should_clear_plan:
        cl.user_session.set("active_plan", None)

    if active_mode == "agent" and agent_state == "lobby":
        cancel_triggers = ["exit", "cancel", "stop", "quit", "nevermind", "abort", "no"]
        if query_lower in cancel_triggers:
            cl.user_session.set("active_mode", "rag")
            cl.user_session.set("agent_state", None)
            cl.user_session.set("collector", None)
            await cl.Message(content="🛑 Cancelled. What else would you like to know?").send()
            return

        # ── Comparison request detected in lobby ──────────────────────────
        detected_plans = detect_specific_plans(query)
        compare_triggers = ["compare", "vs", "versus", "difference", "comparison", "between"]
        if len(detected_plans) >= 2 or (detected_plans and any(t in query_lower for t in compare_triggers)):
            await _handle_comparison_from_lobby(query, detected_plans)
            return

        from insurance_agent.agent.profile_collector import parse_goal, extract_fields_from_text
        goal = parse_goal(query)
        extracted = extract_fields_from_text(query)

        if goal and not extracted.get('age') and not extracted.get('annual_income'):
            await _handle_recommend(query)
            return

        if "start" in query_lower or "yes" in query_lower or "recommend" in query_lower or extracted:
            cl.user_session.set("agent_state", "collecting")
            if not cl.user_session.get("collector"):
                cl.user_session.set("collector", ProfileCollector())
            await _handle_collecting(query)
            return

        cl.user_session.set("active_mode", "rag")
        cl.user_session.set("agent_state", None)
        cl.user_session.set("collector", None)
        active_mode = "rag"

    cancel_triggers = ["exit", "cancel", "stop", "quit", "nevermind", "abort", "no"]
    if active_mode == "agent" and query_lower in cancel_triggers:
        cl.user_session.set("active_mode", "rag")
        cl.user_session.set("agent_state", None)
        cl.user_session.set("agent_profile", None)
        cl.user_session.set("collector", None)
        await cl.Message(content="🛑 Cancelled. I've exited the recommendation process. What else would you like to know?").send()
        return

    if query_lower in GREETING_RESPONSES:
        await cl.Message(content=GREETING_RESPONSES[query_lower]).send()
        return

    # ── MENTOR PROOFING: Handle Meta Questions Beautifully ──
    if any(phrase in query_lower for phrase in DOCUMENT_QUERIES):
        await show_welcome_dashboard()
        return

    plan_type_answer = check_plan_type_query(query)
    if plan_type_answer:
        await cl.Message(content=plan_type_answer).send()
        return

    # ── Pre-graph IRDAI fast-path (before guardrail runs) ────────────────
    if any(t in query_lower for t in IRDAI_TRIGGERS):
        msg = cl.Message(content="🔍 *Searching live IRDAI databases and combining with policy files...*")
        await msg.send()
        async_groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        session_id  = cl.user_session.get("session_id")
        vectorstore = cl.user_session.get("vectorstore")
        try:
            final_answer = await handle_irdai_query(
                query=query,
                groq_client=async_groq_client,
                vectorstore=vectorstore,
                session_id=session_id
            )
            suggestions = await _generate_dynamic_suggestions(query, final_answer)
            actions = [cl.Action(name="follow_up", payload={"value": sug}, label=sug) for sug in suggestions]
            msg.content = final_answer
            msg.actions  = actions
            await msg.update()
            add_to_memory(query, final_answer)
        except Exception:
            msg.content = "I encountered an error fetching live data right now. Please try again."
            await msg.update()
        return

    # ── Pre-graph calculator fast-path ────────────────────────────────────
    # Catches "how much is the premium" before guardrail can block it
    if any(t in query_lower for t in CALCULATOR_TRIGGERS):
        # Only fast-path if a plan is mentioned or active_plan is set
        # Otherwise let it go through LangGraph so intent node can confirm
        active_plan = cl.user_session.get("active_plan")
        detected    = detect_specific_plans(query)
        if active_plan or detected:
            await _handle_calculator(query)
            return
    # ── Comparison age collection state ──────────────────────────────────
    if active_mode == "agent" and agent_state == "comparison_age":
        from insurance_agent.agent.profile_collector import parse_number, extract_fields_from_text
        extracted = extract_fields_from_text(query)
        age = extracted.get("age")
        if age is None:
            num = parse_number(query)
            if num and 18 <= num <= 75:
                age = int(num)

        if age is None:
            await cl.Message(content="Sorry, I didn't catch your age. Could you type it as a number? *(e.g. 28)*").send()
            return

        detected_plans = cl.user_session.get("pending_comparison_plans", [])
        goal           = cl.user_session.get("pending_comparison_goal", "savings")
        groq_client    = cl.user_session.get("groq_client")
        vectorstore    = cl.user_session.get("vectorstore")
        reranker       = cl.user_session.get("reranker")

        profile = {
            "age":              age,
            "annual_income":    500_000,
            "dependents":       0,
            "monthly_spending": 20_000,
            "goal":             goal,
            "specific_plans":   detected_plans,
            "mode":             "comparison",
        }

        cl.user_session.set("agent_profile", profile)
        cl.user_session.set("agent_state",   "processing")
        cl.user_session.set("active_mode",   "agent")

        plan_names = [PLAN_NAME_MAP.get(p, p) for p in detected_plans]
        cards      = " vs ".join([f"**{n}**" for n in plan_names])

        await cl.Message(content=f"📊 Comparing {cards} for age {age}...").send()
        await _run_agent_pipeline(profile, vectorstore, reranker, groq_client)
        return

    if active_mode == "agent" and agent_state == "collecting":
        await _handle_collecting(query)
        return

    if active_mode == "agent" and agent_state == "confirming":
        await _handle_confirming(query)
        return

    if active_mode == "agent" and agent_state == "processing":
        await cl.Message(content="⏳ Still analyzing your profile — please wait a moment...").send()
        return

    if active_mode == "agent" and agent_state == "done":
        await _handle_post_agent(query)
        return
    
    # --- BUG 3 FIX: Direct comparison intercept (any mode) ---
    detected_plans_early = detect_specific_plans(query)
    compare_triggers_early = ["compare", "vs", "versus", "difference", "comparison", "between"]
    if (
        len(detected_plans_early) >= 2
        or (len(detected_plans_early) == 1 and any(t in query_lower for t in compare_triggers_early))
    ):
        await _handle_comparison_from_lobby(query, detected_plans_early)
        return

    initial_state: InsuranceState = {
        "query":                query,
        "query_lower":          query_lower,
        "is_insurance":         None,
        "intent":               None,
        "route":                None,
        "response":             None,
        "blocked_reason":       None,
        "conversation_history": cl.user_session.get("conversation_history", [])
    }

    final_state = await _routing_graph.ainvoke(initial_state)
    route = final_state.get("route", "factual")

    if route == "blocked":
        await cl.Message(content=random.choice(GUARDRAIL_RESPONSES)).send()
        return
        
    if route == "clarify":
        msg = "Could you specify what you'd like details on? I can help you compare plans, fetch IRDAI stats, or check specific policy rules."
        await cl.Message(content=msg).send()
        return

    if route == "recommend":
        await _handle_recommend(query)
        return

    if route == "irdai":
        msg = cl.Message(content="🔍 *Searching live IRDAI databases and combining with policy files...*")
        await msg.send()
        
        async_groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        session_id = cl.user_session.get("session_id")
        vectorstore = cl.user_session.get("vectorstore")
        
        try:
            final_answer = await handle_irdai_query(
                query=query, 
                groq_client=async_groq_client, 
                vectorstore=vectorstore, 
                session_id=session_id
            )
            suggestions = await _generate_dynamic_suggestions(query, final_answer)
            actions = [cl.Action(name="follow_up", payload={"value": sug}, label=sug) for sug in suggestions]
            
            msg.content = final_answer
            msg.actions = actions
            await msg.update()
            add_to_memory(query, final_answer)
        except Exception:
            msg.content = "I encountered an error fetching live data right now. Please try again in a moment."
            await msg.update()
            
        return
    if route == "calculate":
        await _handle_calculator(query)
        return
    
    memory_prefix = get_memory_context_prefix()
    await _run_rag(query, memory_prefix)


# ── Action Handlers ──────────────────────────────────────────────────────────

@cl.action_callback("follow_up")
async def on_action(action: cl.Action):
    if cl.user_session.get("is_processing"):
        return

    try:
        await action.remove() 
    except Exception:
        pass

    action_value = action.payload.get("value", "Tell me more") if isinstance(action.payload, dict) else str(action.payload)
    await cl.Message(content=action_value, author="User").send()
    
    msg = cl.Message(content=action_value)
    await on_message(msg)

@cl.action_callback("clear_context")
async def on_clear_context(action: cl.Action):
    if cl.user_session.get("is_processing"):
        return
    try:
        await action.remove()
    except Exception:
        pass
        
    cl.user_session.set("active_plan", None)
    await cl.Message(content="🔄 *Policy focus cleared. You can now ask about any plan.*").send()

@cl.action_callback("select_policy")
async def on_policy_select(action: cl.Action):
    if cl.user_session.get("is_processing"):
        return

    try:
        await action.remove()
    except Exception:
        pass

    payload = action.payload
    original_query = payload.get("query", "")
    selected_plan = payload.get("plan", "")
    
    cl.user_session.set("active_plan", selected_plan)
    
    new_query = f"{original_query} for {selected_plan}"
    await cl.Message(content=f"Selected: **{selected_plan}**", author="User").send()
    
    msg = cl.Message(content=new_query)
    await on_message(msg)

@cl.action_callback("start_recommendation")
async def on_start_recommendation(action: cl.Action):
    if cl.user_session.get("is_processing"): return
    try: await action.remove() 
    except: pass
    
    cl.user_session.set("agent_state", "collecting")
    
    # ── FORCE WIPE Profile Collector to prevent memory bleed ──
    collector = ProfileCollector()
    cl.user_session.set("collector", collector)
    cl.user_session.set("agent_profile", None)
        
    missing = collector._missing_fields()
    if missing:
        from insurance_agent.agent.profile_collector import QUESTIONS
        next_q = QUESTIONS[missing[0]]
        await cl.Message(content=f"Great! Let's find the exact best plan for you.\n\n**{next_q}**").send()

@cl.action_callback("cancel_agent")
async def on_cancel_agent(action: cl.Action):
    if cl.user_session.get("is_processing"): return
    try: await action.remove() 
    except: pass
    
    cl.user_session.set("active_mode", "rag")
    cl.user_session.set("agent_state", None)
    cl.user_session.set("collector", None)
    await cl.Message(content="🛑 Cancelled. You can ask me any factual questions about the policies!").send()


# ── Mid-Flow Handlers ────────────────────────────────────────────────────────

async def _handle_collecting(query: str):
    collector = cl.user_session.get("collector")
    
    response_text, partial_profile, is_done = collector.handle(query)
    cl.user_session.set("collector", collector)

    if is_done:
        cl.user_session.set("agent_profile", partial_profile)
        cl.user_session.set("agent_state", "confirming")
        profile_summary = _format_profile_summary(partial_profile)
        
        actions = [cl.Action(name="follow_up", payload={"value": "Yes"}, label="✅ Yes, proceed")]
        
        await cl.Message(
            content=(
                f"{response_text}\n\n{profile_summary}\n\n"
                f"**Does this look correct?**\n"
                f"Reply **YES** to proceed or tell me what to correct."
            ),
            actions=actions
        ).send()
    else:
        if partial_profile and partial_profile.get("mode") != "comparison" and len(partial_profile) >= 1:
            msg = cl.Message(content="*Updating plan matches...*")
            await msg.send()
            hint = await _get_progressive_hint(partial_profile)
            msg.content = f"💡 *{hint}*\n\n{response_text}"
            await msg.update()
        else:
            await cl.Message(content=response_text).send()

async def _handle_calculator(query: str):
    """
    MCP-style premium calculator handler.
    Passes query + conversation context to run_calculator_tool().
    The tool schema is sent to Groq — the LLM extracts parameters autonomously.
    """
    groq_client = cl.user_session.get("groq_client")
 
    # Pass conversation memory as context so LLM can extract
    # age/plan from prior messages (e.g. user already said they are 28)
    # --- BUG 4 FIX: Inject agent_profile into calculator context ---
    context = format_memory()
    agent_profile = cl.user_session.get("agent_profile")
    if agent_profile:
        context = (
            f"[USER PROFILE — already collected. Use these values directly, do NOT ask again.]\n"
            f"Age: {agent_profile.get('age')} | "
            f"Annual Income: ₹{agent_profile.get('annual_income', 0):,} | "
            f"Goal: {agent_profile.get('goal', 'savings')}\n\n"
        ) + context
 
    msg = cl.Message(content="⚙️ *Calculating premium...*")
    await msg.send()
 
    try:
        result = await run_calculator_tool(
            query=query,
            groq_client=groq_client,
            context=context,
        )
        suggestions = await _generate_dynamic_suggestions(query, result)
        actions = [
            cl.Action(name="follow_up", payload={"value": sug}, label=sug)
            for sug in suggestions
        ]
        msg.content = result
        msg.actions  = actions
        await msg.update()
        add_to_memory(query, result)
 
    except Exception as e:
        import traceback
        print(f"[Calculator Error] {traceback.format_exc()}")
        msg.content = "❌ Premium calculation failed. Please try again."
        await msg.update()

async def _handle_confirming(query: str):
    groq_client = cl.user_session.get("groq_client")
    vectorstore = cl.user_session.get("vectorstore")
    reranker    = cl.user_session.get("reranker")
    profile     = cl.user_session.get("agent_profile")
    collector   = cl.user_session.get("collector")

    if query.strip().lower() in ["yes", "y", "correct", "looks good", "proceed", "confirm", "ok", "okay"]:
        cl.user_session.set("agent_state", "processing")
        await cl.Message(content="🚀 **Great! Starting analysis...**").send()
        await _run_agent_pipeline(profile, vectorstore, reranker, groq_client)
    else:
        response_text, updated_profile, is_done = collector.handle(query)
        cl.user_session.set("collector", collector)
        if updated_profile and is_done:
            cl.user_session.set("agent_profile", updated_profile)
            profile_summary = _format_profile_summary(updated_profile)
            actions = [cl.Action(name="follow_up", payload={"value": "Yes"}, label="✅ Yes, proceed")]
            await cl.Message(
                content=(
                    f"Updated! Here's your revised profile:\n\n{profile_summary}\n\n"
                    f"**Does this look correct now?**"
                ),
                actions=actions
            ).send()
        else:
            await cl.Message(content=response_text).send()

async def _handle_post_agent(query: str):
    profile          = cl.user_session.get("agent_profile")
    groq_client      = cl.user_session.get("groq_client")
    vectorstore      = cl.user_session.get("vectorstore")
    reranker         = cl.user_session.get("reranker")
    lf_client        = cl.user_session.get("lf_client")
    session_id       = cl.user_session.get("session_id")

    detected_plans = detect_specific_plans(query)
    if detected_plans and any(t in query.lower() for t in ["compare", "vs", "versus", "difference", "comparison"]):
        updated_profile = dict(profile)
        updated_profile["specific_plans"] = detected_plans
        updated_profile["mode"] = "comparison"
        cl.user_session.set("agent_profile", updated_profile)
        cl.user_session.set("agent_state", "processing")
        plan_names = [PLAN_NAME_MAP.get(p, p) for p in detected_plans]
        await cl.Message(
            content=f"📊 Got it! Comparing **{' vs '.join(plan_names)}** using your existing profile..."
        ).send()
        await _run_agent_pipeline(updated_profile, vectorstore, reranker, groq_client)
        return

    if any(t in query.lower() for t in AGENT_TRIGGERS):
        cl.user_session.set("agent_state", None)
        cl.user_session.set("active_mode",  "rag")
        await _handle_recommend(query)
        return

    query_lower = query.lower()
    if any(t in query_lower for t in IRDAI_TRIGGERS):
        msg = cl.Message(content="🔍 *Searching live IRDAI databases and combining with policy files...*")
        await msg.send()
        
        async_groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        try:
            final_answer = await handle_irdai_query(
                query=query, 
                groq_client=async_groq_client, 
                vectorstore=vectorstore, 
                session_id=session_id
            )
            suggestions = await _generate_dynamic_suggestions(query, final_answer)
            actions = [cl.Action(name="follow_up", payload={"value": sug}, label=sug) for sug in suggestions]
            
            msg.content = final_answer
            msg.actions = actions
            await msg.update()
            add_to_memory(query, final_answer)
        except Exception:
            msg.content = "I encountered an error fetching live data right now. Please try again in a moment."
            await msg.update()
        return

    llm = cl.user_session.get("llm")
    chain  = GUARDRAIL_PROMPT | llm
    
    try:
        result = await chain.ainvoke({"query": query})
        answer = result.content.strip().upper()
    except Exception:
        answer = "YES"
        
    try:
        _trace_guardrail(lf_client, query, answer, session_id)
    except Exception:
        pass

    if answer != "YES":
        await cl.Message(content=random.choice(GUARDRAIL_RESPONSES)).send()
        return

    memory_prefix = get_memory_context_prefix()
    await _run_rag(query, memory_prefix)

async def _handle_comparison_from_lobby(query: str, detected_plans: list):
    """
    Handles comparison requests that come in during lobby or RAG mode
    where no profile exists yet.
    Collects minimal profile (age + goal only) then runs comparison pipeline.
    Falls back to asking for age if not in query.
    """
    groq_client = cl.user_session.get("groq_client")
    vectorstore = cl.user_session.get("vectorstore")
    reranker    = cl.user_session.get("reranker")

    from insurance_agent.agent.profile_collector import extract_fields_from_text, parse_goal

    # Try to extract age from the query itself
    extracted = extract_fields_from_text(query)
    age  = extracted.get("age")
    goal = parse_goal(query) or extracted.get("goal", "savings")

    plan_names = [PLAN_NAME_MAP.get(p, p) for p in detected_plans]
    cards      = " vs ".join([f"**{n}**" for n in plan_names])

    if not age:
        # Store the pending comparison and ask for age only
        cl.user_session.set("pending_comparison_plans", detected_plans)
        cl.user_session.set("pending_comparison_goal",  goal)
        cl.user_session.set("agent_state",  "comparison_age")
        cl.user_session.set("active_mode",  "agent")
        await cl.Message(
            content=(
                f"📊 Great! I'll compare {cards} for you.\n\n"
                f"Just one quick question — **how old are you?** "
                f"This helps me tailor the comparison to your eligibility and premium range."
            )
        ).send()
        return

    # Have age — build minimal profile and run pipeline directly
    profile = {
        "age":                age,
        "annual_income":      extracted.get("annual_income", 500_000),
        "dependents":         extracted.get("dependents", 0),
        "monthly_spending":   extracted.get("monthly_spending", 20_000),
        "goal":               goal,
        "specific_plans":     detected_plans,
        "mode":               "comparison",
    }

    cl.user_session.set("agent_profile", profile)
    cl.user_session.set("agent_state",   "processing")
    cl.user_session.set("active_mode",   "agent")

    await cl.Message(
        content=f"📊 Comparing {cards} — tailored for age {age}..."
    ).send()
    await _run_agent_pipeline(profile, vectorstore, reranker, groq_client)

async def _handle_recommend(query: str):
    from insurance_agent.agent.profile_collector import parse_goal
    
    goal = parse_goal(query)
    
    # ── FORCE WIPE Profile Collector to prevent memory bleed ──
    collector = ProfileCollector()
    cl.user_session.set("collector", collector)
    cl.user_session.set("agent_profile", None)
    
    if goal:
        plans = _get_plans_for_goal(goal)
        if plans:
            plan_str = "\n".join([f"> 🛡️ **{p}**" for p in plans])
            
            goal_display = {
                "protection": "Pure Term / Life Protection",
                "pension": "Pension / Regular Income",
                "lump_sum": "Lump Sum / Savings",
                "savings": "Long-term Savings",
                "child_planning": "Child Education & Planning"
            }.get(goal, goal)

            msg = f"### 🔍 Plans for {goal_display}\n\nBased on your preference, here are the matching policies from my knowledge base:\n{plan_str}\n\n"
            msg += "*To tell you which of these is the exact best fit, I would need a few details like your age and income.*"
            
            actions = [
                cl.Action(name="start_recommendation", payload={"value": "Start Recommendation"}, label="💡 Get a Personalized Match"),
                cl.Action(name="follow_up", payload={"value": f"Compare {', '.join(plans)}"}, label="⚖️ Compare These Plans")
            ]
            
            cl.user_session.set("collector", collector)
            cl.user_session.set("agent_state", "lobby")
            cl.user_session.set("active_mode", "agent")
            
            await cl.Message(content=msg, actions=actions).send()
            return

    cl.user_session.set("agent_state", "lobby")
    cl.user_session.set("active_mode", "agent")
    
    lobby_msg = (
        "### 🏦 LIC Plan Recommendation\n"
        "I have detailed knowledge of the following 7 policies:\n"
        "1. LIC Tech Term\n"
        "2. LIC Jeevan Anand\n"
        "3. LIC New Endowment Plan\n"
        "4. LIC Jeevan Labh\n"
        "5. LIC Jeevan Umang\n"
        "6. LIC New Jeevan Amar\n"
        "7. LIC Children's Money Back Plan\n\n"
        "**How would you like to proceed?**\n"
        "• **Personalized Recommendation:** I will ask you a few details (age, income, goals) to find the perfect fit.\n"
        "• **Filter by Goal:** Ask me to filter them right now (e.g., *'Which ones are for pure term?'* or *'Which provide a lump sum?'*)."
    )
    actions = [
        cl.Action(name="start_recommendation", payload={"value": "Start Recommendation"}, label="💡 Start Personalized Recommendation"),
        cl.Action(name="follow_up", payload={"value": "Which ones are pure term?"}, label="🛡️ Show Pure Term Plans"),
        cl.Action(name="cancel_agent", payload={"value": "Cancel"}, label="❌ Cancel")
    ]
    
    await cl.Message(content=lobby_msg, actions=actions).send()