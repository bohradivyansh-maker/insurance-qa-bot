import os
import sys
import asyncio
import random
import chainlit as cl
from dotenv import load_dotenv
from cache import init_semantic_cache, get_cached_answer, store_cached_answer

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))

from retriever import get_retriever, retrieve_chunks
from reranker import load_reranker, rerank_chunks
from chain import load_llm, run_chain, format_context, expand_query
from langfuse_client import get_langfuse_handler
from embedder import load_embedder
from cache import init_semantic_cache
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from insurance_agent.agent.profile_collector import ProfileCollector, detect_specific_plans
from insurance_agent.agent.plan_shortlister import shortlist_plans
from insurance_agent.agent.plan_analyst import analyse_plan
from insurance_agent.agent.recommender import generate_recommendation

GROQ_MODEL = "llama-3.3-70b-versatile"

PLAN_NAME_MAP = {
    "LIC_Tech-Term.pdf":              "LIC Tech Term",
    "LIC_Jeevan_Anand.pdf":           "LIC Jeevan Anand",
    "LIC_Endowment.pdf":              "LIC New Endowment Plan",
    "LIC_Jeevan_Labh.pdf":            "LIC Jeevan Labh",
    "LIC_Jeevan_Umang.pdf":           "LIC Jeevan Umang",
    "LIC_New_Jeevan_Amar.pdf":        "LIC New Jeevan Amar",
    "LIC_ChidrensMoney_BackPlan.pdf": "LIC Children's Money Back Plan",
}

AGENT_TRIGGERS = [
    "best plan for me",
    "recommend a plan",
    "suggest a plan",
    "suggest the most suitable",
    "suggest the best",
    "which plan should i buy",
    "which plan should i take",
    "which plan should i choose",
    "help me choose a plan",
    "advise me on a plan",
    "what plan should i get",
    "suitable plan for me",
    "most suitable plan",
    "most suitable lic",
]

PLAN_TYPE_MAP = {
    "jeevan anand": "LIC's New Jeevan Anand Plan is a Par, Non-Linked, Life, Individual, Savings Plan.",
    "jeevan labh": "LIC's Jeevan Labh is a Par, Non-Linked, Life, Individual, Savings Plan.",
    "jeevan umang": "LIC's Jeevan Umang is a Par, Non-Linked, Individual, Savings, Whole Life Insurance Plan.",
    "endowment": "LIC's New Endowment Plan is a Par, Non-Linked, Life, Individual, Savings Plan.",
    "money back": "LIC's New Children's Money Back Plan is a Par, Non-Linked, Life, Individual, Savings Plan.",
    "jeevan amar": "LIC's New Jeevan Amar is a Non-Par, Non-Linked, Life, Individual, Pure Risk Plan.",
    "tech term": "LIC's New Tech-Term is a Non-Par, Non-Linked, Life, Individual, Pure Risk Plan.",
}

def check_plan_type_query(query: str) -> str | None:
    query_lower = query.lower()
    plan_type_triggers = [
        "type of plan", "what type", "kind of plan", "what kind of plan",
        "is it a", "is this a", "par or non", "linked or non", "plan type",
        "category of plan", "classified as", "what plan type",
    ]
    if not any(trigger in query_lower for trigger in plan_type_triggers):
        return None
    for keyword, answer in PLAN_TYPE_MAP.items():
        if keyword in query_lower:
            return answer
    return None

GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for an Insurance Policy Assistant.
Your only job is to classify if the user query is related to insurance or not.
Reply with exactly one word: YES or NO.
YES = query is about insurance, policies, premiums, claims, coverage.
YES = query is a greeting like hello, hi, thanks, bye, how are you.
YES = query asks for plan suggestion, recommendation, or comparison even without mentioning insurance explicitly.
NO = query is about anything else completely unrelated to insurance."""),
    ("human", "{query}")
])

GUARDRAIL_RESPONSES = [
    "🤔 Interesting question! Unfortunately my training data was exclusively insurance policies, not the entire internet. Try google.com — it's free!",
    "😄 I appreciate the curiosity! But I am strictly an insurance bot. You may have confused localhost:8000 with google.com.",
    "📋 My knowledge is 100% LIC policies and 0% everything else. Whoever built me was very focused.",
    "🏦 Great question for literally any other AI. I only speak fluent insurance.",
    "😅 I checked all 7 of my policy documents and none of them mention this. Shocking, I know.",
    "🔍 I searched my entire knowledge base — all 678 chunks of it — and found nothing. Try a search engine!",
    "📄 My creators gave me 7 insurance PDFs and told me to figure it out. This was not in the PDFs.",
    "💼 I am a very specialized bot. Insurance questions only please — my policy documents are watching.",
    "🤷 Outside my jurisdiction. I only handle claims, premiums, and existential questions about life cover.",
    "😇 I would love to help but my entire existence is 678 chunks of LIC policy documents. You deserve better."
]

GREETING_RESPONSES = {
    "hello": "👋 Hello! I am your Insurance Policy Assistant. Ask me anything about LIC policies or get a personalized plan recommendation!",
    "hi": "👋 Hi there! How can I help you with your insurance queries today?",
    "hey": "👋 Hey! Ask me anything about LIC policies or say 'suggest a plan for me' to get a personalized recommendation!",
    "thanks": "😊 No problem! Glad I could help. Feel free to ask anything else.",
    "thank you": "😊 Happy to help! Let me know if you have more questions.",
    "bye": "👋 Goodbye! Stay insured and stay protected!",
    "goodbye": "👋 Take care! Come back anytime you have insurance questions.",
    "how are you": "😊 I am doing great and ready to help! Ask about policies or say 'recommend a plan' to get started!",
}

DOCUMENT_QUERIES = [
    "what documents", "which documents", "what pdfs",
    "what are you indexed on", "what do you know about",
    "what policies do you have", "which policies"
]

def should_use_agent(query: str) -> bool:
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in AGENT_TRIGGERS)

async def check_insurance_guardrail(query: str, llm) -> bool:
    chain = GUARDRAIL_PROMPT | llm
    response = await chain.ainvoke({"query": query})
    result = response.content.strip().upper()
    return result == "YES"

def _format_profile_summary(profile: dict) -> str:
    goal_map = {
        "pure life protection": "protection",
        "life protection": "protection",
        "pure protection": "protection",
        "pure risk": "protection",
        "term insurance": "protection",
        "1": "pension", "pension": "pension", "regular pension": "pension",
        "retirement": "pension", "regular income": "pension",
        "2": "lump_sum", "lump sum": "lump_sum", "maturity": "lump_sum",
        "3": "protection", "term": "protection", "protect": "protection", "pure": "protection",
        "4": "savings", "long-term": "savings", "endowment": "savings", "long term": "savings",
        "5": "child_planning", "child": "child_planning", "children": "child_planning",
        "education": "child_planning", "kid": "child_planning",
    }
    goal_display  = goal_map.get(profile.get("goal", ""), profile.get("goal", ""))
    specific      = profile.get("specific_plans", [])
    mode          = profile.get("mode", "suggestion")
    specific_disp = ", ".join([PLAN_NAME_MAP.get(p, p) for p in specific]) if specific else "I'll suggest the best ones"
    mode_display  = "📊 Comparison" if mode == "comparison" else "💡 Suggestion"

    return (
        f"---\n"
        f"📊 **Your Profile (Confirmed):**\n"
        f"- 🎂 Age: {profile.get('age')}\n"
        f"- 💰 Annual Income: ₹{profile.get('annual_income'):,}\n"
        f"- 👨‍👩‍👧 Dependents: {profile.get('dependents')}\n"
        f"- 🏠 Monthly Spending: ₹{profile.get('monthly_spending'):,}\n"
        f"- 🎯 Goal: {goal_display}\n"
        f"- 📋 Plans: {specific_disp}\n"
        f"- 🔀 Mode: {mode_display}\n"
        f"---"
    )

async def _run_agent_pipeline(profile, vectorstore, reranker, groq_client):
    loop = asyncio.get_event_loop()
    mode           = profile.get("mode", "suggestion")
    specific_plans = profile.get("specific_plans", [])

    if mode == "suggestion":
        await cl.Message(
            content="🔍 **Suggestion Mode** — finding the best plans for your profile..."
        ).send()

        async with cl.Step(name="🔍 Step 1: Shortlisting relevant plans...") as step:
            shortlisted, reasoning = await loop.run_in_executor(
                None,
                lambda: shortlist_plans(profile, groq_client)
            )
            display_names = [PLAN_NAME_MAP.get(p, p) for p in shortlisted]
            step.output = f"Selected: {', '.join(display_names)}"

        await cl.Message(
            content=f"📋 **Plans shortlisted:** {', '.join(display_names)}\n\n*{reasoning}*"
        ).send()

    else:
        shortlisted   = specific_plans
        display_names = [PLAN_NAME_MAP.get(p, p) for p in shortlisted]
        reasoning     = "User specified these plans directly for comparison."

        await cl.Message(
            content=f"📊 **Comparison Mode** — analysing {' vs '.join(display_names)} for your profile..."
        ).send()

    plan_analyses = {}
    for plan_file in shortlisted:
        plan_name = PLAN_NAME_MAP.get(plan_file, plan_file)
        async with cl.Step(name=f"📄 Analysing {plan_name}...") as step:
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
            step.output = f"Done — {plan_name}"

    async with cl.Step(name="✅ Step 3: Generating recommendation...") as step:
        recommendation = await loop.run_in_executor(
            None,
            lambda: generate_recommendation(
                profile=profile,
                plan_analyses=plan_analyses,
                shortlist_reasoning=reasoning,
                groq_client=groq_client
            )
        )
        step.output = "Recommendation ready."

    await asyncio.sleep(0.3)
    await cl.Message(content=recommendation).send()
    cl.user_session.set("agent_state", "done")
    await cl.Message(
        content="---\n💬 Feel free to ask follow-up questions about these policies!"
    ).send()


@cl.on_chat_start
async def on_chat_start():
    loading = cl.Message(content="⚙️ Initializing Insurance Assistant...")
    await loading.send()

    vectorstore      = get_retriever()
    reranker         = load_reranker()
    llm              = load_llm()
    langfuse_handler = get_langfuse_handler()

    embedder = load_embedder()
    try:
        init_semantic_cache(embedder)
        print("✅ Redis semantic cache active")
    except Exception as e:
        print(f"⚠️ Cache unavailable, running without cache: {e}")

    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("reranker", reranker)
    cl.user_session.set("llm", llm)
    cl.user_session.set("langfuse_handler", langfuse_handler)
    cl.user_session.set("groq_client", groq_client)
    cl.user_session.set("active_mode", "rag")
    cl.user_session.set("collector", None)
    cl.user_session.set("agent_state", None)
    cl.user_session.set("agent_profile", None)

    await loading.remove()
    await cl.Message(
        content=(
            "👋 **Welcome to the LIC Insurance Assistant!**\n\n"
            "I can help you in two ways:\n\n"
            "🔍 **Ask anything** about LIC policies — benefits, premiums, eligibility, claims\n\n"
            "💡 **Get a personalized recommendation** — just say *'suggest a plan for me'* "
            "or *'compare Jeevan Anand and Jeevan Labh'*\n\n"
            "What would you like to know?"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query       = message.content
    query_lower = query.strip().lower()
    active_mode = cl.user_session.get("active_mode")
    agent_state = cl.user_session.get("agent_state")

    # ── Greetings ──────────────────────────────────────────────────────────
    if query_lower in GREETING_RESPONSES:
        await cl.Message(content=GREETING_RESPONSES[query_lower]).send()
        return

    # ── Document list query ────────────────────────────────────────────────
    if any(phrase in query_lower for phrase in DOCUMENT_QUERIES):
        await cl.Message(
            content=(
                "📚 I am trained on the following LIC policy documents:\n\n"
                "1. LIC Tech Term\n"
                "2. LIC Jeevan Anand\n"
                "3. LIC New Endowment Plan\n"
                "4. LIC Jeevan Labh\n"
                "5. LIC Jeevan Umang\n"
                "6. LIC New Jeevan Amar\n"
                "7. LIC Children's Money Back Plan\n\n"
                "Ask me anything about these policies!"
            )
        ).send()
        return

    # ── Plan type direct lookup ────────────────────────────────────────────
    plan_type_answer = check_plan_type_query(query)
    if plan_type_answer:
        await cl.Message(content=plan_type_answer).send()
        return

    # ── Mid agent profile collection ───────────────────────────────────────
    if active_mode == "agent" and agent_state == "collecting":
        collector   = cl.user_session.get("collector")
        groq_client = cl.user_session.get("groq_client")

        response_text, profile = collector.handle(query)
        cl.user_session.set("collector", collector)

        if profile is None:
            await cl.Message(content=response_text).send()
        else:
            cl.user_session.set("agent_profile", profile)
            cl.user_session.set("agent_state", "confirming")
            profile_summary = _format_profile_summary(profile)
            await cl.Message(
                content=(
                    f"{response_text}\n\n{profile_summary}\n\n"
                    f"✅ **Does this look correct?**\n"
                    f"Reply **YES** to proceed or tell me what to correct "
                    f"(e.g. *'my income is 10L not 8L'*)"
                )
            ).send()
        return

    # ── Confirmation state ─────────────────────────────────────────────────
    if active_mode == "agent" and agent_state == "confirming":
        groq_client = cl.user_session.get("groq_client")
        vectorstore = cl.user_session.get("vectorstore")
        reranker    = cl.user_session.get("reranker")
        profile     = cl.user_session.get("agent_profile")
        collector   = cl.user_session.get("collector")

        if query.strip().lower() in ["yes", "y", "correct", "looks good",
                                     "proceed", "confirm", "ok", "okay"]:
            cl.user_session.set("agent_state", "processing")
            await cl.Message(content="🚀 Great! Starting analysis...").send()
            await _run_agent_pipeline(profile, vectorstore, reranker, groq_client)
        else:
            response_text, updated_profile = collector.handle(query)
            cl.user_session.set("collector", collector)

            if updated_profile:
                cl.user_session.set("agent_profile", updated_profile)
                profile_summary = _format_profile_summary(updated_profile)
                await cl.Message(
                    content=(
                        f"Updated! Here's your revised profile:\n\n{profile_summary}\n\n"
                        f"✅ **Does this look correct now?** Reply YES to proceed."
                    )
                ).send()
            else:
                await cl.Message(content=response_text).send()
        return

    # ── Agent processing in progress ───────────────────────────────────────
    if active_mode == "agent" and agent_state == "processing":
        await cl.Message(
            content="⏳ Still analyzing your profile — please wait a moment..."
        ).send()
        return

    # ── Post agent follow-up ───────────────────────────────────────────────
    if active_mode == "agent" and agent_state == "done":
        profile          = cl.user_session.get("agent_profile")
        groq_client      = cl.user_session.get("groq_client")
        vectorstore      = cl.user_session.get("vectorstore")
        reranker         = cl.user_session.get("reranker")
        llm              = cl.user_session.get("llm")
        langfuse_handler = cl.user_session.get("langfuse_handler")

        detected_plans = detect_specific_plans(query)

        if detected_plans and any(t in query.lower() for t in
                                  ["compare", "vs", "versus", "difference", "comparison"]):
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

        if should_use_agent(query):
            collector = ProfileCollector()
            response_text, profile = collector.handle(query)
            cl.user_session.set("collector", collector)
            cl.user_session.set("active_mode", "agent")

            if profile is not None:
                cl.user_session.set("agent_profile", profile)
                cl.user_session.set("agent_state", "confirming")
                profile_summary = _format_profile_summary(profile)
                await cl.Message(
                    content=f"🔄 Starting a fresh recommendation!\n\n{response_text}\n\n{profile_summary}\n\n✅ **Does this look correct?** Reply YES to proceed."
                ).send()
            else:
                cl.user_session.set("agent_state", "collecting")
                await cl.Message(content=f"🔄 Starting a fresh recommendation!\n\n{response_text}").send()
            return

        # Factual follow-up after agent — route to RAG with query expansion
        # Guardrail first, then cache, then retrieval
        llm = cl.user_session.get("llm")
        is_insurance = await check_insurance_guardrail(query, llm)
        if not is_insurance:
            await cl.Message(content=random.choice(GUARDRAIL_RESPONSES)).send()
            return

        cached = get_cached_answer(query)
        if cached:
            await cl.Message(content=cached).send()
            return

        retrieval_query = expand_query(query)
        retrieved = retrieve_chunks(retrieval_query, vectorstore)
        reranked  = rerank_chunks(query, retrieved, reranker)

        msg = cl.Message(content="")
        await msg.send()

        if reranked is None:
            msg.content = "I cannot find specific information about this in the available policy documents."
            await msg.update()
            return

        context = format_context(reranked)
        from prompts import get_rag_prompt
        prompt = get_rag_prompt()
        chain  = prompt | llm

        full_response = ""
        async for chunk in chain.astream(
            {"context": context, "question": query},
            config={"callbacks": [langfuse_handler]}
        ):
            full_response += chunk.content
            await msg.stream_token(chunk.content)

        store_cached_answer(query, full_response)

        source_elements = []
        seen_sources = set()
        source_num = 1

        for doc, score in reranked:
            source_file = os.path.basename(
                doc.metadata.get('source', 'unknown')
            ).replace('.pdf', '').replace('_', ' ')
            page = doc.metadata.get('page', '?')
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

        msg.elements = source_elements
        await msg.update()
        return

    # ── Default RAG flow ───────────────────────────────────────────────────
    llm              = cl.user_session.get("llm")
    langfuse_handler = cl.user_session.get("langfuse_handler")
    vectorstore      = cl.user_session.get("vectorstore")
    reranker         = cl.user_session.get("reranker")

    is_insurance = await check_insurance_guardrail(query, llm)
    if not is_insurance:
        await cl.Message(content=random.choice(GUARDRAIL_RESPONSES)).send()
        return

    if should_use_agent(query):
        collector = ProfileCollector()
        response_text, profile = collector.handle(query)
        cl.user_session.set("collector", collector)
        cl.user_session.set("active_mode", "agent")

        if profile is not None:
            cl.user_session.set("agent_profile", profile)
            cl.user_session.set("agent_state", "confirming")
            profile_summary = _format_profile_summary(profile)
            await cl.Message(
                content=f"💡 **Recommendation Mode activated!**\n\n{response_text}\n\n{profile_summary}\n\n✅ **Does this look correct?** Reply YES to proceed."
            ).send()
        else:
            cl.user_session.set("agent_state", "collecting")
            await cl.Message(
                content=f"💡 **Recommendation Mode activated!**\n\n{response_text}"
            ).send()
        return

    # Default RAG — guardrail passed, agent not triggered → check cache then retrieve
    cached = get_cached_answer(query)
    if cached:
        await cl.Message(content=cached).send()
        return

    retrieval_query = expand_query(query)
    retrieved = retrieve_chunks(retrieval_query, vectorstore)
    reranked  = rerank_chunks(query, retrieved, reranker)

    msg = cl.Message(content="")
    await msg.send()

    if reranked is None:
        msg.content = "I cannot find specific information about this in the available policy documents."
        await msg.update()
        return

    context = format_context(reranked)
    from prompts import get_rag_prompt
    prompt = get_rag_prompt()
    chain  = prompt | llm

    full_response = ""
    async for chunk in chain.astream(
        {"context": context, "question": query},
        config={"callbacks": [langfuse_handler]}
    ):
        full_response += chunk.content
        await msg.stream_token(chunk.content)

    store_cached_answer(query, full_response)

    source_elements = []
    seen_sources = set()
    source_num = 1

    for doc, score in reranked:
        source_file = os.path.basename(
            doc.metadata.get('source', 'unknown')
        ).replace('.pdf', '').replace('_', ' ')
        page = doc.metadata.get('page', '?')
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

    msg.elements = source_elements
    await msg.update()
    return