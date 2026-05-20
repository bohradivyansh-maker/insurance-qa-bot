# insurance_agent/main.py

import os
import sys
import asyncio
import chainlit as cl
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.retriever import get_retriever
from app.reranker import load_reranker

from agent.profile_collector import ProfileCollector
from agent.plan_shortlister import shortlist_plans
from agent.plan_analyst import analyse_plan
from agent.recommender import generate_recommendation

STATE_COLLECTING = "collecting_profile"
STATE_PROCESSING = "processing"
STATE_DONE       = "done"

WELCOME_MESSAGE = """👋 **Welcome to the LIC Insurance Advisor Agent!**

I'm here to help you find the best LIC policy based on your personal profile.

I'll ask you **5 quick questions** — age, income, dependents, monthly expenses, and your goal.

You can also mention specific plans you want compared at any point —
e.g. *"Compare Jeevan Labh and Jeevan Anand for a 35 year old"*
and I'll switch to **Comparison Mode** automatically.

Let's go! **How old are you?**
"""

PLAN_NAME_MAP = {
    "LIC_Tech-Term.pdf":              "LIC Tech Term",
    "LIC_Jeevan_Anand.pdf":           "LIC Jeevan Anand",
    "LIC_Endowment.pdf":              "LIC New Endowment Plan",
    "LIC_Jeevan_Labh.pdf":            "LIC Jeevan Labh",
    "LIC_Jeevan_Umang.pdf":           "LIC Jeevan Umang",
    "LIC_New_Jeevan_Amar.pdf":        "LIC New Jeevan Amar",
    "LIC_ChidrensMoney_BackPlan.pdf": "LIC Children's Money Back Plan",
}


@cl.on_chat_start
async def on_chat_start():
    loading = cl.Message(content="⚙️ Initializing Insurance Advisor Agent...")
    await loading.send()

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("[Agent] Loading vectorstore...")
    vectorstore = get_retriever()

    print("[Agent] Loading reranker...")
    reranker = load_reranker()

    cl.user_session.set("groq_client", groq_client)
    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("reranker", reranker)
    cl.user_session.set("collector", ProfileCollector())
    cl.user_session.set("state", STATE_COLLECTING)
    cl.user_session.set("profile", None)

    await loading.remove()
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_input  = message.content.strip()
    state       = cl.user_session.get("state")
    groq_client = cl.user_session.get("groq_client")
    vectorstore = cl.user_session.get("vectorstore")
    reranker    = cl.user_session.get("reranker")
    collector   = cl.user_session.get("collector")

    # ── Collecting profile ─────────────────────────────────────────────────
    if state == STATE_COLLECTING:
        response_text, profile = collector.handle(user_input)
        cl.user_session.set("collector", collector)

        if profile is None:
            await cl.Message(content=response_text).send()

        else:
            cl.user_session.set("profile", profile)
            cl.user_session.set("state", STATE_PROCESSING)

            profile_summary = _format_profile_summary(profile)
            await cl.Message(content=f"{response_text}\n\n{profile_summary}").send()

            await _run_agent_pipeline(profile, vectorstore, reranker, groq_client)

    # ── Done — handle follow-up questions ─────────────────────────────────
    elif state == STATE_DONE:
        profile = cl.user_session.get("profile")

        followup_prompt = (
            f"The user received a LIC insurance recommendation based on their profile: "
            f"Age {profile['age']}, Income ₹{profile['annual_income']:,}, "
            f"Dependents {profile['dependents']}, Goal {profile['goal']}.\n\n"
            f"Follow-up question: {user_input}\n\n"
            f"Answer helpfully and concisely. If they want a fresh analysis, ask them to start a new chat."
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": followup_prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        await cl.Message(content=response.choices[0].message.content).send()

    elif state == STATE_PROCESSING:
        await cl.Message(content="⏳ Still analyzing — please wait a moment...").send()


async def _run_agent_pipeline(profile, vectorstore, reranker, groq_client):
    """
    Two-mode agentic pipeline:

    SUGGESTION MODE  (no specific plans named):
        Step 1 — Shortlist 2-3 plans via LLM reasoning  [1 Groq call]
        Step 2 — Analyse each shortlisted plan via RAG   [1 Groq call per plan]
        Step 3 — Generate final recommendation           [1 Groq call]

    COMPARISON MODE  (user named specific plans):
        Step 1 — SKIPPED (plans already known)           [0 Groq calls]
        Step 2 — Analyse each named plan via RAG         [1 Groq call per plan]
        Step 3 — Generate final recommendation           [1 Groq call]
    """

    mode           = profile.get("mode", "suggestion")
    specific_plans = profile.get("specific_plans", [])

    # ── SUGGESTION MODE — Step 1: Shortlist ───────────────────────────────
    if mode == "suggestion":
        await cl.Message(content="🔍 **Suggestion Mode** — finding the best plans for your profile...").send()

        async with cl.Step(name="🔍 Step 1: Shortlisting relevant plans...") as step:
            shortlisted, reasoning = shortlist_plans(profile, groq_client)
            display_names = [PLAN_NAME_MAP.get(p, p) for p in shortlisted]
            step.output = f"Selected: {', '.join(display_names)}"

        await cl.Message(
            content=f"📋 **Plans shortlisted:** {', '.join(display_names)}\n\n*{reasoning}*"
        ).send()

    # ── COMPARISON MODE — Step 1: Skipped ────────────────────────────────
    else:
        shortlisted   = specific_plans
        display_names = [PLAN_NAME_MAP.get(p, p) for p in shortlisted]
        reasoning     = "User specified these plans directly for comparison."

        await cl.Message(
            content=f"📊 **Comparison Mode** — analysing {' vs '.join(display_names)} for your profile..."
        ).send()

    # ── Step 2: Analyse each plan (both modes) ────────────────────────────
    plan_analyses = {}
    for plan_file in shortlisted:
        plan_name = PLAN_NAME_MAP.get(plan_file, plan_file)
        async with cl.Step(name=f"📄 Analysing {plan_name}...") as step:
            analysis = analyse_plan(
                plan_filename=plan_file,
                profile=profile,
                vectorstore=vectorstore,
                reranker=reranker,
                groq_client=groq_client
            )
            plan_analyses[plan_file] = analysis
            step.output = f"Done — {plan_name}"

    # ── Step 3: Generate recommendation (both modes) ──────────────────────
    async with cl.Step(name="✅ Step 3: Generating recommendation...") as step:
        recommendation = generate_recommendation(
            profile=profile,
            plan_analyses=plan_analyses,
            shortlist_reasoning=reasoning,
            groq_client=groq_client
        )
        step.output = "Recommendation ready."

    await asyncio.sleep(0.5)
    await cl.Message(content=recommendation).send()

    cl.user_session.set("state", STATE_DONE)
    await cl.Message(content="---\n💬 Feel free to ask any follow-up questions!").send()


def _format_profile_summary(profile: dict) -> str:
    goal_map = {
        "pension":        "Regular income / pension after retirement",
        "lump_sum":       "Lump sum maturity benefit",
        "protection":     "Pure life protection (term insurance)",
        "savings":        "Long-term savings with life cover",
        "child_planning": "Child education and future planning",
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