# insurance_agent/agent/calculator_tool.py
#
# MCP-style premium calculator tool.
#
# What makes this MCP-style (not just a hardcoded function call):
#   1. The tool is defined as a self-describing JSON schema (PREMIUM_TOOL_SCHEMA)
#   2. The schema is passed to Groq's tool-calling API at runtime
#   3. The LLM reads the schema, decides autonomously when to call the tool,
#      and extracts all parameters from natural language — no regex, no keyword matching
#   4. Your Python only defines the schema and executes the math when called
#   5. Adding a new tool (e.g. rider calculator) = add a new schema entry,
#      no changes to routing code
#
# Groq supports OpenAI-compatible tool calling.
# API: https://console.groq.com/docs/tool-use

import json
import asyncio
from groq import Groq
from insurance_agent.agent.premium_calculator import calculate_premium, format_premium_result

# ─────────────────────────────────────────────────────────────────────────────
#  Tool Schema — the MCP contract
#  The LLM reads this to understand what the tool does and what it needs.
#  This is the key difference from a hardcoded function call.
# ─────────────────────────────────────────────────────────────────────────────

PREMIUM_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate_premium",
        "description": (
            "Calculates the indicative LIC insurance premium for a given plan. "
            "Use this whenever the user asks about premium amount, how much they would pay, "
            "monthly/yearly cost, affordability, or premium for any LIC plan. "
            "Extract all available parameters from the conversation. "
            "Use sensible defaults for missing optional parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_name": {
                    "type": "string",
                    "enum": [
                        "jeevan_labh", "jeevan_anand", "jeevan_umang",
                        "tech_term", "endowment", "money_back", "jeevan_amar"
                    ],
                    "description": "The LIC plan to calculate premium for.",
                },
                "age": {
                    "type": "integer",
                    "description": "Age of the policyholder in years.",
                },
                "sum_assured": {
                    "type": "integer",
                    "description": (
                        "Sum assured (coverage amount) in rupees. "
                        "Convert lakhs to rupees: 10 lakhs = 1000000. "
                        "If not specified, use 1000000 (10 lakhs) as default."
                    ),
                },
                "policy_term": {
                    "type": "integer",
                    "description": (
                        "Policy term in years. "
                        "For Jeevan Labh: 16, 21, or 25. "
                        "For Jeevan Anand: 15, 20, 25, 30, or 35. "
                        "For term plans: 10 to 40. "
                        "If not specified, use 20 as default."
                    ),
                },
                "premium_paying_term": {
                    "type": "integer",
                    "description": (
                        "Premium paying term in years. "
                        "For Jeevan Umang: 15, 20, 25, or 30. "
                        "For limited pay plans this differs from policy term. "
                        "Leave null for regular pay plans where PPT equals policy term."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["yearly", "half_yearly", "quarterly", "monthly"],
                    "description": (
                        "Premium payment frequency. "
                        "Default is 'yearly' unless user specifies monthly/quarterly/half-yearly."
                    ),
                },
            },
            "required": ["plan_name", "age", "sum_assured"],
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  System prompt for the tool-calling LLM call
#  Separate from the main RAG/recommendation prompts.
# ─────────────────────────────────────────────────────────────────────────────

CALCULATOR_SYSTEM_PROMPT = """You are a helpful LIC insurance premium calculator assistant.
When the user asks about premium amounts, costs, or affordability for any LIC plan,
use the calculate_premium tool ONLY if you have all required parameters.

Required parameters before calling the tool:
- plan_name: which LIC plan (must be specified)
- age: policyholder age in years (must be provided by user)
- sum_assured: coverage amount in rupees (must be provided by user)

If ANY of these three are missing, do NOT call the tool.
Instead respond with a short friendly message asking only for the missing ones.

Examples:
- "calculate premium for jeevan anand" → age and sum assured missing → ask: "To calculate your Jeevan Anand premium, I need your age and desired sum assured (e.g. 10 lakhs, 50 lakhs)."
- "premium for jeevan labh, I am 28" → sum assured missing → ask: "Got it — age 28, Jeevan Labh. What sum assured would you like? (e.g. 10 lakhs, 25 lakhs)"
- "monthly premium for 50 lakhs Jeevan Labh as a 30 year old" → all present → call tool

Convert Indian number formats: 10L = 1000000, 50 lakhs = 5000000, 1 crore = 10000000.
Policy term and mode are optional — use 20 years and yearly as defaults if not specified."""


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point — called from main.py on CALCULATE route
# ─────────────────────────────────────────────────────────────────────────────

async def run_calculator_tool(
    query:       str,
    groq_client: Groq,
    context:     str = "",
) -> str:
    """
    MCP-style tool-calling flow:
      1. Send user query + tool schema to Groq
      2. Groq LLM reads schema, extracts parameters, returns tool_call
      3. We execute calculate_premium() with those parameters
      4. We send the result back to Groq for natural language synthesis
      5. Return the final human-readable response

    Args:
        query:       User's natural language question about premium
        groq_client: Sync Groq client from session
        context:     Optional conversation history prefix for parameter extraction

    Returns:
        Formatted premium result string ready for Chainlit display
    """
    loop = asyncio.get_event_loop()

    # Build messages — include context if available for better parameter extraction
    messages = [
        {"role": "system", "content": CALCULATOR_SYSTEM_PROMPT},
    ]
    if context:
        messages.append({
            "role": "system",
            "content": f"[Conversation context for parameter extraction]\n{context}"
        })
    messages.append({"role": "user", "content": query})

    # ── Step 1: Send to Groq with tool schema ────────────────────────────
    # This is the MCP-style call — the LLM decides what to extract
    def _first_call():
        return groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=[PREMIUM_TOOL_SCHEMA],
            tool_choice="auto",      # LLM decides when to call the tool
            temperature=0,
            max_tokens=500,
        )

    try:
        response = await loop.run_in_executor(None, _first_call)
    except Exception as e:
        return f"❌ Premium calculator unavailable: {str(e)}"

    response_message = response.choices[0].message

    # ── Step 2: Check if LLM decided to call the tool ────────────────────
    if not response_message.tool_calls:
        # LLM correctly held back because parameters are missing.
        # Return its clarifying question directly — this is the intended flow.
        return response_message.content or (
            "To calculate your premium I need a few details:\n\n"
            "- **Which plan?** (e.g. Jeevan Anand, Tech Term, Jeevan Labh)\n"
            "- **Your age?**\n"
            "- **Sum assured?** (e.g. 10 lakhs, 25 lakhs, 50 lakhs)\n\n"
            "Optionally you can also tell me the payment mode (monthly/yearly) and policy term."
        )
    tool_call = response_message.tool_calls[0]

    # ── Step 3: Parse parameters the LLM extracted ───────────────────────
    try:
        params = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return "❌ Could not parse calculation parameters. Please try rephrasing."

    # ── Step 4: Execute the actual calculation ───────────────────────────
    # This is pure Python math — no LLM involved here
    calc_result = calculate_premium(
        plan_name=           params.get("plan_name", ""),
        age=                 params.get("age", 30),
        sum_assured=         params.get("sum_assured", 1_000_000),
        policy_term=         params.get("policy_term"),
        premium_paying_term= params.get("premium_paying_term"),
        mode=                params.get("mode", "yearly"),
    )

    # ── Step 5: Format the result directly ──────────────────────────────
    # We use our own formatter for consistent Markdown output
    # rather than sending back to LLM (avoids hallucination of numbers)
    formatted = format_premium_result(calc_result)

    # ── Step 6: If calculation succeeded, add LLM commentary ─────────────
    # Send the raw numbers back to Groq for a brief natural language summary
    # that sits above the formatted table — makes it feel conversational
    if calc_result["success"]:
        try:
            summary_messages = messages + [
                # Simulate the tool call result
                {
                    "role":       "assistant",
                    "content":    None,
                    "tool_calls": [
                        {
                            "id":       tool_call.id,
                            "type":     "function",
                            "function": {
                                "name":      "calculate_premium",
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      json.dumps(calc_result),
                },
                {
                    "role":    "user",
                    "content": (
                        "Write exactly 2 sentences summarizing the key premium figure "
                        "and one noteworthy fact about this plan. "
                        "Do NOT repeat the full table — just a conversational opener. "
                        "Be warm and helpful, not robotic."
                    ),
                },
            ]

            def _summary_call():
                return groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=summary_messages,
                    temperature=0.2,
                    max_tokens=120,
                )

            summary_response = await loop.run_in_executor(None, _summary_call)
            summary_text     = summary_response.choices[0].message.content.strip()

            # Prepend summary above the formatted table
            return f"{summary_text}\n\n{formatted}"

        except Exception:
            # Summary failed — just return the formatted table, that's fine
            return formatted

    return formatted
