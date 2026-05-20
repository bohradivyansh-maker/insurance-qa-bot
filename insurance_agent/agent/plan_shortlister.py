import json
import re
from groq import Groq
from insurance_agent.agent.prompts import get_shortlist_system_prompt, SHORTLIST_USER_TEMPLATE, PLAN_DESCRIPTIONS
from insurance_agent.agent.langfuse_tracker import get_langfuse

langfuse = get_langfuse()

def shortlist_plans(profile: dict, groq_client: Groq) -> tuple[list[str], str]:
    
    if profile.get("specific_plans") and len(profile["specific_plans"]) > 0:
        specified = profile["specific_plans"]
        valid_plans = [p for p in specified if p in PLAN_DESCRIPTIONS]
        if valid_plans:
            return valid_plans, f"User specifically requested comparison of: {', '.join(valid_plans)}"

    user_message = SHORTLIST_USER_TEMPLATE.format(
        age=profile["age"],
        annual_income=profile["annual_income"],
        dependents=profile["dependents"],
        monthly_spending=profile["monthly_spending"],
        goal=profile["goal"]
    )

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="agent_shortlist",
        model="llama-3.3-70b-versatile",
        input=str(profile)
    ) as obs:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": get_shortlist_system_prompt(profile)},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        response_text = response.choices[0].message.content
        obs.update(output=response_text)

    shortlisted_plans, reasoning = _parse_shortlist_response(response_text)

    if not shortlisted_plans:
        print("Warning: Shortlist parsing failed, using fallback plans")
        shortlisted_plans = [
            "LIC_Jeevan_Anand.pdf",
            "LIC_Endowment.pdf",
            "LIC_Tech-Term.pdf"
        ]
        reasoning = "Could not parse LLM shortlist response, using default selection."

    return shortlisted_plans, reasoning


def _parse_shortlist_response(text: str) -> tuple[list[str], str]:
    if "SHORTLIST:" not in text:
        return [], ""
    try:
        json_start = text.index("SHORTLIST:") + len("SHORTLIST:")
        json_str = text[json_start:].strip()
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            data = json.loads(match.group())
            plans = data.get("shortlisted_plans", [])
            reasoning = data.get("elimination_reasoning", "")
            valid_plans = [p for p in plans if p in PLAN_DESCRIPTIONS]
            return valid_plans, reasoning
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Shortlist parse error: {e}")
    return [], ""