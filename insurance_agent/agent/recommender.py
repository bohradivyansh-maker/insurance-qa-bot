# insurance_agent/agent/recommender.py

from groq import Groq
from insurance_agent.agent.prompts import RECOMMENDATION_SYSTEM_PROMPT, RECOMMENDATION_USER_TEMPLATE
from insurance_agent.agent.langfuse_tracker import get_langfuse
langfuse = get_langfuse()

def generate_recommendation(
    profile: dict,
    plan_analyses: dict,
    shortlist_reasoning: str,
    groq_client: Groq
) -> str:
    """
    Takes all plan analyses + user profile and generates final recommendation.
    Returns complete response as a string.
    """
    combined_analyses = "\n\n---\n\n".join(plan_analyses.values())
    if shortlist_reasoning:
        combined_analyses = (
            f"**Why these plans were selected:**\n{shortlist_reasoning}\n\n---\n\n"
            + combined_analyses
        )

    user_message = RECOMMENDATION_USER_TEMPLATE.format(
        age=profile["age"],
        annual_income=profile["annual_income"],
        dependents=profile["dependents"],
        monthly_spending=profile["monthly_spending"],
        goal=profile["goal"],
        plan_analyses=combined_analyses
    )

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="agent_recommendation",
        model="llama-3.3-70b-versatile",
        input=str(profile)
    ) as obs:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        result = response.choices[0].message.content
        obs.update(output=result)

    return result

   