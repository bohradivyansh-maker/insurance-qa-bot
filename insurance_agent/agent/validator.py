# insurance_agent/agent/validator.py
# Profile validation with humorous responses for edge cases.

# ── Validation Bounds ─────────────────────────────────────────────────────────
MIN_AGE = 18
MAX_AGE = 75
MIN_INCOME = 150000          # ₹1.5 lakh
MAX_INCOME = 100000000       # ₹10 crore
MAX_DEPENDENTS = 10
MAX_SPENDING_RATIO = 0.95    # monthly spending can't exceed 95% of monthly income
VALID_GOALS = ["pension", "lump_sum", "protection", "savings", "child_planning"]

# ── Humorous Responses ────────────────────────────────────────────────────────

AGE_TOO_LOW = (
    "Slow down! You need to be at least 18 to buy a LIC policy. "
    "Come back when you're done with school! 📚"
)

AGE_TOO_HIGH = (
    "Wait — {age} years old?! Are you the secret resident of the Himalayas who drinks glacial water "
    "and does yoga at 4am? Jokes aside, LIC policies are available for ages 18 to 75. "
    "Could you double-check your age? 🏔️"
)

AGE_NEGATIVE = (
    "Impressive — you haven't been born yet but you're already planning finances. "
    "Unfortunately LIC requires you to actually exist first. "
    "Could you share your real age? 👶"
)

INCOME_TOO_HIGH = (
    "₹{income} per year?! At that income you could personally sponsor the GDP of a small nation. "
    "We're here for retail insurance, not sovereign wealth funds. "
    "Could you recheck your annual income? 💰"
)

INCOME_TOO_LOW = (
    "An annual income of ₹{income} seems a bit low — that's less than ₹1 lakh a year. "
    "Could you double-check? Even an approximate number works fine. 🙂"
)

INCOME_NEGATIVE = (
    "A negative income? That's called debt, and we feel you. "
    "But LIC still needs a positive income to work with. "
    "What's your approximate annual income? 😅"
)

DEPENDENTS_TOO_HIGH = (
    "{dependents} dependents?! Were you taking notes from Genghis Khan? "
    "That's a lot of people counting on you! "
    "We support up to 10 dependents — could you recheck that number? 👨‍👩‍👧‍👦"
)

DEPENDENTS_NEGATIVE = (
    "Negative dependents — so people are actually depending *less* than zero on you? "
    "That's a new concept! Could you share the actual number? (0 is perfectly fine!) 😄"
)

SPENDING_EXCEEDS_INCOME = (
    "Your monthly spending of ₹{spending} is higher than your monthly income of ₹{monthly_income}. "
    "Sounds like someone has a credit card they're *very* close friends with. 💳 "
    "Could you recheck your monthly expenses?"
)

SPENDING_NEGATIVE = (
    "Negative monthly spending? You're making money by spending — teach us your ways! "
    "Could you share your actual monthly expenses? 🙃"
)

SPENDING_ZERO = (
    "₹0 monthly spending? Either you've achieved true enlightenment "
    "or you forgot to fill this in. Could you share your approximate monthly expenses? 🧘"
)

INVALID_GOAL = (
    "I didn't quite catch your financial goal. Could you clarify what you're looking for? "
    "Options are: regular pension income, lump sum savings, pure life protection, "
    "long-term savings, or child education planning. 🎯"
)

# ── Validator Function ────────────────────────────────────────────────────────

def validate_profile(profile: dict) -> tuple[bool, str]:
    """
    Validates extracted user profile.
    
    Returns:
        (True, "") if all fields are valid
        (False, humorous_error_message) if any field is invalid
    """

    age = profile.get("age")
    annual_income = profile.get("annual_income")
    dependents = profile.get("dependents")
    monthly_spending = profile.get("monthly_spending")
    goal = profile.get("goal")

    # ── Age ───────────────────────────────────────────────────────────────
    if age is None:
        return False, "I didn't catch your age — could you share it? 🙂"
    
    try:
        age = int(age)
    except (ValueError, TypeError):
        return False, "That doesn't look like a valid age. Could you share a number? 🙂"

    if age < 0:
        return False, AGE_NEGATIVE
    if age < MIN_AGE:
        return False, AGE_TOO_LOW
    if age > MAX_AGE:
        return False, AGE_TOO_HIGH.format(age=age)

    # ── Annual Income ─────────────────────────────────────────────────────
    if annual_income is None:
        return False, "I didn't catch your annual income — could you share an approximate figure? 🙂"

    try:
        annual_income = float(annual_income)
    except (ValueError, TypeError):
        return False, "That doesn't look like a valid income. Could you share a number? 🙂"

    if annual_income < 0:
        return False, INCOME_NEGATIVE
    if annual_income < MIN_INCOME:
        return False, INCOME_TOO_LOW.format(income=int(annual_income))
    if annual_income > MAX_INCOME:
        return False, INCOME_TOO_HIGH.format(income=f"{int(annual_income):,}")

    # ── Dependents ────────────────────────────────────────────────────────
    if dependents is None:
        return False, "I didn't catch the number of dependents — could you share that? (0 is fine!) 🙂"

    try:
        dependents = int(dependents)
    except (ValueError, TypeError):
        return False, "That doesn't look like a valid number of dependents. Could you share a number? 🙂"

    if dependents < 0:
        return False, DEPENDENTS_NEGATIVE
    if dependents > MAX_DEPENDENTS:
        return False, DEPENDENTS_TOO_HIGH.format(dependents=dependents)

    # ── Monthly Spending ──────────────────────────────────────────────────
    if monthly_spending is None:
        return False, "I didn't catch your monthly spending — could you share an approximate figure? 🙂"

    try:
        monthly_spending = float(monthly_spending)
    except (ValueError, TypeError):
        return False, "That doesn't look like a valid spending amount. Could you share a number? 🙂"

    if monthly_spending < 0:
        return False, SPENDING_NEGATIVE
    if monthly_spending == 0:
        return False, SPENDING_ZERO

    monthly_income = annual_income / 12
    if monthly_spending > monthly_income * MAX_SPENDING_RATIO:
        return False, SPENDING_EXCEEDS_INCOME.format(
            spending=f"{int(monthly_spending):,}",
            monthly_income=f"{int(monthly_income):,}"
        )

    # ── Goal ──────────────────────────────────────────────────────────────
    if goal not in VALID_GOALS:
        return False, INVALID_GOAL

    # ── All Good ──────────────────────────────────────────────────────────
    return True, ""