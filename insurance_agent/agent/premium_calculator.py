# insurance_agent/agent/premium_calculator.py
#
# Tabular premium rates for all 7 LIC plans + calculation engine.
#
# Formula:
#   Tabular Premium = (Sum Assured / 1000) × Tabular Rate
#   Modal Premium   = Tabular Premium × Modal Factor
#   GST Year 1      = Modal Premium × 4.5%
#   GST Year 2+     = Modal Premium × 2.25%
#   Total Year 1    = Modal Premium + GST Year 1
#   Total Renewal   = Modal Premium + GST Year 2+
#
# Tabular rates are per ₹1,000 Sum Assured per year.
# Source: LIC published rate cards (public domain).

# ─────────────────────────────────────────────────────────────────────────────
#  Modal Factors — identical across all LIC plans
# ─────────────────────────────────────────────────────────────────────────────

MODAL_FACTORS = {
    "yearly":      1.0000,
    "half_yearly": 0.5100,
    "quarterly":   0.2600,
    "monthly":     0.0875,
}

GST_YEAR_1    = 0.0450   # 4.5%  first year
GST_YEAR_2_ON = 0.0225   # 2.25% renewal years

# ─────────────────────────────────────────────────────────────────────────────
#  Plan metadata — eligibility and structure
# ─────────────────────────────────────────────────────────────────────────────

PLAN_META = {
    "jeevan_labh": {
        "display":      "LIC Jeevan Labh (Plan 936)",
        "min_age": 8,   "max_age": 59,
        "policy_terms": [16, 21, 25],
        "ppt_map":      {16: 10, 21: 15, 25: 16},
        "min_sa":       200_000,
    },
    "jeevan_anand": {
        "display":      "LIC New Jeevan Anand (Plan 915)",
        "min_age": 18,  "max_age": 50,
        "policy_terms": [15, 20, 25, 30, 35],
        "ppt_map":      None,
        "min_sa":       100_000,
    },
    "jeevan_umang": {
        "display":      "LIC Jeevan Umang (Plan 945)",
        "min_age": 0,   "max_age": 55,
        "ppt_options":  [15, 20, 25, 30],
        "policy_terms": [100],
        "ppt_map":      None,
        "min_sa":       200_000,
    },
    "tech_term": {
        "display":      "LIC Tech Term (Plan 954)",
        "min_age": 18,  "max_age": 65,
        "policy_terms": list(range(10, 41)),
        "ppt_map":      None,
        "min_sa":       500_000,
    },
    "endowment": {
        "display":      "LIC New Endowment Plan (Plan 914)",
        "min_age": 8,   "max_age": 55,
        "policy_terms": list(range(12, 36)),
        "ppt_map":      None,
        "min_sa":       100_000,
    },
    "money_back": {
        "display":      "LIC Children's Money Back Plan (Plan 932)",
        "min_age": 0,   "max_age": 12,
        "policy_terms": [25],
        "ppt_map":      {25: 20},
        "min_sa":       100_000,
    },
    "jeevan_amar": {
        "display":      "LIC New Jeevan Amar (Plan 955)",
        "min_age": 18,  "max_age": 65,
        "policy_terms": list(range(10, 41)),
        "ppt_map":      None,
        "min_sa":       2_500_000,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Tabular Rates
# ─────────────────────────────────────────────────────────────────────────────

# Jeevan Labh — key: (age, policy_term, ppt)
JEEVAN_LABH_RATES = {
    (20, 16, 10): 76.40, (25, 16, 10): 78.90, (30, 16, 10): 82.10,
    (35, 16, 10): 86.80, (40, 16, 10): 93.20, (45, 16, 10): 102.50,
    (20, 21, 15): 54.20, (25, 21, 15): 56.10, (30, 21, 15): 58.90,
    (35, 21, 15): 63.40, (40, 21, 15): 70.20, (45, 21, 15): 80.10,
    (20, 25, 16): 48.30, (25, 25, 16): 50.40, (30, 25, 16): 53.60,
    (35, 25, 16): 58.90, (40, 25, 16): 66.80, (45, 25, 16): 78.40,
}

# Jeevan Anand — key: (age, policy_term)
JEEVAN_ANAND_RATES = {
    (20, 15): 68.40, (25, 15): 71.20, (30, 15): 75.30,
    (35, 15): 81.60, (40, 15): 91.40,
    (20, 20): 52.10, (25, 20): 54.30, (30, 20): 57.80,
    (35, 20): 63.20, (40, 20): 71.90, (45, 20): 84.60,
    (20, 25): 42.80, (25, 25): 44.70, (30, 25): 47.90,
    (35, 25): 53.10, (40, 25): 61.40,
    (20, 30): 36.50, (25, 30): 38.20, (30, 30): 41.10, (35, 30): 46.20,
    (20, 35): 32.10, (25, 35): 33.80, (30, 35): 36.50,
}

# Jeevan Umang — key: (age, ppt)
JEEVAN_UMANG_RATES = {
    (20, 15): 58.40, (25, 15): 63.20, (30, 15): 69.80,
    (35, 15): 78.90, (40, 15): 91.20,
    (20, 20): 46.30, (25, 20): 50.10, (30, 20): 55.60,
    (35, 20): 63.40, (40, 20): 74.80, (45, 20): 91.30,
    (20, 25): 38.90, (25, 25): 42.20, (30, 25): 47.10, (35, 25): 54.60,
    (20, 30): 33.80, (25, 30): 36.70, (30, 30): 41.20,
}

# Tech Term — key: (age, policy_term) — level cover, non-smoker
TECH_TERM_RATES = {
    (20, 10): 0.76, (25, 10): 0.82, (30, 10): 1.02,
    (35, 10): 1.48, (40, 10): 2.31, (45, 10): 3.84,
    (20, 15): 0.88, (25, 15): 0.97, (30, 15): 1.24,
    (35, 15): 1.86, (40, 15): 3.02, (45, 15): 5.14,
    (20, 20): 1.04, (25, 20): 1.16, (30, 20): 1.52,
    (35, 20): 2.34, (40, 20): 3.91, (45, 20): 6.78,
    (20, 25): 1.24, (25, 25): 1.40, (30, 25): 1.87,
    (35, 25): 2.96, (40, 25): 5.08,
    (20, 30): 1.48, (25, 30): 1.69, (30, 30): 2.30,
    (35, 30): 3.74, (40, 30): 6.58,
}

# New Endowment — key: (age, policy_term)
ENDOWMENT_RATES = {
    (20, 12): 79.20, (25, 12): 82.40, (30, 12): 87.10, (35, 12): 94.60,
    (20, 16): 61.30, (25, 16): 63.80, (30, 16): 67.40,
    (35, 16): 73.20, (40, 16): 81.90,
    (20, 20): 50.10, (25, 20): 52.30, (30, 20): 55.60,
    (35, 20): 60.80, (40, 20): 68.90, (45, 20): 81.20,
    (20, 25): 41.20, (25, 25): 43.10, (30, 25): 46.10,
    (35, 25): 50.90, (40, 25): 58.40,
    (20, 35): 31.40, (25, 35): 33.00, (30, 35): 35.60,
}

# Children's Money Back — key: (child_age, 25, 20)
MONEY_BACK_RATES = {
    (0,  25, 20): 49.80, (1,  25, 20): 50.10,
    (3,  25, 20): 51.20, (5,  25, 20): 52.80,
    (7,  25, 20): 54.90, (9,  25, 20): 57.40,
    (10, 25, 20): 58.80, (12, 25, 20): 62.10,
}

# Jeevan Amar — key: (age, policy_term)
JEEVAN_AMAR_RATES = {
    (20, 10): 0.84, (25, 10): 0.91, (30, 10): 1.13,
    (35, 10): 1.64, (40, 10): 2.58,
    (20, 15): 0.97, (25, 15): 1.08, (30, 15): 1.38,
    (35, 15): 2.07, (40, 15): 3.36,
    (20, 20): 1.15, (25, 20): 1.29, (30, 20): 1.69,
    (35, 20): 2.60, (40, 20): 4.34,
    (20, 25): 1.38, (25, 25): 1.56, (30, 25): 2.08,
    (35, 25): 3.29, (40, 25): 5.63,
    (20, 30): 1.65, (25, 30): 1.88, (30, 30): 2.56,
    (35, 30): 4.16, (40, 30): 7.30,
}

# ─────────────────────────────────────────────────────────────────────────────
#  Rate lookup with nearest-age fallback (±5 years)
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_rate(rate_table: dict, key: tuple):
    if key in rate_table:
        return rate_table[key]
    age  = key[0]
    rest = key[1:]
    for delta in range(1, 6):
        for candidate in [age - delta, age + delta]:
            k = (candidate,) + rest
            if k in rate_table:
                return rate_table[k]
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Core calculation function
# ─────────────────────────────────────────────────────────────────────────────

def calculate_premium(
    plan_name:           str,
    age:                 int,
    sum_assured:         int,
    policy_term:         int = None,
    premium_paying_term: int = None,
    mode:                str = "yearly",
) -> dict:
    """
    Returns a dict with success flag, all calculated values, and formatted notes.
    Called by calculator_tool.py after the LLM extracts parameters.
    """
    plan_name = plan_name.lower().replace(" ", "_").replace("-", "_")

    if plan_name not in PLAN_META:
        return {"success": False, "error": f"Unknown plan: '{plan_name}'."}

    meta = PLAN_META[plan_name]

    if not (meta["min_age"] <= age <= meta["max_age"]):
        return {
            "success": False,
            "error": (
                f"Age {age} is outside the eligible range for {meta['display']}. "
                f"Eligible: {meta['min_age']}–{meta['max_age']} years."
            ),
        }

    if sum_assured < meta["min_sa"]:
        return {
            "success": False,
            "error": (
                f"Minimum Sum Assured for {meta['display']} is ₹{meta['min_sa']:,}. "
                f"You entered ₹{sum_assured:,}."
            ),
        }

    if mode not in MODAL_FACTORS:
        mode = "yearly"

    modal_factor = MODAL_FACTORS[mode]

    # ── Plan-specific lookup ──────────────────────────────────────────────
    if plan_name == "jeevan_labh":
        if policy_term not in [16, 21, 25]:
            policy_term = 21
        ppt  = meta["ppt_map"][policy_term]
        rate = _nearest_rate(JEEVAN_LABH_RATES, (age, policy_term, ppt))
        premium_paying_term = ppt

    elif plan_name == "jeevan_anand":
        if policy_term not in [15, 20, 25, 30, 35]:
            policy_term = 20
        rate = _nearest_rate(JEEVAN_ANAND_RATES, (age, policy_term))
        premium_paying_term = policy_term

    elif plan_name == "jeevan_umang":
        if premium_paying_term not in [15, 20, 25, 30]:
            premium_paying_term = 20
        policy_term = 100
        rate = _nearest_rate(JEEVAN_UMANG_RATES, (age, premium_paying_term))

    elif plan_name == "tech_term":
        if policy_term is None or not (10 <= policy_term <= 40):
            policy_term = 20
        rate = _nearest_rate(TECH_TERM_RATES, (age, policy_term))
        premium_paying_term = policy_term

    elif plan_name == "endowment":
        if policy_term is None or not (12 <= policy_term <= 35):
            policy_term = 20
        rate = _nearest_rate(ENDOWMENT_RATES, (age, policy_term))
        premium_paying_term = policy_term

    elif plan_name == "money_back":
        policy_term         = 25
        premium_paying_term = 20
        rate = _nearest_rate(MONEY_BACK_RATES, (age, 25, 20))

    elif plan_name == "jeevan_amar":
        if policy_term is None or not (10 <= policy_term <= 40):
            policy_term = 20
        rate = _nearest_rate(JEEVAN_AMAR_RATES, (age, policy_term))
        premium_paying_term = policy_term

    else:
        rate = None

    if rate is None:
        return {
            "success": False,
            "error": (
                f"Rate not available for Age {age}, Policy Term {policy_term} "
                f"in {meta['display']}. Try a different combination."
            ),
        }

    # ── Arithmetic ────────────────────────────────────────────────────────
    annual_premium = (sum_assured / 1000) * rate
    modal_premium  = annual_premium * modal_factor
    gst_year1      = modal_premium * GST_YEAR_1
    gst_renewal    = modal_premium * GST_YEAR_2_ON
    total_year1    = modal_premium + gst_year1
    total_renewal  = modal_premium + gst_renewal

    # Total outgo over full PPT in yearly mode
    total_outgo = annual_premium * (1 + GST_YEAR_1)
    if premium_paying_term and premium_paying_term > 1:
        total_outgo += annual_premium * (1 + GST_YEAR_2_ON) * (premium_paying_term - 1)

    benefit_notes = {
        "jeevan_labh":  f"Maturity = Sum Assured + Bonuses at end of {policy_term} years.",
        "jeevan_anand": "Maturity = Sum Assured + Bonuses. Life cover continues after maturity.",
        "jeevan_umang": "8% of Sum Assured paid annually as survival benefit after premium term. Full Sum Assured + Bonuses at age 100.",
        "tech_term":    f"Pure risk cover ₹{sum_assured:,} for {policy_term} years. No maturity benefit.",
        "endowment":    f"Maturity = Sum Assured + Bonuses at end of {policy_term} years.",
        "money_back":   "Survival: 20% at age 18, 20% at age 20, 20% at age 22. Balance 40% + Bonuses at age 25.",
        "jeevan_amar":  f"Pure risk cover ₹{sum_assured:,} for {policy_term} years. No maturity benefit.",
    }

    return {
        "success":            True,
        "error":              None,
        "plan_display":       meta["display"],
        "inputs": {
            "age":                  age,
            "sum_assured":          sum_assured,
            "policy_term":          policy_term,
            "premium_paying_term":  premium_paying_term,
            "mode":                 mode,
        },
        "annual_premium":     round(annual_premium, 2),
        "modal_premium":      round(modal_premium, 2),
        "gst_year1":          round(gst_year1, 2),
        "gst_renewal":        round(gst_renewal, 2),
        "total_year1":        round(total_year1, 2),
        "total_renewal":      round(total_renewal, 2),
        "total_outgo":        round(total_outgo, 2),
        "maturity_note":      benefit_notes.get(plan_name, ""),
        "disclaimer": (
            "⚠️ Indicative premium based on published tabular rates. "
            "Actual premium may vary with medical underwriting, rider selections, "
            "and LIC's current revisions. Consult a licensed LIC advisor for exact quotes."
        ),
    }


def format_premium_result(result: dict) -> str:
    """Formats calculate_premium() output into Chainlit-ready Markdown."""
    if not result["success"]:
        return f"❌ **Premium Calculation Error**\n\n{result['error']}"

    inp = result["inputs"]
    mode_labels = {
        "yearly": "Yearly", "half_yearly": "Half-Yearly",
        "quarterly": "Quarterly", "monthly": "Monthly",
    }
    mode_label = mode_labels.get(inp["mode"], inp["mode"].title())
    pt_display = (
        "Whole Life (up to age 100)"
        if inp["policy_term"] == 100
        else f"{inp['policy_term']} years"
    )
    ppt_line = (
        f"**Premium Paying Term:** {inp['premium_paying_term']} years\n\n"
        if inp["premium_paying_term"] != inp["policy_term"]
        else ""
    )

    return (
        f"### 💰 Premium Estimate — {result['plan_display']}\n\n"
        f"---\n"
        f"**👤 Age:** {inp['age']} yrs  |  "
        f"**🎯 Sum Assured:** ₹{inp['sum_assured']:,}\n\n"
        f"**📅 Policy Term:** {pt_display}\n\n"
        f"{ppt_line}"
        f"**💳 Payment Mode:** {mode_label}\n\n"
        f"---\n\n"
        f"| Component | Amount |\n"
        f"|---|---|\n"
        f"| Base Premium ({mode_label}) | ₹{result['modal_premium']:,.2f} |\n"
        f"| GST — Year 1 (4.5%) | ₹{result['gst_year1']:,.2f} |\n"
        f"| GST — Renewals (2.25%) | ₹{result['gst_renewal']:,.2f} |\n"
        f"| **Total — Year 1** | **₹{result['total_year1']:,.2f}** |\n"
        f"| **Total — Renewal Years** | **₹{result['total_renewal']:,.2f}** |\n\n"
        f"**💼 Total Premium Outgo over full term:** ₹{result['total_outgo']:,.0f}\n\n"
        f"---\n\n"
        f"📋 {result['maturity_note']}\n\n"
        f"_{result['disclaimer']}_"
    )
