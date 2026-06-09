# insurance_agent/agent/profile_collector.py
#
# Smart profile collector — extracts whatever user provides upfront,
# confirms extracted fields, asks only for missing ones.
# Zero LLM calls during collection.

import re

FIELDS = ["age", "annual_income", "dependents", "monthly_spending", "goal"]

QUESTIONS = {
    "age":             "How old are you? (Please enter your age in years)",
    "annual_income":   "What is your approximate annual income? (e.g. 5 lakhs, 12L, 8,00,000)",
    "dependents":      "How many dependents do you have? (0 is fine!)",
    "monthly_spending":"What are your approximate monthly expenses? (e.g. 20k, 30,000, 1.5L)",
    "goal":            (
        "What is your primary financial goal? Please pick one:\n"
        "1. Regular pension / income after retirement\n"
        "2. Lump sum savings at maturity\n"
        "3. Pure life protection (term insurance)\n"
        "4. Long-term savings with life cover\n"
        "5. Child education and future planning"
    ),
}

ERRORS = {
    "age_negative":        "A negative age? So you haven't been born yet but you're already planning finances. Impressive. What's your actual age? 👶",
    "age_too_low":         "You need to be at least 18 to buy a LIC policy. Come back when you're done with school! 📚",
    "age_too_high":        "{} years old?! LIC policies are for ages 18–75. What's your actual age? 🏔️",
    "age_invalid":         "Could you type your age as a number? (e.g. 28)",
    "income_negative":     "A negative income? That's called debt. LIC needs a positive income. What's your approximate annual income? 😅",
    "income_too_low":      "₹{} per year seems very low. Even an approximate number works — what's your rough annual income?",
    "income_too_high":     "₹{} per year?! At that income you could sponsor a small nation's GDP. Could you recheck? 💰",
    "income_invalid":      "Try something like: 5 lakhs, 8L, 6,00,000, 25k",
    "dependents_negative": "Negative dependents — new concept! Just enter 0 if you have none. 😄",
    "dependents_too_high": "{} dependents?! We support up to 10. Could you recheck? 👨‍👩‍👧‍👦",
    "dependents_invalid":  "Please enter the number of dependents as a digit (e.g. 0, 2, 4)",
    "spending_negative":   "Negative spending — you make money by spending? What are your actual monthly expenses? 🙃",
    "spending_zero":       "₹0 monthly spending? What are your approximate monthly expenses? 🧘",
    "spending_exceeds":    "Monthly spending of ₹{} is more than your monthly income of ₹{}. Could you recheck? 💳",
    "spending_invalid":    "Try something like: 20k, 30,000, 1.5L",
    "goal_invalid":        "Please pick a number 1–5 or type one of: pension, lump sum, protection, savings, child planning",
}

MIN_AGE    = 18
MAX_AGE    = 75
MIN_INCOME = 100_000
MAX_INCOME = 100_000_000
MAX_DEP    = 10

def parse_number(text: str):
    text = str(text).strip().lower()
    text = text.replace("₹", "").replace(",", "").strip()
    match = re.search(r'(-?\d+(?:\.\d+)?)\s*(crore|cr|lakh|l|k)?', text)
    if not match:
        return None
    num = float(match.group(1))
    multiplier = match.group(2) or ""
    if multiplier in ("crore", "cr"):
        num *= 10_000_000
    elif multiplier in ("lakh", "l"):
        num *= 100_000
    elif multiplier == "k":
        num *= 1_000
    return num

GOAL_MAP = {
    "1": "pension",        "pension": "pension",       "regular": "pension",
    "retirement": "pension","income after": "pension",
    "2": "lump_sum",       "lump": "lump_sum",         "lump sum": "lump_sum",
    "maturity": "lump_sum","savings at": "lump_sum",
    "3": "protection",     "term": "protection",       "pure life": "protection",
    "protect": "protection","pure": "protection",      "life protection": "protection",
    "4": "savings",        "long-term": "savings",     "endowment": "savings",
    "long term": "savings","savings with": "savings",
    "5": "child_planning", "child": "child_planning",  "children": "child_planning",
    "education": "child_planning", "kid": "child_planning",
}

def parse_goal(text: str):
    text = text.strip().lower()
    # Single digits only match if the ENTIRE input is that digit
    # prevents "22" matching "2" → lump_sum
    if text in ("1", "2", "3", "4", "5"):
        return GOAL_MAP[text]
    # All other keys — substring match, longest first, skip single digits
    ordered_keys = [k for k in sorted(GOAL_MAP.keys(), key=len, reverse=True) if len(k) > 1]
    for key in ordered_keys:
        if key in text:
            return GOAL_MAP[key]
    return None

PLAN_KEYWORDS = {
    "jeevan labh":  "LIC_Jeevan_Labh.pdf",
    "jeevan anand": "LIC_Jeevan_Anand.pdf",
    "jeevan umang": "LIC_Jeevan_Umang.pdf",
    "tech term":    "LIC_Tech-Term.pdf",
    "endowment":    "LIC_Endowment.pdf",
    "money back":   "LIC_ChidrensMoney_BackPlan.pdf",
    "jeevan amar":  "LIC_New_Jeevan_Amar.pdf",
}

def detect_specific_plans(text: str) -> list:
    text = text.lower()
    return [fname for kw, fname in PLAN_KEYWORDS.items() if kw in text]

def detect_mode(specific_plans: list) -> str:
    return "comparison" if specific_plans else "suggestion"

def extract_fields_from_text(text: str) -> dict:
    extracted = {}
    text_lower = text.lower()

    age_patterns = [
        r'\bage\s*(?:is\s*|of\s*)?(\d{1,3})',
        r'(\d{1,3})\s*(?:years?\s*old|yr\s*old|yo\b)',
        r'(\d{1,3})[- ]year[- ]old',
        r'\bi\s*am\s*(\d{1,3})',
        r'\biam\s*(\d{1,3})',
        r'^(\d{1,3})$',
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = parse_number(match.group(1))
            if val and MIN_AGE <= val <= MAX_AGE:
                extracted["age"] = int(val)
                break

    income_patterns = [
        r'earn(?:ing)?\s*([\d,.]+\s*(?:lakh|l|cr|crore|k)?)',
        r'annual\s*income\s*(?:is\s*|of\s*)?([\d,.]+\s*(?:lakh|l|cr|crore|k)?)',
        r'income\s*(?:is\s*|of\s*)?([\d,.]+\s*(?:lakh|l|cr|crore|k)?)',
        r'salary\s*(?:is\s*|of\s*)?([\d,.]+\s*(?:lakh|l|cr|crore|k)?)',
        r'([\d,.]+\s*(?:lakh|l|cr|crore)\s*(?:per\s*year|annually|pa|lpa)?)',
        r'([\d,.]+\s*lpa)',
    ]
    for pattern in income_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = parse_number(match.group(1))
            if val and MIN_INCOME <= val <= MAX_INCOME:
                extracted["annual_income"] = int(val)
                break

    dep_patterns = [
        r'(\d+)\s*depend(?:e|a)n',
        r'no\s*depend(?:e|a)n',
        r'zero\s*depend(?:e|a)n',
    ]
    for pattern in dep_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if "no " in pattern or "zero" in pattern:
                extracted["dependents"] = 0
            else:
                val = int(match.group(1))
                if 0 <= val <= MAX_DEP:
                    extracted["dependents"] = val
            break

    spend_patterns = [
        r'monthly\s*(?:expenses?|spending|expenditure)\s*(?:of\s*)?(?:₹\s*)?([\d,.]+\s*(?:lakh|l|k)?)',
        r'(?:expenses?|spending|expenditure)\s*(?:of\s*)?(?:₹\s*)?([\d,.]+\s*(?:lakh|l|k)?)',
        r'spend\s*(?:₹\s*)?([\d,.]+\s*(?:lakh|l|k)?)\s*(?:monthly|per month|a month)',
        r'(?:₹\s*)?([\d,.]+\s*(?:lakh|l|k)?)\s*(?:monthly|per month|a month)',
    ]
    for pattern in spend_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = parse_number(match.group(1))
            if val and val > 0:
                extracted["monthly_spending"] = int(val)
                break

    goal = parse_goal(text)
    if goal:
        extracted["goal"] = goal

    return extracted


def _validate_field(field: str, value, collected: dict):
    if field == "age":
        if value < 0: return False, ERRORS["age_negative"]
        if value < MIN_AGE: return False, ERRORS["age_too_low"]
        if value > MAX_AGE: return False, ERRORS["age_too_high"].format(int(value))
        return True, None

    if field == "annual_income":
        if value < 0: return False, ERRORS["income_negative"]
        if value < MIN_INCOME: return False, ERRORS["income_too_low"].format(f"{int(value):,}")
        if value > MAX_INCOME: return False, ERRORS["income_too_high"].format(f"{int(value):,}")
        return True, None

    if field == "dependents":
        if value < 0: return False, ERRORS["dependents_negative"]
        if value > MAX_DEP: return False, ERRORS["dependents_too_high"].format(value)
        return True, None

    if field == "monthly_spending":
        if value < 0: return False, ERRORS["spending_negative"]
        if value == 0: return False, ERRORS["spending_zero"]
        if "annual_income" in collected:
            monthly_income = collected["annual_income"] / 12
            if value >= monthly_income * 0.95:
                return False, ERRORS["spending_exceeds"].format(
                    f"{int(value):,}", f"{int(monthly_income):,}"
                )
        return True, None

    if field == "goal":
        return True, None

    return True, None


class ProfileCollector:
    def __init__(self):
        self.current_field_index = 0
        self.collected = {
            "specific_plans": [],
            "mode": "suggestion"
        }
        self.done = False
        self.initial_extraction_done = False

    @property
    def current_field(self):
        for field in FIELDS:
            if field not in self.collected:
                return field
        return None

    def _update_plan_detection(self, user_message: str):
        newly_detected = detect_specific_plans(user_message)
        if newly_detected:
            existing = self.collected.get("specific_plans", [])
            merged = list(set(existing + newly_detected))
            self.collected["specific_plans"] = merged
            self.collected["mode"] = detect_mode(merged)

    def _missing_fields(self) -> list:
        return [f for f in FIELDS if f not in self.collected]

    def _format_extracted_summary(self, extracted: dict) -> str:
        goal_display = {
            "pension": "Regular pension / income after retirement",
            "lump_sum": "Lump sum savings at maturity",
            "protection": "Pure life protection (term insurance)",
            "savings": "Long-term savings with life cover",
            "child_planning": "Child education and future planning",
        }
        lines = []
        if "age" in extracted: lines.append(f"- Age: {extracted['age']} years")
        if "annual_income" in extracted: lines.append(f"- Annual income: ₹{extracted['annual_income']:,}")
        if "dependents" in extracted: lines.append(f"- Dependents: {extracted['dependents']}")
        if "monthly_spending" in extracted: lines.append(f"- Monthly expenses: ₹{extracted['monthly_spending']:,}")
        if "goal" in extracted: lines.append(f"- Goal: {goal_display.get(extracted['goal'], extracted['goal'])}")
        return "\n".join(lines)

    def handle(self, user_message: str) -> tuple[str, dict, bool]:
        """
        Returns (response_text, partial_or_full_profile_dict, is_complete_boolean)
        """
        if self.done:
            return "Your profile is already complete!", self.collected, True

        self._update_plan_detection(user_message)

        if not self.initial_extraction_done:
            self.initial_extraction_done = True
            extracted = extract_fields_from_text(user_message)

            if extracted:
                valid_extracted = {}
                validation_errors = []
                for field, value in extracted.items():
                    is_valid, error = _validate_field(field, value, extracted)
                    if is_valid:
                        valid_extracted[field] = value
                        self.collected[field] = value
                    else:
                        validation_errors.append(error)

                missing = self._missing_fields()

                if not missing:
                    self.done = True
                    summary = self._format_extracted_summary(valid_extracted)
                    return (
                        f"Got it! Here's what I picked up:\n\n{summary}\n\nStarting analysis... 🎯"
                    ), self.collected, True

                summary = self._format_extracted_summary(valid_extracted)
                next_field = missing[0]
                next_q = QUESTIONS[next_field]

                confirmation = f"Here's what I gathered:\n\n{summary}\n\n"
                if validation_errors:
                    confirmation += f"⚠️ Some values needed correction: {'; '.join(validation_errors)}\n\n"
                confirmation += f"**{next_q}**"

                return confirmation, self.collected, False

            return (
                f"I'll ask you a few quick questions to find the best plan for you.\n\n"
                f"**{QUESTIONS[FIELDS[0]]}**"
            ), self.collected, False

        field = self.current_field

        if field is None:
            self.done = True
            return "Perfect! I have everything I need. 🎯", self.collected, True

        parse_logic = {
            "age": (parse_number, ERRORS["age_invalid"]),
            "annual_income": (parse_number, ERRORS["income_invalid"]),
            "monthly_spending": (parse_number, ERRORS["spending_invalid"]),
        }

        if field in parse_logic:
            parser, error_msg = parse_logic[field]
            num = parser(user_message)
            if num is None:
                return error_msg, self.collected, False
            
            is_valid, error = _validate_field(field, num, self.collected)
            if not is_valid:
                return error, self.collected, False
                
            self.collected[field] = int(num)

        elif field == "dependents":
            all_nums = [float(x) for x in re.findall(r'\d+', user_message.replace(",", ""))]
            if not all_nums:
                if any(w in user_message.lower() for w in ["no ", "zero", "none", "nobody"]):
                    all_nums = [0]
                else:
                    return ERRORS["dependents_invalid"], self.collected, False
                    
            total = int(sum(all_nums)) if len(all_nums) > 1 else int(all_nums[0])
            is_valid, error = _validate_field("dependents", total, self.collected)
            if not is_valid:
                return error, self.collected, False
            self.collected["dependents"] = total

        elif field == "goal":
            goal = parse_goal(user_message)
            if goal is None:
                return ERRORS["goal_invalid"], self.collected, False
            self.collected["goal"] = goal

        missing = self._missing_fields()
        if not missing:
            self.done = True
            return "Perfect! I have everything I need. 🎯", self.collected, True
            
        success_msgs = {
            "age": f"Got it — age {self.collected.get('age')}! 👍",
            "annual_income": f"Annual income ₹{self.collected.get('annual_income', 0):,} noted! 💰",
            "dependents": f"{self.collected.get('dependents')} dependent(s) noted! 👨‍👩‍👧",
            "monthly_spending": f"Monthly spending ₹{self.collected.get('monthly_spending', 0):,} noted! 🏠",
        }
        
        prefix = success_msgs.get(field, "Got it!")
        return f"{prefix}\n\n**{QUESTIONS[missing[0]]}**", self.collected, False