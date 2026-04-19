from pathlib import Path
import re

from flask import Flask, redirect, render_template_string, request, session, url_for

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = DATA_DIR / "qwen-1.5b-finetuned"

APP_SESSION_VERSION = "v11"

SYSTEM_PROMPT = (
    "You are a friendly AI financial advisor. "
    "Write ONE short friendly opening sentence (max 30 words) that acknowledges the user's "
    "financial goal and tells them you have prepared a personalised plan. "
    "Do NOT use markdown. Do NOT repeat or invent any numbers. Just a warm intro sentence."
)

PROFILE_FIELDS = [
    "monthly_income",
    "monthly_expenses",
    "current_savings",
    "current_debt",
    "financial_goal",
    "risk_tolerance",
    "investment_horizon",
]

FIELD_QUESTIONS = {
    "monthly_income":     "What is your average monthly income after tax? (e.g. '5000' or '$5,000/month')",
    "monthly_expenses":   "What are your average total monthly expenses — including rent, food, bills, transport, subscriptions, etc.? (e.g. '3500' or '$3,500/month')",
    "current_savings":    "How much do you currently have in savings or cash? (e.g. '10000' or '$10k')",
    "current_debt":       "What is your total current debt — including any loans, credit cards, or BNPL? (Say 'none' or '0' if you have no debt.)",
    "financial_goal":     "What is your main financial goal right now?\n  Examples: 'buy a house', 'build an emergency fund', 'start investing', 'pay off debt', 'save more money'",
    "risk_tolerance":     "What is your investment risk tolerance?\n  - low (I want to protect my money)\n  - medium (I'm okay with some ups and downs)\n  - high (I want maximum growth and can handle volatility)",
    "investment_horizon": "What is your investment time horizon?\n  - short term (0–2 years)\n  - medium term (2–5 years)\n  - long term (5+ years)",
}

FIELD_LABELS = {
    "monthly_income":     "monthly income",
    "monthly_expenses":   "monthly expenses",
    "current_savings":    "current savings",
    "current_debt":       "current debt",
    "financial_goal":     "financial goal",
    "risk_tolerance":     "risk tolerance",
    "investment_horizon": "investment horizon",
}

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secret-key"

tokenizer  = None
model      = None
using_lora = False
load_error = None


# ---------------------------------------------------------------------------
# Model loading (optional — app works without it)
# ---------------------------------------------------------------------------

def load_model():
    global tokenizer, model, using_lora, load_error
    if not HAS_TORCH:
        load_error = "PyTorch/Transformers not installed — running in rule-based mode."
        print(load_error)
        return
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, quantization_config=bnb_cfg,
                device_map="auto", trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True,
            )

        model      = base_model
        using_lora = False
        load_error = None

        adapter_path = LORA_PATH / "adapter_config.json"
        if adapter_path.exists():
            try:
                model      = PeftModel.from_pretrained(base_model, str(LORA_PATH), local_files_only=True)
                using_lora = True
                print("LoRA adapter loaded successfully.")
            except Exception as exc:
                load_error = f"LoRA adapter found but could not be loaded: {exc}"
                print(load_error)
        else:
            load_error = f"LoRA adapter not found at {LORA_PATH} — using base model."
            print(load_error)

    except Exception as exc:
        model = tokenizer = None
        using_lora = False
        load_error = f"Model initialization failed: {exc}"
        print(load_error)


load_model()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_default_profile():
    return {field: None for field in PROFILE_FIELDS}


def initialize_session_if_needed():
    if session.get("app_session_version") != APP_SESSION_VERSION:
        session.clear()
        session["app_session_version"] = APP_SESSION_VERSION
        session["profile"]             = get_default_profile()
        session["chat_history"]        = []
        session["guided_mode"]         = False
        session["profile_completed"]   = False
        session["pending_correction"]  = None
        session["pending_clarify"]     = None
        session["followup_topic"]      = None


def get_session_profile():
    profile = session.get("profile")
    if profile is None:
        profile = get_default_profile()
        session["profile"] = profile
    return profile


def save_session_profile(profile):
    session["profile"] = profile
    session.modified   = True


def get_chat_history():
    history = session.get("chat_history")
    if history is None:
        history = []
        session["chat_history"] = history
    return history


def save_chat_history(messages):
    session["chat_history"] = messages
    session.modified         = True


def get_guided_mode():        return session.get("guided_mode", False)
def set_guided_mode(v):       session["guided_mode"] = v; session.modified = True
def get_profile_completed():  return session.get("profile_completed", False)
def set_profile_completed(v): session["profile_completed"] = v; session.modified = True
def get_pending_correction(): return session.get("pending_correction")
def set_pending_correction(v):session["pending_correction"] = v; session.modified = True
def get_pending_clarify():    return session.get("pending_clarify")
def set_pending_clarify(v):   session["pending_clarify"] = v; session.modified = True
def get_followup_topic():     return session.get("followup_topic")
def set_followup_topic(v):    session["followup_topic"] = v; session.modified = True


def reset_session():
    session.clear()
    session["app_session_version"] = APP_SESSION_VERSION
    session["profile"]             = get_default_profile()
    session["chat_history"]        = []
    session["guided_mode"]         = False
    session["profile_completed"]   = False
    session["pending_correction"]  = None
    session["pending_clarify"]     = None
    session["followup_topic"]      = None


# ---------------------------------------------------------------------------
# Value parsing helpers
# ---------------------------------------------------------------------------

def parse_number(value):
    if value is None:
        return None
    text = str(value).strip()
    cleaned = text.replace("$", "").replace(",", "").strip()
    match = re.search(r"^([0-9]+(?:\.[0-9]+)?)\s*(k|K)?$", cleaned)
    if not match:
        match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(k|K)?\b", cleaned)
        if not match:
            return None
    number = float(match.group(1))
    if match.group(2):
        number *= 1000
    return round(number, 2)


def is_zero_debt(text: str) -> bool:
    t = text.lower().strip()
    zero_phrases = [
        "none", "no", "nil", "zero", "nothing", "n/a",
        "no debt", "no debts", "no loans", "no loan",
        "don't have debt", "dont have debt", "do not have debt",
        "no credit card", "no credit card debt", "i have no debt",
        "i don't have any debt", "i have zero debt", "0",
    ]
    return t in zero_phrases or any(t == p or t.startswith(p + " ") or t.endswith(" " + p) for p in zero_phrases)


def normalize_risk_tolerance(text):
    if not text:
        return None
    t = text.lower().strip()
    if any(w in t for w in ["low risk", "conservative", "safe", "protect", "low"]):
        return "low"
    if any(w in t for w in ["medium risk", "moderate", "balanced", "medium", "med", "some risk"]):
        return "medium"
    if any(w in t for w in ["high risk", "aggressive", "growth", "high", "maximum"]):
        return "high"
    return None


def normalize_investment_horizon(text):
    if not text:
        return None
    t = text.lower().strip()
    if "short" in t or "0-2" in t or "1 year" in t or "2 year" in t:
        return "short term"
    if "medium" in t or "mid" in t or "2-5" in t or "3 year" in t or "4 year" in t or "5 year" in t:
        return "medium term"
    if "long" in t or "5+" in t or "10 year" in t or "decade" in t:
        return "long term"
    return None


# ---------------------------------------------------------------------------
# Field tracking
# ---------------------------------------------------------------------------

def get_current_asking_field(profile):
    for field in PROFILE_FIELDS:
        if profile.get(field) is None:
            return field
    return None


def get_next_missing_question(profile):
    field = get_current_asking_field(profile)
    return FIELD_QUESTIONS[field] if field else None


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def extract_field_answer(user_input: str, field: str, profile: dict):
    text  = user_input.strip()
    lower = text.lower()

    if field in ("monthly_income", "monthly_expenses", "current_savings"):
        negative_words = ["no ", "none", "zero", "nil", "nothing", "don't have", "not have"]
        if any(w in lower for w in negative_words) and field != "current_savings":
            return None, False

        value = parse_number(text)
        if value is not None:
            if value <= 0:
                return None, False
            if value >= 100_000:
                return value, True
            return value, False
        return None, False

    if field == "current_debt":
        if is_zero_debt(text):
            return 0, False
        value = parse_number(text)
        if value is not None:
            return value, True
        return None, False

    if field == "financial_goal":
        goal_keywords = [
            "house", "deposit", "invest", "investing", "emergency", "debt",
            "save", "saving", "retire", "retirement", "travel", "car", "education",
            "budget", "wealth", "grow", "income",
        ]
        if len(text) > 2 and any(k in lower for k in goal_keywords):
            return text.strip(), False
        if len(text) > 4:
            return text.strip(), False
        return None, False

    if field == "risk_tolerance":
        risk = normalize_risk_tolerance(lower)
        return risk, False

    if field == "investment_horizon":
        horizon = normalize_investment_horizon(lower)
        return horizon, False

    return None, False


# ---------------------------------------------------------------------------
# Correction detection
# ---------------------------------------------------------------------------

CORRECTION_PATTERNS = [
    (r"(?:update|change|fix|correct|actually|no[,.]?\s+my|wait[,.]?\s+my|sorry[,.]?\s+my|my)\s+"
     r"(?:monthly\s+)?income\s+(?:is|was|=|:)?\s*\$?([0-9.,kK]+)",               "monthly_income"),
    (r"(?:update|change|fix|correct|actually|no[,.]?\s+my|wait[,.]?\s+my|sorry[,.]?\s+my|my)\s+"
     r"(?:monthly\s+)?(?:expenses?|spending|spend|bills)\s+(?:is|are|was|=|:)?\s*\$?([0-9.,kK]+)", "monthly_expenses"),
    (r"(?:update|change|fix|correct|actually|no[,.]?\s+my|wait[,.]?\s+my|sorry[,.]?\s+my|my)\s+"
     r"(?:current\s+)?savings?\s+(?:is|are|was|=|:)?\s*\$?([0-9.,kK]+)",         "current_savings"),
    (r"(?:update|change|fix|correct|actually|no[,.]?\s*|wait|sorry)?\s*"
     r"(?:i\s+(?:have|don'?t\s+have|do\s+not\s+have)\s+)?(?:no|zero|nil|none)\s+(?:debt|loans?|credit)", "current_debt_zero"),
    (r"(?:update|change|fix|correct|actually|no[,.]?\s+my|wait[,.]?\s+my|sorry[,.]?\s+my|my)\s+"
     r"(?:current\s+)?(?:debt|loan|loans)\s+(?:is|are|was|=|:)?\s*\$?([0-9.,kK]+)", "current_debt"),
]


def detect_correction(text, profile):
    lower = text.lower().strip()
    for pattern, field in CORRECTION_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            if field == "current_debt_zero":
                return "current_debt", 0
            raw = m.group(1) if m.lastindex and m.lastindex >= 1 else None
            if raw:
                value = parse_number(raw)
                if value is not None:
                    current_asking = get_current_asking_field(profile)
                    if profile.get(field) is not None or field != current_asking:
                        return field, value
    return None, None


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def calculate_financial_metrics(profile):
    income   = float(profile.get("monthly_income")  or 0)
    expenses = float(profile.get("monthly_expenses") or 0)
    savings  = float(profile.get("current_savings")  or 0)
    debt     = float(profile.get("current_debt")     or 0)
    surplus  = income - expenses
    return {
        "income":           income,
        "expenses":         expenses,
        "savings":          savings,
        "debt":             debt,
        "surplus":          surplus,
        "savings_rate":     round((surplus / income * 100) if income > 0 else 0, 1),
        "emergency_months": round((savings / expenses)     if expenses > 0 else 0, 1),
    }


# ---------------------------------------------------------------------------
# Rule-based advice
# ---------------------------------------------------------------------------

def build_rule_based_advice(profile):
    m         = calculate_financial_metrics(profile)
    income    = m["income"]
    expenses  = m["expenses"]
    savings   = m["savings"]
    debt      = m["debt"]
    surplus   = m["surplus"]
    em_months = m["emergency_months"]
    goal      = str(profile.get("financial_goal") or "improve finances")
    risk      = str(profile.get("risk_tolerance") or "medium")
    horizon   = str(profile.get("investment_horizon") or "medium term")
    goal_lower = goal.lower()

    target_em    = expenses * 6
    em_gap       = max(0.0, target_em - savings)
    months_to_em = (em_gap / surplus) if surplus > 0 and em_gap > 0 else None

    lines = []

    lines.append("YOUR FINANCIAL PROFILE (as provided)")
    lines.append("=" * 50)
    lines.append(f"  Monthly income:       ${income:,.2f}  ← you told me this")
    lines.append(f"  Monthly expenses:     ${expenses:,.2f}  ← you told me this")
    lines.append(f"  Current savings:      ${savings:,.2f}  ← you told me this")
    lines.append(f"  Current debt:         ${debt:,.2f}  ← you told me this")
    lines.append(f"  Goal:                 {goal}")
    lines.append(f"  Risk tolerance:       {risk}")
    lines.append(f"  Investment horizon:   {horizon}")
    lines.append("")

    lines.append("CALCULATED FIGURES (showing working)")
    lines.append("=" * 50)
    lines.append(f"  Monthly surplus:")
    lines.append(f"    = income − expenses")
    lines.append(f"    = ${income:,.2f} − ${expenses:,.2f}")
    lines.append(f"    = ${surplus:,.2f}/month")
    lines.append("")
    lines.append(f"  Savings rate:")
    if income > 0:
        lines.append(f"    = (surplus ÷ income) × 100")
        lines.append(f"    = (${surplus:,.2f} ÷ ${income:,.2f}) × 100")
        lines.append(f"    = {m['savings_rate']:.1f}%")
    else:
        lines.append(f"    = N/A (no income recorded)")
    lines.append("")
    lines.append(f"  Emergency fund (current coverage):")
    if expenses > 0:
        lines.append(f"    = current savings ÷ monthly expenses")
        lines.append(f"    = ${savings:,.2f} ÷ ${expenses:,.2f}")
        lines.append(f"    = {em_months:.2f} months covered")
    else:
        lines.append(f"    = N/A (no expenses recorded)")
    lines.append("")
    lines.append(f"  Emergency fund target (6 months of expenses):")
    lines.append(f"    = 6 × ${expenses:,.2f} = ${target_em:,.2f}")
    lines.append("")
    lines.append(f"  Emergency fund gap:")
    lines.append(f"    = ${target_em:,.2f} − ${savings:,.2f} = ${em_gap:,.2f} still needed")
    if months_to_em is not None and surplus > 0:
        lines.append(f"")
        lines.append(f"  Time to fill emergency fund gap:")
        lines.append(f"    = ${em_gap:,.2f} ÷ ${surplus:,.2f} = {months_to_em:.1f} months ({months_to_em/12:.1f} years)")
    lines.append("")
    lines.append("=" * 50)
    lines.append("")

    lines.append("YOUR PERSONALISED STRATEGY")
    lines.append("=" * 50)
    lines.append("")
    step = 1

    # Emergency Fund
    if em_months < 3:
        lines.append(f"STEP {step}: BUILD YOUR EMERGENCY FUND (Top Priority)")
        lines.append(f"  Status: {em_months:.1f} months covered — below the 3-month minimum.")
        lines.append(f"  Target: 6 months of expenses = ${target_em:,.2f}")
        lines.append(f"  Gap:    ${em_gap:,.2f} remaining")
        if surplus > 0 and months_to_em is not None:
            lines.append(f"  Timeline: ${em_gap:,.2f} ÷ ${surplus:,.2f}/month = {months_to_em:.0f} months")
            lines.append(f"  Action: Auto-transfer your full ${surplus:,.2f} surplus into a high-yield")
            lines.append(f"          savings account each payday until this goal is reached.")
        else:
            lines.append(f"  Action: Your surplus is currently ${surplus:,.2f}. Review your expenses")
            lines.append(f"          to create a positive monthly surplus before building this fund.")
    elif em_months < 6:
        lines.append(f"STEP {step}: TOP UP YOUR EMERGENCY FUND")
        lines.append(f"  Status: {em_months:.1f} months covered — almost at the 6-month target.")
        lines.append(f"  Target: 6 months = ${target_em:,.2f}")
        lines.append(f"  Gap:    ${em_gap:,.2f} still needed")
        if surplus > 0 and months_to_em is not None:
            lines.append(f"  Timeline: {months_to_em:.0f} months to complete at ${surplus:,.2f}/month")
    else:
        lines.append(f"STEP {step}: EMERGENCY FUND — COMPLETE ✓")
        lines.append(f"  Status: {em_months:.1f} months covered — above the 6-month target.")
        lines.append(f"  No action needed. Your safety net is solid.")
    step += 1
    lines.append("")

    # Debt
    if debt > 0:
        lines.append(f"STEP {step}: PAY DOWN DEBT")
        lines.append(f"  Total debt: ${debt:,.2f}")
        if surplus > 0:
            suggested_debt_pmt = surplus * 0.4
            months_to_clear    = debt / suggested_debt_pmt
            lines.append(f"  Suggested monthly payment: 40% of surplus")
            lines.append(f"    = 0.40 × ${surplus:,.2f} = ${suggested_debt_pmt:,.2f}/month")
            lines.append(f"  Estimated payoff: {months_to_clear:.0f} months ({months_to_clear/12:.1f} years)")
            lines.append(f"  Method: Avalanche — pay highest-rate debt first")
        else:
            lines.append(f"  Reduce expenses to create surplus for debt repayment.")
    else:
        lines.append(f"STEP {step}: DEBT — NONE ✓")
        lines.append(f"  You have no debt. Your full ${surplus:,.2f}/month surplus is available")
        lines.append(f"  to direct entirely toward your goal.")
    step += 1
    lines.append("")

    # Goal-specific step
    if "house" in goal_lower or "deposit" in goal_lower:
        lines.append(f"STEP {step}: SAVING FOR A HOUSE DEPOSIT")
        lines.append(f"  Monthly available (after emergency fund): ${surplus:,.2f}")
        lines.append(f"")
        lines.append(f"  Deposit milestone estimates:")
        for target_dep in [50_000, 80_000, 100_000, 150_000, 200_000]:
            already  = max(0.0, savings - target_em)
            need     = max(0.0, target_dep - already)
            if surplus > 0 and need > 0:
                em_wait  = months_to_em or 0
                dep_mths = need / surplus
                total    = em_wait + dep_mths
                lines.append(f"    ${target_dep:>9,.0f} deposit: {dep_mths:.0f} months saving + {em_wait:.0f} emergency = {total:.0f} months total ({total/12:.1f} yrs)")
        lines.append(f"")
        if risk == "low":
            lines.append(f"  Keep deposit in: High-interest savings account or term deposit.")
        elif risk == "medium":
            lines.append(f"  Keep deposit in: HISA for <2yrs, term deposit or bond ETF for 2+yrs.")
        else:
            lines.append(f"  Keep deposit in: HISA for near-term, balanced ETF for 3+ years away.")

    elif "invest" in goal_lower or "investing" in goal_lower or "stock" in goal_lower:
        lines.append(f"STEP {step}: STARTING TO INVEST")
        lines.append(f"  Risk profile: {risk} | Horizon: {horizon}")
        available_to_invest = surplus * 0.5 if em_months >= 6 and debt == 0 else surplus * 0.3
        pct = 50 if em_months >= 6 and debt == 0 else 30
        lines.append(f"")
        lines.append(f"  Suggested monthly investment: {pct}% of surplus")
        lines.append(f"    = {pct}% × ${surplus:,.2f} = ${available_to_invest:,.2f}/month")
        lines.append(f"")
        if risk == "low":
            lines.append(f"  Options for low-risk investor:")
            lines.append(f"    → High-interest savings account (ING, Macquarie, UBank)")
            lines.append(f"    → Term deposits (fixed rate, government-protected)")
            lines.append(f"    → Government bond ETFs (e.g. IAF on ASX)")
        elif risk == "medium":
            lines.append(f"  Options for medium-risk investor:")
            lines.append(f"    → Broad index ETF (e.g. VAS for ASX, IVV for US market)")
            lines.append(f"    → All-in-one diversified ETF (e.g. VDHG or DHHF)")
            lines.append(f"    → Dollar-cost average: invest ${available_to_invest:,.2f} same day each month")
        else:
            lines.append(f"  Options for high-risk investor:")
            lines.append(f"    → Broad share market ETFs (VAS, NDQ, IVV)")
            lines.append(f"    → Individual ASX or US stocks (requires research)")
            lines.append(f"    → Suggested split: 60-70% ETFs, 20-30% stocks, 10% speculative")
        lines.append(f"")
        lines.append(f"  How to start (Australia):")
        lines.append(f"    1. Open a brokerage account: CommSec, Stake, Pearler, or SelfWealth")
        lines.append(f"    2. Buy a broad index ETF first — low fees, instant diversification")
        lines.append(f"    3. Set up automatic monthly investment on payday")

    elif "emergen" in goal_lower:
        lines.append(f"STEP {step}: BUILDING YOUR EMERGENCY FUND")
        lines.append(f"  This is your stated goal — see Step 1 for full details and timeline.")
        lines.append(f"  Best account: high-yield savings account (instant access, APRA-protected).")

    elif "debt" in goal_lower or "reduc" in goal_lower or "pay off" in goal_lower:
        lines.append(f"STEP {step}: ELIMINATING YOUR DEBT")
        if debt > 0:
            lines.append(f"  This is your stated goal — see Step 2 for full payoff timeline.")
        else:
            lines.append(f"  You have no debt — great position! Focus on building wealth instead.")

    elif "save" in goal_lower or "saving" in goal_lower:
        lines.append(f"STEP {step}: SAVING MORE MONEY")
        lines.append(f"  Current savings rate: {m['savings_rate']:.1f}% of income")
        lines.append(f"  Target: aim for 20%+ savings rate = ${income * 0.2:,.2f}/month")
        lines.append(f"  Gap: ${max(0, income * 0.2 - surplus):,.2f}/month more to save")
        lines.append(f"  Action: Automate ${surplus:,.2f} transfer on payday into a HISA.")
    else:
        lines.append(f"STEP {step}: WORKING TOWARDS YOUR GOAL: {goal.upper()}")
        lines.append(f"  Available each month: ${surplus:,.2f} surplus")
        lines.append(f"  Action: Set up an automatic transfer on payday so savings grow consistently.")
    step += 1
    lines.append("")

    # Surplus allocation
    if surplus > 0:
        lines.append(f"STEP {step}: SUGGESTED MONTHLY SURPLUS ALLOCATION")
        lines.append(f"  Total available: ${surplus:,.2f}/month")
        lines.append("")
        if em_months < 6 and em_gap > 0:
            em_alloc     = min(surplus, em_gap)
            remaining    = max(0.0, surplus - em_alloc)
            debt_alloc   = remaining * 0.5 if debt > 0 else 0
            invest_alloc = remaining - debt_alloc
            lines.append(f"  Emergency fund top-up:  ${em_alloc:,.2f}/month (priority)")
            lines.append(f"  Remaining after emergency fund: ${remaining:,.2f}/month")
            if debt > 0:
                lines.append(f"  Debt repayment:         ${debt_alloc:,.2f}/month (50% of remaining)")
            if invest_alloc > 0:
                lines.append(f"  Goal / investing:       ${invest_alloc:,.2f}/month")
        elif debt > 0:
            lines.append(f"  Debt repayment:         ${surplus*0.4:,.2f}/month (40%)")
            lines.append(f"  Investing / savings:    ${surplus*0.5:,.2f}/month (50%)")
            lines.append(f"  Buffer:                 ${surplus*0.1:,.2f}/month (10%)")
        else:
            lines.append(f"  Main goal / investing:  ${surplus*0.7:,.2f}/month (70%)")
            lines.append(f"  Buffer / fun money:     ${surplus*0.2:,.2f}/month (20%)")
            lines.append(f"  Irregular expenses:     ${surplus*0.1:,.2f}/month (10%)")
        lines.append("")
        lines.append(f"  (Adjust these splits to match your personal priorities.)")
        step += 1
        lines.append("")

    lines.append(f"IMPORTANT REMINDER")
    lines.append(f"  All figures are based solely on the data you provided.")
    lines.append(f"  As a {risk}-risk investor with a {horizon} horizon, always prioritise")
    lines.append(f"  stability and never invest money you cannot afford to lose.")
    lines.append(f"  These recommendations are educational — consult a licensed financial advisor")
    lines.append(f"  for personalised legal or tax advice.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# yfinance — Real-time market data with validation
# ---------------------------------------------------------------------------

KNOWN_TICKERS = {
    "AAPL","TSLA","MSFT","GOOGL","GOOG","AMZN","NVDA","META","NFLX","UBER",
    "BRKB","BRK","JPM","JNJ","PG","KO","DIS","BA","GE","XOM","WMT","V","MA",
    "SPY","QQQ","IVV","VTI","VOO","AGG","GLD","SLV",
    "BHP","CBA","WBC","ANZ","NAB","RIO","WES","TLS","MQG","CSL","FMG","WOW",
    "TCL","STO","VAS","VGS","IOZ","A200","NDQ","VDHG","DHHF",
    "SPX","NDX","DJI",
}

# Sanity bounds for financial data
SANITY_BOUNDS = {
    "dividendYield": (0.0, 0.30),   # 0% to 30% — reject wild values like 38%
    "trailingPE":    (0.0, 2000.0), # reject negative or absurd P/E
}


def sanitize_yf_value(key, value):
    """Return value if within sanity bounds, else None."""
    if value is None:
        return None
    bounds = SANITY_BOUNDS.get(key)
    if bounds and not (bounds[0] <= value <= bounds[1]):
        return None
    return value


def extract_ticker_from_text(text: str):
    m = re.search(r'\$([A-Z]{1,5})\b', text)
    if m:
        return m.group(1).upper()

    stock_kw = r'(?:stock|shares?|price|etf|ticker|asx|listed)'
    m = re.search(r'\b([A-Z]{2,5})\s+' + stock_kw, text)
    if m:
        return m.group(1).upper()
    m = re.search(stock_kw + r'\s+(?:of\s+)?([A-Z]{2,5})\b', text)
    if m:
        return m.group(1).upper()

    upper = text.upper()
    for tok in KNOWN_TICKERS:
        if re.search(r'\b' + re.escape(tok) + r'\b', upper):
            return tok

    name_map = {
        "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT",
        "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN",
        "nvidia": "NVDA", "meta": "META", "netflix": "NFLX",
        "bhp": "BHP.AX", "commonwealth bank": "CBA.AX", "westpac": "WBC.AX",
        "anz": "ANZ.AX", "nab": "NAB.AX", "rio tinto": "RIO.AX",
        "wesfarmers": "WES.AX", "telstra": "TLS.AX", "macquarie": "MQG.AX",
        "csl": "CSL.AX", "fortescue": "FMG.AX", "woolworths": "WOW.AX",
    }
    lower = text.lower()
    for name, ticker in name_map.items():
        if name in lower:
            return ticker

    return None


def is_stock_price_query(text: str) -> bool:
    """Detect queries that are SPECIFICALLY asking for a stock price/data."""
    lower = text.lower()
    price_phrases = [
        "stock price", "share price", "market price", "current price",
        "trading at", "worth today", "price today", "what is", "how much is",
        "52 week", "market cap", "dividend", "p/e", "pe ratio",
        "price of", "value of",
    ]
    # Has $ ticker format
    if re.search(r'\$[A-Z]{1,5}\b', text):
        return True
    # Has known ticker + price keyword
    if extract_ticker_from_text(text) and any(p in lower for p in price_phrases):
        return True
    # Explicit price-only phrases
    if any(p in lower for p in ["stock price", "share price", "price today", "trading at"]):
        return True
    return False


def is_investment_advice_query(text: str) -> bool:
    """Detect queries asking whether to invest in something (not just price lookup)."""
    lower = text.lower()
    advice_phrases = [
        "should i invest", "should i buy", "is it worth", "worth investing",
        "good investment", "recommend", "given my profile", "for my profile",
        "should i add", "is it a good buy", "what do you think about investing",
        "advice on", "thoughts on buying", "should i get",
    ]
    return any(p in lower for p in advice_phrases)


def fetch_yfinance_response(raw_ticker: str, profile=None) -> str:
    if not HAS_YF:
        return "Live market data is not available (yfinance not installed)."

    def _try_fetch(symbol):
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="5d")
            if hist.empty:
                return None, None
            return t, hist
        except Exception:
            return None, None

    ticker, hist = _try_fetch(raw_ticker)
    if hist is None and "." not in raw_ticker:
        ticker, hist = _try_fetch(raw_ticker + ".AX")
        if hist is not None:
            raw_ticker = raw_ticker + ".AX"

    if hist is None or ticker is None:
        return (
            f"I couldn't find real-time data for '{raw_ticker}'. "
            "Please double-check the ticker symbol (e.g. AAPL, BHP.AX) and try again."
        )

    try:
        info      = ticker.info or {}
        price     = hist["Close"].iloc[-1]
        currency  = info.get("currency", "USD")
        name      = info.get("longName") or info.get("shortName") or raw_ticker

        lines = [f"Real-time Market Data: {name} ({raw_ticker.upper()})"]
        lines.append("-" * 45)
        lines.append(f"  Current price:  {currency} ${price:,.2f}")

        if len(hist) > 1:
            prev  = hist["Close"].iloc[-2]
            chg   = price - prev
            pct   = chg / prev * 100
            arrow = "+" if chg >= 0 else ""
            lines.append(f"  Daily change:   {arrow}{currency} ${chg:,.2f}  ({chg:+.2f}%)")

        hi52 = sanitize_yf_value("fiftyTwoWeekHigh", info.get("fiftyTwoWeekHigh"))
        lo52 = sanitize_yf_value("fiftyTwoWeekLow",  info.get("fiftyTwoWeekLow"))
        pe   = sanitize_yf_value("trailingPE",       info.get("trailingPE"))
        div  = sanitize_yf_value("dividendYield",    info.get("dividendYield"))

        if hi52: lines.append(f"  52-week high:   {currency} ${hi52:,.2f}")
        if lo52: lines.append(f"  52-week low:    {currency} ${lo52:,.2f}")
        if pe:   lines.append(f"  P/E ratio:      {pe:.1f}")
        if div:  lines.append(f"  Dividend yield: {div*100:.2f}%")

        if info.get("marketCap"):
            mc     = info["marketCap"]
            mc_str = f"${mc/1e12:.2f}T" if mc >= 1e12 else f"${mc/1e9:.1f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"
            lines.append(f"  Market cap:     {mc_str}")
        if info.get("sector"):
            lines.append(f"  Sector:         {info['sector']}")
        lines.append("")
        lines.append("Data provided by Yahoo Finance. For information only — not financial advice.")

        return "\n".join(lines)

    except Exception as e:
        print(f"yfinance parse error: {e}")
        return f"I found data for {raw_ticker.upper()} but had trouble formatting it. Please try again."


def build_stock_investment_advice(ticker_str: str, profile: dict, price_data: str) -> str:
    """
    Build profile-aware investment advice for a specific stock/ETF.
    price_data is the formatted price string already fetched.
    """
    m        = calculate_financial_metrics(profile)
    risk     = str(profile.get("risk_tolerance") or "medium")
    horizon  = str(profile.get("investment_horizon") or "medium term")
    surplus  = m["surplus"]
    debt     = m["debt"]
    em_months = m["emergency_months"]

    ticker_upper = ticker_str.upper().replace(".AX", "")
    is_etf = ticker_upper in {"VAS","VGS","IVV","VOO","VTI","QQQ","SPY","NDQ","VDHG","DHHF","IOZ","A200","IAF","VAF","AGG"}
    is_asx = ".AX" in ticker_str.upper() or ticker_upper in {"VAS","CBA","BHP","WBC","ANZ","NAB","RIO","WES","TLS","MQG","CSL","FMG","WOW","NDQ","VDHG","DHHF"}

    lines = [price_data, ""]
    lines.append("=" * 45)
    lines.append(f"INVESTMENT ADVICE FOR YOUR PROFILE")
    lines.append("=" * 45)
    lines.append(f"  Risk tolerance:   {risk}")
    lines.append(f"  Time horizon:     {horizon}")
    lines.append(f"  Monthly surplus:  ${surplus:,.2f}")
    lines.append(f"  Emergency fund:   {em_months:.1f} months covered")
    lines.append(f"  Current debt:     ${debt:,.2f}")
    lines.append("")

    # Prerequisites check
    warnings = []
    if em_months < 3:
        warnings.append(f"⚠️  Your emergency fund covers only {em_months:.1f} months (target: 6 months = ${m['expenses']*6:,.2f}). Build this before investing.")
    if debt > 0:
        warnings.append(f"⚠️  You have ${debt:,.2f} in debt. High-interest debt should generally be paid off before investing.")

    if warnings:
        lines.append("PREREQUISITES TO ADDRESS FIRST:")
        for w in warnings:
            lines.append(f"  {w}")
        lines.append("")

    # ETF vs individual stock advice
    if is_etf:
        lines.append(f"ABOUT {ticker_upper} (ETF):")
        lines.append(f"  ETFs like {ticker_upper} are generally well-suited to your {risk} risk profile.")
        lines.append(f"  They offer instant diversification and lower fees than managed funds.")
        lines.append("")
        if risk == "low":
            lines.append(f"VERDICT: CONSIDER WITH CAUTION")
            lines.append(f"  Even diversified ETFs carry market risk. If capital protection is your")
            lines.append(f"  priority, a HISA or term deposit may suit you better.")
        elif risk == "medium":
            lines.append(f"VERDICT: SUITABLE FOR YOUR PROFILE ✓")
            lines.append(f"  A diversified ETF like {ticker_upper} aligns well with medium risk + {horizon} horizon.")
            lines.append(f"  Consider investing ${min(surplus*0.5, 1000):,.0f}–${min(surplus*0.7, 2000):,.0f}/month via dollar-cost averaging.")
        else:
            lines.append(f"VERDICT: WELL-SUITED TO YOUR PROFILE ✓")
            lines.append(f"  For a high-risk investor with {horizon} horizon, {ticker_upper} is a solid core holding.")
            lines.append(f"  Consider investing ${min(surplus*0.6, 2000):,.0f}–${min(surplus*0.8, 3000):,.0f}/month.")
    else:
        lines.append(f"ABOUT {ticker_upper} (Individual Stock):")
        lines.append(f"  Individual stocks carry higher concentration risk than diversified ETFs.")
        lines.append("")
        if risk == "low":
            lines.append(f"VERDICT: NOT RECOMMENDED FOR YOUR PROFILE")
            lines.append(f"  Individual stocks like {ticker_upper} are too volatile for a low-risk investor.")
            lines.append(f"  Consider: term deposits, government bond ETFs, or a HISA instead.")
        elif risk == "medium":
            lines.append(f"VERDICT: SUITABLE AS A SMALL PORTFOLIO COMPONENT")
            lines.append(f"  As a medium-risk investor, keep individual stocks like {ticker_upper} to")
            lines.append(f"  no more than 10–20% of your investment portfolio.")
            max_monthly = min(surplus * 0.1, 500)
            lines.append(f"  Suggested maximum: ${max_monthly:,.0f}/month (10% of ${surplus:,.2f} surplus)")
            lines.append(f"  Build your core with index ETFs (e.g. VAS, IVV) first.")
        else:
            lines.append(f"VERDICT: SUITABLE FOR A GROWTH PORTFOLIO")
            lines.append(f"  With high risk tolerance and {horizon} horizon, {ticker_upper} can fit a")
            lines.append(f"  growth-focused portfolio. Keep to 20–30% of total portfolio.")
            max_monthly = min(surplus * 0.25, 1500)
            lines.append(f"  Suggested maximum: ${max_monthly:,.0f}/month alongside diversified ETFs.")

    lines.append("")
    lines.append("SUGGESTED MONTHLY INVESTMENT PLAN:")
    if em_months >= 3 and debt == 0:
        invest_pool = surplus * (0.5 if em_months >= 6 else 0.3)
        lines.append(f"  Available to invest: ${invest_pool:,.2f}/month")
        if is_etf:
            lines.append(f"  → {ticker_upper}: ${invest_pool:,.2f}/month (can be 100% of investment pool)")
        else:
            lines.append(f"  → Core ETFs (e.g. VAS/IVV): ${invest_pool*0.8:,.2f}/month (80%)")
            lines.append(f"  → {ticker_upper}: ${invest_pool*0.2:,.2f}/month (20% max for individual stocks)")
    else:
        lines.append(f"  Focus on emergency fund / debt first. Invest only when prerequisites are met.")

    lines.append("")
    lines.append("How to buy (Australia):")
    if is_asx:
        lines.append(f"  → CommSec, Pearler, Stake, or SelfWealth for ASX-listed securities")
    else:
        lines.append(f"  → Stake or Interactive Brokers for US-listed stocks")
        lines.append(f"  → Note: USD conversion fees apply — factor this into your cost")

    lines.append("")
    lines.append("⚠️  This is educational information only, not financial advice.")
    lines.append("    Consult a licensed financial adviser before investing.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Follow-up helpers
# ---------------------------------------------------------------------------

VAGUE_FOLLOWUP_PHRASES = [
    "more help", "need help", "help me", "what else", "anything else", "tell me more",
    "more advice", "what should i do", "what now", "next steps", "other options",
    "more options", "what about", "i need advice", "give me advice", "any advice",
    "any suggestions", "what do you suggest", "help", "i need more", "more information",
]


def is_vague_followup(text: str) -> bool:
    lower      = text.lower().strip()
    word_count = len(lower.split())
    return word_count <= 8 and any(p in lower for p in VAGUE_FOLLOWUP_PHRASES)


def get_clarifying_followup_menu(profile) -> str:
    goal = str(profile.get("financial_goal", "")).lower()
    m    = calculate_financial_metrics(profile)

    base = (
        "Of course! Here are some specific things I can help you with next.\n"
        "Just ask one of these (or in your own words):\n\n"
    )

    if "house" in goal or "deposit" in goal:
        options = (
            "1. 'How long until I have enough for a house deposit?'\n"
            "2. 'How can I save more each month toward my house?'\n"
            "3. 'Where should I keep my house deposit savings?'\n"
            "4. 'Should I use a term deposit or savings account?'\n"
            "5. Look up a stock price — e.g. '$CBA price' or '$AAPL price'\n"
        )
    elif "invest" in goal or "investing" in goal:
        options = (
            "1. 'How do I start investing with my monthly surplus?'\n"
            "2. 'What ETFs are good for my risk profile?'\n"
            "3. 'What is the difference between ETFs and individual stocks?'\n"
            "4. 'Should I invest in NVDA?' or 'Should I invest in VAS?'\n"
            "5. 'How should I split my surplus between investing and saving?'\n"
        )
    elif "debt" in goal or "reduc" in goal:
        options = (
            "1. 'How long will it take to pay off my debt?'\n"
            "2. 'Explain the avalanche vs snowball debt method'\n"
            "3. 'How much of my surplus should go to debt vs savings?'\n"
            "4. 'When should I start investing after paying off debt?'\n"
            "5. Look up a stock price — e.g. '$AAPL'\n"
        )
    else:
        options = (
            "1. 'How long until I reach my savings goal?'\n"
            "2. 'How can I reduce my monthly expenses?'\n"
            "3. 'How do I start investing?'\n"
            "4. 'What is the best savings account in Australia?'\n"
            "5. 'Should I invest in VAS ETF?'\n"
        )

    surplus_hint = (
        f"\nRemember: your ${m['surplus']:,.2f}/month surplus is your main tool. "
        "Ask me how to put it to work!"
    )
    return base + options + surplus_hint


TIMELINE_PATTERNS  = ["how long", "when can i", "how many months", "how many years", "timeline", "how soon", "when will i"]
BUDGET_PATTERNS_F  = ["save more", "reduce expenses", "cut costs", "spend less", "budget", "lower my expenses", "increase savings", "trim", "save faster"]
INVEST_BASICS_PATS = ["how do i invest", "how to invest", "start investing", "what etf", "which etf", "index fund", "where to invest", "how should i invest", "begin invest", "difference between etf"]
SPLIT_PATTERNS     = ["split", "allocate", "divide", "how much to", "portion", "percentage", "how much should i put"]
DEPOSIT_WHERE      = ["where to keep", "where should i keep", "best place", "high yield", "term deposit", "savings account", "where to save", "keep my deposit", "keep my savings"]
DEBT_TIME_PATTERNS = ["pay off my debt", "clear my debt", "debt free", "how long.*debt", "how long.*loan"]
AVALANCHE_PATTERNS = ["avalanche", "snowball", "debt method", "which method"]


def handle_specific_followup_rule_based(user_input: str, profile: dict):
    lower = user_input.lower()
    m     = calculate_financial_metrics(profile)
    goal  = str(profile.get("financial_goal", "")).lower()
    risk  = str(profile.get("risk_tolerance", "medium"))
    hor   = str(profile.get("investment_horizon", "medium term"))
    income    = m["income"]
    expenses  = m["expenses"]
    savings   = m["savings"]
    debt      = m["debt"]
    surplus   = m["surplus"]
    em_months = m["emergency_months"]
    target_em = expenses * 6
    em_gap    = max(0.0, target_em - savings)

    if any(p in lower for p in TIMELINE_PATTERNS) and ("house" in lower or "deposit" in lower or "house" in goal):
        if surplus <= 0:
            return (
                f"Your monthly surplus is currently ${surplus:,.2f} "
                f"(income ${income:,.2f} - expenses ${expenses:,.2f}). "
                "You need a positive surplus to save for a house deposit."
            )
        months_to_em_val = round(em_gap / surplus) if em_gap > 0 and surplus > 0 else 0
        lines = [
            "House Deposit Savings Timeline:",
            "",
            f"  Your monthly surplus: ${income:,.2f} − ${expenses:,.2f} = ${surplus:,.2f}",
            f"  Emergency fund gap:   ${em_gap:,.2f} = {months_to_em_val} months at ${surplus:,.2f}/month",
            "",
            "  After completing emergency fund, full surplus goes to house deposit:",
            "",
        ]
        for target in [50_000, 80_000, 100_000, 150_000, 200_000]:
            already = max(0.0, savings - target_em)
            need    = max(0.0, target - already)
            if surplus > 0 and need > 0:
                dep_mths = need / surplus
                total    = months_to_em_val + dep_mths
                lines.append(f"  ${target:>8,.0f} deposit: {dep_mths:.0f} months + {months_to_em_val} emergency = {total:.0f} months total ({total/12:.1f} yrs)")
        lines.append("")
        lines.append("Tip: Keep deposit savings in a high-interest savings account or term deposit.")
        return "\n".join(lines)

    if any(p in lower for p in BUDGET_PATTERNS_F):
        savings_rate = m["savings_rate"]
        lines = [
            "Ways to Boost Your Monthly Surplus:",
            "",
            f"  Current surplus:    ${surplus:,.2f}/month  ({savings_rate:.1f}% savings rate)",
            f"  Current expenses:   ${expenses:,.2f}/month",
            "",
            "  Quick wins:",
            "  - Cancel unused subscriptions (streaming, gym, apps): saves $50–$200/month",
            "  - Meal prep instead of eating out: saves $200–$400/month",
            "  - Compare insurance and utilities annually: saves $50–$150/month",
            "  - Use a budget app (YNAB, Frollo, or MoneyBrilliant)",
            "",
            "  Bigger moves:",
            "  - Renegotiate rent or consider a housemate",
            "  - Refinance any high-interest debt to a lower rate",
            "  - Direct any pay rise straight to savings",
            "",
            f"  Even saving an extra $300/month = $3,600 more per year.",
        ]
        return "\n".join(lines)

    if any(p in lower for p in INVEST_BASICS_PATS):
        invest_amt = surplus * (0.5 if debt == 0 and em_months >= 6 else 0.3)
        pct        = 50 if debt == 0 and em_months >= 6 else 30
        if risk == "low":
            options = (
                "  - High-interest savings account (ING, Macquarie, UBank)\n"
                "  - Term deposits (fixed rate, government-protected up to $250k)\n"
                "  - Government bond ETFs (e.g. IAF on ASX)\n"
                "  - Avoid: individual stocks, crypto"
            )
        elif risk == "medium":
            options = (
                "  - Diversified index ETF (e.g. VAS for ASX, IVV for US)\n"
                "  - All-in-one ETF (VDHG or DHHF — set and forget)\n"
                "  - Dollar-cost average: same amount invested each month\n"
                "  - ETFs vs Stocks: ETFs = instant diversification, lower risk;\n"
                "    individual stocks = higher potential return but also higher risk"
            )
        else:
            options = (
                "  - Broad share ETFs (VAS, NDQ, IVV)\n"
                "  - Individual ASX or US stocks (research required)\n"
                "  - Suggested split: 60–70% ETFs, 20–30% stocks, 10% speculative"
            )
        lines = [
            f"Getting Started with Investing (risk: {risk} | horizon: {hor}):",
            "",
            f"  Suggested monthly investment: {pct}% of your ${surplus:,.2f} surplus = ${invest_amt:,.2f}/month",
            "",
            f"  Recommended options for your risk profile:",
            options,
            "",
            "  How to start (Australia):",
            "  1. Open a brokerage: CommSec, Stake, Pearler, or SelfWealth",
            "  2. Buy a broad index ETF — low fees, instant diversification",
            "  3. Auto-invest on payday each month",
            "  4. Leave it alone and let compounding work",
            "",
            "  Ask me 'Should I invest in NVDA?' or '$VAS price' for specific stock advice.",
            "",
            "  Reminder: Investments can fall as well as rise. Never invest money you cannot afford to lose.",
        ]
        return "\n".join(lines)

    if any(p in lower for p in SPLIT_PATTERNS):
        lines = [f"Suggested Surplus Split for ${surplus:,.2f}/month:"]
        lines.append("")
        if em_months < 6 and em_gap > 0:
            em_alloc  = min(surplus, em_gap)
            remaining = max(0.0, surplus - em_alloc)
            d_alloc   = remaining * 0.4 if debt > 0 else 0
            i_alloc   = remaining - d_alloc
            lines.append(f"  Emergency fund top-up:  ${em_alloc:,.2f}/month")
            lines.append(f"  Remaining:              ${remaining:,.2f}/month")
            if debt > 0:
                lines.append(f"  Debt repayment:         ${d_alloc:,.2f}/month  (40% of remaining)")
            if i_alloc > 0:
                lines.append(f"  Investing / goal:       ${i_alloc:,.2f}/month")
        elif debt > 0:
            lines.append(f"  Debt repayment:         ${surplus*0.4:,.2f}/month  (40%)")
            lines.append(f"  Investing / savings:    ${surplus*0.5:,.2f}/month  (50%)")
            lines.append(f"  Buffer:                 ${surplus*0.1:,.2f}/month  (10%)")
        else:
            lines.append(f"  Main goal / investing:  ${surplus*0.7:,.2f}/month  (70%)")
            lines.append(f"  Buffer / fun money:     ${surplus*0.2:,.2f}/month  (20%)")
            lines.append(f"  Irregular expenses:     ${surplus*0.1:,.2f}/month  (10%)")
        lines.append("")
        lines.append("Adjust these ratios to match your personal priorities.")
        return "\n".join(lines)

    if any(p in lower for p in DEPOSIT_WHERE):
        if risk == "low":
            recs = (
                "  1. High-interest savings account (HISA) — ING, Macquarie, UBank\n"
                "     Pros: instant access, APRA protected up to $250k\n\n"
                "  2. Term deposit — lock for 3–12 months for a fixed rate\n"
                "     Pros: guaranteed return, no market risk"
            )
        elif risk == "medium":
            recs = (
                "  1. HISA for money needed within 2 years\n"
                "  2. Term deposits for money needed in 1–3 years\n"
                "  3. Defensive bond ETF (e.g. IAF, VAF) for money 3+ years away"
            )
        else:
            recs = (
                "  1. HISA for near-term portion (< 2 years away)\n"
                "  2. Balanced ETF for longer-term portion (3+ years away)\n"
                "     Note: never use volatile assets for a near-term goal"
            )
        return "\n".join([
            f"Where to Keep Your Savings (risk profile: {risk}):",
            "",
            recs,
            "",
            "For a house deposit you plan to use in < 3 years: always use capital-safe accounts.",
        ])

    if debt > 0 and any(re.search(p, lower) for p in DEBT_TIME_PATTERNS):
        if surplus <= 0:
            return (
                f"With income ${income:,.2f} and expenses ${expenses:,.2f}, "
                f"your surplus is ${surplus:,.2f}. Reduce expenses to free up cash for debt repayment."
            )
        debt_payment = surplus * 0.4
        months_      = debt / debt_payment
        lines = [
            "Debt Payoff Timeline:",
            "",
            f"  Total debt:               ${debt:,.2f}",
            f"  Suggested monthly payment: 40% of ${surplus:,.2f} = ${debt_payment:,.2f}/month",
            f"  Estimated payoff: {months_:.0f} months ({months_/12:.1f} years)",
            "",
            "  Speed it up:",
            f"  - 50%: ${surplus*0.5:,.2f}/month → {debt/(surplus*0.5):.0f} months",
            f"  - 60%: ${surplus*0.6:,.2f}/month → {debt/(surplus*0.6):.0f} months",
            "",
            "  Method: Avalanche — pay minimums on all debts, put all extra onto highest-rate first.",
        ]
        return "\n".join(lines)

    if any(p in lower for p in AVALANCHE_PATTERNS):
        return (
            "Avalanche vs Snowball Debt Repayment:\n\n"
            "  Avalanche method:\n"
            "  - Pay minimums on all debts\n"
            "  - Direct all extra to the HIGHEST INTEREST RATE debt first\n"
            "  - Mathematically optimal — minimises total interest paid\n\n"
            "  Snowball method:\n"
            "  - Pay minimums on all debts\n"
            "  - Direct all extra to the SMALLEST BALANCE debt first\n"
            "  - Quick psychological wins — great for motivation\n"
            "  - Costs more in total interest than avalanche\n\n"
            f"  Recommendation for you ({risk} risk, ${surplus:,.2f}/month surplus):\n"
            f"  Use the Avalanche method to save the most interest on ${debt:,.2f} of debt."
        )

    return None


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def clean_model_output(text: str) -> str:
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _call_llm(messages_list, max_new_tokens=80):
    if not HAS_TORCH or model is None or tokenizer is None:
        return None
    try:
        text   = tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=600)
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return clean_model_output(reply) or None
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


def generate_llm_intro(profile):
    goal    = profile.get("financial_goal", "improve finances")
    risk    = profile.get("risk_tolerance", "medium")
    horizon = profile.get("investment_horizon", "medium term")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"User goal: {goal}. Risk: {risk}. Horizon: {horizon}. "
            "Write one warm friendly sentence (max 25 words) introducing their personalised plan. "
            "No markdown. No numbers."
        )},
    ]
    reply = _call_llm(messages, max_new_tokens=60)
    if reply:
        first = reply.split(".")[0].strip()
        if 5 < len(first) <= 200:
            return first + "."
    return None


def generate_final_advice(profile):
    goal      = str(profile.get("financial_goal") or "improve your finances")
    llm_intro = generate_llm_intro(profile)
    if not llm_intro:
        llm_intro = (
            f"I understand your desire for steady growth with minimal risk, "
            f"and I'll tailor a balanced strategy just for you."
        )
    return f"{llm_intro}\n\n{build_rule_based_advice(profile)}"


# ---------------------------------------------------------------------------
# Main follow-up dispatcher — handles ALL post-profile questions
# ---------------------------------------------------------------------------

def handle_followup(user_input: str, profile: dict) -> str:
    # 1. Investment advice for a specific stock (e.g. "should I invest in NVDA?")
    if is_investment_advice_query(user_input):
        ticker = extract_ticker_from_text(user_input)
        if ticker and HAS_YF:
            price_data = fetch_yfinance_response(ticker, profile=None)  # raw price data
            return build_stock_investment_advice(ticker, profile, price_data)
        elif ticker:
            # No yfinance, but we can still give profile-based advice
            return build_stock_investment_advice(ticker, profile, f"[Live price data unavailable for {ticker.upper()}]")
        else:
            return (
                "I can give you personalised investment advice for specific stocks or ETFs!\n\n"
                "Just mention the ticker, for example:\n"
                "  - 'Should I invest in NVDA?'\n"
                "  - 'Is VAS a good ETF for my profile?'\n"
                "  - 'Should I buy Apple shares?'\n"
            )

    # 2. Pure price lookup (e.g. "$AAPL price")
    if is_stock_price_query(user_input):
        ticker = extract_ticker_from_text(user_input)
        if ticker:
            return fetch_yfinance_response(ticker, profile)
        return (
            "I can look up real-time stock or ETF prices!\n\n"
            "Include the ticker symbol, for example:\n"
            "  - '$AAPL price'\n"
            "  - 'BHP.AX stock'\n"
            "  - 'NVDA price'\n\n"
            "For Australian ASX stocks, add '.AX' (e.g. CBA.AX, WBC.AX)."
        )

    # 3. Vague request — show menu
    if is_vague_followup(user_input):
        return get_clarifying_followup_menu(profile)

    # 4. Specific rule-based follow-up (timeline, budgeting, debt, etc.)
    rule_answer = handle_specific_followup_rule_based(user_input, profile)
    if rule_answer:
        return rule_answer

    # 5. LLM fallback
    m = calculate_financial_metrics(profile)
    messages = [
        {"role": "system", "content": (
            "You are a helpful financial advisor. Answer in 3–5 plain-text sentences. "
            "Use ONLY the numbers in the profile below. No markdown. No invented figures. "
            "If you don't know, say so and suggest the user rephrase."
        )},
        {"role": "user", "content": (
            f"User profile: income ${m['income']:,.2f}/mo, expenses ${m['expenses']:,.2f}/mo, "
            f"surplus ${m['surplus']:,.2f}/mo, savings ${m['savings']:,.2f}, debt ${m['debt']:,.2f}. "
            f"Goal: {profile.get('financial_goal')}. Risk: {profile.get('risk_tolerance')}. "
            f"Horizon: {profile.get('investment_horizon')}.\n\n"
            f"Question: {user_input.strip()}"
        )},
    ]
    reply = _call_llm(messages, max_new_tokens=150)
    if reply and len(reply) > 20:
        return reply

    # 6. Final fallback — show full plan
    return (
        "Here is a reminder of your full personalised financial plan:\n\n"
        + build_rule_based_advice(profile)
    )


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------

SMALL_TALK = {
    "how are you":       "I'm doing great, thanks for asking! Ready to help you with your finances. What would you like help with today?",
    "how are you doing": "I'm doing well! Ready to help with budgeting, saving, or investing. What's on your mind?",
    "what can you do":   (
        "I can help you with:\n"
        "- Budgeting and expense tracking\n"
        "- Building an emergency fund\n"
        "- Managing and reducing debt\n"
        "- Saving for a goal (house, travel, etc.)\n"
        "- Beginner investing advice\n"
        "- Whether to invest in specific stocks or ETFs (e.g. 'Should I invest in NVDA?')\n"
        "- Real-time stock and ETF prices (e.g. '$AAPL price')\n\n"
        "Just tell me what you need!"
    ),
    "who are you":  "I'm your personal AI Financial Advisor, here to help with budgeting, saving, debt, and investing.",
    "thank you":    "You're welcome! Feel free to ask any other financial questions.",
    "thanks":       "Happy to help! Let me know if you have more questions.",
    "ok":           "Great! Is there anything else you'd like to know about your finances?",
    "okay":         "Got it! Feel free to ask any follow-up questions.",
}


def is_greeting(text):
    c         = text.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "hiya", "yo"]
    return c in greetings or any(c.startswith(g + " ") or c.startswith(g + ",") or c == g for g in greetings)


def get_small_talk_response(text):
    return SMALL_TALK.get(text.lower().strip().rstrip("?!."))


def has_financial_topic(text):
    lower    = text.lower()
    triggers = [
        "save", "saving", "budget", "budgeting", "debt", "loan",
        "finance", "financial", "money", "income", "expenses",
        "invest", "investment", "retirement", "emergency fund",
        "house", "deposit", "buy a house", "wealth", "risk",
        "help me save", "help saving", "financial advice",
        "gold", "stocks", "shares", "etf", "fund",
        "advice", "suggest", "recommend", "plan", "need help",
        "stock price", "share price", "ticker",
    ]
    return any(t in lower for t in triggers)


def is_affirmative(text):
    return text.lower().strip() in {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "right", "confirm", "please", "y"}


def is_negative(text):
    return text.lower().strip() in {"no", "nope", "nah", "cancel", "wrong", "incorrect", "don't", "dont", "n"}


# ---------------------------------------------------------------------------
# Clarification helpers
# ---------------------------------------------------------------------------

def maybe_needs_annual_clarification(value: float, field: str) -> bool:
    if field in ("monthly_income", "monthly_expenses") and value >= 50_000:
        return True
    return False


def build_clarification_prompt(field: str, value: float) -> str:
    if field in ("monthly_income", "monthly_expenses"):
        monthly  = round(value / 12, 2)
        label    = FIELD_LABELS[field]
        return (
            f"Just to confirm — you entered ${value:,.2f} as your {label}.\n\n"
            f"  Is this your MONTHLY {label}? (${value:,.2f}/month)\n"
            f"  Or is this your ANNUAL {label}? (which would be ${monthly:,.2f}/month)\n\n"
            "Please reply 'monthly' or 'annual'."
        )
    label = FIELD_LABELS[field]
    return f"Just to confirm — is ${value:,.2f} your {label}? (yes / no)"


# ---------------------------------------------------------------------------
# Main reply builder
# ---------------------------------------------------------------------------

def create_assistant_reply(user_input, profile):
    chat_history      = get_chat_history()
    guided_mode       = get_guided_mode()
    profile_completed = get_profile_completed()
    pending_corr      = get_pending_correction()
    pending_clarify   = get_pending_clarify()

    # ── 0a. Pending clarify (monthly vs annual, or confirm debt) ─────────────
    if pending_clarify:
        field = pending_clarify["field"]
        value = pending_clarify["value"]
        lower = user_input.lower().strip()

        if field in ("monthly_income", "monthly_expenses"):
            if "annual" in lower or "year" in lower or "yearly" in lower or "pa" in lower:
                monthly_val = round(value / 12, 2)
                profile[field] = monthly_val
                save_session_profile(profile)
                set_pending_clarify(None)
                ack  = f"Got it! I've recorded your {FIELD_LABELS[field]} as ${monthly_val:,.2f}/month (${value:,.2f} ÷ 12)."
                next_q = get_next_missing_question(profile)
                if next_q:
                    return f"{ack}\n\n{next_q}"
                set_profile_completed(True)
                return f"{ack}\n\n{generate_final_advice(profile)}"
            elif "month" in lower or is_affirmative(user_input):
                profile[field] = value
                save_session_profile(profile)
                set_pending_clarify(None)
                ack  = f"Got it! I've recorded your {FIELD_LABELS[field]} as ${value:,.2f}/month."
                next_q = get_next_missing_question(profile)
                if next_q:
                    return f"{ack}\n\n{next_q}"
                set_profile_completed(True)
                return f"{ack}\n\n{generate_final_advice(profile)}"
            else:
                return f"Please reply 'monthly' (if ${value:,.2f} is your monthly figure) or 'annual' (if it's your yearly figure)."

        elif field == "current_debt":
            if is_affirmative(user_input):
                profile[field] = value
                save_session_profile(profile)
                set_pending_clarify(None)
                ack  = f"Got it! I've recorded your total debt as ${value:,.2f}."
                next_q = get_next_missing_question(profile)
                if next_q:
                    return f"{ack}\n\n{next_q}"
                set_profile_completed(True)
                return f"{ack}\n\n{generate_final_advice(profile)}"
            elif is_negative(user_input):
                set_pending_clarify(None)
                return f"No problem — let me ask again.\n\n{FIELD_QUESTIONS['current_debt']}"
            else:
                return f"Please reply 'yes' to confirm your total debt is ${value:,.2f}, or 'no' to re-enter."

    # ── 0b. Pending correction confirmation ──────────────────────────────────
    if pending_corr:
        field, new_value = pending_corr["field"], pending_corr["value"]
        if is_affirmative(user_input):
            profile[field] = new_value
            save_session_profile(profile)
            set_pending_correction(None)
            label   = FIELD_LABELS[field]
            display = f"${new_value:,.2f}" if isinstance(new_value, (int, float)) else str(new_value)
            ack     = f"Got it! I've updated your {label} to {display}."
            if get_profile_completed():
                return f"{ack}\n\nHere is your updated recommendation:\n\n{generate_final_advice(profile)}"
            next_q = get_next_missing_question(profile)
            if next_q:
                return f"{ack}\n\n{next_q}"
            set_profile_completed(True)
            return f"{ack}\n\n{generate_final_advice(profile)}"
        elif is_negative(user_input):
            set_pending_correction(None)
            return f"No problem! Your {FIELD_LABELS[field]} remains unchanged. What would you like to do?"
        else:
            return f"Please reply 'yes' to confirm updating your {FIELD_LABELS[field]}, or 'no' to keep it."

    # ── 1. Small talk ─────────────────────────────────────────────────────────
    st = get_small_talk_response(user_input)
    if st:
        return st

    if is_greeting(user_input) and len(chat_history) <= 2:
        return (
            "Hi there! I'm your personal AI Financial Advisor. I'm here to help with "
            "budgeting, saving, managing debt, building an emergency fund, and beginner investing.\n\n"
            "I can also:\n"
            "  - Look up real-time stock and ETF prices — e.g. '$AAPL price' or 'BHP.AX stock'\n"
            "  - Give personalised advice on whether to invest in a stock — e.g. 'Should I invest in NVDA?'\n\n"
            "Just tell me what kind of financial advice you're looking for! For example:\n"
            "  - 'I want investment advice'\n"
            "  - 'Help me save more money'\n"
            "  - 'I need help with debt'\n"
            "  - '$TSLA price'\n\n"
            "What financial goal can I help you with today?"
        )

    # ── 2. Post-profile follow-up (any question after profile is complete) ────
    if guided_mode and profile_completed:
        return handle_followup(user_input, profile)

    # ── 3. Pure market query before guided mode starts ────────────────────────
    if is_stock_price_query(user_input) and not guided_mode:
        ticker = extract_ticker_from_text(user_input)
        if ticker:
            return fetch_yfinance_response(ticker, None)
        return (
            "I can look up real-time stock prices!\n\n"
            "Include the ticker symbol, for example:\n"
            "  - '$AAPL price'\n"
            "  - 'BHP.AX stock'\n"
            "  - 'NVDA price'\n\n"
            "Or tell me your financial goal and I'll build a personalised plan!"
        )

    # ── 4. Correction detection ───────────────────────────────────────────────
    if guided_mode:
        corr_field, corr_value = detect_correction(user_input, profile)
        if corr_field is not None:
            label       = FIELD_LABELS[corr_field]
            old_val     = profile.get(corr_field)
            new_display = f"${corr_value:,.2f}" if isinstance(corr_value, (int, float)) else str(corr_value)
            if old_val is not None:
                old_display = f"${old_val:,.2f}" if isinstance(old_val, (int, float)) else str(old_val)
                set_pending_correction({"field": corr_field, "value": corr_value})
                return (
                    f"I see you'd like to update your {label} from {old_display} to {new_display}. "
                    "Can you confirm? (yes / no)"
                )
            else:
                profile[corr_field] = corr_value
                save_session_profile(profile)
                msg_parts = [f"Got it! I've recorded your {label} as {new_display}."]
                next_q    = get_next_missing_question(profile)
                if next_q:
                    msg_parts.append(next_q)
                    return "\n\n".join(msg_parts)
                set_profile_completed(True)
                return "\n\n".join(msg_parts + [generate_final_advice(profile)])

    # ── 5. Start guided mode ──────────────────────────────────────────────────
    if not guided_mode and has_financial_topic(user_input):
        set_guided_mode(True)
        save_session_profile(profile)
        next_q = get_next_missing_question(profile)
        if next_q:
            return (
                "Great! I'd love to help you with that. To give you the best personalised "
                "advice, I need a few details about your finances. Let's go one at a time.\n\n"
                + next_q
            )
        set_profile_completed(True)
        return generate_final_advice(profile)

    # ── 6. Collect remaining profile fields ───────────────────────────────────
    if guided_mode and not profile_completed:
        current_field = get_current_asking_field(profile)
        if current_field is None:
            set_profile_completed(True)
            return (
                "Thank you! I now have all the details I need. "
                "Here is your personalised financial recommendation:\n\n"
                + generate_final_advice(profile)
            )

        value, needs_clarify = extract_field_answer(user_input, current_field, profile)

        if value is None:
            field_hint = _get_invalid_hint(current_field, user_input)
            return f"{field_hint}\n\n{FIELD_QUESTIONS[current_field]}"

        if maybe_needs_annual_clarification(value, current_field):
            set_pending_clarify({"field": current_field, "value": value})
            return build_clarification_prompt(current_field, value)

        if current_field == "current_debt" and value > 0:
            set_pending_clarify({"field": current_field, "value": value})
            return (
                f"Just to confirm — is your total current debt ${value:,.2f}?\n\n"
                "This includes all loans, credit cards, BNPL, and any other money owed.\n"
                "Reply 'yes' to confirm, or 'no' to re-enter."
            )

        profile[current_field] = value
        save_session_profile(profile)

        if current_field in ("monthly_income", "monthly_expenses", "current_savings"):
            ack = f"Got it — {FIELD_LABELS[current_field]}: ${float(value):,.2f}"
        elif current_field == "current_debt":
            ack = "Got it — no debt recorded." if value == 0 else f"Got it — total debt: ${float(value):,.2f}"
        else:
            ack = f"Got it — {FIELD_LABELS[current_field]}: {value}"

        next_q = get_next_missing_question(profile)
        if next_q:
            return f"{ack}\n\n{next_q}"
        set_profile_completed(True)
        return (
            f"{ack}\n\n"
            "Thank you! I now have all the details I need. "
            "Here is your personalised financial recommendation:\n\n"
            + generate_final_advice(profile)
        )

    # ── 7. Default fallback ───────────────────────────────────────────────────
    return (
        "Hi! I'm your personal AI Financial Advisor. I can help with saving, budgeting, "
        "debt management, beginner investing, and real-time stock prices.\n\n"
        "Tell me your financial goal to get started, or ask e.g. '$AAPL price' or 'Should I invest in NVDA?'!"
    )


def _get_invalid_hint(field: str, user_input: str) -> str:
    lower = user_input.lower().strip()
    if field == "monthly_income":
        if any(w in lower for w in ["none", "no income", "unemployed", "zero"]):
            return "Please enter your best estimate, or type '0' if you truly have no income."
        return f"I couldn't read a number from '{user_input}'. Please enter the dollar amount, e.g. '5000' or '$5,000'."
    if field == "monthly_expenses":
        return f"I couldn't read a number from '{user_input}'. Please enter your total monthly expenses, e.g. '3500'."
    if field == "current_savings":
        return f"I couldn't read a number from '{user_input}'. Please enter your total savings, e.g. '10000' or '$10k'."
    if field == "current_debt":
        return f"I couldn't understand '{user_input}'. Please enter your total debt (e.g. '5000'), or type 'none' if you have no debt."
    if field == "financial_goal":
        return f"I didn't catch a clear goal from '{user_input}'. Please describe your goal in a few words, e.g. 'invest in stocks' or 'buy a house'."
    if field == "risk_tolerance":
        return "Please reply with 'low', 'medium', or 'high' to describe your risk tolerance."
    if field == "investment_horizon":
        return "Please reply with 'short term', 'medium term', or 'long term'."
    return f"I didn't understand '{user_input}'. Please try again."


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Financial Advisor</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; display: flex;
      justify-content: center; align-items: center;
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #1f2937;
    }
    .container {
      width: 100%; max-width: 960px; margin: 20px; padding: 30px;
      background: #ffffff; border-radius: 24px;
      box-shadow: 0 24px 80px rgba(0,0,0,0.12);
    }
    h1 { margin: 0 0 12px; font-size: 2.5rem; text-align: center; }
    .subtitle { margin: 0 0 8px; text-align: center; color: #6b7280; font-size: 1rem; }
    .yfin-badge {
      text-align: center; color: #10b981; font-size: 0.85rem;
      margin-bottom: 20px; font-weight: 600;
    }
    .toolbar {
      display: flex; justify-content: space-between; align-items: center;
      background: #f8fafc; border: 1px solid #e2e8f0;
      border-radius: 16px; padding: 16px 20px; margin-bottom: 24px;
    }
    .status { color: #10b981; font-weight: 600; }
    .error  { color: #ef4444; font-size: 0.95rem; margin-top: 6px; }
    .clear-btn {
      background: #ef4444; color: #fff; border: none;
      border-radius: 12px; padding: 10px 18px;
      text-decoration: none; font-weight: 600; cursor: pointer;
    }
    .chat-box {
      min-height: 400px; max-height: 620px; overflow-y: auto;
      padding: 16px; background: #f8fafc;
      border-radius: 20px; border: 1px solid #e2e8f0; margin-bottom: 20px;
    }
    .message {
      padding: 12px 16px; border-radius: 18px; margin-bottom: 10px;
      max-width: 90%; box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      line-height: 1.7; white-space: pre-wrap; font-size: 0.96rem;
    }
    .message.user {
      margin-left: auto;
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: #fff; font-weight: 500;
    }
    .message.assistant {
      margin-right: auto; background: #fff;
      color: #1f2937; border: 1px solid #e2e8f0;
      font-family: 'Courier New', monospace; font-size: 0.90rem;
    }
    .input-row { display: flex; gap: 16px; align-items: center; }
    input[type=text] {
      flex: 1; padding: 16px 20px; border: 2px solid #e2e8f0;
      border-radius: 999px; font-size: 1rem; outline: none;
    }
    input[type=text]:focus {
      border-color: #667eea;
      box-shadow: 0 0 0 4px rgba(102,126,234,0.12);
    }
    button[type=submit] {
      background: linear-gradient(135deg, #667eea, #764ba2);
      border: none; border-radius: 999px; color: #fff;
      padding: 16px 28px; font-size: 1rem; cursor: pointer; font-weight: 700;
    }
    @media (max-width: 768px) {
      .container { padding: 20px; }
      h1 { font-size: 2rem; }
      .chat-box { min-height: 320px; }
      .input-row { flex-direction: column; }
    }
  </style>
  <script>
    window.addEventListener('load', function() {
      var box = document.querySelector('.chat-box');
      if (box) box.scrollTop = box.scrollHeight;
    });
  </script>
</head>
<body>
  <div class="container">
    <h1>💰 AI Financial Advisor</h1>
    <p class="subtitle">Powered by rule-based logic + Qwen 2.5 1.5B LoRA (optional)</p>
    <p class="yfin-badge">📈 Live market data via yfinance · Ask '$AAPL price' or 'Should I invest in NVDA?'</p>
    <div class="toolbar">
      <div>
        <div class="status">{{ status }}</div>
        {% if error %}<div class="error">⚠️ {{ error }}</div>{% endif %}
      </div>
      <a href="{{ url_for('clear') }}" class="clear-btn">🗑 Clear Chat</a>
    </div>
    <div class="chat-box" id="chatBox">
      {% for message in messages %}
        <div class="message {{ message.role }}">{{ message.content | e }}</div>
      {% endfor %}
    </div>
    <form method="post" action="{{ url_for('index') }}">
      <div class="input-row">
        <input name="message" type="text" placeholder="Type your message, ask '$AAPL price', or 'Should I invest in NVDA?'..." autocomplete="off" required autofocus>
        <button type="submit">Send ➤</button>
      </div>
    </form>
  </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    initialize_session_if_needed()
    profile  = get_session_profile()
    messages = get_chat_history()
    error    = None

    if request.method == "POST":
        user_input = request.form.get("message", "").strip()
        if not user_input:
            messages.append({"role": "assistant", "content": "Please enter a message so I can help you."})
        else:
            messages.append({"role": "user", "content": user_input})
            answer = create_assistant_reply(user_input, profile)
            messages.append({"role": "assistant", "content": answer})
            save_chat_history(messages)

    if using_lora:
        status = "✅ LoRA fine-tuned adapter loaded successfully."
    elif model is not None:
        status = "⚠️ Base model loaded (LoRA adapter not found)."
    elif HAS_TORCH:
        status = "❌ Model initialization failed — running in rule-based mode."
        error  = load_error
    else:
        status = "ℹ️ Running in rule-based mode (PyTorch not installed)."
        error  = load_error

    return render_template_string(HTML_TEMPLATE, messages=messages, status=status, error=error)


@app.route("/clear")
def clear():
    reset_session()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)