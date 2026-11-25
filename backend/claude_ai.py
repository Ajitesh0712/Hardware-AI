import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter
import re
from backend.use_cases import detect_use_cases

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")
_GEMINI_DISABLED = False


def has_gemini_key():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(key and not key.lower().startswith(("your_", "replace", "paste_")))

SUPPORTED_CATEGORIES = {
    "laptop", "mobile", "headphones", "tablet", "monitor",
    "keyboard", "mouse", "speaker", "camera", "smartwatch", "television",
    "printer", "networking", "power_banks", "storage", "charger", "computer_components",
}


def infer_category_from_text(text: str):
    """Infer an explicit product type locally so API failure cannot erase it."""
    query = str(text or "").lower()
    checks = [
        (r"\b(?:graphics? cards?|video cards?)\b", "computer_components"),
        (r"\b(?:laptops?|notebooks?|macbooks?|chromebooks?)\b", "laptop"),
        (r"\b(?:smart\s*tvs?|tvs?|televisions?|oled\s*tvs?|qled\s*tvs?)\b", "television"),
        (r"\b(?:printers?|scanners?)\b", "printer"),
        (r"\b(?:routers?|wi[ -]?fi extenders?|modems?)\b", "networking"),
        (r"\b(?:power banks?|portable chargers?)\b", "power_banks"),
        (r"\b(?:external ssds?|external hard drives?|memory cards?|usb drives?)\b", "storage"),
        (r"\b(?:chargers?|charging cables?)\b", "charger"),
        (r"\b(?:ear ?buds?|earphones?|headphones?|headsets?|airpods?|galaxy buds?)\b", "headphones"),
        (r"\b(?:tablets?|ipad|galaxy tab)\b", "tablet"),
        (r"\b(?:monitors?|projectors?|displays?)\b", "monitor"),
        (r"\bkeyboards?\b", "keyboard"),
        (r"\bmice\b|\bmouse\b", "mouse"),
        (r"\b(?:speakers?|soundbars?)\b", "speaker"),
        (r"\b(?:smart ?watches?|fitness bands?)\b", "smartwatch"),
        (r"\b(?:watches?|fitbit)\b", "smartwatch"),
        (r"\bcameras?\b", "camera"),
        (r"\b(?:smartphones?|mobiles?|phones?|cell ?phones?|iphone)\b", "mobile"),
        (r"\b(?:galaxy\s+[azs]\s?\d{1,2}|pixel\s+\d{1,2}|oneplus\s+\d{1,2}|redmi\s+(?:note\s+)?\d{1,2}|poco\s+[a-z]\d{1,2})\b", "mobile"),
    ]
    for pattern, category in checks:
        if re.search(pattern, query):
            return category
    # In this shopping assistant, a bare RTX/GTX/RX model search means a
    # laptop search unless the user explicitly asked for a graphics card.
    if re.search(r"\b(?:rtx|gtx|geforce|radeon|rx)\s*\d{3,4}\b", query):
        return "laptop"
    return None


def _extract_hardware_models(text: str) -> list[str]:
    """Keep explicitly requested model numbers as hard match requirements."""
    query = str(text or "").lower()
    patterns = (
        r"\b(?:geforce\s+)?(?:rtx|gtx)\s*\d{3,4}\s*(?:ti|super)?\b",
        r"\b(?:radeon\s+)?rx\s*\d{3,4}\s*(?:xt|xtx)?\b",
        r"\b(?:core\s+)?i[3579]-?\d{4,5}[a-z]{0,2}\b",
        r"\bryzen\s+[3579]\s+\d{4,5}[a-z]{0,2}\b",
        r"\b(?:intel\s+)?core\s+ultra\s+[579]\b",
        r"\bm[1-5](?:\s+(?:pro|max|ultra))?\b",
        r"\biphone\s+\d{1,2}(?:\s+(?:pro|max|plus|mini)){0,2}\b",
        r"\bgalaxy\s+[azs]\s?\d{1,2}(?:\s+(?:fe|plus|ultra))?\b",
        r"\bpixel\s+\d{1,2}(?:\s+(?:pro|xl|a))?\b",
        r"\boneplus\s+\d{1,2}(?:r)?\b",
    )
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, query, re.I):
            model = re.sub(r"\s+", " ", match.group(0)).strip()
            model = re.sub(r"^(?:geforce|radeon)\s+", "", model)
            if model not in found:
                found.append(model)
    return found


def _infer_use_case(text: str) -> str:
    query = str(text or "").lower()
    if re.search(r"\b(?:gaming|games?|gamer)\b", query):
        return "gaming"
    if re.search(r"\b(?:3d|rendering|cad)\b", query):
        return "3d rendering"
    if re.search(r"\b(?:video editing|editing videos|video editor)\b", query):
        return "video editing"
    if re.search(r"\b(?:coding|programming|developer|software development)\b", query):
        return "coding"
    if re.search(r"\b(?:design|graphic design|photoshop)\b", query):
        return "design"
    if re.search(r"\b(?:student|school|college|university)\b", query):
        return "student"
    if re.search(r"\b(?:business|office|work)\b", query):
        return "business"
    return "general"


def _local_priority_specs(text: str) -> dict:
    query = str(text or "").lower()
    models = _extract_hardware_models(query)
    ram = re.search(r"\b(\d{1,3})\s*gb\s*ram\b", query)
    storage = re.search(r"\b(\d{1,3})\s*(tb|gb)\s*(?:ssd|storage|hdd|nvme)\b", query)
    return {
        "ram_gb": int(ram.group(1)) if ram else None,
        "storage_gb": (int(storage.group(1)) * (1024 if storage.group(2) == "tb" else 1)) if storage else None,
        "gpu_required": any(re.search(r"\b(?:rtx|gtx|geforce|radeon|rx)\b", model) for model in models),
        "gpu_model": next((m for m in models if re.search(r"\b(?:rtx|gtx|geforce|radeon|rx)\b", m)), None),
        "hardware_models": models,
        "display_size": None,
        "battery_life": bool(re.search(r"\b(?:long battery|battery life|all.day battery)\b", query)),
        "portability": bool(re.search(r"\b(?:lightweight|portable|travel)\b", query)),
    }


def _budget_number(value: str) -> float:
    amount = float(value.replace(",", ""))
    return amount * 1_000 if amount < 1_000 and re.search(r"\bk\b|thousand", value, re.I) else amount


def parse_budget_limits(text: str) -> dict:
    """Extract explicit INR budget bounds, including ranges, without Gemini."""
    query = str(text or "").lower().replace("₹", "rs ")
    number = r"(?:rs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|lakh|lac)?"

    def to_amount(match, group=1):
        raw, scale = match.group(group), (match.group(group + 1) or "").lower()
        amount = float(raw.replace(",", ""))
        if scale in {"k", "thousand"}:
            amount *= 1_000
        elif scale in {"lakh", "lac"}:
            amount *= 100_000
        return amount

    range_match = re.search(
        rf"\b(?:between|from)\s+{number}\s+(?:and|to|-)\s+{number}", query
    )
    if range_match:
        values = [to_amount(range_match, 1), to_amount(range_match, 3)]
        return {"budget_min": min(values), "budget_max": max(values)}

    dash_range = re.search(rf"{number}\s*(?:-|to)\s*{number}", query)
    if dash_range:
        values = [to_amount(dash_range, 1), to_amount(dash_range, 3)]
        return {"budget_min": min(values), "budget_max": max(values)}

    maximum = re.search(rf"\b(?:under|below|less than|up to|upto|max(?:imum)?|within)\s+{number}", query)
    if maximum:
        return {"budget_min": None, "budget_max": to_amount(maximum)}

    minimum = re.search(rf"\b(?:over|above|more than|at least|min(?:imum)?)\s+{number}", query)
    if minimum:
        return {"budget_min": to_amount(minimum), "budget_max": None}
    return {"budget_min": None, "budget_max": None}


def _remove_budget_words(text: str) -> str:
    query = str(text or "")
    amount = r"(?:₹|rs\.?|inr)?\s*\d[\d,]*(?:\.\d+)?\s*(?:k|thousand|lakh|lac)?"
    patterns = [
        rf"\b(?:between|from)\s+{amount}\s+(?:and|to|-)\s+{amount}",
        rf"\b(?:under|below|less than|up to|upto|max(?:imum)?|within|over|above|more than|at least|min(?:imum)?)\s+{amount}",
        rf"{amount}\s*(?:-|to)\s*{amount}",
    ]
    for pattern in patterns:
        query = re.sub(pattern, " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query).strip(" ,.-")
    return query or str(text or "").strip()


def parse_user_requirements(user_query: str) -> dict:
    global _GEMINI_DISABLED

    lexical_category = infer_category_from_text(user_query)
    lexical_budget = parse_budget_limits(user_query)
    local_specs = _local_priority_specs(user_query)
    local_use_case = _infer_use_case(user_query)

    # Keep core search usable and fast when Gemini is unconfigured or unavailable.
    if not has_gemini_key() or _GEMINI_DISABLED:
        return {
            **lexical_budget,
            "category": lexical_category,
            "use_case": local_use_case,
            "use_cases": detect_use_cases(user_query),
            "brand_preference": None,
            "keywords": [],
            "intent": "none",
            "modifiers": [],
            "priority_specs": local_specs,
            "required_models": local_specs["hardware_models"],
            "search_query": _remove_budget_words(user_query),
        }

    prompt = f"""
You are an AI assistant that extracts structured shopping intent from user queries.

Return ONLY valid JSON. No explanation. No markdown.

Format:
{{
    "budget_min": null,
    "budget_max": null,
    "category": null,
    "use_case": "general",
    "brand_preference": null,
    "keywords": [],
    "intent": "increase_budget | decrease_budget | set_budget | none",
    "modifiers": [],
    "priority_specs": {{
        "ram_gb": null,
        "storage_gb": null,
        "gpu_required": false,
        "display_size": null,
        "battery_life": false,
        "portability": false
    }},
    "search_query": ""
}}

Rules:
- Do NOT put full query inside category
- Category must ONLY be one of:
  laptop, mobile, headphones, tablet, monitor,
  keyboard, mouse, speaker, camera, smartwatch

Query: {user_query}
"""

    try:

        response = model.generate_content(prompt)

        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]

            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())

        # =========================
        # SAFE CATEGORY VALIDATION
        # =========================

        valid_categories = [
            "laptop",
            "mobile",
            "headphones",
            "tablet",
            "monitor",
            "keyboard",
            "mouse",
            "speaker",
            "camera",
            "smartwatch"
        ]

        if parsed.get("category") not in valid_categories:
            parsed["category"] = None
        # Explicit product words and numeric budget bounds are deterministic.
        if lexical_category:
            parsed["category"] = lexical_category
        if lexical_budget["budget_min"] is not None or lexical_budget["budget_max"] is not None:
            parsed.update(lexical_budget)

        # =========================
        # SAFE DEFAULTS
        # =========================

        parsed.setdefault("budget_min", None)
        parsed.setdefault("budget_max", None)
        parsed.setdefault("use_case", "general")
        parsed.setdefault("keywords", [])
        parsed.setdefault("priority_specs", {})
        parsed["use_case"] = local_use_case if local_use_case != "general" else parsed.get("use_case", "general")
        parsed["priority_specs"] = {**parsed.get("priority_specs", {}), **local_specs}
        parsed["required_models"] = local_specs["hardware_models"]
        parsed["use_cases"] = detect_use_cases(user_query)
        parsed["search_query"] = _remove_budget_words(user_query)

        return parsed

    except Exception as e:
        if re.search(r"403|api.?key|unauthori[sz]ed|permission denied", str(e), re.I):
            _GEMINI_DISABLED = True
        print("[Query Parser] Gemini unavailable; using local parsing.")

        # SAFE FALLBACK
        return {
            "budget_min": None,
            "budget_max": None,
            **lexical_budget,
            "category": lexical_category,
            "use_case": local_use_case,
            "use_cases": detect_use_cases(user_query),
            "brand_preference": None,
            "keywords": [],
            "intent": "none",
            "modifiers": [],
            "priority_specs": local_specs,
            "required_models": local_specs["hardware_models"],
            "search_query": _remove_budget_words(user_query)
        }

def generate_recommendation_summary(products: list, user_query: str) -> str:
    if not products:
        return "No products matched the current category and budget filters."
    category = infer_category_from_text(user_query)
    category_names = {
        "laptop": "laptops", "mobile": "phones", "headphones": "headphones",
        "tablet": "tablets", "monitor": "monitors", "keyboard": "keyboards",
        "mouse": "mice", "speaker": "speakers", "camera": "cameras",
        "smartwatch": "watches", "television": "TVs", "printer": "printers",
        "networking": "networking devices", "power_banks": "power banks",
        "storage": "storage devices", "charger": "chargers", "computer_components": "computer parts",
    }
    category_name = category_names.get(category, "products")
    if len(products) == 1 and category_name.endswith("s"):
        category_name = category_name[:-1]
    models = _extract_hardware_models(user_query)
    model_phrase = ""
    if models:
        model = models[0].upper()
        if re.search(r"\b(?:RTX|GTX|RX)\b", model):
            model_phrase = f" with {model} graphics"
        else:
            model_phrase = f" with {model}"
    use_case_names = {
        "gaming": "gaming", "coding": "coding", "design": "design",
        "video editing": "video editing", "3d modelling": "3D work",
        "student": "school or college", "business": "work", "long battery": "long battery life",
    }
    uses = [use_case_names[name] for name in detect_use_cases(user_query) if name in use_case_names]
    use_case_phrase = (" for " + " and ".join(uses)) if uses else ""
    budgets = parse_budget_limits(user_query)
    budget_phrase = ""
    if budgets.get("budget_min") is not None and budgets.get("budget_max") is not None:
        budget_phrase = f" between ₹{budgets['budget_min']:,.0f} and ₹{budgets['budget_max']:,.0f}"
    elif budgets.get("budget_max") is not None:
        budget_phrase = f" under ₹{budgets['budget_max']:,.0f}"
    elif budgets.get("budget_min") is not None:
        budget_phrase = f" above ₹{budgets['budget_min']:,.0f}"
    return (
        f"I found {len(products)} {category_name}{model_phrase}{use_case_phrase}{budget_phrase}. "
        "The closest matches to your request are first. Check the current price and reviews on the store page."
    )



def analyze_product_sentiment(reviews: list) -> dict:

    if not reviews:
        return {
            "pros": [],
            "cons": [],
            "hidden_issues": [],
            "best_for": [],
            "performance_analysis": "Not enough reviews available.",
            "build_quality": "No detailed build insights available.",
            "battery_analysis": "Battery feedback unavailable.",
            "user_sentiment_summary": "Insufficient review data.",
            "long_term_usage": "No long-term usage feedback.",
            "value_for_money": "Unknown",
            "sentiment": "mixed"
        }

    if not has_gemini_key():
        return {
            "pros": [], "cons": [], "hidden_issues": [], "best_for": [],
            "performance_analysis": "Review text was retrieved; automated analysis is unavailable without a Gemini API key.",
            "build_quality": "Not analyzed.", "battery_analysis": "Not analyzed.",
            "user_sentiment_summary": "Review text is available; sentiment analysis is not configured.",
            "long_term_usage": "Not analyzed.", "value_for_money": "Not analyzed.",
            "sentiment": "mixed"
        }

    # =========================
    # USE MORE REVIEWS
    # =========================

    reviews_text = "\n".join(reviews[:25])

    # =========================
    # EXTRACT RECURRING KEYWORDS
    # =========================

    all_text = " ".join(reviews).lower()

    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)

    stopwords = {
        "this", "that", "with", "have", "very",
        "good", "great", "nice", "best", "product",
        "quality", "price", "really", "using",
        "after", "before", "their", "there",
        "would", "could", "should", "about",
        "because", "while", "where", "which",
        "these", "those", "amazing", "awesome"
    }

    filtered = [
        w for w in words
        if w not in stopwords
    ]

    common_words = Counter(filtered).most_common(25)

    keyword_summary = ", ".join([
        word for word, _ in common_words
    ])

    # =========================
    # ADVANCED AI PROMPT
    # =========================

    prompt = f"""
You are an expert electronics review analyst.

Analyze these reviews deeply and intelligently.

IMPORTANT:
Use recurring patterns, repeated complaints,
and common praise points.

Frequent review keywords:
{keyword_summary}

Customer Reviews:
{reviews_text}

Return ONLY valid JSON.

Format:
{{
    "pros": [],
    "cons": [],
    "hidden_issues": [],
    "best_for": [],
    "performance_analysis": "",
    "build_quality": "",
    "battery_analysis": "",
    "user_sentiment_summary": "",
    "long_term_usage": "",
    "value_for_money": "",
    "sentiment": "positive|mixed|negative"
}}

Rules:
- Generate 5–8 meaningful pros
- Generate 4–6 realistic cons
- Mention recurring complaints
- Mention heating/noise/durability if present
- Mention gaming/productivity suitability
- Mention battery observations
- Mention real-world experience
- Avoid generic phrases like:
  "good quality"
  "nice product"
  "great performance"
- Output should feel like expert analysis
- Be specific and insightful
"""

    try:

        response = model.generate_content(prompt)

        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]

            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())

        # =========================
        # SAFE DEFAULTS
        # =========================

        parsed.setdefault("pros", [])
        parsed.setdefault("cons", [])
        parsed.setdefault("hidden_issues", [])
        parsed.setdefault("best_for", [])
        parsed.setdefault("performance_analysis", "")
        parsed.setdefault("build_quality", "")
        parsed.setdefault("battery_analysis", "")
        parsed.setdefault("user_sentiment_summary", "")
        parsed.setdefault("long_term_usage", "")
        parsed.setdefault("value_for_money", "")
        parsed.setdefault("sentiment", "mixed")

        return parsed

    except Exception as e:

        print("[Sentiment Error]", e)

        return {
            "pros": [],
            "cons": [],
            "hidden_issues": [],
            "best_for": [],
            "performance_analysis": "Could not generate advanced analysis.",
            "build_quality": "Unknown",
            "battery_analysis": "Unknown",
            "user_sentiment_summary": "Review analysis unavailable.",
            "long_term_usage": "No detailed insights available.",
            "value_for_money": "Average",
            "sentiment": "mixed"
        }
