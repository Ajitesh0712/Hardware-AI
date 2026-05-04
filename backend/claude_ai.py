import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from collections import Counter
import re

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")


def parse_user_requirements(user_query: str) -> dict:

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

        # =========================
        # SAFE DEFAULTS
        # =========================

        parsed.setdefault("budget_min", None)
        parsed.setdefault("budget_max", None)
        parsed.setdefault("use_case", "general")
        parsed.setdefault("keywords", [])
        parsed.setdefault("priority_specs", {})
        parsed.setdefault("search_query", user_query)

        return parsed

    except Exception as e:

        print("[Claude] Error:", e)

        # SAFE FALLBACK
        return {
            "budget_min": None,
            "budget_max": None,
            "category": None,
            "use_case": "general",
            "brand_preference": None,
            "keywords": [],
            "intent": "none",
            "modifiers": [],
            "priority_specs": {},
            "search_query": user_query
        }

def generate_recommendation_summary(products: list, user_query: str) -> str:
    product_list = "\n".join([
        f"- {p.get('product_title','Unknown')} | ${p.get('product_price','N/A')} | Rating: {p.get('product_star_rating','N/A')}"
        for p in products[:5]
    ])
    prompt = f"""User asked: "{user_query}"
Top products:
{product_list}
Write a concise 2-sentence recommendation summary. Be specific and helpful."""

    response = model.generate_content(prompt)
    return response.text.strip()



def analyze_product_sentiment(reviews: list) -> dict:

    if not reviews:
        return {
            "pros": ["No detailed reviews available"],
            "cons": ["Limited customer feedback"],
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
            "pros": [
                "Reliable overall performance",
                "Positive customer reception",
                "Useful for daily usage"
            ],
            "cons": [
                "Some recurring complaints reported"
            ],
            "hidden_issues": [],
            "best_for": [],
            "performance_analysis": "Could not generate advanced analysis.",
            "build_quality": "Unknown",
            "battery_analysis": "Unknown",
            "user_sentiment_summary": "Mixed customer feedback.",
            "long_term_usage": "No detailed insights available.",
            "value_for_money": "Average",
            "sentiment": "mixed"
        }