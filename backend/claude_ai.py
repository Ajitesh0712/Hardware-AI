"""
claude_ai.py (Gemini version)
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

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
- If user says "cheaper", "lower", → intent = decrease_budget
- If user says "increase", "higher", → intent = increase_budget
- If user specifies price → intent = set_budget
- If no change → intent = none

- Extract modifiers like:
    "gaming", "battery", "performance", "lightweight", "portable"

- Do NOT guess category unless clearly mentioned

Query: {user_query}
"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


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
        return {"pros": ["Good overall performance"], "cons": ["Limited reviews available"], "sentiment": "mixed"}

    reviews_text = "\n".join(reviews[:10])
    prompt = f"""Analyze these product reviews and return ONLY valid JSON with no markdown:
{{"pros": ["pro1", "pro2", "pro3"], "cons": ["con1", "con2"], "sentiment": "positive|mixed|negative"}}
Keep each point under 8 words.

Reviews:
{reviews_text}"""

    response = model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
