"""
app.py
Main Flask backend for HardwareAI Electronic Recommendation System.
"""

import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from claude_ai import parse_user_requirements, generate_recommendation_summary, analyze_product_sentiment
from amazon_api import search_products, get_product_reviews
from data_processor import process_products, products_to_dict_list
from recommendation_engine import recommend, get_match_label
from database import init_db, save_query, save_recommendations, log_interaction, upsert_product

load_dotenv()

app = Flask(__name__, static_folder="../frontend")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize database on startup
try:
    init_db()
except Exception as e:
    print(f"[WARN] DB init failed (check MySQL config): {e}")


# ─── Serve Frontend ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


# ─── Main Recommend Endpoint ──────────────────────────────────────────────────

@app.route("/api/recommend", methods=["POST"])
def recommend_products():
    """
    POST /api/recommend
    Body: { "query": "...", "session_id": "..." }
    Returns: { "products": [...], "summary": "...", "requirements": {...} }
    """
    data = request.get_json()
    if not data or not data.get("query"):
        return jsonify({"error": "Query is required"}), 400

    user_query = data["query"].strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    print(f"[API] New query: '{user_query}' | Session: {session_id}")

    # Step 1: Parse requirements with gemini AI
    try:
        requirements = parse_user_requirements(user_query)
        print(f"[Claude] Parsed: {requirements}")
    except Exception as e:
        print(f"[Claude] Error: {e}")
        requirements = {
            "budget_max": None, "budget_min": None,
            "category": user_query.lower(), "use_case": "general",
            "search_query": user_query, "keywords": [],
            "priority_specs": {}
        }
    ALLOWED_CATEGORIES = [
        "laptop", "mobile", "smartphone", "headphones",
        "earbuds", "tablet", "monitor", "keyboard",
        "mouse", "speaker", "smartwatch", "camera"
    ]

    print("[DEBUG] Requirements:", requirements)

    raw_category = (requirements.get("category") or "").lower()
    query_lower = user_query.lower()

    if "laptop" in raw_category or "laptop" in query_lower:
        category = "laptop"
    elif "phone" in raw_category or "mobile" in raw_category or "phone" in query_lower:
        category = "mobile"
    elif "headphone" in raw_category or "earbud" in raw_category or "headphone" in query_lower:
        category = "headphones"
    elif "tablet" in raw_category or "tablet" in query_lower:
        category = "tablet"
    elif "monitor" in raw_category or "monitor" in query_lower:
        category = "monitor"
    elif "keyboard" in raw_category or "keyboard" in query_lower:
        category = "keyboard"
    elif "mouse" in raw_category or "mouse" in query_lower:
        category = "mouse"
    elif "speaker" in raw_category or "speaker" in query_lower:
        category = "speaker"
    elif "watch" in raw_category or "watch" in query_lower:
        category = "smartwatch"
    elif "camera" in raw_category or "camera" in query_lower:
        category = "camera"
    else:
        category = ""

    print("[DEBUG] Raw:", raw_category, "| Final:", category)

    if not category:
        return jsonify({
            "error": "Only electronic products are supported right now.",
            "examples": [
                "laptop under 60000",
                "best headphones for gym",
                "gaming mouse under 2000"
            ]
        }), 400


    requirements["category"] = category

    # Step 2: Fetch products from Amazon API
    search_query = requirements.get("search_query", user_query)
    budget_max = requirements.get("budget_max")

    try:
        raw_products = search_products(
            query=search_query,
            category=requirements.get("category", "laptop"),
            budget_max=budget_max,
            page=1
        )
        print(f"[Amazon] Fetched {len(raw_products)} products")
    except Exception as e:
        print(f"[Amazon] Error: {e}")
        return jsonify({"error": "Failed to fetch products from Amazon. Check your RapidAPI key."}), 502

    if not raw_products:
        return jsonify({"error": "No products found. Try adjusting your query or budget."}), 404

    # Step 3: Process and clean data
    df = process_products(raw_products, requirements)
    if df.empty:
        return jsonify({"error": "No products matched your filters after processing."}), 404

    # Step 4: Run ML recommendation engine
    ranked_df = recommend(df, requirements, top_n=6)

    # Step 5: Enrich top products with reviews + sentiment
    products_list = products_to_dict_list(ranked_df)
    for product in products_list[:3]:  # Enrich top 3 to save API calls
        asin = product.get("asin")
        if asin:
            try:
                reviews = get_product_reviews(asin)
                sentiment_data = analyze_product_sentiment(reviews)
                product["pros"] = sentiment_data.get("pros", [])
                product["cons"] = sentiment_data.get("cons", [])
                product["sentiment"] = sentiment_data.get("sentiment", "mixed")
                product["reviews"] = reviews[:3]  # Store top 3 reviews
            except Exception as e:
                print(f"[Sentiment] Error for {asin}: {e}")
                product["pros"] = ["Good value for money", "Solid performance"]
                product["cons"] = ["Limited accessories included"]
                product["sentiment"] = "mixed"
        else:
            product["pros"] = []
            product["cons"] = []
            product["sentiment"] = "mixed"

        # Add match label
        product["match_label"] = get_match_label(product.get("match_score", 0))

    # Step 6: Generate AI summary
    try:
        summary = generate_recommendation_summary(products_list, user_query)
    except Exception as e:
        print(f"[Claude] Summary error: {e}")
        summary = f"Found {len(products_list)} products matching your requirements."

    # Step 7: Save to database
    try:
        query_id = save_query(user_query, requirements, session_id, len(products_list))
        save_recommendations(query_id, products_list)
        for product in products_list:
            upsert_product(product)
    except Exception as e:
        print(f"[DB] Save error (non-critical): {e}")

    return jsonify({
        "success": True,
        "session_id": session_id,
        "requirements": requirements,
        "summary": summary,
        "products": products_list,
        "total_found": len(products_list)
    })


# ─── Refine Search Endpoint ───────────────────────────────────────────────────

@app.route("/api/refine", methods=["POST"])
def refine_search():
    data = request.get_json()
    if not data or not data.get("query"):
        return jsonify({"error": "Query required"}), 400

    user_query = data["query"]
    original_req = data.get("original_requirements", {})
    session_id = data.get("session_id", str(uuid.uuid4()))

    # ─── STEP 1: Merge AI parsed requirements ───
    try:
        new_req = parse_user_requirements(user_query)
        merged = {**original_req}

        for key, val in new_req.items():
            # 🚫 DO NOT overwrite category with null/empty
            if key == "category" and not val:
                continue

            if val is not None and val != {} and val != []:
                merged[key] = val

    except Exception as e:
        print("[Claude refine error]:", e)
        merged = {**original_req}
        merged["search_query"] = user_query
        new_req = {}

    query_lower = user_query.lower()

    # ─── STEP 2: CATEGORY (only if user mentions) ───
    if "laptop" in query_lower:
        merged["category"] = "laptop"
    elif "phone" in query_lower or "mobile" in query_lower:
        merged["category"] = "mobile"
    elif "headphone" in query_lower or "earbud" in query_lower:
        merged["category"] = "headphones"
    elif "tablet" in query_lower:
        merged["category"] = "tablet"
    elif "monitor" in query_lower:
        merged["category"] = "monitor"
    elif "keyboard" in query_lower:
        merged["category"] = "keyboard"
    elif "mouse" in query_lower:
        merged["category"] = "mouse"
    elif "speaker" in query_lower:
        merged["category"] = "speaker"
    elif "watch" in query_lower:
        merged["category"] = "smartwatch"
    elif "camera" in query_lower:
        merged["category"] = "camera"

    # ✅ Keep previous category if not mentioned
    if not merged.get("category"):
        merged["category"] = original_req.get("category", "laptop")

    # ─── STEP 3: AI INTENT HANDLING ───
    intent = new_req.get("intent")

    if intent == "increase_budget":
        if merged.get("budget_max"):
            merged["budget_max"] += 20000

    elif intent == "decrease_budget":
        if merged.get("budget_max"):
            merged["budget_max"] = max(10000, merged["budget_max"] - 20000)

    elif intent == "set_budget":
        if new_req.get("budget_min"):
            merged["budget_min"] = new_req["budget_min"]
        if new_req.get("budget_max"):
            merged["budget_max"] = new_req["budget_max"]

    # ─── STEP 4: MODIFIER HANDLING ───
    modifiers = new_req.get("modifiers", [])

    if "gaming" in modifiers:
        merged["use_case"] = "gaming"

    if "battery" in modifiers:
        merged.setdefault("priority_specs", {})["battery_life"] = True

    if "performance" in modifiers:
        merged.setdefault("priority_specs", {})["gpu_required"] = True

    if "lightweight" in modifiers or "portable" in modifiers:
        merged.setdefault("priority_specs", {})["portability"] = True

    # ─── STEP 5: BUILD SMART SEARCH QUERY ───
    category = merged.get("category", "laptop")
    base_query = merged.get("search_query") or ""

    # 🔥 Always enforce category context
    search_query = f"{category} {base_query} {user_query}".strip()

    # ─── STEP 6: FETCH PRODUCTS ───
    raw_products = search_products(
        query=search_query,
        category=category,
        budget_max=merged.get("budget_max"),
        page=1
    )

    if not raw_products:
        return jsonify({"error": "No products found for refined query."}), 404

    # ─── STEP 7: PROCESS + RANK ───
    df = process_products(raw_products, merged)
    ranked_df = recommend(df, merged, top_n=6)
    products_list = products_to_dict_list(ranked_df)

    # ─── STEP 8: LABELS ───
    for product in products_list[:3]:
        product["match_label"] = get_match_label(product.get("match_score", 0))
        product["pros"] = []
        product["cons"] = []

    # ─── STEP 9: SUMMARY ───
    try:
        summary = generate_recommendation_summary(products_list, user_query)
    except Exception:
        summary = f"Refined results: {len(products_list)} products found."

    return jsonify({
        "success": True,
        "session_id": session_id,
        "requirements": merged,
        "summary": summary,
        "products": products_list
    })
# ─── Interaction Logging ──────────────────────────────────────────────────────

@app.route("/api/interaction", methods=["POST"])
def track_interaction():
    """
    POST /api/interaction
    Body: { "session_id": "...", "asin": "...", "type": "click|purchase|compare" }
    """
    data = request.get_json()
    if data:
        log_interaction(
            session_id=data.get("session_id", ""),
            asin=data.get("asin", ""),
            interaction_type=data.get("type", "click")
        )
    return jsonify({"success": True})


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "version": "2.0",
        "service": "HardwareAI Backend"
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=5001, debug=debug)
