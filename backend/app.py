"""
app.py
Main Flask backend for HardwareAI Electronic Recommendation System.
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
#from amazon_api import get_product_reviews
from backend.claude_ai import (
    generate_recommendation_summary,
    infer_category_from_text,
    parse_budget_limits,
    parse_user_requirements,
    _remove_budget_words,
)
from backend.product_aggregator import search_products, get_reviews
from backend.data_processor import process_products, products_to_dict_list
from backend.recommendation_engine import recommend, get_match_label
from backend.database import init_db, save_query, save_recommendations, log_interaction, upsert_product
import re
load_dotenv()

app = Flask(__name__, static_folder="../frontend")
CORS(app, resources={r"/api/*": {"origins": "*"}})


def _enrich_products_with_reviews(products: list) -> list:
    """Attach real review snippets to every result without serial network waits."""
    def fetch_one(item):
        index, product = item
        asin = product.get("asin")
        try:
            reviews = get_reviews(asin=asin, product_url=product.get("product_url")) if asin else []
        except Exception as exc:
            print(f"[Review Scraper] Failed for {asin}: {exc}")
            reviews = []
        return index, product, reviews

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(products)))) as pool:
        fetched = list(pool.map(fetch_one, enumerate(products)))

    enriched = []
    for index, product, reviews in fetched:
        product["reviews"] = reviews
        # Show source review snippets directly. Per-product LLM calls made the
        # results page wait on several sequential remote requests and could
        # introduce unsupported generic claims when the model timed out.
        product.setdefault("pros", [])
        product.setdefault("cons", [])
        product["sentiment"] = "available_for_review" if reviews else "unavailable"
        enriched.append(product)
    return enriched


def _classify_refinement_context(user_query: str, original_req: dict, new_req: dict) -> str:
    """Decide whether a chat turn keeps the current search or starts a new one."""
    current = str(user_query or "").lower()
    previous = str(original_req.get("search_query") or "").lower()
    old_models = set(original_req.get("required_models") or [])
    new_models = set(new_req.get("required_models") or [])
    old_category = str(original_req.get("category") or "").lower()
    new_category = infer_category_from_text(user_query) or str(new_req.get("category") or "").lower()

    if re.search(r"\b(?:instead|switch(?:\s+to)?|start over|new search|new laptop|different\s+(?:product|laptop|macbook|phone|headphones|tablet|tv))\b", current):
        return "new_search"
    if new_category and old_category and new_category != old_category:
        return "new_search"
    if new_models and new_models != old_models:
        return "new_search"

    named_products = ("apple", "macbook", "samsung", "galaxy", "oneplus", "pixel", "redmi", "poco", "lenovo", "asus", "dell", "hp", "acer", "msi", "sony", "boat", "jbl")
    if any(re.search(rf"\b{re.escape(name)}\b", current) and not re.search(rf"\b{re.escape(name)}\b", previous) for name in named_products):
        return "new_search"

    explicit_product_type = bool(re.search(
        r"\b(?:laptops?|notebooks?|macbooks?|chromebooks?|phones?|smartphones?|mobiles?|headphones?|earphones?|earbuds?|tablets?|ipads?|monitors?|tvs?|televisions?|printers?|routers?|keyboards?|mice|mouses?|speakers?|cameras?|smartwatches?)\b",
        current,
    ))
    # A broad category request clears a previous exact model. Budget-only
    # requests contain no product type and therefore stay in the same context.
    if explicit_product_type and old_models and not new_models:
        return "new_search"
    return "refinement"

# Initialize database on startup
try:
    init_db()
except Exception as e:
    print(f"[WARN] DB init failed (check MySQL config): {e}")


#Serve Frontend

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


#Main Recommend Endpoint 

@app.route("/api/recommend", methods=["POST"])
def recommend_products():
    """
    POST /api/recommend
    Body: { "query": "...", "session_id": "..." }
    Returns: { "products": [...], "summary": "...", "requirements": {...} }
    """
    data = request.get_json(silent=True)  # silent: a non-JSON body gives None, not an HTML 415 page
    if not isinstance(data, dict) or not data.get("query"):
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
    print("[DEBUG] Requirements:", requirements)

    # Explicit product words in the user's query take precedence over an AI
    # classification that may be incomplete or temporarily unavailable.
    category = infer_category_from_text(user_query) or str(
        requirements.get("category") or ""
    ).lower()
    if category in {"smartphone", "phones", "phone"}:
        category = "mobile"

    if not category:
        return jsonify({
            "error": "What kind of product do you need? Try a laptop, phone, headphones, or TV.",
            "examples": [
                "laptop under 60000",
                "best headphones for gym",
                "gaming mouse under 2000"
            ]
        }), 400


    requirements["category"] = category

    # Step 2: Fetch products from Amazon API
    search_query = str(requirements.get("search_query") or user_query).strip()
    gpu_model_search = bool(re.search(r"\b(?:rtx|gtx|geforce|radeon|rx)\s*\d{3,4}\b", user_query, re.I))
    explicit_laptop = bool(re.search(r"\b(?:laptop|notebook|macbook|chromebook)\b", user_query, re.I))
    explicit_graphics_card = bool(re.search(r"\b(?:graphics? card|video card)\b", user_query, re.I))
    if gpu_model_search and category == "laptop" and not explicit_laptop and not explicit_graphics_card:
        # A bare GPU model query is treated as a laptop request, so say
        # "laptop" explicitly to Amazon and avoid desktop graphics cards.
        search_query = f"laptop {search_query}".strip()
    elif not infer_category_from_text(user_query):
        search_query = f"{category} {search_query}".strip()

    try:
        raw_products = search_products(
            query=search_query,
            category=category,
            budget_min=requirements.get("budget_min"),
            budget_max=requirements.get("budget_max"),
            page=1,
        )
        print(f"[Amazon] Fetched {len(raw_products)} products")
    except Exception as e:
        print(f"[Amazon] Error: {e}")
        return jsonify({"error": "Amazon product retrieval failed. Check network access or whether Amazon is blocking the current request."}), 502

    if not raw_products:
        return jsonify({"error": "No products found. Try adjusting your query or budget."}), 404

    # Step 3: Process and clean data
    df = process_products(raw_products, requirements)
    if df.empty:
        return jsonify({"error": "No products matched your filters after processing."}), 404

    # Step 4: Run ML recommendation engine
    ranked_df = recommend(df, requirements, top_n=6)

    # Step 5: Add review snippets for all visible results.
    products_list = products_to_dict_list(ranked_df)
    for product in products_list:
        exact_match = bool(requirements.get("required_models"))
        product["match_label"] = get_match_label(product.get("match_score", 0), exact_match=exact_match)
        if exact_match:
            product["pros"] = [f"Matches the {requirements['required_models'][0].upper()} model you asked for."]
    products_list = _enrich_products_with_reviews(products_list)


    # Step 6: Give a fast, factual summary based on the ranked results.
    summary = generate_recommendation_summary(products_list, user_query)

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


#Refine Search Endpoint

@app.route("/api/refine", methods=["POST"])
def refine_search():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data.get("query"):
        return jsonify({"error": "Query required"}), 400

    user_query = data["query"]
    original_req = data.get("original_requirements")
    if not isinstance(original_req, dict):
        original_req = {}  # e.g. null when the page has no previous search
    session_id = data.get("session_id", str(uuid.uuid4()))

    try:
        new_req = parse_user_requirements(user_query)
    except Exception as exc:
        print("[Refine parse error]:", exc)
        new_req = {
            "category": infer_category_from_text(user_query),
            "search_query": _remove_budget_words(user_query),
            "budget_min": None, "budget_max": None, "required_models": [],
            "priority_specs": {}, "use_case": "general", "intent": "none", "modifiers": [],
        }

    previous_query = str(original_req.get("search_query", ""))
    context_mode = _classify_refinement_context(user_query, original_req, new_req)
    replace_context = context_mode == "new_search"
    category_from_text = infer_category_from_text(user_query) or new_req.get("category")

    #STEP 1: Merge AI parsed requirements
    merged = {} if replace_context else {**original_req}
    for key, val in new_req.items():
        if key == "priority_specs" and isinstance(val, dict):
            if replace_context:
                merged[key] = val
            else:
                old_specs = merged.get("priority_specs") or {}
                meaningful = {k: v for k, v in val.items() if v not in (None, False, [], "")}
                if meaningful:
                    merged[key] = {**old_specs, **meaningful}
            continue
        if key == "required_models":
            if replace_context or val:
                merged[key] = val
            continue
        if key == "use_case" and val == "general" and original_req.get("use_case") and not replace_context:
            continue
        if val is not None and val != {} and val != []:
            merged[key] = val

    query_lower = user_query.lower()

    # STEP 2: preserve the selected category unless the refinement names a new one.
    if category_from_text:
        merged["category"] = category_from_text


    if not merged.get("category"):
        merged["category"] = original_req.get("category", "laptop")

    explicit_budget = parse_budget_limits(user_query)
    if explicit_budget["budget_min"] is not None or explicit_budget["budget_max"] is not None:
        merged.update(explicit_budget)

    #STEP 3: AI INTENT HANDLING

    intent = new_req.get("intent")

    if intent == "increase_budget":

        if merged.get("budget_max") is not None:
            merged["budget_max"] += 20000

    elif intent == "decrease_budget":

        if merged.get("budget_max") is not None:
            merged["budget_max"] = max(
                10000,
                merged["budget_max"] - 20000
            )

    elif intent == "set_budget":

        if new_req.get("budget_min") is not None:
            merged["budget_min"] = new_req["budget_min"]

        if new_req.get("budget_max") is not None:
            merged["budget_max"] = new_req["budget_max"]
                

    #STEP 4: MODIFIER HANDLING
    modifiers = new_req.get("modifiers", [])

    if "gaming" in modifiers:
        merged["use_case"] = "gaming"

    if "battery" in modifiers:
        merged.setdefault("priority_specs", {})["battery_life"] = True

    if "performance" in modifiers:
        merged.setdefault("priority_specs", {})["gpu_required"] = True

    if "lightweight" in modifiers or "portable" in modifiers:
        merged.setdefault("priority_specs", {})["portability"] = True


   # STEP 5: SMART QUERY REFINEMENT

    category = merged.get("category", "laptop")

    previous_query = previous_query.strip()

    current_query = str(user_query).strip()

    query_lower = current_query.lower()
    


# REMOVE OLD BUDGET FILTERS IF NEW ONE EXISTS


    budget_words = [
        "under",
        "below",
        "less than",
        "above",
        "over",
        "greater than"
    ]

    new_budget_detected = any(
        word in query_lower
        for word in budget_words
    )

    clean_previous_query = previous_query

    if new_budget_detected:

        clean_previous_query = re.sub(
            r"(under|below|less than|above|over|greater than)\s*(rs\.?|₹|inr)?\s*\d+[kK]?",
            "",
            clean_previous_query,
            flags=re.IGNORECASE
        ).strip()

    # Select a new search context or continue the existing search.
    if replace_context:
        
        # explicit replace
        combined_query = current_query

    else:

        # refine previous search
        combined_words = []

        for word in (clean_previous_query + " " + current_query).split():

            if word.lower() not in [
                w.lower()
                for w in combined_words
            ]:
                combined_words.append(word)

        combined_query = " ".join(combined_words)

    # FINAL QUERY


    search_query = combined_query.strip()

    merged["search_query"] = search_query

    print("[REFINED QUERY]", search_query)

    # STEP 6: FETCH PRODUCTS 
    store_query = _remove_budget_words(search_query)
    if (
        category == "laptop"
        and re.search(r"\b(?:rtx|gtx|geforce|radeon|rx)\s*\d{3,4}\b", store_query, re.I)
        and not re.search(r"\b(?:laptop|notebook|macbook|chromebook|graphics? card|video card)\b", store_query, re.I)
    ):
        store_query = f"laptop {store_query}"

    try:
        raw_products = search_products(
            query=store_query,
            category=category,
            budget_min=merged.get("budget_min"),
            budget_max=merged.get("budget_max"),
            page=1,
        )
    except Exception as e:
        print(f"[Amazon] Refine error: {e}")
        return jsonify({"error": "Amazon product retrieval failed. Check network access or whether Amazon is blocking the current request."}), 502

    # STEP 7: PROCESS + RANK
    df = process_products(raw_products, merged)
    ranked_df = recommend(df, merged, top_n=6)
    products_list = products_to_dict_list(ranked_df)

    # STEP 8: LABELS
    for product in products_list:
        exact_match = bool(merged.get("required_models"))
        product["match_label"] = get_match_label(product.get("match_score", 0), exact_match=exact_match)
        if exact_match:
            product["pros"] = [f"Matches the {merged['required_models'][0].upper()} model you asked for."]
    products_list = _enrich_products_with_reviews(products_list)

    # ─── STEP 9: SUMMARY ───
    try:
        summary = generate_recommendation_summary(products_list, merged.get("search_query", user_query))
    except Exception:
        summary = f"Refined results: {len(products_list)} products found."

    return jsonify({
        "success": True,
        "context_mode": context_mode,
        "session_id": session_id,
        "requirements": merged,
        "summary": summary,
        "products": products_list
    })
# Interaction Logging 

@app.route("/api/interaction", methods=["POST"])
def track_interaction():
    """
    POST /api/interaction
    Body: { "session_id": "...", "asin": "...", "type": "click|purchase|compare" }
    """
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        try:
            log_interaction(
                session_id=data.get("session_id", ""),
                asin=data.get("asin", ""),
                interaction_type=data.get("type", "click")
            )
        except Exception as e:
            # Logging is optional; the website must keep working without MySQL.
            print(f"[DB] Interaction log skipped: {e}")
            return jsonify({"success": False, "logged": False})
    return jsonify({"success": True})




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
    app.run(host="0.0.0.0", port=port, debug=debug)
