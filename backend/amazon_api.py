import os
import requests
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = "real-time-amazon-data.p.rapidapi.com"

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST
}


def search_products(query: str, category: str = "laptop", budget_max: float = None, page: int = 1) -> list:
    url = "https://real-time-amazon-data.p.rapidapi.com/search"

    params = {
        "query": f"{query} India",
        "page": str(page),
        "country": "IN",
        "sort_by": "RELEVANCE",
        "product_condition": "NEW"
    }

    if budget_max is not None:
        params["max_price"] = str(int(budget_max))

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)

        #HANDLE RATE LIMIT
        if response.status_code == 429:
            print("[Amazon API] Rate limit exceeded (429)")
            return []

        #HANDLE OTHER ERRORS
        if response.status_code != 200:
            print(f"[Amazon API] Bad status: {response.status_code}")
            return []

        #SAFE JSON PARSE
        try:
            data = response.json()
        except:
            print("[Amazon API] Invalid JSON response")
            return []

        if not data:
            return []

        products = data.get("data", {}).get("products", [])
        if not isinstance(products, list):
            print("[Amazon API] Invalid products format")
            return []

        #SAFE NORMALIZATION
        result = []
        for p in products:
            if not p or not isinstance(p, dict):
                continue
            result.append(_normalize_product(p))

        return result

    except requests.exceptions.Timeout:
        print("[Amazon API] Request timed out")
        return []

    except Exception as e:
        print(f"[Amazon API] Unexpected error: {e}")
        return []


def get_product_details(asin: str) -> dict:
    url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
    params = {"asin": asin, "country": "IN"}  # 🇮🇳

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        product = data.get("data", {})
        return _normalize_product_detail(product)

    except Exception as e:
        print(f"[Amazon API] Detail fetch error for {asin}: {e}")
        return {}


def get_product_reviews(asin: str) -> list:
    url = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"

    params = {
        "asin": asin,
        "country": "IN",   
        "page": "1",
        "star_rating": "ALL"
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        reviews = data.get("data", {}).get("reviews", [])
        return [r.get("review_comment", "") for r in reviews if r.get("review_comment")]

    except Exception as e:
        print(f"[Amazon API] Reviews fetch error for {asin}: {e}")
        return []


def _normalize_product(raw: dict) -> dict:
    price_str = str(
        raw.get("product_price")
        or "0"
    )


    if isinstance(price_str, str) and "$" in price_str:
        try:
            usd_value = float(price_str.replace("$", "").replace(",", "").strip())
            inr_value = int(usd_value * 83)  # conversion rate
            price_str = f"₹{inr_value}"
        except:
            price_str = price_str.replace("$", "₹")

    price = _parse_price(price_str)

    return {
        "asin": raw.get("asin", ""),
        "product_title": raw.get("product_title", "Unknown Product"),
        "product_price": price,
        "product_price_str": price_str,  
        "product_star_rating": _safe_float(raw.get("product_star_rating", 0)),
        "product_num_ratings": raw.get("product_num_ratings", 0),
        "product_url": raw.get("product_url", ""),
        "product_photo": raw.get("product_photo", ""),
        "is_best_seller": raw.get("is_best_seller", False),
        "is_amazon_choice": raw.get("is_amazon_choice", False),
        "delivery": raw.get("delivery", ""),
        "specs": {}
    }


def _normalize_product_detail(raw: dict) -> dict:
    specs = {}
    for feature in raw.get("product_information", {}).items():
        specs[feature[0]] = feature[1]

    about = raw.get("about_product", [])
    if about:
        specs["highlights"] = about[:5]

    return {
        "asin": raw.get("asin", ""),
        "product_title": raw.get("product_title", ""),
        "product_price": _parse_price(raw.get("product_price", "0")),
        "product_star_rating": _safe_float(raw.get("product_star_rating", 0)),
        "product_num_ratings": raw.get("product_num_ratings", 0),
        "product_url": raw.get("product_url", ""),
        "product_photo": raw.get("product_photos", [""])[0] if raw.get("product_photos") else "",
        "specs": specs,
        "description": raw.get("product_description", ""),
        "category": raw.get("category_path", [{}])[-1].get("name", "") if raw.get("category_path") else ""
    }


def _parse_price(price_str) -> float:
    if isinstance(price_str, (int, float)):
        return float(price_str)
    try:
        return float(
            str(price_str)
            .replace("$", "")
            .replace("₹", "")
            .replace(",", "")
            .strip()
        )
    except (ValueError, AttributeError):
        return 0.0


def _safe_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0