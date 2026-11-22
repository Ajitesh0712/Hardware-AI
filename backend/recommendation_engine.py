"""Recommendation ranking: learned proxy plus explicit intent matching."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from backend.data_processor import build_feature_vector
from backend.ml_model import predict_score
from backend.product_features import classify_product


USE_CASE_WEIGHTS = {
    "gaming":       {"price": .10, "rating": .15, "ram": .15, "storage": .10, "gpu": .35, "display": .05, "popularity": .10},
    "coding":       {"price": .15, "rating": .15, "ram": .30, "storage": .20, "gpu": .05, "display": .05, "popularity": .10},
    "design":       {"price": .10, "rating": .15, "ram": .20, "storage": .15, "gpu": .20, "display": .15, "popularity": .05},
    "3d rendering": {"price": .05, "rating": .10, "ram": .25, "storage": .15, "gpu": .35, "display": .05, "popularity": .05},
    "video editing": {"price": .10, "rating": .15, "ram": .25, "storage": .20, "gpu": .15, "display": .10, "popularity": .05},
    "student":      {"price": .35, "rating": .20, "ram": .15, "storage": .10, "gpu": .03, "display": .07, "popularity": .10},
    "business":     {"price": .20, "rating": .25, "ram": .15, "storage": .10, "gpu": .03, "display": .07, "popularity": .20},
    "general":      {"price": .20, "rating": .25, "ram": .15, "storage": .10, "gpu": .05, "display": .05, "popularity": .20},
}

_EXPECTED_CATEGORIES = {
    "laptop": {"Laptops"}, "mobile": {"Smartphones"}, "smartphone": {"Smartphones"},
    "headphones": {"Headphones and Earphones"}, "earbuds": {"Headphones and Earphones"},
    "tablet": {"Tablets"}, "monitor": {"Monitors and Projectors"},
    "speaker": {"Speakers"}, "smartwatch": {"Smartwatches"}, "camera": {"Cameras"},
    "mouse": {"Computer Accessories"}, "keyboard": {"Computer Accessories"},
    "television": {"Televisions"}, "printer": {"Printers and Printing Supplies"},
    "networking": {"Networking Devices"}, "power_banks": {"Power Banks"},
    "storage": {"Storage Devices"}, "charger": {"Chargers and Cables"},
    "computer_components": {"Computer Components"},
}


def _match_score_vector(vector, weights, requirements, title):
    named = dict(zip(("price", "rating", "ram", "storage", "gpu", "display", "popularity"), vector))
    priority = requirements.get("priority_specs") or {}
    # When the user specifies a feature, missing title evidence is neutral,
    # while explicit insufficiency affects the corresponding feature score.
    if priority.get("gpu_required") and named["gpu"] == 0.5:
        named["gpu"] = 0.2
    if priority.get("ram_gb") and np.isnan(named["ram"]):
        named["ram"] = 0.45
    if priority.get("storage_gb") and np.isnan(named["storage"]):
        named["storage"] = 0.45
    weight_total = sum(weights.values()) or 1.0
    score = sum(named[key] * weight for key, weight in weights.items()) / weight_total
    query = str(requirements.get("search_query", "")).lower()
    useful = [w for w in re.findall(r"[a-z0-9]+", query) if len(w) > 2 and w not in {"under", "with", "best", "good", "for", "the"}]
    if useful:
        overlap = sum(word in title for word in useful) / len(useful)
        score = 0.90 * score + 0.10 * overlap
    return float(np.clip(score, 0, 1))


def recommend(df: pd.DataFrame, requirements: dict, top_n: int = 5) -> pd.DataFrame:
    """Rank candidates while applying known category and budget constraints."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    ranked = df.copy()
    category = str(requirements.get("category", "")).lower()
    expected = _EXPECTED_CATEGORIES.get(category)
    if expected and "product_category" in ranked:
        known = ranked.product_category.isin(expected)
        ambiguous = ranked.product_category.eq("Other / Unclassified")
        # Keep uncertain listings only if category detection found no clear
        # matches. This prevents accessories and unrelated devices outranking
        # the requested product type.
        ranked = ranked[known] if known.any() else ranked[ambiguous]
        ranked = ranked.copy()

    if ranked.empty:
        return ranked
    prices = pd.to_numeric(ranked["product_price"], errors="coerce")
    ranked = ranked.loc[prices.notna()].copy()
    if requirements.get("budget_min") is not None:
        ranked = ranked[ranked.product_price >= float(requirements["budget_min"])]
    if requirements.get("budget_max") is not None:
        ranked = ranked[ranked.product_price <= float(requirements["budget_max"])]
    if ranked.empty:
        return ranked

    use_case = str(requirements.get("use_case", "general")).lower()
    weights = USE_CASE_WEIGHTS.get(use_case, USE_CASE_WEIGHTS["general"])
    vectors = [build_feature_vector(row, requirements) for _, row in ranked.iterrows()]
    model_scores = [predict_score(row.to_dict(), requirements) for _, row in ranked.iterrows()]
    intent_scores = [
        _match_score_vector(vector, weights, requirements, str(row.get("product_title", "")).lower())
        for vector, (_, row) in zip(vectors, ranked.iterrows())
    ]
    ranked["model_score"] = model_scores
    ranked["intent_score"] = intent_scores
    # Keep the user's needs in front of the offline proxy score. The learned
    # score summarizes catalog patterns; it is not a measure of query accuracy.
    ranked["match_score"] = (0.30 * ranked["model_score"] + 0.70 * ranked["intent_score"]) * 100
    ranked.loc[ranked["is_best_seller"], "match_score"] += 2
    ranked.loc[ranked["is_amazon_choice"], "match_score"] += 1
    ranked["match_score"] = ranked["match_score"].clip(0, 100)
    ranked = ranked.sort_values("match_score", ascending=False)
    ranked = _deduplicate_products(ranked)
    return ranked.head(top_n).reset_index(drop=True)


def _deduplicate_products(df: pd.DataFrame) -> pd.DataFrame:
    seen, keep = [], []

    def variants(title):
        text = str(title).lower()
        values = []
        for pattern in (
            r"\b\d{1,3}\s*gb\s*ram\b",
            r"\b\d{2,4}\s*gb\s*(?:ssd|storage|nvme)\b|\b\d\s*tb\s*(?:ssd|storage|nvme)\b",
            r"\b(?:rtx|gtx)\s*\d{3,4}(?:\s*(?:ti|super))?\b|\brx\s*\d{3,4}(?:\s*(?:xt|xtx))?\b",
            r"\b(?:i[3579]-?\d{4,5}[a-z]{0,2}|ryzen\s+[3579]\s+\d{4,5}[a-z]{0,2})\b",
        ):
            match = re.search(pattern, text)
            values.append(re.sub(r"\s+", "", match.group(0)) if match else None)
        return values

    for idx, row in df.iterrows():
        title = str(row.get("product_title", "")).lower()
        words = set(re.findall(r"[a-z0-9]+", title))
        duplicate = False
        for previous, previous_asin in seen:
            asin = str(row.get("asin", ""))
            if asin and asin == previous_asin:
                duplicate = True
                break
            other = set(re.findall(r"[a-z0-9]+", previous))
            current_specs, previous_specs = variants(title), variants(previous)
            has_different_specs = any(a and b and a != b for a, b in zip(current_specs, previous_specs))
            if words and other and not has_different_specs and len(words & other) / len(words | other) > 0.78:
                duplicate = True
                break
        if not duplicate:
            seen.append((title, str(row.get("asin", ""))))
            keep.append(idx)
    return df.loc[keep]


def compute_similarity_matrix(df: pd.DataFrame, requirements: dict) -> np.ndarray:
    if df.empty or len(df) < 2:
        return np.array([])
    matrix = np.vstack([build_feature_vector(row, requirements) for _, row in df.iterrows()])
    return cosine_similarity(MinMaxScaler().fit_transform(matrix))


def get_match_label(score: float, exact_match: bool = False) -> str:
    """Plain-language fit label for the relative ranking."""
    if exact_match:
        return "Matches your model"
    if score >= 80:
        return "Great fit"
    if score >= 65:
        return "Good fit"
    if score >= 50:
        return "Worth a look"
    return "Not a close fit"
