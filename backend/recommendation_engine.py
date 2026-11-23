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
from backend.use_cases import USE_CASE_WORDS, detect_use_cases, fails_key_need, use_case_checks
from backend.query_words import ACCESSORY_WORDS, REFINEMENT_WORDS, key_words, looks_like_addon, normalize_text, product_words


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
        if known.any():
            ranked = ranked[known]
        else:
            # Unclassified listings are kept only if their title names what
            # the user typed, so an Xbox game is never shown as a "phone".
            wanted = _wanted_product_words(requirements)
            ranked = ranked[ambiguous & ranked["product_title"].map(lambda title: _share_in_title(wanted, title) > 0)]
        ranked = ranked.copy()
    elif not expected:
        # Any other electronics (consoles, drones, foldable phones, ...):
        # there is no fixed category, so keep listings whose title names at
        # least one of the product words the user typed.
        wanted = _wanted_product_words(requirements)
        if wanted:
            names_product = ranked["product_title"].map(lambda title: _share_in_title(wanted, title) > 0)
            ranked = ranked[names_product]
        if "product_category" in ranked:
            ranked = ranked[ranked.product_category.ne("Non-electronic / Miscellaneous")]
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
    # Products that meet more of the user's needs always come first; the
    # older blended score only breaks ties between equally good matches.
    ranked["rank_score"] = ranked["match_score"]
    ranked["fit_score"] = [fit_score_out_of_10(row, requirements) for _, row in ranked.iterrows()]
    ranked = ranked.sort_values(["fit_score", "rank_score"], ascending=False)
    ranked = _deduplicate_products(ranked)
    # The page colours its badge from match_score (0-100), so store the fit there.
    ranked["match_score"] = ranked["fit_score"] * 10
    return ranked.head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Match score out of 10
# ---------------------------------------------------------------------------

def _model_words(requirements):
    words = set()
    for model in requirements.get("required_models") or []:
        words.update(re.findall(r"[a-z0-9]+", str(model).lower()))
    return words


def _wanted_product_words(requirements):
    """Product words the user typed, e.g. ['sony', 'ps5'] for 'i need sony playstation 5'."""
    query = requirements.get("search_query", "")
    return [w for w in product_words(query) if w not in _model_words(requirements)]


def _share_in_title(words, title):
    """Fraction of the words that appear in the title (0.0 to 1.0)."""
    if not words:
        return 1.0
    text = normalize_text(title)
    return sum(1 for word in words if word in text) / len(words)


def _names_a_category(word):
    from backend.claude_ai import infer_category_from_text  # local import keeps startup simple
    return infer_category_from_text(word) is not None


def fit_score_out_of_10(row, requirements):
    """How well one product meets what the user asked for.

    10 = meets every stated need, 1 = meets none. Each need the user gave
    (product type, exact model, budget, RAM, storage, extra words such as
    "gaming") is a check with a weight; the score is the weighted share of
    checks that pass, mapped from 0-1 onto 1-10.
    """
    title = str(row.get("product_title", ""))
    query = requirements.get("search_query", "")
    category = str(requirements.get("category", "")).lower()
    expected = _EXPECTED_CATEGORIES.get(category)
    model_words = _model_words(requirements)
    checks = []  # (weight, how well it is met from 0 to 1)

    if expected:
        # 1) Right type of product, using the title classifier.
        checks.append((3, 1.0 if row.get("product_category") in expected else 0.5))
        # 2) Other words the user typed ("gaming", "lenovo", "oled", ...).
        # Use words ("video", "editing", "3d") are handled by the hardware
        # checks below, so they are not expected to appear in the title.
        extra = [w for w in key_words(query)
                 if w not in model_words and w not in USE_CASE_WORDS and not _names_a_category(w)]
        for word in extra:
            checks.append((1.5, _share_in_title([word], title)))
        # 3) What the user wants to do -> GPU, CPU, RAM and storage needs.
        if category == "laptop":
            use_cases = requirements.get("use_cases") or detect_use_cases(query)
            checks.extend(use_case_checks(
                title, use_cases,
                ram_gb=row.get("extracted_ram_gb"), storage_gb=row.get("extracted_storage_gb"),
            ))
    else:
        # 1) Right product: the title should name what the user typed.
        wanted = _wanted_product_words(requirements)
        share = _share_in_title(wanted, title) if wanted else 0.5
        typed = set(key_words(query))
        title_words = set(re.findall(r"[a-z0-9]+", normalize_text(title)))
        asked_addons = typed & ACCESSORY_WORDS
        if looks_like_addon(title, wanted, typed):
            share *= 0.3  # an add-on (case, controller, game) the user did not ask for
        elif asked_addons and not asked_addons & title_words:
            share *= 0.3  # the user asked for an add-on, this is the main product
        checks.append((3, share))
        # 2) Preference words such as "gaming" or "wireless".
        extra = [w for w in key_words(query) if w in REFINEMENT_WORDS]
        for word in extra:
            checks.append((1.5, _share_in_title([word], title)))

    # 3) Exact model and budget. Products failing these were already removed.
    if requirements.get("required_models"):
        checks.append((3, 1.0))
    if requirements.get("budget_min") is not None or requirements.get("budget_max") is not None:
        checks.append((2, 1.0))

    # 4) RAM / storage: met, not met, or unknown from the title (half credit).
    priority = requirements.get("priority_specs") or {}
    for column, wanted in (("extracted_ram_gb", priority.get("ram_gb")), ("extracted_storage_gb", priority.get("storage_gb"))):
        if wanted:
            value = row.get(column, np.nan)
            checks.append((1, 0.5 if pd.isna(value) else float(float(value) >= float(wanted))))

    total_weight = sum(weight for weight, _ in checks)
    met = sum(weight * value for weight, value in checks) / total_weight
    score = int(round(1 + 9 * met))
    # A laptop that clearly cannot do the job (e.g. no graphics card for 3D
    # work) is capped at 5/10, however well it fits the budget.
    if category == "laptop" and fails_key_need(title, requirements.get("use_cases") or detect_use_cases(query)):
        score = min(score, 5)
    return score


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


def get_match_label(fit_score) -> str:
    """Badge text such as "8/10 match" (10 = meets every stated need)."""
    return f"{int(fit_score)}/10 match"
