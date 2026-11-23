"""Normalize scraped products into the application's internal schema."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from backend.product_features import classify_product, extract_title_specs


def _number(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("₹", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def process_products(raw_products: list, requirements: dict) -> pd.DataFrame:
    """Normalize scraper fields, clean values, add specs and apply budget."""
    if not raw_products:
        return pd.DataFrame()

    df = pd.DataFrame(raw_products).copy()
    aliases = {
        "product_title": ("product_title", "name", "title"),
        "product_price": ("product_price", "discount_price", "price", "product_price_str"),
        "product_star_rating": ("product_star_rating", "review_rating", "rating"),
        "product_num_ratings": ("product_num_ratings", "no_of_ratings", "review_count"),
    }
    for target, sources in aliases.items():
        if target not in df:
            source = next((col for col in sources if col in df), None)
            df[target] = df[source] if source else np.nan

    df["product_title"] = df["product_title"].fillna("").astype(str)
    df["product_price"] = df["product_price"].map(_number)
    df["product_star_rating"] = df["product_star_rating"].map(_number).clip(0, 5)
    df.loc[df["product_star_rating"] <= 0, "product_star_rating"] = np.nan
    df["product_num_ratings"] = df["product_num_ratings"].map(_number).clip(lower=0)
    df["product_num_ratings_missing"] = df["product_num_ratings"].isna()

    # Keep listings with a usable price. Missing ratings remain missing so the
    # saved model's training-fitted imputer handles them identically at inference.
    df = df[df["product_price"].notna() & (df["product_price"] > 0)].copy()
    if df.empty:
        return df.reset_index(drop=True)
    df["product_num_ratings"] = df["product_num_ratings"].fillna(0)

    df["product_category"] = df["product_title"].map(classify_product)
    specs = df.apply(
        lambda row: extract_title_specs(row["product_title"], row["product_category"]),
        axis=1,
        result_type="expand",
    )
    df = pd.concat([df, specs.add_prefix("extracted_")], axis=1)

    # Exact model names/numbers are hard constraints. A query for an RTX 4060
    # must never rank RTX 4050/4070 laptops as if they were matches.
    for requested_model in requirements.get("required_models", []):
        model = str(requested_model).strip().lower()
        if not model:
            continue
        escaped = re.escape(model)
        escaped = re.sub(r"\\\s+", r"[\\s-]*", escaped)
        escaped = escaped.replace(r"\-", r"[ -]?")
        if re.fullmatch(r"m[1-5](?:\s+(?:pro|max|ultra))?", model):
            escaped += r"(?![a-z0-9])"
        if re.fullmatch(r"m[1-5]", model):
            # "M5" means the base chip, just as "RTX 4060" excludes the Ti.
            escaped += r"(?!\s*(?:pro|max|ultra)\b)"
        if re.search(r"\b(?:rtx|gtx)\s*\d{3,4}$", model):
            escaped += r"(?!\s*(?:ti|super)\b)"
        elif re.search(r"\brx\s*\d{3,4}$", model):
            escaped += r"(?!\s*(?:xt|xtx)\b)"
        pattern = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I)
        df = df[df["product_title"].map(lambda title: bool(pattern.search(title)))].copy()
        if df.empty:
            return df.reset_index(drop=True)

    priority = requirements.get("priority_specs") or {}
    for key, wanted in (("extracted_ram_gb", priority.get("ram_gb")), ("extracted_storage_gb", priority.get("storage_gb"))):
        if wanted is not None:
            available = pd.to_numeric(df[key], errors="coerce")
            df = df[available.isna() | (available >= float(wanted))].copy()
            if df.empty:
                return df.reset_index(drop=True)

    budget_min = _number(requirements.get("budget_min")) if requirements.get("budget_min") is not None else np.nan
    budget_max = _number(requirements.get("budget_max")) if requirements.get("budget_max") is not None else np.nan
    if np.isfinite(budget_min):
        df = df[df["product_price"] >= budget_min]
    if np.isfinite(budget_max):
        df = df[df["product_price"] <= budget_max]

    popularity_rating = df["product_star_rating"].fillna(df["product_star_rating"].median()).fillna(3.5)
    mean_rating = float(popularity_rating.mean()) if not df.empty else 3.5
    threshold = float(df["product_num_ratings"].quantile(0.25)) if not df.empty else 0.0
    threshold = max(threshold, 1.0)
    df["popularity_score"] = (
        df["product_num_ratings"] / (df["product_num_ratings"] + threshold) * popularity_rating
        + threshold / (df["product_num_ratings"] + threshold) * mean_rating
    ) / 5.0
    for flag in ("is_best_seller", "is_amazon_choice"):
        if flag not in df:
            df[flag] = False
        df[flag] = df[flag].map(
            lambda value: value.strip().lower() in {"true", "1", "yes"}
            if isinstance(value, str)
            else bool(value) if pd.notna(value) else False
        )
    return df.reset_index(drop=True)


def build_feature_vector(product_row: pd.Series, requirements: dict) -> np.ndarray:
    """Return intent match dimensions used by similarity and ranking."""
    price = float(product_row.get("product_price", 0) or 0)
    budget_max = _number(requirements.get("budget_max")) if requirements.get("budget_max") is not None else price * 1.5
    if not np.isfinite(budget_max) or budget_max <= 0:
        budget_max = max(price * 1.5, 1.0)
    price_score = float(np.clip(1 - price / budget_max, 0, 1))
    rating_value = product_row.get("product_star_rating", np.nan)
    rating_score = float(np.clip(rating_value, 0, 5) / 5) if pd.notna(rating_value) else 0.5

    priority = requirements.get("priority_specs") or {}
    def ratio_score(key, requested, neutral=0.5):
        available = product_row.get(key, np.nan)
        if requested and pd.notna(available):
            return float(np.clip(float(available) / float(requested), 0, 1))
        return neutral

    ram_score = ratio_score("extracted_ram_gb", priority.get("ram_gb"))
    storage_score = ratio_score("extracted_storage_gb", priority.get("storage_gb"))
    gpu_required = bool(priority.get("gpu_required"))
    has_gpu = bool(re.search(r"\b(?:rtx|gtx|geforce|radeon)\b", str(product_row.get("product_title", "")), re.I))
    gpu_score = 1.0 if has_gpu else (0.2 if gpu_required else 0.5)
    display = product_row.get("extracted_screen_inches", np.nan)
    requested_display = priority.get("display_size")
    display_score = float(np.clip(1 - abs(float(display) - float(requested_display)) / float(requested_display), 0, 1)) if requested_display and pd.notna(display) and float(requested_display) > 0 else 0.5
    popularity = float(np.clip(product_row.get("popularity_score", 0.5) or 0.5, 0, 1))
    return np.array([price_score, rating_score, ram_score, storage_score, gpu_score, display_score, popularity])


def products_to_dict_list(df: pd.DataFrame) -> list:
    result = []
    for _, row in df.iterrows():
        def number_or(value, default=0.0):
            return float(value) if pd.notna(value) else default
        def text_or_empty(value):
            return "" if value is None or pd.isna(value) else str(value)
        def spec_text(key, suffix=""):
            value = row.get(f"extracted_{key}", np.nan)
            if key == "storage_gb" and pd.notna(value) and value >= 1024 and value % 1024 == 0:
                return f"{value / 1024:g} TB"
            return f"{value:g}{suffix}" if pd.notna(value) else "N/A"
        result.append({
            "asin": text_or_empty(row.get("asin", "")),
            "product_title": text_or_empty(row.get("product_title", "")),
            "product_price": number_or(row.get("product_price")),
            "product_price_str": f"₹{number_or(row.get('product_price')):,.0f}",
            "product_star_rating": number_or(row.get("product_star_rating"), default=None),
            "product_num_ratings": (
                None if bool(row.get("product_num_ratings_missing", False))
                else int(number_or(row.get("product_num_ratings")))
            ),
            "product_url": text_or_empty(row.get("product_url", "")),
            "product_photo": text_or_empty(row.get("product_photo", "")),
            "is_best_seller": bool(row.get("is_best_seller", False)),
            "is_amazon_choice": bool(row.get("is_amazon_choice", False)),
            "product_category": text_or_empty(row.get("product_category", "Other / Unclassified")),
            "specs": {
                "RAM": spec_text("ram_gb", "GB"),
                "Storage": spec_text("storage_gb", "GB"),
                "Battery": spec_text("battery_mah", " mAh"),
                "Display": spec_text("screen_inches", '"'),
                "Charging": spec_text("charging_w", "W"),
                "Refresh rate": spec_text("refresh_rate_hz", " Hz"),
                "Graphics": _friendly_graphics_name(str(row.get("product_title", ""))),
            },
            "match_score": number_or(row.get("match_score")),
            "fit_score": int(number_or(row.get("fit_score"), default=1)),
            "model_score": number_or(row.get("model_score")),
            "popularity_score": number_or(row.get("popularity_score")),
        })
    return result


def _friendly_graphics_name(title: str) -> str:
    match = re.search(r"\b(?:geforce\s+)?(rtx|gtx)\s*(\d{3,4})(\s*(?:ti|super))?\b|\b(?:radeon\s+)?(rx)\s*(\d{3,4})(\s*(?:xtx?|xt))?\b", title, re.I)
    if not match:
        return "Dedicated graphics" if re.search(r"\b(?:geforce|radeon|discrete graphics)\b", title, re.I) else "Not listed"
    family = match.group(1) or match.group(4)
    number = match.group(2) or match.group(5)
    suffix = (match.group(3) or match.group(6) or "").strip()
    return f"{family.upper()} {number}{(' ' + suffix.title()) if suffix else ''}"
