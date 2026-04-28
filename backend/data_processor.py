import numpy as np
import pandas as pd
import re


def process_products(raw_products: list, requirements: dict) -> pd.DataFrame:
    """
    Main pipeline: takes raw Amazon products and requirements,
    returns a clean, enriched DataFrame ready for the recommendation engine.
    """
    if not raw_products:
        return pd.DataFrame()

    df = pd.DataFrame(raw_products)
    df = _clean_prices(df)
    df = _clean_ratings(df)
    df = _extract_specs_from_title(df)
    df = _apply_budget_filter(df, requirements)
    df = _compute_popularity_score(df)
    df = df.reset_index(drop=True)

    return df


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure price is a float, remove nulls and zero-price items."""
    df["product_price"] = pd.to_numeric(df["product_price"], errors="coerce")
    df = df[df["product_price"].notna()]
    df = df[df["product_price"] > 0]
    return df


def _clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ratings to 0-5 scale, fill missing with median."""
    df["product_star_rating"] = pd.to_numeric(df["product_star_rating"], errors="coerce")
    median_rating = df["product_star_rating"].median()
    if pd.isna(median_rating):
        median_rating = 3.5
    df["product_star_rating"] = df["product_star_rating"].fillna(median_rating)
    df["product_star_rating"] = df["product_star_rating"].clip(0, 5)

    df["product_num_ratings"] = pd.to_numeric(df["product_num_ratings"], errors="coerce").fillna(0)
    return df


def _extract_specs_from_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse common specs from product titles using regex.
    Extracts: RAM, storage, display size, processor hints.
    """
    titles = df["product_title"].fillna("").str.lower()

    # RAM: e.g. "16GB RAM", "32 GB Memory"
    df["extracted_ram_gb"] = titles.str.extract(r'(\d+)\s*gb\s*(?:ram|memory|ddr)', expand=False)
    df["extracted_ram_gb"] = pd.to_numeric(df["extracted_ram_gb"], errors="coerce")

    # Storage: e.g. "512GB SSD", "1TB HDD", "2TB NVMe"
    storage_match = titles.str.extract(r'(\d+(?:\.\d+)?)\s*(tb|gb)\s*(?:ssd|hdd|nvme|storage|emmc)', expand=True)
    def _to_gb(row):
        if pd.isna(row[0]):
            return np.nan
        val = float(row[0])
        return val * 1024 if str(row[1]).lower() == "tb" else val
    df["extracted_storage_gb"] = storage_match.apply(_to_gb, axis=1)

    # Display size: e.g. "15.6 inch", "13.3""
    df["extracted_display_inch"] = titles.str.extract(r'(\d+\.?\d*)\s*(?:inch|"|-inch|\'\')', expand=False)
    df["extracted_display_inch"] = pd.to_numeric(df["extracted_display_inch"], errors="coerce")

    # GPU hint
    df["has_dedicated_gpu"] = titles.str.contains(
        r'rtx|gtx|radeon|nvidia|geforce|arc\s+\w+\s+gpu', regex=True
    )

    # Processor hint
    df["processor_hint"] = titles.str.extract(
        r'(i3|i5|i7|i9|ryzen\s*\d|snapdragon|apple\s*m\d|m1|m2|m3|celeron|pentium)', expand=False
    ).fillna("unknown")

    return df


def _apply_budget_filter(df: pd.DataFrame, requirements: dict) -> pd.DataFrame:
    """Filter products within the user's budget range."""
    budget_min = requirements.get("budget_min")
    budget_max = requirements.get("budget_max")

    if budget_max:
        # Allow 10% over budget to catch good deals
        df = df[df["product_price"] <= budget_max * 1.10]
    if budget_min:
        df = df[df["product_price"] >= budget_min * 0.90]

    return df


def _compute_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a normalized popularity score combining rating and review count.
    Uses Bayesian average to avoid bias from products with very few reviews.
    """
    C = df["product_star_rating"].mean()  # global mean
    m = df["product_num_ratings"].quantile(0.25)  # minimum threshold

    df["popularity_score"] = (
        (df["product_num_ratings"] / (df["product_num_ratings"] + m)) * df["product_star_rating"] +
        (m / (df["product_num_ratings"] + m)) * C
    )

    # Normalize to 0-1
    min_pop = df["popularity_score"].min()
    max_pop = df["popularity_score"].max()
    if max_pop > min_pop:
        df["popularity_score"] = (df["popularity_score"] - min_pop) / (max_pop - min_pop)
    else:
        df["popularity_score"] = 0.5

    return df


def build_feature_vector(product_row: pd.Series, requirements: dict) -> np.ndarray:
    """
    Build a numerical feature vector for a product to use in ML scoring.
    Returns array: [price_score, rating_score, ram_match, storage_match, gpu_match, display_match]
    """
    budget_max = requirements.get("budget_max") or product_row["product_price"] * 1.5
    budget_max = max(budget_max, 1)

    # Price score: lower price within budget = better score
    price_ratio = product_row["product_price"] / budget_max
    price_score = max(0, 1 - price_ratio)  # 1.0 = very cheap, 0 = at/over budget

    # Rating score (normalized)
    rating_score = product_row["product_star_rating"] / 5.0

    # RAM match
    req_ram = requirements.get("priority_specs", {}).get("ram_gb")
    extracted_ram = product_row.get("extracted_ram_gb", np.nan)
    if req_ram and not pd.isna(extracted_ram):
        ram_score = min(extracted_ram / req_ram, 1.5) / 1.5  # capped at 1.5x required
    else:
        ram_score = 0.5  # neutral if unknown

    # Storage match
    req_storage = requirements.get("priority_specs", {}).get("storage_gb")
    extracted_storage = product_row.get("extracted_storage_gb", np.nan)
    if req_storage and not pd.isna(extracted_storage):
        storage_score = min(extracted_storage / req_storage, 2.0) / 2.0
    else:
        storage_score = 0.5

    # GPU match
    req_gpu = requirements.get("priority_specs", {}).get("gpu_required", False)
    has_gpu = product_row.get("has_dedicated_gpu", False)
    gpu_score = 1.0 if (req_gpu and has_gpu) else (0.5 if not req_gpu else 0.2)

    # Display match
    req_display = requirements.get("priority_specs", {}).get("display_size")
    extracted_display = product_row.get("extracted_display_inch", np.nan)
    if req_display and not pd.isna(extracted_display):
        display_score = max(0, 1 - abs(extracted_display - req_display) / req_display)
    else:
        display_score = 0.5

    # Popularity
    pop_score = product_row.get("popularity_score", 0.5)

    return np.array([price_score, rating_score, ram_score, storage_score, gpu_score, display_score, pop_score])


def products_to_dict_list(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of clean dicts for JSON serialization."""
    result = []
    for _, row in df.iterrows():
        result.append({
            "asin": row.get("asin", ""),
            "product_title": row.get("product_title", ""),
            "product_price": float(row.get("product_price", 0)),
            "product_price_str": f"${row.get('product_price', 0):,.2f}",
            "product_star_rating": float(row.get("product_star_rating", 0)),
            "product_num_ratings": int(row.get("product_num_ratings", 0)),
            "product_url": row.get("product_url", ""),
            "product_photo": row.get("product_photo", ""),
            "is_best_seller": bool(row.get("is_best_seller", False)),
            "is_amazon_choice": bool(row.get("is_amazon_choice", False)),
            "specs": {
                "RAM": f"{int(row['extracted_ram_gb'])}GB" if not pd.isna(row.get("extracted_ram_gb", float("nan"))) else "N/A",
                "Storage": f"{int(row['extracted_storage_gb'])}GB" if not pd.isna(row.get("extracted_storage_gb", float("nan"))) else "N/A",
                "Display": f"{row['extracted_display_inch']:.1f}\"" if not pd.isna(row.get("extracted_display_inch", float("nan"))) else "N/A",
                "GPU": "Dedicated" if row.get("has_dedicated_gpu") else "Integrated",
                "Processor": str(row.get("processor_hint", "N/A")).title()
            },
            "match_score": float(row.get("match_score", 0)),
            "popularity_score": float(row.get("popularity_score", 0))
        })
    return result
