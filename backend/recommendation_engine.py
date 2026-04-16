"""
recommendation_engine.py
ML-based recommendation system using Scikit-learn.
Computes match scores between user requirements and products.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from data_processor import build_feature_vector
from ml_model import predict_score


# Weights for each feature dimension based on use case priorities
USE_CASE_WEIGHTS = {
    "gaming": {
        "price": 0.10,
        "rating": 0.15,
        "ram": 0.15,
        "storage": 0.10,
        "gpu": 0.35,
        "display": 0.05,
        "popularity": 0.10
    },
    "coding": {
        "price": 0.15,
        "rating": 0.15,
        "ram": 0.30,
        "storage": 0.20,
        "gpu": 0.05,
        "display": 0.05,
        "popularity": 0.10
    },
    "design": {
        "price": 0.10,
        "rating": 0.15,
        "ram": 0.20,
        "storage": 0.15,
        "gpu": 0.20,
        "display": 0.15,
        "popularity": 0.05
    },
    "3d rendering": {
        "price": 0.05,
        "rating": 0.10,
        "ram": 0.25,
        "storage": 0.15,
        "gpu": 0.35,
        "display": 0.05,
        "popularity": 0.05
    },
    "video editing": {
        "price": 0.10,
        "rating": 0.15,
        "ram": 0.25,
        "storage": 0.20,
        "gpu": 0.15,
        "display": 0.10,
        "popularity": 0.05
    },
    "student": {
        "price": 0.35,
        "rating": 0.20,
        "ram": 0.15,
        "storage": 0.10,
        "gpu": 0.03,
        "display": 0.07,
        "popularity": 0.10
    },
    "business": {
        "price": 0.20,
        "rating": 0.25,
        "ram": 0.15,
        "storage": 0.10,
        "gpu": 0.03,
        "display": 0.07,
        "popularity": 0.20
    },
    "general": {
        "price": 0.20,
        "rating": 0.25,
        "ram": 0.15,
        "storage": 0.10,
        "gpu": 0.05,
        "display": 0.05,
        "popularity": 0.20
    }
}

DEFAULT_WEIGHTS = USE_CASE_WEIGHTS["general"]


def recommend(df: pd.DataFrame, requirements: dict, top_n: int = 5) -> pd.DataFrame:
    """
    Main recommendation function.
    Scores each product against user requirements and returns top N ranked products.
    """
    if df.empty:
        return df

    use_case = requirements.get("use_case", "general").lower()
    weights = USE_CASE_WEIGHTS.get(use_case, DEFAULT_WEIGHTS)
    weight_vector = np.array(list(weights.values()))


    # 🔥 ML-based scoring
    ml_scores = []

    for _, row in df.iterrows():
        product_dict = row.to_dict()
        score = predict_score(product_dict,requirements)
        ml_scores.append(score)

    scores = np.array(ml_scores)

    # Normalize to 0–100
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min()) * 100
    else:
        scores = np.full(len(scores), 75.0)

    df = df.copy()
    df["match_score"] = scores

    # Apply best_seller and amazon_choice bonuses
    df.loc[df["is_best_seller"] == True, "match_score"] += 3
    df.loc[df["is_amazon_choice"] == True, "match_score"] += 2
    df["match_score"] = df["match_score"].clip(0, 100)

    # Sort by match score descending
    df = df.sort_values("match_score", ascending=False)

    # Deduplicate by similar titles (keep highest scored)
    df = _deduplicate_products(df)

    return df.head(top_n).reset_index(drop=True)


def _deduplicate_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove near-duplicate products using title similarity.
    Keeps the highest-scored version of similar products.
    """
    seen_titles = []
    keep_indices = []

    for idx, row in df.iterrows():
        title = str(row["product_title"]).lower()
        title_words = set(title.split())

        is_duplicate = False
        for seen in seen_titles:
            seen_words = set(seen.split())
            if len(title_words) == 0:
                continue
            overlap = len(title_words & seen_words) / len(title_words | seen_words)
            if overlap > 0.7:  # 70% similarity threshold
                is_duplicate = True
                break

        if not is_duplicate:
            seen_titles.append(title)
            keep_indices.append(idx)

    return df.loc[keep_indices]


def compute_similarity_matrix(df: pd.DataFrame, requirements: dict) -> np.ndarray:
    """
    Compute a cosine similarity matrix between all products.
    Used for 'similar products' feature.
    """
    if df.empty or len(df) < 2:
        return np.array([])

    feature_matrix = np.array([
        build_feature_vector(row, requirements) for _, row in df.iterrows()
    ])

    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    return cosine_similarity(feature_matrix)


def get_match_label(score: float) -> str:
    """Return a human-readable match quality label."""
    if score >= 90:
        return "Perfect Match"
    elif score >= 75:
        return "Excellent Match"
    elif score >= 60:
        return "Good Match"
    elif score >= 45:
        return "Fair Match"
    else:
        return "Partial Match"
