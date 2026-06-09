import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)
import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

csv_path = os.path.join(
    BASE_DIR,
    "electronics.csv"
)

df = pd.read_csv(csv_path)


df["discount_price"] = (
    df["discount_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["discount_price"] = pd.to_numeric(
    df["discount_price"],
    errors="coerce"
)

df["actual_price"] = (
    df["actual_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["actual_price"] = pd.to_numeric(
    df["actual_price"],
    errors="coerce"
)

# Clean number of ratings
df["no_of_ratings"] = (
    df["no_of_ratings"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["no_of_ratings"] = pd.to_numeric(
    df["no_of_ratings"],
    errors="coerce"
)

# Clean review ratings
df["review_rating"] = (
    df["review_rating"]
    .astype(str)
    .str.extract(r'(\d+\.\d+)')[0]
)

df["review_rating"] = pd.to_numeric(
    df["review_rating"],
    errors="coerce"
)

# Remove invalid rows
df.dropna(inplace=True)

def extract_features(product):

    title = str(
        product.get("name")
        or product.get("product_title")
        or ""
    ).lower()

    price = float(
        product.get("discount_price")
        or product.get("product_price")
        or 0
    )

    rating = float(
        product.get("review_rating")
        or product.get("product_star_rating")
        or 0
    )

    review_count = float(
        product.get("no_of_ratings")
        or 0
    )


    gaming = 1 if (
        "gaming" in title or
        "rtx" in title
    ) else 0

    wireless = 1 if "wireless" in title else 0

    rgb = 1 if "rgb" in title else 0

    battery = 1 if (
        "battery" in title or
        "mah" in title
    ) else 0

    premium = 1 if (
        "apple" in title or
        "samsung" in title or
        "sony" in title
    ) else 0

    fast_charging = 1 if (
        "fast charging" in title or
        "65w" in title or
        "120w" in title
    ) else 0

    return [
        price,
        rating,
        review_count,
        gaming,
        wireless,
        rgb,
        battery,
        premium,
        fast_charging
    ]


df["discount_percent"] = (
    (df["actual_price"] - df["discount_price"])
    / df["actual_price"]
)

df["target_score"] = (
    (df["review_rating"] * 0.45)
    +
    (np.log1p(df["no_of_ratings"]) * 0.20)
    +
    (df["discount_percent"] * 0.20)
)

# Normalize target score
df["target_score"] = (
    (df["target_score"] - df["target_score"].min())
    /
    (
        df["target_score"].max()
        - df["target_score"].min()
    )
)


X = np.array([
    extract_features(row)
    for _, row in df.iterrows()
])

y = df["target_score"].values


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("[ML] Model trained successfully.")


def predict_score(product, requirements):

    title = str(
        product.get("name")
        or product.get("product_title")
        or ""
    ).lower()
    

    query = str(
        requirements.get("search_query", "")
    ).lower()

    features = extract_features(product)

    score = model.predict([features])[0]
    # =========================
    # BRAND INTENT BOOST
    # =========================

    query = str(
        requirements.get("search_query", "")
    ).lower()

    brand_map = {
        "iphone": ["iphone", "apple"],
        "apple": ["iphone", "apple"],
        "samsung": ["samsung", "galaxy"],
        "oneplus": ["oneplus"],
        "lenovo": ["lenovo"],
        "asus": ["asus"],
        "dell": ["dell"],
        "hp": ["hp"],
        "sony": ["sony"],
        "jbl": ["jbl"],
        "boat": ["boat"]
    }

    for brand, keywords in brand_map.items():

        if brand in query:

            matched = any(
                keyword in title
                for keyword in keywords
            )

            if matched:
                score += 0.35
            else:
                score -= 0.25

    use_case = requirements.get("use_case", "general")


    if query and query in title:
        score += 0.30

    query_words = query.split()

    match_count = 0

    ignored_words = [
        "best",
        "under",
        "with",
        "for",
        "the",
        "a",
        "an",
        "top",
        "good"
    ]

    for word in query_words:

        if word in ignored_words:
            continue

        if word in title:
            match_count += 1

    score += match_count * 0.10


    if use_case == "gaming":

        if (
            "gaming" in title or
            "rtx" in title
        ):
            score += 0.12

        if (
            "144hz" in title or
            "165hz" in title
        ):
            score += 0.05

        if (
            "mechanical" in title or
            "rgb" in title
        ):
            score += 0.04

    elif use_case == "student":

        if features[0] < 70000:
            score += 0.08

        if "battery" in title:
            score += 0.05

        if "portable" in title:
            score += 0.04

    elif use_case == "business":

        if features[1] >= 4.3:
            score += 0.06

        if (
            "portable" in title or
            "lightweight" in title
        ):
            score += 0.05

    elif use_case == "music":

        if (
            "noise cancelling" in title or
            "wireless" in title
        ):
            score += 0.08


    budget_max = requirements.get("budget_max")

    if budget_max is not None:

        if features[0] <= budget_max:
            score += 0.10
        else:
            score -= 0.08



    if features[1] >= 4.5:
        score += 0.05

    elif features[1] <= 3.5:
        score -= 0.05

    if features[2] > 50000:
        score += 0.05

    elif features[2] < 100:
        score -= 0.03



    if features[7]:
        score += 0.08



    if features[8]:
        score += 0.03

 

    if features[4]:
        score += 0.03


    if features[5]:
        score += 0.02

    if features[6]:
        score += 0.03


    if (
        features[0] > 150000 and
        features[1] < 4.2
    ):
        score -= 0.08


    score = max(0.0, min(score, 1.0))

    return float(score)