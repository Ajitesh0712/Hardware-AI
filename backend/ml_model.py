from sklearn.ensemble import RandomForestRegressor
import numpy as np
# Features: [price, rating, ram, storage, gpu, battery]

X = np.array([
    [80000, 4.5, 16, 512, 1, 1],
    [75000, 4.4, 16, 512, 1, 1],
    [70000, 4.2, 16, 512, 1, 0],
    [65000, 4.1, 8, 512, 0, 0],
    [60000, 4.0, 8, 256, 0, 0],
    [55000, 3.9, 8, 256, 0, 0],
    [50000, 3.8, 8, 256, 0, 0],
    [45000, 3.7, 8, 256, 0, 0],
    [40000, 3.6, 8, 256, 0, 0],
    [35000, 3.5, 8, 256, 0, 0],

    [90000, 4.6, 16, 512, 1, 1],
    [100000, 4.7, 16, 1024, 1, 1],
    [110000, 4.8, 32, 1024, 1, 1],
    [120000, 4.7, 32, 1024, 1, 1],
    [130000, 4.8, 32, 1024, 1, 1],

    [85000, 4.5, 16, 512, 1, 1],
    [78000, 4.3, 16, 512, 1, 0],
    [72000, 4.2, 16, 512, 1, 0],
    [68000, 4.1, 16, 512, 0, 0],

    [60000, 4.0, 8, 512, 0, 1],
    [58000, 4.1, 8, 512, 0, 1],
    [55000, 4.0, 8, 512, 0, 1],

    [70000, 4.2, 16, 512, 0, 1],
    [72000, 4.3, 16, 512, 0, 1],
    [75000, 4.4, 16, 512, 1, 1],

    [95000, 4.6, 16, 1024, 1, 1],
    [105000, 4.7, 16, 1024, 1, 1],

    [50000, 3.9, 8, 256, 0, 0],
    [48000, 3.8, 8, 256, 0, 0],
    [47000, 3.7, 8, 256, 0, 0],

    [65000, 4.2, 8, 512, 0, 1],
    [67000, 4.3, 8, 512, 0, 1],

    [80000, 4.5, 16, 512, 1, 1],
    [82000, 4.6, 16, 512, 1, 1],
    [84000, 4.6, 16, 512, 1, 1],

    [90000, 4.7, 16, 1024, 1, 1],
    [92000, 4.7, 16, 1024, 1, 1],

    [100000, 4.8, 32, 1024, 1, 1],
    [110000, 4.8, 32, 1024, 1, 1],
])

y = np.array([
    0.85, 0.83, 0.80, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
    0.88, 0.90, 0.95, 0.94, 0.96,
    0.87, 0.82, 0.80, 0.75,
    0.70, 0.72, 0.71,
    0.78, 0.80, 0.84,
    0.88, 0.90,
    0.55, 0.52, 0.50,
    0.73, 0.75,
    0.85, 0.87, 0.88,
    0.90, 0.91,
    0.95, 0.96
])

model = RandomForestRegressor()
model.fit(X, y)


def extract_features(product):
    price = product.get("product_price", 0)
    rating = product.get("product_star_rating", 0)

    ram = 16 if "16" in product.get("product_title", "") else 8
    storage = 512 if "512" in product.get("product_title", "") else 256
    gpu = 1 if "RTX" in product.get("product_title", "") else 0
    battery = 1 if "battery" in product.get("product_title", "").lower() else 0

    return [price, rating, ram, storage, gpu, battery]


def predict_score(product,requirements):
    features = extract_features(product)

    score = model.predict([features])[0]

    #adjust based on user intent
    use_case = requirements.get("use_case", "general")

    if use_case == "gaming":
        if "RTX" in product.get("product_title", ""):
            score += 0.1

    if requirements.get("budget_max"):
        if product.get("product_price", 0) <= requirements["budget_max"]:
            score += 0.1

    return float(score)