"""Shared product title parsing for offline training and live inference.

The training CSV and live Amazon scraper use different source field names.
This module maps both to one stable feature schema so the saved estimator
always receives the same columns in the same order.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "price_log1p",
    "rating",
    "review_count_log1p",
    "discount_percent",
    "gaming",
    "wireless",
    "rgb",
    "battery_present",
    "premium_brand",
    "fast_charging",
    "ram_gb",
    "storage_gb",
    "battery_mah",
    "screen_inches",
    "charging_w",
    "refresh_rate_hz",
]

_NON_ELECTRONIC = [
    r"\bscissors?\b", r"\b(?:ball|gel|fountain|roller) pen\b",
    r"\bwhiteboard\b", r"\bposter colou?r\b", r"\bpaint brush\b",
    r"\bstationery\b", r"\bnotebook diary\b", r"\bflash cards?\b",
    r"\be-gift card\b", r"\bcopy(?:ier)? paper\b", r"\berasers?\b",
    r"\bhighlighters?\b", r"\bgraphite pencils?\b", r"\bacrylic colou?rs?\b",
]

_ACCESSORIES = {
    "Mobile Accessories": [
        r"\bphone case\b", r"\bmobile case\b", r"\bback cover\b",
        r"\bphone cover\b", r"\btempered glass\b", r"\bscreen protector\b",
        r"\bphone holder\b", r"\bmobile holder\b", r"\bphone mount\b",
        r"\bphone stand\b", r"\bscreen guard\b", r"\bcable protector\b",
    ],
    "Smartwatch Accessories": [
        r"\bwatch strap\b", r"\bwatch band\b", r"\breplacement band\b",
        r"\bsmart ?watch charger\b", r"\bwatch charger\b",
    ],
    "Computer Accessories": [
        r"\blaptop bag\b", r"\blaptop sleeve\b", r"\blaptop stand\b",
        r"\blaptop riser\b", r"\bcooling pad\b", r"\bmouse pad\b",
        r"\bkeyboard protector\b", r"\bkeyboard cover\b", r"\blaptop screen protector\b",
        r"\bcomputer mouse\b", r"\bwireless mouse\b", r"\bmouse\b", r"\bkeyboard\b",
        r"\bmonitor stand\b", r"\busb hub\b",
    ],
    "Camera Accessories": [
        r"\btripod\b", r"\bmonopod\b", r"\bselfie stick\b",
        r"\bcamera mount\b", r"\bcamera holder\b", r"\bring light\b",
    ],
}

_OTHER_CATEGORIES = [
    ("Gaming Accessories", [r"\bgamepad\b", r"\bjoystick\b", r"\bconsole controller\b", r"\bgaming controller\b"]),
    ("Remote Controls", [r"\bremote(?: control| controller)?\b"]),
    ("Streaming Devices", [r"\bfire tv stick\b", r"\bstreaming device\b", r"\bset ?top box\b", r"\bdth box\b"]),
    ("Telephones", [r"\blandline phone\b", r"\bcordless telephone\b", r"\bcorded telephone\b"]),
    ("Printers and Printing Supplies", [r"\bprinter\b", r"\bink cartridge\b", r"\bink bottle\b", r"\btoner cartridge\b", r"\btoner reload kit\b", r"\bprinthead\b", r"\bscanner\b"]),
    ("Networking Devices", [r"\brouter\b", r"\bmodem\b", r"\bnetwork switch\b", r"\bwi[\s-]?fi extender\b", r"\bhotspot\b", r"\bnetwork adapter\b"]),
    ("Storage Devices", [r"\bpen ?drive\b", r"\bflash drive\b", r"\busb drive\b", r"\bexternal hard ?disk\b", r"\bexternal ssd\b", r"\bmemory card\b", r"\bmicro ?sd\b", r"\bsd card\b", r"\bportable ssd\b"]),
    ("Computer Components", [r"\bgraphics? card\b", r"\bgpu\b", r"\bmotherboard\b", r"\bdesktop processor\b", r"\bcomputer processor\b", r"\bddr[345]\b", r"\bpc cabinet\b"]),
    ("Batteries", [r"\b(?:aa|aaa|coin|alkaline|rechargeable) batteries?\b", r"\bcr\d{4}\b", r"\bbatter(?:y|ies)\b"]),
    ("Power Banks", [r"\bpower bank\b", r"\bportable charger\b"]),
    ("Electrical Accessories", [r"\bextension (?:cord|board)\b", r"\bpower strip\b", r"\bspike guard\b", r"\bsurge guard\b"]),
    ("Chargers and Cables", [r"\bcharger\b", r"\bcharging cable\b", r"\busb cable\b", r"\btype[ -]?c cable\b", r"\bcable\b", r"\bpower adapter\b", r"\bwall adapter\b", r"\bpower cord\b", r"\bhdmi cable\b"]),
]

# Product identity patterns use the start of the title. This avoids calling a
# phone a charger merely because its listing says "charger included".
_PHONE_PREFIX = re.compile(
    r"^(?:\(renewed\)\s*)?(?:"
    r"(?:samsung\s+galaxy\s+(?!tab\b|watch\b)[a-z0-9]+)|"
    r"(?:oneplus\s+(?:nord\s+(?:ce\s*)?\d+[a-z0-9]*|\d+[a-z0-9]*))|"
    r"(?:redmi\s+(?!pad\b|buds?\b|watch\b)[a-z0-9]+)|(?:realme\s+(?!buds?\b|watch\b|band\b)[a-z0-9]+)|"
    r"(?:poco\s+[a-z0-9]+)|(?:iqoo\s+[a-z0-9]+)|"
    r"(?:vivo\s+[a-z0-9]+)|(?:oppo\s+[a-z0-9]+)|"
    r"(?:infinix\s+[a-z0-9]+)|(?:tecno\s+[a-z0-9]+)|"
    r"(?:motorola\s+[a-z0-9]+)|(?:google\s+pixel\s+[a-z0-9]+)|"
    r"(?:pixel\s+[a-z0-9]+)|(?:iphone\s+\d{1,2})|"
    r"(?:lava\s+[a-z0-9]+)|(?:nokia\s+[a-z0-9]+))\b"
)

_SCREENS = {"Smartphones", "Laptops", "Tablets", "Smartwatches", "Televisions", "Monitors and Projectors"}
_BATTERY_CATEGORIES = {"Smartphones", "Tablets", "Smartwatches", "Headphones and Earphones", "Speakers", "Power Banks", "Networking Devices", "Cameras", "Batteries"}


def classify_product(title: Any) -> str:
    text = re.sub(r"\s+", " ", str(title or "").lower()).strip()

    if any(re.search(p, text) for p in _NON_ELECTRONIC):
        return "Non-electronic / Miscellaneous"

    # Accessory identity is more specific than a compatible device brand.
    for category, patterns in _ACCESSORIES.items():
        if category == "Computer Accessories" and re.search(r"\b(?:laptop|macbook|chromebook|notebook pc)\b", text):
            # Product titles often list a laptop's keyboard or mouse support.
            # Explicit bag/stand/sleeve patterns still matched above this guard.
            narrowed = [p for p in patterns if p not in {r"\bkeyboard\b", r"\bcomputer mouse\b", r"\bwireless mouse\b", r"\bmouse\b"}]
            if any(re.search(p, text) for p in narrowed):
                return category
            continue
        if any(re.search(p, text) for p in patterns):
            return category

    if _PHONE_PREFIX.search(text):
        return "Smartphones"

    for category, patterns in _OTHER_CATEGORIES:
        if any(re.search(p, text) for p in patterns):
            return category

    if re.search(r"^(?:apple\s+)?iphone\s+\d{1,2}\b", text) or re.search(r"^(?:smartphone|mobile phone|cell phone)\b", text):
        return "Smartphones"

    if re.search(r"\b(?:earbuds?|headphones?|earphones?|headsets?|neckband|tws|airpods)\b", text):
        return "Headphones and Earphones"

    patterns = [
        ("Headphones and Earphones", r"\b(?:earbuds?|headphones?|earphones?|headsets?|neckband|tws|airpods)\b"),
        ("Smartwatches", r"\b(?:smart ?watch|galaxy watch|fitness watch|fitness tracker|fitness band)\b"),
        ("Cameras", r"\b(?:digital camera|dslr|mirrorless camera|action camera|webcam|camcorder|instant camera)\b"),
        ("Televisions", r"\b(?:television|smart tv|led tv|oled tv|qled tv|android tv|bravia)\b"),
        ("Monitors and Projectors", r"\b(?:monitor|projector|computer display)\b"),
        ("Speakers", r"\b(?:speaker|soundbar|home theatre|home theater|subwoofer)\b"),
        ("Tablets", r"\b(?:tablet|ipad|galaxy tab|lenovo tab)\b"),
        ("Laptops", r"\b(?:laptop|macbook|chromebook|notebook computer|notebook pc)\b"),
    ]
    for category, pattern in patterns:
        if re.search(pattern, text):
            return category
    return "Other / Unclassified"


def _first_number(patterns: list[str], text: str) -> float:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return np.nan


def extract_title_specs(title: Any, category: str | None = None) -> dict[str, float]:
    text = re.sub(r"\s+", " ", str(title or "").lower()).strip()
    category = category or classify_product(text)
    out = {k: np.nan for k in ("ram_gb", "storage_gb", "battery_mah", "screen_inches", "charging_w", "refresh_rate_hz")}

    # RAM and storage are only meaningful in device/component/storage titles.
    memory_categories = {"Smartphones", "Laptops", "Tablets", "Computer Components", "Storage Devices"}
    if category in memory_categories:
        for ram in re.finditer(r"(\d{1,3})\s*gb\s*ram\b", text):
            context = text[max(0, ram.start()-22):min(len(text), ram.end()+24)]
            if re.search(r"(?:up ?to|virtual|extendable)|ram plus", context):
                continue
            value = float(ram.group(1))
            if 1 <= value <= 128:
                out["ram_gb"] = value
                break
        if pd.isna(out["ram_gb"]):
            ram = re.search(r"(\d{1,3})\s*gb\s*[+,]\s*\d{1,4}\s*gb\s*(?:storage|rom)\b", text)
            if ram and 1 <= int(ram.group(1)) <= 128:
                out["ram_gb"] = float(ram.group(1))
        storage = re.search(r"(\d{1,4})\s*gb\s*(?:storage|rom|capacity|ssd|hdd|nvme|emmc)\b|(\d{1,3})\s*tb\s*(?:storage|ssd|hdd|nvme)\b", text)
        if storage:
            if storage.group(1):
                value = float(storage.group(1))
            else:
                value = float(storage.group(2)) * 1024
            if 1 <= value <= 8192:
                out["storage_gb"] = value
        if pd.isna(out["storage_gb"]) and category in {"Smartphones", "Tablets"}:
            capacity = re.search(r"(\d{1,2})\s*gb\s*(?:ram)?\s*[,+]\s*(\d{2,4})\s*gb\b", text)
            if capacity and 16 <= int(capacity.group(2)) <= 2048:
                out["storage_gb"] = float(capacity.group(2))

    if category in _BATTERY_CATEGORIES:
        value = _first_number([r"(\d{3,5})\s*mah\b"], text)
        if 100 <= value <= 100000:
            out["battery_mah"] = value

    if category in _SCREENS:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:inches?\b|[\"″])", text)
        if match:
            value = float(match.group(1))
            context = text[max(0, match.start()-35):match.end()+50]
            display_terms = ("display", "screen", "amoled", "oled", "lcd", " led", "resolution", "touchscreen", "retina", "laptop")
            if 1 <= value <= 100 and (category in {"Televisions", "Monitors and Projectors"} or any(t in context for t in display_terms)):
                out["screen_inches"] = value

    if category in {"Chargers and Cables", "Power Banks", "Smartphones", "Tablets", "Smartwatches"}:
        for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*w(?:att)?\b", text):
            context = text[max(0, m.start()-32):m.end()+40]
            if any(t in context for t in ("charg", "adapter", "power delivery", "pd ", "supervooc", "warp charge", "hypercharge")):
                value = float(m.group(1))
                if 1 <= value <= 300:
                    out["charging_w"] = value
                    break

    if category in _SCREENS:
        for m in re.finditer(r"(\d{2,3})\s*hz\b", text):
            context = text[max(0, m.start()-35):m.end()+35]
            if any(t in context for t in ("refresh", "display", "screen", "amoled", "oled", "monitor", "gaming")):
                value = float(m.group(1))
                if 30 <= value <= 360:
                    out["refresh_rate_hz"] = value
                    break
    return out


def _numeric(row: Mapping[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value):
            try:
                return float(str(value).replace(",", "").replace("₹", "").strip())
            except (TypeError, ValueError):
                continue
    return np.nan


def product_features(row: Mapping[str, Any]) -> dict[str, float]:
    """Return one fixed, model-ready row for offline or live product data."""
    title = str(row.get("name") or row.get("product_title") or row.get("title") or "")
    price = _numeric(row, ("discount_price", "product_price", "price"))
    actual = _numeric(row, ("actual_price", "product_original_price", "list_price"))
    rating = _numeric(row, ("review_rating", "product_star_rating", "rating"))
    reviews = _numeric(row, ("no_of_ratings", "product_num_ratings", "review_count"))
    discount = np.nan
    if np.isfinite(price) and np.isfinite(actual) and actual > 0:
        discount = float(np.clip((actual - price) / actual * 100, 0, 100))
    text = title.lower()
    category = classify_product(title)
    features = {
        "price_log1p": np.log1p(price) if np.isfinite(price) and price >= 0 else np.nan,
        "rating": float(np.clip(rating, 0, 5)) if np.isfinite(rating) else np.nan,
        "review_count_log1p": np.log1p(max(reviews, 0)) if np.isfinite(reviews) else np.nan,
        "discount_percent": discount,
        "gaming": float(any(t in text for t in ("gaming", "geforce", "rtx", "gtx"))),
        "wireless": float("wireless" in text or "bluetooth" in text),
        "rgb": float("rgb" in text),
        "battery_present": float(bool(re.search(r"\b(?:battery|\d{3,5}\s*mah)\b", text))),
        "premium_brand": float(any(t in text for t in ("apple", "samsung", "sony"))),
        "fast_charging": float(any(t in text for t in ("fast charging", "fast charge", "quick charge", "supervooc", "warp charge", "hypercharge"))),
    }
    features.update(extract_title_specs(title, category))
    return {key: features.get(key, np.nan) for key in FEATURE_COLUMNS}


def product_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([product_features(row) for row in frame.to_dict("records")], columns=FEATURE_COLUMNS, index=frame.index)
