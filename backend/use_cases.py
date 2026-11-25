"""Turn what a user wants to *do* into laptop hardware needs.

Users often write "laptop for video editing and 3D modelling" instead of
"RTX 4060, 16 GB RAM". This module

  1. finds every use case in the request (there can be more than one),
  2. looks up simple, stated hardware needs for each one,
  3. scores how well a listing title meets those needs (0 to 1 per need).

The needs are rules of thumb for a college project, not benchmarks. Levels
run from 0 (unsuitable) to 1 (strong); "unknown" gets half credit.
"""

import re

# Pattern -> use-case name. Every match is kept, in this order.
USE_CASE_PATTERNS = [
    ("3d modelling", r"\b(?:3d|blender|maya|cad|autocad|solidworks|revit|rendering|render|modell?ing|unreal|unity)\b"),
    ("video editing", r"\b(?:video\s*edit\w*|edit\w*\s+videos?|premiere|davinci|final cut|after effects|content creat\w*|youtube)\b"),
    ("gaming", r"\b(?:gaming|gamer|games?|esports)\b"),
    ("design", r"\b(?:graphic design\w*|designing|designer|photoshop|illustrator|photo edit\w*|lightroom|design)\b"),
    ("coding", r"\b(?:coding|programming|programmer|developer|software development|data science|machine learning)\b"),
    ("student", r"\b(?:students?|study|studying|studies|school|college|university|online classes|notes)\b"),
    ("business", r"\b(?:business|office|work from home|excel|meetings?)\b"),
    # "good battery", "long battery life", "don't want to keep charging it"
    ("long battery", r"\b(?:battery|backup|all[ -]day|unplugged|(?:keep|keeps|constant(?:ly)?|always)\s+(?:it\s+)?charg\w*|charg\w*\s+(?:every|all the)\s*time)\b"),
]

# Minimum RAM / storage (GB) and wanted GPU / CPU level for each use case.
LAPTOP_NEEDS = {
    "3d modelling":  {"ram_gb": 16, "storage_gb": 512, "gpu": 0.8, "cpu": 0.9},
    "video editing": {"ram_gb": 16, "storage_gb": 512, "gpu": 0.55, "cpu": 0.9},
    "gaming":        {"ram_gb": 16, "storage_gb": 512, "gpu": 0.8, "cpu": 0.7},
    "design":        {"ram_gb": 16, "storage_gb": 512, "gpu": 0.4, "cpu": 0.7},
    "coding":        {"ram_gb": 16, "storage_gb": 512, "gpu": 0.0, "cpu": 0.6},
    "student":       {"ram_gb": 8,  "storage_gb": 256, "gpu": 0.0, "cpu": 0.3},
    "business":      {"ram_gb": 8,  "storage_gb": 256, "gpu": 0.0, "cpu": 0.3},
    "long battery":  {"ram_gb": 8,  "storage_gb": 256, "gpu": 0.0, "cpu": 0.0, "battery": 0.7},
}

# Words that describe a use, so they are not expected in a product title.
USE_CASE_WORDS = {
    "3d", "blender", "maya", "cad", "autocad", "solidworks", "revit", "rendering", "render",
    "modelling", "modeling", "unreal", "unity", "video", "videos", "editing", "edit", "editor",
    "premiere", "davinci", "content", "creation", "creator", "youtube", "gaming", "gamer",
    "game", "games", "esports", "graphic", "design", "designing", "designer", "photoshop",
    "illustrator", "photo", "lightroom", "coding", "programming", "programmer", "developer",
    "software", "development", "data", "science", "machine", "learning", "student", "students",
    "school", "college", "university", "online", "classes", "studies", "business", "office",
    "work", "home", "excel", "meeting", "meetings", "study", "studying", "notes",
    "battery", "backup", "life", "day", "all", "unplugged", "charge", "charging", "charged",
}


def detect_use_cases(text):
    """All use cases named in the text, e.g. ['3d modelling', 'video editing']."""
    query = str(text or "").lower()
    return [name for name, pattern in USE_CASE_PATTERNS if re.search(pattern, query)]


def combined_needs(use_cases):
    """The strictest need across all use cases (largest RAM, best GPU, ...)."""
    needs = {}
    for name in use_cases:
        for key, value in LAPTOP_NEEDS.get(name, {}).items():
            needs[key] = max(needs.get(key, 0), value)
    return needs


def laptop_search_hint(use_cases):
    """Words to add to the Amazon search so it returns suitable laptops."""
    needs = combined_needs(use_cases)
    words = []
    if needs.get("battery", 0) > 0 and needs.get("gpu", 0) < 0.5:
        words.append("long battery life")
    if needs.get("gpu", 0) >= 0.8:
        words.append("rtx 4060")     # a hint only: the app turns off exact-model filtering
    elif needs.get("gpu", 0) >= 0.5:
        words.append("rtx")
    if needs.get("ram_gb", 0) >= 16:
        words.append("16gb ram")
    return " ".join(words)


# ---------------------------------------------------------------------------
# Reading hardware levels from a listing title
# ---------------------------------------------------------------------------

def gpu_level(title):
    """0 (no real graphics) to 1 (high-end) from the title."""
    text = str(title or "").lower()
    rtx = re.search(r"\brtx\s*-?\s*(\d{2})(\d{2})\b", text)
    if rtx:
        series, tier = int(rtx.group(1)), int(rtx.group(2))
        level = {50: 0.55, 60: 0.8, 70: 0.95}.get(tier, 1.0 if tier >= 80 else 0.4)
        if series <= 20:
            level -= 0.25      # e.g. RTX 2050
        elif series == 30:
            level -= 0.1       # e.g. RTX 3050 / 3060
        return max(0.1, min(1.0, level))
    if re.search(r"\bm\d\s*max\b", text):
        return 1.0
    if re.search(r"\bm\d\s*pro\b", text):
        return 0.85
    if re.search(r"\b(?:apple\s+)?m\d\b(?:\s+chip)?", text) and re.search(r"\bmacbook\b", text):
        return 0.55
    if re.search(r"\b(?:radeon\s+)?rx\s*\d{4}", text):
        return 0.6
    if re.search(r"\bgtx\s*\d{3,4}\b", text):
        return 0.3
    if re.search(r"\b(?:rtx|geforce|dedicated graphics)\b", text):
        return 0.5
    if re.search(r"\barc\s*a\d{3}m?\b", text):
        return 0.35
    return 0.15  # a laptop title with no GPU named usually has integrated graphics


def cpu_level(title):
    """0 (very basic) to 1 (top class) from the title; 0.5 if not stated."""
    text = str(title or "").lower()
    if re.search(r"\bm\d\s*max\b", text):
        return 1.0
    if re.search(r"\bm\d\s*pro\b", text):
        return 0.95
    if re.search(r"\b(?:celeron|pentium|athlon|mediatek|kompanio)\b", text):
        return 0.1
    tier = re.search(r"\b(?:core\s*)?i([3579])\b|\bi([3579])-|\bryzen\s*([3579])\b|\b(?:core\s+)?ultra\s*([3579])\b|\bcore\s*([3579])\b", text)
    if tier:
        number = int(next(group for group in tier.groups() if group))
        level = {9: 1.0, 7: 0.85, 5: 0.55, 3: 0.3}[number]
        if re.search(r"\b(?:i[3579]-?\d{4,5}|ryzen\s*[3579]\s*\d{4}|ultra\s*[3579]\s*\d{3})h[xs]?\b", text):
            level += 0.1   # H / HX chips are the faster laptop CPUs
        return min(1.0, level)
    if re.search(r"\bmacbook\b", text) and re.search(r"\bm\d\b", text):
        return 0.75
    return 0.5


def battery_hours(title, specs=None):
    """Battery hours from the spec table or title ("up to 12 hours"), else None.

    These are the maker's figures, usually measured with light use.
    """
    value = None
    for key, text in (specs or {}).items():
        if key.strip().lower() == "battery life":
            value = text
    text = str(value or title or "").lower()
    match = re.search(r"(\d{1,2}(?:\.\d)?)\s*(?:\+\s*)?(?:hrs?|hours)\b", text)
    return float(match.group(1)) if match else None


def battery_level(title, specs=None):
    """0 (drains fast) to 1 (lasts all day) from listed hours or the chips used."""
    hours = battery_hours(title, specs)
    if hours:
        for limit, level in ((12, 1.0), (9, 0.8), (7, 0.6), (5, 0.4)):
            if hours >= limit:
                return level
        return 0.2
    text = str(title or "").lower()
    if re.search(r"\b(?:rtx|gtx)\s*-?\s*\d{3,4}|\bradeon\s+rx\b", text):
        return 0.3   # gaming graphics cards use a lot of power
    if re.search(r"\b(?:i[3579]-?\d{4,5}|ryzen\s*[3579]\s*\d{4}|ultra\s*[3579]\s*\d{3})h[xs]?\b", text):
        return 0.4   # H / HX chips trade battery for speed
    if re.search(r"\bmacbook\b|\bsnapdragon\b", text):
        return 0.95  # very power-efficient ARM chips
    if re.search(r"\b(?:i[3579]-?\d{4,5}|ryzen\s*[3579]\s*\d{4}|core\s*[3579]\s*\d{3}|ultra\s*[3579]\s*\d{3})u\b|\bchromebook\b|\blunar lake\b", text):
        return 0.7   # U-series and Chromebooks are made for battery life
    return 0.5


def ram_from_title(title, extracted=None):
    """RAM in GB. Uses the shared extractor first, then '16GB/512GB' style."""
    if extracted is not None and extracted == extracted:  # not NaN
        return float(extracted)
    text = str(title or "").lower()
    match = re.search(r"\b(\d{1,2})\s*gb\s*(?:ram|unified memory|ddr\d|lpddr\d)", text) or re.search(r"\(\s*(\d{1,2})\s*gb\s*/", text)
    return float(match.group(1)) if match else None


def use_case_checks(title, use_cases, ram_gb=None, storage_gb=None):
    """(weight, how well met) pairs for the 1-10 match score."""
    needs = combined_needs(use_cases)
    if not needs:
        return []
    checks = []
    if needs.get("gpu", 0) > 0:
        # The graphics card matters most for 3D, gaming and video work.
        checks.append((5.0 if needs["gpu"] >= 0.5 else 2.5, min(1.0, gpu_level(title) / needs["gpu"])))
    if needs.get("cpu", 0) > 0:
        checks.append((1.5, min(1.0, cpu_level(title) / needs["cpu"])))
    if needs.get("battery", 0) > 0:
        checks.append((4.0, min(1.0, battery_level(title) / needs["battery"])))
    ram = ram_from_title(title, ram_gb)
    checks.append((1.5, 0.5 if ram is None else float(ram >= needs["ram_gb"])))
    if storage_gb is not None and storage_gb == storage_gb:
        checks.append((1.0, float(storage_gb >= needs["storage_gb"])))
    else:
        checks.append((1.0, 0.5))
    return checks


def fails_key_need(title, use_cases):
    """True if the GPU or CPU is clearly too weak (below half of the need).

    Example: a laptop with integrated graphics for 3D modelling. Such a
    listing should not get a high score just because it is in budget.
    """
    needs = combined_needs(use_cases)
    if needs.get("gpu", 0) > 0 and gpu_level(title) / needs["gpu"] < 0.5:
        return True
    if needs.get("cpu", 0) > 0 and cpu_level(title) / needs["cpu"] < 0.5:
        return True
    if needs.get("battery", 0) > 0 and battery_level(title) / needs["battery"] < 0.5:
        return True
    return False

