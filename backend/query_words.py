"""Small text helpers for turning a shopping request into key words.

These are used to
  * search any kind of electronics, not only the fixed categories,
  * decide whether a chat message is a new product or a refinement,
  * check how well a product title matches the request (score out of 10).
"""

import re

# Budget words are handled by parse_budget_limits(), so they are kept in the
# store query for that function but ignored when comparing product titles.
BUDGET_WORDS = {
    "under", "below", "above", "over", "between", "from", "upto", "up", "within",
    "rs", "inr", "k", "lakh", "lac", "thousand",
}

# Words that carry no product meaning ("i need a ...", "show me ...").
FILLER_WORDS = BUDGET_WORDS | {
    "a", "an", "the", "i", "me", "my", "need", "want", "wanna", "looking", "look",
    "for", "show", "find", "get", "buy", "please", "maybe", "some", "any", "something",
    "with", "and", "or", "to", "of", "in", "on", "is", "are", "can", "you", "it", "that",
    "this", "one", "ones", "also", "instead", "like", "just", "only", "other", "options",
    "around", "about", "less", "than", "more", "budget",
    "best", "good", "great", "nice", "new", "latest", "top", "price", "cheap", "cheaper",
    "cheapest", "affordable", "chip", "model", "version",
    "suitable", "smooth", "smoothly", "handle", "handles", "which", "will", "should", "able",
    "use", "using", "usage", "purpose", "task", "tasks", "stuff", "things", "run", "running",
    "heavy", "works", "working", "do", "doing", "be", "its", "all", "well", "lot",
}

# Words that change *how* to pick, not *what* product is wanted.
REFINEMENT_WORDS = {
    "gaming", "gamer", "battery", "life", "long", "lasting", "light", "lightweight",
    "portable", "thin", "slim", "fast", "faster", "performance", "powerful", "ram", "ssd",
    "storage", "memory", "display", "screen", "camera", "bigger", "smaller", "better",
    "higher", "lower", "rated", "rating", "reviews", "student", "students", "office", "work",
    "coding", "programming", "editing", "video", "design", "college", "school", "business",
    "travel", "colour", "color", "black", "white", "blue", "silver", "grey", "gray", "red",
    "inch", "inches", "refresh", "rate", "wireless", "bluetooth", "noise", "cancelling",
    # Spec words: "laptop with i7" or "oled" refine the current search.
    "i3", "i5", "i7", "i9", "intel", "amd", "ryzen", "nvidia", "core", "processor", "cpu",
    "gpu", "graphics", "oled", "amoled", "ips", "fhd", "hd", "qhd", "4k", "retina",
    "touchscreen", "touch", "backlit", "keyboard", "fingerprint", "5g", "4g", "dual", "sim",
    "anc", "waterproof", "fast", "charging", "pro", "max", "plus", "mini", "ultra",
}

# Add-ons that should rank below the main product unless they were asked for.
ACCESSORY_WORDS = {
    "controller", "controllers", "case", "cover", "skin", "skins", "sticker", "stickers",
    "stand", "charger", "cable", "adapter", "bag", "sleeve", "protector", "holder", "mount",
    "strap", "grip", "grips", "dock", "remote", "compatible", "replacement", "game", "games",
    "cooling", "fan", "headset", "tempered", "glass",
}


def normalize_text(text):
    """Lower-case, drop ® and ™, and write 'PlayStation 5' the same way as 'PS5'."""
    text = str(text or "").lower()
    text = re.sub(r"[®™©]", " ", text)
    text = re.sub(r"\bplay\s*station\s*(\d)\b", r"ps\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_number_or_spec(word):
    # "50000", "16gb", "512gb", "144hz", "65w" are specs or budgets, not products.
    return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:k|gb|tb|mb|hz|w|mah|mp|inch)?", word))


def key_words(text):
    """Return the meaningful words of a request, in order, without repeats."""
    words = []
    for word in re.findall(r"[a-z0-9]+", normalize_text(text)):
        if word in FILLER_WORDS or _is_number_or_spec(word) or len(word) < 2:
            continue
        if word not in words:
            words.append(word)
    return words


def product_words(text):
    """Key words that name a product (not refinements such as 'gaming')."""
    return [w for w in key_words(text) if w not in REFINEMENT_WORDS]


def clean_search_text(text):
    """Remove filler words before sending a query to the store."""
    removable = FILLER_WORDS - BUDGET_WORDS
    kept = [w for w in str(text or "").split() if w.lower().strip(",.!?") not in removable]
    return " ".join(kept).strip() or str(text or "").strip()


# Title wording that marks a game or add-on rather than the device itself.
GAME_PATTERNS = (
    r"\b(?:standard|deluxe|complete|gold|ultimate|premium|collector'?s?|launch) (?:edition|edn)\b",
    r"\bedn\b", r"\bvideo game\b", r"\bdigital (?:key|code)\b", r"\bmembership\b",
    r"\bdlc\b", r"\bseason pass\b",
)


def looks_like_addon(title, wanted_words, typed_words):
    """True if the title is an add-on for the wanted product, not the product.

    Examples for wanted_words ["ps5"]: "Disc Drive for PS5", "PS5 Ghost of
    Yotei Complete Edition". Nothing is flagged when the user asked for a
    game or accessory themselves (typed_words contains "game", "controller"...).
    """
    text = normalize_text(title)
    # "Console with DualSense Controller" is a bundle, not a controller, so
    # the few words after "with"/"includes" are not used for this check.
    main_part = re.sub(r"\b(?:with|includes?|including|\+)\s+(?:[a-z0-9]+\s*){1,4}", " ", text)
    title_words = set(re.findall(r"[a-z0-9]+", main_part))
    if (title_words & ACCESSORY_WORDS) - set(typed_words):
        return True
    if set(typed_words) & ACCESSORY_WORDS:
        return False
    for word in wanted_words:
        if re.search(rf"\b(?:for|compatible with)\s+(?:the\s+|sony\s+)?{re.escape(word)}\b", text):
            return True
    return any(re.search(pattern, text) for pattern in GAME_PATTERNS)

