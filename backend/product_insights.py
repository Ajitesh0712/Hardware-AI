"""Explain each product against what the user asked for.

Fills the three boxes on the product page:
  * "Why it fits you"   -> why_it_fits()
  * "Things to know"    -> things_to_know()
  * "Customer comments" -> review_snippets()

Every sentence comes from one of three sources, and nothing is invented:
  1. the listing (title and the product page's spec table),
  2. the user's request (uses, budget, exact model),
  3. real customer reviews (counted mentions, never made-up quotes).
General statements about hardware types (for example that gaming laptops
drain the battery quickly under load) are worded as typical behaviour.
"""

import html
import re

from backend.use_cases import battery_hours, battery_level, combined_needs, cpu_level, detect_use_cases, gpu_level, ram_from_title

USE_NAMES = {
    "3d modelling": "3D modelling", "video editing": "video editing", "gaming": "gaming",
    "design": "design work", "coding": "coding", "student": "studying", "business": "office work",
    "long battery": "long battery life",
}

# What reviewers talk about. A review "mentions" an aspect if any word appears.
ASPECTS = {
    "performance": ["performance", "fast", "speed", "smooth", "lag", "laggy", "slow", "hang", "powerful"],
    "battery life": ["battery", "backup", "drain", "drains"],
    "display": ["display", "screen", "brightness", "colors", "colours"],
    "heat levels": ["heat", "heating", "hot", "temperature", "thermal", "warm"],
    "fan noise": ["fan", "fans", "noise", "noisy", "loud"],
    "build quality": ["build", "sturdy", "plastic", "flimsy", "hinge"],
    "keyboard": ["keyboard", "keys"],
    "sound": ["speaker", "speakers", "sound", "audio"],
    "value for money": ["value", "worth", "money", "price"],
    "delivery or condition": ["delivery", "delivered", "damaged", "defective", "dead", "replacement", "refund", "broken"],
    "customer service": ["service", "support", "warranty"],
}
NEGATIVE_CUES = [
    "not", "no", "never", "bad", "poor", "worst", "issue", "issues", "problem", "problems", "drain",
    "drains", "hot", "heats", "lag", "slow", "hang", "loud", "noisy", "damaged", "defective",
    "dead", "broken", "disappoint", "cons", "weak", "worse", "terrible", "waste",
]
POSITIVE_CUES = [
    "good", "great", "excellent", "amazing", "awesome", "best", "smooth", "fast", "love",
    "nice", "superb", "perfect", "worth", "beauty", "pros", "impressive", "solid",
]


# ---------------------------------------------------------------------------
# Reading facts from the spec table (with the title as a fallback)
# ---------------------------------------------------------------------------

def _spec(specs, *names):
    """Value of a spec: an exact name first, then a name that contains it.

    Exact first matters: "RAM Size" must not pick up "Graphics RAM Size".
    """
    specs = specs or {}
    for name in names:
        for key, value in specs.items():
            if key.strip().lower() == name.lower():
                return value
    for name in names:
        for key, value in specs.items():
            if name.lower() in key.lower() and not (name.lower() == "ram size" and "graphics" in key.lower()):
                return value
    return None


def _number(text):
    match = re.search(r"\d+(?:\.\d+)?", str(text or "").replace(",", ""))
    return float(match.group()) if match else None


def _weight_kg(specs):
    value = _spec(specs, "Item Weight")
    number = _number(value)
    if number is None:
        return None
    return number / 1000 if "gram" in value.lower() and "kilo" not in value.lower() else number


def _gpu_name(title, specs):
    from_specs = _spec(specs, "Graphics Co Processor", "Graphics Coprocessor")
    if from_specs and re.search(r"rtx|gtx|radeon|arc", from_specs, re.I):
        return re.sub(r"\s+", " ", from_specs).strip()
    match = re.search(r"\b(?:nvidia\s+)?(?:geforce\s+)?rtx\s*-?\s*\d{4}(?:\s*ti)?\b|\bradeon\s+rx\s*\d{4}\w*\b", title, re.I)
    if not match:
        return None
    name = re.sub(r"\s+", " ", match.group(0)).strip()
    return re.sub(r"\b(rtx|nvidia|geforce|radeon|rx)\b", lambda m: m.group(1).upper() if m.group(1).lower() in {"rtx", "rx"} else m.group(1).title(), name, flags=re.I)


def _vram_gb(title, specs):
    value = _number(_spec(specs, "Graphics RAM Size", "Graphics Card Ram"))
    if value:
        return value
    match = re.search(r"rtx\s*-?\s*\d{4}\s*[-,(]?\s*(\d{1,2})\s*gb|(\d{1,2})\s*gb\s*(?:gddr\d|rtx)", title, re.I)
    return float(match.group(1) or match.group(2)) if match else None


def _cpu_name(title, specs):
    value = _spec(specs, "CPU Model Number", "CPU Model")
    if value:
        return value
    match = re.search(r"\b(?:intel\s+)?core\s*(?:ultra\s*)?i?[3579][\s-]*\d{0,5}[a-z]{0,2}\b|\bryzen\s*[3579]\s*\d{0,4}[a-z]{0,2}\b|\bapple\s+m\d(?:\s+(?:pro|max))?\b|\bm\d\s+(?:pro|max)?\s*chip\b", title, re.I)
    return match.group(0).strip() if match else None


def _ram_gb(product, specs):
    value = _number(_spec(specs, "RAM Memory Installed", "RAM Size"))
    if value:
        return value
    spec_text = (product.get("specs") or {}).get("RAM", "")
    return _number(spec_text) or ram_from_title(product.get("product_title", ""))


def _storage_gb(product, specs):
    value = _spec(specs, "Hard Drive Size", "Hard Disk Size")
    number = _number(value)
    if number:
        return number * 1024 if value and "tb" in value.lower() else number
    spec_text = (product.get("specs") or {}).get("Storage", "")
    number = _number(spec_text)
    return (number * 1024 if "TB" in spec_text else number) if number else None


def _has_dedicated_gpu(title, specs):
    """Any RTX / GTX / Radeon RX card, or the spec table says "Dedicated"."""
    if re.search(r"\bdedicated\b", str(_spec(specs, "Graphics Description") or ""), re.I):
        return True
    text = title + " " + str(_spec(specs, "Graphics Co Processor") or "")
    return bool(re.search(r"\b(?:rtx|gtx)\s*-?\s*\d{3,4}|\bradeon\s+rx\s*\d{4}|\brx\s*\d{4}m?\b", text, re.I))


def _uses(requirements):
    return requirements.get("use_cases") or detect_use_cases(requirements.get("search_query", ""))


def _is_laptop(product, requirements):
    return requirements.get("category") == "laptop" or product.get("product_category") == "Laptops"


def _is_apple(title):
    return bool(re.search(r"\b(?:macbook|apple)\b", title, re.I))


# ---------------------------------------------------------------------------
# Reviews: count what real reviewers praise and complain about
# ---------------------------------------------------------------------------

def _sentences(text):
    # "Pros: performance. Cons: battery" -> separate parts, so "cons" applies to battery.
    # "good performance but battery drains" -> two parts judged separately.
    parts = re.split(r"(?<=[.!?])\s+|(?=\bcons?\s*:)|(?=\bpros?\s*:)|\bbut\b|\bhowever\b", text.lower())
    return [part.strip() for part in parts if part and part.strip()]


def review_themes(reviews):
    """Count reviews that praise or criticise each aspect.

    Returns (praised, criticised, total), where praised/criticised are lists
    of (aspect, number_of_reviews) sorted by count.
    """
    praised, criticised = {}, {}
    for review in reviews:
        text, stars = review.get("text", ""), review.get("stars")
        found_good, found_bad = set(), set()
        for sentence in _sentences(text):
            for aspect, words in ASPECTS.items():
                if not any(re.search(rf"\b{re.escape(word)}", sentence) for word in words):
                    continue
                is_pros_list = bool(re.match(r"pros?\s*:", sentence))
                is_cons_list = bool(re.match(r"cons?\s*:", sentence))
                negative = is_cons_list or any(re.search(rf"\b{cue}", sentence) for cue in NEGATIVE_CUES)
                positive = is_pros_list or any(re.search(rf"\b{cue}", sentence) for cue in POSITIVE_CUES)
                if negative and not is_pros_list:
                    found_bad.add(aspect)
                elif positive:
                    found_good.add(aspect)
                elif stars is not None:
                    (found_good if stars >= 4 else found_bad if stars <= 2 else set()).add(aspect)
        for aspect in found_good - found_bad:
            praised[aspect] = praised.get(aspect, 0) + 1
        for aspect in found_bad:
            criticised[aspect] = criticised.get(aspect, 0) + 1
    order = lambda counts: sorted(counts.items(), key=lambda item: -item[1])
    return order(praised), order(criticised), len(reviews)


def _theme_sentence(label, themes, total):
    parts = [f"{aspect} ({count} of {total})" for aspect, count in themes[:3]]
    return f"{label}: " + ", ".join(parts) + "." if parts else None


def review_snippets(reviews, limit=5):
    """Real review text for the "Customer comments" box, with the stars given."""
    snippets = []
    for review in reviews[:limit]:
        text = review["text"] if len(review["text"]) <= 280 else review["text"][:277].rsplit(" ", 1)[0] + "..."
        stars = review.get("stars")
        snippets.append(f"{stars:g}/5 stars: {text}" if stars else text)
    return snippets


# ---------------------------------------------------------------------------
# "Why it fits you"
# ---------------------------------------------------------------------------

def _gpu_reason(gpu, vram, level, uses):
    memory = f" with {vram:g} GB of graphics memory" if vram else ""
    if level >= 0.75:
        strength = f"{gpu}{memory}: a strong dedicated graphics card."
    else:
        strength = f"{gpu}{memory}: an entry-level dedicated graphics card."
    benefits = []
    if "gaming" in uses:
        benefits.append("runs most current games smoothly at 1080p" if level >= 0.75 else "handles esports and older games well at 1080p")
    if "3d modelling" in uses:
        benefits.append("speeds up Blender viewport work and GPU rendering (CUDA/OptiX)" if "rtx" in gpu.lower() else "speeds up 3D viewport work")
    if "video editing" in uses:
        benefits.append("speeds up video export and effects with NVIDIA's NVENC encoder in Premiere Pro and DaVinci Resolve" if "rtx" in gpu.lower() else "helps with video effects and export")
    return strength + (" It " + "; it ".join(benefits) + "." if benefits else "")


def _battery_reasons(title, specs, uses):
    """Battery and portability points for students and long-battery requests."""
    lines = []
    if "long battery" in uses:
        hours = battery_hours(title, specs)
        if _is_apple(title):
            lines.append("Apple's own chips are very power-efficient, so MacBooks typically last a full day of classes, notes and browsing on one charge.")
        elif hours and hours >= 8:
            lines.append(f"Amazon lists {hours:g} hours of battery life (the maker's light-use figure), enough for a day of classes without charging.")
        if not _is_apple(title) and not _has_dedicated_gpu(title, specs) and battery_level(title) >= 0.7 and not (hours and hours < 8):
            cpu = _cpu_name(title, specs) or "Its processor"
            lines.append(f"{cpu}: a power-efficient chip with no power-hungry gaming graphics, which helps the battery last longer.")
    weight = _weight_kg(specs)
    if weight and weight <= 1.6 and ({"student", "long battery"} & set(uses)):
        lines.append(f"Weighs {weight:g} kg, light enough to carry to classes every day.")
    return lines


def why_it_fits(product, requirements, page=None):
    page = page or {}
    specs, reviews = page.get("specs") or {}, page.get("reviews") or []
    title = product.get("product_title", "")
    uses = _uses(requirements)
    needs = combined_needs(uses)
    lines = []

    for model in requirements.get("required_models") or []:
        lines.append(f"Has the {model.upper()} you asked for.")

    if _is_laptop(product, requirements):
        gpu, level = _gpu_name(title, specs), gpu_level(title + " " + str(_spec(specs, "Graphics Co Processor") or ""))
        if _is_apple(title) and level >= 0.55 and uses:
            chip = _cpu_name(title, specs) or "Apple M-series chip"
            lines.append(f"{chip}: CPU, graphics and media engines on one efficient chip, so video editing and exports stay fast on battery as well as plugged in.")
        elif gpu and _has_dedicated_gpu(title, specs):
            reason = _gpu_reason(gpu, _vram_gb(title, specs), level, uses)
            if reason:
                lines.append(reason)

        cpu = _cpu_name(title, specs)
        if cpu and needs.get("cpu", 0) >= 0.6 and cpu_level(title + " " + cpu) >= 0.8 and not _is_apple(title):
            lines.append(f"{cpu}: a high-performance processor with many cores, good for rendering, exporting and multitasking.")

        ram, ram_max = _ram_gb(product, specs), _number(_spec(specs, "Ram Memory Maximum Size", "Maximum Memory"))
        if ram and needs.get("ram_gb") and ram >= needs["ram_gb"]:
            use_text = " and ".join(USE_NAMES[u] for u in uses if u in USE_NAMES and u != "long battery")
            upgrade = f", and it can be upgraded to {ram_max:g} GB" if ram_max and ram_max > ram else ""
            lines.append(f"{ram:g} GB RAM meets the {needs['ram_gb']} GB suggested for {use_text}{upgrade}.")

        lines.extend(_battery_reasons(title, specs, uses))

        storage = _storage_gb(product, specs)
        if storage and storage >= 1000:
            lines.append(f"{storage / 1024:g} TB SSD: room for large projects, footage and games.")

        refresh = _number(_spec(specs, "Refresh Rate")) or _number((product.get("specs") or {}).get("Refresh rate"))
        if refresh and refresh >= 120 and "gaming" in uses:
            lines.append(f"{refresh:g} Hz screen: smoother motion in fast games.")

    price, budget = product.get("product_price"), requirements.get("budget_max")
    if price and budget:
        lines.append(f"₹{price:,.0f} is within your ₹{float(budget):,.0f} budget (₹{float(budget) - price:,.0f} to spare).")

    rating, count = product.get("product_star_rating"), product.get("product_num_ratings")
    if rating and count and rating >= 4.0 and count >= 20:
        lines.append(f"Rated {rating:.1f}/5 by {count:,} buyers on Amazon.")

    praised, _, total = review_themes(reviews)
    sentence = _theme_sentence("Reviewers praise", praised, total)
    if sentence:
        lines.append(sentence)
    return [html.escape(line) for line in lines]


# ---------------------------------------------------------------------------
# "Things to know"
# ---------------------------------------------------------------------------

def _laptop_cautions(product, requirements, specs):
    title = product.get("product_title", "")
    uses = _uses(requirements)
    needs = combined_needs(uses)
    level = gpu_level(title + " " + str(_spec(specs, "Graphics Co Processor") or ""))
    apple = _is_apple(title)
    lines = []

    dedicated = _has_dedicated_gpu(title, specs) and not apple
    if dedicated:
        hours, watt_hours = _spec(specs, "Battery Life"), _number(_spec(specs, "Battery Energy", "Watt Hours"))
        listed = ""
        if hours or watt_hours:
            parts = [str(hours).lower()] if hours else []
            if watt_hours:
                parts.append(f"{watt_hours:g} Wh battery")
            listed = f"Amazon lists {', '.join(parts)}, which is for light use. "
        lines.append(
            listed + "Under gaming or rendering, laptops with a dedicated graphics card usually last only about "
            "1–2 hours and run slower on battery, so expect to keep it plugged in for heavy work."
        )
        if cpu_level(title) >= 0.8:
            lines.append("It gets warm and the fans get loud under heavy load; use it on a hard, flat surface (a cooling pad helps).")

    if "long battery" in uses:
        hours = battery_hours(title, specs)
        if dedicated:
            lines.append("Not a good pick if you want to avoid charging often: gaming graphics cards use a lot of power.")
        elif cpu_level(title) >= 0.6 and battery_level(title) <= 0.4:
            lines.append("It uses a high-performance H-series processor, which favours speed over battery life.")
        if hours and hours < 7:
            lines.append(f"Amazon lists only {hours:g} hours of battery (light use), so expect to charge it during a long day.")
        elif hours:
            lines.append("Battery hours are the maker's light-use estimate; bright screens, video calls and heavy apps shorten them.")
        elif not _is_apple(title):
            lines.append("Battery life is not listed; check the product page or reviews for real-world battery figures before buying.")

    weight = _weight_kg(specs)
    if weight and weight >= 2.0:
        lines.append(f"Weighs {weight:g} kg, which is heavy to carry every day.")

    vram = _vram_gb(title, specs)
    heavy = [text for use, text in (("gaming", "new games at high texture settings"), ("3d modelling", "large 3D scenes"))
             if use in uses]
    if vram and vram <= 4 and heavy:
        lines.append(f"Only {vram:g} GB of graphics memory: {' and '.join(heavy)} can run out of it.")

    if needs.get("gpu", 0) >= 0.5 and not dedicated and not apple:
        use_text = " and ".join(USE_NAMES[u] for u in uses if combined_needs([u]).get("gpu", 0) >= 0.5)
        lines.append(f"No dedicated graphics card, so it is not a good choice for {use_text}.")

    ram = _ram_gb(product, specs)
    ram_max = _number(_spec(specs, "Ram Memory Maximum Size", "Maximum Memory"))
    upgrade = f" It can be upgraded to {ram_max:g} GB." if ram_max and ram and ram_max > ram else ""
    if ram and needs.get("ram_gb") and ram < needs["ram_gb"]:
        use_text = " and ".join(USE_NAMES[u] for u in uses if u in USE_NAMES and u != "long battery")
        lines.append(f"{ram:g} GB RAM is below the {needs['ram_gb']} GB suggested for {use_text}.{upgrade}")
    elif ram and ram == 16 and "3d modelling" in uses:
        lines.append(f"16 GB RAM is the starting point for 3D work; complex scenes run better with 32 GB.{upgrade}")

    storage = _storage_gb(product, specs)
    fills = [text for use, text in (("video editing", "4K footage"), ("gaming", "large games")) if use in uses]
    if storage and storage <= 512 and fills:
        lines.append(f"{storage:g} GB storage fills up quickly with {' and '.join(fills)}; plan for an external drive.")

    if apple and "gaming" in uses:
        lines.append("Many popular Windows games, especially ones with anti-cheat, do not run on macOS.")
    if apple and re.search(r"\bair\b", title, re.I) and ({"3d modelling", "video editing"} & set(uses)):
        lines.append("The MacBook Air has no fan, so long renders or exports slow down as it heats up.")

    refresh = _number(_spec(specs, "Refresh Rate"))
    if refresh and refresh <= 60 and "gaming" in uses:
        lines.append("60 Hz screen: fast games look less smooth than on a 120–144 Hz display.")
    brightness = _number(_spec(specs, "Display Brightness"))
    if brightness and brightness <= 300:
        lines.append(f"{brightness:g}-nit screen: fine indoors but hard to read in bright sunlight.")
    return lines


def things_to_know(product, requirements, page=None, reviews_checked=True):
    page = page or {}
    specs, reviews = page.get("specs") or {}, page.get("reviews") or []
    lines = []
    if _is_laptop(product, requirements):
        lines.extend(_laptop_cautions(product, requirements, specs))

    rating, count = product.get("product_star_rating"), product.get("product_num_ratings")
    if not rating:
        lines.append("No Amazon rating is shown yet, so there is little buyer feedback to go on.")
    elif count is not None and count < 20:
        lines.append(f"Only {count} rating{'s' if count != 1 else ''} so far, so the {rating:.1f}/5 average can still change a lot.")
    elif rating < 3.8:
        lines.append(f"The {rating:.1f}/5 average is lower than most well-reviewed products.")

    _, criticised, total = review_themes(reviews)
    sentence = _theme_sentence("Reviewers complain about", criticised, total)
    if sentence:
        lines.append(sentence)
    if not reviews and reviews_checked:
        lines.append("Customer reviews could not be loaded here; read the latest ones on the store page before buying.")
    elif not reviews:
        lines.append("Reviews are loaded for the top 3 results only (to avoid overloading Amazon); open the store page to read them.")
    return [html.escape(line) for line in lines]
