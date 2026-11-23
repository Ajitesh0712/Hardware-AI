import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9"
}


class AmazonBlockedError(RuntimeError):
    """Amazon answered with a robot check / CAPTCHA page instead of results."""


def _looks_blocked(html):
    """Spot Amazon's robot checks. They can come back with HTTP 200."""
    text = html.lower()
    markers = (
        "captcha",
        "enter the characters you see below",
        "api-services-support@amazon.com",
        "triggerinterstitialchallenge",  # small JavaScript challenge page
    )
    if any(marker in text for marker in markers):
        return True
    # A real search page is hundreds of KB. A tiny page with no result
    # blocks is not a genuine "no products" answer.
    return len(html) < 20000 and 's-search-result' not in text


def search_amazon(query, page=1, budget_min=None, budget_max=None):
    """Search Amazon India and enforce any explicit price bounds locally.

    If the first results page cannot be fetched (network error, HTTP error or
    a robot-check page) an exception is raised, so the API can say that the
    store could not be reached instead of saying that no products exist.
    """
    models = _extract_search_models(query)
    pages_to_check = range(int(page), int(page) + (3 if models else 1))
    if budget_min is not None:
        low_price = str(int(float(budget_min)))
    else:
        low_price = None
    if budget_max is not None:
        high_price = str(int(float(budget_max)))
    else:
        high_price = None

    products = []
    seen_asins = set()
    for search_page in pages_to_check:
        params = {"k": query, "page": str(search_page)}
        if low_price is not None:
            params["low-price"] = low_price
        if high_price is not None:
            params["high-price"] = high_price
        try:
            response = requests.get("https://www.amazon.in/s", params=params, headers=HEADERS, timeout=10)
            response.raise_for_status()
            if _looks_blocked(response.text):
                raise AmazonBlockedError("Amazon returned a robot-check page instead of search results")
        except Exception as e:
            print(f"[Amazon Scraper] Error on page {search_page}: {e}")
            if search_page == pages_to_check[0]:
                raise
            break  # keep the products already found on earlier pages

        soup = BeautifulSoup(response.text, "lxml")

        results = soup.select('[data-component-type="s-search-result"]')
        for item in results:
            product = parse_product(item)
            if not product:
                continue
            title = product.get("product_title", "")
            if models and not all(pattern.search(title) for pattern in models):
                continue
            price = product.get("product_price")
            if not price or (budget_min is not None and price < float(budget_min)) or (budget_max is not None and price > float(budget_max)):
                continue
            asin = product.get("asin")
            if asin and asin in seen_asins:
                continue
            if asin:
                seen_asins.add(asin)
            products.append(product)

        # Search deeper only if the exact requested model was too rare on
        # the first results page. Six candidates are enough for ranking.
        if len(products) >= 6 or not models:
            break

    return products


def _extract_search_models(query):
    """Build exact-title matchers for model identifiers in a search."""
    text = str(query or "").lower()
    patterns = (
        r"\b(?:geforce\s+)?(?:rtx|gtx)\s*\d{3,4}\s*(?:ti|super)?\b",
        r"\b(?:radeon\s+)?rx\s*\d{3,4}\s*(?:xt|xtx)?\b",
        r"\b(?:core\s+)?i[3579]-?\d{4,5}[a-z]{0,2}\b",
        r"\bryzen\s+[3579]\s+\d{4,5}[a-z]{0,2}\b",
        r"\b(?:intel\s+)?core\s+ultra\s+[579]\b",
        r"\bm[1-5](?:\s+(?:pro|max|ultra))?\b",
        r"\biphone\s+\d{1,2}(?:\s+(?:pro|max|plus|mini)){0,2}\b",
        r"\bgalaxy\s+[azs]\s?\d{1,2}(?:\s+(?:fe|plus|ultra))?\b",
        r"\bpixel\s+\d{1,2}(?:\s+(?:pro|xl|a))?\b",
        r"\boneplus\s+\d{1,2}(?:r)?\b",
    )
    matchers = []
    for source in patterns:
        for match in re.finditer(source, text, re.I):
            model = re.sub(r"^(?:geforce|radeon)\s+", "", match.group(0).lower())
            escaped = re.escape(model)
            escaped = re.sub(r"\\\s+", r"[\\s-]*", escaped).replace(r"\-", r"[ -]?")
            if re.fullmatch(r"m[1-5]", model):
                escaped += r"(?!\s*(?:pro|max|ultra)\b)"  # base chip only
            if re.search(r"\b(?:rtx|gtx)\s*\d{3,4}$", model):
                escaped += r"(?!\s*(?:ti|super)\b)"
            elif re.search(r"\brx\s*\d{3,4}$", model):
                escaped += r"(?!\s*(?:xt|xtx)\b)"
            matchers.append(re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I))
    return matchers


def _read_title(item):
    """Return the full product title from one search-result card.

    Amazon sometimes shows the brand in its own small <h2> ("Apple") above
    the real title <h2> ("2026 MacBook Air 13 Laptop with M5 chip ...").
    Reading only the first <h2> gave titles like "Apple", so exact-model
    checks (M5, RTX 3060) failed for every product. We take the longest
    heading and put the brand in front of it when it is missing.
    """
    headings = [h.get_text(" ", strip=True) for h in item.select("h2")]
    headings = [text for text in headings if text]
    if not headings:
        return ""
    title = max(headings, key=len)
    brand = headings[0]
    if brand != title and not title.lower().startswith(brand.lower()):
        title = f"{brand} {title}"
    return title


def parse_product(item):

    try:

        link_el = item.select_one(
            "a.a-link-normal.s-no-outline"
        )


        price_el = item.select_one(".a-price .a-offscreen") or item.select_one(".a-price-whole")

        rating_el = item.select_one(".a-icon-alt")

        image_el = item.select_one("img")

        review_el = item.select_one("a[href*='customerReviews']") or item.select_one("span.a-size-base.s-underline-text")

        title = _read_title(item)

        price_str = (
            price_el.get_text(strip=True)
            if price_el else "0"
        )

        # None means "not shown on the search page"; it is not a 0-star rating.
        rating = None
        if rating_el:

            try:
                rating = float(
                    rating_el.get_text(strip=True)
                    .split()[0]
                )

            except:
                pass

        review_count = _parse_review_count(review_el.get_text(" ", strip=True)) if review_el else None
        if not review_count:
            for node in item.select("[aria-label]"):
                label = node.get("aria-label", "")
                if re.search(r"\b(?:ratings?|reviews?)\b", label, re.I):
                    review_count = _parse_review_count(label)
                    if review_count:
                        break
        if not review_count:
            review_count = None  # count not shown, so do not report "0 ratings"

        image_url = (
            image_el.get("src", "")
            if image_el else ""
        )

        product_url = ""

        if link_el:

            href = link_el.get(
                "href",
                ""
            )
            if "/sspa/" in href:
                return None

            if href.startswith("/"):

                product_url = (
                    "https://www.amazon.in"
                    + href
                )

            else:

                product_url = href

        return {

            "asin": item.get(
                "data-asin",
                ""
            ),

            "product_title": title,

            "product_price": parse_price(price_str),

            "product_price_str": (
                price_str if "₹" in price_str else f"₹{price_str}"
            ),

            "product_star_rating": rating,

            "product_num_ratings": review_count,

            "product_url": product_url,

            "product_photo": image_url,

            "is_best_seller": False,

            "is_amazon_choice": False,

            "delivery": "",

            "specs": {}
        }

    except Exception as e:

        print(
            f"[Amazon Scraper] Parse Error: {e}"
        )

        return None



def get_product_details(product_url=None, asin=None):
    try:
        if not product_url and asin:
            product_url = f"https://www.amazon.in/dp/{asin}"
        response=requests.get(product_url,headers=HEADERS,timeout=10)
        response.raise_for_status()
        soup=BeautifulSoup(response.text,"lxml")
        return extract_details(soup)
    except Exception as e:
        print(f"[Details Error] {e}")
        return {}


def get_product_reviews(asin=None, product_url=None):
    """Fetch visible review text from the product's review page, if accessible."""
    if not asin and product_url:
        match = re.search(r"/dp/([A-Z0-9]{10})|/product-reviews/([A-Z0-9]{10})", product_url, re.I)
        if match:
            asin = match.group(1) or match.group(2)
    if not asin:
        return []
    review_url = f"https://www.amazon.in/product-reviews/{asin}/?reviewerType=all_reviews&sortBy=recent&pageNumber=1"
    try:
        response = requests.get(review_url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        reviews = []
        for block in soup.select('[data-hook="review"]'):
            text_el = block.select_one('[data-hook="reviewText"]')
            if text_el:
                text = text_el.get_text(" ", strip=True)
                if text:
                    reviews.append(text)
            if len(reviews) >= 5:
                break
        return reviews
    except Exception as exc:
        print(f"[Review Scraper] Could not retrieve reviews for {asin}: {exc}")
        return []
    
def extract_details(soup):
    details={
        "rating":0.0,
        "review_count":0,
        "reviews":[],
        "specs":{}
    }
    

    #rating

    try:
        rating_el=soup.select_one("#acrPopover")
        if rating_el:

            rating_text = (
                rating_el.get(
                    "title",
                    ""
                )
            )

            details["rating"] = float(
                rating_text.split()[0]
            )
    except:
        pass

    # Review Count

    try:

        review_count_el = soup.select_one(
            "#acrCustomerReviewText"
        )

        if review_count_el:

            review_text = (
                review_count_el
                .get_text(strip=True)
            )

            review_text = (
                review_text
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
            )

            details["review_count"] = int(
                review_text
            )

    except:
        pass

    # Top Reviews

    # Top Reviews

    try:

        reviews = []

        review_blocks = soup.select(
            '[data-hook="review"]'
        )

        for block in review_blocks[:5]:

            text_el = block.select_one(
                '[data-hook="reviewText"]'
            )

            if text_el:

                review_text = text_el.get_text(
                    " ",
                    strip=True
                )

                review_text = review_text.replace(
                    "Brief content visible, double tap to read full content.",
                    ""
                )

                review_text = review_text.replace(
                    "Full content visible, double tap to read brief content.",
                    ""
                )

                review_text = review_text.replace(
                    "Read moreRead less",
                    ""
                )

                review_text = (
                    review_text
                    .replace(
                        "Read more Read less",
                        ""
                    )
                    .strip()
                )

                if review_text:

                    reviews.append(
                        review_text
                    )

        details["reviews"] = reviews

    except Exception as e:

        print(
            "Review Error:",
            e
        )

    # Product Specifications

    try:

        specs = {}

        rows = soup.select(
            "#productDetails_techSpec_section_1 tr"
        )

        if not rows:

            rows = soup.select(
                "#productDetails_detailBullets_sections1 tr"
            )

        for row in rows:

            cols = row.find_all(
                ["th", "td"]
            )

            if len(cols) >= 2:

                key = cols[0].get_text(
                    strip=True
                )

                value = cols[1].get_text(
                    strip=True
                )

                specs[key] = value

        details["specs"] = specs

    except:
        pass

    return details

def parse_price(price_str):
    if price_str is None:
        return 0.0
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(price_str))
    return float(match.group().replace(",", "")) if match else 0.0


def _parse_review_count(value):
    match = re.search(r"\d[\d,]*", str(value or ""))
    return int(match.group().replace(",", "")) if match else 0
