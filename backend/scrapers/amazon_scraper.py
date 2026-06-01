import requests
from bs4 import BeautifulSoup
from urllib.parse import quote


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9"
}


def search_amazon(query, page=1):

    url = (
        f"https://www.amazon.in/s?"
        f"k={quote(query)}"
        f"&page={page}"
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        results = soup.select(
            '[data-component-type="s-search-result"]'
        )

        products = []

        for item in results:

            product = parse_product(item)

            if product:
                products.append(product)

        return products

    except Exception as e:

        print(
            f"[Amazon Scraper] Error: {e}"
        )

        return []


def parse_product(item):

    try:

        title_el = item.select_one("h2 span")

        link_el = item.select_one(
            "a.a-link-normal.s-no-outline"
        )


        price_el = item.select_one(".a-price-whole")

        rating_el = item.select_one(".a-icon-alt")

        image_el = item.select_one("img")

        review_el = item.select_one(
            "span.a-size-base.s-underline-text"
        )

        title = (
            title_el.get_text(strip=True)
            if title_el else ""
        )

        price_str = (
            price_el.get_text(strip=True)
            if price_el else "0"
        )

        rating = 0.0
        if rating_el:

            try:
                rating = float(
                    rating_el.get_text(strip=True)
                    .split()[0]
                )

            except:
                pass

        review_count = 0

        if review_el:

            try:
                review_count = int(
                    review_el.get_text(strip=True)
                    .replace(",", "")
                )

            except:
                pass

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

            "product_price": parse_price(
                price_str
            ),

            "product_price_str": (
                f"₹{price_str}"
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



def get_product_details(product_url):
    try:
        response=requests.get(product_url,headers=HEADERS,timeout=20)
        
        soup=BeautifulSoup(response.text,"lxml")
        return extract_details(soup)
    except Exception as e:
        print(f"[Details Error] {e}")
        return {}
    
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

    try:

        return float(
            str(price_str)
            .replace("₹", "")
            .replace(",", "")
            .strip()
        )

    except:

        return 0.0