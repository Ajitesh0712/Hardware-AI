from backend.scrapers.amazon_scraper import (
    search_amazon,
    get_product_details
)

def search_products(
    query,
    category=None,
    budget_max=None,
    page=1
):
    return search_amazon(
        query=query,
        page=page
    )

def get_reviews(asin=None, product_url=None):

    if product_url:

        details = get_product_details(
            product_url
        )

        return details.get(
            "reviews",
            []
        )

    return []