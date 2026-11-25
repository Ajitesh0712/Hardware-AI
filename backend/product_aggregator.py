from backend.scrapers.amazon_scraper import (
    search_amazon,
    get_product_details,
    get_product_reviews,
    get_product_page_info,
)

def search_products(
    query,
    category=None,
    budget_min=None,
    budget_max=None,
    page=1,
    exact_models=True,
):
    return search_amazon(
        query=query,
        page=page,
        budget_min=budget_min,
        budget_max=budget_max,
        exact_models=exact_models,
    )

def get_reviews(asin=None, product_url=None):
    return get_product_reviews(asin=asin, product_url=product_url)


def get_product_page(asin=None):
    """Real reviews and spec table from the product page (one request)."""
    return get_product_page_info(asin)

