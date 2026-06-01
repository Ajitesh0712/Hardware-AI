from scrapers.amazon_scraper import (
    search_amazon,
    get_product_details
)

products = search_amazon("gaming laptop")

print("Products Found:", len(products))

first_product = products[0]

print("\nTITLE:")
print(first_product["product_title"])

print("\nURL:")
print(first_product["product_url"])

details = get_product_details(
    first_product["product_url"]
)

print("\nDETAILS:")
print(details)