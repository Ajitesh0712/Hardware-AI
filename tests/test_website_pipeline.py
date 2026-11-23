"""Offline checks for the website pipeline (no Amazon, Gemini, or MySQL).

Run from the repository root:

    python -m unittest discover -s tests -v

The real scraper, review fetcher, and database calls are replaced with small
fakes. This lets us check query parsing, chat context, filtering, and ranking
with the same products every time. Passing these tests does NOT prove that a
live Amazon scrape works; that has to be checked separately.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# Turn Gemini off so requirement parsing uses the local rules only.
os.environ["GEMINI_API_KEY"] = ""
os.environ["DB_HOST"] = "127.0.0.1"
os.environ["DB_PORT"] = "1"  # nothing listens here, so init_db fails fast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend.app as app_module  # noqa: E402
from backend.product_features import classify_product  # noqa: E402


def make_product(asin, title, price, rating=4.3, num_ratings=250):
    return {
        "asin": asin,
        "product_title": title,
        "product_price": price,
        "product_star_rating": rating,
        "product_num_ratings": num_ratings,
        "product_url": f"https://www.amazon.in/dp/{asin}",
        "product_photo": "",
        "is_best_seller": False,
        "is_amazon_choice": False,
    }


# A small fake catalogue that looks like Amazon India search results.
CATALOGUE = [
    make_product("LAP4060A", 'Lenovo LOQ Intel Core i5-12450HX, NVIDIA GeForce RTX 4060 8GB, 16GB DDR5 RAM, 512GB SSD, 15.6" FHD 144Hz Gaming Laptop', 89990),
    make_product("LAP4060B", "ASUS TUF Gaming A15, AMD Ryzen 7 7435HS, RTX 4060 8GB, 16GB RAM, 512GB SSD, 15.6-inch Gaming Laptop", 82990),
    make_product("LAP4050A", "HP Victus Intel Core i5-13420H, RTX 4050 6GB, 16GB DDR4 RAM, 512GB SSD Gaming Laptop", 74990),
    make_product("LAP3060A", 'Acer Nitro V Intel Core i5-13420H, NVIDIA GeForce RTX 3060 6GB, 8GB DDR5 RAM, 512GB SSD, 15.6" Gaming Laptop', 64990),
    make_product("GPU4060A", "ZOTAC Gaming GeForce RTX 4060 8GB Twin Edge OC Graphics Card", 29999),
    make_product("GPU3060A", "MSI GeForce RTX 3060 Ventus 2X 12G OC Graphics Card", 26999),
    make_product("MACM5A", 'Apple MacBook Air 13" Laptop with M5 chip, 16GB Unified Memory, 512GB SSD', 114900),
    make_product("MACM5PRO", "Apple 2026 MacBook Pro Laptop with M5 Pro chip, 24GB Unified Memory, 1TB SSD", 269900),
    make_product("MACM4A", 'Apple MacBook Air 13" Laptop with M4 chip, 16GB Unified Memory, 256GB SSD', 99900),
    make_product("LAPBUD1", 'Lenovo IdeaPad Slim 3 Intel Core i5 12th Gen, 16GB RAM, 512GB SSD, 15.6" FHD Laptop', 48990),
    make_product("LAPBUD2", "HP 15s AMD Ryzen 5 5500U, 8GB DDR4 RAM, 512GB SSD, Windows 11, 15.6 inch Laptop", 38990, rating=None, num_ratings=None),
    make_product("BAG1", "Laptop Backpack for 15.6 inch Laptops, Water Resistant", 999),
    make_product("STAND1", "Adjustable Laptop Stand for Desk, Aluminium", 799),
]


class FakeStore:
    """Stands in for Amazon search. It applies only the price bounds, as the
    real scraper does, and records the query text it was asked for."""

    def __init__(self, products=None):
        self.products = CATALOGUE if products is None else products
        self.queries = []

    def search(self, query, category=None, budget_min=None, budget_max=None, page=1):
        self.queries.append(query)
        results = []
        for product in self.products:
            price = product["product_price"]
            if budget_min is not None and price < budget_min:
                continue
            if budget_max is not None and price > budget_max:
                continue
            results.append(dict(product))
        return results


class WebsitePipelineTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.patches = [
            mock.patch.object(app_module, "search_products", side_effect=self.store.search),
            mock.patch.object(app_module, "get_reviews", return_value=[]),
            mock.patch.object(app_module, "save_query", return_value=0),
            mock.patch.object(app_module, "save_recommendations", return_value=None),
            mock.patch.object(app_module, "upsert_product", return_value=None),
        ]
        for patch in self.patches:
            patch.start()
        self.client = app_module.app.test_client()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()

    # ---- small helpers -------------------------------------------------
    def recommend(self, query):
        response = self.client.post("/api/recommend", json={"query": query})
        return response.status_code, response.get_json()

    def refine(self, query, previous_requirements):
        response = self.client.post("/api/refine", json={
            "query": query,
            "original_requirements": previous_requirements,
        })
        return response.status_code, response.get_json()

    @staticmethod
    def asins(data):
        return [product["asin"] for product in data.get("products", [])]

    # ---- first searches ------------------------------------------------
    def test_laptop_with_rtx_4060_returns_only_rtx_4060_laptops(self):
        status, data = self.recommend("laptop with RTX 4060")
        self.assertEqual(status, 200)
        self.assertEqual(sorted(self.asins(data)), ["LAP4060A", "LAP4060B"])
        self.assertEqual(data["requirements"]["category"], "laptop")

    def test_laptop_with_ddr5_in_title_is_still_a_laptop(self):
        title = CATALOGUE[0]["product_title"]
        self.assertEqual(classify_product(title), "Laptops")
        self.assertEqual(classify_product("HP 250 G8 Laptop (11th Gen Intel Core i3 / 8GB DDR4 RAM / 512GB SSD)"), "Laptops")
        # Real components and accessories must keep their own category.
        self.assertEqual(classify_product("Crucial 8GB DDR4 3200MHz Laptop RAM"), "Computer Components")
        self.assertEqual(classify_product("ZOTAC Gaming GeForce RTX 4060 8GB Graphics Card"), "Computer Components")
        self.assertEqual(classify_product("65W Laptop Charger for HP Pavilion"), "Chargers and Cables")
        # Short live titles without RAM/storage are still laptops.
        self.assertEqual(classify_product("Lenovo V15 G4 AMD Ryzen 5 7520U 15.6 inch FHD Laptop"), "Laptops")
        self.assertEqual(classify_product('Apple 2026 MacBook Neo 13" Laptop with A18 Pro chip'), "Laptops")
        self.assertNotEqual(classify_product("Hard shell case cover for MacBook Air M2 chip"), "Laptops")
        # Newer Intel names seen in live Amazon titles.
        self.assertEqual(classify_product("MSI Stealth 14 AI Studio, Intel Ultra 7 155H, Gaming Laptop (16GB/1TB NVMe SSD/Windows 11/RTX 4060)"), "Laptops")
        self.assertEqual(classify_product("Dell 14 Laptop, Intel Core 5 120U, 16GB DDR5 RAM, 512GB SSD, Windows 11"), "Laptops")

    def test_macbook_with_m5_returns_only_m5(self):
        status, data = self.recommend("MacBook with M5 chip")
        self.assertEqual(status, 200)
        self.assertEqual(self.asins(data), ["MACM5A"])

    def test_laptop_under_50000_respects_budget_and_category(self):
        status, data = self.recommend("laptop under 50000")
        self.assertEqual(status, 200)
        self.assertEqual(sorted(self.asins(data)), ["LAPBUD1", "LAPBUD2"])
        for product in data["products"]:
            self.assertLessEqual(product["product_price"], 50000)

    def test_bare_gpu_query_searches_laptops_not_graphics_cards(self):
        status, data = self.recommend("RTX 3060")
        self.assertEqual(status, 200)
        self.assertEqual(self.asins(data), ["LAP3060A"])
        self.assertIn("laptop", self.store.queries[-1].lower())

    def test_missing_rating_is_reported_as_missing(self):
        status, data = self.recommend("laptop under 50000")
        self.assertEqual(status, 200)
        no_rating = next(p for p in data["products"] if p["asin"] == "LAPBUD2")
        self.assertIsNone(no_rating["product_star_rating"])
        self.assertIsNone(no_rating["product_num_ratings"])
        self.assertEqual(no_rating["sentiment"], "unavailable")

    # ---- chat follow-ups -----------------------------------------------
    def test_budget_only_follow_up_refines_current_search(self):
        _, first = self.recommend("laptop with RTX 4060")
        status, data = self.refine("under 85000", first["requirements"])
        self.assertEqual(status, 200)
        self.assertEqual(data["context_mode"], "refinement")
        self.assertEqual(self.asins(data), ["LAP4060B"])
        self.assertEqual(data["requirements"]["required_models"], ["rtx 4060"])
        self.assertEqual(data["requirements"]["budget_max"], 85000)

    def test_new_model_replaces_old_model(self):
        _, first = self.recommend("laptop with RTX 4060")
        status, data = self.refine("MacBook with M5 chip", first["requirements"])
        self.assertEqual(status, 200)
        self.assertEqual(data["context_mode"], "new_search")
        self.assertEqual(data["requirements"]["required_models"], ["m5"])
        self.assertEqual(self.asins(data), ["MACM5A"])

    def test_bare_gpu_after_model_search_starts_new_search(self):
        _, first = self.recommend("laptop with RTX 4060")
        status, data = self.refine("RTX 3060", first["requirements"])
        self.assertEqual(status, 200)
        self.assertEqual(data["context_mode"], "new_search")
        self.assertEqual(self.asins(data), ["LAP3060A"])

    def test_broad_category_after_model_search_clears_model(self):
        _, first = self.recommend("MacBook with M5 chip")
        status, data = self.refine("laptop under 50000", first["requirements"])
        self.assertEqual(status, 200)
        self.assertEqual(data["requirements"].get("required_models"), [])
        self.assertEqual(sorted(self.asins(data)), ["LAPBUD1", "LAPBUD2"])

    def test_new_budget_follow_up_replaces_old_budget(self):
        _, first = self.recommend("laptop under 50000")
        status, data = self.refine("under 70000", first["requirements"])
        self.assertEqual(status, 200)
        self.assertEqual(data["context_mode"], "refinement")
        self.assertEqual(data["requirements"]["budget_max"], 70000)
        self.assertEqual(sorted(self.asins(data)), ["LAP3060A", "LAPBUD1", "LAPBUD2"])

    def test_other_category_with_no_matches_does_not_show_accessories(self):
        _, first = self.recommend("laptop under 50000")
        status, data = self.refine("phone under 20000", first["requirements"])
        self.assertEqual(status, 200)
        self.assertEqual(data["requirements"]["category"], "mobile")
        self.assertEqual(data["products"], [])

    def test_follow_up_with_no_matches_returns_empty_list(self):
        _, first = self.recommend("laptop with RTX 4060")
        status, data = self.refine("under 30000", first["requirements"])
        self.assertEqual(status, 200)
        self.assertEqual(data["products"], [])

    # ---- ranking -------------------------------------------------------
    def test_exact_model_matches_rank_above_other_laptops(self):
        # Without a model filter the GPU laptops are all visible, but a
        # query for RTX 4060 must never show an RTX 4050 laptop.
        status, data = self.recommend("gaming laptop with RTX 4060")
        self.assertEqual(status, 200)
        self.assertNotIn("LAP4050A", self.asins(data))
        self.assertNotIn("GPU4060A", self.asins(data))

    def test_accessories_do_not_appear_in_laptop_results(self):
        status, data = self.recommend("laptop under 50000")
        self.assertEqual(status, 200)
        self.assertNotIn("BAG1", self.asins(data))
        self.assertNotIn("STAND1", self.asins(data))

    # ---- errors --------------------------------------------------------
    def test_no_products_returns_404_with_message(self):
        self.store.products = []
        status, data = self.recommend("laptop with RTX 4060")
        self.assertEqual(status, 404)
        self.assertIn("error", data)

    def test_store_failure_returns_502_not_500(self):
        with mock.patch.object(app_module, "search_products", side_effect=RuntimeError("blocked")):
            status, data = self.recommend("laptop with RTX 4060")
            self.assertEqual(status, 502)
            self.assertIn("error", data)
            status, data = self.refine("under 85000", {"category": "laptop", "search_query": "laptop"})
            self.assertEqual(status, 502)
            self.assertIn("error", data)

    def test_refine_without_previous_requirements_does_not_crash(self):
        response = self.client.post("/api/refine", json={"query": "laptop under 50000", "original_requirements": None})
        self.assertEqual(response.status_code, 200)

    def test_non_json_body_returns_json_error(self):
        response = self.client.post("/api/recommend", data="hello", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_interaction_logging_without_database_does_not_500(self):
        with mock.patch.object(app_module, "log_interaction", side_effect=RuntimeError("no MySQL")):
            response = self.client.post("/api/interaction", json={"session_id": "s", "asin": "A", "type": "click"})
        self.assertEqual(response.status_code, 200)


class ScraperParsingTests(unittest.TestCase):
    """Checks the HTML parser on a tiny hand-written search result."""

    def test_missing_rating_and_review_count_become_none(self):
        from bs4 import BeautifulSoup
        from backend.scrapers import amazon_scraper

        html = """
        <div data-component-type="s-search-result" data-asin="B0TEST0001">
          <h2><span>HP 15s Ryzen 5 Laptop</span></h2>
          <a class="a-link-normal s-no-outline" href="/dp/B0TEST0001">link</a>
          <span class="a-price"><span class="a-offscreen">₹38,990</span></span>
        </div>"""
        item = BeautifulSoup(html, "lxml").select_one('[data-component-type="s-search-result"]')
        product = amazon_scraper.parse_product(item)
        self.assertEqual(product["product_price"], 38990.0)
        self.assertIsNone(product["product_star_rating"])
        self.assertIsNone(product["product_num_ratings"])

    def test_title_is_read_when_brand_has_its_own_heading(self):
        # Live Amazon layout (Nov 2026): a small brand <h2> above the title <h2>.
        from bs4 import BeautifulSoup
        from backend.scrapers import amazon_scraper

        html = """
        <div data-component-type="s-search-result" data-asin="B0TEST0002">
          <h2 class="a-size-mini"><span>Apple</span></h2>
          <a class="a-link-normal s-no-outline" href="/dp/B0TEST0002">link</a>
          <h2 class="a-size-medium"><span>2026 MacBook Air 13" Laptop with M5 chip, 16GB Unified Memory</span></h2>
          <span class="a-price"><span class="a-offscreen">₹1,34,890</span></span>
        </div>"""
        item = BeautifulSoup(html, "lxml").select_one('[data-component-type="s-search-result"]')
        product = amazon_scraper.parse_product(item)
        self.assertEqual(product["product_title"], 'Apple 2026 MacBook Air 13" Laptop with M5 chip, 16GB Unified Memory')
        matchers = amazon_scraper._extract_search_models("macbook with m5")
        self.assertTrue(all(m.search(product["product_title"]) for m in matchers))
        self.assertFalse(all(m.search("Apple MacBook Pro Laptop with M5 Pro chip") for m in matchers))

    def test_blocked_search_page_raises_instead_of_looking_empty(self):
        from backend.scrapers import amazon_scraper

        class FakeResponse:
            text = "<html><body>Enter the characters you see below (captcha)</body></html>"
            status_code = 200

            def raise_for_status(self):
                return None

        with mock.patch.object(amazon_scraper.requests, "get", return_value=FakeResponse()):
            with self.assertRaises(amazon_scraper.AmazonBlockedError):
                amazon_scraper.search_amazon("laptop")

        # Amazon can also send a tiny JavaScript challenge page with HTTP 200.
        FakeResponse.text = "<html><script>function triggerInterstitialChallenge() {}</script></html>"
        with mock.patch.object(amazon_scraper.requests, "get", return_value=FakeResponse()):
            with self.assertRaises(amazon_scraper.AmazonBlockedError):
                amazon_scraper.search_amazon("laptop")


if __name__ == "__main__":
    unittest.main()
