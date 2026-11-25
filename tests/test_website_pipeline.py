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
    make_product("PS5A", "Sony PlayStation®5 Console (Slim) Digital Edition", 44990),
    make_product("PS5C", "Sony DualSense Wireless Controller for PlayStation 5 - White", 5990),
    make_product("XBX1", "Xbox Series X 1TB Console", 54990),
    make_product("XBG1", "Forza Horizon 5 Standard Edition (Xbox Series X|S Game)", 2999),
    make_product("FOLD1", "Samsung Galaxy Z Fold7 5G (Blue Shadow, 12GB RAM, 256GB Storage)", 174999),
    make_product("RTX5090L", "ASUS ROG Strix SCAR 18, Intel Core Ultra 9 275HX, RTX 5090 24GB, 64GB RAM, 2TB SSD, Gaming Laptop", 449990),
    make_product("LAPBAT1", "ASUS Vivobook 14, Intel Core i5-1335U, 16GB RAM, 512GB SSD, Up to 10 Hours Battery, Thin and Light Laptop", 72990),
    make_product("BAG1", "Laptop Backpack for 15.6 inch Laptops, Water Resistant", 999),
    make_product("STAND1", "Adjustable Laptop Stand for Desk, Aluminium", 799),
]


class FakeStore:
    """Stands in for Amazon search. It applies only the price bounds, as the
    real scraper does, and records the query text it was asked for."""

    def __init__(self, products=None):
        self.products = CATALOGUE if products is None else products
        self.queries = []

    def search(self, query, category=None, budget_min=None, budget_max=None, page=1, exact_models=True):
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
            mock.patch.object(app_module, "get_product_page", return_value={"reviews": [], "specs": {}}),
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

    # ---- any electronics, not only fixed groups ------------------------
    def test_product_outside_fixed_groups_is_searched_as_electronics(self):
        status, data = self.recommend("samsung fold")
        self.assertEqual(status, 200)
        self.assertEqual(data["requirements"]["category"], "electronics")
        self.assertEqual(self.asins(data), ["FOLD1"])

    def test_console_ranks_above_its_accessories(self):
        status, data = self.recommend("i need sony playstation 5")
        self.assertEqual(status, 200)
        self.assertEqual(self.asins(data)[0], "PS5A")
        self.assertNotIn("LAP4060A", self.asins(data))
        self.assertEqual(self.store.queries[-1], "sony playstation 5")  # filler words removed

    def test_new_product_after_model_search_starts_new_search(self):
        _, first = self.recommend("rtx 5090")
        status, data = self.refine("ps5", first["requirements"])
        self.assertEqual(status, 200)
        self.assertEqual(data["context_mode"], "new_search")
        self.assertEqual(data["requirements"]["required_models"], [])
        self.assertEqual(self.asins(data)[0], "PS5A")
        self.assertNotIn("RTX5090L", self.asins(data))

    def test_accessory_follow_up_searches_for_previous_product(self):
        _, first = self.recommend("i need sony playstation 5")
        status, data = self.refine("controller", first["requirements"])
        self.assertEqual(status, 200)
        self.assertEqual(data["context_mode"], "accessory")
        self.assertEqual(self.store.queries[-1], "sony ps5 controller")
        self.assertEqual(self.asins(data)[0], "PS5C")

    def test_spec_word_follow_up_still_refines(self):
        _, first = self.recommend("laptop under 50000")
        status, data = self.refine("with i5", first["requirements"])
        self.assertEqual(data["context_mode"], "refinement")
        self.assertEqual(data["requirements"]["budget_max"], 50000)

    # ---- match score out of 10 ------------------------------------------
    def test_match_label_is_score_out_of_10(self):
        _, first = self.recommend("i need sony playstation 5")
        _, data = self.refine("playstation 5 controller", first["requirements"])
        labels = {p["asin"]: p["match_label"] for p in data["products"]}
        self.assertEqual(labels["PS5C"], "10/10 match")
        self.assertLessEqual(int(labels["PS5A"].split("/")[0]), 4)
        for product in data["products"]:
            self.assertTrue(1 <= product["fit_score"] <= 10)
            self.assertEqual(product["match_score"], product["fit_score"] * 10)

    def test_missing_wish_lowers_score(self):
        _, data = self.recommend("gaming laptop under 50000")
        scores = [p["fit_score"] for p in data["products"]]
        self.assertTrue(scores)
        self.assertTrue(all(score < 10 for score in scores))  # neither cheap laptop says "gaming"

    # ---- uses instead of specs ("for video editing and 3d") -------------
    def test_use_case_request_searches_and_ranks_by_hardware(self):
        status, data = self.recommend("i need a laptop which is suitable for smooth video editing and 3d modelling work")
        self.assertEqual(status, 200)
        self.assertEqual(data["requirements"]["use_cases"], ["3d modelling", "video editing"])
        self.assertEqual(self.store.queries[-1], "laptop rtx 4060 16gb ram")
        top_three = self.asins(data)[:3]
        self.assertTrue(set(top_three) <= {"LAP4060A", "LAP4060B", "RTX5090L", "MACM5PRO"})
        self.assertNotIn("LAPBUD2", self.asins(data))  # Ryzen 5 U, 8GB, no GPU
        self.assertIn("video editing", data["summary"])

    def test_laptop_without_gpu_is_capped_for_video_editing(self):
        _, first = self.recommend("laptop under 100000")
        status, data = self.refine("for video editing", first["requirements"])
        self.assertEqual(data["context_mode"], "refinement")
        self.assertEqual(data["requirements"]["budget_max"], 100000)
        scores = {p["asin"]: p["fit_score"] for p in data["products"]}
        self.assertLessEqual(scores["LAPBUD1"], 5)       # integrated graphics
        self.assertEqual(scores["LAP4060B"], 10)         # RTX 4060 + Ryzen 7 HS + 16GB
        self.assertEqual(self.asins(data)[-1], "LAPBUD1")

    def test_hardware_levels_from_titles(self):
        from backend.use_cases import cpu_level, gpu_level
        self.assertGreater(gpu_level("RTX 4060 8GB"), gpu_level("RTX 3050 4GB"))
        self.assertGreater(gpu_level("RTX 3050 4GB"), gpu_level("RTX 2050"))
        self.assertGreater(gpu_level("MacBook Pro with M5 Pro chip"), gpu_level("MacBook Air with M5 chip"))
        self.assertGreater(cpu_level("Intel Core i7-13650HX"), cpu_level("Intel Core i5-1235U"))
        self.assertLess(cpu_level("Intel Celeron N4500"), 0.2)

    # ---- studying + long battery ------------------------------------------
    def test_study_laptop_with_good_battery(self):
        query = "i need a laptop for studying and a good battery life so that i dont have to keep it charging everytime"
        status, data = self.recommend(query)
        self.assertEqual(status, 200)
        self.assertEqual(data["requirements"]["use_cases"], ["student", "long battery"])
        self.assertEqual(self.store.queries[0], "laptop long battery life")   # not the whole sentence
        scores = {p["asin"]: p["fit_score"] for p in data["products"]}
        self.assertIn(self.asins(data)[0], {"LAPBAT1", "MACM5A", "MACM4A"})
        for gaming_laptop in ("LAP4060A", "LAP4060B", "LAP4050A", "RTX5090L"):
            if gaming_laptop in scores:
                self.assertLessEqual(scores[gaming_laptop], 5)

    def test_empty_search_is_retried_once_with_a_shorter_query(self):
        calls = []

        def picky_store(query, **settings):
            calls.append(query)
            return [] if len(calls) == 1 else self.store.search(query, **settings)

        with mock.patch.object(app_module, "search_products", side_effect=picky_store):
            status, data = self.recommend("samsung galaxy fold something unusual words here")
        self.assertEqual(status, 200)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], "samsung galaxy fold unusual")

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

    def setUp(self):
        from backend.scrapers import amazon_scraper
        # The cache and the pause after a block are shared state; start clean.
        amazon_scraper._search_cache.clear()
        amazon_scraper._paused_until = 0.0
        amazon_scraper._reviews_paused_until = 0.0

    tearDown = setUp

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
        amazon_scraper._paused_until = 0.0
        FakeResponse.text = "<html><script>function triggerInterstitialChallenge() {}</script></html>"
        with mock.patch.object(amazon_scraper.requests, "get", return_value=FakeResponse()):
            with self.assertRaises(amazon_scraper.AmazonBlockedError):
                amazon_scraper.search_amazon("laptop")

    def test_after_a_block_amazon_is_not_asked_again_for_a_while(self):
        from backend.scrapers import amazon_scraper

        class BlockedResponse:
            text = "<html>captcha</html>"
            url = "https://www.amazon.in/s"

            def raise_for_status(self):
                return None

        with mock.patch.object(amazon_scraper.requests, "get", return_value=BlockedResponse()) as fake_get:
            with self.assertRaises(amazon_scraper.AmazonBlockedError):
                amazon_scraper.search_amazon("laptop")
            with self.assertRaises(amazon_scraper.AmazonBlockedError):
                amazon_scraper.search_amazon("phone")
            with self.assertRaises(amazon_scraper.AmazonBlockedError):
                amazon_scraper.search_amazon("tv")
        self.assertEqual(fake_get.call_count, 1)  # only the first search reached Amazon
        self.assertGreaterEqual(amazon_scraper.minutes_until_retry(), 1)
        self.assertEqual(amazon_scraper.get_product_reviews(asin="B0TEST0001"), [])  # reviews paused too

    def test_repeated_search_is_answered_from_cache(self):
        from backend.scrapers import amazon_scraper
        product = {"asin": "A1", "product_title": "HP Laptop", "product_price": 40000.0}
        with mock.patch.object(amazon_scraper, "_search_amazon_live", return_value=[product]) as live:
            first = amazon_scraper.search_amazon("laptop")
            second = amazon_scraper.search_amazon("Laptop ")
        self.assertEqual(first, second)
        self.assertEqual(live.call_count, 1)


class PoliteClientAppTests(unittest.TestCase):
    def test_block_gives_clear_wait_message(self):
        from backend.scrapers import amazon_scraper
        client = app_module.app.test_client()
        with mock.patch.object(app_module, "search_products", side_effect=amazon_scraper.AmazonBlockedError("blocked")), \
             mock.patch.object(app_module, "minutes_until_retry", return_value=7):
            response = client.post("/api/recommend", json={"query": "laptop"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("wait about 7 minute", response.get_json()["error"])

    def test_reviews_are_fetched_only_for_top_results(self):
        products = [{"asin": f"A{i}", "product_url": "", "product_title": "HP Laptop"} for i in range(6)]
        page = {"reviews": [{"stars": 5.0, "text": "Good"}], "specs": {}}
        with mock.patch.object(app_module, "get_product_page", return_value=page) as fake_page:
            app_module._enrich_products_with_reviews(products, {"category": "laptop"})
        self.assertEqual(fake_page.call_count, app_module.REVIEWS_FOR_TOP)
        self.assertEqual(products[0]["sentiment"], "available_for_review")
        self.assertEqual(products[0]["reviews"], ["5/5 stars: Good"])
        self.assertEqual(products[5]["sentiment"], "unavailable")
        self.assertTrue(any("top 3 results only" in line for line in products[5]["cons"]))


# Spec values copied from a live Amazon India product page (Lenovo LOQ, Nov 2026).
# Review texts are written in the style of real ones and used only in tests.
LOQ_PAGE = {
    "specs": {
        "Graphics Co Processor": "NVIDIA GeForce RTX 4050", "Graphics RAM Size": "6 GB",
        "CPU Model Number": "Core i7-13650HX", "RAM Size": "16 GB", "Ram Memory Maximum Size": "32 GB",
        "Hard Drive Size": "512 GB", "Item Weight": "2.38 Kilograms", "Battery Life": "5 Hours",
        "Lithium Battery Energy Content": "60 Watt Hours", "Display Refresh Rate in Hertz": "144",
        "Maximum Display Brightness": "300 Nit",
    },
    "reviews": [
        {"stars": 5.0, "text": "Best laptop to buy. All rounder laptop. Pros: performance, display, rgb keyboard. Cons: battery"},
        {"stars": 1.0, "text": "After opening neither charger nor laptop switched on. Damaged product delivered."},
        {"stars": 4.0, "text": "Smooth gaming performance but it heats up a lot while gaming."},
    ],
}
LOQ = {
    "asin": "B0HF7V9646", "product_price": 94990.0, "product_star_rating": 3.0, "product_num_ratings": 4,
    "product_category": "Laptops",
    "product_title": "Lenovo LOQ Intel Core i7-13650HX | NVIDIA RTX 4050 6GB (16GB RAM/512GB SSD/144Hz) Gaming Laptop",
}


class ProductInsightTests(unittest.TestCase):
    def setUp(self):
        from backend import product_insights
        self.insights = product_insights
        self.requirements = {
            "category": "laptop", "budget_max": 100000, "search_query": "gaming laptop under 100000",
            "use_cases": ["gaming"],
        }

    def test_why_it_fits_explains_hardware_for_the_use(self):
        lines = " ".join(self.insights.why_it_fits(LOQ, self.requirements, LOQ_PAGE))
        self.assertIn("RTX 4050 with 6 GB of graphics memory", lines)
        self.assertIn("games", lines)
        self.assertIn("Core i7-13650HX", lines)
        self.assertIn("upgraded to 32 GB", lines)
        self.assertIn("144 Hz screen", lines)
        self.assertIn("₹5,010 to spare", lines)
        self.assertIn("Reviewers praise: performance (2 of 3)", lines)

    def test_things_to_know_covers_battery_weight_and_complaints(self):
        lines = " ".join(self.insights.things_to_know(LOQ, self.requirements, LOQ_PAGE))
        self.assertIn("Amazon lists 5 hours, 60 Wh battery", lines)
        self.assertIn("keep it plugged in", lines)
        self.assertIn("fans get loud", lines)
        self.assertIn("2.38 kg", lines)
        self.assertIn("512 GB storage", lines)
        self.assertIn("Only 4 ratings", lines)
        self.assertIn("battery life (1 of 3)", lines)
        self.assertIn("heat levels (1 of 3)", lines)
        self.assertIn("delivery or condition (1 of 3)", lines)

    def test_needs_change_the_advice(self):
        three_d = {**self.requirements, "use_cases": ["3d modelling"], "search_query": "laptop for 3d modelling"}
        lines = " ".join(self.insights.things_to_know(LOQ, three_d, LOQ_PAGE))
        self.assertIn("complex scenes run better with 32 GB", lines)
        self.assertNotIn("144 Hz", " ".join(self.insights.why_it_fits(LOQ, three_d, LOQ_PAGE)))

    def test_no_dedicated_gpu_is_flagged_for_video_editing(self):
        basic = {"product_title": "HP 15s Ryzen 5 5500U 8GB RAM 512GB SSD Laptop", "product_price": 38990.0,
                 "product_star_rating": None, "product_num_ratings": None, "product_category": "Laptops"}
        needs = {"category": "laptop", "search_query": "laptop for video editing", "use_cases": ["video editing"]}
        lines = " ".join(self.insights.things_to_know(basic, needs, {}))
        self.assertIn("No dedicated graphics card", lines)
        self.assertIn("8 GB RAM is below the 16 GB", lines)
        self.assertIn("No Amazon rating", lines)
        self.assertNotIn("plugged in", lines)

    def test_battery_advice_for_students(self):
        needs = {"category": "laptop", "search_query": "laptop for studying with good battery",
                 "use_cases": ["student", "long battery"]}
        light = {"product_title": "ASUS Vivobook 14, Intel Core i5-1335U, 16GB RAM, 512GB SSD Laptop",
                 "product_price": 55990.0, "product_star_rating": 4.2, "product_num_ratings": 300, "product_category": "Laptops"}
        page = {"specs": {"Battery Life": "10 Hours", "Item Weight": "1.4 Kilograms", "CPU Model Number": "Core i5-1335U"}, "reviews": []}
        fits = " ".join(self.insights.why_it_fits(light, needs, page))
        self.assertIn("10 hours of battery life", fits)
        self.assertIn("power-efficient", fits)
        self.assertIn("1.4 kg", fits)
        know = " ".join(self.insights.things_to_know(LOQ, needs, LOQ_PAGE))
        self.assertIn("Not a good pick if you want to avoid charging often", know)
        self.assertIn("only 5 hours", know)

    def test_text_from_amazon_is_html_escaped(self):
        page = {"specs": {"Graphics Co Processor": "RTX 4060 <script>x</script>"}, "reviews": []}
        lines = " ".join(self.insights.why_it_fits(LOQ, self.requirements, page))
        self.assertNotIn("<script>", lines)

    def test_review_parser_reads_stars_and_removes_boilerplate(self):
        from bs4 import BeautifulSoup
        from backend.scrapers import amazon_scraper
        html = """<div data-hook="review"><i data-hook="review-star-rating"><span>4.0 out of 5 stars</span></i>
        <span data-hook="reviewText">Brief content visible, double tap to read full content. Great screen.</span></div>"""
        reviews = amazon_scraper._read_review_blocks(BeautifulSoup(html, "lxml"))
        self.assertEqual(reviews, [{"stars": 4.0, "text": "Great screen."}])

if __name__ == "__main__":
    unittest.main()
