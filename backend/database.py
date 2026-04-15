"""
database.py
MySQL database layer for storing products, queries, and interactions.
"""

import os
import json
import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Connection pool for better performance
_pool = None


def get_pool():
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="hardwareai_pool",
            pool_size=5,
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            database=os.getenv("DB_NAME", "hardwareai"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            autocommit=True,
            charset="utf8mb4"
        )
    return _pool


def get_connection():
    return get_pool().get_connection()


# ─── Schema Initialization ────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_queries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                query_text TEXT NOT NULL,
                parsed_requirements JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(64),
                result_count INT DEFAULT 0
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                asin VARCHAR(20) UNIQUE NOT NULL,
                product_title TEXT NOT NULL,
                product_price DECIMAL(10,2),
                product_star_rating DECIMAL(3,2),
                product_num_ratings INT,
                product_url TEXT,
                product_photo TEXT,
                specs JSON,
                pros JSON,
                cons JSON,
                sentiment VARCHAR(20),
                is_best_seller BOOLEAN DEFAULT FALSE,
                is_amazon_choice BOOLEAN DEFAULT FALSE,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                category VARCHAR(100)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                query_id INT NOT NULL,
                asin VARCHAR(20) NOT NULL,
                match_score DECIMAL(5,2),
                rank_position INT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES user_queries(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64),
                asin VARCHAR(20),
                interaction_type VARCHAR(30),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        print("[DB] Tables initialized successfully.")
    except Exception as e:
        print(f"[DB] Init error: {e}")
    finally:
        cursor.close()
        conn.close()


# ─── Query Operations ─────────────────────────────────────────────────────────

def save_query(query_text: str, parsed_requirements: dict, session_id: str = None, result_count: int = 0) -> int:
    """Save a user query and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO user_queries (query_text, parsed_requirements, session_id, result_count)
               VALUES (%s, %s, %s, %s)""",
            (query_text, json.dumps(parsed_requirements), session_id, result_count)
        )
        return cursor.lastrowid
    except Exception as e:
        print(f"[DB] Save query error: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()


def get_recent_queries(limit: int = 10) -> list:
    """Retrieve the most recent user queries."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, query_text, created_at, result_count FROM user_queries ORDER BY created_at DESC LIMIT %s",
            (limit,)
        )
        return cursor.fetchall()
    except Exception as e:
        print(f"[DB] Get queries error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# ─── Product Operations ───────────────────────────────────────────────────────

def upsert_product(product: dict) -> bool:
    """Insert or update a product record."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO products 
                (asin, product_title, product_price, product_star_rating,
                 product_num_ratings, product_url, product_photo, specs,
                 pros, cons, sentiment, is_best_seller, is_amazon_choice, category)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                product_price = VALUES(product_price),
                product_star_rating = VALUES(product_star_rating),
                product_num_ratings = VALUES(product_num_ratings),
                specs = VALUES(specs),
                pros = VALUES(pros),
                cons = VALUES(cons),
                sentiment = VALUES(sentiment),
                last_updated = CURRENT_TIMESTAMP
        """, (
            product.get("asin"),
            product.get("product_title"),
            product.get("product_price"),
            product.get("product_star_rating"),
            product.get("product_num_ratings"),
            product.get("product_url"),
            product.get("product_photo"),
            json.dumps(product.get("specs", {})),
            json.dumps(product.get("pros", [])),
            json.dumps(product.get("cons", [])),
            product.get("sentiment", "mixed"),
            product.get("is_best_seller", False),
            product.get("is_amazon_choice", False),
            product.get("category", "")
        ))
        return True
    except Exception as e:
        print(f"[DB] Upsert product error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def get_cached_product(asin: str, max_age_hours: int = 24) -> dict:
    """Retrieve a cached product if it was updated within max_age_hours."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT * FROM products WHERE asin = %s 
               AND last_updated > DATE_SUB(NOW(), INTERVAL %s HOUR)""",
            (asin, max_age_hours)
        )
        row = cursor.fetchone()
        if row:
            row["specs"] = json.loads(row["specs"]) if row["specs"] else {}
            row["pros"] = json.loads(row["pros"]) if row["pros"] else []
            row["cons"] = json.loads(row["cons"]) if row["cons"] else []
        return row
    except Exception as e:
        print(f"[DB] Cache lookup error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


# ─── Recommendation Tracking ──────────────────────────────────────────────────

def save_recommendations(query_id: int, products: list):
    """Save recommendation results for analytics."""
    if not query_id or not products:
        return
    conn = get_connection()
    cursor = conn.cursor()
    try:
        for i, product in enumerate(products):
            cursor.execute(
                """INSERT INTO recommendations (query_id, asin, match_score, rank_position)
                   VALUES (%s, %s, %s, %s)""",
                (query_id, product.get("asin"), product.get("match_score", 0), i + 1)
            )
    except Exception as e:
        print(f"[DB] Save recommendations error: {e}")
    finally:
        cursor.close()
        conn.close()


def log_interaction(session_id: str, asin: str, interaction_type: str):
    """Log user interaction (click, purchase, etc.)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO user_interactions (session_id, asin, interaction_type) VALUES (%s, %s, %s)",
            (session_id, asin, interaction_type)
        )
    except Exception as e:
        print(f"[DB] Interaction log error: {e}")
    finally:
        cursor.close()
        conn.close()
