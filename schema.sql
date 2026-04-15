-- HardwareAI Database Schema
-- Run: mysql -u root -p < schema.sql

CREATE DATABASE IF NOT EXISTS hardwareai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE hardwareai;

-- User Queries: stores every search
CREATE TABLE IF NOT EXISTS user_queries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_text TEXT NOT NULL,
    parsed_requirements JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(64),
    result_count INT DEFAULT 0,
    INDEX idx_session (session_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Products: cached Amazon product data
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
    sentiment VARCHAR(20) DEFAULT 'mixed',
    is_best_seller BOOLEAN DEFAULT FALSE,
    is_amazon_choice BOOLEAN DEFAULT FALSE,
    category VARCHAR(100),
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_asin (asin),
    INDEX idx_price (product_price),
    INDEX idx_rating (product_star_rating)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Recommendations: tracks what was shown for each query
CREATE TABLE IF NOT EXISTS recommendations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_id INT NOT NULL,
    asin VARCHAR(20) NOT NULL,
    match_score DECIMAL(5,2),
    rank_position INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES user_queries(id) ON DELETE CASCADE,
    INDEX idx_query (query_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- User Interactions: click / view / purchase events
CREATE TABLE IF NOT EXISTS user_interactions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64),
    asin VARCHAR(20),
    interaction_type VARCHAR(30),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_asin (asin)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
