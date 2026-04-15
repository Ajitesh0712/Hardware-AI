# HardwareAI — Electronic Recommendation System

AI-powered electronics advisor that understands natural language and recommends real Amazon products using Claude AI + ML.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python 3.11+, Flask |
| AI / NLP | Anthropic Claude API (claude-sonnet-4) |
| Product Data | RapidAPI — Real-Time Amazon Data |
| Data Processing | Pandas, NumPy |
| ML Engine | Scikit-learn (weighted cosine scoring) |
| Database | MySQL 8.0+ |

---

## Project Structure

```
hardwareai/
├── backend/
│   ├── app.py                  # Flask server & API routes
│   ├── claude_ai.py            # Claude NLP parser & sentiment
│   ├── amazon_api.py           # RapidAPI Amazon product fetcher
│   ├── data_processor.py       # Pandas/NumPy cleaning pipeline
│   ├── recommendation_engine.py # Scikit-learn ML scoring
│   └── database.py             # MySQL connection & queries
├── frontend/
│   └── index.html              # Full frontend (updated)
├── schema.sql                  # MySQL table definitions
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### 1. Clone & Install

```bash
cd hardwareai
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

You need:
- **Anthropic API key** → https://console.anthropic.com
- **RapidAPI key** → https://rapidapi.com/real-time-amazon-data/api/real-time-amazon-data
  - Subscribe to the **"Real-Time Amazon Data"** API (free tier: 100 req/month)

### 3. Set Up MySQL

```bash
# Create the database and tables
mysql -u root -p < schema.sql
```

Update `.env` with your MySQL credentials.

### 4. Run the Backend

```bash
cd backend
python app.py
```

Server starts at `http://localhost:5000`

### 5. Open the Frontend

Open `frontend/index.html` in your browser, OR visit `http://localhost:5000` (Flask serves it automatically).

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/api/recommend` | Main search — returns ranked products |
| POST | `/api/refine` | Refine existing search with new constraints |
| POST | `/api/interaction` | Log user interactions (click/view) |
| GET | `/api/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "I need a laptop under $1200 for machine learning and coding"}'
```

---

## How It Works

1. **User types** a natural language query
2. **Claude AI** parses it into structured requirements (budget, use case, specs)
3. **Amazon API** fetches real-time product listings matching the query
4. **Pandas/NumPy** cleans and normalizes the data
5. **Scikit-learn** scores each product using weighted feature vectors tuned per use case (gaming, coding, design, etc.)
6. **Claude AI** analyzes reviews for pros/cons and generates a summary
7. **MySQL** caches products and logs interactions for performance
8. **Frontend** displays ranked results with specs, ratings, pros/cons

---

## Environment Variables

```
ANTHROPIC_API_KEY=      # Anthropic Claude API key
RAPIDAPI_KEY=           # RapidAPI key for Amazon data
DB_HOST=localhost       # MySQL host
DB_PORT=3306
DB_NAME=hardwareai
DB_USER=root
DB_PASSWORD=
FLASK_ENV=development
PORT=5000
```
