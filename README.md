# ElectronicsAI — FCDS Project Based Learning

ElectronicsAI is a product discovery and recommendation prototype for electronics. It combines a seven-phase data science workflow with a Flask application that parses a shopping request, retrieves product listings, extracts comparable attributes, applies budget/category filters, and ranks candidate products.

## Project contents

- `Data Science/01_data_understanding.ipynb` through `07_model_evaluation.ipynb`: the complete FCDS workflow, from data audit and cleaning through exploratory/statistical analysis, feature engineering, model development and final evaluation.
- `backend/product_features.py`: the shared title classifier and feature extractor used by offline feature engineering and online inference.
- `backend/model_artifacts/product_score.joblib`: the saved scikit-learn pipeline loaded by the app.
- `backend/app.py`, `backend/data_processor.py`, `backend/recommendation_engine.py`: the web/API and ranking flow.
- `frontend/index.html`: the browser interface.
- `deliverables/`: lab presentation and project report.

## Run the app locally

Use Python 3.11 or newer and install the compatible dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env` for Gemini-based query parsing and summaries. The current product search/review adapter scrapes Amazon India pages, so the live path requires network access and may be affected by page changes, throttling, or access controls. MySQL persistence is optional; configure `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD` when a database is available. Never commit `.env` or credentials.

Start from the repository root:

```bash
python -m backend.app
```

Open `http://127.0.0.1:5001`. Example requests include “gaming laptop under 60000”, “wireless earbuds”, and “phone with 8 GB RAM”. The backend also exposes `POST /api/recommend` and `POST /api/refine`.

For deployment, the `Procfile` runs `gunicorn backend.app:app`; set the same environment variables in the deployment environment. The Flask development server is intended for a local demonstration.

## FCDS workflow and findings

The analysis uses 5,010 product records and retains all rows through feature engineering. Note that `review_rating` in the source file is the star value of one sampled review (always a whole number 1–5), not a product's average rating. It includes data understanding, cleaning, univariate and bivariate EDA, non-parametric statistical analysis, rule-based category/spec extraction, regression model comparison, and a grouped holdout evaluation. The project report documents the dataset caveats, methods, results, and presentation procedure.

The learned regression target is a **proxy score** derived from rating, review-count popularity, and discount percentage. It is not a user satisfaction label, recommendation probability, or causal outcome. On the reserved 20% holdout (995 rows, grouped by base product title so colour/storage variants of one product stay on the same side of the split), the saved Random Forest achieved MAE 0.00424, RMSE 0.00933, and R² 0.9932. These strong values are expected in part because the predictors contain the same ingredients used to construct the proxy target; they demonstrate pipeline consistency, not real-world recommendation quality. Ranking also includes explicit use-case matching and is not simply the model prediction.

## Model and live-data flow

The shared feature module preserves the same feature names/order for training and inference. The saved pipeline includes training-fitted missing-value imputation; it is loaded once when the backend imports and is not retrained per request. Live products retain missing ratings for the fitted imputer, and incomplete review evidence is shown as unavailable instead of fabricated claims.

The offline training/evaluation workflow is reproducible from the notebooks. The external Amazon page, Gemini API, and MySQL service were not exercised as a connected end-to-end deployment during this project pass; the app integration was checked with representative scraper-shaped records. Live service behavior therefore depends on valid credentials, network access, and current external service behavior.

## Rebuild the model

Run notebooks 01–07 in order from the `Data Science/` directory (for example `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 01_data_understanding.ipynb`). Phase 6 selects a model on grouped training/validation data and writes the model artifact; Phase 7 evaluates it once on the reserved grouped holdout. Do not use the holdout metrics to tune the model. Generated tables, plots, and the fitted artifact are stored under `Data Science/model_outputs/`, `Data Science/phase_outputs/`, and `backend/model_artifacts/`.

## Offline website checks

`tests/test_website_pipeline.py` checks requirement parsing, chat follow-ups, category/model/budget filtering, ranking and error responses through Flask's test client. Amazon, review fetching and MySQL are replaced with fakes and Gemini is switched off, so the tests need no network or credentials:

```bash
python -m unittest discover -s tests -v
```

Passing these tests does not show that live Amazon scraping works; Amazon may return a robot-check page, which the API now reports as a 502 error instead of "no products found".

