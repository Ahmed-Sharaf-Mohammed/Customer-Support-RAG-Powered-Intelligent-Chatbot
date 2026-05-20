# Personalized Recommendation System with Feedback Loop

A Django-based e-commerce recommendation engine that combines three complementary machine learning approaches — Content-Based Filtering, SVD Matrix Factorization, and ALS Collaborative Filtering — into a single Hybrid Predictor. User behavior is tracked in real time and fed back into a retraining pipeline to keep recommendations fresh.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Database Design](#database-design)
- [ML Pipeline](#ml-pipeline)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Retraining the Models](#retraining-the-models)
- [Monitoring & Logs](#monitoring--logs)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

RecoShop delivers personalized product recommendations by learning from two types of user signals:

- **Explicit feedback** — star ratings and written reviews submitted by users.
- **Implicit feedback** — page views, clicks, searches, and cart actions tracked automatically.

Both signals are processed into sparse interaction matrices, used to train three ML models, and combined at inference time through a weighted Hybrid Predictor. A management command (`retrain`) rebuilds the entire pipeline from the latest data, evaluates the new models against the previous run, and hot-reloads the predictor without restarting the server.

---

## Architecture

```
Browser / Client
      │
      ▼
Django Views  ──►  Tracking Layer  ──►  interactions.log
      │                                        │
      ▼                                        ▼
  Services                            Preprocessing Pipeline
      │                                  (encoders, matrices)
      ▼                                        │
ML Inference Layer                             ▼
  ├── ContentBasedPredictor            Training Pipeline
  ├── SVDPredictor                       ├── SVD
  ├── ALSPredictor                       ├── ALS
  └── HybridPredictor ◄──────────────── └── Content-Based
      │
      ▼
  REST API  (/api/recommendations/)
```

MLflow records every training run, hyperparameters, and evaluation metrics. Artifacts (model `.pkl` files, sparse matrices, encoders) are stored under `data/`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Django 5.0 |
| Database | SQLite (development) |
| ML — Explicit CF | SciPy sparse SVD (`scipy.sparse.linalg.svds`) |
| ML — Implicit CF | `implicit` library (ALS) |
| ML — Content-Based | scikit-learn TF-IDF |
| Data Processing | pandas, NumPy, pyarrow |
| Experiment Tracking | MLflow |
| Data Format | Parquet, XLSX, NPZ (sparse matrices), Pickle |

---

## Project Structure

```
project/
├── manage.py
│
├── config/
│   ├── settings.py          # Django settings (DB, logging, MLflow)
│   └── urls.py              # All routes mounted under /api/
│
├── recommender/             # Main Django app
│   ├── models.py            # Item, UserInteraction, UserBrowsingLog, SearchLog
│   ├── views.py             # home, product_list, product_detail, search, dashboard
│   ├── urls.py              # URL definitions for pages and APIs
│   ├── services/
│   │   ├── item_service.py          # get_items_by_ids, get_popular_items, search_items
│   │   └── interaction_service.py   # log_interaction → writes to interactions.log
│   ├── tracking/
│   │   ├── browsing_tracker.py      # log_event, log_search_event
│   │   └── session_tracker.py       # recently viewed, search history, cart
│   ├── api_helpers/
│   │   └── track_api.py             # AJAX endpoints: /api/track/, /api/rate/
│   └── management/commands/
│       ├── import_data.py           # Populate DB from items/interactions files
│       ├── preprocess.py            # Build ML matrices
│       └── retrain.py               # Full retrain + evaluate + MLflow log
│
├── ml/
│   ├── inference/
│   │   ├── predict.py               # SVDPredictor, ALSPredictor, ContentBasedPredictor, HybridPredictor
│   │   ├── recommender.py           # Singleton: get_recommendations(), reload_predictor()
│   │   └── loader.py                # DB item lookup helpers
│   ├── preprocessing/
│   │   ├── global_encoders.py       # user_encoder.pkl + item_encoder.pkl
│   │   ├── explicit_transform.py    # Clean and normalize ratings
│   │   ├── explicit_interaction_matrix.py   # Build explicit_matrix.npz
│   │   └── implicit_interaction_matrix.py   # Build implicit_matrix.npz (browsing)
│   ├── training/
│   │   ├── train.py                 # Train SVD, ALS, Content-Based models
│   │   ├── evaluate.py              # Precision/Recall/MAP/NDCG @5,10,20
│   │   └── tuning.py                # Grid search for best hyperparameters
│   └── pipelines/
│       ├── preprocess_pipeline.py   # Preprocessing orchestrator
│       └── retrain_pipeline.py      # Full pipeline (preprocess → train → eval → MLflow)
│
├── data/
│   ├── raw/                         # Original source files (gitignored)
│   ├── processed/                   # Cleaned parquet/xlsx files
│   ├── artifacts/                   # Encoders and sparse matrices
│   ├── models/                      # Trained model pickle files
│   └── reports/                     # evaluation_report.json, retrain_meta.json
│
└── logs/
    ├── interactions.log             # All user events (views, clicks, ratings, searches)
    ├── ml.log                       # Training, evaluation, pipeline logs
    └── app.log                      # General Django application logs
```

---

## Database Design

### `recommender_item`

Stores the product catalog. Populated once via `import_data` and queried at runtime.

| Column | Type | Notes |
|---|---|---|
| `item_id` | CharField (PK) | Amazon ASIN or equivalent |
| `title` | CharField | Product name |
| `category` | CharField | Primary category (indexed) |
| `categories` | TextField | Full category path |
| `description` | TextField | Long-form description |
| `features` | TextField | Bullet-point features |
| `price` | FloatField | Listed price (nullable) |
| `avg_rating` | FloatField | Pre-computed average |
| `rating_count` | IntegerField | Number of reviews |
| `store` | CharField | Seller / brand |
| `images` | TextField | Raw image URL data |
| `bought_together` | TextField | Raw co-purchase data |

Indexes: `category`, `avg_rating`, `price`.

---

### `recommender_userinteraction`

Stores explicit feedback — ratings and reviews.

| Column | Type | Notes |
|---|---|---|
| `user_id` | CharField | Authenticated user ID |
| `item_id` | CharField | References `Item.item_id` |
| `rating` | FloatField | 1.0 – 5.0 (nullable) |
| `review_text` | TextField | Written review |
| `review_title` | CharField | Review headline |
| `verified` | BooleanField | Verified purchase flag |
| `helpful_votes` | IntegerField | Community upvotes |
| `timestamp` | DateTimeField | Submission time |

Indexes: `(item_id, timestamp)`, `(user_id, timestamp)`.

---

### `recommender_userbrowsinglog`

Stores implicit feedback — all behavioral events.

| Column | Type | Notes |
|---|---|---|
| `user_id` | CharField | User or session ID |
| `item_id` | CharField | Item involved in the event |
| `event_type` | CharField | view, click, add_to_cart, remove_from_cart, search |
| `session_id` | CharField | Browser session |
| `device` | CharField | desktop / mobile / tablet |
| `source` | CharField | homepage, search, recommendation, category, direct |
| `timestamp` | DateTimeField | Auto-set on insert |

Indexes: `(user_id, timestamp)`, `item_id`, `session_id`.

---

### `recommender_searchlog`

Stores search queries for future intent modeling.

| Column | Type | Notes |
|---|---|---|
| `user_id` | CharField | User or session ID |
| `query` | CharField | Raw search text |
| `results_count` | IntegerField | Number of results returned |
| `session_id` | CharField | Browser session |
| `device` | CharField | Device type |
| `timestamp` | DateTimeField | Auto-set on insert |

---

## ML Pipeline

### Preprocessing

```
User interactions (DB)
        │
        ▼
global_encoders.py   →  user_encoder.pkl, item_encoder.pkl
        │
        ├── explicit_transform.py     → cleaned ratings DataFrame
        │       └── explicit_interaction_matrix.py  →  explicit_matrix.npz
        │
        └── implicit_interaction_matrix.py  →  implicit_matrix.npz
```

### Training

Three models are trained independently:

| Model | Algorithm | Input |
|---|---|---|
| SVD | Truncated SVD (`scipy.sparse.linalg.svds`) | `explicit_matrix.npz` |
| ALS | Alternating Least Squares (`implicit`) | `implicit_matrix.npz` |
| Content-Based | TF-IDF cosine similarity (`sklearn`) | Item text fields |

### Inference — HybridPredictor

At request time, the `HybridPredictor` combines scores from all three models using a weighted average and falls back to a popularity-based ranker for new users with no interaction history.

### Evaluation Metrics

Each retrain run computes and logs: **Precision@K**, **Recall@K**, **MAP@K**, **NDCG@K** for K ∈ {5, 10, 20}. Results are compared against the previous run and saved to `data/reports/evaluation_report.json`.

---

## API Endpoints

All routes are mounted under `/api/`.

| Method | URL | Description |
|---|---|---|
| GET | `/api/` | Home page |
| GET | `/api/products/` | Product catalog with filtering and sorting |
| GET | `/api/products/<item_id>/` | Product detail page |
| GET | `/api/search/` | Full-text search |
| GET | `/api/dashboard/` | User dashboard with personalized recommendations |
| GET | `/api/recommendations/?user_id=<id>` | JSON recommendations API |
| POST | `/api/track/` | AJAX: log a user behavior event |
| POST | `/api/rate/` | AJAX: submit an explicit rating |
| GET | `/api/cart/count/` | AJAX: current cart item count |
| GET/POST | `/api/register/` | User registration |
| GET/POST | `/api/login/` | User login |
| POST | `/api/logout/` | User logout |
| GET | `/admin/` | Django admin panel |

### Recommendations API Response

```json
{
  "user_id": "1",
  "count": 6,
  "recommendations": [
    {
      "item_id": "B001XK5L6O",
      "title": "...",
      "category": "Electronics",
      "avg_rating": 4.5,
      "price": 29.99,
      "image": "https://m.media-amazon.com/..."
    }
  ]
}
```

---

## Setup & Installation

### Requirements

- Python 3.11+
- pip

### Install dependencies

```bash
pip install django numpy pandas scipy scikit-learn implicit mlflow openpyxl pyarrow
```

| Package | Purpose |
|---|---|
| django | Web framework |
| numpy / scipy | Sparse matrices and linear algebra |
| pandas | Data loading and transformation |
| scikit-learn | SVD, TF-IDF, LabelEncoder, metrics |
| implicit | ALS collaborative filtering |
| mlflow | Experiment tracking |
| openpyxl | Reading XLSX files |
| pyarrow | Reading and writing Parquet files |

---

## Running the Project

### Step 1 — Create the database tables

```bash
python manage.py migrate
```

This creates four tables: `recommender_item`, `recommender_userinteraction`, `recommender_userbrowsinglog`, and `recommender_searchlog`.

### Step 2 — Import data

```bash
# From XLSX files (included sample data)
python manage.py import_data \
  --items-path data/processed/items.xlsx \
  --interactions-path data/processed/interactions.xlsx

# Or from Parquet files (if available in data/processed/)
python manage.py import_data
```

Verify:

```bash
python manage.py shell -c "
from recommender.models import Item, UserInteraction
print('Items:', Item.objects.count())
print('Interactions:', UserInteraction.objects.count())
"
```

### Step 3 — Build ML matrices

> Skip this step if `data/artifacts/` and `data/models/` already exist (they are included in the zip).

```bash
# Build all matrices
python manage.py preprocess

# Or build individual steps
python manage.py preprocess --step encoders
python manage.py preprocess --step explicit
python manage.py preprocess --step implicit
```

Output files:

```
data/artifacts/user_encoder.pkl
data/artifacts/item_encoder.pkl
data/artifacts/explicit_matrix.npz
data/artifacts/implicit_matrix.npz
```

### Step 4 — Start the server

```bash
python manage.py runserver
```

Open http://127.0.0.1:8000/api/ in your browser.

### Step 5 — Test recommendations

Via API:
```
http://127.0.0.1:8000/api/recommendations/?user_id=1
```

Via Django shell:
```bash
python manage.py shell -c "
from ml.inference.recommender import get_recommendations
recs = get_recommendations(user_id='1', user_item_ids=[], k=5)
print('Recommended IDs:', recs)
"
```

Via the UI:
1. Register at http://127.0.0.1:8000/api/register/
2. Browse 3–4 products
3. Visit http://127.0.0.1:8000/api/dashboard/
4. Check the **Personalized Recommendations** section

---

## Retraining the Models

```bash
# Preview what will run without executing
python manage.py retrain --dry-run

# Full retrain using existing matrices
python manage.py retrain --skip-preprocess

# Full retrain including matrix rebuild
python manage.py retrain

# Full retrain with hyperparameter grid search
python manage.py retrain --retune
```

The retrain pipeline runs eight steps:

1. Update user and item encoders
2. Re-transform explicit ratings
3. Rebuild explicit sparse matrix
4. Rebuild implicit browsing matrix
5. Train SVD, ALS, and Content-Based models
6. Compute Precision/Recall/MAP/NDCG and compare with the previous run
7. Log all parameters, metrics, and artifacts to MLflow
8. Hot-reload the inference predictor (no server restart needed)

### MLflow Dashboard

```bash
mlflow ui --backend-store-uri mlruns/ --port 5000
```

Open http://127.0.0.1:5000 to view all training runs, compare hyperparameters, and download evaluation reports.

---

## Monitoring & Logs

```bash
# User events (views, clicks, ratings, searches)
tail -f logs/interactions.log

# ML logs (training, inference, pipelines)
tail -f logs/ml.log

# General Django application logs
tail -f logs/app.log
```

Sample `interactions.log` entries:

```
2025-05-16 14:23:01 | event=view           user=3  item=B001XK5L6O  source=recommendation  device=desktop
2025-05-16 14:23:45 | event=search         user=3  query='wireless headphones'  results=12
2025-05-16 14:24:10 | event=explicit_rating user=3  item=B001XK5L6O  rating=4.0  verified=True
2025-05-16 14:25:00 | event=add_to_cart    user=3  item=B002GEN8I4   source=recommendation
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `No module named 'implicit'` | Package not installed | `pip install implicit` |
| Empty recommendations | Database is empty | `python manage.py import_data` |
| `data/models/*.pkl not found` | Models not trained yet | `python manage.py retrain --skip-preprocess` |
| `404` on `/` | Routes are under `/api/` | Go to `http://127.0.0.1:8000/api/` |
| `interactions.log` is empty | No events logged yet | Open any product page — events are logged automatically |
| MLflow UI shows nothing | No retrain has been run | Run `python manage.py retrain` first |

---

## Health Checklist

| Check | Expected Result |
|---|---|
| `python manage.py check` | No errors |
| `python manage.py migrate` | `No migrations to apply` |
| `Item.objects.count() > 0` | True (in Django shell) |
| `/api/recommendations/?user_id=1` | `count > 0` in JSON response |
| Dashboard after login | Non-empty recommendations section |
| `python manage.py retrain --dry-run` | Prints all 8 steps |
| `logs/interactions.log` | New entries after any product view |
| `mlflow ui` | Opens and shows experiment runs |
