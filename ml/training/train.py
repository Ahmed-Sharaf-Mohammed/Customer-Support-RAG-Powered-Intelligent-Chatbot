"""
Model Training
──────────────
Three complementary recommendation models:

1. Content-Based Filtering  — TF-IDF on item text (title + category + description + features)
2. SVD Matrix Factorization — explicit ratings (scipy sparse SVD)
3. ALS Collaborative Filter — implicit browsing signals (implicit library)

Run:
    python -m ml.training.train
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import django
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

from recommender.models import Item

# ── Absolute paths ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = str(_PROJECT_ROOT / "data" / "artifacts")
PROCESSED_DIR = str(_PROJECT_ROOT / "data" / "processed")
MODELS_DIR    = str(_PROJECT_ROOT / "data" / "models")
REPORTS_DIR   = str(_PROJECT_ROOT / "data" / "reports")

for d in (MODELS_DIR, REPORTS_DIR):
    os.makedirs(d, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"   💾 Saved → {path}")


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── 1. Content-Based Filtering ───────────────────────────────────────────────

def build_content_based_model(max_features=8000, ngram_range=(1, 2), min_df=2):
    """
    Build a TF-IDF model over item text.
    Aligns rows with item_encoder so encoded item index == matrix row.
    Saves:
        cb_vectorizer.pkl
        cb_tfidf_matrix.npz
        cb_item_id_to_idx.pkl
        cb_idx_to_item_id.pkl
    """
    print("\n━━━ Content-Based Filtering ━━━")

    item_encoder = _load(f"{ARTIFACTS_DIR}/item_encoder.pkl")
    known_items  = set(item_encoder.classes_)

    qs = Item.objects.values("item_id", "title", "category", "description", "features")
    df = pd.DataFrame(list(qs))
    if df.empty:
        print("   ❌ No items in DB — skipping.")
        return None, None

    df = df[df["item_id"].isin(known_items)].copy()
    df["item_encoded"] = item_encoder.transform(df["item_id"])
    df = df.sort_values("item_encoded").reset_index(drop=True)

    # Build corpus: title (x3 weight) + category (x2) + description + features
    df["corpus"] = (
        (df["title"].fillna("")    + " ") * 3 +
        (df["category"].fillna("") + " ") * 2 +
        df["description"].fillna("") + " " +
        df["features"].fillna("")
    )

    print(f"   Items for CB model: {len(df)}")

    vectorizer   = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(df["corpus"])
    print(f"   TF-IDF shape: {tfidf_matrix.shape}")

    item_id_to_idx = dict(zip(df["item_id"], df["item_encoded"]))
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}

    _save(vectorizer,      f"{MODELS_DIR}/cb_vectorizer.pkl")
    save_npz(              f"{MODELS_DIR}/cb_tfidf_matrix.npz", tfidf_matrix)
    _save(item_id_to_idx,  f"{MODELS_DIR}/cb_item_id_to_idx.pkl")
    _save(idx_to_item_id,  f"{MODELS_DIR}/cb_idx_to_item_id.pkl")

    print(f"   ✅ Content-Based model ready.")
    return vectorizer, tfidf_matrix


# ─── 2. SVD Matrix Factorization ──────────────────────────────────────────────

def train_svd_model(n_factors: int = 50):
    """
    Latent-factor model via truncated SVD on the explicit ratings matrix.
    Prediction: R̂ = U · diag(σ) · Vᵀ

    Saves: svd_model.pkl
    """
    print("\n━━━ SVD Matrix Factorization ━━━")

    matrix_path = f"{ARTIFACTS_DIR}/explicit_matrix.npz"
    if not os.path.exists(matrix_path):
        print("   ❌ explicit_matrix.npz not found — run preprocess pipeline first.")
        return None

    matrix = load_npz(matrix_path).astype(np.float32)
    print(f"   Matrix shape : {matrix.shape}")
    print(f"   Non-zeros    : {matrix.nnz}")

    if matrix.nnz == 0:
        print("   ❌ Empty matrix — skipping SVD.")
        return None

    k = min(n_factors, min(matrix.shape) - 1)
    print(f"   Running SVD  : k={k}")

    U, sigma, Vt = svds(matrix, k=k)

    # Sort by descending singular value
    order  = np.argsort(sigma)[::-1]
    U      = U[:, order]
    sigma  = sigma[order]
    Vt     = Vt[order, :]

    svd_model = {
        "U":             U,       # (n_users, k)
        "sigma":         sigma,   # (k,)
        "Vt":            Vt,      # (k, n_items)
        "n_factors":     k,
        "matrix_shape":  matrix.shape,
    }

    _save(svd_model, f"{MODELS_DIR}/svd_model.pkl")
    print(f"   ✅ SVD ready — U:{U.shape}  σ:{sigma.shape}  Vt:{Vt.shape}")
    return svd_model


# ─── 3. ALS Collaborative Filtering ──────────────────────────────────────────

def train_als_model(factors: int = 64, iterations: int = 20,
                    regularization: float = 0.01):
    """
    Matrix factorization optimised for implicit feedback via
    Alternating Least Squares (implicit library).

    Saves: als_model.pkl
    """
    print("\n━━━ ALS Collaborative Filtering ━━━")

    try:
        import implicit
    except ImportError:
        print("   ❌ `implicit` not installed.  pip install implicit")
        return None

    matrix_path = f"{ARTIFACTS_DIR}/implicit_matrix.npz"
    if not os.path.exists(matrix_path):
        print("   ❌ implicit_matrix.npz not found — run preprocess pipeline first.")
        return None

    matrix = load_npz(matrix_path).astype(np.float32)
    print(f"   Matrix shape : {matrix.shape}")
    print(f"   Non-zeros    : {matrix.nnz}")

    if matrix.nnz == 0:
        print("   ❌ Empty matrix — skipping ALS.")
        return None

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        use_gpu=False,
        calculate_training_loss=True,
    )

    # ALS expects item-user matrix (items as rows)
    print(f"   Training ALS : factors={factors}  iter={iterations}  reg={regularization}")
    model.fit(matrix.T.tocsr())

    _save(model, f"{MODELS_DIR}/als_model.pkl")
    print("   ✅ ALS ready.")
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(n_svd_factors: int = 50, als_factors: int = 64,
         als_iterations: int = 20, als_regularization: float = 0.01):
    """Train all three models and save a training summary."""
    print("🚀 Model Training Pipeline\n")

    results = {}

    cb_vectorizer, _ = build_content_based_model()
    results["content_based"] = cb_vectorizer is not None

    svd = train_svd_model(n_factors=n_svd_factors)
    results["svd"] = svd is not None

    als = train_als_model(
        factors=als_factors,
        iterations=als_iterations,
        regularization=als_regularization,
    )
    results["als"] = als is not None

    summary = {
        "models_trained": results,
        "params": {
            "svd": {"n_factors": n_svd_factors},
            "als": {"factors": als_factors, "iterations": als_iterations,
                    "regularization": als_regularization},
        },
    }
    summary_path = f"{MODELS_DIR}/training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training complete:")
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    print(f"Summary saved → {summary_path}")
    return results


if __name__ == "__main__":
    main()
