"""
Hyperparameter Tuning
─────────────────────
Grid search over the key knobs of SVD and ALS.
The best configuration is saved to data/reports/best_params.json and
is automatically picked up by the training & retrain pipelines.

Run:
    python -m ml.training.tuning [--quick]

--quick uses a narrower grid (faster for dev/testing).
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from itertools import product

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = str(_PROJECT_ROOT / "data" / "artifacts")
PROCESSED_DIR = str(_PROJECT_ROOT / "data" / "processed")
MODELS_DIR    = str(_PROJECT_ROOT / "data" / "models")
REPORTS_DIR   = str(_PROJECT_ROOT / "data" / "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Import evaluation helpers
from ml.training.evaluate import (
    evaluate_recommendations,
    temporal_train_test_split,
    random_train_test_split,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _primary_metric(metrics: dict, k: int = 10) -> float:
    """NDCG@k as the optimisation target."""
    return metrics.get(f"@{k}", {}).get("NDCG", 0.0)


def _load_explicit_data():
    path = f"{PROCESSED_DIR}/explicit_aggregated.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}  — run preprocess pipeline first.")
    df = pd.read_parquet(path)
    if "timestamp_unix" in df.columns:
        return temporal_train_test_split(df)
    return random_train_test_split(df)


# ─── SVD Grid Search ─────────────────────────────────────────────────────────

def tune_svd(quick: bool = False, target_k: int = 10) -> dict:
    print("\n━━━ Tuning SVD ━━━")

    from scipy.sparse import load_npz
    from scipy.sparse.linalg import svds

    matrix_path = f"{ARTIFACTS_DIR}/explicit_matrix.npz"
    if not os.path.exists(matrix_path):
        print("   ❌ explicit_matrix.npz not found — skipping.")
        return {}

    matrix = load_npz(matrix_path).astype(np.float32)
    if matrix.nnz == 0:
        print("   ❌ Empty matrix — skipping.")
        return {}

    try:
        train_df, test_df = _load_explicit_data()
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return {}

    if test_df.empty:
        print("   ❌ Empty test set.")
        return {}

    gt   = {uid: grp["item_encoded"].tolist()
            for uid, grp in test_df.groupby("user_encoded")}
    seen = {uid: set(grp["item_encoded"].tolist())
            for uid, grp in train_df.groupby("user_encoded")}

    n_factors_grid = [20, 50] if quick else [20, 50, 100, 150]
    max_eval_k     = max(target_k, 20)

    best_score, best_params, all_results = -1.0, {}, []

    print(f"   Grid: n_factors={n_factors_grid}")

    for n_factors in n_factors_grid:
        k = min(n_factors, min(matrix.shape) - 1)
        U, sigma, Vt = svds(matrix, k=k)
        order = np.argsort(sigma)[::-1]
        predicted = np.dot(U[:, order] * sigma[order], Vt[order, :])

        recs: dict = {}
        for user_idx in gt:
            if user_idx >= predicted.shape[0]:
                continue
            scores = predicted[user_idx].copy()
            for s in seen.get(user_idx, set()):
                if s < len(scores):
                    scores[s] = -np.inf
            top = np.argsort(scores)[::-1][:max_eval_k]
            recs[user_idx] = top.tolist()

        metrics = evaluate_recommendations(recs, gt, (target_k,))
        score   = _primary_metric(metrics, target_k)

        result  = {"n_factors": n_factors, f"ndcg@{target_k}": round(score, 5), **metrics}
        all_results.append(result)
        flag = " ← best" if score > best_score else ""
        print(f"   n_factors={n_factors:>4}  NDCG@{target_k}={score:.4f}{flag}")

        if score > best_score:
            best_score  = score
            best_params = {"n_factors": n_factors}

    print(f"   🏆 Best SVD: {best_params}  NDCG@{target_k}={best_score:.4f}")

    out = {"best_params": best_params, "best_score": best_score,
           "grid_results": all_results}
    with open(f"{REPORTS_DIR}/svd_tuning.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    return best_params


# ─── ALS Grid Search ──────────────────────────────────────────────────────────

def tune_als(quick: bool = False, target_k: int = 10) -> dict:
    print("\n━━━ Tuning ALS ━━━")

    try:
        import implicit
    except ImportError:
        print("   ❌ `implicit` not installed.  pip install implicit")
        return {}

    from scipy.sparse import load_npz

    matrix_path = f"{ARTIFACTS_DIR}/implicit_matrix.npz"
    data_path   = f"{PROCESSED_DIR}/implicit_aggregated.parquet"

    for p in (matrix_path, data_path):
        if not os.path.exists(p):
            print(f"   ❌ Missing: {p}")
            return {}

    matrix = load_npz(matrix_path).astype(np.float32)
    if matrix.nnz == 0:
        print("   ❌ Empty matrix — skipping.")
        return {}

    df = pd.read_parquet(data_path)

    # Hold-out: random 20 % of users
    np.random.seed(42)
    all_users  = df["user_encoded"].unique()
    test_users = set(np.random.choice(
        all_users, size=max(1, int(len(all_users) * 0.2)), replace=False))

    train_df = df[~df["user_encoded"].isin(test_users)]
    test_df  = df[ df["user_encoded"].isin(test_users)]

    if test_df.empty:
        print("   ❌ Empty test set.")
        return {}

    gt   = {uid: grp["item_encoded"].tolist()
            for uid, grp in test_df.groupby("user_encoded")}

    # Grid
    if quick:
        factors_list = [32, 64]
        iters_list   = [15]
        reg_list     = [0.01]
    else:
        factors_list = [32, 64, 128]
        iters_list   = [15, 20, 30]
        reg_list     = [0.01, 0.05, 0.1]

    grid = list(product(factors_list, iters_list, reg_list))
    print(f"   Grid size: {len(grid)} combinations")

    best_score, best_params, all_results = -1.0, {}, []

    for factors, iters, reg in grid:
        model = implicit.als.AlternatingLeastSquares(
            factors=factors, iterations=iters,
            regularization=reg, use_gpu=False,
        )
        model.fit(matrix.T.tocsr())

        recs: dict = {}
        for user_idx in gt:
            if user_idx >= matrix.shape[0]:
                continue
            try:
                ids, _ = model.recommend(
                    user_idx, matrix[user_idx],
                    N=target_k, filter_already_liked_items=True,
                )
                recs[user_idx] = ids[:target_k].tolist()
            except Exception:
                pass

        metrics = evaluate_recommendations(recs, gt, (target_k,))
        score   = _primary_metric(metrics, target_k)

        result = {"factors": factors, "iterations": iters,
                  "regularization": reg, f"ndcg@{target_k}": round(score, 5)}
        all_results.append(result)
        flag = " ← best" if score > best_score else ""
        print(f"   f={factors:>4} i={iters:>3} r={reg}  NDCG@{target_k}={score:.4f}{flag}")

        if score > best_score:
            best_score  = score
            best_params = {"factors": factors, "iterations": iters, "regularization": reg}

    print(f"   🏆 Best ALS: {best_params}  NDCG@{target_k}={best_score:.4f}")

    out = {"best_params": best_params, "best_score": best_score,
           "grid_results": all_results}
    with open(f"{REPORTS_DIR}/als_tuning.json", "w") as f:
        json.dump(out, f, indent=2)

    return best_params


# ─── Content-Based tuning ────────────────────────────────────────────────────

def tune_content_based(quick: bool = False) -> dict:
    """
    Tune TF-IDF vocabulary size and n-gram range.
    We retrain the vectorizer on-the-fly (no ALS/SVD cost).
    """
    print("\n━━━ Tuning Content-Based ━━━")

    try:
        import django
        import os as _os
        _os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        django.setup()
        from recommender.models import Item
    except Exception as e:
        print(f"   ❌ Django not available: {e}")
        return {}

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        train_df, test_df = _load_explicit_data()
        item_encoder = _load(f"{ARTIFACTS_DIR}/item_encoder.pkl")
    except Exception as e:
        print(f"   ❌ {e}")
        return {}

    known_items = set(item_encoder.classes_)
    qs = Item.objects.values("item_id", "title", "category", "description", "features")
    df_items = pd.DataFrame(list(qs))
    df_items = df_items[df_items["item_id"].isin(known_items)].copy()
    df_items["item_encoded"] = item_encoder.transform(df_items["item_id"])
    df_items = df_items.sort_values("item_encoded").reset_index(drop=True)
    df_items["corpus"] = (
        df_items["title"].fillna("")       * 3 + " " +
        df_items["category"].fillna("")    * 2 + " " +
        df_items["description"].fillna("") + " " +
        df_items["features"].fillna("")
    )

    if quick:
        max_feat_list  = [3000, 8000]
        ngram_list     = [(1, 1), (1, 2)]
    else:
        max_feat_list  = [3000, 8000, 15000]
        ngram_list     = [(1, 1), (1, 2), (1, 3)]

    target_k    = 10
    gt          = {uid: set(grp["item_encoded"].tolist())
                   for uid, grp in test_df.groupby("user_encoded")}
    seen_train  = {uid: list(grp["item_encoded"].tolist())
                   for uid, grp in train_df.groupby("user_encoded")}

    best_score, best_params, all_results = -1.0, {}, []
    idx_to_item_id = dict(zip(df_items["item_encoded"], df_items["item_id"]))
    item_id_to_idx = {v: k for k, v in idx_to_item_id.items()}

    from collections import Counter

    for max_feat, ngram in product(max_feat_list, ngram_list):
        vec = TfidfVectorizer(max_features=max_feat, ngram_range=ngram,
                              sublinear_tf=True, stop_words="english", min_df=2)
        try:
            tfidf = vec.fit_transform(df_items["corpus"])
        except Exception:
            continue

        recs: dict = {}
        for user_idx in gt:
            train_items = seen_train.get(user_idx, [])
            if not train_items:
                continue
            counter: Counter = Counter()
            seen_set = set(train_items)
            for enc_idx in train_items[-5:]:
                if enc_idx >= tfidf.shape[0]:
                    continue
                sims = cosine_similarity(tfidf[enc_idx], tfidf).flatten()
                sims[enc_idx] = -1
                for rank, r in enumerate(np.argsort(sims)[::-1][:target_k * 3]):
                    if r not in seen_set:
                        counter[r] += 1.0 / (rank + 1)
            recs[user_idx] = [enc for enc, _ in counter.most_common(target_k)]

        metrics = evaluate_recommendations(recs, {uid: list(s) for uid, s in gt.items()},
                                           (target_k,))
        score   = _primary_metric(metrics, target_k)

        result  = {"max_features": max_feat, "ngram_range": list(ngram),
                   f"ndcg@{target_k}": round(score, 5)}
        all_results.append(result)
        flag = " ← best" if score > best_score else ""
        print(f"   max_feat={max_feat:>6} ngram={ngram}  NDCG@{target_k}={score:.4f}{flag}")

        if score > best_score:
            best_score  = score
            best_params = {"max_features": max_feat, "ngram_range": list(ngram)}

    print(f"   🏆 Best CB: {best_params}  NDCG@{target_k}={best_score:.4f}")

    out = {"best_params": best_params, "best_score": best_score,
           "grid_results": all_results}
    with open(f"{REPORTS_DIR}/cb_tuning.json", "w") as f:
        json.dump(out, f, indent=2)

    return best_params


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(quick: bool = False) -> dict:
    print("🚀 Hyperparameter Tuning\n")
    mode = "quick" if quick else "full"
    print(f"   Mode: {mode}")

    best = {
        "svd": tune_svd(quick=quick),
        "als": tune_als(quick=quick),
        "content_based": tune_content_based(quick=quick),
    }

    out_path = f"{REPORTS_DIR}/best_params.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)

    print(f"\n📄 Best params saved → {out_path}")
    print(json.dumps(best, indent=2))
    return best


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    main(quick=quick)
