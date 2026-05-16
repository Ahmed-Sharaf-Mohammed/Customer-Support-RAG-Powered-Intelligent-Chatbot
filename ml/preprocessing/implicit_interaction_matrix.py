import os
import pickle
import pandas as pd
import numpy as np
import django
from scipy.sparse import csr_matrix, save_npz
from datetime import datetime

# ---------------------------------------------------
# Django setup
# ---------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
from recommender.models import UserBrowsingLog

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = str(_PROJECT_ROOT / "data" / "processed")
ARTIFACTS_DIR = str(_PROJECT_ROOT / "data" / "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------------------------------
# Load encoders (saved from explicit pipeline)
# ---------------------------------------------------
def load_encoders():
    with open(f"{ARTIFACTS_DIR}/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/item_encoder.pkl", "rb") as f:
        item_encoder = pickle.load(f)
    print("✅ Encoders loaded")
    return user_encoder, item_encoder

# ---------------------------------------------------
# Load implicit interactions from DB
# ---------------------------------------------------
def load_implicit_interactions():
    print("📥 Loading UserBrowsingLog from DB...")
    qs = UserBrowsingLog.objects.values(
        "user_id", "item_id", "event_type", "timestamp"
    )
    df = pd.DataFrame(list(qs))
    print(f"✅ Loaded {len(df)} implicit events")
    return df

# ---------------------------------------------------
# Define event weights
# ---------------------------------------------------
def get_event_weight(event_type):
    weights = {
        "view": 1,
        "click": 2,
        "add_to_cart": 4,
        "remove_from_cart": -1,
        "search": 0.5,
    }
    return weights.get(event_type, 0)

# ---------------------------------------------------
# Apply encoding (transform only, no fit)
# ---------------------------------------------------
def transform_ids(df, user_encoder, item_encoder):
    print("🔄 Transforming user_id and item_id (using existing encoders)...")
    
    # استخدام isin مباشرة (أسرع من set)
    known_users_mask = df["user_id"].isin(user_encoder.classes_)
    known_items_mask = df["item_id"].isin(item_encoder.classes_)
    
    unknown_users = (~known_users_mask).sum()
    unknown_items = (~known_items_mask).sum()
    
    if unknown_users > 0:
        print(f"⚠️ Unknown users skipped: {unknown_users}")
    if unknown_items > 0:
        print(f"⚠️ Unknown items skipped: {unknown_items}")
    
    df = df[known_users_mask & known_items_mask]
    
    if len(df) == 0:
        print("❌ No valid interactions after filtering unknowns!")
        return df
    
    df["user_encoded"] = user_encoder.transform(df["user_id"])
    df["item_encoded"] = item_encoder.transform(df["item_id"])
    print("✅ Encoding applied")
    return df

# ---------------------------------------------------
# Add event weight column
# ---------------------------------------------------
def add_weights(df):
    df["weight"] = df["event_type"].apply(get_event_weight)
    df = df[df["weight"] != 0]
    print(f"✅ After filtering zero-weight events: {len(df)} rows")
    return df

# ---------------------------------------------------
# Optionally apply time decay
# ---------------------------------------------------
def apply_time_decay(df, half_life_days=7):
    print("⏳ Applying time decay...")
    # جعل now بدون timezone (naive)
    now = datetime.now()
    # تحويل timestamp إلى naive (إزالة الـ tz info إن وجدت)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["age_days"] = (now - df["timestamp"]).dt.days
    df["decay_factor"] = 0.5 ** (df["age_days"] / half_life_days)
    df["weight_decayed"] = df["weight"] * df["decay_factor"]
    print("✅ Time decay applied")
    return df

# ---------------------------------------------------
# Aggregate duplicates
# ---------------------------------------------------
def aggregate_implicit(df, use_decay=True):
    print("🔄 Aggregating duplicate user-item interactions...")
    weight_col = "weight_decayed" if use_decay and "weight_decayed" in df.columns else "weight"
    agg_df = df.groupby(["user_encoded", "item_encoded"], as_index=False)[weight_col].sum()
    agg_df.rename(columns={weight_col: "implicit_score"}, inplace=True)
    print(f"✅ Aggregated to {len(agg_df)} unique user-item pairs")
    return agg_df

# ---------------------------------------------------
# Build sparse implicit matrix (float32)
# ---------------------------------------------------
def build_implicit_matrix(agg_df):
    print("🏗️ Building implicit sparse matrix...")
    rows = agg_df["user_encoded"]
    cols = agg_df["item_encoded"]
    values = agg_df["implicit_score"].astype(np.float32)
    matrix = csr_matrix((values, (rows, cols)))
    print(f"📐 Matrix shape: {matrix.shape}")
    print(f"📊 Non-zero entries: {matrix.nnz}")
    return matrix

# ---------------------------------------------------
# Save artifacts
# ---------------------------------------------------
def save_implicit_artifacts(matrix, agg_df):
    save_npz(f"{ARTIFACTS_DIR}/implicit_matrix.npz", matrix)
    print("💾 Implicit matrix saved")
    
    agg_df.to_parquet(f"{PROCESSED_DIR}/implicit_aggregated.parquet", index=False)
    print("💾 Aggregated implicit interactions saved")
    
    metadata = {
        "num_users": matrix.shape[0],
        "num_items": matrix.shape[1],
        "num_interactions": matrix.nnz,
        "sparsity": 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
    }
    with open(f"{ARTIFACTS_DIR}/implicit_matrix_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("💾 Implicit matrix metadata saved")

# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------
def main(use_time_decay=True):
    print("🚀 Starting implicit preprocessing pipeline...\n")
    
    user_encoder, item_encoder = load_encoders()
    df = load_implicit_interactions()
    df = transform_ids(df, user_encoder, item_encoder)
    
    if len(df) == 0:
        print("❌ No data to process. Exiting.")
        return None, None
    
    df = add_weights(df)
    if use_time_decay:
        df = apply_time_decay(df)
    agg_df = aggregate_implicit(df, use_decay=use_time_decay)
    matrix = build_implicit_matrix(agg_df)
    save_implicit_artifacts(matrix, agg_df)
    
    print("\n🎉 Implicit preprocessing pipeline completed")
    return matrix, agg_df

if __name__ == "__main__":
    main(use_time_decay=True)