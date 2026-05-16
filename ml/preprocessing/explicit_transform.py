import os
import pickle
import pandas as pd
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from recommender.models import UserInteraction
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = str(_PROJECT_ROOT / "data" / "processed")
ARTIFACTS_DIR = str(_PROJECT_ROOT / "data" / "artifacts")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------
# Helpers for scalers (load or fit)
# ---------------------------------------------------
def get_or_fit_scaler(df, column, scaler_path, feature_range=(0, 1)):
    """
    Load existing scaler if exists, otherwise fit on the whole column and save.
    Returns: scaler, transformed column values.
    """
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        transformed = scaler.transform(df[[column]])
        print(f"✅ Loaded existing scaler from {scaler_path}")
    else:
        scaler = MinMaxScaler(feature_range=feature_range)
        transformed = scaler.fit_transform(df[[column]])
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"✅ Fitted and saved new scaler to {scaler_path}")
    return scaler, transformed

# ---------------------------------------------------
# Load encoders
# ---------------------------------------------------
def load_encoders():
    with open(f"{ARTIFACTS_DIR}/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/item_encoder.pkl", "rb") as f:
        item_encoder = pickle.load(f)
    print("✅ Global encoders loaded")
    return user_encoder, item_encoder

# ---------------------------------------------------
# Load explicit data
# ---------------------------------------------------
def load_explicit_data():
    qs = UserInteraction.objects.values(
        "user_id", "item_id", "rating", "timestamp", "verified", "helpful_votes"
    )
    df = pd.DataFrame(list(qs))
    print(f"📥 Loaded {len(df)} explicit interactions")
    return df

# ---------------------------------------------------
# Clean missing values
# ---------------------------------------------------
def clean_data(df):
    initial_len = len(df)
    df = df.dropna(subset=["rating", "timestamp"])
    print(f"🧹 Dropped {initial_len - len(df)} rows with missing rating/timestamp")
    return df

# ---------------------------------------------------
# Filter only known user_ids and item_ids (safeguard)
# ---------------------------------------------------
def filter_known_ids(df, user_encoder, item_encoder):
    known_users_mask = df["user_id"].isin(user_encoder.classes_)
    known_items_mask = df["item_id"].isin(item_encoder.classes_)
    unknown_users = (~known_users_mask).sum()
    unknown_items = (~known_items_mask).sum()
    if unknown_users:
        print(f"⚠️ {unknown_users} rows with unknown user_id (skipped)")
    if unknown_items:
        print(f"⚠️ {unknown_items} rows with unknown item_id (skipped)")
    return df[known_users_mask & known_items_mask]

# ---------------------------------------------------
# Process timestamps (safe conversion)
# ---------------------------------------------------
def process_timestamps(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_unix"] = df["timestamp"].astype("int64") // 10**9
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    return df

# ---------------------------------------------------
# Transform IDs using global encoders (no fit!)
# ---------------------------------------------------
def transform_ids(df, user_encoder, item_encoder):
    df["user_encoded"] = user_encoder.transform(df["user_id"])
    df["item_encoded"] = item_encoder.transform(df["item_id"])
    return df

# ---------------------------------------------------
# Normalize ratings using persistent scaler
# ---------------------------------------------------
def normalize_ratings(df):
    scaler_path = f"{ARTIFACTS_DIR}/rating_scaler.pkl"
    scaler, normalized = get_or_fit_scaler(df, "rating", scaler_path)
    df["rating_normalized"] = normalized
    return df

# ---------------------------------------------------
# Process verified and helpful_votes (with scaler for helpful)
# ---------------------------------------------------
def process_verified_helpful(df):
    df["verified_int"] = df["verified"].astype(int)
    if "helpful_votes" in df.columns:
        df["helpful_votes"] = df["helpful_votes"].fillna(0)
        scaler_path = f"{ARTIFACTS_DIR}/helpful_scaler.pkl"
        _, normalized = get_or_fit_scaler(df, "helpful_votes", scaler_path)
        df["helpful_votes_normalized"] = normalized
    return df

# ---------------------------------------------------
# Aggregate duplicate (user, item) interactions
# ---------------------------------------------------
def aggregate_explicit(df):
    print("🔄 Aggregating duplicate (user, item) interactions...")
    agg_dict = {
        "rating_normalized": "mean",
        "timestamp_unix": "last",
        "year": "last",
        "month": "last",
        "day": "last",
        "dayofweek": "last",
        "hour": "last",
        "verified_int": "max",
    }
    if "helpful_votes_normalized" in df.columns:
        agg_dict["helpful_votes_normalized"] = "mean"
    initial = len(df)
    df_agg = df.groupby(["user_encoded", "item_encoded"], as_index=False).agg(agg_dict)
    print(f"✅ Reduced from {initial} to {len(df_agg)} rows after aggregation")
    return df_agg

# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------
def main():
    print("🚀 Transforming explicit interactions using global encoders...\n")
    user_enc, item_enc = load_encoders()
    df = load_explicit_data()
    df = clean_data(df)
    df = filter_known_ids(df, user_enc, item_enc)
    if df.empty:
        print("❌ No valid data after filtering. Exiting.")
        return
    df = process_timestamps(df)
    df = transform_ids(df, user_enc, item_enc)
    df = normalize_ratings(df)
    df = process_verified_helpful(df)
    
    # Save raw transformed data (before aggregation)
    df.to_parquet(f"{PROCESSED_DIR}/explicit_transformed_raw.parquet", index=False)
    print("💾 Saved explicit_transformed_raw.parquet (before aggregation)")
    
    # Aggregate and save final version for matrix building
    df_agg = aggregate_explicit(df)
    df_agg.to_parquet(f"{PROCESSED_DIR}/explicit_aggregated.parquet", index=False)
    print("💾 Saved explicit_aggregated.parquet (ready for interaction matrix)")
    
    print("🎉 Transformation completed")

if __name__ == "__main__":
    main()