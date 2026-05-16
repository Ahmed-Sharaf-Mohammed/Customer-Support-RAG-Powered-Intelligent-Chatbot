import os
import pickle
import scipy.sparse as sp
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, save_npz

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = str(_PROJECT_ROOT / "data" / "processed")
ARTIFACTS_DIR = str(_PROJECT_ROOT / "data" / "artifacts")


os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------------------------------
# Load encoded interactions
# ---------------------------------------------------

def load_encoded_data():
    path = f"{PROCESSED_DIR}/explicit_aggregated.parquet"
    print("📥 Loading encoded interactions...")
    df = pd.read_parquet(path)
    print(f"✅ Loaded rows: {len(df)}")
    return df


# ---------------------------------------------------
# Aggregate duplicate interactions
# ---------------------------------------------------

def aggregate_duplicates(df):
    """
    If same user interacted with same item multiple times,
    aggregate using mean for ratings, max for verified, etc.
    """
    print("🔄 Aggregating duplicate interactions...")
    initial_len = len(df)

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

    df = df.groupby(["user_encoded", "item_encoded"], as_index=False).agg(agg_dict)

    print(f"✅ Reduced from {initial_len} to {len(df)} rows after aggregation")
    return df


# ---------------------------------------------------
# Build sparse interaction matrix (float32)
# ---------------------------------------------------

def build_interaction_matrix(df):
    print("🏗️ Building sparse interaction matrix...")
    rows = df["user_encoded"]
    cols = df["item_encoded"]
    values = df["rating_normalized"].astype("float32")   # <-- float32 for efficiency

    matrix = csr_matrix((values, (rows, cols)))
    print("✅ Sparse matrix created")
    print(f"📐 Matrix shape: {matrix.shape}")
    return matrix


# ---------------------------------------------------
# Save matrix
# ---------------------------------------------------

def save_matrix(matrix):
    output_path = f"{ARTIFACTS_DIR}/explicit_matrix.npz"
    save_npz(output_path, matrix)
    print(f"💾 Matrix saved: {output_path}")


# ---------------------------------------------------
# Save matrix metadata
# ---------------------------------------------------

def save_matrix_metadata(matrix):
    num_users, num_items = matrix.shape
    num_interactions = matrix.nnz
    sparsity = 1.0 - (num_interactions / (num_users * num_items))

    metadata = {
        "num_users": int(num_users),
        "num_items": int(num_items),
        "num_interactions": int(num_interactions),
        "sparsity": float(sparsity)
    }

    with open(f"{ARTIFACTS_DIR}/matrix_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("💾 Matrix metadata saved")
    print(f"   Users: {num_users}, Items: {num_items}, Interactions: {num_interactions}")
    print(f"   Sparsity: {sparsity:.6f}")


# ---------------------------------------------------
# Matrix statistics (optional, for console)
# ---------------------------------------------------

def print_matrix_stats(matrix):
    num_users, num_items = matrix.shape
    num_interactions = matrix.nnz
    sparsity = 1.0 - (num_interactions / (num_users * num_items))
    print("\n📊 Matrix Statistics")
    print(f"Users  : {num_users}")
    print(f"Items  : {num_items}")
    print(f"Entries: {num_interactions}")
    print(f"Sparsity: {sparsity:.6f}")


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------

def main():
    print("🚀 Starting interaction matrix pipeline...\n")

    df = load_encoded_data()

    # Build matrix (float32)
    matrix = build_interaction_matrix(df)

    # Save matrix
    save_matrix(matrix)

    # [2] Save metadata
    save_matrix_metadata(matrix)

    # [3] Print stats (optional)
    print_matrix_stats(matrix)

    print("\n🎉 Interaction matrix pipeline completed")
    return matrix, df


# ---------------------------------------------------
# Run
# ---------------------------------------------------

if __name__ == "__main__":
    main()