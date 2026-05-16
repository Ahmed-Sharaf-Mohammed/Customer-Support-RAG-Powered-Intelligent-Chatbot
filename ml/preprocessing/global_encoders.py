import os
import pickle
import pandas as pd
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from recommender.models import UserInteraction, UserBrowsingLog, Item
from sklearn.preprocessing import LabelEncoder

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = str(_PROJECT_ROOT / "data" / "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def get_all_users():
    print("🔍 Fetching all user IDs...")
    users_explicit = set(UserInteraction.objects.values_list("user_id", flat=True).distinct())
    users_implicit = set(UserBrowsingLog.objects.values_list("user_id", flat=True).distinct())
    all_users = sorted(users_explicit.union(users_implicit))   # sorted for reproducibility
    print(f"   Explicit: {len(users_explicit)}, Implicit: {len(users_implicit)}, Total: {len(all_users)}")
    return all_users

def get_all_items():
    print("🔍 Fetching all item IDs...")
    items_explicit = set(UserInteraction.objects.values_list("item_id", flat=True).distinct())
    items_implicit = set(UserBrowsingLog.objects.values_list("item_id", flat=True).distinct())
    items_catalog = set(Item.objects.values_list("item_id", flat=True).distinct())
    all_items = sorted(items_explicit.union(items_implicit).union(items_catalog))
    print(f"   Explicit: {len(items_explicit)}, Implicit: {len(items_implicit)}, Catalog: {len(items_catalog)}, Total: {len(all_items)}")
    return all_items

def build_global_encoders():
    print("🚀 Building global encoders...\n")
    all_users = get_all_users()
    all_items = get_all_items()
    if not all_users or not all_items:
        raise ValueError("Missing users or items")
    
    user_encoder = LabelEncoder().fit(all_users)
    item_encoder = LabelEncoder().fit(all_items)
    
    # Save encoders
    with open(f"{ARTIFACTS_DIR}/user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)
    with open(f"{ARTIFACTS_DIR}/item_encoder.pkl", "wb") as f:
        pickle.dump(item_encoder, f)
    
    # Save metadata
    metadata = {
        "num_users": len(user_encoder.classes_),
        "num_items": len(item_encoder.classes_),
        "users_sample": user_encoder.classes_[:5].tolist(),
        "items_sample": item_encoder.classes_[:5].tolist(),
    }
    with open(f"{ARTIFACTS_DIR}/global_encoder_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"💾 Saved encoders & metadata. Users: {metadata['num_users']}, Items: {metadata['num_items']}")
    return user_encoder, item_encoder

if __name__ == "__main__":
    build_global_encoders()