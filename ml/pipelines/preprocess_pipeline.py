"""
Preprocessing Pipeline
──────────────────────
Orchestrates all preprocessing steps in the correct order:

  Step 1 — Build global user/item LabelEncoders
  Step 2 — Transform explicit (rating) interactions
  Step 3 — Build explicit sparse matrix
  Step 4 — Build implicit sparse matrix (browsing logs, time-decay)

Run:
    python -m ml.pipelines.preprocess_pipeline
"""

import os
import sys
import time
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()


def _step(n: int, title: str):
    bar = "─" * 50
    print(f"\n{bar}")
    print(f"  Step {n}: {title}")
    print(bar)


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s"


def run():
    total_t0 = time.time()
    print("🚀 Preprocessing Pipeline")
    print("=" * 50)

    # ── Step 1: Global encoders ───────────────────────────────────────────────
    _step(1, "Building global user / item encoders")
    t0 = time.time()
    from ml.preprocessing.global_encoders import build_global_encoders
    user_enc, item_enc = build_global_encoders()
    print(f"   ✅ Done ({_elapsed(t0)})")

    # ── Step 2: Explicit transform ────────────────────────────────────────────
    _step(2, "Transforming explicit (rating) interactions")
    t0 = time.time()
    from ml.preprocessing.explicit_transform import main as explicit_transform
    explicit_transform()
    print(f"   ✅ Done ({_elapsed(t0)})")

    # ── Step 3: Explicit matrix ───────────────────────────────────────────────
    _step(3, "Building explicit sparse rating matrix")
    t0 = time.time()
    from ml.preprocessing.explicit_interaction_matrix import main as build_explicit
    build_explicit()
    print(f"   ✅ Done ({_elapsed(t0)})")

    # ── Step 4: Implicit matrix ───────────────────────────────────────────────
    _step(4, "Building implicit matrix (browsing logs + time decay)")
    t0 = time.time()
    from ml.preprocessing.implicit_interaction_matrix import main as build_implicit
    build_implicit(use_time_decay=True)
    print(f"   ✅ Done ({_elapsed(t0)})")

    print(f"\n{'='*50}")
    print(f"🎉 Preprocessing complete  (total: {_elapsed(total_t0)})")
    print(f"{'='*50}")


if __name__ == "__main__":
    run()
