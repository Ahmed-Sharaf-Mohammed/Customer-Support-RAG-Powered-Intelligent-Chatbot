import os
import sys
import json
import time
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = str(_PROJECT_ROOT / "data" / "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)


def _step(n: int, title: str):
    bar = "─" * 50
    print(f"\n{bar}")
    print(f"  Step {n}: {title}")
    print(bar)


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def _load_best_params() -> dict:
    path = f"{REPORTS_DIR}/best_params.json"
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
        print(f"   📂 Loaded tuned params from {path}")
        return params
    print("   ℹ️  No tuning params found — using defaults")
    return {}


def run(tune: bool = False, quick_tune: bool = False) -> dict:
    total_t0 = time.time()
    print("🚀 Training Pipeline")
    print("=" * 50)

    best_params = {}

    # ── Step 0: Tuning (optional) ─────────────────────────────────────────────
    if tune or quick_tune:
        _step(0, "Hyperparameter tuning")
        t0 = time.time()
        from ml.training.tuning import main as run_tuning
        best_params = run_tuning(quick=quick_tune)
        print(f"   ✅ Tuning done ({_elapsed(t0)})")
    else:
        best_params = _load_best_params()

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    _step(1, "Training models")
    t0 = time.time()

    svd_p = best_params.get("svd", {})
    als_p = best_params.get("als", {})
    cb_p  = best_params.get("content_based", {})

    from ml.training.train import main as train_models, build_content_based_model
    results = train_models(
        n_svd_factors     = svd_p.get("n_factors",     50),
        als_factors       = als_p.get("factors",       64),
        als_iterations    = als_p.get("iterations",    20),
        als_regularization= als_p.get("regularization", 0.01),
    )
    print(f"   ✅ Training done ({_elapsed(t0)})")

    # ── Step 2: Evaluate ──────────────────────────────────────────────────────
    _step(2, "Evaluating models")
    t0 = time.time()
    from ml.training.evaluate import run_evaluation
    report = run_evaluation()
    print(f"   ✅ Evaluation done ({_elapsed(t0)})")

    print(f"\n{'='*50}")
    print(f"🎉 Training pipeline complete  (total: {_elapsed(total_t0)})")
    print(f"{'='*50}")

    return report


if __name__ == "__main__":
    tune       = "--tune"       in sys.argv
    quick_tune = "--quick-tune" in sys.argv
    run(tune=tune, quick_tune=quick_tune)
