import os
import json
import time
import logging
import django
from pathlib import Path
from datetime import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

logger = logging.getLogger("ml")

# ── Absolute paths (never depend on CWD) ─────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR   = str(_PROJECT_ROOT / "data" / "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

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
        print(f"   📂 Using tuned params from {path}")
        return params
    print("   ℹ️  No tuned params found — using defaults")
    return {}


def _load_previous_report() -> dict:
    path = f"{REPORTS_DIR}/evaluation_report.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _compare_reports(old: dict, new: dict):
    """Print NDCG@10 delta for each model between old and new reports."""
    print("\n  📊 Performance comparison (NDCG@10)")
    print("  " + "─" * 44)
    print(f"  {'Model':20s}  {'Before':>8}  {'After':>8}  {'Δ':>8}")
    print("  " + "─" * 44)

    for model in ("svd", "als", "content_based"):
        old_v = old.get(model, {}).get("@10", {}).get("NDCG", None)
        new_v = new.get(model, {}).get("@10", {}).get("NDCG", None)
        old_s = f"{old_v:.4f}" if old_v is not None else "   —"
        new_s = f"{new_v:.4f}" if new_v is not None else "   —"
        if old_v is not None and new_v is not None:
            delta = new_v - old_v
            sign  = "+" if delta >= 0 else ""
            d_s   = f"{sign}{delta:.4f}"
            arrow = "📈" if delta > 0.001 else ("📉" if delta < -0.001 else "≈ ")
        else:
            d_s, arrow = "   —", "  "
        print(f"  {model:20s}  {old_s:>8}  {new_s:>8}  {arrow} {d_s}")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run(force_retune: bool = False) -> dict:
    started_at = datetime.now().isoformat(timespec="seconds")
    total_t0   = time.time()

    print(f"🔄 Retrain Pipeline  [{started_at}]")
    print("=" * 50)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    try:
        import mlflow
        from django.conf import settings
        tracking_uri = getattr(settings, "MLFLOW_TRACKING_URI", str(_PROJECT_ROOT / "mlruns"))
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("recommendation-system")
        _mlflow_run = mlflow.start_run(run_name=f"retrain_{started_at}")
        _use_mlflow = True
        print(f"   📊 MLflow tracking → {tracking_uri}")
    except Exception as e:
        _mlflow_run = None
        _use_mlflow = False
        print(f"   ⚠️  MLflow disabled: {e}")

    try:
        # ── Step 1: Update encoders ───────────────────────────────────────────
        _step(1, "Updating global encoders")
        t0 = time.time()
        from ml.preprocessing.global_encoders import build_global_encoders
        build_global_encoders()
        print(f"   ✅ Done ({_elapsed(t0)})")

        # ── Step 2: Re-transform explicit ─────────────────────────────────────
        _step(2, "Re-transforming explicit interactions")
        t0 = time.time()
        from ml.preprocessing.explicit_transform import main as explicit_transform
        explicit_transform()
        print(f"   ✅ Done ({_elapsed(t0)})")

        # ── Step 3: Rebuild explicit matrix ───────────────────────────────────
        _step(3, "Rebuilding explicit sparse matrix")
        t0 = time.time()
        from ml.preprocessing.explicit_interaction_matrix import main as build_explicit
        build_explicit()
        print(f"   ✅ Done ({_elapsed(t0)})")

        # ── Step 4: Rebuild implicit matrix ───────────────────────────────────
        _step(4, "Rebuilding implicit matrix (browsing + time decay)")
        t0 = time.time()
        from ml.preprocessing.implicit_interaction_matrix import main as build_implicit
        build_implicit(use_time_decay=True)
        print(f"   ✅ Done ({_elapsed(t0)})")

        # ── Step 5: Retrain models ────────────────────────────────────────────
        _step(5, "Retraining models")
        t0          = time.time()
        best_params = _load_best_params()
        svd_p       = best_params.get("svd", {})
        als_p       = best_params.get("als", {})

        train_params = dict(
            n_svd_factors      = svd_p.get("n_factors",      50),
            als_factors        = als_p.get("factors",        64),
            als_iterations     = als_p.get("iterations",     20),
            als_regularization = als_p.get("regularization", 0.01),
        )

        if _use_mlflow:
            mlflow.log_params({
                "svd_n_factors":      train_params["n_svd_factors"],
                "als_factors":        train_params["als_factors"],
                "als_iterations":     train_params["als_iterations"],
                "als_regularization": train_params["als_regularization"],
            })

        from ml.training.train import main as train_models
        train_models(**train_params)
        print(f"   ✅ Done ({_elapsed(t0)})")

        # ── Step 6: Evaluate & compare ────────────────────────────────────────
        _step(6, "Evaluating retrained models")
        t0         = time.time()
        old_report = _load_previous_report()
        from ml.training.evaluate import run_evaluation
        new_report = run_evaluation()
        _compare_reports(old_report, new_report)
        print(f"   ✅ Done ({_elapsed(t0)})")

        # ── Log evaluation metrics to MLflow ──────────────────────────────────
        if _use_mlflow:
            for model_name, model_data in new_report.items():
                if not isinstance(model_data, dict):
                    continue
                for k_label, metrics in model_data.items():
                    if not isinstance(metrics, dict):
                        continue
                    k = k_label.replace("@", "")
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(
                                f"{model_name}_{metric_name}_at_{k}",
                                round(float(value), 6),
                            )
            # Save full report as MLflow artifact
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="eval_report_"
            ) as tmp:
                json.dump(new_report, tmp, indent=2)
                tmp_path = tmp.name
            mlflow.log_artifact(tmp_path, artifact_path="evaluation")
            os.unlink(tmp_path)
            print("   📊 Metrics + report logged to MLflow")

        # ── Step 7: Hot-reload inference singleton ────────────────────────────
        _step(7, "Hot-reloading inference predictor")
        try:
            from ml.inference.recommender import reload_predictor
            ok = reload_predictor()
            print(f"   {'✅ Predictor reloaded.' if ok else '⚠️  Reload failed (check logs).'}")
        except Exception as e:
            print(f"   ⚠️  Could not reload predictor: {e}")

        # ── Summary ───────────────────────────────────────────────────────────
        total_seconds = round(time.time() - total_t0, 1)
        retrain_meta = {
            "retrained_at":  started_at,
            "total_seconds": total_seconds,
            "params_used":   best_params,
        }
        meta_path = f"{REPORTS_DIR}/retrain_meta.json"
        with open(meta_path, "w") as f:
            json.dump(retrain_meta, f, indent=2)

        if _use_mlflow:
            mlflow.log_metric("total_seconds", total_seconds)

        print(f"\n{'='*50}")
        print(f"🎉 Retrain pipeline complete  (total: {_elapsed(total_t0)})")
        print(f"{'='*50}")

    finally:
        if _use_mlflow and _mlflow_run is not None:
            mlflow.end_run()

    return new_report


if __name__ == "__main__":
    import sys
    force_retune = "--retune" in sys.argv
    run(force_retune=force_retune)
