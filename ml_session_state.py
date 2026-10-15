"""
ml_session_state.py
===================
Persists ML training history to disk so all tabs can reflect
which model generation is currently active.
"""

import json
import os
from datetime import datetime

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_training_state.json")


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {
            "training_count": 0,
            "last_trained":   None,
            "best_model":     None,
            "data_source":    None,
            "n_samples":      None,
            "results":        {},
            "history":        [],
        }
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(train_result: dict, data_source: str, n_samples: int):
    state = load_state()
    state["training_count"] += 1
    state["last_trained"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["best_model"]      = train_result["best_model"]
    state["data_source"]     = data_source
    state["n_samples"]       = n_samples

    # Serialise results (drop non-JSON-safe objects)
    clean_results = {}
    for model_name, m in train_result["results"].items():
        clean_results[model_name] = {
            "accuracy":         m["accuracy"],
            "precision":        m["precision"],
            "recall":           m["recall"],
            "f1":               m["f1"],
            "roc_auc":          m["roc_auc"],
            "confusion_matrix": m["confusion_matrix"],
        }
    state["results"] = clean_results

    # Keep a short history log
    state["history"].append({
        "run":         state["training_count"],
        "timestamp":   state["last_trained"],
        "best_model":  state["best_model"],
        "data_source": data_source,
        "n_samples":   n_samples,
        "best_f1":     clean_results[state["best_model"]]["f1"],
    })

    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    return state


def get_training_badge(state: dict) -> str:
    if state["training_count"] == 0:
        return "⚪ Not trained yet — using synthetic fallback"
    return (
        f"✅ Model v{state['training_count']} · "
        f"Trained {state['last_trained']} · "
        f"Best: {state['best_model']} · "
        f"{state['n_samples']} samples ({state['data_source']})"
    )