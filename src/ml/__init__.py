"""
ml/  —  Step 5: FreqAI-style ML Signal Bridge
==============================================
Self-contained ML directional-signal layer feeding the LangGraph pipeline.

  features.py       — OHLCV → engineered feature matrix (pandas-only, no TA-Lib)
  labels.py         — forward-direction binary labels (Polymarket Up/Down aligned)
  model.py          — MLSignalModel: LightGBM → sklearn HistGBM → heuristic fallback
  freqai_bridge.py  — lifecycle: adaptive retraining, persistence, generate_ml_signal()
"""
from ml.features      import build_features, latest_feature_row, FEATURE_COLUMNS
from ml.labels        import make_labels, align_xy
from ml.model         import MLSignalModel
from ml.freqai_bridge import FreqAIBridge, clear_model_cache

__all__ = [
    "build_features", "latest_feature_row", "FEATURE_COLUMNS",
    "make_labels", "align_xy",
    "MLSignalModel",
    "FreqAIBridge", "clear_model_cache",
]
