from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import shap

from app.core.config import FEATURE_NAMES
from app.core.logger import get_logger
from app.services.feature_utils import normalize_feature_vector
from app.services.model_service import get_model_service


logger = get_logger(__name__)


class XAIService:
    def __init__(self) -> None:
        self.feature_names = FEATURE_NAMES
        self.model_service = get_model_service()
        self.explainer = self._build_explainer()

    def _build_explainer(self) -> shap.Explainer:
        background = np.array(
            [
                [0.2, 0.1, 0.3, 1.0],
                [0.5, 0.2, 0.4, 2.0],
                [1.0, 0.4, 0.5, 3.0],
                [1.5, 0.6, 0.7, 4.0],
                [2.0, 0.8, 0.9, 5.0],
            ],
            dtype=float,
        )

        def predict_positive_class(data: np.ndarray) -> np.ndarray:
            return self.model_service.predict_positive_probabilities(np.asarray(data, dtype=float))

        return shap.Explainer(predict_positive_class, background)

    def explain(self, features: List[float]) -> Dict[str, Any]:
        attribution = self.get_shap_attributions(features)
        values = np.asarray(attribution["shap_values"], dtype=float)
        absolute_values = np.abs(values)
        ranked_indices = np.argsort(absolute_values)[::-1]

        top_features = [self.feature_names[index] for index in ranked_indices]
        importance_scores = [float(round(absolute_values[index], 6)) for index in ranked_indices]

        logger.info("SHAP explanation generated")

        return {
            "top_features": top_features,
            "importance_scores": importance_scores,
        }

    def get_shap_attributions(self, features: List[float]) -> Dict[str, Any]:
        array = normalize_feature_vector(features).reshape(1, -1)
        explanation = self.explainer(array)
        values = np.asarray(explanation.values[0], dtype=float)

        return {
            "feature_names": list(self.feature_names),
            "shap_values": values.tolist(),
        }


_xai_service = XAIService()


def explain_features(features: List[float]) -> Dict[str, Any]:
    return _xai_service.explain(features)


def get_shap_attributions(features: List[float]) -> Dict[str, Any]:
    return _xai_service.get_shap_attributions(features)