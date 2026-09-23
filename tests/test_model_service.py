from __future__ import annotations

import inspect

import numpy as np
import pytest

from app.services import model_service as model_service_module
from app.services.model_service import get_model_service


def test_heuristic_tabular_adapter_probability_removed() -> None:
    """Regression guard: the hardcoded confidence-blending heuristic must stay removed.

    A prior version of this service silently blended the trained model's output with
    a hand-written formula whenever confidence was near 0.5, which meant predictions
    were not purely model-derived. That code path was deleted; this test fails again
    if anyone reintroduces it.
    """
    assert not hasattr(model_service_module, "_tabular_adapter_probability")
    source = inspect.getsource(model_service_module)
    assert "_tabular_adapter_probability" not in source


@pytest.mark.parametrize(
    "features",
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.4, 0.2, 0.6, 3.0],
        [10.0, -5.0, 2.5, 100.0],
    ],
)
def test_predict_positive_probabilities_in_unit_interval(features: list[float]) -> None:
    service = get_model_service()
    probabilities = service.predict_positive_probabilities(np.asarray(features, dtype=np.float32))

    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0


def test_predict_returns_well_formed_result() -> None:
    service = get_model_service()
    result = service.predict([0.5, 0.3, 0.4, 2.0])

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["threshold_used"] <= 1.0


def test_get_model_service_is_a_singleton() -> None:
    assert get_model_service() is get_model_service()
