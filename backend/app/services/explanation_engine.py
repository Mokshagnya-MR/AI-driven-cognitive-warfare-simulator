from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_FEATURE_LABELS = {
    "velocity": "Propagation velocity",
    "bot_ratio": "Bot participation",
    "echo_density": "Echo chamber density",
    "depth": "Propagation depth",
}


def _humanize_feature_name(feature_name: str) -> str:
    return _FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").strip().title())


def _extract_metric_value(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_shap_input(shap_values: Any, feature_names: Sequence[str] | None = None) -> list[tuple[str, float]]:
    if isinstance(shap_values, Mapping):
        names = list(shap_values.get("feature_names", feature_names or []))
        values = shap_values.get("shap_values", [])
    else:
        names = list(feature_names or [])
        values = shap_values

    array = np.asarray(values, dtype=float).reshape(-1)
    if not names:
        names = [f"feature_{index}" for index in range(len(array))]

    pairs = list(zip(names, array.tolist()))
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return pairs


def _summarize_risk(prediction: int, velocity: float, bot_ratio: float, echo_density: float, depth: float) -> str:
    suspicious_signals = sum(
        [velocity > 6.0, bot_ratio > 0.2, echo_density > 0.4, depth > 5.0]
    )

    if prediction == 1 and velocity > 6.0 and bot_ratio > 0.2:
        return "high"
    if prediction == 1 and suspicious_signals >= 2:
        return "medium"
    if prediction == 0 and suspicious_signals >= 1:
        return "medium"
    return "low"


def _build_reasoning_lines(velocity: float, bot_ratio: float, echo_density: float, depth: float) -> list[str]:
    reasoning: list[str] = []

    if velocity > 6.0:
        reasoning.append("High propagation velocity suggests rapid viral spread.")
    if bot_ratio > 0.2:
        reasoning.append("Elevated bot participation indicates possible coordinated amplification.")
    if echo_density > 0.4:
        reasoning.append("High echo chamber density suggests information is circulating within closed communities.")
    if depth > 5.0:
        reasoning.append("Deep propagation indicates sustained engagement across network layers.")

    if not reasoning:
        reasoning.append("The spread pattern does not show strong signs of automated or highly concentrated amplification.")

    return reasoning


def _build_key_drivers(shap_pairs: Iterable[tuple[str, float]]) -> list[dict[str, Any]]:
    pairs = list(shap_pairs)
    total_abs = float(sum(abs(value) for _, value in pairs)) or 1.0

    key_drivers: list[dict[str, Any]] = []
    for feature_name, value in pairs[:3]:
        importance = abs(value) / total_abs
        direction = "toward misinformation" if value >= 0 else "toward organic spread"
        key_drivers.append(
            {
                "label": _humanize_feature_name(feature_name),
                "explanation": f"{_humanize_feature_name(feature_name)} contributed most to the classification by pushing the model {direction}.",
                "score": float(round(importance, 4)),
                "direction": direction,
            }
        )

    return key_drivers


def generate_explanation(
    prediction: int,
    probability: float,
    metrics: Mapping[str, Any],
    shap_values: Any,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    velocity = _extract_metric_value(metrics, "velocity")
    bot_ratio = _extract_metric_value(metrics, "bot_ratio")
    echo_density = _extract_metric_value(metrics, "echo_chamber_density")
    depth = _extract_metric_value(metrics, "depth")

    summary = (
        "This content shows strong indicators of misinformation propagation."
        if int(prediction) == 1
        else "This content appears to follow organic information spread patterns."
    )

    reasoning = _build_reasoning_lines(velocity, bot_ratio, echo_density, depth)
    key_drivers = _build_key_drivers(_normalize_shap_input(shap_values, feature_names=feature_names))
    risk_level = _summarize_risk(int(prediction), velocity, bot_ratio, echo_density, depth)

    recommendation = (
        "Content should be verified before sharing."
        if int(prediction) == 1
        else "No strong manipulation signals detected."
    )

    if float(probability) >= 0.8 and int(prediction) == 1:
        summary = "This content is strongly aligned with misinformation-like spread behavior."
    elif float(probability) <= 0.3 and int(prediction) == 0:
        summary = "This content is strongly aligned with organic spread behavior."

    return {
        "summary": summary,
        "reasoning": reasoning,
        "risk_level": risk_level,
        "key_drivers": key_drivers,
        "recommendation": recommendation,
    }