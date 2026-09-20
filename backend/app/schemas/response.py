from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict


class GenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class SimulationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_stats: Dict[str, Any]
    propagation_metrics: Dict[str, Any]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: int
    confidence: float
    threshold_used: float


class ExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_features: List[str]
    importance_scores: List[float]


class ExplanationDriverResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    explanation: str
    score: float
    direction: str


class StructuredExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    reasoning: List[str]
    risk_level: Literal["low", "medium", "high"]
    key_drivers: List[ExplanationDriverResponse]
    recommendation: str


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_text: str
    metrics: Dict[str, Any]
    prediction: int
    confidence: float
    threshold_used: float
    explanation: StructuredExplanationResponse