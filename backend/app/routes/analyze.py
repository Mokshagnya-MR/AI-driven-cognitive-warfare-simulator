from fastapi import APIRouter, HTTPException

from app.core.logger import get_logger
from app.schemas.request import AnalyzeNewsRequest, AnalyzeRequest
from app.schemas.response import AnalyzeResponse, StructuredExplanationResponse
from app.services.explanation_engine import generate_explanation
from app.services.llm_service import OllamaServiceError, generate_misinformation
from app.services.feature_utils import metrics_to_feature_vector
from app.services.model_service import get_model_service, predict as model_predict
from app.services.simulation_service import simulate_network
from app.services.xai_service import get_shap_attributions


logger = get_logger(__name__)
router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        generated_text = generate_misinformation(request.topic)
    except OllamaServiceError as exc:
        logger.error("Analysis generation failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    simulation_result = simulate_network(text=generated_text, steps=request.steps)
    graph_stats = simulation_result["graph_stats"]
    propagation_metrics = simulation_result["propagation_metrics"]
    features = metrics_to_feature_vector(graph_stats, propagation_metrics)

    prediction_result = model_predict(features)
    model_service = get_model_service()
    probability = float(model_service.predict_positive_probabilities(features)[0])
    shap_attributions = get_shap_attributions(features)
    explanation_result = generate_explanation(
        prediction=int(prediction_result["prediction"]),
        probability=probability,
        metrics={
            **graph_stats,
            **propagation_metrics,
        },
        shap_values=shap_attributions,
        feature_names=shap_attributions.get("feature_names"),
    )

    return AnalyzeResponse(
        generated_text=generated_text,
        metrics={
            "graph_stats": graph_stats,
            "propagation_metrics": propagation_metrics,
            "features": features,
        },
        prediction=int(prediction_result["prediction"]),
        confidence=float(prediction_result["confidence"]),
        threshold_used=float(prediction_result["threshold_used"]),
        explanation=StructuredExplanationResponse(**explanation_result),
    )


@router.post("/analyze/news", response_model=AnalyzeResponse)
async def analyze_news(request: AnalyzeNewsRequest) -> AnalyzeResponse:
    context_topic = request.title.strip()
    if request.description.strip():
        context_topic = f"{request.title.strip()}\n\nContext:\n{request.description.strip()}"

    try:
        generated_text = generate_misinformation(context_topic)
    except OllamaServiceError as exc:
        logger.error("News analysis generation failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    simulation_result = simulate_network(text=generated_text, steps=request.steps)
    graph_stats = simulation_result["graph_stats"]
    propagation_metrics = simulation_result["propagation_metrics"]
    features = metrics_to_feature_vector(graph_stats, propagation_metrics)

    prediction_result = model_predict(features)
    model_service = get_model_service()
    probability = float(model_service.predict_positive_probabilities(features)[0])
    shap_attributions = get_shap_attributions(features)
    explanation_result = generate_explanation(
        prediction=int(prediction_result["prediction"]),
        probability=probability,
        metrics={
            **graph_stats,
            **propagation_metrics,
        },
        shap_values=shap_attributions,
        feature_names=shap_attributions.get("feature_names"),
    )

    return AnalyzeResponse(
        generated_text=generated_text,
        metrics={
            "graph_stats": graph_stats,
            "propagation_metrics": propagation_metrics,
            "features": features,
        },
        prediction=int(prediction_result["prediction"]),
        confidence=float(prediction_result["confidence"]),
        threshold_used=float(prediction_result["threshold_used"]),
        explanation=StructuredExplanationResponse(**explanation_result),
    )