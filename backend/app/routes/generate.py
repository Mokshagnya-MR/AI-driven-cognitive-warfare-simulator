from fastapi import APIRouter, HTTPException

from app.core.logger import get_logger
from app.schemas.request import TopicRequest
from app.schemas.response import GenerationResponse
from app.services.llm_service import (
    OllamaServiceError,
    generate_counter_narrative,
    generate_misinformation,
    generate_neutral_news,
)


logger = get_logger(__name__)
router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("/misinformation", response_model=GenerationResponse)
async def misinformation(request: TopicRequest) -> GenerationResponse:
    try:
        return GenerationResponse(text=generate_misinformation(request.topic))
    except OllamaServiceError as exc:
        logger.error("Misinformation generation failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/counter", response_model=GenerationResponse)
async def counter_narrative(request: TopicRequest) -> GenerationResponse:
    try:
        return GenerationResponse(text=generate_counter_narrative(request.topic))
    except OllamaServiceError as exc:
        logger.error("Counter narrative generation failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/neutral", response_model=GenerationResponse)
async def neutral_news(request: TopicRequest) -> GenerationResponse:
    try:
        return GenerationResponse(text=generate_neutral_news(request.topic))
    except OllamaServiceError as exc:
        logger.error("Neutral news generation failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc