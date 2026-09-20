from fastapi import APIRouter

from app.schemas.request import FeatureVectorRequest
from app.schemas.response import ExplanationResponse
from app.services.xai_service import explain_features


router = APIRouter(tags=["explain"])


@router.post("/explain", response_model=ExplanationResponse)
async def explain_route(request: FeatureVectorRequest) -> ExplanationResponse:
    result = explain_features(request.features)
    return ExplanationResponse(**result)