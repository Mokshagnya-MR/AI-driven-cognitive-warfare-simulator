from fastapi import APIRouter

from app.schemas.request import FeatureVectorRequest
from app.schemas.response import PredictionResponse
from app.services.model_service import predict


router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_route(request: FeatureVectorRequest) -> PredictionResponse:
    result = predict(request.features)
    return PredictionResponse(**result)