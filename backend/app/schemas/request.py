from typing import List

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import SIMULATION_STEPS


class TopicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1)


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    steps: int = Field(default=SIMULATION_STEPS, ge=1, le=100)
    bot_ratio_override: float | None = Field(default=None, ge=0.0, le=1.0)
    seed_nodes: int | None = Field(default=None, ge=1, le=50)
    influencer_strength: float | None = Field(default=None, ge=0.0, le=1.0)


class FeatureVectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: List[float] = Field(..., min_length=4, max_length=4)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1)
    steps: int = Field(default=SIMULATION_STEPS, ge=1, le=100)


class AnalyzeNewsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1)
    description: str = Field(default="")
    steps: int = Field(default=SIMULATION_STEPS, ge=1, le=100)