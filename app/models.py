from typing import List
from pydantic import BaseModel, Field, validator
import numpy as np

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4)
    request_id: str = Field(default=None)

    @validator('features')
    def check_features(cls, v):
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features cannot contain NaN or Inf")
        return v

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str
    request_id: str = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
