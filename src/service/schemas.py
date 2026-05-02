from pydantic import BaseModel, Field


class PredictPurchaseRequest(BaseModel):
    item_id: int = Field(default=98113, ge=0)
    hour: int = Field(default=12, ge=1, lt=24)
    weekday: int = Field(default=3, ge=1, lt=7)


class PredictPurchaseResponse(BaseModel):
    item_id: int = Field(default=0)
    purchase_probability: float = Field(default=0.0)


class PredictPurchaseErrorResponse(BaseModel):
    error: str
