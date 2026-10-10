"""A tiny REST API around the model. Run: uvicorn api.main:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_value

app = FastAPI(title="House Price API", version="1.0.0")


class HouseFeatures(BaseModel):
 MedInc: float = Field(gt=0, le=20, description="Median income in the block, tens of thousands of USD")
 HouseAge: float = Field(ge=0, le=100, description="Median house age in years")
 AveRooms: float = Field(gt=0, le=50, description="Average rooms per household")
 AveBedrms: float = Field(gt=0, le=20, description="Average bedrooms per household")
 Population: float = Field(gt=0, le=40000, description="People living in the block")
 AveOccup: float = Field(gt=0, le=100, description="Average household members")
 Latitude: float = Field(ge=32, le=43, description="Latitude (California)")
 Longitude: float = Field(ge=-125, le=-114, description="Longitude (California)")


class PriceResponse(BaseModel):
    predicted_price_usd: float
    
    
@app.get("/health")
def health():
 return {"status": "ok"}


@app.post("/predict", response_model=PriceResponse)
def predict(features: HouseFeatures):
 value = predict_value(features.model_dump())
 return PriceResponse(predicted_price_usd=round(value * 100_000, 2))