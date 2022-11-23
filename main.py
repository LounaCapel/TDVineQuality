from fastapi import Body, FastAPI
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field
from modelIA import *

app = FastAPI()

@app.post("/api/predict")
async def predict(vin: Vin):
    return predictionNote(vin)

@app.get("/api/predict")
async def perfect():
    return vinParfait()

@app.get("/api/model")
async def getModel():
    return modeleSerialise()

@app.get("/api/model/description")
async def getModelDescription():
    return description()

@app.put("/api/model")
async def addWine(vin : Vin):
    return enrichir(vin)

@app.post("/api/model/retrain")
async def retrain():
    return entrainer()