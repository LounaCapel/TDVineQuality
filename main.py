from fastapi import FastAPI
from fastapi.responses import FileResponse
from modelIA import *

app = FastAPI()

@app.post("/api/predict")
async def predict(vin: Vin):
    return predictionNote(vin, loadModeleSerialise("RandomForestSerialise.pkl"))

@app.get("/api/predict")
async def perfect():
    return vinParfait('Wines.csv')

@app.get("/api/model")
async def getModel():
    file_path = "RandomForestSerialise.pkl"
    return FileResponse(path=file_path, filename=file_path, media_type='pkl')

@app.get("/api/model/description")
async def getModelDescription():
    return description(loadModeleSerialise("RandomForestSerialise.pkl"), 'Wines.csv')

@app.put("/api/model")
async def addWine(vin : Vin):
    return enrichir(vin, 'Wines.csv')

@app.post("/api/model/retrain")
async def retrain():
    return modeleSerialise(entrainer('Wines.csv'))