#nom, para, retour
from pydantic import BaseModel
from keras import models

class Vin (BaseModel):
    fixedActivity : float

def predictionNote(vin : Vin) -> int:
    return (0)

def vinParfait() -> Vin:
    return ("")

def modeleSerialise() -> models :
    return("")

def description() -> str :
    return ("information")

def enrichir(modele : models,new : Vin ) -> models :
    return (modele)

def entrainer(modele : models) -> models :
    return (modele)
