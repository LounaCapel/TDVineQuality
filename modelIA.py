
#nom, para, retour
from pydantic import BaseModel
from keras import models

class Vin (BaseModel):
    fixedActivity : float

def PredictionNote(vin : Vin) -> int:
    return (0)

def VinParfait() -> Vin:
    return ("")

def ModeleSerialise(modele : models) -> models :
    return()

def Description() -> str :
    return ("information")

def Enrichir(modele : models,new : Vin ) -> models :
    return (modele)

def Entrainer(modele : models) -> models :
    return (modele)
